#!/usr/bin/env python3
"""
annotate.py — Hierarchical DVC annotation pipeline for YouCook2 segmentation data.

Three annotation levels with distinct strategies:
  Level 1: Warped-Time Segmentation — uniform sampling → virtual frame index → map back
  Level 2: Phase-Based Event Detection — uses L1 phases as scope, detects events per phase
  Level 3: Local Temporal Grounding — given L2 event + text query → pinpoint atomic moments

Usage:
    python annotate.py \
        --frames-dir proxy_data/hier_seg_annotation/frames \
        --output-dir proxy_data/hier_seg_annotation/annotations \
        --level 1 \
        --api-base https://api.novita.ai/v3/openai \
        --model pa/gmn-2.5-pr \
        --workers 4 --limit 50

Output:
    annotations/{clip_key}.json
"""

import argparse
import base64
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from prompts import (
    SYSTEM_PROMPT,
    DOMAIN_TAXONOMY,
    get_level1_prompt,
    get_level2_prompt,
    get_level2_check_prompt,
    get_level3_prompt,
    get_level3_check_prompt,
    get_merged_l1l2_prompt,
)


# ─────────────────────────────────────────────────────────────────────────────
# Frame helpers
# ─────────────────────────────────────────────────────────────────────────────

def format_mmss(total_seconds: int) -> str:
    minutes, seconds = divmod(max(0, int(total_seconds)), 60)
    return f"{minutes:02d}:{seconds:02d}"


def encode_frame_to_base64(frame_path: Path, resize_max_width: int = 0, jpeg_quality: int = 60) -> str:
    with Image.open(frame_path) as img:
        img = img.convert("RGB")
        if resize_max_width > 0 and img.width > resize_max_width:
            new_height = max(1, round(img.height * resize_max_width / img.width))
            img = img.resize((resize_max_width, new_height), Image.Resampling.LANCZOS)

        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


def frame_stem_to_index(frame_path: Path, fallback_index: int) -> int:
    """Frame filename stem → real second index (e.g. '0042' → 42)."""
    try:
        return int(frame_path.stem)
    except ValueError:
        return fallback_index


def get_all_frame_files(frame_dir: Path) -> list[Path]:
    """Return sorted list of all JPEG frames in a directory."""
    return sorted(frame_dir.glob("*.jpg"))


def sample_uniform(frame_files: list[Path], n_sample: int) -> list[Path]:
    """Uniformly sample exactly n_sample frames from a list."""
    if not frame_files or n_sample <= 0:
        return []
    if len(frame_files) <= n_sample:
        return list(frame_files)
    stride = (len(frame_files) - 1) / (n_sample - 1)
    return [frame_files[round(i * stride)] for i in range(n_sample)]


def encode_frame_files(
    frame_files: list[Path],
    resize_max_width: int = 0,
    jpeg_quality: int = 60,
) -> list[str]:
    """Encode frame files to base64 JPEG strings."""
    return [
        encode_frame_to_base64(fp, resize_max_width=resize_max_width, jpeg_quality=jpeg_quality)
        for fp in frame_files
    ]


def get_frames_in_time_range(
    frame_dir: Path,
    start_sec: int,
    end_sec: int,
) -> list[Path]:
    """Return frame files whose stem index falls within [start_sec, end_sec]."""
    result = []
    for fp in get_all_frame_files(frame_dir):
        idx = frame_stem_to_index(fp, -1)
        if start_sec <= idx <= end_sec:
            result.append(fp)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Level 1: Warped-Time mapping
# ─────────────────────────────────────────────────────────────────────────────

def build_warped_frame_mapping(
    frame_files: list[Path],
    n_sample: int = 32,
) -> tuple[list[Path], list[dict[str, int]]]:
    """
    Uniformly sample n_sample frames and build a warped_idx → real_sec mapping.

    Returns:
        sampled_files: Frame paths in warped order.
        mapping: List of {"warped_idx": 1..N, "real_sec": <original second>}.
    """
    sampled = sample_uniform(frame_files, n_sample)
    mapping = []
    for warped_idx, fp in enumerate(sampled, 1):
        real_sec = frame_stem_to_index(fp, warped_idx)
        mapping.append({"warped_idx": warped_idx, "real_sec": real_sec})
    return sampled, mapping


def warped_to_real_sec(warped_idx: int, mapping: list[dict[str, int]]) -> int | None:
    """Look up the real-second timestamp for a warped frame index."""
    for entry in mapping:
        if entry["warped_idx"] == warped_idx:
            return entry["real_sec"]
    # Clamp to nearest if exact match not found
    if not mapping:
        return None
    if warped_idx <= mapping[0]["warped_idx"]:
        return mapping[0]["real_sec"]
    if warped_idx >= mapping[-1]["warped_idx"]:
        return mapping[-1]["real_sec"]
    # Linear interpolation between surrounding entries
    for i in range(len(mapping) - 1):
        if mapping[i]["warped_idx"] <= warped_idx <= mapping[i + 1]["warped_idx"]:
            # Snap to closest
            left, right = mapping[i], mapping[i + 1]
            frac = (warped_idx - left["warped_idx"]) / (right["warped_idx"] - left["warped_idx"])
            return round(left["real_sec"] + frac * (right["real_sec"] - left["real_sec"]))
    return None



# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def clip_key_from_path(video_path: str) -> str:
    return Path(video_path).stem


def load_frame_meta(frame_dir: Path) -> dict[str, Any]:
    meta_path = frame_dir / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        with open(meta_path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def count_extracted_frames(frame_dir: Path) -> int:
    return len(list(frame_dir.glob("*.jpg")))


def build_record_from_frame_dir(frame_dir: Path) -> dict | None:
    meta = load_frame_meta(frame_dir)
    source_video_path = meta.get("source_video_path") or meta.get("record_video_path")
    if not source_video_path:
        jpg_files = sorted(frame_dir.glob("*.jpg"))
        if not jpg_files:
            return None
        source_video_path = frame_dir.name

    clip_key = meta.get("clip_key") or frame_dir.name
    annotation_end_sec = meta.get("annotation_end_sec")
    clip_duration = meta.get("annotation_end_sec") or meta.get("window_end_sec")

    return {
        "videos": [source_video_path],
        "metadata": {
            "clip_key": clip_key,
            "clip_end": annotation_end_sec,
            "clip_duration": clip_duration,
            "clip_start": meta.get("annotation_start_sec", 0),
            "video_id": meta.get("video_id") or clip_key,
            "source_mode": meta.get("source_mode"),
        },
    }


def load_records_from_frames_dir(frames_base: Path, limit: int = 0) -> list[dict]:
    records: list[dict] = []
    for frame_dir in sorted(p for p in frames_base.iterdir() if p.is_dir()):
        record = build_record_from_frame_dir(frame_dir)
        if record is not None:
            records.append(record)
    if limit > 0:
        records = records[:limit]
    return records


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI-compatible API client
# ─────────────────────────────────────────────────────────────────────────────

def call_vlm(
    api_base: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_text: str,
    frame_b64_list: list[str],
    frame_labels: list[str],
    max_tokens: int = 8192,
    temperature: float = 0.0,
    retries: int = 3,
) -> str:
    """
    Call a VLM endpoint with interleaved frame images.

    Args:
        frame_labels: Per-frame text labels (e.g. "[Frame 1]" or "[Timestamp 00:42 | Frame 42]").
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai is required: pip install openai")

    key = api_key or os.environ.get("NOVITA_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    client = OpenAI(api_key=key, base_url=api_base)

    content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    for i, b64 in enumerate(frame_b64_list):
        label = frame_labels[i] if i < len(frame_labels) else f"[Frame {i + 1}]"
        content.append({"type": "text", "text": label})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
        })

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": content},
    ]

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content
        except Exception as e:
            last_error = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"API call failed after {retries} attempts: {last_error}")


def parse_json_from_response(text: str) -> dict[str, Any]:
    """Extract and parse the first JSON block from the model response."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    import re
    m = re.search(r"```(?:json)?\s*(\{[\s\S]+?\})\s*```", text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m2 = re.search(r"\{[\s\S]+\}", text)
    if m2:
        try:
            return json.loads(m2.group(0))
        except json.JSONDecodeError:
            pass
    return {"_raw_response": text, "_parse_error": True}


def call_and_parse(
    api_base: str, api_key: str, model: str,
    system_prompt: str, prompt_text: str,
    frame_b64: list[str], frame_labels: list[str],
) -> dict[str, Any] | None:
    """Call VLM and parse response. Retries once on parse failure. Returns None on final failure."""
    raw = call_vlm(api_base, api_key, model, system_prompt, prompt_text, frame_b64, frame_labels)
    parsed = parse_json_from_response(raw)
    if parsed.get("_parse_error"):
        raw = call_vlm(api_base, api_key, model, system_prompt, prompt_text, frame_b64, frame_labels)
        parsed = parse_json_from_response(raw)
        if parsed.get("_parse_error"):
            return None
    return parsed


# ─────────────────────────────────────────────────────────────────────────────
# Per-clip annotation
# ─────────────────────────────────────────────────────────────────────────────

def annotate_clip(
    record: dict,
    frames_base: Path,
    output_dir: Path,
    level: int | str,
    api_base: str,
    api_key: str,
    model: str,
    max_frames_per_call: int,
    resize_max_width: int,
    jpeg_quality: int,
    overwrite: bool,
) -> dict:
    """
    Run the annotation pipeline for a single clip record.

    Returns a status dict: {clip_key, ok, error, skipped}.
    """
    videos = record.get("videos") or []
    if not videos:
        return {"clip_key": "?", "ok": False, "error": "no videos field", "skipped": False}

    vid_path = videos[0]
    meta = record.get("metadata") or {}
    key = str(meta.get("clip_key") or clip_key_from_path(vid_path))
    out_file = output_dir / f"{key}.json"

    # Load existing annotation if present
    existing: dict[str, Any] = {}
    if out_file.exists():
        try:
            with open(out_file, encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = {}

    # Skip if the requested level is already done and not overwriting
    level_key = f"level{level}" if level not in ("2c", "3c", "merged") else ("level2" if level == "2c" else "level3" if level == "3c" else None)
    is_check_mode = level in ("2c", "3c")
    if level == "merged":
        if not overwrite and existing.get("level1") is not None and existing.get("level2") is not None:
            return {"clip_key": key, "ok": True, "error": None, "skipped": True}
    elif is_check_mode:
        check_target = "level2" if level == "2c" else "level3"
        # For check mode, skip if the target level doesn't exist yet
        if existing.get(check_target) is None:
            return {"clip_key": key, "ok": False,
                    "error": f"{check_target} annotation missing; run that level first before check",
                    "skipped": False}
        # Also skip if already checked and not overwriting
        if not overwrite and existing.get(check_target, {}).get("_check_stats") is not None:
            return {"clip_key": key, "ok": True, "error": None, "skipped": True}
    else:
        if not overwrite and existing.get(level_key) is not None:
            return {"clip_key": key, "ok": True, "error": None, "skipped": True}

    # Load frames metadata
    frame_dir = frames_base / key
    frame_meta = load_frame_meta(frame_dir)

    clip_duration = float(
        frame_meta.get("annotation_end_sec")
        or meta.get("clip_end")
        or meta.get("clip_duration")
        or count_extracted_frames(frame_dir)
    )
    n_total_frames = count_extracted_frames(frame_dir)

    try:
        if level == 1:
            result_key, result_val = _annotate_level1(
                frame_dir, clip_duration,
                api_base, api_key, model,
                max_frames_per_call, resize_max_width, jpeg_quality,
            )
        elif level == 2:
            l1 = existing.get("level1")
            if l1 is None:
                return {"clip_key": key, "ok": False,
                        "error": "level1 annotation missing; run level 1 first", "skipped": False}
            result_key, result_val = _annotate_level2(
                frame_dir, clip_duration, l1,
                api_base, api_key, model,
                max_frames_per_call, resize_max_width, jpeg_quality,
            )
        elif level == 3:
            l2 = existing.get("level2")
            if l2 is None:
                return {"clip_key": key, "ok": False,
                        "error": "level2 annotation missing; run level 2 first", "skipped": False}
            result_key, result_val = _annotate_level3(
                frame_dir, clip_duration, l2,
                api_base, api_key, model,
                max_frames_per_call, resize_max_width, jpeg_quality,
            )
        elif level == "2c":
            l1 = existing.get("level1")
            l2 = existing.get("level2")
            if l1 is None or l2 is None:
                return {"clip_key": key, "ok": False,
                        "error": "level1+level2 annotations required for L2 check; run level 1 & 2 first",
                        "skipped": False}
            result_key, result_val = _check_level2(
                frame_dir, clip_duration, l1, l2,
                api_base, api_key, model,
                max_frames_per_call, resize_max_width, jpeg_quality,
            )
        elif level == "3c":
            l2 = existing.get("level2")
            l3 = existing.get("level3")
            if l2 is None or l3 is None:
                return {"clip_key": key, "ok": False,
                        "error": "level2+level3 annotations required for check; run level 2 & 3 first",
                        "skipped": False}
            result_key, result_val = _check_level3(
                frame_dir, clip_duration, l2, l3,
                api_base, api_key, model,
                max_frames_per_call, resize_max_width, jpeg_quality,
            )
        elif level == "merged":
            merged_result = _annotate_merged_l1l2(
                frame_dir, clip_duration,
                api_base, api_key, model,
                max_frames_per_call, resize_max_width, jpeg_quality,
            )
        else:
            return {"clip_key": key, "ok": False, "error": f"unsupported level {level}", "skipped": False}

    except RuntimeError as e:
        return {"clip_key": key, "ok": False, "error": str(e)[:300], "skipped": False}

    # Merge into existing annotation file
    ann: dict[str, Any] = {
        "clip_key": key,
        "video_path": vid_path,
        "source_video_path": frame_meta.get("source_video_path") or vid_path,
        "source_mode": frame_meta.get("source_mode") or "windowed_clip",
        "annotation_start_sec": frame_meta.get("annotation_start_sec"),
        "annotation_end_sec": frame_meta.get("annotation_end_sec") or clip_duration,
        "window_start_sec": frame_meta.get("window_start_sec", meta.get("clip_start")),
        "window_end_sec": frame_meta.get("window_end_sec", meta.get("clip_end")),
        "clip_duration_sec": clip_duration,
        "n_frames": n_total_frames,
        "frame_dir": str(frame_dir),
        "level1": None,
        "level2": None,
        "level3": None,
        **existing,
    }
    if level == "merged":
        ann.update(merged_result)  # overwrites level1, level2, domain, summary
    else:
        ann[result_key] = result_val
    ann["annotated_at"] = datetime.now(timezone.utc).isoformat()

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(ann, f, ensure_ascii=False, indent=2)

    return {"clip_key": key, "ok": True, "error": None, "skipped": False}


# ─────────────────────────────────────────────────────────────────────────────
# Level 1: Warped-Time Macro Phase Segmentation
# ─────────────────────────────────────────────────────────────────────────────

def _annotate_level1(
    frame_dir: Path,
    clip_duration: float,
    api_base: str, api_key: str, model: str,
    max_frames: int, resize_max_width: int, jpeg_quality: int,
) -> tuple[str, dict[str, Any]]:
    """
    Level 1: Warped-Time Segmentation.

    1. Uniformly sample N frames from the full video.
    2. Label them [Frame 1] .. [Frame N] — no real timestamps.
    3. Model predicts phase boundaries on the warped frame axis.
    4. Map warped frame indices back to real seconds via the mapping table.
    """
    all_frames = get_all_frame_files(frame_dir)
    if not all_frames:
        raise RuntimeError(f"no frames found in {frame_dir}")

    sampled_files, mapping = build_warped_frame_mapping(all_frames, n_sample=max_frames)
    frame_b64 = encode_frame_files(sampled_files, resize_max_width=resize_max_width, jpeg_quality=jpeg_quality)

    # Warped labels: [Frame 1], [Frame 2], ...
    frame_labels = [f"[Frame {entry['warped_idx']}]" for entry in mapping]

    prompt_text = get_level1_prompt(n_frames=len(sampled_files))
    parsed = call_and_parse(api_base, api_key, model, SYSTEM_PROMPT, prompt_text, frame_b64, frame_labels)
    if parsed is None:
        raise RuntimeError("level1 JSON parse failed after retry")

    # Map warped frame indices back to real seconds
    phases = parsed.get("macro_phases")
    if isinstance(phases, list):
        for phase in phases:
            if not isinstance(phase, dict):
                continue
            sf = phase.get("start_frame")
            ef = phase.get("end_frame")
            if isinstance(sf, (int, float)):
                phase["start_time"] = warped_to_real_sec(int(sf), mapping)
            if isinstance(ef, (int, float)):
                phase["end_time"] = warped_to_real_sec(int(ef), mapping)

    return "level1", {
        **parsed,
        "_warped_mapping": mapping,
        "_sampling": {
            "n_sampled_frames": len(sampled_files),
            "resize_max_width": resize_max_width,
            "jpeg_quality": jpeg_quality,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Merged L1+L2: Single-Call Phase + Event Detection + Domain
# ─────────────────────────────────────────────────────────────────────────────

def _split_merged_response(
    parsed: dict,
    n_sampled_frames: int,
    resize_max_width: int,
    jpeg_quality: int,
    clip_duration: float,
) -> tuple[dict, dict, str, str]:
    """Split merged VLM response into level1/level2 dicts + domain + summary.

    The VLM outputs events nested inside phases. This function:
    1. Extracts and validates domain/summary
    2. Strips nested events out of phases → flat events list
    3. Tags each event with parent_phase_id
    4. Re-numbers event_id globally by start_time
    """
    domain = parsed.get("domain", "other")
    if domain not in DOMAIN_TAXONOMY:
        domain = "other"
    summary = parsed.get("summary", "")

    raw_phases = parsed.get("macro_phases", [])
    l1_phases: list[dict] = []
    all_events: list[dict] = []

    for phase in raw_phases:
        if not isinstance(phase, dict):
            continue
        phase_id = phase.get("phase_id", len(l1_phases) + 1)

        # Extract nested events, then remove from the phase dict
        phase_events = phase.pop("events", [])

        # Validate phase timestamps
        st = phase.get("start_time")
        et = phase.get("end_time")
        if not (isinstance(st, (int, float)) and isinstance(et, (int, float)) and st < et):
            continue
        phase["start_time"] = int(st)
        phase["end_time"] = min(int(et), int(clip_duration))
        l1_phases.append(phase)

        # Collect events with parent linkage
        for ev in phase_events:
            if not isinstance(ev, dict):
                continue
            ev_st = ev.get("start_time")
            ev_et = ev.get("end_time")
            if not (isinstance(ev_st, (int, float)) and isinstance(ev_et, (int, float)) and ev_st < ev_et):
                continue
            ev["start_time"] = int(ev_st)
            ev["end_time"] = min(int(ev_et), int(clip_duration))
            ev["parent_phase_id"] = phase_id
            all_events.append(ev)

    # Sort events by start_time and re-number globally
    all_events.sort(key=lambda e: (e.get("start_time", 0), e.get("end_time", 0)))
    for i, ev in enumerate(all_events, 1):
        ev["event_id"] = i

    level1 = {
        "macro_phases": l1_phases,
        "_sampling": {
            "n_sampled_frames": n_sampled_frames,
            "resize_max_width": resize_max_width,
            "jpeg_quality": jpeg_quality,
        },
    }
    level2 = {"events": all_events}
    return level1, level2, domain, summary


def _annotate_merged_l1l2(
    frame_dir: Path,
    clip_duration: float,
    api_base: str, api_key: str, model: str,
    max_frames: int, resize_max_width: int, jpeg_quality: int,
) -> dict[str, Any]:
    """
    Merged L1+L2: Single VLM call for phases, events, domain, and summary.

    Uses real timestamps (not warped frames). Samples up to max_frames
    from the full video.

    Returns dict of annotation updates:
      {"level1": ..., "level2": ..., "domain": ..., "summary": ...}
    """
    all_frames = get_all_frame_files(frame_dir)
    if not all_frames:
        raise RuntimeError(f"no frames found in {frame_dir}")

    sampled = sample_uniform(all_frames, max_frames)
    frame_b64 = encode_frame_files(sampled, resize_max_width=resize_max_width, jpeg_quality=jpeg_quality)

    # Real-time labels (same format as L2)
    frame_labels = []
    for fp in sampled:
        idx = frame_stem_to_index(fp, 0)
        frame_labels.append(f"[Timestamp {format_mmss(idx)} | Frame {idx}]")

    duration = int(clip_duration)
    prompt_text = get_merged_l1l2_prompt(n_frames=len(sampled), duration_sec=duration)
    parsed = call_and_parse(api_base, api_key, model, SYSTEM_PROMPT, prompt_text, frame_b64, frame_labels)
    if parsed is None:
        raise RuntimeError("merged L1+L2 JSON parse failed after retry")

    level1, level2, domain, summary = _split_merged_response(
        parsed, len(sampled), resize_max_width, jpeg_quality, clip_duration,
    )

    return {
        "level1": level1,
        "level2": level2,
        "domain": domain,
        "summary": summary,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Level 2: Phase-Based Event Detection
# ─────────────────────────────────────────────────────────────────────────────

def _annotate_level2(
    frame_dir: Path,
    clip_duration: float,
    l1_result: dict[str, Any],
    api_base: str, api_key: str, model: str,
    max_frames: int, resize_max_width: int, jpeg_quality: int,
) -> tuple[str, dict[str, Any]]:
    """
    Level 2: Phase-Based Event Detection.

    Uses L1 macro phases as scope:
      1. For each L1 phase, extract frames in [start_time, end_time].
      2. Detect complete cooking events within that phase.
      3. Collect all events across phases (no NMS needed since phases don't overlap).
    """
    phases = l1_result.get("macro_phases", [])
    phases = sorted(
        [p for p in phases if isinstance(p, dict)],
        key=lambda p: (p.get("start_time", 0), p.get("end_time", 0)),
    )
    if not phases:
        raise RuntimeError("level1 macro_phases missing or empty")

    all_events: list[dict[str, Any]] = []
    phase_calls: list[dict[str, Any]] = []

    for phase in phases:
        phase_id = phase.get("phase_id", len(phase_calls) + 1)
        start_time = phase.get("start_time")
        end_time = phase.get("end_time")
        phase_name = phase.get("phase_name", "")
        narrative_summary = phase.get("narrative_summary", "")

        if not isinstance(start_time, (int, float)) or not isinstance(end_time, (int, float)):
            phase_calls.append({
                "phase_id": phase_id, "phase_name": phase_name,
                "start_time": start_time, "end_time": end_time,
                "skipped": True, "skip_reason": "invalid time",
            })
            continue

        phase_frames = get_frames_in_time_range(frame_dir, int(start_time), int(end_time))
        if not phase_frames:
            phase_calls.append({
                "phase_id": phase_id, "phase_name": phase_name,
                "start_time": start_time, "end_time": end_time,
                "skipped": True, "skip_reason": "no frames",
            })
            continue

        sampled = sample_uniform(phase_frames, max_frames)
        frame_b64 = encode_frame_files(sampled, resize_max_width=resize_max_width, jpeg_quality=jpeg_quality)

        # Real-time labels for L2
        frame_labels = []
        for fp in sampled:
            idx = frame_stem_to_index(fp, 0)
            frame_labels.append(f"[Timestamp {format_mmss(idx)} | Frame {idx}]")

        prompt_text = get_level2_prompt(
            int(start_time), int(end_time),
            phase_name=phase_name, narrative_summary=narrative_summary,
        )
        parsed = call_and_parse(api_base, api_key, model, SYSTEM_PROMPT, prompt_text, frame_b64, frame_labels)

        n_raw = 0
        if parsed is not None:
            events = parsed.get("events")
            if isinstance(events, list):
                for ev in events:
                    if isinstance(ev, dict):
                        st = ev.get("start_time")
                        et = ev.get("end_time")
                        if isinstance(st, (int, float)) and isinstance(et, (int, float)) and st < et:
                            ev["parent_phase_id"] = phase_id
                            all_events.append(ev)
                            n_raw += 1

        phase_calls.append({
            "phase_id": phase_id, "phase_name": phase_name,
            "start_time": start_time, "end_time": end_time,
            "n_sampled_frames": len(sampled),
            "n_events_raw": n_raw,
        })

    # Re-number event_ids globally
    all_events.sort(key=lambda e: (e.get("start_time", 0), e.get("end_time", 0)))
    for i, ev in enumerate(all_events, 1):
        ev["event_id"] = i

    return "level2", {
        "events": all_events,
        "_phase_calls": phase_calls,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Level 3: Local Temporal Grounding
# ─────────────────────────────────────────────────────────────────────────────

def _annotate_level3(
    frame_dir: Path,
    clip_duration: float,
    l2_result: dict[str, Any],
    api_base: str, api_key: str, model: str,
    max_frames: int, resize_max_width: int, jpeg_quality: int,
) -> tuple[str, dict[str, Any]]:
    """
    Level 3: Local Temporal Grounding.

    For each L2 event:
      1. Extract frames within the event time range.
      2. Use the event instruction as the action query.
      3. Model pinpoints atomic state-change moments.
    """
    events = l2_result.get("events", [])
    events = sorted(
        [e for e in events if isinstance(e, dict)],
        key=lambda e: (e.get("start_time", 0), e.get("end_time", 0)),
    )
    if not events:
        raise RuntimeError("level2 events missing or empty")

    all_results: list[dict[str, Any]] = []
    segment_calls: list[dict[str, Any]] = []

    for event in events:
        event_id = event.get("event_id", len(segment_calls) + 1)
        start_time = event.get("start_time")
        end_time = event.get("end_time")
        instruction = event.get("instruction", "")

        if not isinstance(start_time, (int, float)) or not isinstance(end_time, (int, float)):
            segment_calls.append({
                "event_id": event_id, "instruction": instruction,
                "start_time": start_time, "end_time": end_time,
                "skipped": True, "skip_reason": "invalid time",
            })
            continue

        ev_frames = get_frames_in_time_range(frame_dir, int(start_time), int(end_time))
        if not ev_frames:
            segment_calls.append({
                "event_id": event_id, "instruction": instruction,
                "start_time": start_time, "end_time": end_time,
                "skipped": True, "skip_reason": "no frames",
            })
            continue

        sampled = sample_uniform(ev_frames, max_frames)
        frame_b64 = encode_frame_files(sampled, resize_max_width=resize_max_width, jpeg_quality=jpeg_quality)

        # Real-time labels
        frame_labels = []
        for fp in sampled:
            idx = frame_stem_to_index(fp, 0)
            frame_labels.append(f"[Timestamp {format_mmss(idx)} | Frame {idx}]")

        prompt_text = get_level3_prompt(int(start_time), int(end_time), instruction)
        parsed = call_and_parse(api_base, api_key, model, SYSTEM_PROMPT, prompt_text, frame_b64, frame_labels)

        if parsed is None:
            segment_calls.append({
                "event_id": event_id, "instruction": instruction,
                "start_time": start_time, "end_time": end_time,
                "skipped": True, "skip_reason": "parse failed",
            })
            continue

        results = parsed.get("grounding_results")
        if isinstance(results, list):
            for r in results:
                if isinstance(r, dict):
                    r["parent_event_id"] = event_id
                    all_results.append(r)

        segment_calls.append({
            "event_id": event_id, "instruction": instruction,
            "start_time": start_time, "end_time": end_time,
            "n_sampled_frames": len(sampled),
            "n_grounding_results": len(results) if isinstance(results, list) else 0,
        })

    # Sort and re-number
    all_results.sort(key=lambda r: (r.get("start_time", 0), r.get("end_time", 0)))
    for i, r in enumerate(all_results, 1):
        r["action_id"] = i

    return "level3", {
        "grounding_results": all_results,
        "_segment_calls": segment_calls,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Level 3 Check: Model-based Quality Judge & Supplement
# ─────────────────────────────────────────────────────────────────────────────

def _apply_l2_check_results(
    existing_events: list[dict[str, Any]],
    check_parsed: dict[str, Any],
    parent_phase_id: int,
    phase_start: int,
    phase_end: int,
) -> list[dict[str, Any]]:
    """
    Apply check verdicts to existing L2 events for one phase.

    Returns the updated list of events for this phase.
    """
    event_by_id = {e.get("event_id"): e for e in existing_events}

    reviews = check_parsed.get("reviews") or []
    kept_events: list[dict[str, Any]] = []

    for review in reviews:
        if not isinstance(review, dict):
            continue
        eid = review.get("event_id")
        verdict = review.get("verdict", "keep")
        original = event_by_id.get(eid)
        if original is None:
            continue

        if verdict == "keep":
            kept_events.append(original)
        elif verdict == "revise":
            revised = review.get("revised")
            if isinstance(revised, dict):
                updated = dict(original)
                for field in ("start_time", "end_time", "instruction", "visual_keywords"):
                    if field in revised:
                        updated[field] = revised[field]
                st = updated.get("start_time")
                et = updated.get("end_time")
                if isinstance(st, (int, float)) and isinstance(et, (int, float)) and st < et:
                    updated["_checked"] = "revised"
                    kept_events.append(updated)
                else:
                    original["_checked"] = "revise_failed_kept_original"
                    kept_events.append(original)
            else:
                original["_checked"] = "revise_no_data_kept_original"
                kept_events.append(original)
        elif verdict == "remove":
            pass  # intentionally removed
        else:
            kept_events.append(original)

    # Any existing events not mentioned in reviews are kept
    reviewed_ids = {r.get("event_id") for r in reviews if isinstance(r, dict)}
    for e in existing_events:
        if e.get("event_id") not in reviewed_ids:
            kept_events.append(e)

    # Add supplements
    supplements = check_parsed.get("supplements") or []
    for sup in supplements:
        if not isinstance(sup, dict):
            continue
        st = sup.get("start_time")
        et = sup.get("end_time")
        if not (isinstance(st, (int, float)) and isinstance(et, (int, float)) and st < et):
            continue
        sup["parent_phase_id"] = parent_phase_id
        sup["_checked"] = "supplemented"
        kept_events.append(sup)

    return kept_events


def _check_level2(
    frame_dir: Path,
    clip_duration: float,
    l1_result: dict[str, Any],
    l2_result: dict[str, Any],
    api_base: str, api_key: str, model: str,
    max_frames: int, resize_max_width: int, jpeg_quality: int,
) -> tuple[str, dict[str, Any]]:
    """
    Level 2 Check: Model-based quality review and supplement for L2 annotations.

    For each L1 phase:
      1. Gather existing L2 events belonging to this phase.
      2. Re-read phase frames and call the judge model.
      3. Apply verdicts (keep/revise/remove) and add supplemented events.
    """
    phases = l1_result.get("macro_phases", [])
    phases = sorted(
        [p for p in phases if isinstance(p, dict)],
        key=lambda p: (p.get("start_frame", 0), p.get("end_frame", 0)),
    )
    if not phases:
        raise RuntimeError("level1 macro_phases missing or empty")

    existing_events = l2_result.get("events", [])

    # Build warped mapping to convert phase frame boundaries to real seconds
    warped_mapping = l1_result.get("_warped_mapping") or []
    warp_to_sec = {}
    for m in warped_mapping:
        warp_to_sec[m.get("warped_idx")] = m.get("real_sec")

    all_checked: list[dict[str, Any]] = []
    check_calls: list[dict[str, Any]] = []
    stats = {"kept": 0, "revised": 0, "removed": 0, "supplemented": 0}

    for phase in phases:
        phase_id = phase.get("phase_id")
        phase_name = phase.get("phase_name", "")
        narrative = phase.get("narrative_summary", "")

        # Convert warped frame bounds to real seconds
        sf = phase.get("start_frame", 0)
        ef = phase.get("end_frame", 0)
        phase_start = warp_to_sec.get(sf, sf)
        phase_end = warp_to_sec.get(ef, ef)

        # Gather events belonging to this phase
        phase_events = [
            e for e in existing_events
            if isinstance(e, dict) and e.get("parent_phase_id") == phase_id
        ]

        if not phase_events:
            check_calls.append({
                "phase_id": phase_id, "phase_name": phase_name,
                "start_time": phase_start, "end_time": phase_end,
                "skipped": True, "skip_reason": "no existing L2 events",
            })
            continue

        ph_frames = get_frames_in_time_range(frame_dir, int(phase_start), int(phase_end))
        if not ph_frames:
            check_calls.append({
                "phase_id": phase_id, "phase_name": phase_name,
                "start_time": phase_start, "end_time": phase_end,
                "skipped": True, "skip_reason": "no frames",
            })
            all_checked.extend(phase_events)
            continue

        sampled = sample_uniform(ph_frames, max_frames)
        frame_b64 = encode_frame_files(sampled, resize_max_width=resize_max_width, jpeg_quality=jpeg_quality)

        frame_labels = []
        for fp in sampled:
            idx = frame_stem_to_index(fp, 0)
            frame_labels.append(f"[Timestamp {format_mmss(idx)} | Frame {idx}]")

        prompt_text = get_level2_check_prompt(
            int(phase_start), int(phase_end), phase_name, narrative, phase_events,
        )
        parsed = call_and_parse(api_base, api_key, model, SYSTEM_PROMPT, prompt_text, frame_b64, frame_labels)

        if parsed is None:
            check_calls.append({
                "phase_id": phase_id, "phase_name": phase_name,
                "start_time": phase_start, "end_time": phase_end,
                "skipped": True, "skip_reason": "check parse failed",
            })
            all_checked.extend(phase_events)
            continue

        n_before = len(phase_events)
        checked = _apply_l2_check_results(
            phase_events, parsed, phase_id, int(phase_start), int(phase_end),
        )
        all_checked.extend(checked)

        reviews = parsed.get("reviews") or []
        for rv in reviews:
            if isinstance(rv, dict):
                v = rv.get("verdict", "keep")
                if v == "keep":
                    stats["kept"] += 1
                elif v == "revise":
                    stats["revised"] += 1
                elif v == "remove":
                    stats["removed"] += 1
        n_supplements = len(parsed.get("supplements") or [])
        stats["supplemented"] += n_supplements

        check_calls.append({
            "phase_id": phase_id, "phase_name": phase_name,
            "start_time": phase_start, "end_time": phase_end,
            "n_before": n_before,
            "n_after": len(checked),
            "n_supplements": n_supplements,
        })

    # Sort and re-number
    all_checked.sort(key=lambda e: (e.get("start_time", 0), e.get("end_time", 0)))
    for i, e in enumerate(all_checked, 1):
        e["event_id"] = i

    return "level2", {
        "events": all_checked,
        "_check_calls": check_calls,
        "_check_stats": stats,
    }


def _apply_l3_check_results(
    existing_results: list[dict[str, Any]],
    check_parsed: dict[str, Any],
    parent_event_id: int,
    event_start: int,
    event_end: int,
) -> list[dict[str, Any]]:
    """
    Apply check verdicts to existing L3 results for one event.

    Returns the updated list of grounding results for this event.
    """
    result_by_id = {r.get("action_id"): r for r in existing_results}

    reviews = check_parsed.get("reviews") or []
    kept_results: list[dict[str, Any]] = []

    for review in reviews:
        if not isinstance(review, dict):
            continue
        aid = review.get("action_id")
        verdict = review.get("verdict", "keep")
        original = result_by_id.get(aid)
        if original is None:
            continue

        if verdict == "keep":
            kept_results.append(original)
        elif verdict == "revise":
            revised = review.get("revised")
            if isinstance(revised, dict):
                updated = dict(original)
                for field in ("start_time", "end_time", "sub_action", "pre_state", "post_state"):
                    if field in revised:
                        updated[field] = revised[field]
                # Validate revised timestamps stay within event bounds
                st = updated.get("start_time")
                et = updated.get("end_time")
                if isinstance(st, (int, float)) and isinstance(et, (int, float)) and st < et:
                    updated["_checked"] = "revised"
                    kept_results.append(updated)
                else:
                    # Revised timestamps invalid, keep original
                    original["_checked"] = "revise_failed_kept_original"
                    kept_results.append(original)
            else:
                # No revised data provided, keep original
                original["_checked"] = "revise_no_data_kept_original"
                kept_results.append(original)
        elif verdict == "remove":
            pass  # intentionally removed
        else:
            # Unknown verdict, keep original
            kept_results.append(original)

    # Any existing results not mentioned in reviews are kept
    reviewed_ids = {r.get("action_id") for r in reviews if isinstance(r, dict)}
    for r in existing_results:
        if r.get("action_id") not in reviewed_ids:
            kept_results.append(r)

    # Add supplements
    supplements = check_parsed.get("supplements") or []
    for sup in supplements:
        if not isinstance(sup, dict):
            continue
        st = sup.get("start_time")
        et = sup.get("end_time")
        if not (isinstance(st, (int, float)) and isinstance(et, (int, float)) and st < et):
            continue
        sup["parent_event_id"] = parent_event_id
        sup["_checked"] = "supplemented"
        kept_results.append(sup)

    return kept_results


def _check_level3(
    frame_dir: Path,
    clip_duration: float,
    l2_result: dict[str, Any],
    l3_result: dict[str, Any],
    api_base: str, api_key: str, model: str,
    max_frames: int, resize_max_width: int, jpeg_quality: int,
) -> tuple[str, dict[str, Any]]:
    """
    Level 3 Check: Model-based quality review and supplement for L3 annotations.

    For each L2 event:
      1. Gather existing L3 grounding_results belonging to this event.
      2. Re-read event frames and call the judge model.
      3. Apply verdicts (keep/revise/remove) and add supplemented actions.
    """
    events = l2_result.get("events", [])
    events = sorted(
        [e for e in events if isinstance(e, dict)],
        key=lambda e: (e.get("start_time", 0), e.get("end_time", 0)),
    )
    if not events:
        raise RuntimeError("level2 events missing or empty")

    existing_l3 = l3_result.get("grounding_results", [])

    all_checked: list[dict[str, Any]] = []
    check_calls: list[dict[str, Any]] = []
    stats = {"kept": 0, "revised": 0, "removed": 0, "supplemented": 0}

    for event in events:
        event_id = event.get("event_id")
        start_time = event.get("start_time")
        end_time = event.get("end_time")
        instruction = event.get("instruction", "")

        if not isinstance(start_time, (int, float)) or not isinstance(end_time, (int, float)):
            check_calls.append({
                "event_id": event_id, "instruction": instruction,
                "skipped": True, "skip_reason": "invalid time",
            })
            continue

        # Gather existing L3 results for this event
        event_results = [
            r for r in existing_l3
            if isinstance(r, dict) and r.get("parent_event_id") == event_id
        ]

        if not event_results:
            # No existing L3 results for this event → nothing to check
            check_calls.append({
                "event_id": event_id, "instruction": instruction,
                "start_time": start_time, "end_time": end_time,
                "skipped": True, "skip_reason": "no existing L3 results",
            })
            continue

        ev_frames = get_frames_in_time_range(frame_dir, int(start_time), int(end_time))
        if not ev_frames:
            check_calls.append({
                "event_id": event_id, "instruction": instruction,
                "start_time": start_time, "end_time": end_time,
                "skipped": True, "skip_reason": "no frames",
            })
            # Keep originals when we can't check
            all_checked.extend(event_results)
            continue

        sampled = sample_uniform(ev_frames, max_frames)
        frame_b64 = encode_frame_files(sampled, resize_max_width=resize_max_width, jpeg_quality=jpeg_quality)

        frame_labels = []
        for fp in sampled:
            idx = frame_stem_to_index(fp, 0)
            frame_labels.append(f"[Timestamp {format_mmss(idx)} | Frame {idx}]")

        prompt_text = get_level3_check_prompt(
            int(start_time), int(end_time), instruction, event_results,
        )
        parsed = call_and_parse(api_base, api_key, model, SYSTEM_PROMPT, prompt_text, frame_b64, frame_labels)

        if parsed is None:
            # Check failed, keep originals
            check_calls.append({
                "event_id": event_id, "instruction": instruction,
                "start_time": start_time, "end_time": end_time,
                "skipped": True, "skip_reason": "check parse failed",
            })
            all_checked.extend(event_results)
            continue

        # Apply check results
        n_before = len(event_results)
        checked = _apply_l3_check_results(
            event_results, parsed, event_id, int(start_time), int(end_time),
        )
        all_checked.extend(checked)

        # Count stats
        reviews = parsed.get("reviews") or []
        for rv in reviews:
            if isinstance(rv, dict):
                v = rv.get("verdict", "keep")
                if v == "keep":
                    stats["kept"] += 1
                elif v == "revise":
                    stats["revised"] += 1
                elif v == "remove":
                    stats["removed"] += 1
        n_supplements = len(parsed.get("supplements") or [])
        stats["supplemented"] += n_supplements

        check_calls.append({
            "event_id": event_id, "instruction": instruction,
            "start_time": start_time, "end_time": end_time,
            "n_before": n_before,
            "n_after": len(checked),
            "n_supplements": n_supplements,
        })

    # Sort and re-number
    all_checked.sort(key=lambda r: (r.get("start_time", 0), r.get("end_time", 0)))
    for i, r in enumerate(all_checked, 1):
        r["action_id"] = i

    return "level3", {
        "grounding_results": all_checked,
        "_check_calls": check_calls,
        "_check_stats": stats,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hierarchical DVC annotation pipeline for YouCook2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--jsonl", default=None,
                        help="可选：输入 JSONL。若不提供，则直接遍历 --frames-dir 下所有样本。")
    parser.add_argument("--frames-dir", required=True,
                        help="Root directory of pre-extracted 1fps frames")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write per-clip annotation JSON files")
    parser.add_argument("--level", type=str, choices=["1", "2", "3", "2c", "3c", "merged"], default="1",
                        help="Annotation level (1/2/3=annotate, 2c/3c=check, merged=L1+L2+domain in one call)")
    parser.add_argument("--api-base", default="https://api.novita.ai/v3/openai",
                        help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", default="",
                        help="API key (prefers NOVITA_API_KEY env var, then OPENAI_API_KEY)")
    parser.add_argument("--model", default="pa/gmn-2.5-pr",
                        help="Model name to pass to the API")
    parser.add_argument("--max-frames-per-call", type=int, default=32,
                        help="Max frames per API call")
    parser.add_argument("--resize-max-width", type=int, default=384,
                        help="Resize frames before upload; <=0 disables resizing")
    parser.add_argument("--jpeg-quality", type=int, default=60,
                        help="JPEG quality for recompressing frames before upload")
    parser.add_argument("--workers", type=int, default=2,
                        help="Parallel annotation workers")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process at most N clips (0 = all)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-annotate even if the level is already done")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("NOVITA_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""

    frames_base = Path(args.frames_dir)
    if not frames_base.exists():
        print(f"ERROR: frames-dir not found: {frames_base}", file=sys.stderr)
        sys.exit(1)

    records: list[dict] = []
    if args.jsonl:
        jsonl_path = Path(args.jsonl)
        if not jsonl_path.exists():
            print(f"ERROR: JSONL not found: {jsonl_path}", file=sys.stderr)
            sys.exit(1)
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        if args.limit > 0:
            records = records[: args.limit]
    else:
        records = load_records_from_frames_dir(frames_base, limit=args.limit)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    level = int(args.level) if args.level in ("1", "2", "3") else args.level

    print(f"Annotating {len(records)} clips at Level {args.level}")
    print(f"API: {args.api_base}  model: {args.model}  workers: {args.workers}")
    print(f"resize_max_width={args.resize_max_width}  jpeg_quality={args.jpeg_quality}")
    if level == "merged":
        print(f"Merged mode: L1 phases + L2 events + domain + summary in one VLM call")
    elif level == 2:
        print(f"L2 phase-based: events detected per L1 macro phase")
    elif level == "2c":
        print(f"L2 check mode: model-based quality review & supplement for L2 events")
    elif level == "3c":
        print(f"L3 check mode: model-based quality review & supplement for L3 actions")
    print(f"Frames: {frames_base}  Output: {output_dir}\n")

    ok_count = skipped_count = error_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                annotate_clip,
                rec, frames_base, output_dir, level,
                args.api_base, api_key, args.model,
                args.max_frames_per_call,
                args.resize_max_width,
                args.jpeg_quality,
                args.overwrite,
            ): rec
            for rec in records
        }
        total = len(futures)
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            if res["skipped"]:
                skipped_count += 1
                if i % 50 == 0:
                    print(f"[{i}/{total}] SKIP   {res['clip_key']}")
            elif res["ok"]:
                ok_count += 1
                print(f"[{i}/{total}] OK     {res['clip_key']}")
            else:
                error_count += 1
                print(f"[{i}/{total}] ERROR  {res['clip_key']}: {res['error']}")

    print(f"\nFinished: {ok_count} annotated, {skipped_count} skipped, {error_count} errors")
    if error_count > 0:
        print("Re-run with --overwrite to retry failed clips.")


if __name__ == "__main__":
    main()
