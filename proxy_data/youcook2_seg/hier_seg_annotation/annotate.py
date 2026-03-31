#!/usr/bin/env python3
"""
annotate.py — Topology-adaptive hierarchical video annotation pipeline.

Annotation levels:
  merged: L1+L2+Topology single-call — full video frames (1fps)
          → domain + topology classification + macro phases + events
  3:      L3 grounding — topology-routed:
          procedural → per-event frames → state_change micro-actions
          periodic   → per-phase frames → repetition_unit micro-actions
          sequence/flat → skipped automatically
  2c/3c:  Quality check & supplement for L2/L3 results respectively

Recommended workflow:
    # Step 1: L1+L2+Topology merged annotation (1fps full-video frames)
    python annotate.py \\
        --frames-dir frames/ \\
        --output-dir annotations/ \\
        --level merged \\
        --api-base https://api.novita.ai/v3/openai \\
        --model pa/gmn-2.5-pr \\
        --workers 4

    # Step 2: Extract L3 frames (auto-routes by topology: event/phase/skip)
    python extract_frames.py \\
        --annotation-dir annotations/ \\
        --original-video-root /path/to/videos \\
        --output-dir frames_l3/ --fps 2

    # Step 3: L3 annotation (auto-skips sequence/flat)
    python annotate.py \\
        --frames-dir frames/ \\
        --l3-frames-dir frames_l3/ \\
        --output-dir annotations/ \\
        --level 3 \\
        --api-base https://api.novita.ai/v3/openai \\
        --model pa/gmn-2.5-pr \\
        --workers 4

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
    DOMAIN_L2_ALL,
    TOPOLOGY_TYPES,
    TOPOLOGY_TO_L2_MODE,
    TOPOLOGY_TO_L3_MODE,
    get_level2_check_prompt,
    get_level3_prompt,
    get_level3_check_prompt,
    get_merged_check_prompt,
    get_merged_l1l2_prompt,
)


# ─────────────────────────────────────────────────────────────────────────────
# Frame helpers
# ─────────────────────────────────────────────────────────────────────────────

def format_mmss(total_seconds: float) -> str:
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


def frame_index_to_sec(frame_index: int, fps: float = 1.0) -> float:
    """Convert 1-based ffmpeg frame index to real timestamp in seconds."""
    return (frame_index - 1) / fps


def get_all_frame_files(frame_dir: Path) -> list[Path]:
    """Return sorted list of all JPEG frames in a directory."""
    return sorted(frame_dir.glob("*.jpg"))


def sample_uniform(frame_files: list[Path], n_sample: int) -> list[Path]:
    """Uniformly sample up to n_sample frames. n_sample=0 returns all frames."""
    if not frame_files:
        return []
    if n_sample <= 0:
        return list(frame_files)
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
    start_sec: float,
    end_sec: float,
    fps: float = 1.0,
) -> list[Path]:
    """Return frame files whose timestamp (derived from stem index and fps) falls within [start_sec, end_sec]."""
    result = []
    for fp in get_all_frame_files(frame_dir):
        idx = frame_stem_to_index(fp, -1)
        time_sec = frame_index_to_sec(idx, fps)
        if start_sec <= time_sec <= end_sec:
            result.append(fp)
    return result


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
    """Extract and parse the first JSON object from the model response.

    Always returns a dict.  If the VLM produces a JSON array instead of
    an object, the first dict element is returned (with ``_unwrapped_array``
    flag); if no dict element exists the response is treated as a parse error.
    """
    def _ensure_dict(obj: Any) -> dict[str, Any]:
        """Unwrap a list → dict if possible, else signal parse error."""
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    item["_unwrapped_array"] = True
                    return item
            return {"_raw_response": text, "_parse_error": True}
        return {"_raw_response": text, "_parse_error": True}

    text = text.strip()
    try:
        return _ensure_dict(json.loads(text))
    except json.JSONDecodeError:
        pass
    import re
    m = re.search(r"```(?:json)?\s*(\{[\s\S]+?\})\s*```", text)
    if m:
        try:
            return _ensure_dict(json.loads(m.group(1)))
        except json.JSONDecodeError:
            pass
    m2 = re.search(r"\{[\s\S]+\}", text)
    if m2:
        try:
            return _ensure_dict(json.loads(m2.group(0)))
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
    level: str,
    api_base: str,
    api_key: str,
    model: str,
    max_frames_per_call: int,
    resize_max_width: int,
    jpeg_quality: int,
    overwrite: bool,
    l3_frames_dir: Path | None = None,
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
    is_check_mode = level in ("2c", "3c")
    if level == "merged":
        if not overwrite and existing.get("level1") is not None and existing.get("level2") is not None:
            return {"clip_key": key, "ok": True, "error": None, "skipped": True}
    elif level == "3":
        if not overwrite and existing.get("level3") is not None:
            return {"clip_key": key, "ok": True, "error": None, "skipped": True}
    elif is_check_mode:
        check_target = "level2" if level == "2c" else "level3"
        if existing.get(check_target) is None:
            return {"clip_key": key, "ok": False,
                    "error": f"{check_target} annotation missing; run that level first before check",
                    "skipped": False}
        if not overwrite and existing.get(check_target, {}).get("_check_stats") is not None:
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

    # For L3/3c, l3_frames_dir is the base dir for per-event subfolders:
    #   {l3_frames_dir}/{clip_key}_ev{event_id}/
    # frame_dir (full-video 1fps) is used as fallback when per-event dir is absent.

    try:
        if level == "merged":
            merged_result = _annotate_merged_l1l2(
                frame_dir, clip_duration,
                api_base, api_key, model,
                max_frames_per_call, resize_max_width, jpeg_quality,
            )
        elif level == "3":
            # ── Topology-aware L3 routing ──
            topology_type = existing.get("topology_type", "procedural")
            topology_confidence = existing.get("topology_confidence", 1.0)
            l3_mode = existing.get("l3_mode") or TOPOLOGY_TO_L3_MODE.get(topology_type, "state_change")

            # Conservative threshold: low confidence → skip L3
            if topology_confidence < 0.6:
                l3_mode = "skip"

            if l3_mode == "skip":
                result_key, result_val = "level3", {
                    "micro_type": "skip",
                    "grounding_results": [],
                    "_segment_calls": [],
                    "_skip_reason": f"l3_mode=skip (topology={topology_type}, conf={topology_confidence:.2f})",
                }
            elif topology_type == "periodic":
                # periodic: L3 from phases, not events
                l1 = existing.get("level1")
                if l1 is None:
                    return {"clip_key": key, "ok": False,
                            "error": "level1 annotation missing; run merged first", "skipped": False}
                result_key, result_val = _annotate_level3(
                    frame_dir, clip_duration, existing.get("level2") or {"events": []},
                    api_base, api_key, model,
                    max_frames_per_call, resize_max_width, jpeg_quality,
                    l3_base=l3_frames_dir,
                    clip_key_str=key,
                    topology_type=topology_type,
                    l1_result=l1,
                )
            else:
                # default: leaf-node routing (events + eventless phases)
                l2 = existing.get("level2")
                if l2 is None:
                    return {"clip_key": key, "ok": False,
                            "error": "level2 annotation missing; run merged first", "skipped": False}
                result_key, result_val = _annotate_level3(
                    frame_dir, clip_duration, l2,
                    api_base, api_key, model,
                    max_frames_per_call, resize_max_width, jpeg_quality,
                    l3_base=l3_frames_dir,
                    clip_key_str=key,
                    topology_type=topology_type,
                    l1_result=existing.get("level1"),
                )
        elif level == "2c":
            l1 = existing.get("level1")
            l2 = existing.get("level2")
            if l1 is None or l2 is None:
                return {"clip_key": key, "ok": False,
                        "error": "level1+level2 annotations required for L2 check; run merged first",
                        "skipped": False}
            result_key, result_val = _check_level2(
                frame_dir, clip_duration, l1, l2,
                api_base, api_key, model,
                max_frames_per_call, resize_max_width, jpeg_quality,
            )
        elif level == "merged_c":
            l1 = existing.get("level1")
            l2 = existing.get("level2")
            if l1 is None or l2 is None:
                return {"clip_key": key, "ok": False,
                        "error": "level1+level2 annotations required for merged check; run merged first",
                        "skipped": False}
            checked_l1, checked_l2 = _check_merged_l1l2(
                frame_dir, clip_duration, l1, l2,
                summary=existing.get("summary", ""),
                topology_type=existing.get("topology_type", "procedural"),
                topology_confidence=float(existing.get("topology_confidence", 0.5)),
                api_base=api_base, api_key=api_key, model=model,
                max_frames=max_frames_per_call,
                resize_max_width=resize_max_width, jpeg_quality=jpeg_quality,
                global_phase_criterion=existing.get("global_phase_criterion", ""),
            )
            # merged_c writes both level1 and level2 — handled specially below
            result_key = "merged_c"
            result_val = {"level1": checked_l1, "level2": checked_l2}
        elif level == "3c":
            l2 = existing.get("level2")
            l3 = existing.get("level3")
            if l2 is None or l3 is None:
                return {"clip_key": key, "ok": False,
                        "error": "level2+level3 annotations required for check; run merged & level 3 first",
                        "skipped": False}
            result_key, result_val = _check_level3(
                frame_dir, clip_duration, l2, l3,
                api_base, api_key, model,
                max_frames_per_call, resize_max_width, jpeg_quality,
                l3_base=l3_frames_dir,
                clip_key_str=key,
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
        ann.update(merged_result)  # overwrites level1, level2, domain_l1, domain_l2, summary
    elif level == "merged_c":
        ann["level1"] = result_val["level1"]
        ann["level2"] = result_val["level2"]
    else:
        ann[result_key] = result_val
    ann["annotated_at"] = datetime.now(timezone.utc).isoformat()

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(ann, f, ensure_ascii=False, indent=2)

    return {"clip_key": key, "ok": True, "error": None, "skipped": False}


# ─────────────────────────────────────────────────────────────────────────────
# Merged L1+L2: Single-Call Phase + Event Detection + Domain
# ─────────────────────────────────────────────────────────────────────────────

def _split_merged_response(
    parsed: dict,
    n_sampled_frames: int,
    resize_max_width: int,
    jpeg_quality: int,
    clip_duration: float,
) -> dict[str, Any]:
    """Split merged VLM response into a flat dict of annotation fields.

    The VLM outputs events nested inside phases. This function:
    1. Extracts and validates domain_l1/domain_l2/summary
    2. Extracts and validates topology_type/confidence/reason/l2_mode/l3_mode
    3. Strips nested events out of phases → flat events list
    4. Tags each event with parent_phase_id
    5. Re-numbers event_id globally by start_time

    Returns dict with keys: level1, level2, domain_l1, domain_l2, summary,
        topology_type, topology_confidence, topology_reason, l2_mode, l3_mode.
    """
    domain_l1 = parsed.get("domain_l1", "other")
    if domain_l1 not in DOMAIN_TAXONOMY:
        domain_l1 = "other"
    domain_l2 = parsed.get("domain_l2", "other")
    if domain_l2 not in DOMAIN_L2_ALL:
        domain_l2 = "other"
    summary = parsed.get("summary", "")
    global_phase_criterion = parsed.get("global_phase_criterion", "")

    # ── Topology extraction ──
    topology_type = parsed.get("topology_type", "procedural")
    if topology_type not in TOPOLOGY_TYPES:
        topology_type = "procedural"

    topology_confidence = parsed.get("topology_confidence")
    if not isinstance(topology_confidence, (int, float)):
        topology_confidence = 0.5
    topology_confidence = max(0.0, min(1.0, float(topology_confidence)))

    topology_reason = str(parsed.get("topology_reason", ""))

    l2_mode = parsed.get("l2_mode") or TOPOLOGY_TO_L2_MODE.get(topology_type, "workflow")
    l3_mode = TOPOLOGY_TO_L3_MODE.get(topology_type, "state_change")

    # Conservative override: very low confidence → treat as flat
    if topology_confidence < 0.5:
        topology_type = "flat"
        l2_mode = "skip"
        l3_mode = "skip"

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
    return {
        "level1": level1,
        "level2": level2,
        "domain_l1": domain_l1,
        "domain_l2": domain_l2,
        "summary": summary,
        "global_phase_criterion": global_phase_criterion,
        "topology_type": topology_type,
        "topology_confidence": topology_confidence,
        "topology_reason": topology_reason,
        "l2_mode": l2_mode,
        "l3_mode": l3_mode,
    }


def _annotate_merged_l1l2(
    frame_dir: Path,
    clip_duration: float,
    api_base: str, api_key: str, model: str,
    max_frames: int, resize_max_width: int, jpeg_quality: int,
) -> dict[str, Any]:
    """
    Merged L1+L2+Topology: Single VLM call for topology, phases, events, domain, and summary.

    Uses real timestamps (not warped frames). Samples up to max_frames
    from the full video.

    Returns dict of annotation updates including topology_type, l2_mode, l3_mode.
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

    return _split_merged_response(
        parsed, len(sampled), resize_max_width, jpeg_quality, clip_duration,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Level 3: Local Temporal Grounding
# ─────────────────────────────────────────────────────────────────────────────

def _annotate_level3(
    frame_dir: Path,
    clip_duration: float,
    l2_result: dict[str, Any],
    api_base: str, api_key: str, model: str,
    max_frames: int, resize_max_width: int, jpeg_quality: int,
    l3_base: Path | None = None,
    clip_key_str: str = "",
    topology_type: str = "procedural",
    l1_result: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    Level 3: Local Temporal Grounding (topology-aware).

    Source selection by topology:
      - procedural: iterate L2 events → per-event frames in {l3_base}/{clip_key}_ev{id}/
      - periodic:   iterate L1 phases → per-phase frames in {l3_base}/{clip_key}_ph{id}/
    Falls back to filtering the full-video 1fps frame dir when dedicated dir is absent.
    """
    # ── Build source list based on topology (leaf-node collection) ──
    sources: list[dict[str, Any]] = []

    if topology_type == "periodic" and l1_result is not None:
        # periodic: L3 sources from L1 phases
        phases = l1_result.get("macro_phases", [])
        for phase in phases:
            if not isinstance(phase, dict):
                continue
            sources.append({
                "source_id": phase.get("phase_id", len(sources) + 1),
                "start_time": phase.get("start_time"),
                "end_time": phase.get("end_time"),
                "instruction": phase.get("narrative_summary") or phase.get("phase_name", ""),
                "_source_type": "phase",
            })
    else:
        # Leaf-node collection: phases without events become leaf nodes,
        # phases with events contribute their events as leaf nodes.
        events = l2_result.get("events", [])
        l1_phases = (l1_result or {}).get("macro_phases", []) if l1_result else []

        if l1_phases:
            # Build phase_id → events mapping
            phase_events: dict[int, list[dict]] = {}
            for ev in events:
                if not isinstance(ev, dict):
                    continue
                pid = ev.get("parent_phase_id")
                if pid is not None:
                    phase_events.setdefault(pid, []).append(ev)

            for phase in l1_phases:
                if not isinstance(phase, dict):
                    continue
                pid = phase.get("phase_id")
                children = phase_events.get(pid, [])
                if children:
                    # Phase has events → events are leaf nodes
                    for ev in children:
                        sources.append({
                            "source_id": ev.get("event_id", len(sources) + 1),
                            "start_time": ev.get("start_time"),
                            "end_time": ev.get("end_time"),
                            "instruction": ev.get("instruction", ""),
                            "_source_type": "event",
                        })
                else:
                    # Phase has no events → phase itself is leaf node
                    sources.append({
                        "source_id": phase.get("phase_id", len(sources) + 1),
                        "start_time": phase.get("start_time"),
                        "end_time": phase.get("end_time"),
                        "instruction": phase.get("narrative_summary") or phase.get("phase_name", ""),
                        "_source_type": "phase",
                    })
        else:
            # Fallback: no L1 data available, use events directly (backward-compat)
            for ev in events:
                if not isinstance(ev, dict):
                    continue
                sources.append({
                    "source_id": ev.get("event_id", len(sources) + 1),
                    "start_time": ev.get("start_time"),
                    "end_time": ev.get("end_time"),
                    "instruction": ev.get("instruction", ""),
                    "_source_type": "event",
                })

    sources.sort(key=lambda s: (s.get("start_time") or 0, s.get("end_time") or 0))

    if not sources:
        raise RuntimeError("no sources (events/phases) available for L3 grounding")

    meta = load_frame_meta(frame_dir)
    fps = float(meta.get("fps", 1.0))

    micro_type = "repetition_unit" if topology_type == "periodic" else "state_change"
    all_results: list[dict[str, Any]] = []
    segment_calls: list[dict[str, Any]] = []

    for source in sources:
        source_id = source["source_id"]
        start_time = source["start_time"]
        end_time = source["end_time"]
        instruction = source["instruction"]
        source_type = source["_source_type"]
        key_prefix = "ph" if source_type == "phase" else "ev"

        if not isinstance(start_time, (int, float)) or not isinstance(end_time, (int, float)):
            segment_calls.append({
                f"parent_{source_type}_id": source_id, "instruction": instruction,
                "start_time": start_time, "end_time": end_time,
                "skipped": True, "skip_reason": "invalid time",
            })
            continue

        # Try per-source frame dir first (high-fps extracted by extract_frames.py L3 mode)
        src_dir = (l3_base / f"{clip_key_str}_{key_prefix}{source_id}") if (l3_base and clip_key_str) else None
        using_dedicated = src_dir is not None and src_dir.exists() and len(list(src_dir.glob("*.jpg"))) > 0

        if using_dedicated:
            src_meta = load_frame_meta(src_dir)
            src_fps = float(src_meta.get("fps", 2.0))
            src_start = float(src_meta.get("event_start_sec", start_time))
            src_frames = get_all_frame_files(src_dir)

            def make_labels(frames: list[Path], _fps: float = src_fps, _start: float = src_start) -> list[str]:
                labels = []
                for fp in frames:
                    idx = frame_stem_to_index(fp, 0)
                    t_abs = _start + frame_index_to_sec(idx, _fps)
                    labels.append(f"[Timestamp {format_mmss(t_abs)} | Frame {idx}]")
                return labels
        else:
            src_fps = fps
            src_frames = get_frames_in_time_range(frame_dir, start_time, end_time, fps)

            def make_labels(frames: list[Path], _fps: float = fps) -> list[str]:
                labels = []
                for fp in frames:
                    idx = frame_stem_to_index(fp, 0)
                    t = frame_index_to_sec(idx, _fps)
                    labels.append(f"[Timestamp {format_mmss(t)} | Frame {idx}]")
                return labels

        if not src_frames:
            segment_calls.append({
                f"parent_{source_type}_id": source_id, "instruction": instruction,
                "start_time": start_time, "end_time": end_time,
                "skipped": True, "skip_reason": "no frames",
            })
            continue

        sampled = sample_uniform(src_frames, max_frames)
        frame_b64 = encode_frame_files(sampled, resize_max_width=resize_max_width, jpeg_quality=jpeg_quality)
        frame_labels = make_labels(sampled)

        prompt_text = get_level3_prompt(
            int(start_time), int(end_time), instruction,
            topology_type=topology_type,
        )
        parsed = call_and_parse(api_base, api_key, model, SYSTEM_PROMPT, prompt_text, frame_b64, frame_labels)

        if parsed is None:
            segment_calls.append({
                f"parent_{source_type}_id": source_id, "instruction": instruction,
                "start_time": start_time, "end_time": end_time,
                "skipped": True, "skip_reason": "parse failed",
            })
            continue

        results = parsed.get("grounding_results")
        source_criterion = parsed.get("micro_split_criterion", "")
        if isinstance(results, list):
            for r in results:
                if isinstance(r, dict):
                    r[f"parent_{source_type}_id"] = source_id
                    all_results.append(r)

        segment_calls.append({
            f"parent_{source_type}_id": source_id, "instruction": instruction,
            "start_time": start_time, "end_time": end_time,
            "n_sampled_frames": len(sampled),
            "n_grounding_results": len(results) if isinstance(results, list) else 0,
            "frame_source": f"per_{source_type}" if using_dedicated else "full_video_filtered",
            "micro_split_criterion": source_criterion,
        })

    # Sort and re-number
    all_results.sort(key=lambda r: (r.get("start_time", 0), r.get("end_time", 0)))
    for i, r in enumerate(all_results, 1):
        r["action_id"] = i

    # Pick first non-empty micro_split_criterion from segment calls
    micro_split_criterion = ""
    for sc in segment_calls:
        c = sc.get("micro_split_criterion", "")
        if c:
            micro_split_criterion = c
            break

    return "level3", {
        "micro_type": micro_type,
        "micro_split_criterion": micro_split_criterion,
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
        key=lambda p: (p.get("start_time", 0), p.get("end_time", 0)),
    )
    if not phases:
        raise RuntimeError("level1 macro_phases missing or empty")

    existing_events = l2_result.get("events", [])

    all_checked: list[dict[str, Any]] = []
    check_calls: list[dict[str, Any]] = []
    stats = {"kept": 0, "revised": 0, "removed": 0, "supplemented": 0}

    for phase in phases:
        phase_id = phase.get("phase_id")
        phase_name = phase.get("phase_name", "")
        narrative = phase.get("narrative_summary", "")

        phase_start = phase.get("start_time", 0)
        phase_end = phase.get("end_time", 0)

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

        # Count stats from actual apply results
        for r in checked:
            tag = r.get("_checked")
            if tag == "revised":
                stats["revised"] += 1
            elif tag == "supplemented":
                stats["supplemented"] += 1
            else:
                stats["kept"] += 1
        stats["removed"] += n_before - sum(1 for r in checked if r.get("_checked") != "supplemented")

        check_calls.append({
            "phase_id": phase_id, "phase_name": phase_name,
            "start_time": phase_start, "end_time": phase_end,
            "n_before": n_before,
            "n_after": len(checked),
            "n_supplements": sum(1 for r in checked if r.get("_checked") == "supplemented"),
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
    l3_base: Path | None = None,
    clip_key_str: str = "",
) -> tuple[str, dict[str, Any]]:
    """
    Level 3 Check: Model-based quality review and supplement for L3 annotations.

    For each L2 event:
      1. Prefer per-event frames from {l3_base}/{clip_key}_ev{event_id}/ (high-fps).
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

    meta = load_frame_meta(frame_dir)
    fps = float(meta.get("fps", 1.0))

    existing_l3 = l3_result.get("grounding_results", [])
    micro_type = l3_result.get("micro_type", "state_change")
    micro_split_criterion = l3_result.get("micro_split_criterion", "")

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
            check_calls.append({
                "event_id": event_id, "instruction": instruction,
                "start_time": start_time, "end_time": end_time,
                "skipped": True, "skip_reason": "no existing L3 results",
            })
            continue

        # Try per-event dir first
        ev_dir = (l3_base / f"{clip_key_str}_ev{event_id}") if (l3_base and clip_key_str) else None
        using_per_event = ev_dir is not None and ev_dir.exists() and len(list(ev_dir.glob("*.jpg"))) > 0

        if using_per_event:
            ev_meta = load_frame_meta(ev_dir)
            ev_fps = float(ev_meta.get("fps", 2.0))
            ev_start = float(ev_meta.get("event_start_sec", start_time))
            ev_frames = get_all_frame_files(ev_dir)

            def make_labels(frames: list[Path], _ev_fps: float = ev_fps, _ev_start: float = ev_start) -> list[str]:
                labels = []
                for fp in frames:
                    idx = frame_stem_to_index(fp, 0)
                    t_abs = _ev_start + frame_index_to_sec(idx, _ev_fps)
                    labels.append(f"[Timestamp {format_mmss(t_abs)} | Frame {idx}]")
                return labels
        else:
            ev_frames = get_frames_in_time_range(frame_dir, start_time, end_time, fps)

            def make_labels(frames: list[Path], _fps: float = fps) -> list[str]:
                labels = []
                for fp in frames:
                    idx = frame_stem_to_index(fp, 0)
                    t = frame_index_to_sec(idx, _fps)
                    labels.append(f"[Timestamp {format_mmss(t)} | Frame {idx}]")
                return labels

        if not ev_frames:
            check_calls.append({
                "event_id": event_id, "instruction": instruction,
                "start_time": start_time, "end_time": end_time,
                "skipped": True, "skip_reason": "no frames",
            })
            all_checked.extend(event_results)
            continue

        sampled = sample_uniform(ev_frames, max_frames)
        frame_b64 = encode_frame_files(sampled, resize_max_width=resize_max_width, jpeg_quality=jpeg_quality)
        frame_labels = make_labels(sampled)

        prompt_text = get_level3_check_prompt(
            int(start_time), int(end_time), instruction, event_results,
            micro_type=micro_type, micro_split_criterion=micro_split_criterion,
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

        # Count stats from actual apply results
        for r in checked:
            tag = r.get("_checked")
            if tag == "revised":
                stats["revised"] += 1
            elif tag == "supplemented":
                stats["supplemented"] += 1
            else:
                stats["kept"] += 1
        stats["removed"] += n_before - sum(1 for r in checked if r.get("_checked") != "supplemented")

        check_calls.append({
            "event_id": event_id, "instruction": instruction,
            "start_time": start_time, "end_time": end_time,
            "n_before": n_before,
            "n_after": len(checked),
            "n_supplements": sum(1 for r in checked if r.get("_checked") == "supplemented"),
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
# Merged L1+L2 Check: Simultaneous Phase + Event Quality Review
# ─────────────────────────────────────────────────────────────────────────────

def _apply_l1_check_results(
    existing_phases: list[dict[str, Any]],
    check_parsed: dict[str, Any],
    clip_duration: float,
) -> list[dict[str, Any]]:
    """
    Apply check verdicts to existing L1 phases.

    Returns the updated list of macro phases.
    """
    phase_by_id = {p.get("phase_id"): p for p in existing_phases}

    reviews = check_parsed.get("phase_reviews") or []
    kept_phases: list[dict[str, Any]] = []

    for review in reviews:
        if not isinstance(review, dict):
            continue
        pid = review.get("phase_id")
        verdict = review.get("verdict", "keep")
        original = phase_by_id.get(pid)
        if original is None:
            continue

        if verdict == "keep":
            kept_phases.append(original)
        elif verdict == "revise":
            revised = review.get("revised")
            if isinstance(revised, dict):
                updated = dict(original)
                for field in ("start_time", "end_time", "phase_name", "narrative_summary"):
                    if field in revised:
                        updated[field] = revised[field]
                st = updated.get("start_time")
                et = updated.get("end_time")
                if isinstance(st, (int, float)) and isinstance(et, (int, float)) and st < et:
                    updated["_checked"] = "revised"
                    kept_phases.append(updated)
                else:
                    original["_checked"] = "revise_failed_kept_original"
                    kept_phases.append(original)
            else:
                original["_checked"] = "revise_no_data_kept_original"
                kept_phases.append(original)
        elif verdict == "remove":
            pass  # intentionally removed
        else:
            kept_phases.append(original)

    # Any existing phases not mentioned in reviews are kept
    reviewed_ids = {r.get("phase_id") for r in reviews if isinstance(r, dict)}
    for p in existing_phases:
        if p.get("phase_id") not in reviewed_ids:
            kept_phases.append(p)

    # Add supplements
    supplements = check_parsed.get("phase_supplements") or []
    for sup in supplements:
        if not isinstance(sup, dict):
            continue
        st = sup.get("start_time")
        et = sup.get("end_time")
        if not (isinstance(st, (int, float)) and isinstance(et, (int, float)) and st < et):
            continue
        sup["start_time"] = int(st)
        sup["end_time"] = min(int(et), int(clip_duration))
        sup["_checked"] = "supplemented"
        kept_phases.append(sup)

    return kept_phases


def _check_merged_l1l2(
    frame_dir: Path,
    clip_duration: float,
    l1_result: dict[str, Any],
    l2_result: dict[str, Any],
    summary: str,
    topology_type: str,
    topology_confidence: float,
    api_base: str, api_key: str, model: str,
    max_frames: int, resize_max_width: int, jpeg_quality: int,
    global_phase_criterion: str = "",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Merged L1+L2 Check: Model-based quality review for both L1 phases and L2 events.

    Sends full-video 1fps frames with existing L1+L2 annotations for simultaneous
    review, mirroring the merged annotation call.

    Returns (checked_l1, checked_l2) dicts.
    """
    phases = l1_result.get("macro_phases", [])
    phases = sorted(
        [p for p in phases if isinstance(p, dict)],
        key=lambda p: (p.get("start_time", 0), p.get("end_time", 0)),
    )
    if not phases:
        raise RuntimeError("level1 macro_phases missing or empty")

    existing_events = l2_result.get("events", [])

    # Sample full-video frames (same as merged annotation)
    all_frames = get_all_frame_files(frame_dir)
    if not all_frames:
        raise RuntimeError(f"no frames found in {frame_dir}")

    sampled = sample_uniform(all_frames, max_frames)
    frame_b64 = encode_frame_files(sampled, resize_max_width=resize_max_width, jpeg_quality=jpeg_quality)

    frame_labels = []
    for fp in sampled:
        idx = frame_stem_to_index(fp, 0)
        frame_labels.append(f"[Timestamp {format_mmss(idx)} | Frame {idx}]")

    prompt_text = get_merged_check_prompt(
        n_frames=len(sampled),
        duration_sec=int(clip_duration),
        summary=summary,
        topology_type=topology_type,
        topology_confidence=topology_confidence,
        l1_phases=phases,
        l2_events=existing_events,
        global_phase_criterion=global_phase_criterion,
    )
    parsed = call_and_parse(api_base, api_key, model, SYSTEM_PROMPT, prompt_text, frame_b64, frame_labels)

    if parsed is None:
        raise RuntimeError("merged check: VLM parse failed")

    # ── Apply L1 phase verdicts ──
    checked_phases = _apply_l1_check_results(phases, parsed, clip_duration)

    # Sort and re-number phases
    checked_phases.sort(key=lambda p: (p.get("start_time", 0), p.get("end_time", 0)))
    for i, p in enumerate(checked_phases, 1):
        p["phase_id"] = i

    # Build old→new phase_id mapping for event parent_phase_id fixup
    # (supplements get new IDs from re-numbering above)
    valid_phase_ids = {p["phase_id"] for p in checked_phases}

    # ── Apply L2 event verdicts ──
    event_by_id = {e.get("event_id"): e for e in existing_events}
    event_reviews = parsed.get("event_reviews") or []
    kept_events: list[dict[str, Any]] = []

    for review in event_reviews:
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
            pass
        else:
            kept_events.append(original)

    # Any existing events not mentioned in reviews are kept
    reviewed_eids = {r.get("event_id") for r in event_reviews if isinstance(r, dict)}
    for e in existing_events:
        if e.get("event_id") not in reviewed_eids:
            kept_events.append(e)

    # Add event supplements
    event_supplements = parsed.get("event_supplements") or []
    for sup in event_supplements:
        if not isinstance(sup, dict):
            continue
        st = sup.get("start_time")
        et = sup.get("end_time")
        if not (isinstance(st, (int, float)) and isinstance(et, (int, float)) and st < et):
            continue
        sup["_checked"] = "supplemented"
        kept_events.append(sup)

    # Drop events whose parent_phase_id no longer exists
    final_events = []
    for e in kept_events:
        ppid = e.get("parent_phase_id")
        if ppid is not None and ppid not in valid_phase_ids:
            continue  # orphaned by phase removal
        final_events.append(e)

    # Sort and re-number events
    final_events.sort(key=lambda e: (e.get("start_time", 0), e.get("end_time", 0)))
    for i, e in enumerate(final_events, 1):
        e["event_id"] = i

    # ── Aggregate stats from actual results ──
    l1_stats = {"kept": 0, "revised": 0, "removed": 0, "supplemented": 0}
    original_phase_ids = {p.get("phase_id") for p in phases}
    for p in checked_phases:
        tag = p.get("_checked")
        if tag == "revised":
            l1_stats["revised"] += 1
        elif tag == "supplemented":
            l1_stats["supplemented"] += 1
        else:
            l1_stats["kept"] += 1
    l1_stats["removed"] = len(original_phase_ids) - (l1_stats["kept"] + l1_stats["revised"])
    if l1_stats["removed"] < 0:
        l1_stats["removed"] = 0

    l2_stats = {"kept": 0, "revised": 0, "removed": 0, "supplemented": 0}
    original_event_ids = {e.get("event_id") for e in existing_events}
    for e in final_events:
        tag = e.get("_checked")
        if tag == "revised":
            l2_stats["revised"] += 1
        elif tag == "supplemented":
            l2_stats["supplemented"] += 1
        elif e.get("event_id") in original_event_ids or tag is None:
            l2_stats["kept"] += 1
    l2_stats["removed"] = len(original_event_ids) - (l2_stats["kept"] + l2_stats["revised"])
    if l2_stats["removed"] < 0:
        l2_stats["removed"] = 0

    checked_l1 = {
        "macro_phases": checked_phases,
        "_check_stats": l1_stats,
    }
    checked_l2 = {
        "events": final_events,
        "_check_stats": l2_stats,
    }

    return checked_l1, checked_l2

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hierarchical video annotation pipeline (merged L1+L2, L3, and quality checks)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--jsonl", default=None,
                        help="可选：输入 JSONL。若不提供，则直接遍历 --frames-dir 下所有样本。")
    parser.add_argument("--frames-dir", required=True,
                        help="Root directory of pre-extracted frames")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write per-clip annotation JSON files")
    parser.add_argument("--l3-frames-dir", default=None,
                        help="High-FPS frames directory for L3/3c (falls back to --frames-dir)")
    parser.add_argument("--level", type=str, choices=["merged", "3", "2c", "3c", "merged_c"], default="merged",
                        help="Annotation level (merged=L1+L2+domain, 3=L3 grounding, 2c/3c=check, merged_c=L1+L2 check)")
    parser.add_argument("--api-base", default="https://api.novita.ai/v3/openai",
                        help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", default="",
                        help="API key (prefers NOVITA_API_KEY env var, then OPENAI_API_KEY)")
    parser.add_argument("--model", default="pa/gmn-2.5-pr",
                        help="Model name to pass to the API")
    parser.add_argument("--max-frames-per-call", type=int, default=0,
                        help="Max frames per API call (0 = no limit, send all frames)")
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

    level = args.level
    l3_frames_dir = Path(args.l3_frames_dir) if args.l3_frames_dir else None

    print(f"Annotating {len(records)} clips at Level {args.level}")
    print(f"API: {args.api_base}  model: {args.model}  workers: {args.workers}")
    print(f"resize_max_width={args.resize_max_width}  jpeg_quality={args.jpeg_quality}")
    if level == "merged":
        print(f"Merged mode: L1 phases + L2 events + domain + summary in one VLM call")
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
                l3_frames_dir,
            ): rec
            for rec in records
        }
        total = len(futures)
        for i, fut in enumerate(as_completed(futures), 1):
            rec = futures[fut]
            clip_key = (rec.get("videos") or ["?"])[0].rsplit("/", 1)[-1].rsplit(".", 1)[0] if rec.get("videos") else "?"
            try:
                res = fut.result()
            except Exception as exc:
                error_count += 1
                print(f"[{i}/{total}] CRASH  {clip_key}: {type(exc).__name__}: {exc}", flush=True)
                continue
            if res["skipped"]:
                skipped_count += 1
                print(f"\r[{i}/{total}] skip={skipped_count} ok={ok_count} err={error_count}", end="", flush=True)
            elif res["ok"]:
                ok_count += 1
                print(f"\n[{i}/{total}] OK     {res['clip_key']}", flush=True)
            else:
                error_count += 1
                print(f"\n[{i}/{total}] ERROR  {res['clip_key']}: {res['error']}", flush=True)

    print(f"\n\nFinished: {ok_count} annotated, {skipped_count} skipped, {error_count} errors", flush=True)
    if error_count > 0:
        print("Re-run with --overwrite to retry failed clips.")


if __name__ == "__main__":
    main()
