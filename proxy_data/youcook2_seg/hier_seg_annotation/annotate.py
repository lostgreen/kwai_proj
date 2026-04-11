#!/usr/bin/env python3
"""
annotate.py — Scene-first hierarchical video annotation pipeline.

Two-pass architecture:
  Pass 1: Scene merge/split + event captioning + domain classification (1fps)
  Pass 2: Per-event L3 sub-action annotation (1fps, only event frames)
  + L1 phase aggregation from events

Usage:
    python annotate.py \\
        --frames-dir frames/ \\
        --output-dir annotations/ \\
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
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Global thread-safe token usage tracker
# ─────────────────────────────────────────────────────────────────────────────
_token_lock = threading.Lock()
_token_usage = {
    "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "api_calls": 0,
    "est_text_chars": 0, "est_image_b64_bytes": 0, "n_images": 0,
}


def get_token_usage() -> dict[str, int]:
    """Return a snapshot of accumulated token usage."""
    with _token_lock:
        return dict(_token_usage)


def reset_token_usage() -> None:
    """Reset accumulated token usage counters."""
    with _token_lock:
        for k in _token_usage:
            _token_usage[k] = 0


def _accumulate_usage(usage, text_chars: int = 0, image_b64_bytes: int = 0, n_images: int = 0) -> None:
    """Accumulate token usage from an API response."""
    if usage is None:
        return
    with _token_lock:
        _token_usage["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
        _token_usage["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
        _token_usage["total_tokens"] += getattr(usage, "total_tokens", 0) or 0
        _token_usage["api_calls"] += 1
        _token_usage["est_text_chars"] += text_chars
        _token_usage["est_image_b64_bytes"] += image_b64_bytes
        _token_usage["n_images"] += n_images


from archetypes import (
    SYSTEM_PROMPT,
    DOMAIN_L2_ALL,
    resolve_domain_l1,
    get_l1_aggregation_prompt,
    get_scene_first_prompt,
    get_scene_first_l3_prompt,
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


def load_scene_boundaries(frame_dir: Path) -> list[int]:
    """Load scene boundary frame indices from scenes.json.

    Returns a list of 1-based frame indices where scene breaks occur.
    Returns empty list if scenes.json is absent (graceful degradation).
    """
    scenes_path = frame_dir / "scenes.json"
    if not scenes_path.exists():
        return []
    try:
        with open(scenes_path, encoding="utf-8") as f:
            data = json.load(f)
        indices = data.get("boundary_frame_indices", [])
        return [int(i) for i in indices if isinstance(i, (int, float))]
    except Exception:
        return []


def load_scenes_as_segments(frame_dir: Path, clip_duration: float) -> list[dict]:
    """Convert scenes.json boundary data into ordered scene segment dicts.

    Returns [{scene_id, start_time, end_time}] (start/end in whole seconds).
    Falls back to a single full-clip segment when scenes.json is absent or empty.
    """
    scenes_path = frame_dir / "scenes.json"
    if not scenes_path.exists():
        return [{"scene_id": 1, "start_time": 0, "end_time": int(clip_duration)}]
    try:
        with open(scenes_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return [{"scene_id": 1, "start_time": 0, "end_time": int(clip_duration)}]

    boundaries = data.get("boundary_timestamps_sec", [])
    segment_starts = [0.0] + list(boundaries)
    segment_ends = list(boundaries) + [clip_duration]

    segments: list[dict] = []
    for i, (st, et) in enumerate(zip(segment_starts, segment_ends), 1):
        st_int = int(round(st))
        et_int = min(int(round(et)), int(clip_duration))
        if et_int > st_int:
            segments.append({"scene_id": i, "start_time": st_int, "end_time": et_int})

    return segments or [{"scene_id": 1, "start_time": 0, "end_time": int(clip_duration)}]


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
    images_first: bool = False,
) -> str:
    """
    Call a VLM endpoint with interleaved frame images.

    Args:
        frame_labels: Per-frame text labels (e.g. "[Frame 1]" or "[Timestamp 00:42 | Frame 42]").
        images_first: If True, place interleaved [label+image] before the text prompt.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai is required: pip install openai")

    # Provider detection
    _is_azure = (
        os.environ.get("USE_AZURE", "").lower() in ("1", "true", "yes")
        or "azure" in api_base.lower()
    )
    _is_openrouter = "openrouter.ai" in api_base.lower()
    _is_novita = "novita.ai" in api_base.lower()

    # API key: prefer explicit arg, then provider-specific env var, then generic fallback
    if api_key:
        key = api_key
    elif _is_azure:
        key = os.environ.get("AZURE_OPENAI_API_KEY", "")
    elif _is_openrouter:
        key = os.environ.get("OPENROUTER_API_KEY", "")
    elif _is_novita:
        key = os.environ.get("NOVITA_API_KEY", "")
    else:
        key = os.environ.get("OPENAI_API_KEY", "")

    if _is_azure:
        from openai import AzureOpenAI
        client = AzureOpenAI(
            azure_endpoint=api_base,
            api_key=key,
            api_version=os.environ.get("AZURE_API_VERSION", "2025-01-01-preview"),
        )
    else:
        client = OpenAI(api_key=key, base_url=api_base)

    content: list[dict[str, Any]] = []
    if not images_first:
        content.append({"type": "text", "text": user_text})
    for i, b64 in enumerate(frame_b64_list):
        label = frame_labels[i] if i < len(frame_labels) else f"[Frame {i + 1}]"
        content.append({"type": "text", "text": label})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
        })
    if images_first:
        content.append({"type": "text", "text": user_text})

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": content},
    ]

    # Pre-compute text/image sizes for usage tracking
    text_chars = len(system_prompt) + len(user_text) + sum(len(l) for l in frame_labels)
    image_b64_bytes = sum(len(b) for b in frame_b64_list)

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            create_kwargs: dict[str, Any] = dict(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            # json_object mode — skip for providers that may not support it
            if not _is_openrouter:
                create_kwargs["response_format"] = {"type": "json_object"}
            # Gemini-specific low-res hint — skip for Azure / OpenRouter / non-Gemini
            if not _is_azure and not _is_openrouter:
                create_kwargs["extra_body"] = {
                    "generation_config": {
                        "media_resolution": "MEDIA_RESOLUTION_LOW",
                    }
                }
            # OpenRouter: pass reasoning config if enabled via env
            if _is_openrouter and os.environ.get("OPENROUTER_REASONING", "").lower() in ("1", "true", "yes"):
                create_kwargs.setdefault("extra_body", {})["reasoning"] = {"enabled": True}
            resp = client.chat.completions.create(**create_kwargs)
            _accumulate_usage(resp.usage, text_chars, image_b64_bytes, len(frame_b64_list))
            if resp.usage:
                pt = getattr(resp.usage, "prompt_tokens", 0) or 0
                ct = getattr(resp.usage, "completion_tokens", 0) or 0
                print(f"    [call] prompt={pt:,} compl={ct:,} imgs={len(frame_b64_list)} b64={image_b64_bytes//1024}KB", flush=True)
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
    images_first: bool = False,
) -> dict[str, Any] | None:
    """Call VLM and parse response. Retries once on parse failure. Returns None on final failure."""
    raw = call_vlm(api_base, api_key, model, system_prompt, prompt_text, frame_b64, frame_labels,
                   images_first=images_first)
    parsed = parse_json_from_response(raw)
    if parsed.get("_parse_error"):
        raw = call_vlm(api_base, api_key, model, system_prompt, prompt_text, frame_b64, frame_labels,
                       images_first=images_first)
        parsed = parse_json_from_response(raw)
        if parsed.get("_parse_error"):
            return None
    return parsed


def _validate_sub_actions(
    sub_actions: list,
    event_start: int,
    event_end: int,
    event_id: int,
) -> list[dict]:
    """Validate and filter L3 sub-actions within a single L2 event."""
    if not isinstance(sub_actions, list):
        return []
    valid: list[dict] = []
    for sa in sub_actions:
        if not isinstance(sa, dict):
            continue
        sa_st = sa.get("start_time")
        sa_et = sa.get("end_time")
        if not (isinstance(sa_st, (int, float)) and isinstance(sa_et, (int, float)) and sa_st < sa_et):
            print(f"    WARN: sub_action in event {event_id} dropped "
                  f"(invalid timestamps: st={sa_st} et={sa_et}, "
                  f"label={str(sa.get('sub_action', ''))[:60]})", flush=True)
            continue
        sa_st = round(sa_st)
        sa_et = round(sa_et)
        # Clamp to event boundaries
        sa_st = max(sa_st, event_start)
        sa_et = min(sa_et, event_end)
        if sa_et - sa_st < 1:
            print(f"    WARN: sub_action in event {event_id} dropped after clamp "
                  f"(duration<1s: {sa_st}-{sa_et}, bounds={event_start}-{event_end})", flush=True)
            continue
        sa["start_time"] = sa_st
        sa["end_time"] = sa_et
        valid.append(sa)
    # Re-number action_id
    for i, sa in enumerate(valid, 1):
        sa["action_id"] = i
    return valid

def _build_level3_from_stage1(
    stage1_result: dict[str, Any],
    merged_events: list[dict],
) -> dict:
    """Build level3 dict from inline L3 sub-actions extracted in Stage 1.

    Maps stage1 L3 results to the final (re-numbered) events after L1 aggregation.
    """
    l3_results = stage1_result.get("_l3_results", [])
    if not l3_results:
        return {"micro_type": "sub_action", "grounding_results": []}

    event_phase_map = {ev["event_id"]: ev.get("parent_phase_id") for ev in merged_events}
    event_lookup = {ev["event_id"]: ev for ev in merged_events}

    grounding_results: list[dict] = []
    for l3r in l3_results:
        eid = l3r["event_id"]
        ev_match = event_lookup.get(eid)
        if ev_match is None:
            continue
        grounding_results.append({
            "event_id": eid,
            "parent_phase_id": event_phase_map.get(eid),
            "event_start": ev_match["start_time"],
            "event_end": ev_match["end_time"],
            "event_instruction": ev_match.get("instruction", ""),
            "sub_actions": l3r["sub_actions"],
        })

    return {"micro_type": "sub_action", "grounding_results": grounding_results}


def _merge_l1_aggregation(
    stage1_result: dict[str, Any],
    stage2_parsed: dict,
    clip_duration: float,
) -> dict[str, Any]:
    """Merge Stage 1 (L2 events) + Stage 2 (L1 phases) into annotation output.
    Phase timestamps are derived from member events (not from VLM output).
    """
    events = list(stage1_result["_stage1_events"])
    events_by_id = {ev["event_id"]: ev for ev in events}

    raw_phases = stage2_parsed.get("macro_phases", [])
    if not isinstance(raw_phases, list):
        raw_phases = []

    l1_phases: list[dict] = []
    assigned_event_ids: set[int] = set()

    for phase in raw_phases:
        if not isinstance(phase, dict):
            continue
        member_ids = phase.get("member_event_ids", [])
        if not isinstance(member_ids, list):
            continue
        member_events = [events_by_id[eid] for eid in member_ids if eid in events_by_id]
        if not member_events:
            continue

        phase_start = min(ev["start_time"] for ev in member_events)
        phase_end = max(ev["end_time"] for ev in member_events)

        l1_phases.append({
            "phase_id": phase.get("phase_id", len(l1_phases) + 1),
            "start_time": round(phase_start),
            "end_time": min(round(phase_end), round(clip_duration)),
            "phase_name": phase.get("phase_name", ""),
            "narrative_summary": phase.get("narrative_summary", ""),
            "event_split_criterion": phase.get("event_split_criterion", ""),
            "_member_event_ids": [eid for eid in member_ids if eid in events_by_id],
        })
        assigned_event_ids.update(eid for eid in member_ids if eid in events_by_id)

    # Handle orphan events — assign to temporally nearest phase
    orphan_ids = set(events_by_id.keys()) - assigned_event_ids
    if orphan_ids and l1_phases:
        for oid in orphan_ids:
            orphan_ev = events_by_id[oid]
            best_phase = min(
                l1_phases,
                key=lambda p: min(
                    abs(orphan_ev["start_time"] - p["start_time"]),
                    abs(orphan_ev["end_time"] - p["end_time"]),
                ),
            )
            best_phase["_member_event_ids"].append(oid)
            best_phase["start_time"] = min(best_phase["start_time"], round(orphan_ev["start_time"]))
            best_phase["end_time"] = max(best_phase["end_time"], min(round(orphan_ev["end_time"]), round(clip_duration)))
            print(f"    INFO: orphan event {oid} assigned to phase {best_phase['phase_id']}", flush=True)
    elif orphan_ids:
        # No phases at all — create a single catch-all phase
        l1_phases.append({
            "phase_id": 1,
            "start_time": min(events_by_id[eid]["start_time"] for eid in orphan_ids),
            "end_time": min(max(events_by_id[eid]["end_time"] for eid in orphan_ids), round(clip_duration)),
            "phase_name": stage1_result.get("summary", "Main activity")[:100],
            "narrative_summary": stage1_result.get("video_caption", ""),
            "event_split_criterion": "single-phase fallback",
            "_member_event_ids": list(orphan_ids),
        })

    # Sort phases by start_time, re-number phase_id
    l1_phases.sort(key=lambda p: (p["start_time"], p["end_time"]))
    for i, phase in enumerate(l1_phases, 1):
        phase["phase_id"] = i

    # Assign parent_phase_id to every event
    for phase in l1_phases:
        for eid in phase["_member_event_ids"]:
            if eid in events_by_id:
                events_by_id[eid]["parent_phase_id"] = phase["phase_id"]
        del phase["_member_event_ids"]

    # Sort events by start_time, re-number event_id
    events.sort(key=lambda e: (e.get("start_time", 0), e.get("end_time", 0)))
    for i, ev in enumerate(events, 1):
        ev["event_id"] = i

    # Strip key_frame_indices from final events (internal field, not needed downstream)
    for ev in events:
        ev.pop("key_frame_indices", None)

    level1 = {
        "macro_phases": l1_phases,
        "_sampling": stage1_result.get("_sampling", {}),
    }
    level2 = {"events": events}

    return {
        "level1": level1,
        "level2": level2,
        "summary": stage1_result.get("summary", ""),
        "global_phase_criterion": stage1_result.get("global_phase_criterion", ""),
        "archetype": stage1_result.get("archetype", "tutorial"),
        "domain_l2": stage1_result.get("domain_l2", "other"),
        "domain_l2_note": stage1_result.get("domain_l2_note", ""),
        "domain_l1": stage1_result.get("domain_l1", "other"),
        "topology_type": stage1_result.get("topology_type", "procedural"),
        "video_caption": stage1_result.get("video_caption", ""),
    }


def _annotate_l1_aggregation(
    frame_dir: Path,
    clip_duration: float,
    stage1_result: dict[str, Any],
    api_base: str, api_key: str, model: str,
    resize_max_width: int, jpeg_quality: int,
    fps: float = 1.0,
) -> dict[str, Any]:
    """Stage 2 of bottom-up pipeline: L1 phase aggregation from L2 events.

    Selects key frames (1-2 per event from Stage 1), sends them with the
    event list to the VLM, and gets L1 phase groupings.

    Falls back to wrapping all events in a single phase if VLM fails.
    """
    events = stage1_result["_stage1_events"]
    all_frames = get_all_frame_files(frame_dir)
    frame_by_idx: dict[int, Path] = {}
    for fp in all_frames:
        frame_by_idx[frame_stem_to_index(fp, -1)] = fp

    # Collect key frames per event
    key_frame_files: list[Path] = []
    key_frame_labels: list[str] = []
    for ev in events:
        for kf_idx in ev.get("key_frame_indices", []):
            fp = frame_by_idx.get(kf_idx)
            if fp is not None:
                key_frame_files.append(fp)
                key_frame_labels.append(
                    f"[Event {ev['event_id']} KeyFrame | "
                    f"Timestamp {format_mmss(frame_index_to_sec(kf_idx, fps=fps))} | Frame {kf_idx}]"
                )

    if not key_frame_files:
        print("    WARN: no valid key frames for L1 aggregation, using single-phase fallback", flush=True)
        return _merge_l1_aggregation(
            stage1_result,
            {"macro_phases": []},
            clip_duration,
        )

    frame_b64 = encode_frame_files(
        key_frame_files, resize_max_width=resize_max_width, jpeg_quality=jpeg_quality,
    )
    print(f"  [L1-agg] key frames: {len(key_frame_files)} from {len(events)} events", flush=True)

    # Build events JSON for the prompt (strip key_frame_indices)
    events_for_prompt = []
    for ev in events:
        events_for_prompt.append({
            "event_id": ev["event_id"],
            "start_time": ev["start_time"],
            "end_time": ev["end_time"],
            "instruction": ev.get("instruction", ""),
            "dense_caption": ev.get("dense_caption", ""),
            "l3_feasible": ev.get("l3_feasible", False),
        })

    import json as _json
    events_json_str = _json.dumps(events_for_prompt, indent=2, ensure_ascii=False)

    prompt_text = get_l1_aggregation_prompt(
        events_json=events_json_str,
        summary=stage1_result.get("summary", ""),
        global_phase_criterion=stage1_result.get("global_phase_criterion", ""),
        n_events=len(events),
        duration_sec=int(clip_duration),
    )

    parsed = call_and_parse(api_base, api_key, model, SYSTEM_PROMPT, prompt_text, frame_b64, key_frame_labels)

    if parsed is None:
        print("    WARN: L1 aggregation parse failed, using single-phase fallback", flush=True)
        parsed = {"macro_phases": []}

    return _merge_l1_aggregation(stage1_result, parsed, clip_duration)
# ─────────────────────────────────────────────────────────────────────────────
# Scene-First Pipeline (v9): scene-anchored merge → caption → inline L3
# ─────────────────────────────────────────────────────────────────────────────

def _split_scene_first_response(
    parsed: dict,
    scenes: list[dict],
    n_sampled_frames: int,
    resize_max_width: int,
    jpeg_quality: int,
    clip_duration: float,
    fps: float = 1.0,
    effective_fps: float | None = None,
) -> dict[str, Any]:
    """Parse and validate the scene_first VLM response.

    - Events reference scene_ids instead of raw timestamps.
    - start_time / end_time are computed from the scenes lookup table.
    - merge_reason is validated and required for multi-scene events.
    - scene_ids must partition all input scenes without overlap.
    """
    # ── Classification fields ─────────────────────────────────────────────────
    summary = parsed.get("summary", "")
    global_phase_criterion = parsed.get("global_phase_criterion", "")

    domain_l2 = parsed.get("domain_l2", "other")
    if domain_l2 not in DOMAIN_L2_ALL:
        domain_l2 = "other"
    domain_l2_note = str(parsed.get("domain_l2_note") or "") if domain_l2 == "other" else ""

    video_caption = str(parsed.get("video_caption", ""))

    # Paradigm / topology — hardcoded defaults (not requested from VLM in scene-first)
    archetype = "tutorial"
    topology_type = "procedural"

    # ── Build scenes lookup ──────────────────────────────────────────────────
    scenes_by_id: dict[int, dict] = {s["scene_id"]: s for s in scenes}
    all_scene_ids: set[int] = set(scenes_by_id.keys())
    n_scenes = len(scenes)

    # ── Validate and convert events ──────────────────────────────────────────
    raw_events = parsed.get("events", [])
    if not isinstance(raw_events, list):
        raw_events = []

    covered_scene_ids: set[int] = set()
    valid_events: list[dict] = []

    for ev in raw_events:
        if not isinstance(ev, dict):
            continue

        split_reason = ev.get("split_reason")
        is_split = bool(split_reason and str(split_reason).strip())

        # Validate scene_ids
        raw_sids = ev.get("scene_ids", [])
        if not isinstance(raw_sids, list) or not raw_sids:
            print(f"    WARN: event dropped — missing or empty scene_ids", flush=True)
            continue
        valid_sids = sorted({int(s) for s in raw_sids if isinstance(s, (int, float)) and int(s) in scenes_by_id})
        if not valid_sids:
            print(f"    WARN: event dropped — scene_ids {raw_sids} not in known scenes", flush=True)
            continue

        # Mark scenes as covered (for coverage check)
        covered_scene_ids.update(valid_sids)

        if is_split:
            # SPLIT: model provides explicit timestamps; validate within parent scene bounds
            if len(valid_sids) != 1:
                print(f"    WARN: split event has {len(valid_sids)} scene_ids; using first", flush=True)
                valid_sids = [valid_sids[0]]
            parent_scene = scenes_by_id[valid_sids[0]]
            ev_start_raw = ev.get("start_time")
            ev_end_raw = ev.get("end_time")
            if not (isinstance(ev_start_raw, (int, float)) and isinstance(ev_end_raw, (int, float))
                    and ev_start_raw < ev_end_raw):
                print(f"    WARN: split event in scene {valid_sids[0]} has invalid timestamps "
                      f"({ev_start_raw}-{ev_end_raw}) — clamping to scene boundaries", flush=True)
                ev_start = parent_scene["start_time"]
                ev_end = parent_scene["end_time"]
            else:
                ev_start = max(int(round(ev_start_raw)), parent_scene["start_time"])
                ev_end = min(int(round(ev_end_raw)), parent_scene["end_time"])
            merge_reason = None
        else:
            # KEEP / MERGE: timestamps derived from scene boundaries
            # Enforce consecutiveness — non-consecutive merges are invalid
            if len(valid_sids) > 1:
                expected = list(range(valid_sids[0], valid_sids[-1] + 1))
                if valid_sids != expected:
                    # Non-consecutive merge: keep only the first scene to preserve timeline integrity
                    print(f"    WARN: scene_ids {valid_sids} are NOT consecutive (interleaved merge) — "
                          f"trimming to first scene [{valid_sids[0]}]", flush=True)
                    valid_sids = [valid_sids[0]]
            ev_start = min(scenes_by_id[sid]["start_time"] for sid in valid_sids)
            ev_end = max(scenes_by_id[sid]["end_time"] for sid in valid_sids)

            # merge_reason validation
            merge_reason = ev.get("merge_reason")
            if len(valid_sids) > 1:
                if not merge_reason or not str(merge_reason).strip():
                    merge_reason = f"Merged scenes {valid_sids} (no explicit reason provided)"
                    print(f"    WARN: merge event covers {valid_sids} but merge_reason is empty", flush=True)
                else:
                    merge_reason = str(merge_reason).strip()
            else:
                merge_reason = None

        ev_end = min(ev_end, int(clip_duration))
        if ev_end <= ev_start:
            print(f"    WARN: event dropped — zero/negative duration after clamping "
                  f"(scene_ids={valid_sids}, {ev_start}-{ev_end})", flush=True)
            continue

        # key_frame_indices
        raw_kf = ev.get("key_frame_indices", [])
        valid_kf: list[int] = []
        if isinstance(raw_kf, list):
            for idx in raw_kf:
                if isinstance(idx, (int, float)) and 1 <= int(idx) <= n_sampled_frames:
                    valid_kf.append(int(idx))
        if not valid_kf:
            mid_sec = (ev_start + ev_end) / 2
            mid_frame = max(1, min(n_sampled_frames, round(mid_sec * fps) + 1))
            valid_kf = [mid_frame]

        valid_events.append({
            "start_time": ev_start,
            "end_time": ev_end,
            "scene_ids": valid_sids,
            "merge_reason": merge_reason,
            "split_reason": str(split_reason).strip() if is_split else None,
            "instruction": str(ev.get("instruction", "")),
            "dense_caption": str(ev.get("dense_caption", "")),
            "visual_keywords": ev.get("visual_keywords", []) if isinstance(ev.get("visual_keywords"), list) else [],
            "key_frame_indices": valid_kf[:2],
            "l3_worthy": bool(ev.get("l3_worthy", False)),
            "l3_feasible": False,  # will be set by separate L3 pass
            "l3_reason": "",
        })

    # ── Synthesize fallback events for uncovered scenes ───────────────────────
    uncovered = sorted(all_scene_ids - covered_scene_ids)
    if uncovered:
        print(f"    WARN: {len(uncovered)} scene(s) not covered by VLM events — "
              f"creating single-scene fallback events for: {uncovered}", flush=True)
        for sid in uncovered:
            sc = scenes_by_id[sid]
            mid_sec = (sc["start_time"] + sc["end_time"]) / 2
            mid_frame = max(1, min(n_sampled_frames, round(mid_sec * fps) + 1))
            valid_events.append({
                "start_time": sc["start_time"],
                "end_time": sc["end_time"],
                "scene_ids": [sid],
                "merge_reason": None,
                "split_reason": None,
                "instruction": f"Scene {sid}: visual content (fallback — VLM did not annotate this scene)",
                "dense_caption": "",
                "visual_keywords": [],
                "key_frame_indices": [mid_frame],
                "l3_feasible": False,
                "l3_reason": "fallback — scene not covered by VLM",
            })

    # Sort by start_time and number events
    valid_events.sort(key=lambda e: (e["start_time"], e["end_time"]))
    for i, ev in enumerate(valid_events, 1):
        ev["event_id"] = i

    return {
        "_stage1_events": valid_events,
        "_l3_results": [],  # L3 is done in separate pass 2
        "summary": summary,
        "global_phase_criterion": global_phase_criterion,
        "archetype": archetype,
        "domain_l2": domain_l2,
        "domain_l2_note": domain_l2_note,
        "domain_l1": resolve_domain_l1(domain_l2),
        "topology_type": topology_type,
        "video_caption": video_caption,
        "_sampling": {
            "n_sampled_frames": n_sampled_frames,
            "resize_max_width": resize_max_width,
            "jpeg_quality": jpeg_quality,
            "fps": fps,
            "effective_fps": effective_fps or fps,
            "n_input_scenes": n_scenes,
        },
    }


def _annotate_scene_first(
    frame_dir: Path,
    clip_duration: float,
    api_base: str, api_key: str, model: str,
    max_frames: int, resize_max_width: int, jpeg_quality: int,
    fps: float = 1.0,
) -> dict[str, Any]:
    """Stage 1 of scene-first pipeline: merge scenes → caption events → inline L3.

    Uses pre-detected scenes as hard anchors. The model decides which adjacent
    scenes to merge, writes captions, and annotates L3 sub-actions.
    Event timestamps are derived from scene_ids, not from VLM output.
    """
    all_frames = get_all_frame_files(frame_dir)
    if not all_frames:
        raise RuntimeError(f"no frames found in {frame_dir}")

    # Frames are already extracted at 1fps; use them directly
    effective_fps = fps

    sampled = sample_uniform(all_frames, max_frames)
    frame_b64 = encode_frame_files(sampled, resize_max_width=resize_max_width, jpeg_quality=jpeg_quality)

    with Image.open(sampled[0]) as _img:
        orig_w, orig_h = _img.size
    avg_b64_len = sum(len(b) for b in frame_b64) // max(len(frame_b64), 1)
    avg_jpeg_kb = avg_b64_len * 3 // 4 // 1024
    print(f"  [scene-first] frames: {len(sampled)}/{len(all_frames)} @ {fps}fps "
          f"orig={orig_w}x{orig_h} resize_max={resize_max_width} q={jpeg_quality} avg_jpeg={avg_jpeg_kb}KB", flush=True)

    # Too short → skip
    if clip_duration < 15:
        return {
            "_stage1_events": [], "_l3_results": [],
            "summary": "", "global_phase_criterion": "",
            "archetype": "tutorial",
            "domain_l2": "other", "domain_l1": "other", "topology_type": "procedural",
            "video_caption": "",
            "_sampling": {
                "n_sampled_frames": len(sampled), "resize_max_width": resize_max_width,
                "jpeg_quality": jpeg_quality, "fps": fps, "effective_fps": effective_fps,
                "n_input_scenes": 0,
            },
        }

    # Load scenes (hard anchors)
    scenes = load_scenes_as_segments(frame_dir, clip_duration)
    n_scenes = len(scenes)
    print(f"  [scene-first] input scenes: {n_scenes}", flush=True)

    # Mark [SCENE BREAK] frames
    scene_boundaries_raw = load_scene_boundaries(frame_dir)
    sampled_indices = [frame_stem_to_index(fp, 0) for fp in sampled]
    scene_boundary_set: set[int] = set()
    boundary_tol = 2 if fps >= 2.0 else 1  # tolerance for matching boundary to nearest frame
    for b_idx in scene_boundaries_raw:
        if sampled_indices:
            nearest = min(sampled_indices, key=lambda s, b=b_idx: abs(s - b))
            if abs(nearest - b_idx) <= boundary_tol:
                scene_boundary_set.add(nearest)

    frame_labels = []
    for fp in sampled:
        idx = frame_stem_to_index(fp, 0)
        # Use plain seconds (not MM:SS) to match JSON output format and avoid confusion
        ts_sec = int(round(frame_index_to_sec(idx, fps=fps)))
        ts_label = f"[t={ts_sec}s]"
        if idx in scene_boundary_set:
            ts_label = f"[SCENE BREAK] {ts_label}"
        frame_labels.append(ts_label)

    import json as _json
    scenes_json_str = _json.dumps(scenes, ensure_ascii=False)

    prompt_text = get_scene_first_prompt(
        n_frames=len(sampled),
        duration_sec=int(clip_duration),
        n_scenes=n_scenes,
        scenes_json_str=scenes_json_str,
    )

    # One-time prompt dump: save the first clip's full prompt for verification
    if not hasattr(_annotate_scene_first, "_prompt_logged"):
        _debug_path = Path("_debug_scene_first_prompt.txt")
        try:
            lines = [f"=== SYSTEM PROMPT ===\n{SYSTEM_PROMPT}\n",
                     "=== USER MESSAGE (images_first=True) ===\n",
                     "--- FRAMES (sent first) ---"]
            for _lbl in frame_labels:
                lines.append(f"{_lbl}\n  [IMAGE_PLACEHOLDER]")
            lines.append("--- FRAMES END ---\n")
            lines.append(f"--- PROMPT TEXT ---\n{prompt_text}\n--- PROMPT TEXT END ---")
            _debug_path.write_text("\n".join(lines), encoding="utf-8")
            print(f"  [scene-first] prompt saved to {_debug_path.resolve()}", flush=True)
        except Exception as _e:
            print(f"  [scene-first] WARN: failed to save debug prompt: {_e}", flush=True)
        _annotate_scene_first._prompt_logged = True

    parsed = call_and_parse(api_base, api_key, model, SYSTEM_PROMPT, prompt_text, frame_b64, frame_labels,
                            images_first=True)
    if parsed is None:
        raise RuntimeError("scene-first JSON parse failed after retry")

    return _split_scene_first_response(
        parsed, scenes, len(sampled), resize_max_width, jpeg_quality, clip_duration,
        fps=fps, effective_fps=effective_fps,
    )


def _annotate_scene_first_l3(
    frame_dir: Path,
    events: list[dict],
    clip_duration: float,
    api_base: str, api_key: str, model: str,
    resize_max_width: int, jpeg_quality: int,
    fps: float = 1.0,
    l3_log_dir: Path | None = None,
) -> tuple[list[dict], list[dict]]:
    """Pass 2: per-event L3 sub-split using only the event's frames at 1fps.

    Returns (updated_events, l3_results) where updated_events have l3_feasible set.
    """
    all_frames = get_all_frame_files(frame_dir)

    # Frames are already at 1fps; use them directly
    # Build index: timestamp (seconds) → Path
    frame_by_sec: dict[int, Path] = {}
    for fp in all_frames:
        idx = frame_stem_to_index(fp, -1)
        if idx >= 0:
            sec = int(round(frame_index_to_sec(idx, fps=fps)))
            frame_by_sec[sec] = fp

    # Load scene boundaries for [SCENE BREAK] labels
    scene_boundaries_raw = load_scene_boundaries(frame_dir)
    sampled_indices = sorted(frame_by_sec.values(), key=lambda p: frame_stem_to_index(p, 0))
    all_sampled_idx = [frame_stem_to_index(fp, 0) for fp in sampled_indices]
    boundary_tol = 2 if fps >= 2.0 else 1
    scene_boundary_set: set[int] = set()
    for b_idx in scene_boundaries_raw:
        if all_sampled_idx:
            nearest = min(all_sampled_idx, key=lambda s, b=b_idx: abs(s - b))
            if abs(nearest - b_idx) <= boundary_tol:
                scene_boundary_set.add(nearest)

    l3_results: list[dict] = []
    for ev in events:
        ev_start = ev["start_time"]
        ev_end = ev["end_time"]
        ev_duration = ev_end - ev_start
        n_scenes = len(ev.get("scene_ids", [1]))

        # Deterministic L3 eligibility: multi-scene OR (l3_worthy AND duration > 10s)
        if n_scenes >= 2:
            l3_eligible = True
        elif ev.get("l3_worthy", False) and ev_duration > 10:
            l3_eligible = True
        else:
            l3_eligible = False

        if not l3_eligible:
            ev["l3_feasible"] = False
            ev["l3_reason"] = f"skip: {n_scenes} scene(s), {ev_duration}s, l3_worthy={ev.get('l3_worthy', False)}"
            continue

        # Select frames within [ev_start, ev_end]
        event_frames: list[Path] = []
        for sec in sorted(frame_by_sec.keys()):
            if ev_start <= sec <= ev_end:
                event_frames.append(frame_by_sec[sec])
        if not event_frames:
            ev["l3_feasible"] = False
            ev["l3_reason"] = "no frames in event range"
            continue

        frame_b64 = encode_frame_files(event_frames, resize_max_width=resize_max_width, jpeg_quality=jpeg_quality)
        frame_labels = []
        for fp in event_frames:
            idx = frame_stem_to_index(fp, 0)
            ts_sec = int(round(frame_index_to_sec(idx, fps=fps)))
            ts_label = f"[t={ts_sec}s]"
            if idx in scene_boundary_set:
                ts_label = f"[SCENE BREAK] {ts_label}"
            frame_labels.append(ts_label)

        prompt_text = get_scene_first_l3_prompt(
            n_frames=len(event_frames),
            start_time=ev_start,
            end_time=ev_end,
            instruction=ev.get("instruction", ""),
            dense_caption=ev.get("dense_caption", ""),
            scene_ids=str(ev.get("scene_ids", [])),
            n_scenes=n_scenes,
        )

        # One-time L3 prompt dump for verification
        if not hasattr(_annotate_scene_first_l3, "_prompt_logged"):
            _debug_l3_path = Path("_debug_scene_first_l3_prompt.txt")
            try:
                lines = [f"=== SYSTEM PROMPT ===\n{SYSTEM_PROMPT}\n",
                         "=== USER MESSAGE (images_first=True) ===\n",
                         "--- FRAMES (sent first) ---"]
                for _lbl in frame_labels:
                    lines.append(f"{_lbl}\n  [IMAGE_PLACEHOLDER]")
                lines.append("--- FRAMES END ---\n")
                lines.append(f"--- PROMPT TEXT ---\n{prompt_text}\n--- PROMPT TEXT END ---")
                _debug_l3_path.write_text("\n".join(lines), encoding="utf-8")
                print(f"    [L3] prompt saved to {_debug_l3_path.resolve()}", flush=True)
            except Exception as _e:
                print(f"    [L3] WARN: failed to save debug prompt: {_e}", flush=True)
            _annotate_scene_first_l3._prompt_logged = True

        raw_response = call_vlm(api_base, api_key, model, SYSTEM_PROMPT, prompt_text,
                                frame_b64, frame_labels, images_first=True)
        parsed = parse_json_from_response(raw_response)
        if parsed.get("_parse_error"):
            # Retry once
            raw_response = call_vlm(api_base, api_key, model, SYSTEM_PROMPT, prompt_text,
                                    frame_b64, frame_labels, images_first=True)
            parsed = parse_json_from_response(raw_response)

        # Save raw VLM response to log directory
        if l3_log_dir is not None:
            clip_key = frame_dir.name
            log_file = l3_log_dir / f"{clip_key}_ev{ev['event_id']}.json"
            try:
                log_data = {
                    "clip_key": clip_key,
                    "event_id": ev["event_id"],
                    "event_range": f"{ev_start}-{ev_end}s",
                    "n_scenes": n_scenes,
                    "n_frames": len(event_frames),
                    "frame_labels": frame_labels,
                    "raw_vlm_response": raw_response,
                    "parsed": parsed,
                }
                log_file.write_text(json.dumps(log_data, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass

        if parsed.get("_parse_error"):
            ev["l3_feasible"] = False
            ev["l3_reason"] = "L3 VLM call failed"
            continue

        raw_subs = parsed.get("sub_actions", [])
        valid_subs = _validate_sub_actions(raw_subs, ev_start, ev_end, ev["event_id"])
        ev["l3_feasible"] = len(valid_subs) > 0
        ev["l3_reason"] = f"{len(valid_subs)} sub-actions" if valid_subs else "no valid sub-actions"
        if not valid_subs and raw_subs:
            # Debug: VLM returned sub_actions but all were invalid
            print(f"    [L3] DEBUG event {ev['event_id']}: VLM returned {len(raw_subs)} sub_actions, "
                  f"all invalid. Raw: {json.dumps(raw_subs, ensure_ascii=False)[:500]}", flush=True)
        if valid_subs:
            l3_results.append({
                "event_id": ev["event_id"],
                "parent_phase_id": None,
                "sub_actions": valid_subs,
            })
        print(f"    [L3] event {ev['event_id']} ({ev_start}-{ev_end}s, {n_scenes} scene(s), "
              f"{len(event_frames)} frames): {len(valid_subs)} sub-actions", flush=True)

    return events, l3_results


# ─────────────────────────────────────────────────────────────────────────────
# Per-clip annotation
# ─────────────────────────────────────────────────────────────────────────────

def annotate_clip(
    record: dict,
    frames_base: Path,
    output_dir: Path,
    api_base: str,
    api_key: str,
    model: str,
    max_frames_per_call: int,
    resize_max_width: int,
    jpeg_quality: int,
    overwrite: bool,
) -> dict:
    """
    Run the scene-first annotation pipeline for a single clip record.

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

    # Skip if already done and not overwriting
    if not overwrite and existing.get("level1") is not None and existing.get("level2") is not None and existing.get("level3") is not None:
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
        # ── Scene-anchored: Pass 1 (merge+caption @1fps) → Pass 2 (per-event L3 @1fps) → L1 agg ──
        fps = float(frame_meta.get("fps") or 1.0)
        stage1_result = _annotate_scene_first(
            frame_dir, clip_duration,
            api_base, api_key, model,
            max_frames_per_call, resize_max_width, jpeg_quality,
            fps=fps,
        )

        # Pass 2: per-event L3 sub-split
        events = stage1_result.get("_stage1_events", [])
        if events:
            l3_log_dir = output_dir.parent / "l3_logs"
            l3_log_dir.mkdir(parents=True, exist_ok=True)
            events, l3_results = _annotate_scene_first_l3(
                frame_dir, events, clip_duration,
                api_base, api_key, model,
                resize_max_width, jpeg_quality,
                fps=fps,
                l3_log_dir=l3_log_dir,
            )
            stage1_result["_stage1_events"] = events
            stage1_result["_l3_results"] = l3_results

        n_events = len(events)
        n_l3 = len(stage1_result.get("_l3_results", []))
        n_merged = sum(
            1 for ev in events
            if len(ev.get("scene_ids", [])) > 1
        )
        n_split = sum(
            1 for ev in events
            if ev.get("split_reason")
        )
        n_scenes_in = stage1_result.get("_sampling", {}).get("n_input_scenes", 0)
        print(f"  [{key}] scene-first: {n_scenes_in} scenes → {n_events} events "
              f"({n_merged} merged, {n_split} splits), {n_l3} with L3, "
              f"domain={stage1_result.get('domain_l2')}"
              f"{' SKIP' if stage1_result.get('feasibility', {}).get('skip') else ''}",
              flush=True)

        if stage1_result.get("_stage1_events"):
            merged_result = _annotate_l1_aggregation(
                frame_dir, clip_duration, stage1_result,
                api_base, api_key, model,
                resize_max_width, jpeg_quality,
                fps=fps,
            )
            merged_result["level3"] = _build_level3_from_stage1(
                stage1_result, merged_result.get("level2", {}).get("events", []),
            )
            # Carry scene-level metadata into the annotation output
            merged_result["scene_first_stats"] = {
                "n_input_scenes": n_scenes_in,
                "n_events": n_events,
                "n_merged_events": n_merged,
                "n_split_events": n_split,
                "n_single_scene_events": n_events - n_merged - n_split,
            }
        else:
            merged_result = {
                "level1": {"macro_phases": [], "_sampling": stage1_result.get("_sampling", {})},
                "level2": {"events": []},
                "level3": {"micro_type": "sub_action", "grounding_results": []},
                "scene_first_stats": {
                    "n_input_scenes": n_scenes_in,
                    "n_events": 0,
                    "n_merged_events": 0,
                    "n_single_scene_events": 0,
                },
            }
            for k in ("summary", "global_phase_criterion", "archetype",
                       "domain_l2", "domain_l1", "topology_type",
                       "video_caption"):
                merged_result[k] = stage1_result.get(k, "")

        n_phases = len(merged_result.get("level1", {}).get("macro_phases", []))
        n_final_events = len(merged_result.get("level2", {}).get("events", []))
        n_l3_final = len(merged_result.get("level3", {}).get("grounding_results", []))
        print(f"  [{key}] L1-agg: {n_phases} phases, {n_final_events} events, {n_l3_final} L3 events",
              flush=True)

    except RuntimeError as e:
        return {"clip_key": key, "ok": False, "error": str(e)[:300], "skipped": False}

    # Merge into existing annotation file
    scene_boundaries = load_scene_boundaries(frame_dir)
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
        "scene_detection": {
            "n_boundaries": len(scene_boundaries),
            "boundary_frame_indices": scene_boundaries,
        } if scene_boundaries else None,
        "level1": None,
        "level2": None,
        "level3": None,
        **existing,
    }
    ann.update(merged_result)
    ann["annotated_at"] = datetime.now(timezone.utc).isoformat()

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(ann, f, ensure_ascii=False, indent=2)

    return {"clip_key": key, "ok": True, "error": None, "skipped": False}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scene-first hierarchical video annotation pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--jsonl", default=None,
                        help="可选：输入 JSONL。若不提供，则直接遍历 --frames-dir 下所有样本。")
    parser.add_argument("--frames-dir", required=True,
                        help="Root directory of pre-extracted frames")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write per-clip annotation JSON files")
    parser.add_argument("--api-base", default="https://api.novita.ai/v3/openai",
                        help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", default="",
                        help="API key (prefers NOVITA_API_KEY env var, then OPENAI_API_KEY)")
    parser.add_argument("--model", default="pa/gmn-2.5-pr",
                        help="Model name to pass to the API")
    parser.add_argument("--max-frames-per-call", type=int, default=0,
                        help="Max frames per API call (0 = no limit, send all frames)")
    parser.add_argument("--resize-max-width", type=int, default=0,
                        help="Resize frames before upload; <=0 disables resizing")
    parser.add_argument("--jpeg-quality", type=int, default=60,
                        help="JPEG quality for recompressing frames before upload")
    parser.add_argument("--workers", type=int, default=2,
                        help="Parallel annotation workers")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process at most N clips (0 = all)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-annotate even if the level is already done")
    # Legacy: accept but ignore --level and --l3-frames-dir for backward compat
    parser.add_argument("--level", default="scene_first", help=argparse.SUPPRESS)
    parser.add_argument("--l3-frames-dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--classify-frames", type=int, default=64, help=argparse.SUPPRESS)
    args = parser.parse_args()

    # api_key resolved per-provider inside call_vlm(); pass through args.api_key
    api_key = args.api_key

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

    print(f"Annotating {len(records)} clips (scene-first, two-pass)")
    print(f"API: {args.api_base}  model: {args.model}  workers: {args.workers}")
    print(f"resize_max_width={args.resize_max_width}  jpeg_quality={args.jpeg_quality}")
    print(f"Frames: {frames_base}  Output: {output_dir}\n")

    ok_count = skipped_count = error_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                annotate_clip,
                rec, frames_base, output_dir,
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
                u = get_token_usage()
                print(f"\n[{i}/{total}] OK     {res['clip_key']}  (tokens: in={u['prompt_tokens']:,} out={u['completion_tokens']:,} calls={u['api_calls']})", flush=True)
            else:
                error_count += 1
                print(f"\n[{i}/{total}] ERROR  {res['clip_key']}: {res['error']}", flush=True)

    print(f"\n\nFinished: {ok_count} annotated, {skipped_count} skipped, {error_count} errors", flush=True)
    if error_count > 0:
        print("Re-run with --overwrite to retry failed clips.")

    # Token usage summary
    usage = get_token_usage()
    if usage["api_calls"] > 0:
        print(f"\n── Token Usage ──")
        print(f"  API calls:        {usage['api_calls']}")
        print(f"  Prompt tokens:    {usage['prompt_tokens']:,}")
        print(f"  Completion tokens:{usage['completion_tokens']:,}")
        print(f"  Total tokens:     {usage['total_tokens']:,}")
        if ok_count > 0:
            print(f"  Avg per clip:     {usage['total_tokens'] // ok_count:,} tokens")
        # Estimate text vs image token breakdown
        est_text_tokens = usage["est_text_chars"] // 4  # ~4 chars/token
        est_image_tokens = usage["prompt_tokens"] - est_text_tokens
        n_img = usage["n_images"]
        img_b64_mb = usage["est_image_b64_bytes"] / 1_048_576
        print(f"  ── Breakdown (estimated) ──")
        print(f"  Text chars sent:  {usage['est_text_chars']:,}  (~{est_text_tokens:,} tokens)")
        print(f"  Images sent:      {n_img:,}  ({img_b64_mb:.1f} MB base64)")
        if est_image_tokens > 0 and n_img > 0:
            print(f"  Image tokens:     ~{est_image_tokens:,}  (~{est_image_tokens // n_img:,} per image)")


if __name__ == "__main__":
    main()
