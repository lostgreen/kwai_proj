#!/usr/bin/env python3
"""
annotate.py — Archetype-driven hierarchical video annotation pipeline.

Annotation levels:
  merged: Step 0 classify (64 frames → archetype + domain_l2)
          + Step 1 archetype-driven L1+L2 annotation (full video frames)
  3:      L3 grounding — archetype-routed:
          tutorial/educational → per-event → state_change micro-actions
          performance          → per-phase → repetition_unit micro-actions
          cinematic/vlog/sports → per-event → interaction_unit micro-actions
          talk/ambient         → L3 skipped automatically

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

    # Step 3: L3 annotation (auto-skips flat; sequence now supported in v2)
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
    PARADIGM_IDS,
    PARADIGM_TO_TOPOLOGY,
    TOPOLOGY_TYPES,
    TOPOLOGY_TO_L2_MODE,
    TOPOLOGY_TO_L3_MODE,
    ARCHETYPE_IDS,
    ARCHETYPE_TO_TOPOLOGY,
    TOPOLOGY_TO_DEFAULT_ARCHETYPE,
    resolve_domain_l1,
    get_archetype,
    get_classification_prompt,
    get_archetype_merged_prompt,
    get_unified_merged_prompt,
    get_universal_merged_prompt,
    get_archetype_l3_prompt,
    get_active_levels,
    get_l3_parent_type,
    get_l2_first_prompt,
    get_l1_aggregation_prompt,
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
# Archetype helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_archetype(existing: dict[str, Any]) -> str:
    """Resolve archetype from existing annotation, with backward compat."""
    archetype = existing.get("archetype")
    if archetype and archetype in ARCHETYPE_IDS:
        return archetype
    # Fall back: topology → default archetype
    topology = existing.get("topology_type", "")
    return TOPOLOGY_TO_DEFAULT_ARCHETYPE.get(topology, "ambient")


def _classify_archetype(
    frame_dir: Path,
    clip_duration: float,
    api_base: str, api_key: str, model: str,
    max_frames: int = 64,
    resize_max_width: int = 0, jpeg_quality: int = 60,
) -> dict[str, Any]:
    """Step 0: Classify video archetype + domain_l2 using a small number of frames."""
    all_frames = get_all_frame_files(frame_dir)
    if not all_frames:
        raise RuntimeError(f"no frames found in {frame_dir}")

    sampled = sample_uniform(all_frames, max_frames)
    frame_b64 = encode_frame_files(sampled, resize_max_width=resize_max_width, jpeg_quality=jpeg_quality)

    frame_labels = []
    for fp in sampled:
        idx = frame_stem_to_index(fp, 0)
        frame_labels.append(f"[Timestamp {format_mmss(frame_index_to_sec(idx))} | Frame {idx}]")

    duration = int(clip_duration)
    prompt_text = get_classification_prompt(n_frames=len(sampled), duration_sec=duration)
    parsed = call_and_parse(api_base, api_key, model, SYSTEM_PROMPT, prompt_text, frame_b64, frame_labels)

    if parsed is None:
        # Fallback: default to tutorial archetype
        return {
            "archetype": "tutorial",
            "archetype_confidence": 0.0,
            "archetype_reason": "classification failed, defaulting to tutorial",
            "domain_l2": "other",
            "topology_type": "procedural",
        }

    archetype = parsed.get("archetype", "tutorial")
    if archetype not in ARCHETYPE_IDS:
        archetype = "tutorial"

    domain_l2 = parsed.get("domain_l2", "other")
    if domain_l2 not in DOMAIN_L2_ALL:
        domain_l2 = "other"

    archetype_confidence = parsed.get("archetype_confidence")
    if not isinstance(archetype_confidence, (int, float)):
        archetype_confidence = 0.5
    archetype_confidence = max(0.0, min(1.0, float(archetype_confidence)))

    archetype_reason = str(parsed.get("archetype_reason", ""))
    topology_type = ARCHETYPE_TO_TOPOLOGY.get(archetype, "procedural")

    return {
        "archetype": archetype,
        "archetype_confidence": archetype_confidence,
        "archetype_reason": archetype_reason,
        "domain_l2": domain_l2,
        "topology_type": topology_type,
    }


# ─────────────────────────────────────────────────────────────────────────────
# v7: Bottom-up pipeline — L2-first dense captioning + L1 aggregation
# ─────────────────────────────────────────────────────────────────────────────

def _split_l2_first_response(
    parsed: dict,
    n_sampled_frames: int,
    resize_max_width: int,
    jpeg_quality: int,
    clip_duration: float,
) -> dict[str, Any]:
    """Split Stage 1 (L2-first) VLM response into validated annotation fields.

    Unlike _split_merged_response() which un-nests events from phases, this
    processes a FLAT events list directly (no phases in the VLM output).

    Returns dict with _stage1_events (temporary), classification fields,
    and aggregation hints (summary, global_phase_criterion).
    """
    summary = parsed.get("summary", "")
    global_phase_criterion = parsed.get("global_phase_criterion", "")

    # ── Classification fields (same validation as _split_merged_response) ──
    archetype = parsed.get("paradigm", "tutorial")
    if archetype not in PARADIGM_IDS:
        archetype = "tutorial"

    archetype_confidence = parsed.get("paradigm_confidence")
    if not isinstance(archetype_confidence, (int, float)):
        archetype_confidence = 0.5
    archetype_confidence = max(0.0, min(1.0, float(archetype_confidence)))

    archetype_reason = str(parsed.get("paradigm_reason", ""))

    domain_l2 = parsed.get("domain_l2", "other")
    if domain_l2 not in DOMAIN_L2_ALL:
        domain_l2 = "other"

    topology_type = PARADIGM_TO_TOPOLOGY.get(archetype, "procedural")
    video_caption = str(parsed.get("video_caption", ""))

    # Feasibility
    raw_feas = parsed.get("feasibility", {})
    if not isinstance(raw_feas, dict):
        raw_feas = {}
    feas_score = raw_feas.get("score")
    if not isinstance(feas_score, (int, float)):
        feas_score = 0.5
    feas_score = max(0.0, min(1.0, float(feas_score)))
    feas_skip = bool(raw_feas.get("skip", False))
    feas_skip_reason = raw_feas.get("skip_reason")
    valid_skip_reasons = {"talk_dominant", "ambient_static", "low_visual_dynamics", "too_short", "low_feasibility"}
    if feas_skip_reason not in valid_skip_reasons:
        feas_skip_reason = None
    est_events = raw_feas.get("estimated_n_events", 0)
    visual_dynamics = raw_feas.get("visual_dynamics", "medium")
    if visual_dynamics not in {"high", "medium", "low"}:
        visual_dynamics = "medium"

    # Programmatic overrides
    if visual_dynamics == "low" and est_events < 2 and not feas_skip:
        feas_skip = True
        feas_skip_reason = "low_visual_dynamics"
    if feas_score < 0.4 and not feas_skip:
        feas_skip = True
        feas_skip_reason = "low_feasibility"

    feasibility = {
        "score": feas_score,
        "skip": feas_skip,
        "skip_reason": feas_skip_reason,
        "estimated_n_phases": 0,
        "estimated_n_events": int(est_events) if isinstance(est_events, (int, float)) else 0,
        "visual_dynamics": visual_dynamics,
    }

    # Video metadata
    raw_meta = parsed.get("video_metadata", {})
    if not isinstance(raw_meta, dict):
        raw_meta = {}
    camera_style = raw_meta.get("camera_style", "unknown")
    if camera_style not in {"static_tripod", "handheld", "multi_angle", "first_person"}:
        camera_style = "unknown"
    editing_style = raw_meta.get("editing_style", "unknown")
    if editing_style not in {"continuous", "jump_cut", "montage", "mixed"}:
        editing_style = "unknown"
    video_metadata = {
        "has_text_overlay": bool(raw_meta.get("has_text_overlay", False)),
        "has_narration": bool(raw_meta.get("has_narration", False)),
        "camera_style": camera_style,
        "editing_style": editing_style,
    }

    # ── Flat events validation (no phase nesting) ─────────────────────────
    raw_events = parsed.get("events", [])
    if not isinstance(raw_events, list):
        raw_events = []

    valid_events: list[dict] = []
    for ev in raw_events:
        if not isinstance(ev, dict):
            continue
        ev_st = ev.get("start_time")
        ev_et = ev.get("end_time")
        if not (isinstance(ev_st, (int, float)) and isinstance(ev_et, (int, float)) and ev_st < ev_et):
            print(f"    WARN: event dropped (invalid timestamps: st={ev_st} et={ev_et})", flush=True)
            continue
        ev["start_time"] = round(ev_st)
        ev["end_time"] = min(round(ev_et), round(clip_duration))
        # L3 feasibility — force false for short events
        ev.setdefault("l3_feasible", True)
        if ev["end_time"] - ev["start_time"] < 10:
            ev["l3_feasible"] = False
        ev["l3_feasible"] = bool(ev["l3_feasible"])
        ev.setdefault("l3_reason", "")
        # Validate key_frame_indices
        raw_kf = ev.get("key_frame_indices", [])
        valid_kf: list[int] = []
        if isinstance(raw_kf, list):
            for idx in raw_kf:
                if isinstance(idx, (int, float)) and 1 <= int(idx) <= n_sampled_frames:
                    valid_kf.append(int(idx))
        if not valid_kf:
            # Fallback: frame at the midpoint of the event
            mid_sec = (ev["start_time"] + ev["end_time"]) / 2
            mid_frame = max(1, min(n_sampled_frames, round(mid_sec) + 1))
            valid_kf = [mid_frame]
        ev["key_frame_indices"] = valid_kf[:2]
        valid_events.append(ev)

    # Merge short events (<5s) into adjacent events
    merged_events: list[dict] = []
    for ev in valid_events:
        dur = ev["end_time"] - ev["start_time"]
        if dur >= 5:
            merged_events.append(ev)
        elif merged_events:
            prev = merged_events[-1]
            prev["end_time"] = max(prev["end_time"], ev["end_time"])
            print(f"    INFO: short event ({dur}s) merged into previous event", flush=True)
        else:
            merged_events.append(ev)

    # Sort by start_time and re-number
    merged_events.sort(key=lambda e: (e.get("start_time", 0), e.get("end_time", 0)))
    for i, ev in enumerate(merged_events, 1):
        ev["event_id"] = i

    return {
        "_stage1_events": merged_events,
        "summary": summary,
        "global_phase_criterion": global_phase_criterion,
        "archetype": archetype,
        "archetype_confidence": archetype_confidence,
        "archetype_reason": archetype_reason,
        "domain_l2": domain_l2,
        "domain_l1": resolve_domain_l1(domain_l2),
        "topology_type": topology_type,
        "video_caption": video_caption,
        "feasibility": feasibility,
        "video_metadata": video_metadata,
        "_sampling": {
            "n_sampled_frames": n_sampled_frames,
            "resize_max_width": resize_max_width,
            "jpeg_quality": jpeg_quality,
        },
    }


def _annotate_l2_first(
    frame_dir: Path,
    clip_duration: float,
    api_base: str, api_key: str, model: str,
    max_frames: int, resize_max_width: int, jpeg_quality: int,
) -> dict[str, Any]:
    """Stage 1 of bottom-up pipeline: L2-first dense video captioning.

    Detects L2 events directly from all frames without prior L1 phase annotation.
    Also produces classification fields and key_frame_indices per event.
    """
    all_frames = get_all_frame_files(frame_dir)
    if not all_frames:
        raise RuntimeError(f"no frames found in {frame_dir}")

    sampled = sample_uniform(all_frames, max_frames)
    frame_b64 = encode_frame_files(sampled, resize_max_width=resize_max_width, jpeg_quality=jpeg_quality)

    with Image.open(sampled[0]) as _img:
        orig_w, orig_h = _img.size
    avg_b64_len = sum(len(b) for b in frame_b64) // max(len(frame_b64), 1)
    avg_jpeg_kb = avg_b64_len * 3 // 4 // 1024
    print(f"  [L2-first] frames: {len(sampled)}/{len(all_frames)} orig={orig_w}x{orig_h} "
          f"resize_max={resize_max_width} q={jpeg_quality} avg_jpeg={avg_jpeg_kb}KB", flush=True)

    frame_labels = []
    for fp in sampled:
        idx = frame_stem_to_index(fp, 0)
        frame_labels.append(f"[Timestamp {format_mmss(frame_index_to_sec(idx))} | Frame {idx}]")

    duration = int(clip_duration)

    # Too short → skip
    if clip_duration < 15:
        return {
            "_stage1_events": [],
            "summary": "",
            "global_phase_criterion": "",
            "archetype": "tutorial",
            "archetype_confidence": 0.0,
            "archetype_reason": "too short for annotation",
            "domain_l2": "other",
            "domain_l1": "other",
            "topology_type": "procedural",
            "video_caption": "",
            "feasibility": {
                "score": 0.0, "skip": True, "skip_reason": "too_short",
                "estimated_n_phases": 0, "estimated_n_events": 0, "visual_dynamics": "low",
            },
            "video_metadata": {
                "has_text_overlay": False, "has_narration": False,
                "camera_style": "unknown", "editing_style": "unknown",
            },
            "_sampling": {
                "n_sampled_frames": len(sampled),
                "resize_max_width": resize_max_width,
                "jpeg_quality": jpeg_quality,
            },
        }

    prompt_text = get_l2_first_prompt(n_frames=len(sampled), duration_sec=duration)
    parsed = call_and_parse(api_base, api_key, model, SYSTEM_PROMPT, prompt_text, frame_b64, frame_labels)
    if parsed is None:
        raise RuntimeError("L2-first JSON parse failed after retry")

    return _split_l2_first_response(
        parsed, len(sampled), resize_max_width, jpeg_quality, clip_duration,
    )


def _merge_l1_aggregation(
    stage1_result: dict[str, Any],
    stage2_parsed: dict,
    clip_duration: float,
) -> dict[str, Any]:
    """Merge Stage 1 (L2 events) + Stage 2 (L1 phases) into backward-compatible annotation.

    Builds level1 and level2 dicts with identical structure to _split_merged_response().
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
        l3_feasible = any(ev.get("l3_feasible", False) for ev in member_events)

        l1_phases.append({
            "phase_id": phase.get("phase_id", len(l1_phases) + 1),
            "start_time": round(phase_start),
            "end_time": min(round(phase_end), round(clip_duration)),
            "phase_name": phase.get("phase_name", ""),
            "narrative_summary": phase.get("narrative_summary", ""),
            "event_split_criterion": phase.get("event_split_criterion", ""),
            "l3_feasible": l3_feasible,
            "l3_reason": phase.get("l3_reason", ""),
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
            "l3_feasible": any(events_by_id[eid].get("l3_feasible", False) for eid in orphan_ids),
            "l3_reason": "fallback",
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

    any_l3 = any(p.get("l3_feasible", False) for p in l1_phases)
    l3_feasibility = {
        "suitable": any_l3,
        "reason": "per-phase assessment (bottom-up)",
        "estimated_l3_actions": 0,
    }

    return {
        "level1": level1,
        "level2": level2,
        "summary": stage1_result.get("summary", ""),
        "global_phase_criterion": stage1_result.get("global_phase_criterion", ""),
        "l3_feasibility": l3_feasibility,
        "archetype": stage1_result.get("archetype", "tutorial"),
        "archetype_confidence": stage1_result.get("archetype_confidence", 0.5),
        "archetype_reason": stage1_result.get("archetype_reason", ""),
        "domain_l2": stage1_result.get("domain_l2", "other"),
        "domain_l1": stage1_result.get("domain_l1", "other"),
        "topology_type": stage1_result.get("topology_type", "procedural"),
        "video_caption": stage1_result.get("video_caption", ""),
        "feasibility": stage1_result.get("feasibility", {}),
        "video_metadata": stage1_result.get("video_metadata", {}),
    }


def _annotate_l1_aggregation(
    frame_dir: Path,
    clip_duration: float,
    stage1_result: dict[str, Any],
    api_base: str, api_key: str, model: str,
    resize_max_width: int, jpeg_quality: int,
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
                    f"Timestamp {format_mmss(frame_index_to_sec(kf_idx))} | Frame {kf_idx}]"
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
    classify_frames: int = 64,
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
    if level in ("merged", "l2_first"):
        if not overwrite and existing.get("level1") is not None and existing.get("level2") is not None:
            return {"clip_key": key, "ok": True, "error": None, "skipped": True}
    elif level == "3":
        if not overwrite and existing.get("level3") is not None:
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

    # For L3, l3_frames_dir is the base dir for per-event subfolders:
    #   {l3_frames_dir}/{clip_key}_ev{event_id}/
    # frame_dir (full-video 1fps) is used as fallback when per-event dir is absent.

    try:
        if level == "l2_first":
            # ── Bottom-up: Stage 1 (L2 events) + Stage 2 (L1 aggregation) ──
            stage1_result = _annotate_l2_first(
                frame_dir, clip_duration,
                api_base, api_key, model,
                max_frames_per_call, resize_max_width, jpeg_quality,
            )
            n_events = len(stage1_result.get("_stage1_events", []))
            print(f"  [{key}] L2-first: {n_events} events, "
                  f"paradigm={stage1_result.get('archetype')} "
                  f"domain={stage1_result.get('domain_l2')}"
                  f"{' SKIP' if stage1_result.get('feasibility', {}).get('skip') else ''}",
                  flush=True)

            if stage1_result.get("_stage1_events"):
                merged_result = _annotate_l1_aggregation(
                    frame_dir, clip_duration, stage1_result,
                    api_base, api_key, model,
                    resize_max_width, jpeg_quality,
                )
            else:
                merged_result = {
                    "level1": {"macro_phases": [], "_sampling": stage1_result.get("_sampling", {})},
                    "level2": {"events": []},
                    "l3_feasibility": {"suitable": False, "reason": "no events", "estimated_l3_actions": 0},
                }
                # Carry over classification fields
                for k in ("summary", "global_phase_criterion", "archetype",
                           "archetype_confidence", "archetype_reason",
                           "domain_l2", "domain_l1", "topology_type",
                           "video_caption", "feasibility", "video_metadata"):
                    merged_result[k] = stage1_result.get(k, "")

            n_phases = len(merged_result.get("level1", {}).get("macro_phases", []))
            n_final_events = len(merged_result.get("level2", {}).get("events", []))
            print(f"  [{key}] L1-agg: {n_phases} phases, {n_final_events} events", flush=True)

        elif level == "merged":
            # Step 0: Classify archetype + domain (64 frames, lightweight)
            classify_result = _classify_archetype(
                frame_dir, clip_duration,
                api_base, api_key, model,
                max_frames=classify_frames,
                resize_max_width=resize_max_width, jpeg_quality=jpeg_quality,
            )
            archetype_id = classify_result["archetype"]
            print(f"  [{key}] archetype={archetype_id} "
                  f"conf={classify_result['archetype_confidence']:.2f} "
                  f"domain={classify_result['domain_l2']}", flush=True)

            # Step 1: Paradigm-driven merged L1+L2 annotation (all frames)
            merged_result = _annotate_merged_l1l2(
                frame_dir, clip_duration,
                api_base, api_key, model,
                max_frames_per_call, resize_max_width, jpeg_quality,
                archetype_id=archetype_id,
            )
            # Merge classify metadata into annotation result
            merged_result.update(classify_result)
            print(f"  [{key}] "
                  f"phases={len(merged_result.get('level1', {}).get('macro_phases', []))} "
                  f"events={len(merged_result.get('level2', {}).get('events', []))}"
                  f"{' SKIP' if merged_result.get('feasibility', {}).get('skip') else ''}",
                  flush=True)
        elif level == "3":
            # ── Archetype-aware L3 routing ──
            archetype_id = _resolve_archetype(existing)
            cfg = get_archetype(archetype_id)
            topology_type = ARCHETYPE_TO_TOPOLOGY.get(archetype_id, "procedural")

            # Check if L3 is enabled for this archetype
            if not cfg.l3.enabled:
                result_key, result_val = "level3", {
                    "micro_type": "skip",
                    "grounding_results": [],
                    "_segment_calls": [],
                    "_skip_reason": f"L3 disabled for archetype={archetype_id}",
                }
            # Check L3 feasibility from merged annotation (VLM assessment)
            elif existing.get("l3_feasibility", {}).get("suitable") is False:
                feas_reason = existing.get("l3_feasibility", {}).get("reason", "")
                print(f"  [{key}] L3 skipped: not suitable ({feas_reason})", flush=True)
                result_key, result_val = "level3", {
                    "micro_type": "skip",
                    "grounding_results": [],
                    "_segment_calls": [],
                    "_skip_reason": f"L3 not suitable: {feas_reason}",
                }
            else:
                l3_parent = get_l3_parent_type(archetype_id)
                if l3_parent == "phase":
                    # L3 from phases (e.g., performance archetype)
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
                        archetype_id=archetype_id,
                        l1_result=l1,
                    )
                else:
                    # L3 from events (default: leaf-node routing)
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
                        archetype_id=archetype_id,
                        l1_result=existing.get("level1"),
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
    if level in ("merged", "l2_first"):
        ann.update(merged_result)  # overwrites level1, level2, archetype, domain_l2, summary
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
    1. Extracts and validates classification fields (paradigm, domain, feasibility)
    2. Extracts and validates summary
    3. Strips nested events out of phases → flat events list
    4. Tags each event with parent_phase_id
    5. Re-numbers event_id globally by start_time

    Returns dict with keys: level1, level2, summary, global_phase_criterion,
    archetype, domain_l2, topology_type, feasibility, video_metadata, video_caption.
    """
    summary = parsed.get("summary", "")
    global_phase_criterion = parsed.get("global_phase_criterion", "")

    # ── Classification fields (unified prompt v5/v6) ──────────────────
    archetype = parsed.get("paradigm", "tutorial")
    if archetype not in PARADIGM_IDS:
        archetype = "tutorial"

    archetype_confidence = parsed.get("paradigm_confidence")
    if not isinstance(archetype_confidence, (int, float)):
        archetype_confidence = 0.5
    archetype_confidence = max(0.0, min(1.0, float(archetype_confidence)))

    archetype_reason = str(parsed.get("paradigm_reason", ""))

    domain_l2 = parsed.get("domain_l2", "other")
    if domain_l2 not in DOMAIN_L2_ALL:
        domain_l2 = "other"

    topology_type = PARADIGM_TO_TOPOLOGY.get(archetype, "procedural")

    video_caption = str(parsed.get("video_caption", ""))

    # Feasibility
    raw_feas = parsed.get("feasibility", {})
    if not isinstance(raw_feas, dict):
        raw_feas = {}
    feas_score = raw_feas.get("score")
    if not isinstance(feas_score, (int, float)):
        feas_score = 0.5
    feas_score = max(0.0, min(1.0, float(feas_score)))
    feas_skip = bool(raw_feas.get("skip", False))
    feas_skip_reason = raw_feas.get("skip_reason")
    valid_skip_reasons = {"talk_dominant", "ambient_static", "low_visual_dynamics", "too_short", "low_feasibility"}
    if feas_skip_reason not in valid_skip_reasons:
        feas_skip_reason = None
    est_phases = raw_feas.get("estimated_n_phases", 0)
    est_events = raw_feas.get("estimated_n_events", 0)
    visual_dynamics = raw_feas.get("visual_dynamics", "medium")
    if visual_dynamics not in {"high", "medium", "low"}:
        visual_dynamics = "medium"

    # Programmatic override rules (ported from stage1_classify.py)
    if visual_dynamics == "low" and est_events < 2 and not feas_skip:
        feas_skip = True
        feas_skip_reason = "low_visual_dynamics"
    if feas_score < 0.4 and not feas_skip:
        feas_skip = True
        feas_skip_reason = "low_feasibility"

    feasibility = {
        "score": feas_score,
        "skip": feas_skip,
        "skip_reason": feas_skip_reason,
        "estimated_n_phases": int(est_phases) if isinstance(est_phases, (int, float)) else 0,
        "estimated_n_events": int(est_events) if isinstance(est_events, (int, float)) else 0,
        "visual_dynamics": visual_dynamics,
    }

    # Video metadata
    raw_meta = parsed.get("video_metadata", {})
    if not isinstance(raw_meta, dict):
        raw_meta = {}
    camera_style = raw_meta.get("camera_style", "unknown")
    if camera_style not in {"static_tripod", "handheld", "multi_angle", "first_person"}:
        camera_style = "unknown"
    editing_style = raw_meta.get("editing_style", "unknown")
    if editing_style not in {"continuous", "jump_cut", "montage", "mixed"}:
        editing_style = "unknown"
    video_metadata = {
        "has_text_overlay": bool(raw_meta.get("has_text_overlay", False)),
        "has_narration": bool(raw_meta.get("has_narration", False)),
        "camera_style": camera_style,
        "editing_style": editing_style,
    }

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
            print(f"    WARN: phase {phase_id} dropped (invalid timestamps: st={st} et={et})", flush=True)
            continue
        phase["start_time"] = round(st)
        phase["end_time"] = min(round(et), round(clip_duration))
        # Per-phase L3 feasibility (v5: replaces video-level l3_feasibility)
        phase.setdefault("l3_feasible", True)
        phase["l3_feasible"] = bool(phase["l3_feasible"])
        phase.setdefault("l3_reason", "")
        l1_phases.append(phase)

        # Collect events with parent linkage
        phase_valid_events: list[dict] = []
        for ev in phase_events:
            if not isinstance(ev, dict):
                continue
            ev_st = ev.get("start_time")
            ev_et = ev.get("end_time")
            if not (isinstance(ev_st, (int, float)) and isinstance(ev_et, (int, float)) and ev_st < ev_et):
                print(f"    WARN: event in phase {phase_id} dropped (invalid timestamps: st={ev_st} et={ev_et})", flush=True)
                continue
            ev["start_time"] = round(ev_st)
            ev["end_time"] = min(round(ev_et), round(clip_duration))
            ev["parent_phase_id"] = phase_id
            # Per-event L3 feasibility — force false for short events
            ev.setdefault("l3_feasible", True)
            if ev["end_time"] - ev["start_time"] < 10:
                ev["l3_feasible"] = False
            ev["l3_feasible"] = bool(ev["l3_feasible"])
            ev.setdefault("l3_reason", "")
            phase_valid_events.append(ev)

        # Merge short events (<5s) into adjacent events within the same phase
        merged_events: list[dict] = []
        for ev in phase_valid_events:
            dur = ev["end_time"] - ev["start_time"]
            if dur >= 5:
                merged_events.append(ev)
            elif merged_events:
                # Merge into the previous event by extending its end_time
                prev = merged_events[-1]
                prev["end_time"] = max(prev["end_time"], ev["end_time"])
                print(f"    INFO: short event ({dur}s) in phase {phase_id} merged into previous event", flush=True)
            else:
                # No previous event — keep it anyway (first event in phase)
                merged_events.append(ev)

        all_events.extend(merged_events)

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

    # L3 feasibility: aggregate from per-phase l3_feasible flags
    # Video is L3-feasible if ANY phase is L3-feasible
    any_l3 = any(p.get("l3_feasible", True) for p in l1_phases)
    l3_feasibility = {
        "suitable": any_l3,
        "reason": "per-phase assessment",
        "estimated_l3_actions": 0,
    }

    return {
        "level1": level1,
        "level2": level2,
        "summary": summary,
        "global_phase_criterion": global_phase_criterion,
        "l3_feasibility": l3_feasibility,
        # Classification fields (unified prompt v5)
        "archetype": archetype,
        "archetype_confidence": archetype_confidence,
        "archetype_reason": archetype_reason,
        "domain_l2": domain_l2,
        "domain_l1": resolve_domain_l1(domain_l2),
        "topology_type": topology_type,
        "video_caption": video_caption,
        "feasibility": feasibility,
        "video_metadata": video_metadata,
    }


def _annotate_merged_l1l2(
    frame_dir: Path,
    clip_duration: float,
    api_base: str, api_key: str, model: str,
    max_frames: int, resize_max_width: int, jpeg_quality: int,
    archetype_id: str = "tutorial",
) -> dict[str, Any]:
    """
    Paradigm-driven merged L1+L2 annotation (Step 1 of 2-step pipeline).

    Uses archetype-specific prompts. Archetype/domain classification
    is done separately in Step 0 (_classify_archetype).

    Returns dict of annotation updates including level1, level2, summary.
    """
    all_frames = get_all_frame_files(frame_dir)
    if not all_frames:
        raise RuntimeError(f"no frames found in {frame_dir}")

    sampled = sample_uniform(all_frames, max_frames)
    frame_b64 = encode_frame_files(sampled, resize_max_width=resize_max_width, jpeg_quality=jpeg_quality)

    # Log original frame size (before resize) for cost analysis
    with Image.open(sampled[0]) as _img:
        orig_w, orig_h = _img.size
    avg_b64_len = sum(len(b) for b in frame_b64) // max(len(frame_b64), 1)
    avg_jpeg_kb = avg_b64_len * 3 // 4 // 1024
    print(f"  frames: {len(sampled)}/{len(all_frames)} orig={orig_w}x{orig_h} "
          f"resize_max={resize_max_width} q={jpeg_quality} avg_jpeg={avg_jpeg_kb}KB", flush=True)

    # Real-time labels (same format as L2)
    frame_labels = []
    for fp in sampled:
        idx = frame_stem_to_index(fp, 0)
        frame_labels.append(f"[Timestamp {format_mmss(frame_index_to_sec(idx))} | Frame {idx}]")

    duration = int(clip_duration)

    # Programmatic pre-check: too short → skip
    if clip_duration < 15:
        return {
            "level1": {"macro_phases": [], "_sampling": {
                "n_sampled_frames": len(sampled),
                "resize_max_width": resize_max_width,
                "jpeg_quality": jpeg_quality,
            }},
            "level2": {"events": []},
            "summary": "",
            "global_phase_criterion": "",
            "l3_feasibility": {"suitable": False, "reason": "too short", "estimated_l3_actions": 0},
            "archetype": "tutorial",
            "archetype_confidence": 0.0,
            "archetype_reason": "too short for annotation",
            "domain_l2": "other",
            "domain_l1": "other",
            "topology_type": "procedural",
            "video_caption": "",
            "feasibility": {
                "score": 0.0, "skip": True, "skip_reason": "too_short",
                "estimated_n_phases": 0, "estimated_n_events": 0, "visual_dynamics": "low",
            },
            "video_metadata": {
                "has_text_overlay": False, "has_narration": False,
                "camera_style": "unknown", "editing_style": "unknown",
            },
        }

    prompt_text = get_archetype_merged_prompt(
        archetype_id=archetype_id,
        n_frames=len(sampled), duration_sec=duration,
    )
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
    archetype_id: str = "tutorial",
    l1_result: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    Level 3: Local Temporal Grounding (archetype-aware).

    Source selection by archetype L3 parent type:
      - parent="event":  iterate L2 events → per-event frames
      - parent="phase":  iterate L1 phases → per-phase frames
    Falls back to filtering the full-video 1fps frame dir when dedicated dir is absent.
    """
    cfg = get_archetype(archetype_id)
    l3_parent = get_l3_parent_type(archetype_id)

    # ── Build source list based on archetype L3 parent ──
    sources: list[dict[str, Any]] = []

    if l3_parent == "phase" and l1_result is not None:
        # periodic: L3 sources from L1 phases
        phases = l1_result.get("macro_phases", [])
        for phase in phases:
            if not isinstance(phase, dict):
                continue
            # Skip phases marked as L3-infeasible
            if not phase.get("l3_feasible", True):
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

        # Build phase_id → l3_feasible mapping
        phase_l3_ok: dict[int, bool] = {}
        for p in l1_phases:
            if isinstance(p, dict):
                phase_l3_ok[p.get("phase_id")] = p.get("l3_feasible", True)

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
                # Skip phases marked as L3-infeasible
                if not phase_l3_ok.get(pid, True):
                    continue
                children = phase_events.get(pid, [])
                if children:
                    # Phase has events → events are leaf nodes
                    for ev in children:
                        # Skip events marked as L3-infeasible
                        if not ev.get("l3_feasible", True):
                            continue
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

    micro_type = cfg.l3.micro_type or "state_change"
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

        prompt_text = get_archetype_l3_prompt(
            archetype_id=archetype_id,
            clip_start_sec=int(start_time),
            clip_end_sec=int(end_time),
            action_query=instruction,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hierarchical video annotation pipeline (merged L1+L2, L3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--jsonl", default=None,
                        help="可选：输入 JSONL。若不提供，则直接遍历 --frames-dir 下所有样本。")
    parser.add_argument("--frames-dir", required=True,
                        help="Root directory of pre-extracted frames")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write per-clip annotation JSON files")
    parser.add_argument("--l3-frames-dir", default=None,
                        help="High-FPS frames directory for L3 (falls back to --frames-dir)")
    parser.add_argument("--level", type=str, choices=["merged", "l2_first", "3"], default="merged",
                        help="Annotation level (merged=top-down L1+L2, l2_first=bottom-up L2→L1, 3=L3 grounding)")
    parser.add_argument("--classify-frames", type=int, default=64,
                        help="Number of frames for Step 0 archetype classification")
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

    level = args.level
    l3_frames_dir = Path(args.l3_frames_dir) if args.l3_frames_dir else None

    print(f"Annotating {len(records)} clips at Level {args.level}")
    print(f"API: {args.api_base}  model: {args.model}  workers: {args.workers}")
    print(f"resize_max_width={args.resize_max_width}  jpeg_quality={args.jpeg_quality}")
    if level == "merged":
        print(f"Merged mode: Step 0 classify ({args.classify_frames} frames) + Step 1 archetype-driven L1+L2")
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
                args.classify_frames,
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
