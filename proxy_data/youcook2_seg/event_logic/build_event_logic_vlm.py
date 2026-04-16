#!/usr/bin/env python3
"""
build_event_logic_vlm.py — VLM-powered Event Logic data construction pipeline.

Calls a text-only LLM "Task Architect" (via Novita/OpenAI-compatible API) to
design three types of event-logic questions from hierarchical segmentation
annotations:

  1. predict_next   — Predict the next step (MCQ, letter answer)
  2. fill_blank     — Fill-in-the-blank (MCQ, letter answer)
  3. sort           — Sequence sorting (digit-sequence answer)

The LLM receives the annotation's "action script" (text only) and returns
structured JSON specifying context IDs, correct answer, and distractors.
This script then resolves IDs to video clip paths and assembles EasyR1
training records.

Usage:

    ANN=/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/hier_seg_annotation/annotations_fixed_gmn25
    CLIPS=/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/hier_seg_annotation/clips
    OUT=/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/event_logic/vlm

    python build_event_logic_vlm.py \\
        --annotation-dir $ANN --clip-dir $CLIPS \\
        --output-dir $OUT \\
        --api-base https://api.novita.ai/v3/openai \\
        --model pa/gmn-2.5-fls \\
        --tasks predict_next fill_blank sort \\
        --workers 8 --temperature 0.7 \\
        --complete-only --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ── sys.path setup (same pattern as build_event_shuffle.py) ────────────────
_PROXY_DATA_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROXY_DATA_DIR not in sys.path:
    sys.path.insert(0, _PROXY_DATA_DIR)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from shared.seg_source import (  # noqa: E402
    load_annotations,
    get_l2_event_atomic_path,
    get_l3_action_atomic_path,
)
from vlm_task_prompts import (  # noqa: E402
    TASK_ARCHITECT_SYSTEM_PROMPT,
    get_predict_next_user_prompt,
    get_fill_blank_user_prompt,
    get_sequence_sort_user_prompt,
)
from prompts import (  # noqa: E402
    get_add_prompt_generic,
    get_replace_prompt_generic,
    get_sort_prompt_generic,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
# Suppress noisy HTTP-level logs from the openai library
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

ALL_TASKS = ("predict_next", "fill_blank", "sort")


# =====================================================================
# Clip path collection helpers (for --collect-clips-only mode)
# =====================================================================

def _collect_clips_predict_next(parsed: dict, ann: dict, clip_dir: str) -> list[str]:
    """Collect clip paths needed by predict_next task (context clips only)."""
    paths = []
    for cid in parsed.get("context_ids", []):
        item = resolve_item(cid, ann, clip_dir)
        if item:
            paths.append(item["clip_path"])
    return paths


def _collect_clips_fill_blank(parsed: dict, ann: dict, clip_dir: str) -> list[str]:
    """Collect clip paths needed by fill_blank task (before + after, not missing)."""
    paths = []
    for bid in parsed.get("before_ids", []):
        item = resolve_item(bid, ann, clip_dir)
        if item:
            paths.append(item["clip_path"])
    for aid in parsed.get("after_ids", []):
        item = resolve_item(aid, ann, clip_dir)
        if item:
            paths.append(item["clip_path"])
    return paths


def _collect_clips_sort(parsed: dict, ann: dict, clip_dir: str) -> list[str]:
    """Collect clip paths needed by sort task (all ordered clips)."""
    paths = []
    for oid in parsed.get("ordered_ids", []):
        item = resolve_item(oid, ann, clip_dir)
        if item:
            paths.append(item["clip_path"])
    return paths


_TASK_CLIP_COLLECTORS = {
    "predict_next": _collect_clips_predict_next,
    "fill_blank": _collect_clips_fill_blank,
    "sort": _collect_clips_sort,
}


# =====================================================================
# Token tracking (thread-safe, same pattern as reclassify_domain.py)
# =====================================================================

_token_lock = threading.Lock()
_token_usage: dict[str, int] = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "api_calls": 0,
}


def _accumulate_usage(usage) -> None:
    if usage is None:
        return
    with _token_lock:
        _token_usage["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
        _token_usage["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
        _token_usage["total_tokens"] += getattr(usage, "total_tokens", 0) or 0
        _token_usage["api_calls"] += 1


# =====================================================================
# Script text generation
# =====================================================================

def build_script_text(ann: dict) -> str | None:
    """Convert an annotation JSON to a readable "Action Script" for the LLM.

    Returns None if the annotation is missing L1/L2.

    L3 format: nested — grounding_results[].sub_actions[] (from annotate.py).
    """
    l1 = ann.get("level1")
    l2 = ann.get("level2")
    if not l1 or not l2 or l2.get("_parse_error"):
        return None

    lines = ["[Video Info]"]
    lines.append(f"Category: {ann.get('domain_l1', 'other')} > {ann.get('domain_l2', 'other')}")
    caption = ann.get("video_caption", "").strip()
    if caption:
        lines.append(f"Summary: {caption}")
    lines.append("")
    lines.append("--- Action Script ---")

    phases = l1.get("macro_phases", [])
    events = l2.get("events", [])
    l3 = ann.get("level3")
    grounding_results = (
        l3.get("grounding_results", [])
        if l3 and not l3.get("_parse_error")
        else []
    )

    # Build lookup: event_id -> sorted sub_actions (nested format)
    actions_by_event: dict[int, list[dict]] = {}
    for gr in grounding_results:
        if not isinstance(gr, dict):
            continue
        eid = gr.get("event_id")
        subs = gr.get("sub_actions", [])
        if eid is not None and isinstance(subs, list):
            actions_by_event[eid] = sorted(
                [s for s in subs if isinstance(s, dict)],
                key=lambda s: s.get("start_time", 0),
            )

    # Build lookup: phase_id -> sorted events
    events_by_phase: dict[int, list[dict]] = defaultdict(list)
    for ev in events:
        if isinstance(ev, dict) and ev.get("parent_phase_id") is not None:
            events_by_phase[ev["parent_phase_id"]].append(ev)
    for pid in events_by_phase:
        events_by_phase[pid].sort(key=lambda e: e.get("start_time", 0))

    for phase in sorted(phases, key=lambda p: p.get("start_time", 0)):
        if not isinstance(phase, dict):
            continue
        ph_id = phase.get("phase_id")
        ph_name = phase.get("phase_name", "Unnamed Phase")
        ph_st = int(phase.get("start_time", 0))
        ph_et = int(phase.get("end_time", 0))
        lines.append(f"\n[Phase {ph_id}]: {ph_name} ({ph_st}s - {ph_et}s)")

        for ev in events_by_phase.get(ph_id, []):
            ev_id = ev.get("event_id")
            instr = ev.get("instruction", "").strip()
            ev_st = int(ev.get("start_time", 0))
            ev_et = int(ev.get("end_time", 0))
            lines.append(f"  [Event {ev_id}]: {instr} ({ev_st}s - {ev_et}s)")

            for act in actions_by_event.get(ev_id, []):
                act_id = act.get("action_id")
                sub = act.get("sub_action", "").strip()
                act_st = int(act.get("start_time", 0))
                act_et = int(act.get("end_time", 0))
                if sub:
                    lines.append(
                        f"    - Action {ev_id}.{act_id}: {sub} ({act_st}s - {act_et}s)"
                    )

    return "\n".join(lines)


# =====================================================================
# ID parsing & item resolution
# =====================================================================

_ID_RE = re.compile(r"(Phase|Event|Action)\s+(\d+)(?:\.(\d+))?")


def parse_item_id(id_str: str) -> tuple[str, int, int | None]:
    """Parse "Event 3" -> ("Event", 3, None), "Action 4.2" -> ("Action", 4, 2)."""
    m = _ID_RE.match(id_str.strip())
    if not m:
        raise ValueError(f"Unrecognized item ID format: {id_str!r}")
    level = m.group(1)
    primary = int(m.group(2))
    sub = int(m.group(3)) if m.group(3) else None
    return level, primary, sub


def resolve_item(
    id_str: str,
    ann: dict,
    clip_dir: str,
) -> dict | None:
    """Resolve an item ID to its clip path + text description.

    Returns {"id", "text", "clip_path", "start", "end"} or None.
    L3 is nested format: grounding_results[].sub_actions[].
    """
    try:
        level, primary, sub = parse_item_id(id_str)
    except ValueError:
        return None

    clip_key = ann.get("clip_key", "")

    if level == "Event":
        events = ann.get("level2", {}).get("events", [])
        ev = next((e for e in events if e.get("event_id") == primary), None)
        if not ev:
            return None
        return {
            "id": id_str,
            "text": ev.get("instruction", "").strip(),
            "clip_path": get_l2_event_atomic_path(
                clip_key, primary,
                int(ev["start_time"]), int(ev["end_time"]),
                clip_dir,
            ),
            "start": int(ev["start_time"]),
            "end": int(ev["end_time"]),
        }

    elif level == "Action":
        # primary = parent event_id, sub = action_id
        if sub is None:
            return None
        l3 = ann.get("level3", {})
        for gr in l3.get("grounding_results", []):
            if not isinstance(gr, dict) or gr.get("event_id") != primary:
                continue
            for sa in gr.get("sub_actions", []):
                if isinstance(sa, dict) and sa.get("action_id") == sub:
                    return {
                        "id": id_str,
                        "text": sa.get("sub_action", "").strip(),
                        "clip_path": get_l3_action_atomic_path(
                            clip_key, sub, primary,
                            int(sa["start_time"]), int(sa["end_time"]),
                            clip_dir,
                        ),
                        "start": int(sa["start_time"]),
                        "end": int(sa["end_time"]),
                    }
        return None

    return None


def _check_granularity(id_list: list[str]) -> bool:
    """Return True if all IDs share the same granularity (Event or Action)."""
    levels = set()
    for id_str in id_list:
        try:
            level, _, _ = parse_item_id(id_str)
            levels.add(level)
        except ValueError:
            return False
    return len(levels) == 1


def _check_contiguity(id_list: list[str]) -> bool:
    """Return True if IDs form a contiguous sequence with no gaps.

    For Event IDs: event numbers must be strictly consecutive (1,2,3 — not 1,3).

    For Action IDs two cases are handled:
    - Same parent event: sub-IDs must be consecutive (3.1,3.2,3.3 — not 3.1,3.3).
    - Cross-event (e.g. Action 8.2 → Action 9.1): the parent event numbers must
      be monotonically increasing (no skipped events), and within each event the
      sub-IDs that appear must themselves be consecutive. Cross-event transitions
      are natural temporal progressions and are allowed.
    """
    if len(id_list) < 2:
        return True
    parsed = []
    for id_str in id_list:
        try:
            level, primary, sub = parse_item_id(id_str)
            parsed.append((level, primary, sub))
        except ValueError:
            return False

    first_level = parsed[0][0]
    if first_level == "Event":
        ids = [p[1] for p in parsed]
        return ids == list(range(ids[0], ids[0] + len(ids)))

    elif first_level == "Action":
        parents = [p[1] for p in parsed]
        sub_ids = [p[2] for p in parsed]

        # All sub IDs must be present (no Actions without sub-index)
        if any(s is None for s in sub_ids):
            return False

        # Group by parent event, preserving order
        groups: list[tuple[int, list[int]]] = []
        for parent, sub in zip(parents, sub_ids):
            if groups and groups[-1][0] == parent:
                groups[-1][1].append(sub)
            else:
                groups.append((parent, [sub]))

        # Parent event IDs must be monotonically increasing (no skips)
        event_ids = [g[0] for g in groups]
        if event_ids != sorted(set(event_ids)):
            return False  # repeated or out-of-order event IDs

        # Within each event, the sub-IDs that appear must be consecutive
        for _, subs in groups:
            if subs != list(range(subs[0], subs[0] + len(subs))):
                return False

        return True

    return False


# =====================================================================
# Video resolution probing & concatenation with inline black frame
# =====================================================================


def _get_video_resolution(path: str) -> tuple[int, int]:
    """Return (width, height) of video at *path* via ffprobe. Falls back to (320, 240)."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0",
                path,
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            if len(parts) == 2:
                w, h = int(parts[0]), int(parts[1])
                if w > 0 and h > 0:
                    return w, h
        log.warning("ffprobe failed for %s (rc=%d): %s",
                    path, result.returncode, result.stderr.strip()[:200])
    except Exception as exc:
        log.warning("ffprobe exception for %s: %s", path, exc)
    return 320, 240


def _concat_clips_with_black(
    before_paths: list[str],
    after_paths: list[str],
    output_path: str,
    black_duration: float = 2.0,
) -> bool:
    """Concatenate *before* clips + an inline black frame + *after* clips.

    The black frame is generated on-the-fly via ffmpeg's ``lavfi`` source at
    the same resolution as the first available clip — no pre-generated
    placeholder file required.  All streams are normalised to the reference
    resolution so codec / size mismatches are handled in a single pass.
    """
    if os.path.exists(output_path):
        return True

    all_clip_paths = before_paths + after_paths
    if not all_clip_paths:
        return False

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Probe reference resolution from first existing clip
    ref_path = next((p for p in all_clip_paths if os.path.exists(p)), None)
    if ref_path is None:
        log.error("concat_with_black: no clip files exist on disk for %s — "
                  "cannot determine resolution. First path: %s",
                  output_path, all_clip_paths[0] if all_clip_paths else "N/A")
        return False
    ref_w, ref_h = _get_video_resolution(ref_path)

    scale_pad = (
        f"scale={ref_w}:{ref_h}:force_original_aspect_ratio=decrease,"
        f"pad={ref_w}:{ref_h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=2"
    )

    # Build ffmpeg inputs and filter_complex parts
    inputs: list[str] = []
    filter_parts: list[str] = []
    idx = 0

    # Before clips
    for p in before_paths:
        inputs.extend(["-i", p])
        filter_parts.append(f"[{idx}:v]{scale_pad}[v{idx}]")
        idx += 1

    # Inline black frame (lavfi — no file needed)
    black_idx = idx
    inputs.extend([
        "-f", "lavfi",
        "-i", f"color=c=black:s={ref_w}x{ref_h}:d={black_duration}:r=2",
    ])
    filter_parts.append(f"[{black_idx}:v]setsar=1[v{black_idx}]")
    idx += 1

    # After clips
    for p in after_paths:
        inputs.extend(["-i", p])
        filter_parts.append(f"[{idx}:v]{scale_pad}[v{idx}]")
        idx += 1

    # Concat all streams
    concat_in = "".join(f"[v{i}]" for i in range(idx))
    filter_parts.append(f"{concat_in}concat=n={idx}:v=1:a=0[out]")

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", ";".join(filter_parts),
        "-map", "[out]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-an",
        output_path,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        log.error("concat_with_black failed for %s: %s", output_path, exc)
        if os.path.exists(output_path):
            os.unlink(output_path)
        return False


def _concat_video_files(paths: list[str], output_path: str) -> bool:
    """Concatenate a list of video files into one using ffmpeg concat demuxer.

    Stream-copy first (fast); falls back to re-encode if codecs mismatch.
    Returns True on success.
    """
    if os.path.exists(output_path):
        return True

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    list_path = output_path + ".concat.txt"
    try:
        with open(list_path, "w", encoding="utf-8") as f:
            for p in paths:
                f.write(f"file '{p}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", list_path,
            "-c", "copy",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if result.returncode != 0:
            # Fallback: re-encode with normalization (handles mismatched resolution/fps/codec).
            # Probe first file for reference resolution so we don't downscale to 320x240.
            log.debug("concat stream-copy failed for %s, trying re-encode. stderr: %s",
                      output_path, result.stderr.decode(errors="replace")[:300])
            ref_w, ref_h = _get_video_resolution(paths[0]) if paths else (320, 240)
            cmd_reencode = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", list_path,
                "-vf", (
                    f"scale={ref_w}:{ref_h}:force_original_aspect_ratio=decrease,"
                    f"pad={ref_w}:{ref_h}:(ow-iw)/2:(oh-ih)/2"
                ),
                "-r", "2",
                "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-an",
                output_path,
            ]
            subprocess.run(cmd_reencode, check=True, capture_output=True, timeout=120)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        log.error("concat failed for %s: %s", output_path, exc)
        if os.path.exists(output_path):
            os.unlink(output_path)
        return False
    finally:
        if os.path.exists(list_path):
            os.unlink(list_path)


# =====================================================================
# LLM call
# =====================================================================

def call_task_architect(
    api_base: str,
    api_key: str,
    model: str,
    user_prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    retries: int = 3,
    label: str = "",
) -> str:
    """Call the LLM task architect. Returns raw response text."""
    from openai import OpenAI

    # Provider-specific key resolution
    if api_key:
        key = api_key
    elif "novita.ai" in api_base.lower():
        key = os.environ.get("NOVITA_API_KEY", "")
    elif "openrouter.ai" in api_base.lower():
        key = os.environ.get("OPENROUTER_API_KEY", "")
    else:
        key = os.environ.get("OPENAI_API_KEY", "")

    client = OpenAI(api_key=key, base_url=api_base)
    messages = [
        {"role": "system", "content": TASK_ARCHITECT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    last_error = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            _accumulate_usage(resp.usage)
            usage = resp.usage
            if usage:
                with _token_lock:
                    total_calls = _token_usage["api_calls"]
                    total_tokens = _token_usage["total_tokens"]
                log.info(
                    "[LLM] %s  in=%d out=%d | cumulative: calls=%d tokens=%d",
                    label or "call",
                    getattr(usage, "prompt_tokens", 0) or 0,
                    getattr(usage, "completion_tokens", 0) or 0,
                    total_calls,
                    total_tokens,
                )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            last_error = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise last_error  # type: ignore[misc]


# =====================================================================
# JSON parsing (3-tier fallback, same as reclassify_domain.py)
# =====================================================================

def parse_json_from_response(text: str) -> dict | None:
    """Extract JSON dict from LLM response using 3-tier fallback."""
    text = text.strip()
    # Tier 1: direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # Tier 2: markdown code block
    m = re.search(r"```(?:json)?\s*(\{[\s\S]+?\})\s*```", text)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    # Tier 3: first {...}
    m2 = re.search(r"\{[\s\S]+\}", text)
    if m2:
        try:
            obj = json.loads(m2.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return None


# =====================================================================
# Record assembly per task type
# =====================================================================

def _assemble_predict_next(
    parsed: dict,
    ann: dict,
    clip_dir: str,
    complete_only: bool,
    rng: random.Random,
    cot: bool = False,
) -> list[dict]:
    """Assemble Predict-Next MCQ record from validated LLM output."""
    context_ids = parsed.get("context_ids", [])
    correct_next_id = parsed.get("correct_next_id", "")
    correct_text = parsed.get("correct_next_text", "")
    distractors = parsed.get("distractors", [])
    granularity = parsed.get("granularity", "")
    clip_key = ann.get("clip_key", "?")

    if len(context_ids) < 1 or not correct_next_id or not correct_text or len(distractors) < 3:
        log.warning("[predict_next] %s: REJECT basic validation failed (ctx=%d, next=%r, dist=%d)",
                  clip_key, len(context_ids), bool(correct_next_id), len(distractors))
        return []

    # Trim to first 3 distractors if LLM returned more
    distractors = distractors[:3]

    # Granularity consistency
    all_ids = list(context_ids) + [correct_next_id]
    if not _check_granularity(all_ids):
        log.warning("[predict_next] %s: granularity mismatch in %s", clip_key, all_ids)
        return []

    # Contiguity check removed: non-action shots between clips can create
    # legitimate gaps in ID sequence (e.g. Action 8.2 -> Action 9.1)
    context_items = []
    for cid in context_ids:
        item = resolve_item(cid, ann, clip_dir)
        if item is None:
            log.warning("[predict_next] %s: resolve_item returned None for %s", clip_key, cid)
            return []
        if complete_only and not os.path.exists(item["clip_path"]):
            log.warning("[predict_next] %s: clip missing: %s", clip_key, item["clip_path"])
            return []
        context_items.append(item)

    # Verify correct_next_id exists
    correct_item = resolve_item(correct_next_id, ann, clip_dir)
    if correct_item is None:
        log.warning("[predict_next] %s: resolve_item returned None for correct_next %s", clip_key, correct_next_id)
        return []

    # Build MCQ: correct + 3 distractors, shuffled
    options = [(correct_text, True)] + [(d, False) for d in distractors]
    rng.shuffle(options)
    correct_idx = next(i for i, (_, is_correct) in enumerate(options) if is_correct)
    correct_letter = chr(ord("A") + correct_idx)
    option_texts = [text for text, _ in options]

    # Concatenate context clips into single video; fall back to multi-video on failure
    context_paths = [item["clip_path"] for item in context_items]
    ids_tag = "_".join(cid.replace(" ", "").replace(".", "_") for cid in context_ids)
    concat_dir = os.path.join(clip_dir, "concat")
    concat_path = os.path.join(concat_dir, f"{clip_key}_pn_{ids_tag}.mp4")

    ok = _concat_video_files(context_paths, concat_path)
    if ok:
        videos = [concat_path]
        # Single video: prompt shows num_ctx steps in one video
        prompt = get_add_prompt_generic(len(context_items), option_texts, cot=cot)
    else:
        # Fallback: multi-video (original behavior)
        log.warning("predict_next concat failed for %s, using multi-video fallback", clip_key)
        videos = context_paths
        prompt = get_add_prompt_generic(len(context_items), option_texts, cot=cot)

    return [{
        "messages": [{"role": "user", "content": prompt}],
        "prompt": prompt,
        "answer": correct_letter,
        "videos": videos,
        "data_type": "video",
        "problem_type": "event_logic_predict_next",
        "metadata": {
            "clip_key": ann.get("clip_key", ""),
            "granularity": granularity,
            "context_ids": context_ids,
            "correct_next_id": correct_next_id,
            "correct_text": correct_text,
            "distractors": distractors,
            "domain_l1": ann.get("domain_l1", "other"),
            "domain_l2": ann.get("domain_l2", "other"),
            "source": "vlm_task_architect",
        },
    }]


def _assemble_fill_blank(
    parsed: dict,
    ann: dict,
    clip_dir: str,
    complete_only: bool,
    rng: random.Random,
    cot: bool = False,
) -> list[dict]:
    """Assemble Fill-in-the-Blank MCQ record from validated LLM output."""
    before_ids = parsed.get("before_ids", [])
    missing_id = parsed.get("missing_id", "")
    after_ids = parsed.get("after_ids", [])
    correct_text = parsed.get("correct_text", "")
    distractors = parsed.get("distractors", [])
    granularity = parsed.get("granularity", "")

    clip_key = ann.get("clip_key", "?")

    if not before_ids or not missing_id or not after_ids or not correct_text or len(distractors) < 3:
        log.warning("[fill_blank] %s: REJECT basic validation failed (before=%d, miss=%r, after=%d, dist=%d)",
                    clip_key, len(before_ids), bool(missing_id), len(after_ids), len(distractors))
        return []

    # Trim to first 3 distractors if LLM returned more
    distractors = distractors[:3]

    # Granularity consistency
    all_ids = list(before_ids) + [missing_id] + list(after_ids)
    if not _check_granularity(all_ids):
        log.warning("[fill_blank] %s: granularity mismatch in %s", clip_key, all_ids)
        return []

    # Contiguity check removed: non-action shots between clips can create
    # legitimate gaps in ID sequence (e.g. Action 8.2 -> Action 9.1)
    before_items = [resolve_item(bid, ann, clip_dir) for bid in before_ids]
    after_items = [resolve_item(aid, ann, clip_dir) for aid in after_ids]
    missing_item = resolve_item(missing_id, ann, clip_dir)

    if any(b is None for b in before_items) or any(a is None for a in after_items):
        log.warning("[fill_blank] %s: resolve_item returned None for before/after IDs", clip_key)
        return []
    if missing_item is None:
        log.warning("[fill_blank] %s: resolve_item returned None for missing_id %s", clip_key, missing_id)
        return []

    if complete_only:
        all_items = before_items + after_items
        missing_paths = [item["clip_path"] for item in all_items if not os.path.exists(item["clip_path"])]  # type: ignore[union-attr]
        if missing_paths:
            log.warning("[fill_blank] %s: %d clips missing on disk, first: %s",
                        clip_key, len(missing_paths), missing_paths[0])
            return []

    # Build MCQ
    options = [(correct_text, True)] + [(d, False) for d in distractors]
    rng.shuffle(options)
    correct_idx = next(i for i, (_, is_correct) in enumerate(options) if is_correct)
    correct_letter = chr(ord("A") + correct_idx)
    option_texts = [t for t, _ in options]

    total_steps = len(before_ids) + 1 + len(after_ids)
    missing_pos = len(before_ids)  # 0-indexed

    prompt = get_replace_prompt_generic(total_steps, missing_pos, option_texts, cot=cot)

    # Concatenate: before clips → inline black frame → after clips → single video
    before_paths = [item["clip_path"] for item in before_items]  # type: ignore[union-attr]
    after_paths = [item["clip_path"] for item in after_items]  # type: ignore[union-attr]

    # Output path: {clip_dir}/concat/{clip_key}_fill_{missing_id_safe}.mp4
    missing_id_safe = missing_id.replace(" ", "_").replace(".", "_")
    concat_dir = os.path.join(clip_dir, "concat")
    concat_path = os.path.join(concat_dir, f"{clip_key}_{missing_id_safe}.mp4")

    ok = _concat_clips_with_black(before_paths, after_paths, concat_path)
    if ok:
        videos = [concat_path]
    else:
        # Fallback: multi-video without black frame
        log.warning("fill_blank concat failed for %s, using multi-video fallback", clip_key)
        videos = before_paths + after_paths
        # Rebuild prompt with explicit step-list format for multi-video
        labels = [chr(ord("A") + i) for i in range(len(option_texts))]
        step_lines = []
        step_lines.append("Watch the following process carefully. The sequence has a [MISSING] step.")
        step_lines.append("Context Sequence:")
        video_idx = 0
        for i in range(total_steps):
            if i == missing_pos:
                step_lines.append(f"Step {i + 1}: [MISSING]")
            else:
                step_lines.append(f"Step {i + 1}: <video>")
                video_idx += 1
        step_lines += ["", "Based on the chronological visual content, "
                       "pick the correct textual option to fill in the [MISSING] step.", "Options:"]
        for label, opt in zip(labels, option_texts):
            step_lines.append(f"{label}. {opt}")
        step_lines.append("")
        if cot:
            step_lines.append(f"Think step by step inside <think> </think> tags, then provide your "
                              f"final answer (a single letter from {', '.join(labels)}) inside <answer> </answer> tags.")
        else:
            step_lines.append(f"Provide your answer (a single letter from {', '.join(labels)}) inside <answer> </answer> tags.")
        prompt = "\n".join(step_lines)

    return [{
        "messages": [{"role": "user", "content": prompt}],
        "prompt": prompt,
        "answer": correct_letter,
        "videos": videos,
        "data_type": "video",
        "problem_type": "event_logic_fill_blank",
        "metadata": {
            "clip_key": ann.get("clip_key", ""),
            "granularity": granularity,
            "before_ids": before_ids,
            "missing_id": missing_id,
            "after_ids": after_ids,
            "correct_text": correct_text,
            "distractors": distractors,
            "domain_l1": ann.get("domain_l1", "other"),
            "domain_l2": ann.get("domain_l2", "other"),
            "source": "vlm_task_architect",
        },
    }]


def _assemble_sort(
    parsed: dict,
    ann: dict,
    clip_dir: str,
    complete_only: bool,
    rng: random.Random,
    cot: bool = False,
) -> list[dict]:
    """Assemble VLM-curated Sort record from validated LLM output."""
    ordered_ids = parsed.get("ordered_ids", [])
    granularity = parsed.get("granularity", "")

    if len(ordered_ids) < 3 or len(ordered_ids) > 9:
        return []

    if not _check_granularity(ordered_ids):
        return []

    # Resolve all items
    items = []
    for oid in ordered_ids:
        item = resolve_item(oid, ann, clip_dir)
        if item is None:
            return []
        if complete_only and not os.path.exists(item["clip_path"]):
            return []
        items.append(item)

    # Shuffle (retry to avoid identity permutation)
    n = len(items)
    indices = list(range(n))
    shuf_idx = indices[:]
    for _ in range(20):
        rng.shuffle(shuf_idx)
        if shuf_idx != indices:
            break
    if shuf_idx == indices:
        return []

    # Inverse permutation: answer[i] = 1-based clip at chronological position i
    inverse = [0] * n
    for clip_idx, orig_idx in enumerate(shuf_idx):
        inverse[orig_idx] = clip_idx + 1
    answer = "".join(str(x) for x in inverse)

    shuffled_items = [items[i] for i in shuf_idx]
    prompt = get_sort_prompt_generic(n, cot=cot)
    videos = [item["clip_path"] for item in shuffled_items]

    return [{
        "messages": [{"role": "user", "content": prompt}],
        "prompt": prompt,
        "answer": answer,
        "videos": videos,
        "data_type": "video",
        "problem_type": "event_logic_sort",
        "metadata": {
            "clip_key": ann.get("clip_key", ""),
            "granularity": granularity,
            "ordered_ids": ordered_ids,
            "shuffled_indices": shuf_idx,
            "instructions": [item["text"] for item in shuffled_items],
            "domain_l1": ann.get("domain_l1", "other"),
            "domain_l2": ann.get("domain_l2", "other"),
            "source": "vlm_task_architect",
        },
    }]


# =====================================================================
# Per-annotation processing
# =====================================================================

_TASK_PROMPT_BUILDERS = {
    "predict_next": get_predict_next_user_prompt,
    "fill_blank": get_fill_blank_user_prompt,
    "sort": get_sequence_sort_user_prompt,
}

_TASK_ASSEMBLERS = {
    "predict_next": _assemble_predict_next,
    "fill_blank": _assemble_fill_blank,
    "sort": _assemble_sort,
}


def process_one_annotation(
    ann: dict,
    clip_dir: str,
    api_base: str,
    api_key: str,
    model: str,
    tasks: set[str],
    complete_only: bool,
    temperature: float,
    cache_dir: str,
    rng_seed: int,
    collect_only: bool = False,
    cot: bool = False,
) -> dict:
    """Process one annotation: build script, call LLM for each task, assemble records.

    When collect_only=True: skips clip existence checks and returns needed clip paths
    instead of assembled records (keys: clip_key, status, needed_clips, errors).

    Otherwise returns dict with keys: clip_key, status, predict_next, fill_blank, sort, errors.
    """
    clip_key = ann.get("clip_key", "")
    rng = random.Random(rng_seed)
    result: dict = {
        "clip_key": clip_key,
        "status": "ok",
        "predict_next": [],
        "fill_blank": [],
        "sort": [],
        "needed_clips": [],
        "errors": [],
    }

    # 1. Build script text
    script_text = build_script_text(ann)
    if script_text is None:
        l1 = ann.get("level1")
        l2 = ann.get("level2")
        log.warning("[skip] %s: build_script_text=None (l1=%s, l2=%s, l2_parse_error=%s)",
                    clip_key, bool(l1), bool(l2),
                    l2.get("_parse_error") if isinstance(l2, dict) else "N/A")
        result["status"] = "skip"
        return result

    # 2. Call LLM for each enabled task
    for task_name in tasks:
        prompt_builder = _TASK_PROMPT_BUILDERS[task_name]

        # Check cache
        cached = _load_cache(cache_dir, clip_key, task_name)
        if cached is not None:
            parsed = cached
        else:
            user_prompt = prompt_builder(script_text)
            try:
                raw = call_task_architect(
                    api_base, api_key, model, user_prompt,
                    temperature=temperature,
                    label=f"{clip_key}/{task_name}",
                )
            except Exception as e:
                result["errors"].append(f"{task_name}: LLM call failed: {e}")
                continue

            parsed = parse_json_from_response(raw)
            if parsed is None:
                result["errors"].append(f"{task_name}: JSON parse failed")
                continue

            # Save cache
            _save_cache(cache_dir, clip_key, task_name, parsed)

        if not parsed.get("suitable", False):
            log.info("[%s] %s/%s: suitable=false (from %s), skipping",
                     "cache" if cached is not None else "llm", clip_key, task_name,
                     parsed.get("reasoning", "no reasoning")[:120])
            continue

        if collect_only:
            # Collect clip paths without existence check
            collector = _TASK_CLIP_COLLECTORS[task_name]
            result["needed_clips"].extend(collector(parsed, ann, clip_dir))
        else:
            assembler = _TASK_ASSEMBLERS[task_name]
            records = assembler(parsed, ann, clip_dir, complete_only, rng, cot=cot)
            result[task_name].extend(records)

    if not collect_only:
        has_records = any(result[t] for t in ALL_TASKS if t in tasks)
        if result["errors"] and not has_records:
            result["status"] = "error"

    return result


# =====================================================================
# Cache (optional disk-based resume)
# =====================================================================

def _cache_path(cache_dir: str, clip_key: str, task: str) -> str:
    return os.path.join(cache_dir, f"{clip_key}_{task}.json")


def _load_cache(cache_dir: str, clip_key: str, task: str) -> dict | None:
    if not cache_dir:
        return None
    p = _cache_path(cache_dir, clip_key, task)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _save_cache(cache_dir: str, clip_key: str, task: str, data: dict) -> None:
    if not cache_dir:
        return
    os.makedirs(cache_dir, exist_ok=True)
    p = _cache_path(cache_dir, clip_key, task)
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except OSError:
        pass


# =====================================================================
# Balanced sampling (from build_event_shuffle.py pattern)
# =====================================================================

def _balanced_sample_by_domain(
    records: list[dict],
    budget: int,
    rng: random.Random,
) -> list[dict]:
    """Two-tier balanced sampling by domain_l1 -> domain_l2."""
    if len(records) <= budget:
        rng.shuffle(records)
        return records

    by_l1: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        d = rec.get("metadata", {}).get("domain_l1", "other")
        by_l1[d].append(rec)

    n_l1 = len(by_l1)
    base_per_l1 = budget // max(n_l1, 1)

    sampled: list[dict] = []
    shortfall = 0
    overflows: list[list[dict]] = []

    for d in sorted(by_l1.keys()):
        pool = by_l1[d]
        rng.shuffle(pool)
        if len(pool) <= base_per_l1:
            sampled.extend(pool)
            shortfall += base_per_l1 - len(pool)
        else:
            sampled.extend(pool[:base_per_l1])
            overflows.append(pool[base_per_l1:])

    # Redistribute shortfall
    extra = shortfall
    if extra > 0 and overflows:
        per_overflow = extra // len(overflows)
        for leftovers in overflows:
            take = min(per_overflow, len(leftovers))
            sampled.extend(leftovers[:take])

    rng.shuffle(sampled)
    return sampled[:budget]


# =====================================================================
# IO
# =====================================================================

def write_jsonl(records: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="VLM-powered Event Logic data construction pipeline"
    )
    parser.add_argument("--annotation-dir", "-a", required=True,
                        help="Annotation JSON directory")
    parser.add_argument("--clip-dir", required=True,
                        help="Atomic clips root (L2/, L3/ subdirs)")
    parser.add_argument("--output-dir", "-o", required=True,
                        help="Output directory (train.jsonl, val.jsonl, stats.json)")

    # LLM config
    parser.add_argument("--api-base", default="https://api.novita.ai/v3/openai")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default="pa/gmn-2.5-fls")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--workers", type=int, default=8,
                        help="Concurrent LLM call threads")

    # Task selection
    parser.add_argument("--tasks", nargs="+", choices=ALL_TASKS, default=list(ALL_TASKS))

    # Filtering
    parser.add_argument("--complete-only", action="store_true",
                        help="Only keep records whose clip files exist on disk")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max annotations to process (0 = all)")

    # Output config
    parser.add_argument("--train-budget", type=int, default=-1,
                        help="Max training samples per task (-1 = unlimited)")
    parser.add_argument("--val-count", type=int, default=100,
                        help="Total validation samples (split across tasks)")

    # Cache / debug
    parser.add_argument("--cache-dir", default="",
                        help="Cache LLM responses for resume")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only build script texts, skip LLM calls")
    parser.add_argument("--collect-clips-only", action="store_true",
                        help="Run LLM design (using cache if available) and write needed "
                             "clip paths to <output-dir>/needed_clips.txt without assembling "
                             "records or checking clip existence. Use this before clip cutting.")
    parser.add_argument("--prompt-style", choices=["direct", "cot"], default="direct",
                        help="Prompt format: 'direct' omits <think> tags (default), "
                             "'cot' includes chain-of-thought reasoning instructions.")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    rng = random.Random(args.seed)
    task_set = set(args.tasks)
    use_cot = (args.prompt_style == "cot")

    # ---- 1. Load annotations ----
    log.info("Loading annotations from: %s", args.annotation_dir)
    annotations = load_annotations(args.annotation_dir, complete_only=False,
                                   limit=args.limit if args.limit > 0 else 0)
    log.info("Loaded %d annotations", len(annotations))

    # ---- Dry run: only show script texts ----
    if args.dry_run:
        ok = 0
        for ann in annotations:
            script = build_script_text(ann)
            if script:
                ok += 1
                if ok <= 3:
                    log.info("=== Script [%s] ===\n%s", ann.get("clip_key", ""), script[:2000])
        log.info("Dry run: %d / %d annotations have valid script texts", ok, len(annotations))
        return

    # ---- Collect-clips-only: write needed clip paths, skip assembly ----
    if args.collect_clips_only:
        log.info("=== COLLECT-CLIPS-ONLY mode: gathering needed clip paths ===")
        all_clip_paths: set[str] = set()

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    process_one_annotation,
                    ann, args.clip_dir,
                    args.api_base, args.api_key, args.model,
                    task_set, False, args.temperature,
                    args.cache_dir,
                    rng.randint(0, 2**31),
                    True,  # collect_only
                ): ann.get("clip_key", "")
                for ann in annotations
            }

            done_count = 0
            for fut in as_completed(futures):
                done_count += 1
                try:
                    result = fut.result()
                    all_clip_paths.update(result.get("needed_clips", []))
                    if result.get("errors"):
                        clip_key = futures[fut]
                        log.warning("Errors for %s: %s", clip_key, result["errors"])
                except Exception as e:
                    clip_key = futures[fut]
                    log.error("Error processing %s: %s", clip_key, e)

                if done_count % 100 == 0 or done_count == len(futures):
                    log.info("Progress: %d / %d  (collected %d clip paths so far)",
                             done_count, len(futures), len(all_clip_paths))

        # Diagnostic summary for collect-clips-only mode
        _status_counts: dict[str, int] = defaultdict(int)
        for fut in futures:
            try:
                r = fut.result()
                _status_counts[r.get("status", "unknown")] += 1
            except Exception:
                _status_counts["exception"] += 1
        log.info("Collect-clips summary: %s  |  total unique clips: %d",
                 dict(_status_counts), len(all_clip_paths))

        os.makedirs(args.output_dir, exist_ok=True)
        clips_list_path = os.path.join(args.output_dir, "needed_clips.txt")
        with open(clips_list_path, "w", encoding="utf-8") as f:
            for p in sorted(all_clip_paths):
                f.write(p + "\n")
        log.info("Written %d unique clip paths to: %s", len(all_clip_paths), clips_list_path)
        log.info("Next step: prepare_all_clips.py --clip-list %s", clips_list_path)
        return

    # ---- 2. Process annotations in parallel ----
    all_results: list[dict] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                process_one_annotation,
                ann, args.clip_dir,
                args.api_base, args.api_key, args.model,
                task_set, args.complete_only, args.temperature,
                args.cache_dir,
                rng.randint(0, 2**31),
                False,   # collect_only
                use_cot,
            ): ann.get("clip_key", "")
            for ann in annotations
        }

        done_count = 0
        for fut in as_completed(futures):
            done_count += 1
            try:
                result = fut.result()
            except Exception as e:
                clip_key = futures[fut]
                result = {
                    "clip_key": clip_key, "status": "error",
                    "errors": [str(e)],
                    "predict_next": [], "fill_blank": [], "sort": [],
                }
            all_results.append(result)

            if done_count % 100 == 0 or done_count == len(futures):
                log.info("Progress: %d / %d", done_count, len(futures))

    # ---- 3. Aggregate records by task type ----
    task_records: dict[str, list[dict]] = {t: [] for t in args.tasks}
    for result in all_results:
        for task in args.tasks:
            task_records[task].extend(result.get(task, []))

    for task in args.tasks:
        log.info("  %s: %d records", task, len(task_records[task]))

    # ---- 4. Train/val split + balanced sampling per task ----
    all_train: list[dict] = []
    all_val: list[dict] = []
    n_tasks = len(args.tasks)
    val_per_task = max(1, args.val_count // max(n_tasks, 1))

    for task in args.tasks:
        records = task_records[task]
        rng.shuffle(records)
        n_val = min(val_per_task, max(1, len(records) // 5))
        val = records[:n_val]
        train_pool = records[n_val:]

        if args.train_budget > 0 and len(train_pool) > args.train_budget:
            train = _balanced_sample_by_domain(train_pool, args.train_budget, rng)
        else:
            train = train_pool

        all_train.extend(train)
        all_val.extend(val)
        log.info("  %s: %d train + %d val", task, len(train), len(val))

    rng.shuffle(all_train)
    rng.shuffle(all_val)

    # ---- 5. Write output ----
    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.jsonl")
    val_path = os.path.join(args.output_dir, "val.jsonl")
    write_jsonl(all_train, train_path)
    write_jsonl(all_val, val_path)

    # Per-task files
    for task in args.tasks:
        task_train = [r for r in all_train if r.get("problem_type", "").endswith(task)]
        task_val = [r for r in all_val if r.get("problem_type", "").endswith(task)]
        write_jsonl(task_train, os.path.join(args.output_dir, f"train_{task}.jsonl"))
        write_jsonl(task_val, os.path.join(args.output_dir, f"val_{task}.jsonl"))
        log.info("  %s: %d train + %d val → train_%s.jsonl / val_%s.jsonl",
                 task, len(task_train), len(task_val), task, task)

    # Stats
    stats = {
        "total_annotations": len(annotations),
        "processed_ok": sum(1 for r in all_results if r["status"] == "ok"),
        "skipped": sum(1 for r in all_results if r["status"] == "skip"),
        "errored": sum(1 for r in all_results if r["status"] == "error"),
        "records_by_task": {t: len(task_records[t]) for t in args.tasks},
        "train_total": len(all_train),
        "val_total": len(all_val),
        "train_by_type": dict(Counter(r["problem_type"] for r in all_train)),
        "val_by_type": dict(Counter(r["problem_type"] for r in all_val)),
        "train_by_domain_l1": dict(Counter(
            r.get("metadata", {}).get("domain_l1", "other") for r in all_train
        )),
        "token_usage": dict(_token_usage),
    }
    stats_path = os.path.join(args.output_dir, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # Summary
    log.info("=== Output ===")
    log.info("  Train: %d  ->  %s", len(all_train), train_path)
    log.info("  Val:   %d  ->  %s", len(all_val), val_path)
    log.info("  Stats: %s", stats_path)
    log.info("  Token usage: %s", _token_usage)

    if all_train:
        ex = all_train[0]
        log.info("=== Example record ===")
        log.info("  problem_type: %s", ex["problem_type"])
        log.info("  answer: %s", ex["answer"])
        log.info("  videos (%d): %s", len(ex["videos"]), ex["videos"][0] if ex["videos"] else "N/A")
        log.info("  prompt (first 200):\n  %s", ex["prompt"][:200])


if __name__ == "__main__":
    main()
