#!/usr/bin/env python3
"""
annotate_gseg.py — Grounding+Segmentation annotation pipeline.

Sends video frames to a VLM and asks it to generate:
  1. An abstract reasoning query (for the student model)
  2. Ground-truth segmentation (for reward computation)

Supports multi-task output: one video can produce multiple (query, segments)
pairs when the video contains distinct activity threads.

Usage:
    python annotate_gseg.py \
        --frames-dir /path/to/frames \
        --output-dir /path/to/annotations \
        --api-base https://api.novita.ai/v3/openai \
        --model pa/gmn-2.5-fl \
        --workers 4

Input:  frames/{clip_key}/*.jpg  + optional meta.json
Output: annotations/{clip_key}.json
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Reuse VLM infrastructure from the existing hier_seg_annotation pipeline
_HIER_SEG_DIR = os.path.join(os.path.dirname(__file__), "..", "hier_seg_annotation")
sys.path.insert(0, _HIER_SEG_DIR)

from annotate import (
    call_vlm,
    parse_json_from_response,
    get_all_frame_files,
    sample_uniform,
    encode_frame_files,
    format_mmss,
    frame_stem_to_index,
    frame_index_to_sec,
    load_frame_meta,
    load_records_from_frames_dir,
)

from prompts_gseg import SYSTEM_PROMPT, get_annotation_prompt


# ─────────────────────────────────────────────────────────────────────────────
# JSON parsing (handles both array and single-object VLM responses)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_tasks_from_response(raw: str) -> list[dict] | None:
    """Parse VLM response into a list of task dicts.

    Accepts JSON array ``[{task}, ...]`` or a single JSON object ``{task}``.
    Returns None on parse failure.
    """
    text = raw.strip()

    # Try direct parse
    for candidate in _extract_json_candidates(text):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, list):
            return [t for t in obj if isinstance(t, dict)]
        if isinstance(obj, dict):
            return [obj]

    return None


def _extract_json_candidates(text: str) -> list[str]:
    """Yield candidate JSON strings from raw VLM output."""
    yield text
    # From ```json ... ``` blocks
    for m in re.finditer(r"```(?:json)?\s*([\[\{][\s\S]+?[\]\}])\s*```", text):
        yield m.group(1)
    # Bare array or object
    m = re.search(r"(\[[\s\S]+\])", text)
    if m:
        yield m.group(1)
    m = re.search(r"(\{[\s\S]+\})", text)
    if m:
        yield m.group(1)


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
    max_frames: int,
    resize_max_width: int,
    jpeg_quality: int,
    overwrite: bool,
) -> str:
    """Annotate a single clip. Returns status string."""
    meta = record.get("metadata", {})
    clip_key = meta.get("clip_key", "unknown")
    out_path = output_dir / f"{clip_key}.json"

    # Skip if already annotated
    if out_path.exists() and not overwrite:
        try:
            with open(out_path, encoding="utf-8") as f:
                existing = json.load(f)
            if existing.get("tasks") and len(existing["tasks"]) > 0:
                return f"skip  {clip_key}"
        except Exception:
            pass

    # Load frames
    frame_dir = frames_base / clip_key
    if not frame_dir.is_dir():
        return f"error {clip_key}: frame dir not found"

    frame_files = get_all_frame_files(frame_dir)
    if not frame_files:
        return f"error {clip_key}: no frames"

    # Sample frames
    sampled = sample_uniform(frame_files, max_frames)
    n_frames = len(sampled)

    # Determine duration
    duration = meta.get("clip_duration") or meta.get("clip_end")
    if not duration:
        last_idx = frame_stem_to_index(sampled[-1], n_frames)
        duration = int(frame_index_to_sec(last_idx, 1.0)) + 1
    duration = int(duration)

    # Encode frames + build labels
    frame_b64 = encode_frame_files(sampled, resize_max_width, jpeg_quality)
    frame_labels = []
    for fp in sampled:
        idx = frame_stem_to_index(fp, 0)
        sec = frame_index_to_sec(idx, 1.0)
        frame_labels.append(f"[Timestamp {format_mmss(sec)} | Frame {idx}]")

    # Build prompt
    prompt_text = get_annotation_prompt(n_frames, duration)

    # Call VLM (retry once on parse failure)
    print(f"  annotating {clip_key} ({n_frames} frames, {duration}s) …", flush=True)
    tasks = None
    for attempt in range(2):
        raw = call_vlm(
            api_base, api_key, model,
            SYSTEM_PROMPT, prompt_text,
            frame_b64, frame_labels,
        )
        tasks = _parse_tasks_from_response(raw)
        if tasks:
            break

    if not tasks:
        return f"error {clip_key}: VLM parse failure"

    # Validate each task
    valid_tasks = []
    for t in tasks:
        if not t.get("query") or not t.get("segments"):
            continue
        validated = {
            "query_style": t.get("query_style", ""),
            "query": t["query"],
            "video_summary": t.get("video_summary", ""),
            "domain": t.get("domain", ""),
            "noise_description": t.get("noise_description"),
            "grounding": _validate_grounding(t.get("grounding", {}), duration),
            "segments": _validate_segments(t.get("segments", []), duration),
            "reasoning_trace": t.get("reasoning_trace", ""),
        }
        if validated["segments"]:
            valid_tasks.append(validated)

    if not valid_tasks:
        return f"error {clip_key}: no valid tasks after validation"

    # Build output annotation
    annotation = {
        "clip_key": clip_key,
        "source_video_path": record.get("videos", [""])[0],
        "clip_duration_sec": duration,
        "n_frames": n_frames,
        "frame_dir": str(frame_dir),
        "n_tasks": len(valid_tasks),
        "tasks": valid_tasks,
        "annotated_at": datetime.now(timezone.utc).isoformat(),
    }

    # Write
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(annotation, f, indent=2, ensure_ascii=False)

    total_seg = sum(len(t["segments"]) for t in valid_tasks)
    styles = ",".join(t["query_style"] for t in valid_tasks)
    return f"ok    {clip_key}: {len(valid_tasks)} tasks, {total_seg} segments, style={styles}"


def _validate_grounding(g: dict, duration: int) -> dict:
    """Clamp grounding times to valid range."""
    return {
        "start_time": max(0, min(int(g.get("start_time", 0)), duration)),
        "end_time": max(0, min(int(g.get("end_time", duration)), duration)),
        "rationale": g.get("rationale", ""),
    }


def _validate_segments(segments: list, duration: int) -> list:
    """Validate and clamp segment timestamps."""
    result = []
    for seg in segments:
        s = max(0, min(int(seg.get("start_time", 0)), duration))
        e = max(0, min(int(seg.get("end_time", 0)), duration))
        if e <= s:
            continue
        result.append({
            "id": seg.get("id", len(result) + 1),
            "start_time": s,
            "end_time": e,
            "label": seg.get("label", ""),
        })
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Grounding+Segmentation annotation: VLM generates abstract query + GT.",
    )
    parser.add_argument("--frames-dir", type=Path, required=True,
                        help="Root dir of pre-extracted 1fps frames (subdir per clip)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for annotation JSONs")
    parser.add_argument("--api-base", default="https://api.novita.ai/v3/openai")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default="pa/gmn-2.5-fl")
    parser.add_argument("--max-frames-per-call", type=int, default=0,
                        help="Max frames to send per VLM call (0 = no limit)")
    parser.add_argument("--resize-max-width", type=int, default=0)
    parser.add_argument("--jpeg-quality", type=int, default=60)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0,
                        help="Process at most N clips (0 = all)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    # Load records from frames directory
    records = load_records_from_frames_dir(args.frames_dir, args.limit)
    print(f"Found {len(records)} clips in {args.frames_dir}", flush=True)

    if not records:
        print("No clips to process.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    counts = {"ok": 0, "skip": 0, "error": 0}

    def _process(record):
        return annotate_clip(
            record,
            args.frames_dir,
            args.output_dir,
            args.api_base,
            args.api_key,
            args.model,
            args.max_frames_per_call,
            args.resize_max_width,
            args.jpeg_quality,
            args.overwrite,
        )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process, r): r for r in records}
        for fut in as_completed(futures):
            try:
                status = fut.result()
            except Exception as exc:
                clip_key = futures[fut].get("metadata", {}).get("clip_key", "?")
                status = f"error {clip_key}: {exc}"
            tag = status.split()[0]
            counts[tag] = counts.get(tag, 0) + 1
            print(f"  [{counts['ok']+counts['skip']+counts['error']}/{len(records)}] {status}", flush=True)

    print(f"\nDone: {counts['ok']} annotated, {counts['skip']} skipped, {counts['error']} errors")


if __name__ == "__main__":
    main()
