#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check and refine caption pairs using a VLM (e.g. Gemini).

For each caption pair, sends forward/reverse/shuffle video frames together with
the existing captions to a VLM.  The VLM judges whether the forward and reverse
captions are textually distinguishable and, if not, rewrites them with clear
start→end state transitions and balanced word count.

Input:
  - caption_pairs.jsonl   (from annotate_event_captions.py)
  - manifest JSONL        (for video paths)

Output:
  - refined_caption_pairs.jsonl   (drop-in replacement for caption_pairs.jsonl)

Example:
python proxy_data/temporal_aot/check_and_refine_captions.py \\
  --caption-pairs /tmp/aot_annotations/caption_pairs.jsonl \\
  --manifest-jsonl /tmp/aot_event_manifest.jsonl \\
  --output /tmp/aot_annotations/refined_caption_pairs.jsonl \\
  --api-base https://generativelanguage.googleapis.com/v1beta/openai \\
  --model gemini-2.5-flash \\
  --workers 4
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm.auto import tqdm

from prompts import SYSTEM_PROMPT, get_check_and_refine_prompt


# ---------------------------------------------------------------------------
# Frame sampling (reuse logic from annotate_event_captions.py)
# ---------------------------------------------------------------------------


def sample_video_frames_by_fps(
    video_path: str,
    target_fps: float,
    max_frames: int,
) -> list[Image.Image]:
    try:
        import decord
    except ImportError as exc:
        raise ImportError("decord is required: pip install decord") from exc

    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    total = len(vr)
    if total == 0:
        return []

    video_fps = vr.get_avg_fps() or 25.0
    duration = total / video_fps
    n_frames = min(max(1, round(duration * target_fps)), max_frames)
    if n_frames >= total:
        indices = list(range(total))
    elif n_frames == 1:
        indices = [total // 2]
    else:
        stride = (total - 1) / (n_frames - 1)
        indices = [round(i * stride) for i in range(n_frames)]
    frames = vr.get_batch(indices).asnumpy()
    return [Image.fromarray(frame).convert("RGB") for frame in frames]


def sample_video_frames_segment_aware(
    video_path: str,
    target_fps: float,
    max_frames: int,
    n_segments: int,
    segment_sec: float,
) -> list[Image.Image]:
    try:
        import decord
    except ImportError as exc:
        raise ImportError("decord is required: pip install decord") from exc

    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    total = len(vr)
    if total == 0:
        return []

    video_fps = vr.get_avg_fps() or 25.0
    frames_per_seg = max(1, round(segment_sec * target_fps))
    total_duration = total / video_fps

    indices: list[int] = []
    for seg_idx in range(n_segments):
        seg_start = seg_idx * segment_sec
        seg_end = min((seg_idx + 1) * segment_sec, total_duration)
        frame_start = int(seg_start * video_fps)
        frame_end = min(int(seg_end * video_fps), total - 1)
        if frame_end <= frame_start:
            indices.append(min(frame_start, total - 1))
            continue
        if frames_per_seg == 1:
            indices.append((frame_start + frame_end) // 2)
        else:
            stride = (frame_end - frame_start) / (frames_per_seg - 1)
            for j in range(frames_per_seg):
                idx = min(round(frame_start + j * stride), total - 1)
                indices.append(idx)

    seen: set[int] = set()
    unique_indices: list[int] = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            unique_indices.append(idx)
    unique_indices = unique_indices[:max_frames]

    frames = vr.get_batch(unique_indices).asnumpy()
    return [Image.fromarray(frame).convert("RGB") for frame in frames]


# ---------------------------------------------------------------------------
# Image encoding & VLM call
# ---------------------------------------------------------------------------


def image_to_data_url(img: Image.Image, jpeg_quality: int, resize_max_width: int) -> str:
    if resize_max_width > 0 and img.width > resize_max_width:
        new_height = max(1, round(img.height * resize_max_width / img.width))
        img = img.resize((resize_max_width, new_height), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    raw = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{raw}"


def call_vlm(
    api_base: str,
    api_key: str,
    model: str,
    system_prompt: str,
    content: list[dict[str, Any]],
    retries: int,
) -> dict[str, Any]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("openai is required: pip install openai") from exc

    client = OpenAI(
        api_key=api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY") or "",
        base_url=api_base,
    )

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
                temperature=0.0,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content
            return json.loads(raw)
        except Exception as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"VLM call failed after {retries} attempts: {last_error}")


# ---------------------------------------------------------------------------
# Build multi-video content block
# ---------------------------------------------------------------------------


def _build_frame_section(
    label: str,
    frames: list[Image.Image],
    jpeg_quality: int,
    resize_max_width: int,
) -> list[dict[str, Any]]:
    """Build a labeled section of frames for the VLM content array."""
    section: list[dict[str, Any]] = [
        {"type": "text", "text": f"\n=== {label} ===\n"},
    ]
    for idx, frame in enumerate(frames, 1):
        section.append({"type": "text", "text": f"[Frame {idx}]"})
        section.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_to_data_url(frame, jpeg_quality, resize_max_width),
                    "detail": "low",
                },
            }
        )
    return section


# ---------------------------------------------------------------------------
# Per-record check & refine
# ---------------------------------------------------------------------------


def check_one(
    pair: dict[str, Any],
    manifest_record: dict[str, Any] | None,
    api_base: str,
    api_key: str,
    model: str,
    max_frames: int,
    fps: float,
    shuffle_fps: float,
    resize_max_width: int,
    jpeg_quality: int,
    retries: int,
) -> dict[str, Any]:
    """Check and optionally refine one caption pair."""
    forward_path = pair["forward_video_path"]
    reverse_path = pair.get("reverse_video_path") or ""
    shuffle_path = pair.get("shuffle_video_path") or ""

    forward_caption = pair["forward_caption"]
    reverse_caption = pair["reverse_caption"]
    shuffle_caption = pair.get("shuffle_caption", "") or ""

    # --- Sample frames from each video ---
    forward_frames = sample_video_frames_by_fps(forward_path, target_fps=fps, max_frames=max_frames)

    if reverse_path and os.path.isfile(reverse_path):
        reverse_frames = sample_video_frames_by_fps(reverse_path, target_fps=fps, max_frames=max_frames)
    else:
        reverse_frames = []

    shuffle_frames: list[Image.Image] = []
    if shuffle_path and os.path.isfile(shuffle_path):
        if manifest_record:
            actual_dur = manifest_record.get("actual_duration_sec") or manifest_record.get("duration_sec") or 0.0
            n_segments = int(manifest_record.get("shuffle_n_segments") or 0)
            if n_segments <= 0 and actual_dur > 0:
                _old_seg_sec = manifest_record.get("shuffle_segment_sec", 2.0) or 2.0
                n_segments = int(actual_dur // _old_seg_sec)
            segment_dur = (actual_dur / n_segments) if n_segments > 0 else 2.0
            if n_segments >= 2:
                shuffle_frames = sample_video_frames_segment_aware(
                    shuffle_path,
                    target_fps=shuffle_fps,
                    max_frames=max_frames,
                    n_segments=n_segments,
                    segment_sec=segment_dur,
                )
            else:
                shuffle_frames = sample_video_frames_by_fps(shuffle_path, target_fps=shuffle_fps, max_frames=max_frames)
        else:
            shuffle_frames = sample_video_frames_by_fps(shuffle_path, target_fps=shuffle_fps, max_frames=max_frames)

    # --- Build multi-video content ---
    content: list[dict[str, Any]] = []
    content.extend(_build_frame_section("Forward Video", forward_frames, jpeg_quality, resize_max_width))
    if reverse_frames:
        content.extend(_build_frame_section("Reverse Video", reverse_frames, jpeg_quality, resize_max_width))
    if shuffle_frames and shuffle_caption:
        content.extend(_build_frame_section("Shuffle Video", shuffle_frames, jpeg_quality, resize_max_width))

    # --- Append text prompt ---
    prompt_text = get_check_and_refine_prompt(
        forward_caption=forward_caption,
        reverse_caption=reverse_caption,
        shuffle_caption=shuffle_caption if shuffle_frames else "",
    )
    content.append({"type": "text", "text": prompt_text})

    # --- Call VLM ---
    system_prompt = (
        "You are an expert video caption reviewer. "
        "Your job is to ensure temporal captions accurately and distinctly describe "
        "the observed action sequences in forward, reverse, and shuffled video clips. "
        "Be precise about state changes and their direction."
    )
    resp = call_vlm(api_base, api_key, model, system_prompt, content, retries)

    # --- Build output record ---
    was_refined = not resp.get("distinguishable", True)
    result = dict(pair)  # copy all original fields
    result["original_forward_caption"] = forward_caption
    result["original_reverse_caption"] = reverse_caption
    result["original_shuffle_caption"] = shuffle_caption
    result["was_refined"] = was_refined
    result["check_distinguishable"] = resp.get("distinguishable", True)
    result["check_reason"] = resp.get("reason", "")

    # Update captions (refined or unchanged)
    result["forward_caption"] = resp.get("forward_caption", forward_caption).strip()
    result["reverse_caption"] = resp.get("reverse_caption", reverse_caption).strip()
    if shuffle_caption:
        result["shuffle_caption"] = resp.get("shuffle_caption", shuffle_caption).strip()

    # Recompute is_different
    result["is_different"] = result["forward_caption"] != result["reverse_caption"]

    return result


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_jsonl(path: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_manifest_index(path: str) -> dict[str, dict[str, Any]]:
    """Load manifest and index by clip_key."""
    index: dict[str, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                index[record["clip_key"]] = record
    return index


def load_done_keys(path: str) -> set[str]:
    """Load clip_keys already processed in the output file."""
    keys: set[str] = set()
    if not os.path.isfile(path):
        return keys
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                keys.add(json.loads(line)["clip_key"])
    return keys


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check and refine caption pairs for temporal AoT data."
    )
    parser.add_argument("--caption-pairs", required=True, help="Input caption_pairs.jsonl")
    parser.add_argument("--manifest-jsonl", required=True, help="Manifest JSONL (for video paths & shuffle metadata)")
    parser.add_argument("--output", required=True, help="Output refined_caption_pairs.jsonl")
    parser.add_argument("--api-base", required=True, help="OpenAI-compatible API base URL")
    parser.add_argument("--model", required=True, help="Model name (e.g. gemini-2.5-flash)")
    parser.add_argument("--api-key", default="", help="API key (defaults to OPENAI_API_KEY or GEMINI_API_KEY)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--max-frames", type=int, default=32, help="Max frames per video section")
    parser.add_argument("--fps", type=float, default=1.0, help="Frame rate for forward/reverse sampling")
    parser.add_argument("--shuffle-fps", type=float, default=2.0, help="Frame rate for shuffle sampling")
    parser.add_argument("--resize-max-width", type=int, default=768, help="Resize frame max width before upload")
    parser.add_argument("--jpeg-quality", type=int, default=60, help="JPEG quality for upload")
    parser.add_argument("--retries", type=int, default=3, help="API retry count per sample")
    parser.add_argument("--max-samples", type=int, default=0, help="Max pairs to process (0 = all)")
    parser.add_argument(
        "--resume", action="store_true", default=True,
        help="Skip pairs already in output file and append new results.",
    )
    parser.add_argument(
        "--only-check", action="store_true", default=False,
        help="Only run check (no refinement). Useful for statistics.",
    )
    args = parser.parse_args()

    pairs = load_jsonl(args.caption_pairs)
    manifest_index = load_manifest_index(args.manifest_jsonl)
    print(f"Loaded {len(pairs)} caption pairs, {len(manifest_index)} manifest records.")

    if args.max_samples > 0:
        pairs = pairs[:args.max_samples]

    # Resume support
    done_keys: set[str] = set()
    if args.resume:
        done_keys = load_done_keys(args.output)
        if done_keys:
            skipped = sum(1 for p in pairs if p["clip_key"] in done_keys)
            print(f"Resuming: skipping {skipped}/{len(pairs)} already-processed pairs.")
            pairs = [p for p in pairs if p["clip_key"] not in done_keys]

    if not pairs:
        print("No pairs to process.")
        return

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    file_mode = "a" if args.resume and done_keys else "w"

    refined_count = 0
    unchanged_count = 0
    error_count = 0

    with open(args.output, file_mode, encoding="utf-8", buffering=1) as fout:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    check_one,
                    pair,
                    manifest_index.get(pair["clip_key"]),
                    args.api_base,
                    args.api_key,
                    args.model,
                    args.max_frames,
                    args.fps,
                    args.shuffle_fps,
                    args.resize_max_width,
                    args.jpeg_quality,
                    args.retries,
                ): pair["clip_key"]
                for pair in pairs
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Checking captions",
                unit="pair",
            ):
                clip_key = futures[future]
                try:
                    result = future.result()
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    if result.get("was_refined", False):
                        refined_count += 1
                    else:
                        unchanged_count += 1
                except Exception as exc:
                    error_count += 1
                    print(f"[ERROR] {clip_key}: {exc}")

    total = refined_count + unchanged_count + error_count
    print(
        f"\nDone. total={total} refined={refined_count} unchanged={unchanged_count} "
        f"errors={error_count}"
    )
    print(f"Output: {args.output}")
    if refined_count > 0:
        print(
            f"  {refined_count}/{refined_count + unchanged_count} pairs were refined "
            f"({100 * refined_count / max(1, refined_count + unchanged_count):.1f}%)"
        )


if __name__ == "__main__":
    main()
