#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Annotate forward/reverse/shuffle captions for event clips listed in a manifest JSONL.

Output files under --output-dir:
- forward_captions.jsonl
- reverse_captions.jsonl
- caption_pairs.jsonl          (always written)
- shuffle_captions.jsonl       (only when manifest contains shuffle_video_path)

Example:
python proxy_data/temporal_aot/annotate_event_captions.py \
  --manifest-jsonl /tmp/aot_event_manifest.jsonl \
  --output-dir /tmp/aot_annotations \
  --api-base http://localhost:8000/v1 \
  --model Qwen3-VL-7B \
  --workers 4 \
  --max-samples 200
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

from prompts import SYSTEM_PROMPT, get_forward_reverse_caption_prompt, get_shuffle_caption_prompt


def sample_video_frames_by_fps(
    video_path: str,
    target_fps: float,
    max_frames: int,
) -> list[Image.Image]:
    """Sample frames at a fixed target FPS (e.g. 1fps for caption, 2fps for training).

    The number of sampled frames = min(ceil(duration * target_fps), max_frames).
    Frames are uniformly distributed across the video.
    """
    try:
        import decord
    except ImportError as exc:
        raise ImportError("decord is required for annotation: pip install decord") from exc

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
    """Sample frames at ``target_fps`` while ensuring each segment gets at least one frame.

    For shuffled videos cut into ``n_segments`` segments of ``segment_sec``
    seconds each, this guarantees visual coverage of every segment so the VLM
    can describe the per-segment content.
    """
    try:
        import decord
    except ImportError as exc:
        raise ImportError("decord is required for annotation: pip install decord") from exc

    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    total = len(vr)
    if total == 0:
        return []

    video_fps = vr.get_avg_fps() or 25.0
    # Frames per segment = max(1, round(segment_sec * target_fps))
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

    # Deduplicate while preserving order, then cap at max_frames
    seen: set[int] = set()
    unique_indices: list[int] = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            unique_indices.append(idx)
    unique_indices = unique_indices[:max_frames]

    frames = vr.get_batch(unique_indices).asnumpy()
    return [Image.fromarray(frame).convert("RGB") for frame in frames]


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
    user_text: str,
    frames: list[Image.Image],
    resize_max_width: int,
    jpeg_quality: int,
    retries: int,
) -> dict[str, Any]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("openai is required: pip install openai") from exc

    client = OpenAI(
        api_key=api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("NOVITA_API_KEY") or "",
        base_url=api_base,
    )

    content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    for idx, frame in enumerate(frames, 1):
        content.append({"type": "text", "text": f"[Frame {idx}]"})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_to_data_url(frame, jpeg_quality, resize_max_width), "detail": "low"},
            }
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
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"caption annotation failed: {last_error}")


def annotate_one(
    record: dict,
    api_base: str,
    api_key: str,
    model: str,
    max_frames: int,
    fwd_rev_fps: float,
    shuffle_fps: float,
    resize_max_width: int,
    jpeg_quality: int,
    retries: int,
) -> tuple[dict, dict, dict, dict | None]:
    """Annotate one manifest record.

    Returns (forward_item, reverse_item, pair_item, shuffle_item).
    shuffle_item is None when the manifest record has no shuffle_video_path.
    """
    forward_path = record["forward_video_path"]
    reverse_path = record.get("reverse_video_path") or forward_path
    shuffle_path = record.get("shuffle_video_path") or ""

    fwd_rev_prompt = get_forward_reverse_caption_prompt()
    forward_frames = sample_video_frames_by_fps(forward_path, target_fps=fwd_rev_fps, max_frames=max_frames)
    reverse_frames = sample_video_frames_by_fps(reverse_path, target_fps=fwd_rev_fps, max_frames=max_frames)
    forward_resp = call_vlm(api_base, api_key, model, SYSTEM_PROMPT, fwd_rev_prompt, forward_frames, resize_max_width, jpeg_quality, retries)
    reverse_resp = call_vlm(api_base, api_key, model, SYSTEM_PROMPT, fwd_rev_prompt, reverse_frames, resize_max_width, jpeg_quality, retries)

    base = {
        "clip_key": record["clip_key"],
        "event_id": record.get("event_id"),
        "duration_sec": record.get("duration_sec"),
    }
    forward_direction_clear = bool(forward_resp.get("direction_clear", True))
    reverse_direction_clear = bool(reverse_resp.get("direction_clear", True))

    forward_item = {
        **base,
        "direction": "forward",
        "video_path": forward_path,
        "caption": forward_resp.get("caption", "").strip(),
        "confidence": float(forward_resp.get("confidence", 0.0) or 0.0),
        "direction_clear": forward_direction_clear,
    }
    reverse_item = {
        **base,
        "direction": "reverse",
        "video_path": reverse_path,
        "caption": reverse_resp.get("caption", "").strip(),
        "confidence": float(reverse_resp.get("confidence", 0.0) or 0.0),
        "direction_clear": reverse_direction_clear,
    }

    shuffle_item: dict | None = None
    if shuffle_path:
        # Compute segment count for segment-aware sampling & prompt
        # New manifests store shuffle_n_segments directly; fall back to deriving
        # from the legacy shuffle_segment_sec field for backward compatibility.
        actual_dur = record.get("actual_duration_sec") or record.get("duration_sec") or 0.0
        n_segments = int(record.get("shuffle_n_segments") or 0)
        if n_segments <= 0 and actual_dur > 0:
            _old_seg_sec = record.get("shuffle_segment_sec", 2.0) or 2.0
            n_segments = int(actual_dur // _old_seg_sec)
        segment_dur = (actual_dur / n_segments) if n_segments > 0 else 2.0

        shuf_prompt = get_shuffle_caption_prompt(
            n_segments=n_segments,
            segment_sec=segment_dur,
        )
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
        shuffle_resp = call_vlm(api_base, api_key, model, SYSTEM_PROMPT, shuf_prompt, shuffle_frames, resize_max_width, jpeg_quality, retries)
        shuffle_item = {
            **base,
            "direction": "shuffle",
            "video_path": shuffle_path,
            "caption": shuffle_resp.get("caption", "").strip(),
            "confidence": float(shuffle_resp.get("confidence", 0.0) or 0.0),
            # direction_clear is always False for shuffle clips by construction
            "direction_clear": False,
            "n_segments": n_segments,
        }

    pair = {
        **base,
        "forward_video_path": forward_path,
        "reverse_video_path": reverse_path,
        "shuffle_video_path": shuffle_path,
        "forward_caption": forward_item["caption"],
        "reverse_caption": reverse_item["caption"],
        "shuffle_caption": shuffle_item["caption"] if shuffle_item else "",
        "forward_confidence": forward_item["confidence"],
        "reverse_confidence": reverse_item["confidence"],
        "shuffle_confidence": shuffle_item["confidence"] if shuffle_item else 0.0,
        "forward_direction_clear": forward_direction_clear,
        "reverse_direction_clear": reverse_direction_clear,
        "is_different": forward_item["caption"] != reverse_item["caption"],
    }
    return forward_item, reverse_item, pair, shuffle_item


def load_manifest(path: str, max_samples: int) -> list[dict]:
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if max_samples > 0:
        records = records[:max_samples]
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate forward/reverse/shuffle captions for event-level AoT data.")
    parser.add_argument("--manifest-jsonl", required=True, help="Input manifest JSONL")
    parser.add_argument("--output-dir", required=True, help="Directory to save JSONL outputs")
    parser.add_argument("--api-base", required=True, help="OpenAI-compatible API base")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--api-key", default="", help="API key, defaults to OPENAI_API_KEY/NOVITA_API_KEY")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--max-frames", type=int, default=32, help="Hard cap on frames per clip")
    parser.add_argument("--fwd-rev-fps", type=float, default=1.0, help="Frame sampling rate for forward/reverse caption (default: 1fps)")
    parser.add_argument("--shuffle-fps", type=float, default=2.0, help="Frame sampling rate for shuffle caption (default: 2fps, matches training input)")
    parser.add_argument("--resize-max-width", type=int, default=768, help="Resize frames before upload")
    parser.add_argument("--jpeg-quality", type=int, default=60, help="JPEG quality for upload")
    parser.add_argument("--retries", type=int, default=3, help="API retry count")
    parser.add_argument("--max-samples", type=int, default=0, help="Max number of manifest records to annotate")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    forward_out_path = os.path.join(args.output_dir, "forward_captions.jsonl")
    reverse_out_path = os.path.join(args.output_dir, "reverse_captions.jsonl")
    shuffle_out_path = os.path.join(args.output_dir, "shuffle_captions.jsonl")
    pair_out_path = os.path.join(args.output_dir, "caption_pairs.jsonl")

    records = load_manifest(args.manifest_jsonl, max_samples=args.max_samples)
    has_shuffle = any(r.get("shuffle_video_path") for r in records)
    pair_count = 0

    open_files: dict[str, Any] = {
        "forward": open(forward_out_path, "w", encoding="utf-8", buffering=1),
        "reverse": open(reverse_out_path, "w", encoding="utf-8", buffering=1),
        "pair": open(pair_out_path, "w", encoding="utf-8", buffering=1),
    }
    if has_shuffle:
        open_files["shuffle"] = open(shuffle_out_path, "w", encoding="utf-8", buffering=1)

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    annotate_one,
                    record,
                    args.api_base,
                    args.api_key,
                    args.model,
                    args.max_frames,
                    args.fwd_rev_fps,
                    args.shuffle_fps,
                    args.resize_max_width,
                    args.jpeg_quality,
                    args.retries,
                )
                for record in records
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Annotating captions", unit="clip"):
                f_item, r_item, p_item, s_item = future.result()
                open_files["forward"].write(json.dumps(f_item, ensure_ascii=False) + "\n")
                open_files["reverse"].write(json.dumps(r_item, ensure_ascii=False) + "\n")
                open_files["pair"].write(json.dumps(p_item, ensure_ascii=False) + "\n")
                if s_item is not None and "shuffle" in open_files:
                    open_files["shuffle"].write(json.dumps(s_item, ensure_ascii=False) + "\n")
                pair_count += 1
    finally:
        for fh in open_files.values():
            fh.close()

    print(f"Wrote {pair_count} caption pairs to {pair_out_path}")
    if has_shuffle:
        print(f"Wrote shuffle captions to {shuffle_out_path}")


if __name__ == "__main__":
    main()
