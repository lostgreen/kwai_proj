#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Annotate recipe-instruction-style step captions for L2 event clips.

These captions serve as the TEXT CONTEXT in T→V Event Logic tasks (add_t2v, replace_t2v):
instead of watching video clips as context, the model reads these step descriptions,
then selects the correct video from multiple-choice options.

Output: l2_step_captions.jsonl  — one entry per unique clip_key, indexed by clip_key.

─── Manifest format (--manifest-jsonl) ───────────────────────────────────────
Each line is a JSON object with at minimum:
    {"clip_key": "...", "video_path": "/abs/path/to/clip.mp4"}
Optional fields (used as metadata / annotation hints):
    "event_id", "video_id", "duration_sec", "instruction"

You can generate the manifest in two ways:
  1. From an existing Event Logic dataset JSONL (recommended, no server access needed):
       python annotate_l2_step_captions.py --from-dataset proxy_data/event_logic/data/proxy_train_text_options.jsonl ...
  2. From a hand-crafted manifest listing all L2 clips you want annotated.

─── Example usage ────────────────────────────────────────────────────────────
# Generate from existing V→T dataset (extracts all unique video clips):
python proxy_data/event_logic/annotate_l2_step_captions.py \\
    --from-dataset proxy_data/event_logic/data/proxy_train_text_options.jsonl \\
    --output proxy_data/event_logic/data/l2_step_captions.jsonl \\
    --api-base http://localhost:8000/v1 \\
    --model Qwen3-VL-7B \\
    --workers 4

# Or from a pre-built manifest:
python proxy_data/event_logic/annotate_l2_step_captions.py \\
    --manifest-jsonl proxy_data/event_logic/data/l2_clips_manifest.jsonl \\
    --output proxy_data/event_logic/data/l2_step_captions.jsonl \\
    --api-base https://api.novita.ai/v3/openai \\
    --model qwen/qwen2.5-vl-72b-instruct \\
    --workers 8
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

from prompts import STEP_CAPTION_SYSTEM_PROMPT, get_step_caption_prompt


# ─────────────────────────────────────────────────────────────────────────────
# Video frame sampling (same pattern as annotate_event_captions.py)
# ─────────────────────────────────────────────────────────────────────────────

def sample_video_frames_by_fps(
    video_path: str,
    target_fps: float,
    max_frames: int,
) -> list[Image.Image]:
    """Sample frames at a fixed FPS. Returns [] on failure."""
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


# ─────────────────────────────────────────────────────────────────────────────
# VLM helpers
# ─────────────────────────────────────────────────────────────────────────────

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
        content.append({
            "type": "image_url",
            "image_url": {"url": image_to_data_url(frame, jpeg_quality, resize_max_width), "detail": "low"},
        })

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
                max_tokens=256,
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"VLM call failed after {retries} retries: {last_error}")


# ─────────────────────────────────────────────────────────────────────────────
# Per-clip annotation
# ─────────────────────────────────────────────────────────────────────────────

def annotate_one(
    record: dict,
    api_base: str,
    api_key: str,
    model: str,
    max_frames: int,
    fps: float,
    resize_max_width: int,
    jpeg_quality: int,
    retries: int,
) -> dict:
    """Annotate one event clip. Returns a caption record."""
    clip_key = record["clip_key"]
    video_path = record["video_path"]
    instruction_hint = record.get("instruction", "")

    frames = sample_video_frames_by_fps(video_path, target_fps=fps, max_frames=max_frames)
    if not frames:
        raise ValueError(f"No frames extracted from {video_path}")

    user_text = get_step_caption_prompt(instruction_hint=instruction_hint)
    resp = call_vlm(
        api_base, api_key, model,
        STEP_CAPTION_SYSTEM_PROMPT, user_text, frames,
        resize_max_width, jpeg_quality, retries,
    )

    return {
        "clip_key": clip_key,
        "video_path": video_path,
        "event_id": record.get("event_id"),
        "video_id": record.get("video_id"),
        "duration_sec": record.get("duration_sec"),
        "instruction_hint": instruction_hint,
        "caption": resp.get("caption", "").strip(),
        "confidence": float(resp.get("confidence", 0.0) or 0.0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Manifest loading
# ─────────────────────────────────────────────────────────────────────────────

def load_manifest_jsonl(path: str, max_samples: int) -> list[dict]:
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


def build_manifest_from_dataset(dataset_path: str) -> list[dict]:
    """
    Extract unique (clip_key, video_path) pairs from an existing Event Logic dataset JSONL.

    Works with both proxy_train_text_options.jsonl (V→T) and any other JSONL that has
    a "videos" field containing video clip paths.

    clip_key is derived from the video filename (basename without extension).
    """
    seen: dict[str, dict] = {}  # clip_key -> record

    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            meta = item.get("metadata", {})
            videos: list[str] = item.get("videos", [])

            for vpath in videos:
                clip_key = Path(vpath).stem  # e.g. "WlHWRPyA7_g_event04_95_112"
                if clip_key not in seen:
                    seen[clip_key] = {
                        "clip_key": clip_key,
                        "video_path": vpath,
                        "video_id": meta.get("video_id", ""),
                        "event_id": None,
                        "duration_sec": None,
                        "instruction": "",
                    }

    records = list(seen.values())
    print(f"[manifest] Extracted {len(records)} unique clips from {dataset_path}")
    return records


def load_existing_index(path: str) -> dict[str, dict]:
    """Load existing l2_step_captions.jsonl and return dict keyed by clip_key."""
    index: dict[str, dict] = {}
    if not os.path.exists(path):
        return index
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            index[item["clip_key"]] = item
    return index


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Annotate recipe-instruction-style captions for L2 event clips "
            "(used as text context in T→V Event Logic tasks)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input: one of two modes
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--manifest-jsonl",
        help="Input manifest JSONL. Each line: {clip_key, video_path, optional: instruction, event_id, video_id, duration_sec}",
    )
    input_group.add_argument(
        "--from-dataset",
        metavar="DATASET_JSONL",
        help=(
            "Build manifest automatically from an existing Event Logic dataset JSONL "
            "(e.g. proxy_train_text_options.jsonl). Extracts all unique video clips."
        ),
    )

    parser.add_argument("--output", required=True, help="Output path for l2_step_captions.jsonl")
    parser.add_argument("--api-base", required=True, help="OpenAI-compatible API base URL")
    parser.add_argument("--model", required=True, help="VLM model name")
    parser.add_argument("--api-key", default="", help="API key (falls back to OPENAI_API_KEY / NOVITA_API_KEY env)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel annotation workers (default: 4)")
    parser.add_argument("--fps", type=float, default=1.0, help="Frame sampling rate in fps (default: 1.0)")
    parser.add_argument("--max-frames", type=int, default=16, help="Max frames per clip (default: 16)")
    parser.add_argument("--resize-max-width", type=int, default=768, help="Resize frames to this max width before upload (default: 768)")
    parser.add_argument("--jpeg-quality", type=int, default=70, help="JPEG compression quality (default: 70)")
    parser.add_argument("--retries", type=int, default=3, help="Number of API retries per clip (default: 3)")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit number of clips to annotate (0 = all)")
    parser.add_argument(
        "--confidence-threshold", type=float, default=0.0,
        help="Only write entries with confidence >= this value (default: 0.0 = keep all)",
    )
    parser.add_argument(
        "--no-resume", action="store_true", default=False,
        help="Overwrite output file instead of resuming from existing annotations",
    )
    args = parser.parse_args()

    # ── Load manifest ─────────────────────────────────────────────────────────
    if args.from_dataset:
        records = build_manifest_from_dataset(args.from_dataset)
        if args.max_samples > 0:
            records = records[: args.max_samples]
    else:
        records = load_manifest_jsonl(args.manifest_jsonl, max_samples=args.max_samples)

    # Deduplicate by clip_key
    seen_keys: set[str] = set()
    deduped: list[dict] = []
    for r in records:
        if r["clip_key"] not in seen_keys:
            seen_keys.add(r["clip_key"])
            deduped.append(r)
    records = deduped
    print(f"Loaded {len(records)} unique clips.")

    # ── Resume ────────────────────────────────────────────────────────────────
    resume = not args.no_resume
    done_index = load_existing_index(args.output) if resume else {}
    if done_index:
        print(f"Resuming: skipping {len(done_index)} already-annotated clips.")
    records = [r for r in records if r["clip_key"] not in done_index]
    print(f"Clips to annotate: {len(records)}")

    if not records:
        print("Nothing to do — all clips already annotated.")
        return

    # ── Annotate ──────────────────────────────────────────────────────────────
    os.makedirs(Path(args.output).parent, exist_ok=True)
    file_mode = "a" if resume and done_index else "w"
    written = 0
    skipped = 0
    failed = 0

    with open(args.output, file_mode, encoding="utf-8", buffering=1) as out_f:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    annotate_one,
                    record,
                    args.api_base,
                    args.api_key,
                    args.model,
                    args.max_frames,
                    args.fps,
                    args.resize_max_width,
                    args.jpeg_quality,
                    args.retries,
                ): record["clip_key"]
                for record in records
            }

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Annotating step captions",
                unit="clip",
            ):
                clip_key = futures[future]
                try:
                    result = future.result()
                    if result["confidence"] >= args.confidence_threshold:
                        out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        written += 1
                    else:
                        print(
                            f"[SKIP] {clip_key}: confidence {result['confidence']:.2f} "
                            f"< threshold {args.confidence_threshold:.2f}"
                        )
                        skipped += 1
                except Exception as exc:
                    print(f"[ERROR] {clip_key}: {exc}")
                    failed += 1

    total_done = len(done_index) + written
    print(f"\nDone.")
    print(f"  Written this run : {written}")
    print(f"  Skipped (low conf): {skipped}")
    print(f"  Failed            : {failed}")
    print(f"  Total in output   : {total_done}")
    print(f"  Output            : {args.output}")


if __name__ == "__main__":
    main()
