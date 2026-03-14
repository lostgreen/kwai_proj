#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Annotate forward/reverse captions for event clips listed in a manifest JSONL.

Output files under --output-dir:
- forward_captions.jsonl
- reverse_captions.jsonl
- caption_pairs.jsonl

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

from prompts import SYSTEM_PROMPT, get_forward_reverse_caption_prompt


def sample_video_frames(video_path: str, max_frames: int) -> list[Image.Image]:
    try:
        import decord
    except ImportError as exc:
        raise ImportError("decord is required for annotation: pip install decord") from exc

    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    total = len(vr)
    if total == 0:
        return []
    if max_frames <= 0 or total <= max_frames:
        indices = list(range(total))
    else:
        stride = (total - 1) / (max_frames - 1)
        indices = [round(i * stride) for i in range(max_frames)]
    frames = vr.get_batch(indices).asnumpy()
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
    resize_max_width: int,
    jpeg_quality: int,
    retries: int,
) -> tuple[dict, dict, dict]:
    forward_path = record["forward_video_path"]
    reverse_path = record.get("reverse_video_path") or forward_path

    prompt = get_forward_reverse_caption_prompt()
    forward_frames = sample_video_frames(forward_path, max_frames=max_frames)
    reverse_frames = sample_video_frames(reverse_path, max_frames=max_frames)
    forward_resp = call_vlm(api_base, api_key, model, SYSTEM_PROMPT, prompt, forward_frames, resize_max_width, jpeg_quality, retries)
    reverse_resp = call_vlm(api_base, api_key, model, SYSTEM_PROMPT, prompt, reverse_frames, resize_max_width, jpeg_quality, retries)

    base = {
        "clip_key": record["clip_key"],
        "event_id": record.get("event_id"),
        "duration_sec": record.get("duration_sec"),
    }
    forward_item = {
        **base,
        "direction": "forward",
        "video_path": forward_path,
        "caption": forward_resp.get("caption", "").strip(),
        "confidence": float(forward_resp.get("confidence", 0.0) or 0.0),
    }
    reverse_item = {
        **base,
        "direction": "reverse",
        "video_path": reverse_path,
        "caption": reverse_resp.get("caption", "").strip(),
        "confidence": float(reverse_resp.get("confidence", 0.0) or 0.0),
    }
    pair = {
        **base,
        "forward_video_path": forward_path,
        "reverse_video_path": reverse_path,
        "forward_caption": forward_item["caption"],
        "reverse_caption": reverse_item["caption"],
        "forward_confidence": forward_item["confidence"],
        "reverse_confidence": reverse_item["confidence"],
        "is_different": forward_item["caption"] != reverse_item["caption"],
    }
    return forward_item, reverse_item, pair


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
    parser = argparse.ArgumentParser(description="Annotate forward/reverse captions for event-level AoT data.")
    parser.add_argument("--manifest-jsonl", required=True, help="Input manifest JSONL")
    parser.add_argument("--output-dir", required=True, help="Directory to save JSONL outputs")
    parser.add_argument("--api-base", required=True, help="OpenAI-compatible API base")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--api-key", default="", help="API key, defaults to OPENAI_API_KEY/NOVITA_API_KEY")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--max-frames", type=int, default=16, help="Max frames sampled from each clip")
    parser.add_argument("--resize-max-width", type=int, default=768, help="Resize frames before upload")
    parser.add_argument("--jpeg-quality", type=int, default=60, help="JPEG quality for upload")
    parser.add_argument("--retries", type=int, default=3, help="API retry count")
    parser.add_argument("--max-samples", type=int, default=0, help="Max number of manifest records to annotate")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    forward_path = os.path.join(args.output_dir, "forward_captions.jsonl")
    reverse_path = os.path.join(args.output_dir, "reverse_captions.jsonl")
    pair_path = os.path.join(args.output_dir, "caption_pairs.jsonl")

    records = load_manifest(args.manifest_jsonl, max_samples=args.max_samples)
    pair_count = 0

    with (
        open(forward_path, "w", encoding="utf-8", buffering=1) as forward_file,
        open(reverse_path, "w", encoding="utf-8", buffering=1) as reverse_file,
        open(pair_path, "w", encoding="utf-8", buffering=1) as pair_file,
        ThreadPoolExecutor(max_workers=args.workers) as executor,
    ):
        futures = [
            executor.submit(
                annotate_one,
                record,
                args.api_base,
                args.api_key,
                args.model,
                args.max_frames,
                args.resize_max_width,
                args.jpeg_quality,
                args.retries,
            )
            for record in records
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Annotating captions", unit="clip"):
            f_item, r_item, p_item = future.result()
            forward_file.write(json.dumps(f_item, ensure_ascii=False) + "\n")
            reverse_file.write(json.dumps(r_item, ensure_ascii=False) + "\n")
            pair_file.write(json.dumps(p_item, ensure_ascii=False) + "\n")
            forward_file.flush()
            reverse_file.flush()
            pair_file.flush()
            pair_count += 1

    print(f"Wrote {pair_count} caption pairs to {pair_path}")


if __name__ == "__main__":
    main()
