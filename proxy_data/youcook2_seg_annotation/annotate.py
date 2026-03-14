#!/usr/bin/env python3
"""
annotate.py — Hierarchical DVC annotation pipeline for YouCook2 segmentation data.

Supports 3 annotation levels (Level 1 is active; Level 2 & 3 are reserved).

Usage:
    # 旧模式：基于 JSONL 对应的样本进行标注
    python annotate.py \
        --jsonl proxy_data/youcook2_train_easyr1.jsonl \
        --frames-dir proxy_data/youcook2_seg_annotation/frames \
        --output-dir proxy_data/youcook2_seg_annotation/annotations \
        --level 1 \
        --api-base http://localhost:8000/v1 \
        --model Qwen3-VL-7B \
        [--api-key YOUR_KEY] \
        [--workers 2] \
        [--limit 50]

    # 新模式：直接标注 frames 目录下的所有原视频抽帧结果
    python annotate.py \
        --frames-dir proxy_data/youcook2_seg_annotation/frames \
        --output-dir proxy_data/youcook2_seg_annotation/annotations \
        --level 1 \
        --api-base http://localhost:8000/v1 \
        --model Qwen3-VL-7B

Output:
    annotations/{clip_key}.json  — per-clip annotation result:
    {
        "clip_key": "GLd3aX16zBg_90_174",
        "video_path": "...",
        "clip_duration_sec": 84.0,
        "level1": {<macro_phases>},   # from Level 1 LLM call
        "level2": null,               # filled by level 2 pass
        "level3": null,               # filled by level 3 pass
        "n_frames": 84,
        "frame_dir": "...",
        "annotated_at": "2025-..."
    }
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
    get_level1_prompt,
    get_level2_prompt,
    get_level3_prompt,
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


def frames_to_base64(
    frame_dir: Path,
    max_frames: int = 64,
    sample_every: int = 1,
    resize_max_width: int = 0,
    jpeg_quality: int = 60,
) -> tuple[list[str], list[int]]:
    """
    Load JPEG frames from `frame_dir`, optionally sub-sample and resize them.

    Returns:
        (b64_list, indices) where b64_list are raw base64-encoded JPEG strings and
        indices are frame numbers derived from filenames.
    """
    frame_files = sorted(frame_dir.glob("*.jpg"))
    if not frame_files:
        return [], []
    sample_every = max(1, sample_every)
    if sample_every > 1:
        frame_files = frame_files[::sample_every]
    if max_frames > 0 and len(frame_files) > max_frames:
        stride = (len(frame_files) - 1) / (max_frames - 1)
        frame_files = [frame_files[round(i * stride)] for i in range(max_frames)]
    b64_list = []
    indices = []
    for fp in frame_files:
        b64_list.append(encode_frame_to_base64(
            fp,
            resize_max_width=resize_max_width,
            jpeg_quality=jpeg_quality,
        ))
        try:
            indices.append(int(fp.stem))
        except ValueError:
            indices.append(len(indices) + 1)
    return b64_list, indices


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
# OpenAI-compatible API client via openai library
# ─────────────────────────────────────────────────────────────────────────────

def call_vlm(
    api_base: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_text: str,
    frame_b64_list: list[str],
    frame_indices: list[int],
    max_tokens: int = 8192,
    temperature: float = 0.0,
    retries: int = 3,
) -> str:
    """
    Call a VLM endpoint (OpenAI-compatible) with interleaved frame images.

    Message layout:
        system: system_prompt
        user:   [text: user_text]
                [text: "[Timestamp MM:SS | Frame {i}]"] [image: frame_i]  × n_frames

    Uses response_format={"type": "json_object"} for structured output.
    API key is taken from `api_key` or NOVITA_API_KEY / OPENAI_API_KEY env vars.
    Frame payloads follow the Novita example format:
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai is required: pip install openai")

    key = api_key or os.environ.get("NOVITA_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    client = OpenAI(api_key=key, base_url=api_base)

    # Build user content: prompt text first, then interleaved frame labels + images
    content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    for i, b64 in enumerate(frame_b64_list):
        fid = frame_indices[i] if i < len(frame_indices) else i + 1
        content.append({"type": "text", "text": f"[Timestamp {format_mmss(fid)} | Frame {fid}]"})
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
    """
    Extract and parse the first JSON block from the model response.
    Tries bare JSON first, then looks for ```json ... ``` fence.
    """
    text = text.strip()
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting from code fence
    import re
    m = re.search(r"```(?:json)?\s*(\{[\s\S]+?\})\s*```", text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Try finding first { ... } block
    m2 = re.search(r"\{[\s\S]+\}", text)
    if m2:
        try:
            return json.loads(m2.group(0))
        except json.JSONDecodeError:
            pass
    return {"_raw_response": text, "_parse_error": True}


# ─────────────────────────────────────────────────────────────────────────────
# Per-clip annotation
# ─────────────────────────────────────────────────────────────────────────────

def annotate_clip(
    record: dict,
    frames_base: Path,
    output_dir: Path,
    level: int,
    api_base: str,
    api_key: str,
    model: str,
    max_frames_per_call: int,
    level1_target_fps: float,
    level1_resize_max_width: int,
    level1_jpeg_quality: int,
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
    level_key = f"level{level}"
    if not overwrite and existing.get(level_key) is not None:
        return {"clip_key": key, "ok": True, "error": None, "skipped": True}

    # Load frames
    frame_dir = frames_base / key
    frame_meta = load_frame_meta(frame_dir)
    source_fps = float(frame_meta.get("fps") or 1.0)

    # Get clip duration from metadata
    clip_duration = float(
        frame_meta.get("annotation_end_sec")
        or meta.get("clip_end")
        or meta.get("clip_duration")
        or count_extracted_frames(frame_dir)
    )
    n_total_frames = count_extracted_frames(frame_dir)

    # Build annotation
    try:
        if level == 1:
            sample_every = 1
            if level1_target_fps > 0 and source_fps > level1_target_fps:
                sample_every = max(1, round(source_fps / level1_target_fps))
            frame_b64, frame_indices = frames_to_base64(
                frame_dir,
                max_frames=max_frames_per_call,
                sample_every=sample_every,
                resize_max_width=level1_resize_max_width,
                jpeg_quality=level1_jpeg_quality,
            )
            if not frame_b64:
                return {"clip_key": key, "ok": False,
                        "error": f"no frames found in {frame_dir}", "skipped": False}
            prompt_text = get_level1_prompt(clip_duration)
            raw = call_vlm(api_base, api_key, model, SYSTEM_PROMPT,
                           prompt_text, frame_b64, frame_indices)
            parsed = parse_json_from_response(raw)
            result_key = "level1"
            result_val = {
                **parsed,
                "_sampling": {
                    "source_fps": source_fps,
                    "target_fps": level1_target_fps,
                    "sample_every": sample_every,
                    "resize_max_width": level1_resize_max_width,
                    "jpeg_quality": level1_jpeg_quality,
                    "n_sampled_frames": len(frame_indices),
                    "sampled_frame_indices": frame_indices,
                },
            }

        elif level == 2:
            frame_b64, frame_indices = frames_to_base64(frame_dir, max_frames=max_frames_per_call)
            if not frame_b64:
                return {"clip_key": key, "ok": False,
                        "error": f"no frames found in {frame_dir}", "skipped": False}
            l1 = existing.get("level1")
            if l1 is None:
                return {"clip_key": key, "ok": False,
                        "error": "level1 annotation missing; run level 1 first", "skipped": False}
            prompt_text = get_level2_prompt(clip_duration, l1)
            raw = call_vlm(api_base, api_key, model, SYSTEM_PROMPT,
                           prompt_text, frame_b64, frame_indices)
            parsed = parse_json_from_response(raw)
            result_key = "level2"
            result_val = parsed

        elif level == 3:
            frame_b64, frame_indices = frames_to_base64(frame_dir, max_frames=max_frames_per_call)
            if not frame_b64:
                return {"clip_key": key, "ok": False,
                        "error": f"no frames found in {frame_dir}", "skipped": False}
            l1 = existing.get("level1")
            l2 = existing.get("level2")
            if l1 is None or l2 is None:
                return {"clip_key": key, "ok": False,
                        "error": "level1/level2 annotation missing; run previous levels first",
                        "skipped": False}
            prompt_text = get_level3_prompt(clip_duration, l1, l2)
            raw = call_vlm(api_base, api_key, model, SYSTEM_PROMPT,
                           prompt_text, frame_b64, frame_indices)
            parsed = parse_json_from_response(raw)
            result_key = "level3"
            result_val = parsed
        else:
            return {"clip_key": key, "ok": False, "error": f"unsupported level {level}", "skipped": False}

    except NotImplementedError as e:
        return {"clip_key": key, "ok": False, "error": str(e), "skipped": False}
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
        result_key: result_val,
        "annotated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(ann, f, ensure_ascii=False, indent=2)

    return {"clip_key": key, "ok": True, "error": None, "skipped": False}


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
    parser.add_argument("--level", type=int, choices=[1, 2, 3], default=1,
                        help="Annotation level to run (1=macro, 2=activity, 3=step)")
    parser.add_argument("--api-base", default="https://api.novita.ai/v3/openai",
                        help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", default="",
                        help="API key (prefers NOVITA_API_KEY env var, then OPENAI_API_KEY)")
    parser.add_argument("--model", default="pa/gmn-2.5-pr",
                        help="Model name to pass to the API")
    parser.add_argument("--max-frames-per-call", type=int, default=32,
                        help="Max frames to include per API call (memory limit)")
    parser.add_argument("--level1-target-fps", type=float, default=0.5,
                        help="Effective FPS for Level 1 on top of pre-extracted frames")
    parser.add_argument("--level1-resize-max-width", type=int, default=384,
                        help="Resize Level 1 frames before upload; <=0 disables resizing")
    parser.add_argument("--level1-jpeg-quality", type=int, default=60,
                        help="JPEG quality used when recompressing Level 1 frames before upload")
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

    print(f"Annotating {len(records)} clips at Level {args.level}")
    print(f"API: {args.api_base}  model: {args.model}  workers: {args.workers}")
    if args.level == 1:
        print(
            "Level 1 sampling: "
            f"target_fps={args.level1_target_fps}  "
            f"resize_max_width={args.level1_resize_max_width}  "
            f"jpeg_quality={args.level1_jpeg_quality}"
        )
    print(f"Frames: {frames_base}  Output: {output_dir}\n")

    ok_count = skipped_count = error_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                annotate_clip,
                rec, frames_base, output_dir, args.level,
                args.api_base, api_key, args.model,
                args.max_frames_per_call,
                args.level1_target_fps,
                args.level1_resize_max_width,
                args.level1_jpeg_quality,
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
