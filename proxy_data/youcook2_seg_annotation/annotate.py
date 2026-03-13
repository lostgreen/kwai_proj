#!/usr/bin/env python3
"""
annotate.py — Hierarchical DVC annotation pipeline for YouCook2 windowed clips.

Supports 3 annotation levels (Level 1 is active; Level 2 & 3 are reserved).

Usage:
    # Annotate all clips at Level 1 only (single level, recommended first pass):
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

    # If frames are not pre-extracted, you can pass --video-dir to extract inline:
    python annotate.py ... --video-dir /path/to/Youcook2_windowed

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
from pathlib import Path
from typing import Any

from prompts import (
    SYSTEM_PROMPT,
    get_level1_prompt,
    get_level2_prompt,
    get_level3_prompt,
)


# ─────────────────────────────────────────────────────────────────────────────
# Frame helpers
# ─────────────────────────────────────────────────────────────────────────────

def frames_to_base64(frame_dir: Path, max_frames: int = 64) -> list[str]:
    """
    Load JPEG frames from `frame_dir`, evenly sample up to `max_frames`,
    and return them as base64 data URLs.
    """
    frame_files = sorted(frame_dir.glob("*.jpg"))
    if not frame_files:
        return []
    if len(frame_files) > max_frames:
        stride = (len(frame_files) - 1) / (max_frames - 1)
        frame_files = [frame_files[round(i * stride)] for i in range(max_frames)]
    result = []
    for fp in frame_files:
        with open(fp, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
            result.append(f"data:image/jpeg;base64,{b64}")
    return result


def clip_key_from_path(video_path: str) -> str:
    return Path(video_path).stem


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI-compatible API client (no extra dependencies beyond stdlib + requests)
# ─────────────────────────────────────────────────────────────────────────────

def call_vlm(
    api_base: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_text: str,
    frame_b64_list: list[str],
    max_tokens: int = 2048,
    temperature: float = 0.1,
    retries: int = 3,
) -> str:
    """
    Call an OpenAI-compatible VLM endpoint with vision (interleaved images).

    Frames are sent as image_url content items before the text prompt.
    Returns the assistant's response text.
    """
    try:
        import requests
    except ImportError:
        raise ImportError("requests is required: pip install requests")

    # Build content list: images first, then text instruction
    content: list[dict[str, Any]] = []
    for b64 in frame_b64_list:
        content.append({
            "type": "image_url",
            "image_url": {"url": b64},
        })
    content.append({"type": "text", "text": user_text})

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": content},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    last_error = None
    for attempt in range(retries):
        try:
            r = requests.post(
                f"{api_base.rstrip('/')}/chat/completions",
                json=payload,
                headers=headers,
                timeout=120,
            )
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            last_error = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # exponential back-off
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
    key = clip_key_from_path(vid_path)
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
    frame_b64 = frames_to_base64(frame_dir, max_frames=max_frames_per_call)
    if not frame_b64:
        return {"clip_key": key, "ok": False,
                "error": f"no frames found in {frame_dir}", "skipped": False}

    # Get clip duration from metadata
    meta = record.get("metadata") or {}
    clip_duration = float(meta.get("clip_duration") or len(frame_b64))

    # Build annotation
    try:
        if level == 1:
            prompt_text = get_level1_prompt(clip_duration)
            raw = call_vlm(api_base, api_key, model, SYSTEM_PROMPT, prompt_text, frame_b64)
            parsed = parse_json_from_response(raw)
            result_key = "level1"
            result_val = parsed

        elif level == 2:
            l1 = existing.get("level1")
            if l1 is None:
                return {"clip_key": key, "ok": False,
                        "error": "level1 annotation missing; run level 1 first", "skipped": False}
            prompt_text = get_level2_prompt(clip_duration, l1)
            raw = call_vlm(api_base, api_key, model, SYSTEM_PROMPT, prompt_text, frame_b64)
            parsed = parse_json_from_response(raw)
            result_key = "level2"
            result_val = parsed

        elif level == 3:
            l1 = existing.get("level1")
            l2 = existing.get("level2")
            if l1 is None or l2 is None:
                return {"clip_key": key, "ok": False,
                        "error": "level1/level2 annotation missing; run previous levels first",
                        "skipped": False}
            prompt_text = get_level3_prompt(clip_duration, l1, l2)
            raw = call_vlm(api_base, api_key, model, SYSTEM_PROMPT, prompt_text, frame_b64)
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
        "clip_duration_sec": clip_duration,
        "n_frames": len(frame_b64),
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
    parser.add_argument("--jsonl", required=True,
                        help="Input JSONL (e.g. youcook2_train_easyr1.jsonl)")
    parser.add_argument("--frames-dir", required=True,
                        help="Root directory of pre-extracted 1fps frames")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write per-clip annotation JSON files")
    parser.add_argument("--level", type=int, choices=[1, 2, 3], default=1,
                        help="Annotation level to run (1=macro, 2=activity, 3=step)")
    parser.add_argument("--api-base", default="http://localhost:8000/v1",
                        help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", default="",
                        help="API key (can also set OPENAI_API_KEY env var)")
    parser.add_argument("--model", default="Qwen3-VL-7B",
                        help="Model name to pass to the API")
    parser.add_argument("--max-frames-per-call", type=int, default=32,
                        help="Max frames to include per API call (memory limit)")
    parser.add_argument("--workers", type=int, default=2,
                        help="Parallel annotation workers")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process at most N clips (0 = all)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-annotate even if the level is already done")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")

    # Load JSONL
    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        print(f"ERROR: JSONL not found: {jsonl_path}", file=sys.stderr)
        sys.exit(1)

    records: list[dict] = []
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

    frames_base = Path(args.frames_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Annotating {len(records)} clips at Level {args.level}")
    print(f"API: {args.api_base}  model: {args.model}  workers: {args.workers}")
    print(f"Frames: {frames_base}  Output: {output_dir}\n")

    ok_count = skipped_count = error_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                annotate_clip,
                rec, frames_base, output_dir, args.level,
                args.api_base, api_key, args.model,
                args.max_frames_per_call, args.overwrite,
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
