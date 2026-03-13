#!/usr/bin/env python3
"""
extract_frames.py — Extract 1fps frames from YouCook2 windowed clips using ffmpeg.

Usage:
    python extract_frames.py \
        --jsonl proxy_data/youcook2_train_easyr1.jsonl \
        --video-dir /path/to/Youcook2_windowed \
        --output-dir proxy_data/youcook2_seg_annotation/frames \
        [--workers 8] \
        [--limit 100]

Output layout:
    frames/
        {video_id}_{start}_{end}/
            0001.jpg   ← frame at t=1s relative to clip start
            0002.jpg
            ...
"""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def resolve_video_path(original_path: str, video_dir: str | None) -> Path:
    """
    Resolve the actual video file path.

    If `video_dir` is given, replace the original directory component with it.
    Otherwise use the original path as-is.
    """
    p = Path(original_path)
    if video_dir:
        return Path(video_dir) / p.name
    return p


def extract_clip_frames(
    video_path: Path,
    output_dir: Path,
    fps: float = 1.0,
    max_frames: int = 0,
) -> list[Path]:
    """
    Use ffmpeg to extract frames at `fps` from `video_path`.

    Args:
        video_path:  Path to the video file.
        output_dir:  Directory to write JPEG frames into.
        fps:         Frames per second (default 1.0).
        max_frames:  If > 0, stop after this many frames (0 = no limit).
    Returns:
        Sorted list of extracted frame paths.
    Raises:
        RuntimeError: If ffmpeg exits with a non-zero code.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(output_dir / "%04d.jpg")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", "2",        # JPEG quality
    ]
    if max_frames > 0:
        cmd += ["-frames:v", str(max_frames)]
    cmd += [pattern]

    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {video_path}:\n{result.stderr[-500:]}"
        )

    frames = sorted(output_dir.glob("*.jpg"))
    return frames


def clip_key(video_path: str) -> str:
    """Return 'videoId_start_end' from a windowed clip filename."""
    return Path(video_path).stem


def process_record(
    record: dict,
    video_dir: str | None,
    output_base: Path,
    fps: float,
    max_frames: int,
    overwrite: bool,
) -> dict:
    """
    Extract frames for a single JSONL record.

    Returns a status dict: {clip_key, n_frames, skipped, error}.
    """
    videos = record.get("videos") or []
    if not videos:
        return {"clip_key": "?", "n_frames": 0, "skipped": True, "error": "no videos field"}

    orig_path = videos[0]
    video_path = resolve_video_path(orig_path, video_dir)
    key = clip_key(orig_path)
    out_dir = output_base / key

    if not overwrite and out_dir.exists() and len(list(out_dir.glob("*.jpg"))) > 0:
        n = len(list(out_dir.glob("*.jpg")))
        return {"clip_key": key, "n_frames": n, "skipped": True, "error": None}

    if not video_path.exists():
        return {"clip_key": key, "n_frames": 0, "skipped": True,
                "error": f"video not found: {video_path}"}

    try:
        frames = extract_clip_frames(video_path, out_dir, fps=fps, max_frames=max_frames)
        return {"clip_key": key, "n_frames": len(frames), "skipped": False, "error": None}
    except RuntimeError as e:
        return {"clip_key": key, "n_frames": 0, "skipped": False, "error": str(e)[:200]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract 1fps frames from YC2 windowed clips")
    parser.add_argument("--jsonl", required=True,
                        help="Input JSONL file (e.g. youcook2_train_easyr1.jsonl)")
    parser.add_argument("--video-dir", default=None,
                        help="Override base directory for video files. "
                             "If omitted, uses original paths from JSONL.")
    parser.add_argument("--output-dir", required=True,
                        help="Root directory to write frame folders under")
    parser.add_argument("--fps", type=float, default=1.0,
                        help="Frames per second to extract (default: 1.0)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Max frames per clip (0 = no limit)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel ffmpeg workers (default: 4)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process at most N clips (0 = all)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-extract even if output dir already has frames")
    args = parser.parse_args()

    # Load JSONL
    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        print(f"ERROR: JSONL not found: {jsonl_path}", file=sys.stderr)
        sys.exit(1)

    records = []
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

    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(records)} clips → {output_base}")
    print(f"FPS={args.fps}  max_frames={args.max_frames or 'unlimited'}  workers={args.workers}")

    done = skipped = errors = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                process_record, rec, args.video_dir, output_base,
                args.fps, args.max_frames, args.overwrite
            ): rec
            for rec in records
        }
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            if res["error"]:
                errors += 1
                print(f"[{i}/{len(records)}] ERROR  {res['clip_key']}: {res['error']}")
            elif res["skipped"]:
                skipped += 1
                if i % 100 == 0:
                    print(f"[{i}/{len(records)}] (skip) {res['clip_key']} ({res['n_frames']} frames)")
            else:
                done += 1
                print(f"[{i}/{len(records)}] OK     {res['clip_key']} → {res['n_frames']} frames")

    print(f"\nDone: {done} extracted, {skipped} skipped, {errors} errors")


if __name__ == "__main__":
    main()
