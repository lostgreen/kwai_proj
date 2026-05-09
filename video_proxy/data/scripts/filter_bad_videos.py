#!/usr/bin/env python3
"""Pre-filter training JSONL by validating video files with decord.

Reads each sample, tries to open every video path with decord,
and drops samples where any video fails to decode.

Usage:
    python local_scripts/filter_bad_videos.py \
        --input  /path/to/train.jsonl \
        --output /path/to/train_clean.jsonl \
        [--bad    /path/to/bad_samples.jsonl] \
        [--workers 8]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def _check_video(video_path: str) -> tuple[str, str | None]:
    """Return (path, error_msg | None)."""
    try:
        import decord

        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(video_path, num_threads=1)
        fps = vr.get_avg_fps()
        n_frames = len(vr)
        if n_frames == 0:
            return video_path, "0 frames"
        if fps <= 0:
            return video_path, f"invalid fps={fps}"
        return video_path, None
    except Exception as e:
        return video_path, str(e)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input JSONL")
    parser.add_argument("--output", required=True, help="Output (valid samples) JSONL")
    parser.add_argument("--bad", default=None, help="Output JSONL for rejected samples (optional)")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers for video check")
    args = parser.parse_args()

    # ---------- load ----------
    with open(args.input) as f:
        records = [json.loads(line) for line in f if line.strip()]
    print(f"[filter] Loaded {len(records)} records from {args.input}")

    # ---------- collect unique video paths ----------
    all_videos: set[str] = set()
    for rec in records:
        for v in rec.get("videos", []):
            all_videos.add(v)
    print(f"[filter] {len(all_videos)} unique video paths to check")

    # ---------- validate videos ----------
    bad_videos: dict[str, str] = {}  # path -> error
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_check_video, v): v for v in all_videos}
        done = 0
        for fut in as_completed(futures):
            done += 1
            path, err = fut.result()
            if err is not None:
                bad_videos[path] = err
            if done % 200 == 0 or done == len(futures):
                print(f"[filter] Checked {done}/{len(futures)} videos, {len(bad_videos)} bad so far")

    print(f"[filter] Bad videos: {len(bad_videos)}/{len(all_videos)}")
    for p, e in sorted(bad_videos.items()):
        print(f"  BAD: {p}  ({e})")

    # ---------- filter records ----------
    kept, dropped = [], []
    for rec in records:
        video_paths = rec.get("videos", [])
        bad_in_rec = [v for v in video_paths if v in bad_videos]
        if bad_in_rec:
            rec["_filter_reason"] = {v: bad_videos[v] for v in bad_in_rec}
            dropped.append(rec)
        else:
            kept.append(rec)

    print(f"[filter] Kept {len(kept)}, dropped {len(dropped)} records")

    # ---------- write ----------
    os.makedirs(Path(args.output).parent, exist_ok=True)
    with open(args.output, "w") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[filter] Wrote {args.output}")

    if args.bad and dropped:
        os.makedirs(Path(args.bad).parent, exist_ok=True)
        with open(args.bad, "w") as f:
            for rec in dropped:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[filter] Wrote bad samples to {args.bad}")


if __name__ == "__main__":
    main()
