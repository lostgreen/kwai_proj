#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build event-level AoT manifests from existing proxy data.

This script treats a single event clip as the atomic unit.
It can:
1. collect unique event clips from proxy/mixed JSONL
2. optionally export reversed clips offline
3. optionally build T2V composite clips: forward + black + reverse

Example:
python proxy_data/temporal_aot/build_event_aot_data.py \
  --input-jsonl proxy_data/proxy_train_easyr1.jsonl \
  --output-jsonl /tmp/aot_event_manifest.jsonl \
  --reverse-dir /tmp/aot_reverse \
  --composite-dir /tmp/aot_t2v \
  --make-reverse \
  --make-composite \
  --max-samples 500
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
from pathlib import Path


EVENT_RE = re.compile(r"(?P<video>.+)_event(?P<event>\d+)_(?P<start>\d+)_(?P<end>\d+)\.mp4$")


def parse_event_meta(video_path: str) -> dict:
    name = os.path.basename(video_path)
    match = EVENT_RE.match(name)
    meta = {
        "video_path": video_path,
        "clip_key": Path(video_path).stem,
        "event_id": None,
        "start_sec": None,
        "end_sec": None,
        "duration_sec": None,
        "source_video_id": None,
    }
    if not match:
        return meta

    start_sec = int(match.group("start"))
    end_sec = int(match.group("end"))
    meta.update(
        {
            "event_id": int(match.group("event")),
            "start_sec": start_sec,
            "end_sec": end_sec,
            "duration_sec": end_sec - start_sec,
            "source_video_id": match.group("video"),
        }
    )
    return meta


def load_unique_event_clips(input_jsonl: str, min_duration: int) -> list[dict]:
    seen = set()
    items: list[dict] = []
    with open(input_jsonl, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            metadata = obj.get("metadata", {})
            for video_path in obj.get("videos", []):
                if not isinstance(video_path, str):
                    continue
                if video_path in seen:
                    continue
                meta = parse_event_meta(video_path)
                duration = meta.get("duration_sec")
                if duration is not None and duration < min_duration:
                    continue
                if not os.path.exists(video_path):
                    continue
                seen.add(video_path)
                meta["recipe_type"] = metadata.get("recipe_type")
                meta["source_task_type"] = metadata.get("task_type")
                items.append(meta)
    return items


def run_ffmpeg(cmd: list[str], dry_run: bool) -> None:
    if dry_run:
        print("[DRY RUN]", " ".join(cmd))
        return
    subprocess.run(cmd, check=True)


def build_reverse_clip(src_path: str, dst_path: str, dry_run: bool) -> None:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        src_path,
        "-vf",
        "reverse",
        "-an",
        dst_path,
    ]
    run_ffmpeg(cmd, dry_run=dry_run)


def build_black_clip(dst_path: str, duration_sec: float, dry_run: bool) -> None:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c=black:s=768x432:d={duration_sec}:r=25",
        "-an",
        dst_path,
    ]
    run_ffmpeg(cmd, dry_run=dry_run)


def build_composite_clip(forward_path: str, black_path: str, reverse_path: str, dst_path: str, dry_run: bool) -> None:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        forward_path,
        "-i",
        black_path,
        "-i",
        reverse_path,
        "-filter_complex",
        "[0:v][1:v][2:v]concat=n=3:v=1:a=0[outv]",
        "-map",
        "[outv]",
        dst_path,
    ]
    run_ffmpeg(cmd, dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build event-level AoT manifest and optional reverse/composite videos.")
    parser.add_argument("--input-jsonl", required=True, help="Source JSONL, e.g. proxy_train_easyr1.jsonl")
    parser.add_argument("--output-jsonl", required=True, help="Output manifest JSONL")
    parser.add_argument("--max-samples", type=int, default=0, help="Max number of unique event clips to keep (0 = all)")
    parser.add_argument("--min-duration", type=int, default=3, help="Minimum event clip duration in seconds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--reverse-dir", default="", help="Directory to save reversed clips")
    parser.add_argument("--composite-dir", default="", help="Directory to save T2V composite clips")
    parser.add_argument("--black-video", default="", help="Reusable black screen clip path")
    parser.add_argument("--black-gap-sec", type=float, default=2.0, help="Black gap duration for T2V composite")
    parser.add_argument("--make-reverse", action="store_true", help="Export reversed clips with ffmpeg")
    parser.add_argument("--make-composite", action="store_true", help="Export T2V composite clips")
    parser.add_argument("--dry-run", action="store_true", help="Print ffmpeg commands without executing them")
    args = parser.parse_args()

    random.seed(args.seed)
    records = load_unique_event_clips(args.input_jsonl, min_duration=args.min_duration)
    random.shuffle(records)
    if args.max_samples > 0:
        records = records[: args.max_samples]

    black_video_path = args.black_video
    if args.make_composite and not black_video_path:
        if not args.composite_dir:
            raise ValueError("--composite-dir is required when --make-composite is set")
        black_video_path = os.path.join(args.composite_dir, f"black_{args.black_gap_sec:.1f}s.mp4")
        if args.dry_run or not os.path.exists(black_video_path):
            build_black_clip(black_video_path, duration_sec=args.black_gap_sec, dry_run=args.dry_run)

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as out:
        for record in records:
            clip_key = record["clip_key"]
            forward_path = record["video_path"]
            reverse_path = ""
            composite_path = ""

            if args.make_reverse:
                if not args.reverse_dir:
                    raise ValueError("--reverse-dir is required when --make-reverse is set")
                reverse_path = os.path.join(args.reverse_dir, f"{clip_key}_rev.mp4")
                if args.dry_run or not os.path.exists(reverse_path):
                    build_reverse_clip(forward_path, reverse_path, dry_run=args.dry_run)

            if args.make_composite:
                if not reverse_path:
                    if not args.reverse_dir:
                        raise ValueError("--reverse-dir is required to build composite clips")
                    reverse_path = os.path.join(args.reverse_dir, f"{clip_key}_rev.mp4")
                    if args.dry_run or not os.path.exists(reverse_path):
                        build_reverse_clip(forward_path, reverse_path, dry_run=args.dry_run)
                composite_path = os.path.join(args.composite_dir, f"{clip_key}_t2v.mp4")
                if args.dry_run or not os.path.exists(composite_path):
                    build_composite_clip(forward_path, black_video_path, reverse_path, composite_path, dry_run=args.dry_run)

            out_record = {
                **record,
                "forward_video_path": forward_path,
                "reverse_video_path": reverse_path,
                "composite_video_path": composite_path,
                "black_gap_sec": args.black_gap_sec if args.make_composite else 0.0,
            }
            out.write(json.dumps(out_record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} records to {args.output_jsonl}")


if __name__ == "__main__":
    main()
