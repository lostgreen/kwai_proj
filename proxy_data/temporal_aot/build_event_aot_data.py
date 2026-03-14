#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build event-level AoT manifests from existing event clip annotations.

This script treats a single event clip as the atomic unit.
It can:
1. collect unique event clips from extracted clip annotations or proxy JSONL
2. optionally export reversed clips offline
3. optionally build T2V composite clips: forward + black + reverse

Recommended input is the extracted clip database JSON, for example:
{
  "GLd3aX16zBg": [
    {
      "clip_path": ".../GLd3aX16zBg_event00_90_102.mp4",
      "original_video_id": "GLd3aX16zBg",
      "recipe_type": "113",
      "subset": "training",
      "sentence": "spread margarine on two slices of white bread",
      "segment_in_original": [90, 102],
      "event_id": 0,
      "sequence_index": 0
    }
  ]
}
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm


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


def merge_clip_metadata(base_meta: dict, clip_info: dict) -> dict:
    merged = dict(base_meta)
    segment = clip_info.get("segment_in_original")
    if isinstance(segment, list) and len(segment) == 2:
        start_sec, end_sec = segment
        if isinstance(start_sec, (int, float)) and isinstance(end_sec, (int, float)):
            merged["start_sec"] = int(start_sec)
            merged["end_sec"] = int(end_sec)
            merged["duration_sec"] = int(end_sec) - int(start_sec)

    merged["event_id"] = clip_info.get("event_id", merged.get("event_id"))
    merged["source_video_id"] = clip_info.get("original_video_id", merged.get("source_video_id"))
    merged["recipe_type"] = clip_info.get("recipe_type")
    merged["subset"] = clip_info.get("subset")
    merged["sentence"] = clip_info.get("sentence")
    merged["sequence_index"] = clip_info.get("sequence_index")
    return merged


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


def load_event_clips_from_db(clip_db_json: str, min_duration: int, subset: str) -> list[dict]:
    seen = set()
    items: list[dict] = []
    with open(clip_db_json, encoding="utf-8") as f:
        db = json.load(f)

    if not isinstance(db, dict):
        raise ValueError("Clip database JSON must be a dict: video_id -> list[clip_info]")

    for original_video_id, clips in db.items():
        if not isinstance(clips, list):
            continue
        for clip_info in clips:
            if not isinstance(clip_info, dict):
                continue
            clip_path = clip_info.get("clip_path")
            if not isinstance(clip_path, str):
                continue
            if clip_path in seen:
                continue
            if subset and clip_info.get("subset") != subset:
                continue
            meta = parse_event_meta(clip_path)
            meta = merge_clip_metadata(meta, clip_info)
            if not meta.get("source_video_id"):
                meta["source_video_id"] = original_video_id
            duration = meta.get("duration_sec")
            if duration is not None and duration < min_duration:
                continue
            if not os.path.exists(clip_path):
                continue
            seen.add(clip_path)
            meta["video_path"] = clip_path
            items.append(meta)
    return items


def probe_video(video_path: str) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        video_path,
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(result.stdout or "{}")


def parse_float(value: Any) -> float | None:
    try:
        if value in (None, "", "N/A"):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_video_stats(probe_data: dict[str, Any]) -> dict[str, Any]:
    streams = probe_data.get("streams", [])
    video_stream = None
    for stream in streams:
        if isinstance(stream, dict) and stream.get("codec_type") == "video":
            video_stream = stream
            break

    if video_stream is None:
        raise ValueError("no video stream found")

    actual_duration = parse_float(video_stream.get("duration"))
    if actual_duration is None:
        actual_duration = parse_float(probe_data.get("format", {}).get("duration"))

    nb_frames = video_stream.get("nb_frames")
    if nb_frames not in (None, "", "N/A"):
        try:
            nb_frames = int(nb_frames)
        except (TypeError, ValueError):
            nb_frames = None
    else:
        nb_frames = None

    return {
        "actual_duration_sec": actual_duration,
        "width": video_stream.get("width"),
        "height": video_stream.get("height"),
        "codec_name": video_stream.get("codec_name"),
        "nb_frames": nb_frames,
    }


def validate_record(
    record: dict,
    min_duration: int,
    max_duration_diff_sec: float,
) -> tuple[bool, dict[str, Any]]:
    video_path = record["video_path"]
    if not os.path.exists(video_path):
        return False, {"reason": "missing_file"}

    try:
        probe_data = probe_video(video_path)
        stats = extract_video_stats(probe_data)
    except subprocess.CalledProcessError as exc:
        return False, {"reason": "ffprobe_failed", "detail": str(exc)}
    except (json.JSONDecodeError, ValueError) as exc:
        return False, {"reason": "invalid_video", "detail": str(exc)}

    actual_duration = stats.get("actual_duration_sec")
    if actual_duration is None:
        return False, {"reason": "missing_duration"}
    if actual_duration < float(min_duration):
        return False, {"reason": "actual_duration_too_short", "actual_duration_sec": actual_duration}

    expected_duration = record.get("duration_sec")
    if expected_duration is not None:
        duration_diff = abs(float(expected_duration) - actual_duration)
        if duration_diff > max_duration_diff_sec:
            return False, {
                "reason": "duration_mismatch",
                "expected_duration_sec": expected_duration,
                "actual_duration_sec": actual_duration,
                "duration_diff_sec": duration_diff,
            }
        stats["duration_diff_sec"] = duration_diff
    else:
        stats["duration_diff_sec"] = None

    return True, stats


def filter_valid_records(
    records: list[dict],
    min_duration: int,
    max_duration_diff_sec: float,
) -> tuple[list[dict], list[dict]]:
    valid_records: list[dict] = []
    skipped_records: list[dict] = []
    for record in tqdm(records, desc="Validating clips", unit="clip"):
        is_valid, info = validate_record(
            record,
            min_duration=min_duration,
            max_duration_diff_sec=max_duration_diff_sec,
        )
        if not is_valid:
            skipped_records.append(
                {
                    "clip_key": record.get("clip_key"),
                    "video_path": record.get("video_path"),
                    **info,
                }
            )
            continue
        record.update(info)
        valid_records.append(record)
    return valid_records, skipped_records


def run_ffmpeg(cmd: list[str], dry_run: bool) -> None:
    if dry_run:
        print("[DRY RUN]", " ".join(cmd))
        return
    quiet_cmd = [
        cmd[0],
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostats",
        *cmd[1:],
    ]
    subprocess.run(quiet_cmd, check=True)


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
    parser.add_argument("--clip-db-json", default="", help="Extracted clip database JSON, recommended input")
    parser.add_argument("--input-jsonl", default="", help="Fallback source JSONL, e.g. proxy_train_easyr1.jsonl")
    parser.add_argument("--output-jsonl", required=True, help="Output manifest JSONL")
    parser.add_argument("--max-samples", type=int, default=0, help="Max number of unique event clips to keep (0 = all)")
    parser.add_argument("--min-duration", type=int, default=3, help="Minimum event clip duration in seconds")
    parser.add_argument(
        "--max-duration-diff-sec",
        type=float,
        default=2.0,
        help="Max allowed diff between metadata duration and probed video duration",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--subset", default="", help="Optional subset filter for clip DB input, e.g. training or validation")
    parser.add_argument("--reverse-dir", default="", help="Directory to save reversed clips")
    parser.add_argument("--composite-dir", default="", help="Directory to save T2V composite clips")
    parser.add_argument("--invalid-report-jsonl", default="", help="Optional JSONL path to save skipped invalid clips")
    parser.add_argument("--black-video", default="", help="Reusable black screen clip path")
    parser.add_argument("--black-gap-sec", type=float, default=2.0, help="Black gap duration for T2V composite")
    parser.add_argument("--make-reverse", action="store_true", help="Export reversed clips with ffmpeg")
    parser.add_argument("--make-composite", action="store_true", help="Export T2V composite clips")
    parser.add_argument("--dry-run", action="store_true", help="Print ffmpeg commands without executing them")
    args = parser.parse_args()

    random.seed(args.seed)
    if bool(args.clip_db_json) == bool(args.input_jsonl):
        raise ValueError("Specify exactly one of --clip-db-json or --input-jsonl")

    if args.clip_db_json:
        records = load_event_clips_from_db(
            args.clip_db_json,
            min_duration=args.min_duration,
            subset=args.subset,
        )
    else:
        records = load_unique_event_clips(args.input_jsonl, min_duration=args.min_duration)
    records, skipped_records = filter_valid_records(
        records,
        min_duration=args.min_duration,
        max_duration_diff_sec=args.max_duration_diff_sec,
    )
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

    written_count = 0
    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as out:
        for record in tqdm(records, desc="Building AoT data", unit="clip"):
            clip_key = record["clip_key"]
            forward_path = record["video_path"]
            reverse_path = ""
            composite_path = ""

            try:
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
            except subprocess.CalledProcessError as exc:
                skipped_records.append(
                    {
                        "clip_key": clip_key,
                        "video_path": forward_path,
                        "reason": "ffmpeg_generation_failed",
                        "detail": str(exc),
                    }
                )
                continue

            out_record = {
                **record,
                "forward_video_path": forward_path,
                "reverse_video_path": reverse_path,
                "composite_video_path": composite_path,
                "black_gap_sec": args.black_gap_sec if args.make_composite else 0.0,
            }
            out.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            written_count += 1

    if args.invalid_report_jsonl:
        os.makedirs(os.path.dirname(args.invalid_report_jsonl) or ".", exist_ok=True)
        with open(args.invalid_report_jsonl, "w", encoding="utf-8") as f:
            for item in skipped_records:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Wrote {written_count} records to {args.output_jsonl}")
    print(f"Skipped {len(skipped_records)} invalid/failed records")
    if args.invalid_report_jsonl:
        print(f"Wrote invalid report to {args.invalid_report_jsonl}")


if __name__ == "__main__":
    main()
