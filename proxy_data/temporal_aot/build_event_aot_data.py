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
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
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


def parse_ratio(value: Any) -> float | None:
    if value in (None, "", "N/A", "0/0"):
        return None
    try:
        return float(Fraction(str(value)))
    except (TypeError, ValueError, ZeroDivisionError):
        return parse_float(value)


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
        "avg_frame_rate": parse_ratio(video_stream.get("avg_frame_rate")) or parse_ratio(video_stream.get("r_frame_rate")),
    }


def load_bad_sample_index(bad_samples_jsonl: str) -> tuple[set[str], set[str]]:
    clip_keys: set[str] = set()
    video_paths: set[str] = set()
    with open(bad_samples_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            clip_key = obj.get("clip_key")
            if isinstance(clip_key, str) and clip_key:
                clip_keys.add(clip_key)
            for path_key in ("video_path", "forward_video_path"):
                video_path = obj.get(path_key)
                if isinstance(video_path, str) and video_path:
                    video_paths.add(video_path)
    return clip_keys, video_paths


def filter_known_bad_records(
    records: list[dict],
    bad_clip_keys: set[str],
    bad_video_paths: set[str],
) -> tuple[list[dict], list[dict]]:
    kept_records: list[dict] = []
    skipped_records: list[dict] = []
    for record in records:
        clip_key = record.get("clip_key")
        video_path = record.get("video_path")
        if (isinstance(clip_key, str) and clip_key in bad_clip_keys) or (
            isinstance(video_path, str) and video_path in bad_video_paths
        ):
            skipped_records.append(
                {
                    "clip_key": clip_key,
                    "video_path": video_path,
                    "reason": "known_bad_sample",
                }
            )
            continue
        kept_records.append(record)
    return kept_records, skipped_records


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


def ensure_composite_stats(record: dict) -> dict[str, Any]:
    width = record.get("width")
    height = record.get("height")
    fps = record.get("avg_frame_rate")
    if width and height and fps:
        return {
            "width": width,
            "height": height,
            "avg_frame_rate": fps,
        }

    probe_data = probe_video(record["video_path"])
    stats = extract_video_stats(probe_data)
    return {
        "width": stats.get("width"),
        "height": stats.get("height"),
        "avg_frame_rate": stats.get("avg_frame_rate"),
    }


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


def verify_video_readable(video_path: str) -> bool:
    """Quick check that ffprobe can open the output and find a video stream."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", video_path],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0 and "video" in (result.stdout or "")
    except (subprocess.TimeoutExpired, OSError):
        return False


def run_ffmpeg_with_fallback(cmd: list[str], dst_path: str, dry_run: bool) -> None:
    """Run ffmpeg with fast encoding; on verification failure, retry with default preset."""
    run_ffmpeg(cmd, dry_run=dry_run)
    if dry_run:
        return
    if verify_video_readable(dst_path):
        return
    # Fallback: replace ultrafast with medium and retry
    fallback_cmd = []
    for token in cmd:
        if token == "ultrafast":
            fallback_cmd.append("medium")
        else:
            fallback_cmd.append(token)
    os.remove(dst_path)
    run_ffmpeg(fallback_cmd, dry_run=False)


# Shared fast-encoding flags: ultrafast preset cuts encode time 5-8x vs default medium.
FAST_ENCODE_FLAGS = ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", "-threads", "1"]


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
        *FAST_ENCODE_FLAGS,
        dst_path,
    ]
    run_ffmpeg_with_fallback(cmd, dst_path, dry_run=dry_run)


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
        *FAST_ENCODE_FLAGS,
        dst_path,
    ]
    run_ffmpeg_with_fallback(cmd, dst_path, dry_run=dry_run)


def format_fps_value(fps: float | None) -> str:
    if fps is None or fps <= 0:
        return "25"
    return f"{fps:.6f}".rstrip("0").rstrip(".")


def build_composite_clip(
    forward_path: str,
    black_path: str,
    reverse_path: str,
    dst_path: str,
    width: int | None,
    height: int | None,
    fps: float | None,
    dry_run: bool,
) -> None:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    black_filters = []
    if width and height:
        black_filters.append(f"scale={width}:{height}")
    black_filters.append(f"fps={format_fps_value(fps)}")
    black_filters.extend(["setsar=1", "format=yuv420p"])
    black_filter = ",".join(black_filters)
    filter_complex = (
        "[0:v]setsar=1,format=yuv420p[fwd];"
        f"[1:v]{black_filter}[gap];"
        "[2:v]setsar=1,format=yuv420p[rev];"
        "[fwd][gap][rev]concat=n=3:v=1:a=0[outv]"
    )
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
        filter_complex,
        "-map",
        "[outv]",
        *FAST_ENCODE_FLAGS,
        dst_path,
    ]
    run_ffmpeg_with_fallback(cmd, dst_path, dry_run=dry_run)


def build_shuffle_clip(
    src_path: str,
    dst_path: str,
    actual_duration: float,
    segment_sec: float,
    seed: int,
    dry_run: bool,
) -> int:
    """Shuffle the temporal order of fixed-length segments using a single ffmpeg filter_complex.

    Returns the number of segments used, or raises ValueError if the clip is too short.
    A different random seed produces a different shuffle permutation, so callers can
    generate multiple distinct shuffled variants from the same source clip.
    """
    n_full = int(actual_duration // segment_sec)
    if n_full < 2:
        raise ValueError(
            f"Clip too short for shuffle: duration={actual_duration:.2f}s requires at least "
            f"2×{segment_sec}s segments"
        )

    rng = random.Random(seed)
    order = list(range(n_full))
    # Guarantee the shuffled order differs from the original
    while order == list(range(n_full)):
        rng.shuffle(order)

    # Build a single-pass filter_complex: trim each segment in shuffled order, then concat
    trim_filters = []
    for out_idx, seg_idx in enumerate(order):
        start = seg_idx * segment_sec
        end = start + segment_sec
        trim_filters.append(
            f"[0:v]trim=start={start:.6f}:end={end:.6f},setpts=PTS-STARTPTS,format=yuv420p[s{out_idx}]"
        )
    concat_inputs = "".join(f"[s{i}]" for i in range(len(order)))
    trim_filters.append(f"{concat_inputs}concat=n={len(order)}:v=1:a=0[outv]")
    filter_complex = ";".join(trim_filters)

    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", src_path,
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-an",
        *FAST_ENCODE_FLAGS,
        dst_path,
    ]
    run_ffmpeg_with_fallback(cmd, dst_path, dry_run=dry_run)
    return len(order)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build event-level AoT manifest and optional reverse/composite/shuffle videos.")
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
    parser.add_argument("--shuffle-dir", default="", help="Directory to save temporally-shuffled clips")
    parser.add_argument("--invalid-report-jsonl", default="", help="Optional JSONL path to save skipped invalid clips")
    parser.add_argument(
        "--bad-samples-jsonl",
        default="",
        help="Optional JSONL path of previously known bad samples to skip by clip_key/video_path",
    )
    parser.add_argument("--black-video", default="", help="Reusable black screen clip path")
    parser.add_argument("--black-gap-sec", type=float, default=2.0, help="Black gap duration for T2V composite")
    parser.add_argument("--make-reverse", action="store_true", help="Export reversed clips with ffmpeg")
    parser.add_argument("--make-composite", action="store_true", help="Export T2V composite clips")
    parser.add_argument("--make-shuffle", action="store_true", help="Export temporally-shuffled clips with ffmpeg")
    parser.add_argument(
        "--shuffle-segment-sec",
        type=float,
        default=2.0,
        help="Segment length in seconds for temporal shuffling (default: 2.0)",
    )
    parser.add_argument(
        "--min-shuffle-segments",
        type=int,
        default=3,
        help="Minimum number of full segments required to generate a shuffle clip (default: 3)",
    )
    parser.add_argument(
        "--build-workers",
        type=int,
        default=4,
        help="Parallel workers for ffmpeg generation (default: 4). Each worker runs one ffmpeg subprocess.",
    )
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
    skipped_records: list[dict] = []
    if args.bad_samples_jsonl:
        bad_clip_keys, bad_video_paths = load_bad_sample_index(args.bad_samples_jsonl)
        records, known_bad_records = filter_known_bad_records(records, bad_clip_keys, bad_video_paths)
        skipped_records.extend(known_bad_records)
    else:
        records, invalid_records = filter_valid_records(
            records,
            min_duration=args.min_duration,
            max_duration_diff_sec=args.max_duration_diff_sec,
        )
        skipped_records.extend(invalid_records)
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

    # -----------------------------------------------------------------------
    # Per-record generation worker (runs in thread pool)
    # -----------------------------------------------------------------------
    skipped_lock = threading.Lock()

    def generate_one(record: dict) -> dict | None:
        """Generate reverse/composite/shuffle clips for one record.

        Returns the completed output record on success, or None on failure
        (failure details are appended to skipped_records via the lock).
        """
        clip_key = record["clip_key"]
        forward_path = record["video_path"]
        reverse_path = ""
        composite_path = ""
        shuffle_path = ""
        actual_duration = record.get("actual_duration_sec") or record.get("duration_sec") or 0.0

        try:
            if args.make_reverse or args.make_composite:
                if not args.reverse_dir:
                    raise ValueError("--reverse-dir is required when --make-reverse or --make-composite is set")
                reverse_path = os.path.join(args.reverse_dir, f"{clip_key}_rev.mp4")
                if args.dry_run or not os.path.exists(reverse_path):
                    build_reverse_clip(forward_path, reverse_path, dry_run=args.dry_run)

            if args.make_composite:
                composite_stats = ensure_composite_stats(record)
                composite_path = os.path.join(args.composite_dir, f"{clip_key}_t2v.mp4")
                if args.dry_run or not os.path.exists(composite_path):
                    build_composite_clip(
                        forward_path,
                        black_video_path,
                        reverse_path,
                        composite_path,
                        width=composite_stats.get("width"),
                        height=composite_stats.get("height"),
                        fps=composite_stats.get("avg_frame_rate"),
                        dry_run=args.dry_run,
                    )

            if args.make_shuffle:
                if not args.shuffle_dir:
                    raise ValueError("--shuffle-dir is required when --make-shuffle is set")
                n_full = int(actual_duration // args.shuffle_segment_sec)
                if n_full < args.min_shuffle_segments:
                    raise ValueError(
                        f"Skipping shuffle: only {n_full} full {args.shuffle_segment_sec}s segments "
                        f"(need {args.min_shuffle_segments})"
                    )
                shuffle_path = os.path.join(args.shuffle_dir, f"{clip_key}_shuf.mp4")
                if args.dry_run or not os.path.exists(shuffle_path):
                    build_shuffle_clip(
                        forward_path,
                        shuffle_path,
                        actual_duration=actual_duration,
                        segment_sec=args.shuffle_segment_sec,
                        seed=args.seed + hash(clip_key) % (2 ** 31),
                        dry_run=args.dry_run,
                    )

        except (subprocess.CalledProcessError, json.JSONDecodeError, ValueError) as exc:
            with skipped_lock:
                skipped_records.append(
                    {
                        "clip_key": clip_key,
                        "video_path": forward_path,
                        "reason": "ffmpeg_generation_failed",
                        "detail": str(exc),
                    }
                )
            return None

        return {
            **record,
            "forward_video_path": forward_path,
            "reverse_video_path": reverse_path,
            "composite_video_path": composite_path,
            "shuffle_video_path": shuffle_path,
            "shuffle_segment_sec": args.shuffle_segment_sec if args.make_shuffle else 0.0,
            "black_gap_sec": args.black_gap_sec if args.make_composite else 0.0,
        }

    # -----------------------------------------------------------------------
    # Run generation in parallel, write results in deterministic order
    # -----------------------------------------------------------------------
    written_count = 0
    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)

    workers = max(1, args.build_workers)
    with (
        open(args.output_jsonl, "w", encoding="utf-8") as out,
        ThreadPoolExecutor(max_workers=workers) as executor,
    ):
        futures = {executor.submit(generate_one, record): record for record in records}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Building AoT data", unit="clip"):
            out_record = future.result()
            if out_record is not None:
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
