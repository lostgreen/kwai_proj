#!/usr/bin/env python3
"""Validate mixed-task experiment JSONL before launching training."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc


def is_number_or_none(value: Any) -> bool:
    return value is None or isinstance(value, (int, float)) and not isinstance(value, bool)


def check_frame_policy(
    path: Path,
    row: dict[str, Any],
    line_no: int,
    max_frames: int,
    check_frame_files: bool,
) -> list[str]:
    errors: list[str] = []
    videos = row.get("videos")
    if not isinstance(videos, list) or not videos or not isinstance(videos[0], list):
        errors.append(f"{path}:{line_no}: videos is not a non-empty frame-list")
        return errors

    for vid_idx, frames in enumerate(videos):
        if not isinstance(frames, list) or not frames:
            errors.append(f"{path}:{line_no}: videos[{vid_idx}] is empty or not a list")
            continue
        if not all(isinstance(frame, str) and frame for frame in frames):
            errors.append(f"{path}:{line_no}: videos[{vid_idx}] contains non-string frame paths")
        if max_frames > 0 and len(frames) > max_frames:
            errors.append(f"{path}:{line_no}: videos[{vid_idx}] has {len(frames)} frames > max_frames={max_frames}")
        if check_frame_files:
            for frame in (frames[0], frames[-1]):
                if not Path(frame).is_file():
                    errors.append(f"{path}:{line_no}: frame file missing: {frame}")

    meta = row.get("metadata") or {}
    sampling = meta.get("experiment_frame_sampling")
    if not isinstance(sampling, dict):
        errors.append(f"{path}:{line_no}: missing metadata.experiment_frame_sampling")
        return errors

    for rule_idx, rule in enumerate(sampling.get("rules") or []):
        if not isinstance(rule, dict):
            errors.append(f"{path}:{line_no}: rules[{rule_idx}] is not an object")
            continue
        if not is_number_or_none(rule.get("min_sec")):
            errors.append(f"{path}:{line_no}: rules[{rule_idx}].min_sec is not numeric/null: {rule.get('min_sec')!r}")
        if not is_number_or_none(rule.get("max_sec")):
            errors.append(f"{path}:{line_no}: rules[{rule_idx}].max_sec is not numeric/null: {rule.get('max_sec')!r}")
        if not is_number_or_none(rule.get("fps")):
            errors.append(f"{path}:{line_no}: rules[{rule_idx}].fps is not numeric/null: {rule.get('fps')!r}")

    for video_idx, video_meta in enumerate(sampling.get("videos") or []):
        if not isinstance(video_meta, dict):
            errors.append(f"{path}:{line_no}: sampling.videos[{video_idx}] is not an object")
            continue
        for key in ("duration_sec", "base_fps", "target_fps"):
            if not is_number_or_none(video_meta.get(key)):
                errors.append(f"{path}:{line_no}: sampling.videos[{video_idx}].{key} invalid: {video_meta.get(key)!r}")
        for key in ("max_frames", "input_frames", "after_fps_frames", "output_frames"):
            if not isinstance(video_meta.get(key), int) or isinstance(video_meta.get(key), bool):
                errors.append(f"{path}:{line_no}: sampling.videos[{video_idx}].{key} is not int: {video_meta.get(key)!r}")

    return errors


def arrow_check(path: Path, rows: list[dict[str, Any]]) -> list[str]:
    try:
        import pyarrow as pa
    except Exception as exc:  # pragma: no cover - depends on env
        return [f"{path}: pyarrow import failed: {exc!r}"]

    try:
        pa.Table.from_pylist(rows)
    except Exception as exc:
        return [f"{path}: pyarrow conversion failed: {exc!r}"]
    return []


def check_file(path: Path, args: argparse.Namespace) -> int:
    if not path.is_file():
        print(f"[check-experiment] missing: {path}", file=sys.stderr)
        return 1

    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    applied: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    max_frame_len: Counter[str] = Counter()
    errors: list[str] = []

    try:
        for line_no, row in iter_jsonl(path):
            rows.append(row)
            if not isinstance(row, dict):
                errors.append(f"{path}:{line_no}: row is not a JSON object")
                continue
            task = str(row.get("problem_type") or "unknown")
            counts[task] += 1
            videos = row.get("videos") or []
            if isinstance(videos, list):
                for frames in videos:
                    if isinstance(frames, list):
                        max_frame_len[task] = max(max_frame_len[task], len(frames))
            meta = row.get("metadata") or {}
            sampling = meta.get("experiment_frame_sampling")
            if isinstance(sampling, dict):
                applied[task] += 1
                for video_meta in sampling.get("videos") or []:
                    if isinstance(video_meta, dict):
                        sources[str(video_meta.get("source") or "unknown")] += 1
                errors.extend(check_frame_policy(path, row, line_no, args.max_frames, args.check_frame_files))
            else:
                skipped[task] += 1
                if args.require_frame_policy:
                    errors.append(f"{path}:{line_no}: missing metadata.experiment_frame_sampling")
    except ValueError as exc:
        print(f"[check-experiment] {exc}", file=sys.stderr)
        return 1

    if args.expect_no_skipped and sum(skipped.values()) > 0:
        errors.append(f"{path}: skipped frame policy records: {dict(skipped)}")

    if args.arrow_check:
        errors.extend(arrow_check(path, rows))

    print("=" * 70)
    print(f"[check-experiment] {path}")
    print(f"total:       {len(rows)}")
    print(f"counts:      {dict(counts)}")
    print(f"applied:     {dict(applied)}")
    print(f"skipped:     {dict(skipped)}")
    print(f"sources:     {dict(sources)}")
    print(f"max_frames:  {dict(max_frame_len)}")
    print(f"errors:      {len(errors)}")
    for error in errors[: args.max_errors]:
        print(f"  - {error}")
    if len(errors) > args.max_errors:
        print(f"  ... {len(errors) - args.max_errors} more")
    return 1 if errors else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", action="append", required=True, help="Experiment JSONL path; repeatable")
    parser.add_argument("--max-frames", type=int, default=256)
    parser.add_argument("--require-frame-policy", action="store_true")
    parser.add_argument("--expect-no-skipped", action="store_true")
    parser.add_argument("--check-frame-files", action="store_true", help="Check first/last frame path for each video")
    parser.add_argument("--arrow-check", action="store_true", help="Run pyarrow.Table.from_pylist on the loaded rows")
    parser.add_argument("--max-errors", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exit_code = 0
    for raw_path in args.jsonl:
        exit_code |= check_file(Path(raw_path), args)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
