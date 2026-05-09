#!/usr/bin/env python3
"""Validate hier-seg shared-frame manifests for phase-crop frame ablations."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

HIER_L1 = "temporal_seg_hier_L1"
HIER_L2 = "temporal_seg_hier_L2"
HIER_L3 = "temporal_seg_hier_L3_seg"


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


def as_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def close(a: float | None, b: float | None, tol: float = 1e-3) -> bool:
    return a is not None and b is not None and abs(a - b) <= tol


def check_record(path: Path, line_no: int, row: dict[str, Any], check_files: bool) -> list[str]:
    errors: list[str] = []
    problem_type = row.get("problem_type")
    meta = row.get("metadata") or {}
    shared = meta.get("shared_source_frames")
    videos = row.get("videos")

    if problem_type not in {HIER_L1, HIER_L2, HIER_L3}:
        errors.append(f"{path}:{line_no}: unexpected problem_type={problem_type!r}")
        return errors
    if not isinstance(shared, dict):
        errors.append(f"{path}:{line_no}: missing metadata.shared_source_frames")
        return errors
    if not isinstance(videos, list) or not videos or not isinstance(videos[0], list) or not videos[0]:
        errors.append(f"{path}:{line_no}: videos is not a non-empty frame-list")
        return errors

    start = as_float(shared.get("segment_start_sec"))
    end = as_float(shared.get("segment_end_sec"))
    if start is None or end is None or end <= start:
        errors.append(f"{path}:{line_no}: invalid shared segment [{start}, {end}]")

    if problem_type == HIER_L1:
        if not close(start, 0.0):
            errors.append(f"{path}:{line_no}: L1 shared segment must start at 0, got {start}")

    if problem_type == HIER_L2:
        if meta.get("l2_mode") == "full":
            errors.append(f"{path}:{line_no}: L2 is still full-video mode")
        phase_start = as_float(meta.get("phase_start_sec"))
        phase_end = as_float(meta.get("phase_end_sec"))
        if phase_start is None or phase_end is None or phase_end <= phase_start:
            errors.append(f"{path}:{line_no}: L2 missing valid phase_start_sec/phase_end_sec")
        if not close(start, phase_start) or not close(end, phase_end):
            errors.append(
                f"{path}:{line_no}: L2 shared span [{start}, {end}] "
                f"does not match phase span [{phase_start}, {phase_end}]"
            )

    if problem_type == HIER_L3:
        clip_start = as_float(meta.get("clip_start_sec"))
        clip_end = as_float(meta.get("clip_end_sec"))
        if clip_start is None or clip_end is None or clip_end <= clip_start:
            errors.append(f"{path}:{line_no}: L3 missing valid clip_start_sec/clip_end_sec")
        if not close(start, clip_start) or not close(end, clip_end):
            errors.append(
                f"{path}:{line_no}: L3 shared span [{start}, {end}] "
                f"does not match clip span [{clip_start}, {clip_end}]"
            )

    if check_files:
        for frame in (videos[0][0], videos[0][-1]):
            if not Path(frame).is_file():
                errors.append(f"{path}:{line_no}: frame file missing: {frame}")

    return errors


def check_file(path: Path, args: argparse.Namespace) -> int:
    if not path.is_file():
        print(f"[check-hier-phasecrop] missing: {path}", file=sys.stderr)
        return 1

    counts: Counter[str] = Counter()
    durations: dict[str, list[float]] = {HIER_L1: [], HIER_L2: [], HIER_L3: []}
    errors: list[str] = []

    try:
        for line_no, row in iter_jsonl(path):
            if not isinstance(row, dict):
                errors.append(f"{path}:{line_no}: row is not a JSON object")
                continue
            problem_type = str(row.get("problem_type") or "unknown")
            counts[problem_type] += 1
            meta = row.get("metadata") or {}
            shared = meta.get("shared_source_frames") or {}
            start = as_float(shared.get("segment_start_sec"))
            end = as_float(shared.get("segment_end_sec"))
            if start is not None and end is not None and end > start and problem_type in durations:
                durations[problem_type].append(end - start)
            errors.extend(check_record(path, line_no, row, args.check_frame_files))
    except ValueError as exc:
        print(f"[check-hier-phasecrop] {exc}", file=sys.stderr)
        return 1

    for required in (HIER_L1, HIER_L2, HIER_L3):
        if counts.get(required, 0) == 0:
            errors.append(f"{path}: no records for {required}")

    print("=" * 70)
    print(f"[check-hier-phasecrop] {path}")
    print(f"counts: {dict(counts)}")
    for problem_type, values in durations.items():
        if values:
            values_sorted = sorted(values)
            mean = sum(values) / len(values)
            p50 = values_sorted[len(values_sorted) // 2]
            print(f"{problem_type}: n={len(values)} mean={mean:.1f}s p50={p50:.1f}s max={max(values):.1f}s")
    print(f"errors: {len(errors)}")
    for error in errors[: args.max_errors]:
        print(f"  - {error}")
    if len(errors) > args.max_errors:
        print(f"  ... {len(errors) - args.max_errors} more")
    return 1 if errors else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", action="append", required=True, help="Hier shared-frame JSONL; repeatable")
    parser.add_argument("--check-frame-files", action="store_true")
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
