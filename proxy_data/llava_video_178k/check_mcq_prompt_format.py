#!/usr/bin/env python3
"""Check LLaVA MCQ JSONL prompt/answer format and frame JSONL structure."""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from proxy_data.llava_video_178k.convert_mcq_to_direct import DIRECT_INSTRUCTION  # noqa: E402


_ANSWER_LETTER = re.compile(r"^[A-Z]$")


def expand_paths(paths: list[str], globs: list[str]) -> list[Path]:
    expanded: list[Path] = []
    for value in list(paths or []) + list(globs or []):
        if any(ch in value for ch in "*?["):
            expanded.extend(Path(p) for p in sorted(glob.glob(value)))
        else:
            expanded.append(Path(value))
    unique: dict[str, Path] = {}
    for path in expanded:
        unique[str(path)] = path
    return list(unique.values())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Failed to parse {path}:{line_no}: {exc}") from exc
            row["_check_line_no"] = line_no
            rows.append(row)
    return rows


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def is_low_reward_llava(row: dict[str, Any]) -> bool:
    meta = row.get("metadata") or {}
    values = [
        meta.get("mcq_base_training_source"),
        meta.get("curation_source"),
        meta.get("dataset"),
    ]
    if "llava_rollout_mean_reward" in meta:
        return True
    return any("llava_rollout" in str(value).lower() for value in values if value is not None)


def frame_lists_ok(row: dict[str, Any], check_files: bool) -> tuple[bool, str]:
    videos = row.get("videos") or []
    if not videos:
        return False, "missing videos"
    if not isinstance(videos[0], list):
        return False, "videos[0] is not a frame list"
    for vid_idx, frames in enumerate(videos):
        if not isinstance(frames, list) or not frames:
            return False, f"videos[{vid_idx}] empty or not a list"
        if check_files:
            for frame in frames:
                if not Path(str(frame)).exists():
                    return False, f"missing frame file: {frame}"
    meta = row.get("metadata") or {}
    if "video_fps_override" not in meta:
        return False, "missing metadata.video_fps_override"
    if "offline_frame_extraction" not in meta:
        return False, "missing metadata.offline_frame_extraction"
    return True, ""


def check_record(
    row: dict[str, Any],
    source_path: Path,
    strict_prompt: bool,
    min_mean_reward: float,
    max_mean_reward: float,
) -> list[str]:
    errors: list[str] = []
    where = f"{source_path}:{row.get('_check_line_no', '?')}"
    meta = row.get("metadata") or {}

    if row.get("problem_type") != "llava_mcq":
        errors.append(f"{where}: problem_type != llava_mcq")
    if row.get("data_type") != "video":
        errors.append(f"{where}: data_type != video")

    answer = str(row.get("answer") or "").strip().upper()
    if not _ANSWER_LETTER.match(answer):
        errors.append(f"{where}: answer is not a single uppercase letter")

    prompt = str(row.get("prompt") or "")
    if "<video>" not in prompt:
        errors.append(f"{where}: prompt missing <video>")
    if strict_prompt and DIRECT_INSTRUCTION not in prompt:
        errors.append(f"{where}: prompt is not direct-answer MCQ template")

    messages = row.get("messages") or []
    if not messages or messages[0].get("content") != prompt:
        errors.append(f"{where}: messages[0].content does not match prompt")

    if is_low_reward_llava(row):
        mean_reward = safe_float(meta.get("llava_rollout_mean_reward"))
        if math.isnan(mean_reward):
            errors.append(f"{where}: low-reward row missing llava_rollout_mean_reward")
        elif mean_reward < min_mean_reward or mean_reward > max_mean_reward:
            errors.append(
                f"{where}: low-reward mean {mean_reward} outside "
                f"[{min_mean_reward}, {max_mean_reward}]"
            )

    if not row.get("videos"):
        errors.append(f"{where}: missing videos")
    return errors


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_source: Counter[str] = Counter()
    by_duration_bucket: Counter[str] = Counter()
    low_reward = 0
    for row in rows:
        meta = row.get("metadata") or {}
        by_source[str(meta.get("source") or "unknown")] += 1
        by_duration_bucket[str(meta.get("duration_bucket") or "unknown")] += 1
        if is_low_reward_llava(row):
            low_reward += 1
    return {
        "records": len(rows),
        "low_reward_records": low_reward,
        "by_source": dict(by_source.most_common()),
        "by_duration_bucket": dict(by_duration_bucket.most_common()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check LLaVA MCQ prompt/answer and frame JSONL structure")
    parser.add_argument("--jsonl", action="append", default=[], help="Plain MCQ JSONL to check; repeatable")
    parser.add_argument("--jsonl-glob", action="append", default=[], help="Glob for plain MCQ JSONL files")
    parser.add_argument("--frames-jsonl", action="append", default=[], help="Frame-list MCQ JSONL to check; repeatable")
    parser.add_argument("--frames-glob", action="append", default=[], help="Glob for frame-list MCQ JSONL files")
    parser.add_argument("--summary-json", default="", help="Optional summary JSON path")
    parser.add_argument("--max-errors", type=int, default=50)
    parser.add_argument("--min-mean-reward", type=float, default=0.0)
    parser.add_argument("--max-mean-reward", type=float, default=0.375)
    parser.add_argument("--check-frame-files", action="store_true", help="Also check every frame path exists")
    parser.add_argument("--no-strict-prompt", action="store_true", help="Do not require the direct-answer instruction")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jsonl_paths = expand_paths(args.jsonl, args.jsonl_glob)
    frame_paths = expand_paths(args.frames_jsonl, args.frames_glob)
    all_paths = jsonl_paths + frame_paths
    if not all_paths:
        raise SystemExit("No files to check. Pass --jsonl/--jsonl-glob and/or --frames-jsonl/--frames-glob.")

    missing = [str(path) for path in all_paths if not path.is_file()]
    if missing:
        raise SystemExit("Missing check input file(s):\n" + "\n".join(missing))

    errors: list[str] = []
    summaries: dict[str, Any] = {}
    strict_prompt = not args.no_strict_prompt

    for path in jsonl_paths:
        rows = load_jsonl(path)
        summaries[str(path)] = summarize_rows(rows)
        for row in rows:
            errors.extend(check_record(row, path, strict_prompt, args.min_mean_reward, args.max_mean_reward))

    for path in frame_paths:
        rows = load_jsonl(path)
        summaries[str(path)] = summarize_rows(rows)
        for row in rows:
            errors.extend(check_record(row, path, strict_prompt, args.min_mean_reward, args.max_mean_reward))
            ok, reason = frame_lists_ok(row, args.check_frame_files)
            if not ok:
                errors.append(f"{path}:{row.get('_check_line_no', '?')}: {reason}")

    summary = {
        "checked_jsonl_files": [str(path) for path in jsonl_paths],
        "checked_frame_jsonl_files": [str(path) for path in frame_paths],
        "num_errors": len(errors),
        "summaries": summaries,
        "mean_reward_range": [args.min_mean_reward, args.max_mean_reward],
    }
    if args.summary_json:
        out_path = Path(args.summary_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("==========================================")
    print(" LLaVA MCQ format check")
    print(f" Plain JSONL files: {len(jsonl_paths)}")
    print(f" Frame JSONL files: {len(frame_paths)}")
    print(f" Errors:           {len(errors)}")
    if args.summary_json:
        print(f" Summary:          {args.summary_json}")
    print("==========================================")

    if errors:
        preview = "\n".join(errors[: args.max_errors])
        raise SystemExit(f"LLaVA MCQ format check failed:\n{preview}")


if __name__ == "__main__":
    main()
