#!/usr/bin/env python3
"""Check TG JSONL prompt/answer format and TimeLens selection metadata."""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from proxy_data.temporal_grounding.build_dataset import (  # noqa: E402
    PROMPT_TEMPLATE_NO_COT,
    format_answer_text,
)


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


def safe_pair(value: Any) -> tuple[float, float] | None:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            start = float(value[0])
            end = float(value[1])
        except (TypeError, ValueError):
            return None
        if start >= 0 and end > start:
            return start, end
    return None


def sentence_from(row: dict[str, Any]) -> str:
    meta = row.get("metadata") or {}
    for key in ("query", "sentence"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def is_timelens(row: dict[str, Any]) -> bool:
    meta = row.get("metadata") or {}
    values = [
        meta.get("dataset"),
        meta.get("dataset_name"),
        meta.get("curation_source"),
        meta.get("tg_base_training_source"),
    ]
    return any("timelens" in str(value).lower() for value in values if value is not None)


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
    timelens_min_iou: float,
    timelens_max_iou: float,
) -> list[str]:
    errors: list[str] = []
    where = f"{source_path}:{row.get('_check_line_no', '?')}"
    meta = row.get("metadata") or {}

    if row.get("problem_type") != "temporal_grounding":
        errors.append(f"{where}: problem_type != temporal_grounding")
    if row.get("data_type") != "video":
        errors.append(f"{where}: data_type != video")

    sentence = sentence_from(row)
    if not sentence:
        errors.append(f"{where}: missing metadata.sentence/query")

    timestamp = safe_pair(meta.get("timestamp"))
    if timestamp is None:
        errors.append(f"{where}: invalid metadata.timestamp")
    else:
        expected_answer = format_answer_text(timestamp[0], timestamp[1])
        if row.get("answer") != expected_answer:
            errors.append(f"{where}: answer mismatch; expected {expected_answer!r}")

    prompt = str(row.get("prompt") or "")
    if sentence and strict_prompt:
        expected_prompt = PROMPT_TEMPLATE_NO_COT.format(sentence=sentence)
        if prompt != expected_prompt:
            errors.append(f"{where}: prompt is not TG-Bench natural-language template")
    elif "Please find the visual event described by a sentence" not in prompt:
        errors.append(f"{where}: prompt missing TG-Bench preamble")

    messages = row.get("messages") or []
    if not messages or messages[0].get("content") != prompt:
        errors.append(f"{where}: messages[0].content does not match prompt")

    if is_timelens(row):
        mean_iou = meta.get("timelens_rollout_mean_iou")
        try:
            mean_iou_f = float(mean_iou)
        except (TypeError, ValueError):
            errors.append(f"{where}: TimeLens row missing timelens_rollout_mean_iou")
        else:
            if math.isnan(mean_iou_f) or mean_iou_f < timelens_min_iou or mean_iou_f > timelens_max_iou:
                errors.append(
                    f"{where}: TimeLens mean IoU {mean_iou_f} outside "
                    f"[{timelens_min_iou}, {timelens_max_iou}]"
                )

    if not row.get("videos"):
        errors.append(f"{where}: missing videos")
    return errors


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_dataset: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    timelens = 0
    for row in rows:
        meta = row.get("metadata") or {}
        by_dataset[str(meta.get("dataset") or meta.get("dataset_name") or "unknown")] += 1
        by_source[str(meta.get("source") or "unknown")] += 1
        if is_timelens(row):
            timelens += 1
    return {
        "records": len(rows),
        "timelens_records": timelens,
        "by_dataset": dict(by_dataset.most_common()),
        "by_source": dict(by_source.most_common()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check TG prompt/answer format and frame JSONL structure")
    parser.add_argument("--jsonl", action="append", default=[], help="Plain TG JSONL to check; repeatable")
    parser.add_argument("--jsonl-glob", action="append", default=[], help="Glob for plain TG JSONL files")
    parser.add_argument("--frames-jsonl", action="append", default=[], help="Frame-list TG JSONL to check; repeatable")
    parser.add_argument("--frames-glob", action="append", default=[], help="Glob for frame-list TG JSONL files")
    parser.add_argument("--summary-json", default="", help="Optional summary JSON path")
    parser.add_argument("--max-errors", type=int, default=50)
    parser.add_argument("--timelens-min-iou", type=float, default=0.1)
    parser.add_argument("--timelens-max-iou", type=float, default=0.4)
    parser.add_argument("--check-frame-files", action="store_true", help="Also check every frame path exists")
    parser.add_argument("--no-strict-prompt", action="store_true", help="Only require TG-Bench preamble, not exact prompt")
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
            errors.extend(check_record(row, path, strict_prompt, args.timelens_min_iou, args.timelens_max_iou))

    for path in frame_paths:
        rows = load_jsonl(path)
        summaries[str(path)] = summarize_rows(rows)
        for row in rows:
            errors.extend(check_record(row, path, strict_prompt, args.timelens_min_iou, args.timelens_max_iou))
            ok, reason = frame_lists_ok(row, args.check_frame_files)
            if not ok:
                errors.append(f"{path}:{row.get('_check_line_no', '?')}: {reason}")

    summary = {
        "checked_jsonl_files": [str(path) for path in jsonl_paths],
        "checked_frame_jsonl_files": [str(path) for path in frame_paths],
        "num_errors": len(errors),
        "summaries": summaries,
        "timelens_iou_range": [args.timelens_min_iou, args.timelens_max_iou],
    }
    if args.summary_json:
        out_path = Path(args.summary_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("==========================================")
    print(" TG format check")
    print(f" Plain JSONL files: {len(jsonl_paths)}")
    print(f" Frame JSONL files: {len(frame_paths)}")
    print(f" Errors:           {len(errors)}")
    if args.summary_json:
        print(f" Summary:          {args.summary_json}")
    print("==========================================")

    if errors:
        preview = "\n".join(errors[: args.max_errors])
        raise SystemExit(f"TG format check failed:\n{preview}")


if __name__ == "__main__":
    main()
