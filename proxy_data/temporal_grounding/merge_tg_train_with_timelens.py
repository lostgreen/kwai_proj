#!/usr/bin/env python3
"""Merge base TimeRFT TG data with selected TimeLens TG samples."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Failed to parse {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def norm_float(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return ""


def first_video_key(record: dict[str, Any]) -> str:
    videos = record.get("videos") or []
    if not videos:
        return ""
    first = videos[0]
    if isinstance(first, list):
        return str(first[0]) if first else ""
    return str(first)


def dedupe_key(record: dict[str, Any]) -> tuple[str, str, str, str, str]:
    meta = record.get("metadata") or {}
    dataset = str(meta.get("dataset") or meta.get("dataset_name") or meta.get("source") or "unknown")
    video_uid = str(meta.get("video_uid") or meta.get("video_id") or meta.get("clip_key") or first_video_key(record))
    sentence = str(meta.get("query") or meta.get("sentence") or record.get("query") or "").strip().lower()
    timestamp = meta.get("timestamp") or []
    start = norm_float(timestamp[0]) if isinstance(timestamp, list) and len(timestamp) >= 2 else ""
    end = norm_float(timestamp[1]) if isinstance(timestamp, list) and len(timestamp) >= 2 else ""
    return dataset, video_uid, sentence, start, end


def tag_record(record: dict[str, Any], training_source: str, dataset: str | None) -> dict[str, Any]:
    out = dict(record)
    meta = dict(out.get("metadata") or {})
    meta["tg_base_training_source"] = training_source
    if dataset and not meta.get("dataset"):
        meta["dataset"] = dataset
    out["metadata"] = meta
    out["data_type"] = out.get("data_type") or "video"
    out["problem_type"] = out.get("problem_type") or "temporal_grounding"
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_source: Counter[str] = Counter()
    by_dataset: Counter[str] = Counter()
    by_training_source: Counter[str] = Counter()
    videos: set[str] = set()
    for row in rows:
        meta = row.get("metadata") or {}
        by_source[str(meta.get("source") or "unknown")] += 1
        by_dataset[str(meta.get("dataset") or meta.get("dataset_name") or "unknown")] += 1
        by_training_source[str(meta.get("tg_base_training_source") or "unknown")] += 1
        video_uid = str(meta.get("video_uid") or meta.get("video_id") or meta.get("clip_key") or first_video_key(row))
        if video_uid:
            videos.add(video_uid)
    return {
        "records": len(rows),
        "unique_videos": len(videos),
        "by_source": dict(by_source.most_common()),
        "by_dataset": dict(by_dataset.most_common()),
        "by_training_source": dict(by_training_source.most_common()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge rewritten TimeRFT TG with selected TimeLens TG")
    parser.add_argument("--base", required=True, help="Rewritten TimeRFT/base TG JSONL")
    parser.add_argument("--timelens", required=True, help="Selected TimeLens TG JSONL")
    parser.add_argument("--output", required=True, help="Merged TG train JSONL")
    parser.add_argument("--summary-json", default="", help="Optional summary JSON path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-shuffle", action="store_true", help="Keep base then TimeLens order")
    parser.add_argument(
        "--duplicate-policy",
        choices=["keep-first", "keep-all"],
        default="keep-first",
        help="How to handle duplicate dataset/video/query/timestamp keys",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_path = Path(args.base)
    timelens_path = Path(args.timelens)
    output_path = Path(args.output)
    summary_path = Path(args.summary_json) if args.summary_json else output_path.with_suffix(".summary.json")

    if not base_path.is_file():
        raise SystemExit(f"Base TG file not found: {base_path}")
    if not timelens_path.is_file():
        raise SystemExit(f"TimeLens TG file not found: {timelens_path}")

    base_rows = [tag_record(row, "timerft_base", "TimeRFT") for row in load_jsonl(base_path)]
    timelens_rows = [tag_record(row, "timelens_iou_selected", "TimeLens-100K") for row in load_jsonl(timelens_path)]

    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    duplicates = 0
    for row in base_rows + timelens_rows:
        key = dedupe_key(row)
        if args.duplicate_policy == "keep-first" and key in seen:
            duplicates += 1
            continue
        seen.add(key)
        merged.append(row)

    if not args.no_shuffle:
        random.Random(args.seed).shuffle(merged)

    write_jsonl(output_path, merged)
    summary = {
        "base_input": str(base_path),
        "timelens_input": str(timelens_path),
        "output": str(output_path),
        "base_records": len(base_rows),
        "timelens_records": len(timelens_rows),
        "merged_records": len(merged),
        "duplicates_skipped": duplicates,
        "merged_summary": summarize(merged),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("==========================================")
    print(" TG train merge done")
    print(f" Base:        {base_path} ({len(base_rows)} records)")
    print(f" TimeLens:    {timelens_path} ({len(timelens_rows)} records)")
    print(f" Merged:      {output_path} ({len(merged)} records)")
    print(f" Duplicates:  {duplicates}")
    print(f" Summary:     {summary_path}")
    print("==========================================")


if __name__ == "__main__":
    main()
