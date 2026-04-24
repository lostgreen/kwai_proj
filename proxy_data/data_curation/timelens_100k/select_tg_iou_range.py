#!/usr/bin/env python3
"""Select TimeLens TG rollout samples by query-level mean IoU.

This joins the query-level rollout input with ``analysis/query_stats.jsonl`` and
keeps records whose mean IoU falls in the requested inclusive range.

Usage from train/:
    python proxy_data/data_curation/timelens_100k/select_tg_iou_range.py \
        --input-jsonl proxy_data/data_curation/results/timelens_100k_short/tg_rollout_qwen3_vl_8b_roll8/tg_rollout_input.jsonl \
        --query-stats proxy_data/data_curation/results/timelens_100k_short/tg_rollout_qwen3_vl_8b_roll8/analysis/query_stats.jsonl \
        --output-jsonl proxy_data/temporal_grounding/data/tg_timelens_iou0p1_0p4.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from proxy_data.temporal_grounding.build_dataset import (  # noqa: E402
    PROMPT_TEMPLATE_NO_COT,
    format_answer_text,
)


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


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_stats(path: Path) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        query_id = str(row.get("query_id") or row.get("metadata_id") or "")
        if not query_id:
            continue
        stats[query_id] = row
    return stats


def query_id_of(record: dict[str, Any]) -> str:
    meta = record.get("metadata") or {}
    return str(meta.get("id") or record.get("metadata_id") or "")


def annotate_selected_record(
    record: dict[str, Any],
    stat: dict[str, Any],
    min_iou: float,
    max_iou: float,
) -> dict[str, Any]:
    out = dict(record)
    meta = dict(out.get("metadata") or {})
    sentence = str(meta.get("query") or meta.get("sentence") or "").strip()
    timestamp = meta.get("timestamp") or []
    if sentence and isinstance(timestamp, list) and len(timestamp) == 2:
        try:
            start = float(timestamp[0])
            end = float(timestamp[1])
            if start >= 0 and end > start:
                prompt = PROMPT_TEMPLATE_NO_COT.format(sentence=sentence)
                out["prompt"] = prompt
                out["messages"] = [{"role": "user", "content": prompt}]
                out["answer"] = format_answer_text(start, end)
                meta["sentence"] = sentence
                meta.setdefault("query", sentence)
                meta["timestamp"] = [round(start, 2), round(end, 2)]
                meta["prompt_format"] = "tgbench_natural_language_v1"
                meta["answer_format"] = "tgbench_natural_language_v1"
        except (TypeError, ValueError):
            pass
    meta["curation_source"] = "timelens_short_qwen3vl_roll8"
    meta["timelens_selection_iou_min"] = min_iou
    meta["timelens_selection_iou_max"] = max_iou
    meta["timelens_rollout_mean_iou"] = safe_float(stat.get("mean_iou"))
    meta["timelens_rollout_std_iou"] = safe_float(stat.get("std_iou"))
    meta["timelens_rollout_min_iou"] = safe_float(stat.get("min_iou"))
    meta["timelens_rollout_max_iou"] = safe_float(stat.get("max_iou"))
    meta["timelens_rollout_num_rollouts"] = int(stat.get("num_rollouts") or 0)
    meta.setdefault("dataset", "TimeLens-100K")
    out["metadata"] = meta
    out["data_type"] = out.get("data_type") or "video"
    out["problem_type"] = out.get("problem_type") or "temporal_grounding"
    return out


def build_summary(rows: list[dict[str, Any]], selected: list[dict[str, Any]], missing_stats: int) -> dict[str, Any]:
    source_counter: Counter[str] = Counter()
    duration_counter: Counter[str] = Counter()
    video_uids: set[str] = set()
    for row in selected:
        meta = row.get("metadata") or {}
        source_counter[str(meta.get("source") or "unknown")] += 1
        duration_counter[str(meta.get("duration_bucket") or "unknown")] += 1
        video_uid = str(meta.get("video_uid") or meta.get("video_id") or meta.get("clip_key") or "")
        if video_uid:
            video_uids.add(video_uid)
    return {
        "input_queries": len(rows),
        "selected_queries": len(selected),
        "selected_videos": len(video_uids),
        "missing_stats": missing_stats,
        "selected_by_source": dict(source_counter.most_common()),
        "selected_by_duration_bucket": dict(duration_counter.most_common()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select TimeLens TG samples by mean IoU range")
    parser.add_argument("--input-jsonl", required=True, help="TimeLens query-level rollout input JSONL")
    parser.add_argument("--query-stats", required=True, help="analysis/query_stats.jsonl produced by analyze_tg_rollout.py")
    parser.add_argument("--output-jsonl", required=True, help="Selected TimeLens TG JSONL")
    parser.add_argument("--summary-json", default="", help="Optional summary JSON path")
    parser.add_argument("--min-iou", type=float, default=0.1, help="Inclusive lower mean-IoU bound")
    parser.add_argument("--max-iou", type=float, default=0.4, help="Inclusive upper mean-IoU bound")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional cap after filtering; 0 keeps all")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    stats_path = Path(args.query_stats)
    output_path = Path(args.output_jsonl)
    summary_path = Path(args.summary_json) if args.summary_json else output_path.with_suffix(".summary.json")

    if args.min_iou > args.max_iou:
        raise SystemExit(f"--min-iou ({args.min_iou}) cannot exceed --max-iou ({args.max_iou})")
    if not input_path.is_file():
        raise SystemExit(f"Input JSONL not found: {input_path}")
    if not stats_path.is_file():
        raise SystemExit(f"Query stats not found: {stats_path}")

    rows = load_jsonl(input_path)
    stats = load_stats(stats_path)
    selected: list[dict[str, Any]] = []
    missing_stats = 0

    for row in rows:
        query_id = query_id_of(row)
        stat = stats.get(query_id)
        if stat is None:
            missing_stats += 1
            continue
        mean_iou = safe_float(stat.get("mean_iou"))
        if mean_iou is None:
            continue
        if args.min_iou <= mean_iou <= args.max_iou:
            selected.append(annotate_selected_record(row, stat, args.min_iou, args.max_iou))

    if args.max_samples > 0 and len(selected) > args.max_samples:
        rng = random.Random(args.seed)
        selected = rng.sample(selected, args.max_samples)
        selected.sort(key=lambda row: query_id_of(row))

    write_jsonl(output_path, selected)
    summary = build_summary(rows, selected, missing_stats)
    summary["iou_range"] = [args.min_iou, args.max_iou]
    summary["query_stats"] = str(stats_path)
    summary["input_jsonl"] = str(input_path)
    summary["output_jsonl"] = str(output_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("==========================================")
    print(" TimeLens TG IoU-range selection")
    print(f" Input queries:    {len(rows)}")
    print(f" Stats rows:       {len(stats)}")
    print(f" IoU range:        [{args.min_iou}, {args.max_iou}]")
    print(f" Selected queries: {len(selected)}")
    print(f" Selected videos:  {summary['selected_videos']}")
    print(f" Missing stats:    {missing_stats}")
    print(f" Output:           {output_path}")
    print(f" Summary:          {summary_path}")
    print("==========================================")


if __name__ == "__main__":
    main()
