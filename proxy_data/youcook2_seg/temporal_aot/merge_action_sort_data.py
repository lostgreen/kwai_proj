#!/usr/bin/env python3
"""Merge action-AoT and sort-direction JSONL files for rollout/training.

Typical usage:
    python proxy_data/youcook2_seg/temporal_aot/merge_action_sort_data.py \
        --train-input seg_action/train.jsonl sort_direction/train.jsonl \
        --val-input seg_action/val.jsonl sort_direction/val.jsonl \
        --output-dir merged_aot_direction
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: str) -> list[dict]:
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def merge_files(paths: list[str], seed: int) -> list[dict]:
    merged: list[dict] = []
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    for path in paths:
        for rec in load_jsonl(path):
            key = (
                rec.get("prompt", ""),
                rec.get("answer", ""),
                tuple(rec.get("videos") or []),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(rec)
    random.Random(seed).shuffle(merged)
    return merged


def summarize(records: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for rec in records:
        counts[str(rec.get("problem_type", "unknown"))] += 1
    return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge multiple AoT/direction JSONL files into combined train/val splits",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-input", nargs="+", required=True)
    parser.add_argument("--val-input", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_records = merge_files(args.train_input, args.seed)
    val_records = merge_files(args.val_input, args.seed + 1)

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    stats_path = output_dir / "stats.json"

    write_jsonl(train_records, str(train_path))
    write_jsonl(val_records, str(val_path))

    stats = {
        "train_total": len(train_records),
        "val_total": len(val_records),
        "train_inputs": args.train_input,
        "val_inputs": args.val_input,
        "train_by_problem_type": summarize(train_records),
        "val_by_problem_type": summarize(val_records),
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"Train: {len(train_records)} -> {train_path}")
    print(f"Val:   {len(val_records)} -> {val_path}")
    print(f"Stats: {stats_path}")
    print("Train by problem_type:")
    for task, count in stats["train_by_problem_type"].items():
        print(f"  {task}: {count}")
    print("Val by problem_type:")
    for task, count in stats["val_by_problem_type"].items():
        print(f"  {task}: {count}")


if __name__ == "__main__":
    main()
