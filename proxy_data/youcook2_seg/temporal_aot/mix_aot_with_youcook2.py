#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mix YouCook2 temporal segmentation data with Temporal AoT V2T/T2V data.

Default behavior:
- sample 400 train records from each source
- sample 30 val records from each source
- keep train/val disjoint within each source
- assign problem_type="temporal_seg" to YouCook2 records when missing

Example:
python proxy_data/temporal_aot/mix_aot_with_youcook2.py \
  --seg-jsonl proxy_data/youcook2_train_easyr1.jsonl \
  --v2t-jsonl proxy_data/temporal_aot/data/v2t_train.jsonl \
  --t2v-jsonl proxy_data/temporal_aot/data/t2v_train.jsonl \
  --train-output proxy_data/temporal_aot/data/mixed_aot_train.jsonl \
  --val-output proxy_data/temporal_aot/data/mixed_aot_val.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter


def load_jsonl(path: str) -> list[dict]:
    items: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[WARN] Skip invalid JSON at {path}:{line_no}: {exc}")
    return items


def normalize_problem_type(records: list[dict], default_problem_type: str) -> list[dict]:
    normalized: list[dict] = []
    for item in records:
        copied = dict(item)
        if not copied.get("problem_type") and default_problem_type:
            copied["problem_type"] = default_problem_type
        normalized.append(copied)
    return normalized


def sample_train_val(records: list[dict], train_n: int, val_n: int, seed: int) -> tuple[list[dict], list[dict]]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)

    actual_val_n = min(val_n, len(shuffled))
    val_records = shuffled[:actual_val_n]

    remaining = shuffled[actual_val_n:]
    actual_train_n = min(train_n, len(remaining))
    train_records = remaining[:actual_train_n]
    return train_records, val_records


def write_jsonl(path: str, records: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def add_source_tag(records: list[dict], source_name: str) -> list[dict]:
    tagged: list[dict] = []
    for record in records:
        copied = dict(record)
        metadata = dict(copied.get("metadata") or {})
        metadata["mix_source"] = source_name
        copied["metadata"] = metadata
        tagged.append(copied)
    return tagged


def print_split_stats(name: str, total: int, train_records: list[dict], val_records: list[dict]) -> None:
    print(
        f"{name}: total={total}, "
        f"train={len(train_records)}, "
        f"val={len(val_records)}, "
        f"unused={max(total - len(train_records) - len(val_records), 0)}"
    )


def print_problem_type_stats(name: str, records: list[dict]) -> None:
    counts = Counter(record.get("problem_type", "(empty)") for record in records)
    joined = ", ".join(f"{task}={count}" for task, count in sorted(counts.items()))
    print(f"{name} problem types: {joined}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mix YouCook2 temporal_seg with AoT V2T/T2V JSONL data.")
    parser.add_argument("--seg-jsonl", required=True, help="Input YouCook2 temporal segmentation JSONL")
    parser.add_argument("--v2t-jsonl", default="", help="Input AoT V2T JSONL (optional)")
    parser.add_argument("--t2v-jsonl", default="", help="Input AoT T2V JSONL (optional)")
    parser.add_argument("--train-output", required=True, help="Output mixed train JSONL")
    parser.add_argument("--val-output", required=True, help="Output mixed val JSONL")
    parser.add_argument("--train-per-source", type=int, default=400, help="Train samples to keep from each source")
    parser.add_argument("--val-per-source", type=int, default=30, help="Val samples to keep from each source")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    seg_records = normalize_problem_type(load_jsonl(args.seg_jsonl), default_problem_type="temporal_seg")
    v2t_records = normalize_problem_type(load_jsonl(args.v2t_jsonl) if args.v2t_jsonl else [], default_problem_type="aot_v2t")
    t2v_records = normalize_problem_type(load_jsonl(args.t2v_jsonl) if args.t2v_jsonl else [], default_problem_type="aot_t2v")

    seg_train, seg_val = sample_train_val(seg_records, args.train_per_source, args.val_per_source, args.seed + 11)
    v2t_train, v2t_val = sample_train_val(v2t_records, args.train_per_source, args.val_per_source, args.seed + 22)
    t2v_train, t2v_val = sample_train_val(t2v_records, args.train_per_source, args.val_per_source, args.seed + 33)

    seg_train = add_source_tag(seg_train, "youcook2_temporal_seg")
    seg_val = add_source_tag(seg_val, "youcook2_temporal_seg")
    v2t_train = add_source_tag(v2t_train, "temporal_aot_v2t")
    v2t_val = add_source_tag(v2t_val, "temporal_aot_v2t")
    t2v_train = add_source_tag(t2v_train, "temporal_aot_t2v")
    t2v_val = add_source_tag(t2v_val, "temporal_aot_t2v")

    mixed_train = seg_train + v2t_train + t2v_train
    mixed_val = seg_val + v2t_val + t2v_val
    random.Random(args.seed + 101).shuffle(mixed_train)
    random.Random(args.seed + 202).shuffle(mixed_val)

    write_jsonl(args.train_output, mixed_train)
    write_jsonl(args.val_output, mixed_val)

    print("=== Source Split Stats ===")
    print_split_stats("youcook2_temporal_seg", len(seg_records), seg_train, seg_val)
    print_split_stats("temporal_aot_v2t", len(v2t_records), v2t_train, v2t_val)
    print_split_stats("temporal_aot_t2v", len(t2v_records), t2v_train, t2v_val)

    print("\n=== Mixed Output Stats ===")
    print(f"train: {len(mixed_train)} -> {args.train_output}")
    print_problem_type_stats("train", mixed_train)
    print(f"val:   {len(mixed_val)} -> {args.val_output}")
    print_problem_type_stats("val", mixed_val)


if __name__ == "__main__":
    main()
