#!/usr/bin/env python3
"""
sample_mixed_dataset.py — Sample balanced train/val splits from all 3 hierarchy levels.

- Strips metadata field (not needed by EasyR1)
- Samples equal records from each level
- Shuffles and writes combined JSONL files

Usage:
    python sample_mixed_dataset.py \
        --input-dir datasets \
        --output-dir datasets \
        --train-n 1000 \
        --val-n 100 \
        --seed 42
"""

import argparse
import json
import random
from pathlib import Path


LEVEL_FILES = {
    1: "youcook2_hier_L1_train_clipped.jsonl",
    2: "youcook2_hier_L2_train_clipped.jsonl",
    3: "youcook2_hier_L3_train_clipped.jsonl",
}

# Only these keys are consumed by EasyR1
KEEP_KEYS = {"messages", "prompt", "answer", "videos", "data_type", "problem_type"}


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def strip_metadata(record: dict) -> dict:
    return {k: v for k, v in record.items() if k in KEEP_KEYS}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample balanced mixed-level dataset for EasyR1 training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", default="datasets",
                        help="Directory containing the per-level JSONL files")
    parser.add_argument("--output-dir", default="datasets",
                        help="Directory to write the mixed JSONL files")
    parser.add_argument("--train-n", type=int, default=1000,
                        help="Total number of training records")
    parser.add_argument("--val-n", type=int, default=100,
                        help="Total number of validation records")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_levels = len(LEVEL_FILES)
    # Equal split across levels; remainder goes to the last level
    train_base = args.train_n // n_levels
    val_base   = args.val_n   // n_levels
    train_rem  = args.train_n - train_base * n_levels
    val_rem    = args.val_n   - val_base   * n_levels

    all_train: list[dict] = []
    all_val:   list[dict] = []

    for i, (level, filename) in enumerate(sorted(LEVEL_FILES.items())):
        is_last = (i == n_levels - 1)
        t_n = train_base + (train_rem if is_last else 0)
        v_n = val_base   + (val_rem   if is_last else 0)

        path = input_dir / filename
        if not path.exists():
            print(f"  SKIP L{level}: {path} not found")
            continue

        records = load_jsonl(path)
        rng.shuffle(records)
        print(f"  L{level}: {len(records)} records available, need train={t_n} val={v_n}")

        total_need = t_n + v_n
        if len(records) < total_need:
            print(f"    WARNING: not enough records ({len(records)} < {total_need}), "
                  f"adjusting proportionally")
            v_n = max(1, round(len(records) * v_n / total_need))
            t_n = len(records) - v_n

        val_records   = records[:v_n]
        train_records = records[v_n: v_n + t_n]

        print(f"    → train={len(train_records)}, val={len(val_records)}")
        all_train.extend(strip_metadata(r) for r in train_records)
        all_val.extend(strip_metadata(r)   for r in val_records)

    # Final shuffle of combined splits
    rng.shuffle(all_train)
    rng.shuffle(all_val)

    train_path = output_dir / "youcook2_hier_mixed_train.jsonl"
    val_path   = output_dir / "youcook2_hier_mixed_val.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for r in all_train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for r in all_val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nOutput:")
    print(f"  Train : {train_path}  ({len(all_train)} records)")
    print(f"  Val   : {val_path}  ({len(all_val)} records)")


if __name__ == "__main__":
    main()
