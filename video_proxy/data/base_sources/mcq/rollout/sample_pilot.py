#!/usr/bin/env python3
"""
Stratified pilot sampling from MCQ JSONL.

Samples up to N records per (duration_bucket, source) grid cell
for efficient pilot rollout evaluation.

Usage:
    python sample_pilot.py \
        --input mcq_all.jsonl \
        --output pilot_sample.jsonl \
        --per-cell 500 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Stratified pilot sampling for MCQ rollout")
    parser.add_argument("--input", required=True, help="Full MCQ JSONL from prepare_mcq.py")
    parser.add_argument("--output", required=True, help="Output sampled JSONL")
    parser.add_argument("--per-cell", type=int, default=500,
                        help="Max samples per (duration_bucket, source) cell")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Load and group by cell
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            meta = rec.get("metadata", {})
            key = (meta.get("duration_bucket", "unknown"), meta.get("source", "unknown"))
            cells[key].append(rec)

    # Sample
    sampled = []
    print(f"{'duration_bucket':>15s}  {'source':>20s}  {'total':>8s}  {'sampled':>8s}")
    print("-" * 60)
    for key in sorted(cells.keys()):
        pool = cells[key]
        rng.shuffle(pool)
        n = min(args.per_cell, len(pool))
        sampled.extend(pool[:n])
        bucket, source = key
        print(f"{bucket:>15s}  {source:>20s}  {len(pool):>8d}  {n:>8d}")

    # Shuffle final output
    rng.shuffle(sampled)

    # Write
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for rec in sampled:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nTotal sampled: {len(sampled)} from {len(cells)} cells")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
