#!/usr/bin/env python3
"""
Post-rollout filtering and balanced downsampling.

Reads the rollout report JSONL from offline_rollout_filter.py, filters QA pairs
by mean accuracy range [min_acc, max_acc], then balanced-downsamples across
(duration_bucket, source) grid to reach the target total count.

Usage:
    python filter_and_downsample.py \
        --report rollout_report.jsonl \
        --input mcq_all.jsonl \
        --output final_train.jsonl \
        --min-acc 0.25 --max-acc 0.5 \
        --target-total 1000 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: str) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def main():
    parser = argparse.ArgumentParser(
        description="Filter by rollout accuracy range + balanced downsample"
    )
    parser.add_argument("--report", required=True,
                        help="Rollout report JSONL from offline_rollout_filter.py")
    parser.add_argument("--input", required=True,
                        help="Original MCQ JSONL (to retrieve full records for kept indices)")
    parser.add_argument("--output", required=True,
                        help="Output JSONL for RL training")
    parser.add_argument("--min-acc", type=float, default=0.25,
                        help="Minimum mean accuracy to keep (inclusive, closed interval)")
    parser.add_argument("--max-acc", type=float, default=0.5,
                        help="Maximum mean accuracy to keep (inclusive, closed interval)")
    parser.add_argument("--target-total", type=int, default=1000,
                        help="Target total number of records after downsampling "
                             "(0 = keep all filtered records)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stats-only", action="store_true",
                        help="Only print accuracy stats without producing output")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Load report
    print(f"Loading report: {args.report}")
    reports = load_jsonl(args.report)
    print(f"  {len(reports)} records in report")

    # Load original input — build content-based lookup
    print(f"Loading input: {args.input}")
    originals = load_jsonl(args.input)
    print(f"  {len(originals)} original records")

    # Content-based lookup: (prompt, answer) → original record
    originals_by_content: dict[tuple[str, str], dict] = {}
    for rec in originals:
        key = (rec.get("prompt", ""), rec.get("answer", ""))
        originals_by_content[key] = rec

    def _lookup_original(report_entry: dict) -> dict | None:
        """Find original record matching a report entry (content-based, index fallback)."""
        # Content match first
        key = (report_entry.get("prompt", ""), report_entry.get("answer", ""))
        rec = originals_by_content.get(key)
        if rec is not None:
            return rec
        # Index fallback
        idx = report_entry.get("index", -1)
        if 0 <= idx < len(originals):
            return originals[idx]
        return None

    # --- Stats pass ---
    acc_by_cell: dict[tuple[str, str], list[float]] = defaultdict(list)
    for report in reports:
        mean_reward = report.get("mean_reward", 0.0)
        rec = _lookup_original(report)
        if rec is not None:
            meta = rec.get("metadata", {})
            cell = (meta.get("duration_bucket", "unknown"), meta.get("source", "unknown"))
            acc_by_cell[cell].append(mean_reward)

    print(f"\n{'duration_bucket':>15s}  {'source':>20s}  {'n':>6s}  {'mean_acc':>10s}  {'in_range':>10s}")
    print("-" * 70)
    total_in_range = 0
    for cell in sorted(acc_by_cell.keys()):
        accs = acc_by_cell[cell]
        mean_acc = sum(accs) / len(accs)
        in_range = sum(1 for a in accs if args.min_acc <= a <= args.max_acc)
        total_in_range += in_range
        bucket, source = cell
        print(f"{bucket:>15s}  {source:>20s}  {len(accs):>6d}  {mean_acc:>10.4f}  {in_range:>10d}")
    print(f"\nTotal in accuracy range [{args.min_acc}, {args.max_acc}]: {total_in_range}")

    if args.stats_only:
        return

    # --- Filter pass ---
    kept_by_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for report in reports:
        mean_reward = report.get("mean_reward", 0.0)

        if not (args.min_acc <= mean_reward <= args.max_acc):
            continue

        rec = _lookup_original(report)
        if rec is None:
            continue

        meta = rec.get("metadata", {})
        cell = (meta.get("duration_bucket", "unknown"), meta.get("source", "unknown"))
        kept_by_cell[cell].append(rec)

    total_kept = sum(len(v) for v in kept_by_cell.values())
    print(f"\nAfter accuracy filter: {total_kept} records across {len(kept_by_cell)} cells")

    if total_kept == 0:
        print("  No records passed the filter!")
        return

    if args.target_total <= 0 or args.target_total >= total_kept:
        sampled = []
        for pool in kept_by_cell.values():
            sampled.extend(pool)
        rng.shuffle(sampled)

        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            for rec in sampled:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if args.target_total <= 0:
            print("\nTarget total <= 0, keeping all filtered records without downsampling.")
        else:
            print("\nTarget total >= filtered count, keeping all filtered records.")
        print(f"Final output: {len(sampled)} records → {args.output}")
        return

    # --- Balanced downsample ---
    n_cells = len(kept_by_cell)
    per_cell = max(1, args.target_total // n_cells)
    remainder = args.target_total - per_cell * n_cells

    sampled = []
    overflow_cells = []

    print(f"\nBalanced downsample: target={args.target_total}, "
          f"cells={n_cells}, per_cell={per_cell}")
    print(f"{'duration_bucket':>15s}  {'source':>20s}  {'available':>10s}  {'sampled':>10s}")
    print("-" * 60)

    for cell in sorted(kept_by_cell.keys()):
        pool = kept_by_cell[cell]
        rng.shuffle(pool)
        n = min(per_cell, len(pool))
        sampled.extend(pool[:n])
        bucket, source = cell
        print(f"{bucket:>15s}  {source:>20s}  {len(pool):>10d}  {n:>10d}")

        # Track cells with leftover capacity
        if len(pool) > per_cell:
            overflow_cells.append((cell, pool[per_cell:]))

    # Distribute remainder from overflow cells
    if remainder > 0 and overflow_cells:
        rng.shuffle(overflow_cells)
        for cell, extra in overflow_cells:
            take = min(remainder, len(extra))
            sampled.extend(extra[:take])
            remainder -= take
            if remainder <= 0:
                break

    # Final shuffle
    rng.shuffle(sampled)

    # Write
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for rec in sampled:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nFinal output: {len(sampled)} records → {args.output}")


if __name__ == "__main__":
    main()
