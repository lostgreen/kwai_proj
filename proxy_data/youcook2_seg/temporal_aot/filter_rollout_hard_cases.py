#!/usr/bin/env python3
"""Filter rollout reports into a hard-case training set.

Keeps samples whose mean reward is within a configurable range (default:
[0.0, 0.5]) and optionally downsamples to a target total while balancing across
problem_type and, within each problem_type, metadata.domain_l1.

Placed in the temporal_aot folder so the full data-generation pipeline stays
co-located with the builders.
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


def _sample_nested(records: list[dict], target: int, nested_key: str, rng: random.Random) -> list[dict]:
    if len(records) <= target or target <= 0 or not nested_key:
        picked = list(records)
        rng.shuffle(picked)
        return picked[:target] if target > 0 else picked

    by_nested: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        meta = rec.get("metadata") or {}
        by_nested[str(meta.get(nested_key, "unknown"))].append(rec)

    groups = sorted(by_nested)
    per_group = max(1, target // max(len(groups), 1))
    remainder = target - per_group * len(groups)

    sampled: list[dict] = []
    leftovers: list[dict] = []
    for group in groups:
        pool = list(by_nested[group])
        rng.shuffle(pool)
        take = min(per_group, len(pool))
        sampled.extend(pool[:take])
        leftovers.extend(pool[take:])

    if remainder > 0 and leftovers:
        rng.shuffle(leftovers)
        sampled.extend(leftovers[:remainder])

    rng.shuffle(sampled)
    return sampled[:target]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter rollout report by mean_reward and downsample hard cases",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--report", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-mean-reward", type=float, default=0.0)
    parser.add_argument("--max-mean-reward", type=float, default=0.5)
    parser.add_argument("--target-total", type=int, default=10000,
                        help="0 or negative means keep all filtered samples")
    parser.add_argument("--nested-balance-key", default="domain_l1",
                        help="metadata key used within each problem_type bucket")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    reports = load_jsonl(args.report)
    originals = load_jsonl(args.input)

    originals_by_content = {
        (rec.get("prompt", ""), rec.get("answer", "")): rec
        for rec in originals
    }

    def lookup_original(report_entry: dict) -> dict | None:
        key = (report_entry.get("prompt", ""), report_entry.get("answer", ""))
        record = originals_by_content.get(key)
        if record is not None:
            return record
        idx = report_entry.get("index", -1)
        if isinstance(idx, int) and 0 <= idx < len(originals):
            return originals[idx]
        return None

    candidates: list[dict] = []
    seen_keys: set[tuple[str, str]] = set()
    for report in reports:
        mean_reward = float(report.get("mean_reward", 0.0))
        if not (args.min_mean_reward <= mean_reward <= args.max_mean_reward):
            continue
        record = lookup_original(report)
        if record is None:
            continue
        key = (record.get("prompt", ""), record.get("answer", ""))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        candidates.append(record)

    print(f"Filtered hard cases: {len(candidates)} / {len(reports)} "
          f"(mean_reward in [{args.min_mean_reward}, {args.max_mean_reward}])")

    if args.target_total <= 0 or len(candidates) <= args.target_total:
        final_records = list(candidates)
        rng.shuffle(final_records)
    else:
        by_type: dict[str, list[dict]] = defaultdict(list)
        for rec in candidates:
            by_type[str(rec.get("problem_type", "unknown"))].append(rec)

        task_types = sorted(by_type)
        per_type = max(1, args.target_total // max(len(task_types), 1))
        remainder = args.target_total - per_type * len(task_types)

        final_records: list[dict] = []
        leftovers: list[dict] = []
        for task in task_types:
            pool = by_type[task]
            picked = _sample_nested(pool, per_type, args.nested_balance_key, rng)
            final_records.extend(picked)
            picked_keys = {(rec.get("prompt", ""), rec.get("answer", "")) for rec in picked}
            leftovers.extend([rec for rec in pool if (rec.get("prompt", ""), rec.get("answer", "")) not in picked_keys])

        if remainder > 0 and leftovers:
            rng.shuffle(leftovers)
            final_records.extend(leftovers[:remainder])
        rng.shuffle(final_records)
        final_records = final_records[:args.target_total]

    write_jsonl(final_records, args.output)
    print(f"Output: {len(final_records)} records -> {args.output}")

    by_type_out: dict[str, int] = defaultdict(int)
    for rec in final_records:
        by_type_out[str(rec.get("problem_type", "unknown"))] += 1
    print("By problem_type:")
    for task in sorted(by_type_out):
        print(f"  {task}: {by_type_out[task]}")


if __name__ == "__main__":
    main()
