#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Curate a fixed-size subset from offline-filtered AoT MCQ data.

Prioritises samples whose mean rollout reward is close to 0.5 (highest
GRPO training signal) and balances difficulty tiers:

  - medium  [mid_lo, mid_hi]   → 60 % of target  (default)
  - hard    (0, mid_lo)        → 30 %
  - easy    (mid_hi, 1)        → 10 %

Samples with mean_reward == 0.0 (all-wrong) or == 1.0 (all-correct) are
excluded — the former because the base model has zero capability on them,
the latter because they provide no RL gradient.

When --per-type-quota is given (JSON dict), the target count is split
across problem_types and each type is sampled independently with the same
tier ratios.

Example:
  python curate_1k_samples.py \
    --report-jsonl  ablations/exp1/offline_filter_report.jsonl \
    --train-jsonl   ablations/exp1/mixed_train.jsonl \
    --output-jsonl  ablations/exp1/mixed_train.curated_1k.jsonl \
    --target-count  1000

  # Mixed experiment (exp7): 500 v2t binary + 500 v2t 3way
  python curate_1k_samples.py \
    --report-jsonl  ablations/exp1/offline_filter_report.jsonl,ablations/exp2/offline_filter_report.jsonl \
    --train-jsonl   ablations/exp1/mixed_train.jsonl,ablations/exp2/mixed_train.jsonl \
    --output-jsonl  ablations/exp7/mixed_train.curated_1k.jsonl \
    --target-count  1000 \
    --per-type-quota '{"aot_v2t": 500, "aot_3way_v2t": 500}'
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Curate a fixed-size difficulty-balanced subset.")
    p.add_argument("--report-jsonl", required=True,
                   help="Comma-separated offline_filter_report.jsonl path(s)")
    p.add_argument("--train-jsonl", required=True,
                   help="Comma-separated mixed_train.jsonl path(s), aligned with --report-jsonl")
    p.add_argument("--output-jsonl", required=True,
                   help="Output curated JSONL")
    p.add_argument("--target-count", type=int, default=1000)
    p.add_argument("--mid-ratio", type=float, default=0.6)
    p.add_argument("--hard-ratio", type=float, default=0.3)
    p.add_argument("--easy-ratio", type=float, default=0.1)
    p.add_argument("--mid-lo", type=float, default=0.3,
                   help="Lower bound of medium tier (inclusive)")
    p.add_argument("--mid-hi", type=float, default=0.7,
                   help="Upper bound of medium tier (inclusive)")
    p.add_argument("--per-type-quota", default="",
                   help='JSON dict, e.g. \'{"aot_v2t": 500, "aot_3way_v2t": 500}\'')
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_report_and_train(report_paths: list[str], train_paths: list[str]):
    """Load report entries and training samples, pairing them by (file_idx, line_index).

    Returns a list of (report_entry, train_sample) tuples for samples that
    passed the offline filter (keep=True).
    """
    paired: list[tuple[dict, dict]] = []

    for report_path, train_path in zip(report_paths, train_paths):
        # Load train samples indexed by line position
        train_samples: list[dict] = []
        with open(train_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    train_samples.append(json.loads(line))

        # Load report and pair
        with open(report_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if not entry.get("keep", False):
                    continue
                idx = entry["index"]
                if idx < len(train_samples):
                    paired.append((entry, train_samples[idx]))

    return paired


def compute_mean_reward(entry: dict) -> float:
    rewards = entry.get("rewards", [])
    if not rewards:
        return 0.0
    return sum(rewards) / len(rewards)


def sample_tier(candidates: list, count: int, rng: random.Random) -> list:
    """Take up to `count` items from pre-sorted candidates."""
    if len(candidates) <= count:
        return list(candidates)
    return list(candidates[:count])


def curate_for_type(
    pairs: list[tuple[dict, dict]],
    target: int,
    mid_lo: float,
    mid_hi: float,
    mid_ratio: float,
    hard_ratio: float,
    easy_ratio: float,
    rng: random.Random,
) -> list[dict]:
    """Select `target` samples with difficulty-tier balancing."""

    # Compute mean reward and classify
    scored: list[tuple[float, dict, dict]] = []
    for entry, sample in pairs:
        mr = compute_mean_reward(entry)
        # Exclude all-wrong (mr == 0.0) and all-correct (mr == 1.0)
        # Note: offline filter already removed all-same-reward, but
        #       mr could still be 0.0 if all 8 rollouts scored 0.
        if mr <= 0.0 or mr >= 1.0:
            continue
        scored.append((mr, entry, sample))

    # Classify into tiers
    hard = []   # (0, mid_lo)
    medium = []  # [mid_lo, mid_hi]
    easy = []    # (mid_hi, 1)

    for mr, entry, sample in scored:
        if mr < mid_lo:
            hard.append((mr, sample))
        elif mr > mid_hi:
            easy.append((mr, sample))
        else:
            medium.append((mr, sample))

    # Sort within tiers (priority order)
    medium.sort(key=lambda x: abs(x[0] - 0.5))         # closest to 0.5 first
    hard.sort(key=lambda x: -x[0])                       # closest to mid_lo first
    easy.sort(key=lambda x: x[0])                        # closest to mid_hi first

    # Compute tier targets
    n_mid = int(target * mid_ratio)
    n_hard = int(target * hard_ratio)
    n_easy = target - n_mid - n_hard  # remainder to easy

    # Sample from each tier
    mid_selected = sample_tier(medium, n_mid, rng)
    hard_selected = sample_tier(hard, n_hard, rng)
    easy_selected = sample_tier(easy, n_easy, rng)

    # Redistribute shortfalls to medium (most valuable tier)
    total_selected = len(mid_selected) + len(hard_selected) + len(easy_selected)
    shortfall = target - total_selected

    if shortfall > 0:
        # Try to fill from medium overflow
        already_mid = len(mid_selected)
        extra_mid = sample_tier(medium[already_mid:], shortfall, rng)
        mid_selected.extend(extra_mid)
        shortfall -= len(extra_mid)

    if shortfall > 0:
        # Then from hard overflow
        already_hard = len(hard_selected)
        extra_hard = sample_tier(hard[already_hard:], shortfall, rng)
        hard_selected.extend(extra_hard)
        shortfall -= len(extra_hard)

    if shortfall > 0:
        # Finally from easy overflow
        already_easy = len(easy_selected)
        extra_easy = sample_tier(easy[already_easy:], shortfall, rng)
        easy_selected.extend(extra_easy)

    result = [s for _, s in mid_selected + hard_selected + easy_selected]

    # Stats
    n_total_avail = len(scored)
    print(f"  Available (excl 0.0/1.0): {n_total_avail}  "
          f"(hard={len(hard)}, medium={len(medium)}, easy={len(easy)})")
    print(f"  Selected: {len(result)}  "
          f"(hard={len(hard_selected)}, medium={len(mid_selected)}, easy={len(easy_selected)})")

    return result


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    report_paths = [p.strip() for p in args.report_jsonl.split(",") if p.strip()]
    train_paths = [p.strip() for p in args.train_jsonl.split(",") if p.strip()]

    if len(report_paths) != len(train_paths):
        print("ERROR: --report-jsonl and --train-jsonl must have the same number of paths",
              file=sys.stderr)
        sys.exit(1)

    print(f"Loading {len(report_paths)} report/train file pair(s) ...")
    all_pairs = load_report_and_train(report_paths, train_paths)
    print(f"Total keep=True pairs: {len(all_pairs)}")

    # Parse per-type quota
    per_type_quota: dict[str, int] = {}
    if args.per_type_quota:
        per_type_quota = json.loads(args.per_type_quota)
        total_quota = sum(per_type_quota.values())
        if total_quota != args.target_count:
            print(f"WARNING: sum of per-type quotas ({total_quota}) != target_count ({args.target_count}). "
                  f"Using quota sum.", file=sys.stderr)
            args.target_count = total_quota

    output: list[dict] = []

    if per_type_quota:
        # Group pairs by problem_type
        by_type: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
        for entry, sample in all_pairs:
            ptype = entry.get("problem_type") or sample.get("problem_type", "unknown")
            by_type[ptype].append((entry, sample))

        for ptype, quota in per_type_quota.items():
            print(f"\n[{ptype}] target={quota}")
            type_pairs = by_type.get(ptype, [])
            if not type_pairs:
                print(f"  WARNING: no samples found for type '{ptype}'!", file=sys.stderr)
                continue
            selected = curate_for_type(
                type_pairs, quota,
                args.mid_lo, args.mid_hi,
                args.mid_ratio, args.hard_ratio, args.easy_ratio,
                rng,
            )
            output.extend(selected)
    else:
        print(f"\nAll types combined, target={args.target_count}")
        output = curate_for_type(
            all_pairs, args.target_count,
            args.mid_lo, args.mid_hi,
            args.mid_ratio, args.hard_ratio, args.easy_ratio,
            rng,
        )

    # Shuffle final output
    rng.shuffle(output)

    # Write
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for sample in output:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(output)} samples to {args.output_jsonl}")
    if len(output) < args.target_count:
        print(f"WARNING: only {len(output)}/{args.target_count} samples available. "
              f"Consider lowering --target-count or generating more data.", file=sys.stderr)


if __name__ == "__main__":
    main()
