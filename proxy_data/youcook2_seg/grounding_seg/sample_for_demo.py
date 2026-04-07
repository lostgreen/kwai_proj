#!/usr/bin/env python3
"""
sample_for_demo.py — Balanced sampling from JSONL by a grouping field.

Supports grouping by:
  - ``source``          (candidates.jsonl: how_to_step, cosmo_cap, ...)
  - ``_screen.domain_l1`` (screen_keep.jsonl: educational, entertainment, ...)
  - any dot-separated nested key

Usage:
    # By source (candidates.jsonl)
    python sample_for_demo.py --input candidates.jsonl --output sampled.jsonl

    # By domain_l1 (screen_keep.jsonl)
    python sample_for_demo.py --input screen_keep.jsonl --output sampled.jsonl \
        --group-by _screen.domain_l1
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def _get_nested(rec: dict, key: str) -> str:
    """Resolve dot-separated nested key, e.g. ``_screen.domain_l1``."""
    obj = rec
    for part in key.split("."):
        if isinstance(obj, dict):
            obj = obj.get(part)
        else:
            return "unknown"
    return str(obj) if obj is not None else "unknown"


def main():
    parser = argparse.ArgumentParser(
        description="Balanced sampling from JSONL by a grouping field.",
    )
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--group-by", default="_screen.domain_l1",
                        help="Dot-separated field to group by (default: _screen.domain_l1)")
    parser.add_argument("--per-group", type=int, default=50,
                        help="Max records per group (default: 50)")
    parser.add_argument("--min-duration", type=float, default=30.0,
                        help="Minimum video duration in seconds (default: 30)")
    parser.add_argument("--max-duration", type=float, default=600.0,
                        help="Maximum video duration in seconds (default: 600)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Load and group
    by_group: dict[str, list[str]] = defaultdict(list)
    total = 0
    filtered = 0
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            rec = json.loads(line)
            dur = rec.get("duration", 0)
            if dur < args.min_duration or dur > args.max_duration:
                filtered += 1
                continue
            group = _get_nested(rec, args.group_by)
            by_group[group].append(line)

    print(f"Total records: {total}")
    print(f"Filtered by duration [{args.min_duration}, {args.max_duration}]: {filtered}")
    print(f"Remaining: {total - filtered}")
    print(f"Groups (by {args.group_by}): {len(by_group)}")

    # Sample
    sampled = []
    for group in sorted(by_group):
        pool = by_group[group]
        rng.shuffle(pool)
        take = min(len(pool), args.per_group)
        sampled.extend(pool[:take])
        print(f"  {group}: {len(pool)} → {take}")

    rng.shuffle(sampled)

    # Write
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for line in sampled:
            f.write(line + "\n")

    print(f"\nSampled {len(sampled)} records → {args.output}")


if __name__ == "__main__":
    main()
