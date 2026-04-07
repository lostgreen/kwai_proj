#!/usr/bin/env python3
"""
sample_for_demo.py — Balanced sampling from candidates.jsonl by source (domain).

Reads a JSONL file where each line has a ``source`` field (e.g. how_to_step,
cosmo_cap, hacs, ...) and outputs a sub-sampled JSONL with at most N records
per source, preserving the original format.

Usage:
    python sample_for_demo.py \
        --input candidates.jsonl \
        --output sampled.jsonl \
        --per-source 50 \
        --seed 42
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Balanced sampling from JSONL by source field.",
    )
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--per-source", type=int, default=50,
                        help="Max records per source (default: 50)")
    parser.add_argument("--min-duration", type=float, default=30.0,
                        help="Minimum video duration in seconds (default: 30)")
    parser.add_argument("--max-duration", type=float, default=600.0,
                        help="Maximum video duration in seconds (default: 600)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Load and group by source
    by_source: dict[str, list[str]] = defaultdict(list)
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
            source = rec.get("source", "unknown")
            by_source[source].append(line)

    print(f"Total records: {total}")
    print(f"Filtered by duration [{args.min_duration}, {args.max_duration}]: {filtered}")
    print(f"Remaining: {total - filtered}")
    print(f"Sources: {len(by_source)}")

    # Sample
    sampled = []
    for source in sorted(by_source):
        pool = by_source[source]
        rng.shuffle(pool)
        take = min(len(pool), args.per_source)
        sampled.extend(pool[:take])
        print(f"  {source}: {len(pool)} → {take}")

    rng.shuffle(sampled)

    # Write
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for line in sampled:
            f.write(line + "\n")

    print(f"\nSampled {len(sampled)} records → {args.output}")


if __name__ == "__main__":
    main()
