#!/usr/bin/env python3
"""
Resume helper: create a reduced input JSONL by removing already-processed items.

Reads an existing rollout report to find which indices have been processed,
then outputs a new JSONL containing only the remaining items with their
original indices preserved in metadata._original_index.

Usage:
    python resume_helper.py \
        --input pilot_sample.jsonl \
        --report rollout_report.jsonl \
        --output remaining.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Create remaining-items JSONL for resume")
    parser.add_argument("--input", required=True, help="Original input JSONL")
    parser.add_argument("--report", required=True, help="Existing (partial) report JSONL")
    parser.add_argument("--output", required=True, help="Output JSONL with remaining items")
    # For shard-based DP: merge multiple shard reports
    parser.add_argument("--report-shards", nargs="*", default=[],
                        help="Additional shard report files to merge")
    args = parser.parse_args()

    # Collect processed indices from all report files
    processed: set[int] = set()
    report_files = [args.report] + list(args.report_shards)
    for rpath in report_files:
        if not Path(rpath).is_file():
            continue
        with open(rpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                idx = entry.get("index", -1)
                if idx >= 0:
                    processed.add(idx)

    print(f"Processed indices: {len(processed)}")

    # Load original input
    items = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    print(f"Original input: {len(items)}")

    # Write remaining items, preserving original index
    remaining = 0
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for idx, item in enumerate(items):
            if idx not in processed:
                # Store original index so we can merge reports later
                item.setdefault("metadata", {})["_original_index"] = idx
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                remaining += 1

    print(f"Remaining: {remaining}")
    print(f"Output: {args.output}")

    # Exit with code 0 if there's work to do, 42 if all done
    if remaining == 0:
        print("All items already processed!")
        raise SystemExit(42)


if __name__ == "__main__":
    main()
