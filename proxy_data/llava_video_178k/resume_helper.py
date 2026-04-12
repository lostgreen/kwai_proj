#!/usr/bin/env python3
"""
Resume helper: create a reduced input JSONL by removing already-processed items.

Supports two matching modes:
  - content (default): match by (prompt, answer) pair — works even when the
    input JSONL has changed (e.g. different PER_CELL, re-shuffled).
  - index: match by positional index (legacy behaviour, fragile).

Usage:
    python resume_helper.py \
        --input pilot_sample.jsonl \
        --report rollout_report.jsonl \
        --output remaining.jsonl

    # Force index-based matching (legacy)
    python resume_helper.py --match-mode index ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_report_files(report_files: list[str]):
    """Load all report JSONL lines from one or more files."""
    entries: list[dict] = []
    for rpath in report_files:
        if not Path(rpath).is_file():
            continue
        with open(rpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entries.append(json.loads(line))
    return entries


def main():
    parser = argparse.ArgumentParser(description="Create remaining-items JSONL for resume")
    parser.add_argument("--input", required=True, help="Original input JSONL")
    parser.add_argument("--report", required=True, help="Existing (partial) report JSONL")
    parser.add_argument("--output", required=True, help="Output JSONL with remaining items")
    parser.add_argument("--report-shards", nargs="*", default=[],
                        help="Additional shard report files to merge")
    parser.add_argument("--match-mode", choices=["content", "index"], default="content",
                        help="How to identify already-processed items (default: content)")
    args = parser.parse_args()

    report_files = [args.report] + list(args.report_shards)
    report_entries = _load_report_files(report_files)
    print(f"Report entries loaded: {len(report_entries)}")

    # Load original input
    items: list[dict] = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    print(f"Original input: {len(items)}")

    if args.match_mode == "content":
        # Content-based matching: (prompt, answer) pair
        processed_keys: set[tuple[str, str]] = set()
        for entry in report_entries:
            key = (entry.get("prompt", ""), entry.get("answer", ""))
            processed_keys.add(key)
        print(f"Unique processed (prompt, answer) keys: {len(processed_keys)}")

        remaining = 0
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            for idx, item in enumerate(items):
                key = (item.get("prompt", ""), item.get("answer", ""))
                if key not in processed_keys:
                    item.setdefault("metadata", {})["_original_index"] = idx
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    remaining += 1
    else:
        # Index-based matching (legacy)
        processed_indices: set[int] = set()
        for entry in report_entries:
            idx = entry.get("index", -1)
            if idx >= 0:
                processed_indices.add(idx)
        print(f"Processed indices: {len(processed_indices)}")

        remaining = 0
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            for idx, item in enumerate(items):
                if idx not in processed_indices:
                    item.setdefault("metadata", {})["_original_index"] = idx
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    remaining += 1

    print(f"Remaining: {remaining}")
    print(f"Output: {args.output}")

    if remaining == 0:
        print("All items already processed!")
        raise SystemExit(42)


if __name__ == "__main__":
    main()
