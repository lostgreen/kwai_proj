#!/usr/bin/env python3
"""Build a final Event Logic harder training mix.

Inputs:
  - hard PN/FB cases selected by rollout filtering
  - frame-list sort records generated from the VLM Event Logic cache

The final mix keeps filtered PN/FB cases first, then fills the remaining
budget with sort records whose ordered sequence is longest.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"failed to parse {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"expected object in {path}:{line_no}")
            rows.append(row)
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _record_key(row: dict[str, Any]) -> tuple[str, str, str]:
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    row_id = str(row.get("metadata_id") or meta.get("id") or "").strip()
    if row_id:
        return ("id", str(row.get("problem_type") or ""), row_id)
    return (
        "content",
        str(row.get("problem_type") or ""),
        json.dumps(
            {
                "prompt": row.get("prompt", ""),
                "answer": row.get("answer", ""),
                "videos": row.get("videos") or [],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
    )


def _sort_length(row: dict[str, Any]) -> int:
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    ordered_ids = meta.get("ordered_ids")
    if isinstance(ordered_ids, list):
        return len(ordered_ids)
    videos = row.get("videos")
    if isinstance(videos, list):
        return len(videos)
    answer = str(row.get("answer") or "")
    return len(answer)


def _dedupe(rows: list[dict[str, Any]], seen: set[tuple[str, str, str]]) -> tuple[list[dict[str, Any]], int]:
    unique: list[dict[str, Any]] = []
    duplicates = 0
    for row in rows:
        key = _record_key(row)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        unique.append(row)
    return unique, duplicates


def _summarize_by_type(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get("problem_type") or "unknown") for row in rows).items()))


def build_harder_mix(
    *,
    hard_paths: list[str | Path],
    sort_path: str | Path,
    output_path: str | Path,
    stats_path: str | Path | None = None,
    target_total: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    if target_total <= 0:
        raise ValueError("target_total must be positive")

    seen: set[tuple[str, str, str]] = set()
    hard_input_count = 0
    hard_records: list[dict[str, Any]] = []
    hard_duplicate_count = 0
    for path in hard_paths:
        rows = load_jsonl(path)
        hard_input_count += len(rows)
        unique, duplicates = _dedupe(rows, seen)
        hard_duplicate_count += duplicates
        hard_records.extend(unique)

    sort_rows = [row for row in load_jsonl(sort_path) if row.get("problem_type") == "event_logic_sort"]
    sort_rows.sort(
        key=lambda row: (
            -_sort_length(row),
            str((row.get("metadata") or {}).get("clip_key", "")) if isinstance(row.get("metadata"), dict) else "",
            str(row.get("answer", "")),
            str(row.get("prompt", "")),
        )
    )
    sort_unique, sort_duplicate_count = _dedupe(sort_rows, seen)

    if len(hard_records) >= target_total:
        selected_hard = random.Random(seed).sample(hard_records, target_total)
        selected_sort: list[dict[str, Any]] = []
    else:
        selected_hard = list(hard_records)
        remaining = target_total - len(selected_hard)
        selected_sort = sort_unique[:remaining]

    selected = selected_hard + selected_sort
    selected_before_shuffle_by_source = {
        "hard": len(selected_hard),
        "sort": len(selected_sort),
    }
    random.Random(seed).shuffle(selected)
    write_jsonl(selected, output_path)

    stats = {
        "stage": "event-logic-harder-mix",
        "target_total": target_total,
        "seed": seed,
        "hard_paths": [str(Path(path)) for path in hard_paths],
        "sort_path": str(Path(sort_path)),
        "output_path": str(Path(output_path)),
        "input_hard_count": hard_input_count,
        "hard_unique_count": len(hard_records),
        "hard_duplicate_count": hard_duplicate_count,
        "input_sort_count": len(sort_rows),
        "sort_unique_count": len(sort_unique),
        "sort_duplicate_count": sort_duplicate_count,
        "selected_total": len(selected),
        "selected_before_shuffle_by_source": selected_before_shuffle_by_source,
        "selected_by_problem_type": _summarize_by_type(selected),
        "selected_sort_lengths": [_sort_length(row) for row in selected_sort],
    }
    if stats_path is not None:
        stats_out = Path(stats_path)
        stats_out.parent.mkdir(parents=True, exist_ok=True)
        stats_out.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge filtered PN/FB hard cases with longest Event Logic sort frame-list records",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--hard-input", action="append", required=True,
                        help="Filtered PN/FB hard_cases.jsonl. Repeat for each file.")
    parser.add_argument("--sort-input", required=True, help="Frame-list event_logic_sort JSONL")
    parser.add_argument("--output", required=True, help="Final mixed train JSONL")
    parser.add_argument("--stats-output", default="", help="Optional stats JSON path")
    parser.add_argument("--target-total", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats_output = args.stats_output or str(Path(args.output).with_suffix(".stats.json"))
    stats = build_harder_mix(
        hard_paths=[Path(path) for path in args.hard_input],
        sort_path=Path(args.sort_input),
        output_path=Path(args.output),
        stats_path=Path(stats_output),
        target_total=args.target_total,
        seed=args.seed,
    )
    print(f"[event-logic-harder-mix] selected {stats['selected_total']} / target {stats['target_total']}")
    print(f"[event-logic-harder-mix] hard={stats['selected_before_shuffle_by_source']['hard']} "
          f"sort={stats['selected_before_shuffle_by_source']['sort']}")
    print(f"[event-logic-harder-mix] output: {stats['output_path']}")
    print(f"[event-logic-harder-mix] stats: {stats_output}")
    print("[event-logic-harder-mix] by_problem_type:")
    for problem_type, count in stats["selected_by_problem_type"].items():
        print(f"  {problem_type}: {count}")


if __name__ == "__main__":
    main()
