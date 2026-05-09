from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .io import write_json, write_jsonl
from .sources import duration, load_records, normalize_dataset, source_key, to_unified_record


def passes_duration(record: dict[str, Any], min_duration: float, max_duration: float) -> bool:
    dur = duration(record)
    return min_duration <= dur <= max_duration


def select_records(
    records: list[dict[str, Any]],
    *,
    dataset: str,
    min_duration: float,
    max_duration: float,
    per_source: int = 0,
    target_total: int = 0,
    balanced_total: bool = False,
    seed: int = 42,
) -> list[dict[str, Any]]:
    normalize_dataset(dataset)
    rng = random.Random(seed)
    filtered = [record for record in records if passes_duration(record, min_duration, max_duration)]

    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in filtered:
        by_source[source_key(record)].append(record)

    pooled_by_source: dict[str, list[dict[str, Any]]] = {}
    for source, pool in sorted(by_source.items()):
        ranked = list(pool)
        rng.shuffle(ranked)
        ranked.sort(key=lambda item: (duration(item), _video_id(item)))
        pooled_by_source[source] = ranked[:per_source] if per_source > 0 else ranked

    if target_total <= 0:
        return [item for source in sorted(pooled_by_source) for item in pooled_by_source[source]]

    if balanced_total:
        return _allocate_balanced_total(pooled_by_source, target_total)

    merged = [item for source in sorted(pooled_by_source) for item in pooled_by_source[source]]
    rng.shuffle(merged)
    return merged[:target_total]


def curate_dataset(
    *,
    dataset: str,
    input_path: str | Path,
    output_dir: str | Path,
    video_root: str | Path | None = None,
    min_duration: float = 60.0,
    max_duration: float = 240.0,
    per_source: int = 0,
    target_total: int = 0,
    balanced_total: bool = False,
    seed: int = 42,
    write_screen_keep: bool = True,
) -> dict[str, Any]:
    dataset = normalize_dataset(dataset)
    records = load_records(dataset, input_path)
    selected = select_records(
        records,
        dataset=dataset,
        min_duration=min_duration,
        max_duration=max_duration,
        per_source=per_source,
        target_total=target_total,
        balanced_total=balanced_total,
        seed=seed,
    )
    unified = [to_unified_record(dataset, record, video_root) for record in selected]

    out_dir = Path(output_dir)
    duration_keep = out_dir / "duration_keep.jsonl"
    write_jsonl(duration_keep, unified)
    if write_screen_keep:
        write_jsonl(out_dir / "screen_keep.jsonl", unified)

    summary = {
        "dataset": dataset,
        "input": str(Path(input_path)),
        "total": len(records),
        "kept": len(unified),
        "rejected_duration": len(records) - len([r for r in records if passes_duration(r, min_duration, max_duration)]),
        "min_duration": min_duration,
        "max_duration": max_duration,
        "per_source": per_source,
        "target_total": target_total,
        "balanced_total": balanced_total,
        "seed": seed,
        "by_source": _count_by_source(selected),
        "outputs": {
            "duration_keep": str(duration_keep),
            "screen_keep": str(out_dir / "screen_keep.jsonl") if write_screen_keep else None,
        },
    }
    write_json(out_dir / "duration_summary.json", summary)
    return summary


def _video_id(record: dict[str, Any]) -> str:
    return str(record.get("video") or record.get("video_path") or "")


def _count_by_source(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        counts[source_key(record)] += 1
    return dict(sorted(counts.items()))


def _allocate_balanced_total(
    by_source: dict[str, list[dict[str, Any]]],
    total: int,
) -> list[dict[str, Any]]:
    sources = sorted(by_source)
    if total <= 0 or not sources:
        return []

    quota = {source: total // len(sources) for source in sources}
    remainder = total - sum(quota.values())
    for source in sorted(sources, key=lambda key: len(by_source[key]), reverse=True)[:remainder]:
        quota[source] += 1

    selected: list[dict[str, Any]] = []
    taken: dict[str, int] = {}
    overflow = 0
    for source in sources:
        n_take = min(quota[source], len(by_source[source]))
        selected.extend(by_source[source][:n_take])
        taken[source] = n_take
        overflow += quota[source] - n_take

    if overflow > 0:
        for source in sorted(sources, key=lambda key: len(by_source[key]), reverse=True):
            if overflow <= 0:
                break
            remaining = by_source[source][taken[source]:]
            if not remaining:
                continue
            extra = min(overflow, len(remaining))
            selected.extend(remaining[:extra])
            overflow -= extra

    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Duration-first data curation")
    parser.add_argument("--dataset", required=True, choices=["et_instruct_164k", "timelens_100k"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--video-root", default=None)
    parser.add_argument("--min-duration", type=float, default=60.0)
    parser.add_argument("--max-duration", type=float, default=240.0)
    parser.add_argument("--per-source", type=int, default=0)
    parser.add_argument("--target-total", type=int, default=0)
    parser.add_argument("--balanced-total", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-screen-keep", action="store_true")
    args = parser.parse_args()

    summary = curate_dataset(
        dataset=args.dataset,
        input_path=args.input,
        output_dir=args.output_dir,
        video_root=args.video_root,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        per_source=args.per_source,
        target_total=args.target_total,
        balanced_total=args.balanced_total,
        seed=args.seed,
        write_screen_keep=not args.no_screen_keep,
    )
    print(f"Duration curation done: kept {summary['kept']} / {summary['total']}")
    print(f"Output: {summary['outputs']['duration_keep']}")


if __name__ == "__main__":
    main()
