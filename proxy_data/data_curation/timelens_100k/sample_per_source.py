#!/usr/bin/env python3
"""
从 text_filter.py 产出的 passed_timelens.jsonl 中采样，并转为统一 JSONL。

输出 JSONL 格式（兼容 local_screen.py / extract_frames.py 输入）:
  - 将 TimeLens 原始 schema (video_path, events, ...) 转为
    统一 schema (videos[], metadata{clip_key, clip_start, clip_end, ...})
  - 保留原始字段在 _tl_raw 中供溯源

用法:
    python sample_per_source.py \
        --input results/passed_timelens.jsonl \
        --output results/sample_dev.jsonl \
        --per-source 2 --seed 42 \
        --video-root /path/to/video_shards

    python sample_per_source.py \
        --input results/passed_timelens.jsonl \
        --output results/sample_3k.jsonl \
        --video-root /path/to/video_shards \
        --total 3000 --balanced-total
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def to_unified_record(raw: dict, video_root: str | None = None) -> dict:
    """将 TimeLens text_filter 输出转为统一格式。"""
    video_rel = raw.get("video_path", "")
    duration = raw.get("duration", 0)
    source = raw.get("source", "unknown")

    if video_root:
        video_path = str(Path(video_root) / video_rel)
    else:
        video_path = video_rel

    clip_key = Path(video_rel).stem

    # Count events
    events = raw.get("events", [])
    n_events = len(events)

    return {
        "videos": [video_path],
        "metadata": {
            "clip_key": clip_key,
            "video_id": clip_key,
            "clip_start": 0,
            "clip_end": duration,
            "clip_duration": duration,
            "original_duration": duration,
            "is_full_video": True,
            "source": source,
        },
        "source": source,
        "dataset": "TimeLens-100K",
        "duration": duration,
        "_tl_raw": {
            "video_path": video_rel,
            "n_events": n_events,
            "events_summary": [
                {"query": ev.get("query", ""), "span": ev.get("span", [])}
                for ev in events[:10]  # keep first 10 for reference
            ],
        },
        "_origin": raw.get("_origin"),
    }


def rank_pool(pool: list[dict], rng: random.Random) -> list[dict]:
    ranked = list(pool)
    rng.shuffle(ranked)
    ranked.sort(
        key=lambda item: (
            -len(item.get("events") or []),
            float(item.get("duration") or 0.0),
            item.get("video_path") or "",
        )
    )
    return ranked


def allocate_balanced_total(by_source: dict[str, list[dict]], total: int) -> list[dict]:
    sources = sorted(by_source.keys())
    if total <= 0 or not sources:
        return []

    base = total // len(sources)
    quota = {source: base for source in sources}
    remainder = total - base * len(sources)
    by_size = sorted(sources, key=lambda source: len(by_source[source]), reverse=True)
    for idx in range(remainder):
        quota[by_size[idx % len(by_size)]] += 1

    selected: list[dict] = []
    overflow = 0
    taken = {source: 0 for source in sources}

    for source in sources:
        pool = by_source[source]
        n_take = min(quota[source], len(pool))
        selected.extend(pool[:n_take])
        taken[source] = n_take
        overflow += quota[source] - n_take

    if overflow > 0:
        for source in by_size:
            if overflow <= 0:
                break
            pool = by_source[source]
            remaining = len(pool) - taken[source]
            if remaining <= 0:
                continue
            extra = min(overflow, remaining)
            selected.extend(pool[taken[source]:taken[source] + extra])
            taken[source] += extra
            overflow -= extra

    return selected


def main():
    parser = argparse.ArgumentParser(description="每个 source 随机采样 N 条 (TimeLens, 输出 JSONL)")
    parser.add_argument("--input", required=True, help="passed_timelens.jsonl 路径")
    parser.add_argument("--output", required=True, help="输出 JSONL 路径")
    parser.add_argument("--per-source", type=int, default=0, help="每个 source 最多采样条数；0 = 不设上限")
    parser.add_argument("--total", type=int, default=0, help="总采样条数；0 = 不设上限")
    parser.add_argument("--balanced-total", action="store_true",
                        help="当 --total > 0 时，按 source 尽量均衡分配总量")
    parser.add_argument("--video-root", default=None,
                        help="视频根目录（拼接到 video_path 前）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    records = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_source[r.get("source", "unknown")].append(r)

    pooled_by_source: dict[str, list[dict]] = {}
    for source in sorted(by_source.keys()):
        ranked_pool = rank_pool(by_source[source], rng)
        if args.per_source > 0:
            ranked_pool = ranked_pool[:args.per_source]
        pooled_by_source[source] = ranked_pool
        cap_label = args.per_source if args.per_source > 0 else "all"
        print(f"  {source}: {len(by_source[source])} available -> pooled {len(ranked_pool)} (cap={cap_label})")

    if args.total > 0:
        if args.balanced_total:
            sampled_raw = allocate_balanced_total(pooled_by_source, args.total)
        else:
            merged_pool = [item for pool in pooled_by_source.values() for item in pool]
            rng.shuffle(merged_pool)
            sampled_raw = merged_pool[:args.total]
    else:
        sampled_raw = [item for pool in pooled_by_source.values() for item in pool]

    sampled_by_source: dict[str, int] = defaultdict(int)
    for item in sampled_raw:
        sampled_by_source[item.get("source", "unknown")] += 1

    print(f"\nTotal: {len(sampled_raw)} records (from {len(by_source)} sources)")
    if args.total > 0:
        print("Selected per source:")
        for source in sorted(sampled_by_source.keys()):
            print(f"  {source}: {sampled_by_source[source]}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for raw in sampled_raw:
            unified = to_unified_record(raw, args.video_root)
            f.write(json.dumps(unified, ensure_ascii=False) + "\n")

    print(f"Wrote {args.output} ({len(sampled_raw)} records, JSONL)")


if __name__ == "__main__":
    main()
