#!/usr/bin/env python3
"""
从 text_filter.py 产出的 passed_timelens.jsonl 中，每个 source 随机抽 N 条。

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


def main():
    parser = argparse.ArgumentParser(description="每个 source 随机采样 N 条 (TimeLens, 输出 JSONL)")
    parser.add_argument("--input", required=True, help="passed_timelens.jsonl 路径")
    parser.add_argument("--output", required=True, help="输出 JSONL 路径")
    parser.add_argument("--per-source", type=int, default=2, help="每个 source 采样条数")
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

    sampled_raw = []
    for source in sorted(by_source.keys()):
        pool = by_source[source]
        rng.shuffle(pool)
        n = min(args.per_source, len(pool))
        sampled_raw.extend(pool[:n])
        print(f"  {source}: {len(pool)} available -> sampled {n}")

    print(f"\nTotal: {len(sampled_raw)} records (from {len(by_source)} sources)")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for raw in sampled_raw:
            unified = to_unified_record(raw, args.video_root)
            f.write(json.dumps(unified, ensure_ascii=False) + "\n")

    print(f"Wrote {args.output} ({len(sampled_raw)} records, JSONL)")


if __name__ == "__main__":
    main()
