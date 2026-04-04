#!/usr/bin/env python3
"""
从 text_filter.py 产出的 passed.jsonl 中，每个 source 随机抽 N 条。

用法:
    python sample_per_source.py \
        --input results/passed.jsonl \
        --output results/sample_preview.json \
        --per-source 2 --seed 42
"""

import argparse
import json
import random
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description="每个 source 随机采样 N 条")
    parser.add_argument("--input", required=True, help="passed.jsonl 路径")
    parser.add_argument("--output", required=True, help="输出 JSON 路径")
    parser.add_argument("--per-source", type=int, default=2, help="每个 source 采样条数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # 加载
    records = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # 按 source 分组
    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_source[r.get("source", "unknown")].append(r)

    # 每个 source 抽 N 条
    sampled = []
    for source in sorted(by_source.keys()):
        pool = by_source[source]
        rng.shuffle(pool)
        n = min(args.per_source, len(pool))
        sampled.extend(pool[:n])
        print(f"  {source}: {len(pool)} available → sampled {n}")

    print(f"\n总计: {len(sampled)} 条 (from {len(by_source)} sources)")

    # 写出
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)
    print(f"✅ 已写入 {args.output}")


if __name__ == "__main__":
    main()
