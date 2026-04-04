#!/usr/bin/env python3
"""
从 text_filter.py 产出的 passed.jsonl 中，每个 source 随机抽 N 条。

输出 JSONL 格式（兼容 extract_frames.py --jsonl 输入）:
  - 将 ET-Instruct 原始 schema (video, duration, ...) 转为
    统一 schema (videos[], metadata{clip_key, clip_start, clip_end, ...})
  - 保留原始字段在 _et_raw 中供溯源

用法:
    python sample_per_source.py \
        --input results/passed.jsonl \
        --output results/sample_dev.jsonl \
        --per-source 2 --seed 42 \
        --video-root /m2v_intern/xuboshen/zgw/data/ET-Instruct-164K/videos
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def to_unified_record(raw: dict, video_root: str | None = None) -> dict:
    """将 ET-Instruct text_filter 输出转为 extract_frames.py 统一格式。"""
    video_rel = raw.get("video", "")
    duration = raw.get("duration", 0)
    source = raw.get("source", "unknown")

    # video path
    if video_root:
        video_path = str(Path(video_root) / video_rel)
    else:
        video_path = video_rel

    # clip_key: 视频文件 stem（去掉目录和扩展名）
    clip_key = Path(video_rel).stem

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
        "dataset": "ET-Instruct-164K",
        "duration": duration,
        "_et_raw": {
            "video": video_rel,
            "task": raw.get("task"),
            "tgt": raw.get("tgt"),
            "n_events": len(raw.get("tgt", [])) // 2,
        },
        "_origin": raw.get("_origin"),
    }


def main():
    parser = argparse.ArgumentParser(description="每个 source 随机采样 N 条 (输出 JSONL)")
    parser.add_argument("--input", required=True, help="passed.jsonl 路径")
    parser.add_argument("--output", required=True, help="输出 JSONL 路径")
    parser.add_argument("--per-source", type=int, default=2, help="每个 source 采样条数")
    parser.add_argument("--video-root", default=None,
                        help="视频根目录（拼接到 video 相对路径前）")
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
    sampled_raw = []
    for source in sorted(by_source.keys()):
        pool = by_source[source]
        rng.shuffle(pool)
        n = min(args.per_source, len(pool))
        sampled_raw.extend(pool[:n])
        print(f"  {source}: {len(pool)} available → sampled {n}")

    print(f"\n总计: {len(sampled_raw)} 条 (from {len(by_source)} sources)")

    # 转为统一格式 & 写出 JSONL
    with open(args.output, "w", encoding="utf-8") as f:
        for raw in sampled_raw:
            unified = to_unified_record(raw, args.video_root)
            f.write(json.dumps(unified, ensure_ascii=False) + "\n")

    print(f"✅ 已写入 {args.output} ({len(sampled_raw)} records, JSONL)")


if __name__ == "__main__":
    main()
