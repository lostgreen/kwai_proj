#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据混合脚本：将 proxy 代理任务数据 + youcook2 时序分割数据合并，
并为每条样本设置正确的 problem_type 字段。

输出: 一个合并后的 JSONL 文件，每行包含:
    - problem_type: add / delete / replace / sort / temporal_seg
    - 其余字段与 EasyR1 格式一致

用法:
    python proxy_data/merge_datasets.py \
        --proxy proxy_data/proxy_train_easyr1.jsonl \
        --seg   proxy_data/youcook2_train_easyr1.jsonl \
        --output proxy_data/mixed_train.jsonl \
        --stats
"""

import json
import argparse
import os
import random
from collections import Counter


def load_and_tag(filepath: str, default_problem_type: str = "") -> list[dict]:
    """读取 JSONL，为每条样本设置 problem_type。"""
    samples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] {filepath} 第 {line_no} 行 JSON 解析失败: {e}")
                continue

            # 设置 problem_type
            # 优先从 metadata.task_type 获取（proxy 数据）
            meta = d.get("metadata", {})
            task_type = meta.get("task_type", "")

            if task_type:
                d["problem_type"] = task_type  # add / delete / replace / sort
            elif default_problem_type:
                d["problem_type"] = default_problem_type
            # else: keep existing problem_type

            samples.append(d)
    return samples


def main():
    parser = argparse.ArgumentParser(description="合并多来源训练数据并设置 problem_type")
    parser.add_argument("--proxy", required=True, help="代理任务数据 (proxy_train_easyr1.jsonl)")
    parser.add_argument("--seg", required=True, help="时序分割数据 (youcook2_train_easyr1.jsonl)")
    parser.add_argument("--output", "-o", required=True, help="输出合并后的 JSONL 文件")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--stats", action="store_true", help="打印统计信息")
    args = parser.parse_args()

    random.seed(args.seed)

    # 加载并标注
    print(f"📂 Loading proxy data: {args.proxy}")
    proxy_samples = load_and_tag(args.proxy, default_problem_type="")
    print(f"   → {len(proxy_samples)} samples")

    print(f"📂 Loading temporal seg data: {args.seg}")
    seg_samples = load_and_tag(args.seg, default_problem_type="temporal_seg")
    print(f"   → {len(seg_samples)} samples")

    # 合并
    all_samples = proxy_samples + seg_samples

    # 打乱（TaskHomogeneousBatchSampler 会自行按任务分组，但文件层面可先洗牌）
    random.shuffle(all_samples)

    # 写入
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\n✅ 合并完成: {len(all_samples)} 个样本 → {args.output}")

    if args.stats:
        counts = Counter(s.get("problem_type", "(empty)") for s in all_samples)
        print(f"\n📊 任务分布:")
        for task, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            pct = 100.0 * cnt / len(all_samples)
            print(f"  {task:20s}: {cnt:5d}  ({pct:.1f}%)")


if __name__ == "__main__":
    main()
