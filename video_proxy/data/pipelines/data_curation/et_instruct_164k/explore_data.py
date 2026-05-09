"""
ET-Instruct-164K 数据探索脚本

用法:
    python explore_data.py --json_path /path/to/et_instruct_164k_vid.json [--sample_n 5]

功能:
    1. 打印 JSON 顶层结构 & 样本数
    2. 分析字段分布
    3. 按 domain 统计
    4. 抽样打印
    5. 输出 raw_stats.json 和 domain_stats.json
"""

import json
import argparse
import os
import random
from collections import Counter, defaultdict
from pathlib import Path


def load_json(path: str):
    """加载 JSON 文件，支持 list 或 dict 格式。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def print_structure(data, max_depth: int = 3, prefix: str = ""):
    """递归打印 JSON 结构（不打印具体值，只打印类型和键）。"""
    if isinstance(data, dict):
        print(f"{prefix}dict with {len(data)} keys: {list(data.keys())[:20]}")
        if max_depth > 0:
            for k in list(data.keys())[:5]:
                print(f"{prefix}  [{k}]:")
                print_structure(data[k], max_depth - 1, prefix + "    ")
    elif isinstance(data, list):
        print(f"{prefix}list with {len(data)} items")
        if max_depth > 0 and len(data) > 0:
            print(f"{prefix}  [0]:")
            print_structure(data[0], max_depth - 1, prefix + "    ")
    else:
        print(f"{prefix}{type(data).__name__}: {str(data)[:100]}")


def analyze_samples(samples: list, output_dir: str):
    """分析样本列表，输出统计信息。"""
    print(f"\n{'='*60}")
    print(f"总样本数: {len(samples)}")
    print(f"{'='*60}\n")

    # 字段统计
    field_counter = Counter()
    for s in samples:
        if isinstance(s, dict):
            for k in s.keys():
                field_counter[k] += 1

    print("字段分布:")
    for field, count in field_counter.most_common():
        pct = count / len(samples) * 100
        print(f"  {field}: {count} ({pct:.1f}%)")

    # 尝试识别 domain / video_source 字段
    domain_field = None
    for candidate in ["source", "domain", "video_source", "dataset", "data_source"]:
        if candidate in field_counter:
            domain_field = candidate
            break

    # 如果没有直接的 domain 字段，尝试从 video 路径推断
    domain_stats = defaultdict(int)
    if domain_field:
        for s in samples:
            domain_stats[s.get(domain_field, "unknown")] += 1
        print(f"\nDomain 分布 (字段: {domain_field}):")
    else:
        # 尝试从 video 路径推断 domain
        video_field = None
        for candidate in ["video", "video_path", "video_id", "vid"]:
            if candidate in field_counter:
                video_field = candidate
                break
        if video_field:
            for s in samples:
                v = s.get(video_field, "")
                if isinstance(v, str) and "/" in v:
                    domain = v.split("/")[0]
                    domain_stats[domain] += 1
                elif isinstance(v, list) and len(v) > 0:
                    domain = str(v[0]).split("/")[0]
                    domain_stats[domain] += 1
                else:
                    domain_stats["unknown"] += 1
            print(f"\nDomain 分布 (从 {video_field} 路径推断):")
        else:
            print("\n⚠️ 未找到 domain 或 video 字段，无法统计域分布")

    for domain, count in sorted(domain_stats.items(), key=lambda x: -x[1]):
        pct = count / len(samples) * 100
        print(f"  {domain}: {count} ({pct:.1f}%)")

    # 时长统计（如果有）
    duration_field = None
    for candidate in ["duration", "video_duration", "length"]:
        if candidate in field_counter:
            duration_field = candidate
            break
    if duration_field:
        durations = [s[duration_field] for s in samples if duration_field in s and isinstance(s[duration_field], (int, float))]
        if durations:
            print(f"\n时长统计 (字段: {duration_field}):")
            print(f"  min: {min(durations):.1f}s, max: {max(durations):.1f}s")
            print(f"  mean: {sum(durations)/len(durations):.1f}s")
            print(f"  median: {sorted(durations)[len(durations)//2]:.1f}s")

    # 保存统计
    os.makedirs(output_dir, exist_ok=True)
    stats = {
        "total_samples": len(samples),
        "fields": dict(field_counter),
        "domain_stats": dict(domain_stats),
    }
    if duration_field and durations:
        stats["duration"] = {
            "min": min(durations),
            "max": max(durations),
            "mean": sum(durations) / len(durations),
            "count_with_duration": len(durations),
        }

    with open(os.path.join(output_dir, "raw_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "domain_stats.json"), "w", encoding="utf-8") as f:
        json.dump(dict(domain_stats), f, indent=2, ensure_ascii=False)

    print(f"\n统计已保存到: {output_dir}/raw_stats.json, domain_stats.json")
    return stats


def print_samples(samples: list, n: int = 5):
    """随机抽样打印。"""
    n = min(n, len(samples))
    chosen = random.sample(samples, n)
    print(f"\n{'='*60}")
    print(f"随机抽样 {n} 条:")
    print(f"{'='*60}")
    for i, s in enumerate(chosen):
        print(f"\n--- Sample {i+1} ---")
        print(json.dumps(s, indent=2, ensure_ascii=False)[:2000])


def main():
    parser = argparse.ArgumentParser(description="ET-Instruct-164K 数据探索")
    parser.add_argument("--json_path", required=True, help="JSON 文件路径")
    parser.add_argument("--sample_n", type=int, default=5, help="抽样打印数量")
    parser.add_argument("--output_dir", default="results", help="统计输出目录")
    args = parser.parse_args()

    print(f"加载: {args.json_path}")
    data = load_json(args.json_path)

    print(f"\n顶层结构:")
    print_structure(data)

    # 确定样本列表
    if isinstance(data, list):
        samples = data
    elif isinstance(data, dict):
        # 可能需要从某个键取出样本列表
        if len(data) == 1:
            samples = list(data.values())[0]
        else:
            # 按长度找最大的 list 值
            max_key = max(data.keys(), key=lambda k: len(data[k]) if isinstance(data[k], list) else 0)
            samples = data[max_key] if isinstance(data[max_key], list) else list(data.values())
            print(f"使用键 '{max_key}' 作为样本列表")
    else:
        print("⚠️ 无法识别数据格式")
        return

    analyze_samples(samples, args.output_dir)
    print_samples(samples, args.sample_n)


if __name__ == "__main__":
    main()
