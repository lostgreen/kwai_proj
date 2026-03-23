"""
prepare_data.py — 三层分割消融实验数据准备

从 per-level _clipped.jsonl 中按指定层级/变体筛选、split val、合并输出。

用法:
    # L2 only
    python prepare_data.py --levels L2 --val-per-level 20 --output-dir ./data/exp1

    # L3 sequential only
    python prepare_data.py --levels L3_seq --val-per-level 20 --output-dir ./data/exp2

    # L2 + L3 sequential
    python prepare_data.py --levels L2 L3_seq --val-per-level 20 --output-dir ./data/exp5

    # L1 + L2 + L3 both (seq + shuffled)
    python prepare_data.py --levels L1 L2 L3_both --val-per-level 20 --output-dir ./data/exp7
"""

import argparse
import json
import os
import random
from collections import Counter


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DEFAULT_DATA_ROOT = os.path.join(REPO_ROOT, "proxy_data", "youcook2_seg_annotation", "datasets")

# 层级 -> 文件名
LEVEL_FILES = {
    "L1": "youcook2_hier_L1_train_clipped.jsonl",
    "L2": "youcook2_hier_L2_train_clipped.jsonl",
    "L3": "youcook2_hier_L3_train_clipped.jsonl",
    "L3_seg": "youcook2_hier_L3_seg_train_clipped.jsonl",
}

# 有效的 level 选项
VALID_LEVELS = {"L1", "L2", "L3_seq", "L3_shuf", "L3_both", "L3_seg"}


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def filter_l3(records, order):
    """Filter L3 records by shuffled flag."""
    if order == "seq":
        return [r for r in records if not r.get("metadata", {}).get("shuffled", False)]
    elif order == "shuf":
        return [r for r in records if r.get("metadata", {}).get("shuffled", False)]
    else:  # both
        return records


def main():
    parser = argparse.ArgumentParser(description="Hier Seg 消融实验数据准备")
    parser.add_argument(
        "--levels", nargs="+", required=True,
        help="层级列表: L1, L2, L3_seq, L3_shuf, L3_both",
    )
    parser.add_argument(
        "--val-per-level", type=int, default=20,
        help="每层验证集条数 (default: 20)",
    )
    parser.add_argument(
        "--data-root", type=str, default=DEFAULT_DATA_ROOT,
        help="per-level _clipped.jsonl 所在目录",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="输出 train.jsonl / val.jsonl 的目录",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 验证 levels
    for lv in args.levels:
        if lv not in VALID_LEVELS:
            raise ValueError(f"Unknown level: {lv}. Valid: {VALID_LEVELS}")

    rng = random.Random(args.seed)

    all_train = []
    all_val = []

    # 确定需要加载哪些文件
    levels_to_load = set()
    l3_order = None
    for lv in args.levels:
        if lv == "L3_seg":
            levels_to_load.add("L3_seg")
        elif lv.startswith("L3"):
            levels_to_load.add("L3")
            l3_order = lv.split("_")[1] if "_" in lv else "both"
        else:
            levels_to_load.add(lv)

    for level in sorted(levels_to_load):
        filepath = os.path.join(args.data_root, LEVEL_FILES[level])
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")

        records = load_jsonl(filepath)

        # L3 过滤
        if level == "L3" and l3_order:
            records = filter_l3(records, l3_order)

        rng.shuffle(records)

        # Split val
        n_val = min(args.val_per_level, len(records) // 5)
        val_records = records[:n_val]
        train_records = records[n_val:]

        all_train.extend(train_records)
        all_val.extend(val_records)

        print(f"  {level}: {len(train_records)} train + {n_val} val (total {len(records)})")

    # Shuffle combined sets
    rng.shuffle(all_train)
    rng.shuffle(all_val)

    # Write output
    train_path = os.path.join(args.output_dir, "train.jsonl")
    val_path = os.path.join(args.output_dir, "val.jsonl")
    write_jsonl(all_train, train_path)
    write_jsonl(all_val, val_path)

    # 打印统计
    train_types = Counter(r["problem_type"] for r in all_train)
    val_types = Counter(r["problem_type"] for r in all_val)

    print(f"\n{'='*50}")
    print(f"  Output: {args.output_dir}")
    print(f"  Train: {len(all_train)} samples")
    for pt, cnt in sorted(train_types.items()):
        print(f"    {pt}: {cnt}")
    print(f"  Val: {len(all_val)} samples")
    for pt, cnt in sorted(val_types.items()):
        print(f"    {pt}: {cnt}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
