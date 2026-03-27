"""
prepare_v2_ablation_data.py — V2 通用 Prompt Ablation 数据准备

从现有 L1/L2/L3 JSONL 中读取，替换 prompt 为 V2 版本变体（V1-V4）。
- L3: 从 grounding 格式（含 action query 列表）转为自由分割格式（只需 duration）
- answer / videos / metadata 保持不变

用法:
    # V1 (baseline) for L2 only
    python prepare_v2_ablation_data.py --levels L2 --variant V1 --output-dir ./data/v2_V1

    # V4 (gran+cot) for L1+L2+L3
    python prepare_v2_ablation_data.py --levels L1 L2 L3 --variant V4 --output-dir ./data/v2_V4

    # All four variants for L1+L2+L3 at once
    python prepare_v2_ablation_data.py --levels L1 L2 L3 --variant all --output-dir ./data/v2_ablation
"""

import argparse
import json
import os
import random
import re
from collections import Counter

from prompt_variants_v2 import PROMPT_VARIANTS_V2, VARIANT_DESCRIPTIONS_V2, RESPONSE_LEN_HINTS


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
DEFAULT_DATA_ROOT = os.path.join(REPO_ROOT, "proxy_data", "youcook2_seg_annotation", "datasets")

LEVEL_FILES = {
    "L1": "youcook2_hier_L1_train_clipped.jsonl",
    "L2": "youcook2_hier_L2_train_clipped.jsonl",
    "L3": "youcook2_hier_L3_train_clipped.jsonl",
}

# L3 的新 problem_type（自由分割）
L3_PROBLEM_TYPE_NEW = "temporal_seg_hier_L3_seg"


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


def extract_duration_from_prompt(prompt_text: str) -> int:
    """从 prompt 中提取 duration（秒）。"""
    m = re.search(r"(\d+)s (?:cooking )?video clip", prompt_text)
    if m:
        return int(m.group(1))
    m = re.search(r"timestamps 0 to (\d+)", prompt_text)
    if m:
        return int(m.group(1))
    return 128


def extract_n_frames_from_prompt(prompt_text: str) -> int:
    """从 L1 prompt 中提取 n_frames。"""
    m = re.search(r"(\d+) frames uniformly sampled", prompt_text)
    if m:
        return int(m.group(1))
    m = re.search(r"numbered 1 to (\d+)", prompt_text)
    if m:
        return int(m.group(1))
    return 256


def rewrite_prompt(record: dict, level: str, variant: str) -> dict:
    """
    重写一条记录的 prompt 为指定 V2 变体。

    L3 特殊处理：
      - 旧: prompt 含 action query 列表（grounding 格式）
      - 新: 仅使用 duration 参数（自由分割格式），problem_type 改为 L3_seg
    """
    record = dict(record)
    old_prompt = record.get("prompt", "")

    template = PROMPT_VARIANTS_V2[level][variant]

    if level == "L1":
        n_frames = extract_n_frames_from_prompt(old_prompt)
        new_prompt_body = template.format(n_frames=n_frames)
    else:  # L2 or L3 — both use duration
        duration = extract_duration_from_prompt(old_prompt)
        new_prompt_body = template.format(duration=duration)

    # 重建完整 prompt（带 <video> 前缀）
    new_prompt = f"<video>\n\n{new_prompt_body}"

    record["prompt"] = new_prompt
    record["messages"] = [{"role": "user", "content": new_prompt}]

    # L3: 更新 problem_type 为分割版本（对应 F1-IoU reward）
    if level == "L3":
        record["problem_type"] = L3_PROBLEM_TYPE_NEW

    return record


def process_variant(levels, variant, data_root, rng, val_per_level, train_per_level):
    """
    处理单个 V2 variant，返回 (train, val)。

    val_per_level:   每层取多少条 val（精确值）
    train_per_level: 每层最多取多少条 train（None = 全部剩余）
    """
    all_train = []
    all_val = []

    for level in sorted(levels):
        filepath = os.path.join(data_root, LEVEL_FILES[level])
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")

        records = load_jsonl(filepath)

        # L3: shuffled 版本只保留 sequential（一个 event clip 一条记录，不重复）
        if level == "L3":
            records = [r for r in records if not r.get("metadata", {}).get("shuffled", False)]
            print(f"  [L3] Filtered to sequential only: {len(records)} records")

        rewritten = [rewrite_prompt(r, level, variant) for r in records]

        rng_copy = random.Random(rng.random())
        rng_copy.shuffle(rewritten)

        # --- val: 精确 val_per_level 条（不超过数据量的 1/5）---
        n_val = min(val_per_level, len(rewritten) // 5)
        level_val = rewritten[:n_val]
        level_train_pool = rewritten[n_val:]

        # --- train: 最多 train_per_level 条（None = 全部）---
        if train_per_level is not None:
            level_train = level_train_pool[:train_per_level]
        else:
            level_train = level_train_pool

        all_val.extend(level_val)
        all_train.extend(level_train)

        print(f"  {level}/{variant}: {len(level_train)} train + {len(level_val)} val "
              f"(available={len(rewritten)}, response_len={RESPONSE_LEN_HINTS[variant]})")

    rng.shuffle(all_train)
    rng.shuffle(all_val)
    return all_train, all_val


def main():
    parser = argparse.ArgumentParser(description="V2 Prompt Ablation 数据准备")
    parser.add_argument(
        "--levels", nargs="+", required=True,
        choices=["L1", "L2", "L3"],
        help="层级列表，支持 L1 L2 L3 任意组合",
    )
    parser.add_argument(
        "--variant", type=str, required=True,
        choices=["V1", "V2", "V3", "V4", "all"],
        help="Prompt 变体 (V1-V4) 或 all",
    )
    parser.add_argument(
        "--val-per-level", type=int, default=100,
        help="每层 val 条数（精确值，默认 100）",
    )
    parser.add_argument(
        "--train-per-level", type=int, default=400,
        help="每层 train 最多条数（默认 400；设为 -1 表示用全部剩余数据）",
    )
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    variants = ["V1", "V2", "V3", "V4"] if args.variant == "all" else [args.variant]
    train_per_level = None if args.train_per_level < 0 else args.train_per_level

    for variant in variants:
        print(f"\n{'=' * 60}")
        print(f"  Variant {variant}: {VARIANT_DESCRIPTIONS_V2[variant]}")
        print(f"  val_per_level={args.val_per_level}, train_per_level={train_per_level or 'all'}")
        print(f"{'=' * 60}")

        out_dir = os.path.join(args.output_dir, variant) if args.variant == "all" else args.output_dir

        train, val = process_variant(
            args.levels, variant, args.data_root,
            random.Random(rng.random()), args.val_per_level, train_per_level,
        )

        train_path = os.path.join(out_dir, "train.jsonl")
        val_path = os.path.join(out_dir, "val.jsonl")
        write_jsonl(train, train_path)
        write_jsonl(val, val_path)

        train_types = Counter(r["problem_type"] for r in train)
        val_types = Counter(r["problem_type"] for r in val)

        print(f"\n  Output: {out_dir}")
        print(f"  Train: {len(train)} samples")
        for pt, cnt in sorted(train_types.items()):
            print(f"    {pt}: {cnt}")
        print(f"  Val: {len(val)} samples")
        for pt, cnt in sorted(val_types.items()):
            print(f"    {pt}: {cnt}")

        if train:
            ex = train[0]
            print(f"\n  --- Example prompt (first 300 chars) ---")
            print(f"  {ex['prompt'][:300]}")
            print(f"  --- Answer ---")
            print(f"  {ex['answer']}")


if __name__ == "__main__":
    main()
