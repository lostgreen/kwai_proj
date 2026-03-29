"""
prepare_v2_ablation_data.py — V2 通用 Prompt Ablation 数据准备

从 build_hier_data.py 的输出（合并的 train.jsonl / val.jsonl）中读取，
替换 prompt 为 V2 版本变体（V1-V4）。

- L3: 若为 grounding 格式 (problem_type=temporal_seg_hier_L3)，转为自由分割格式 (temporal_seg_hier_L3_seg)
- answer / videos / metadata 保持不变

用法:
    # 从 build_hier_data.py 输出读取 (推荐)
    python prepare_v2_ablation_data.py --levels L1 L2 L3 --variant V1 --data-root /path/to/base --output-dir ./data/v2_V1

    # All four variants
    python prepare_v2_ablation_data.py --levels L1 L2 L3 --variant all --data-root /path/to/base --output-dir ./data/v2_ablation
"""

import argparse
import json
import os
import random
import re
from collections import Counter


def _load_prompt_module(version: str):
    """加载 prompt 模板（V3 版本: 边界判据 + 稀疏采样感知）。"""
    from prompt_variants_v3 import (
        PROMPT_VARIANTS_V3 as variants,
        VARIANT_DESCRIPTIONS_V3 as descriptions,
        RESPONSE_LEN_HINTS,
    )
    return variants, descriptions, RESPONSE_LEN_HINTS


# 默认使用 v3
_PROMPT_VERSION = os.environ.get("PROMPT_VERSION", "v3")
PROMPT_VARIANTS_V2, VARIANT_DESCRIPTIONS_V2, RESPONSE_LEN_HINTS = _load_prompt_module(_PROMPT_VERSION)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

# problem_type → level mapping (用于从合并文件中按层过滤)
PROBLEM_TYPE_TO_LEVEL = {
    "temporal_seg_hier_L1": "L1",
    "temporal_seg_hier_L2": "L2",
    "temporal_seg_hier_L3": "L3",
    "temporal_seg_hier_L3_seg": "L3",
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
    """从 prompt 中提取 duration（秒）。适用于 L1/L2/L3。"""
    m = re.search(r"(\d+)s (?:cooking )?video clip", prompt_text)
    if m:
        return int(m.group(1))
    m = re.search(r"timestamps 0 to (\d+)", prompt_text)
    if m:
        return int(m.group(1))
    return 128


def rewrite_prompt(record: dict, level: str, variant: str) -> dict:
    """
    重写一条记录的 prompt 为指定 V2 变体。

    L1/L2/L3 全部统一使用 duration 参数。
    L3 特殊处理：problem_type 改为 L3_seg（自由分割）。
    """
    record = dict(record)
    old_prompt = record.get("prompt", "")

    template = PROMPT_VARIANTS_V2[level][variant]
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


def load_records_by_level(data_root: str, levels: list[str]) -> dict[str, list[dict]]:
    """从 build_hier_data.py 输出的合并 train.jsonl 中按 problem_type 分层读取。"""
    train_path = os.path.join(data_root, "train.jsonl")
    val_path = os.path.join(data_root, "val.jsonl")

    # 合并 train + val（后续由本脚本重新 split）
    all_records = []
    for path in [train_path, val_path]:
        if os.path.exists(path):
            all_records.extend(load_jsonl(path))

    if not all_records:
        raise FileNotFoundError(f"No data found in {data_root}/train.jsonl or val.jsonl")

    # 按 problem_type 分类
    by_level: dict[str, list[dict]] = {lv: [] for lv in levels}
    for r in all_records:
        pt = r.get("problem_type", "")
        lv = PROBLEM_TYPE_TO_LEVEL.get(pt)
        if lv and lv in by_level:
            by_level[lv].append(r)

    return by_level


def process_variant(levels, variant, data_root, rng, val_per_level, train_per_level):
    """
    处理单个 V2 variant，返回 (train, val)。

    val_per_level:   每层取多少条 val（精确值）
    train_per_level: 每层最多取多少条 train（None = 全部剩余）
    """
    all_train = []
    all_val = []

    records_by_level = load_records_by_level(data_root, sorted(levels))

    for level in sorted(levels):
        records = records_by_level[level]

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
    parser = argparse.ArgumentParser(description="V2/V3 Prompt Ablation 数据准备")
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
        "--prompt-version", type=str, default=None,
        choices=["v2", "v3"],
        help="Prompt 模板版本: v2 (语义描述) 或 v3 (边界判据+稀疏采样). "
             "默认使用环境变量 PROMPT_VERSION 或 v2",
    )
    parser.add_argument(
        "--val-per-level", type=int, default=100,
        help="每层 val 条数（精确值，默认 100）",
    )
    parser.add_argument(
        "--train-per-level", type=int, default=400,
        help="每层 train 最多条数（默认 400；设为 -1 表示用全部剩余数据）",
    )
    parser.add_argument("--data-root", type=str, required=True,
                        help="build_hier_data.py 输出目录（含 train.jsonl / val.jsonl）")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 运行时切换 prompt 版本（命令行优先于环境变量）
    global PROMPT_VARIANTS_V2, VARIANT_DESCRIPTIONS_V2, RESPONSE_LEN_HINTS
    if args.prompt_version:
        PROMPT_VARIANTS_V2, VARIANT_DESCRIPTIONS_V2, RESPONSE_LEN_HINTS = _load_prompt_module(args.prompt_version)
        print(f"[prepare] Using prompt version: {args.prompt_version}")

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
