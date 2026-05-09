"""Event Logic data handler.

Train: External train_*.jsonl from event_logic VLM pipeline → domain_l1 balanced sample
Val: External val_*.jsonl or sample from train → stratified by problem_type + domain_l1

problem_types: event_logic_predict_next, event_logic_fill_blank, event_logic_sort

VLM 管线输出 (--el-train 指定路径):
  train_predict_next.jsonl / val_predict_next.jsonl
  train_fill_blank.jsonl   / val_fill_blank.jsonl
  train_sort.jsonl         / val_sort.jsonl
  train.jsonl / val.jsonl  (全量合并)

消融时 --el-train 可指向单任务文件 (train_predict_next.jsonl) 或全量 (train.jsonl)。
采样按 problem_type × domain_l1 两级分层。
"""

from __future__ import annotations

import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

from .common import load_jsonl, nested_stratified_sample, write_jsonl

NAME = "event_logic"
PROBLEM_TYPES = [
    "event_logic_predict_next",
    "event_logic_fill_blank",
    "event_logic_sort",
]

# ---- 文件命名约定 ----
_VAL_PREFIX = "event_logic_val"


def add_cli_args(parser: ArgumentParser) -> None:
    g = parser.add_argument_group("Event Logic")
    g.add_argument("--el-train", help="Event Logic train JSONL (单任务或全量)")
    g.add_argument("--el-val-source", help="Event Logic val source JSONL (单任务或全量)")
    g.add_argument("--el-target", type=int, default=2000, help="Event Logic train sample target")
    g.add_argument("--val-el-n", type=int, default=100, help="Event Logic val sample size")


def setup_base(data_root: str, args: Namespace, force: bool, seed: int) -> None:
    val_dir = os.path.join(data_root, "val")
    os.makedirs(val_dir, exist_ok=True)

    val_n = args.val_el_n
    el_val = os.path.join(val_dir, f"{_VAL_PREFIX}_{val_n}.jsonl")
    if force or not os.path.exists(el_val):
        print(f"\n>>> Event Logic val (sample {val_n}, stratified by problem_type × domain_l1)...")
        source = getattr(args, "el_val_source", None) or getattr(args, "el_train", None)
        if not source or not os.path.exists(source):
            print(f"  [event_logic] WARN: val/train source not found: {source}")
            return
        all_records = load_jsonl(source)
        sampled = nested_stratified_sample(
            all_records, val_n,
            key="problem_type", nested_key="domain_l1", seed=seed,
        )
        write_jsonl(sampled, el_val)
    else:
        print(f"\n>>> Event Logic val exists: {el_val} — skip")


def load_train(data_root: str, args: Namespace) -> list[dict]:
    path = args.el_train
    if not path or not os.path.exists(path):
        print(f"  [event_logic] WARN: train source not found: {path}")
        return []
    return load_jsonl(path)


def sample_train(records: list[dict], target: int, seed: int) -> list[dict]:
    if target <= 0:
        return list(records)
    return nested_stratified_sample(
        records, target,
        key="problem_type", nested_key="domain_l1", seed=seed,
    )


def load_val(data_root: str, args: Namespace | None = None) -> list[dict]:
    """加载 val 数据。

    优先使用 --el-val-source，不够 --val-el-n 时从 --el-train 补充（domain_l1 均衡）。
    """
    target_n = getattr(args, "val_el_n", 100) if args is not None else 100
    seed = getattr(args, "seed", 42) if args is not None else 42

    records: list[dict] = []

    # 1. 尝试 --el-val-source
    if args is not None:
        source = getattr(args, "el_val_source", None)
        if source and os.path.exists(source):
            records = load_jsonl(source)
            print(f"  [event_logic] val from --el-val-source: {source} ({len(records)})")

    # 2. 回退: 共享 val 目录
    if not records:
        val_dir = os.path.join(data_root, "val")
        for f in sorted(Path(val_dir).glob(f"{_VAL_PREFIX}_*.jsonl")):
            records = load_jsonl(str(f))
            break

    # 3. 不够时从 train 补充
    if 0 < len(records) < target_n and args is not None:
        train_path = getattr(args, "el_train", None)
        if train_path and os.path.exists(train_path):
            # 用 prompt 去重（避免 val 和 train 重叠）
            val_prompts = {r.get("prompt", "") for r in records}
            train_all = load_jsonl(train_path)
            candidates = [r for r in train_all if r.get("prompt", "") not in val_prompts]
            need = target_n - len(records)
            supplement = nested_stratified_sample(
                candidates, need,
                key="problem_type", nested_key="domain_l1", seed=seed,
            )
            print(f"  [event_logic] val supplemented from train: +{len(supplement)} (total {len(records) + len(supplement)})")
            records.extend(supplement)

    return records
