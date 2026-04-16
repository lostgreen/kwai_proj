"""Event Logic data handler.

Train: External train.jsonl from event_logic VLM pipeline → stratified sample
Val: External val.jsonl or sample from train → stratified by problem_type
problem_types: event_logic_sort
"""

from __future__ import annotations

import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

from .common import load_jsonl, stratified_sample, write_jsonl

NAME = "event_logic"
PROBLEM_TYPES = [
    "event_logic_sort",
]

# ---- 文件命名约定 ----
_VAL_PREFIX = "event_logic_val"


def add_cli_args(parser: ArgumentParser) -> None:
    g = parser.add_argument_group("Event Logic")
    g.add_argument("--el-train", help="Event Logic train JSONL")
    g.add_argument("--el-val-source", help="Event Logic val source JSONL (or sample from train)")
    g.add_argument("--el-target", type=int, default=2000, help="Event Logic train sample target")
    g.add_argument("--val-el-n", type=int, default=100, help="Event Logic val sample size")


def setup_base(data_root: str, args: Namespace, force: bool, seed: int) -> None:
    val_dir = os.path.join(data_root, "val")
    os.makedirs(val_dir, exist_ok=True)

    val_n = args.val_el_n
    el_val = os.path.join(val_dir, f"{_VAL_PREFIX}_{val_n}.jsonl")
    if force or not os.path.exists(el_val):
        print(f"\n>>> Event Logic val (sample {val_n}, stratified by problem_type)...")
        # 优先使用专用 val source，否则从 train 采样
        source = getattr(args, "el_val_source", None) or getattr(args, "el_train", None)
        if not source or not os.path.exists(source):
            print(f"  [event_logic] WARN: val/train source not found: {source}")
            return
        all_records = load_jsonl(source)
        sampled = stratified_sample(all_records, val_n, key="problem_type", seed=seed)
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
    return stratified_sample(records, target, key="problem_type", seed=seed)


def load_val(data_root: str) -> list[dict]:
    val_dir = os.path.join(data_root, "val")
    for f in sorted(Path(val_dir).glob(f"{_VAL_PREFIX}_*.jsonl")):
        return load_jsonl(str(f))
    return []
