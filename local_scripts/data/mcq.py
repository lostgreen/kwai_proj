"""LLaVA Video MCQ data handler.

Train: Copy from LLaVA pipeline output (train_final.jsonl)
Val: Stratified sample from train by metadata.data_source
"""

from __future__ import annotations

import os
import shutil
from argparse import ArgumentParser, Namespace
from pathlib import Path

from .common import load_jsonl, nested_stratified_sample, random_sample, write_jsonl

NAME = "mcq"
PROBLEM_TYPES = ["llava_mcq"]

# ---- 文件命名约定 ----
_TRAIN_FILE = "mcq_train_filtered.jsonl"
_VAL_PREFIX = "mcq_val"


def add_cli_args(parser: ArgumentParser) -> None:
    g = parser.add_argument_group("LLaVA MCQ")
    g.add_argument("--mcq-source", help="MCQ train_final.jsonl path")
    g.add_argument("--val-mcq-n", type=int, default=150, help="MCQ val sample size")


def setup_base(data_root: str, args: Namespace, force: bool, seed: int) -> None:
    base_dir = os.path.join(data_root, "base")
    val_dir = os.path.join(data_root, "val")
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # ── MCQ train ──
    mcq_train = os.path.join(base_dir, _TRAIN_FILE)
    if force or not os.path.exists(mcq_train):
        print("\n>>> MCQ train (copy from source)...")
        if not args.mcq_source or not os.path.exists(args.mcq_source):
            print(f"  [mcq] WARN: source not found: {args.mcq_source}")
            return
        shutil.copy2(args.mcq_source, mcq_train)
        print(f"  MCQ train: {sum(1 for _ in open(mcq_train))} samples")
    else:
        print(f"\n>>> MCQ train exists: {mcq_train} — skip")

    # ── MCQ val (按 data_source 分层) ──
    val_n = args.val_mcq_n
    mcq_val = os.path.join(val_dir, f"{_VAL_PREFIX}_{val_n}.jsonl")
    if force or not os.path.exists(mcq_val):
        print(f"\n>>> MCQ val (sample {val_n}, stratified by data_source)...")
        if os.path.exists(mcq_train):
            records = load_jsonl(mcq_train)
            sampled = nested_stratified_sample(
                records, val_n,
                key="problem_type",
                nested_key="data_source",
                seed=seed,
            )
            write_jsonl(sampled, mcq_val)
        else:
            print("  [mcq] WARN: MCQ train not available")
    else:
        print(f"\n>>> MCQ val exists: {mcq_val} — skip")


def load_train(data_root: str, args: Namespace) -> list[dict]:
    path = os.path.join(data_root, "base", _TRAIN_FILE)
    return load_jsonl(path)


def sample_train(records: list[dict], target: int, seed: int) -> list[dict]:
    # MCQ 全量使用，不采样
    return list(records)


def load_val(data_root: str) -> list[dict]:
    val_dir = os.path.join(data_root, "val")
    for f in sorted(Path(val_dir).glob(f"{_VAL_PREFIX}_*.jsonl")):
        return load_jsonl(str(f))
    return []
