"""Temporal Grounding data handler.

Train: 直接复制 run_pipeline.sh 预构建的 TimeRFT validated JSONL
Val: 从 run_pipeline.sh 预构建的 TVGBench validated JSONL 中随机采样

前置条件:
  bash proxy_data/temporal_grounding/run_pipeline.sh
  → tg_timerft_max256s_validated.jsonl  (train)
  → tg_tvgbench_max256s_validated.jsonl (val)
"""

from __future__ import annotations

import os
import shutil
from argparse import ArgumentParser, Namespace
from pathlib import Path

from .common import load_jsonl, random_sample, write_jsonl

NAME = "tg"
PROBLEM_TYPES = ["temporal_grounding"]

_TRAIN_FILE = "tg_train.jsonl"
_VAL_PREFIX = "tg_val"


def add_cli_args(parser: ArgumentParser) -> None:
    g = parser.add_argument_group("Temporal Grounding")
    g.add_argument(
        "--tg-train-source",
        help="Pre-built TimeRFT JSONL (e.g. tg_timerft_max256s_validated.jsonl)",
    )
    g.add_argument(
        "--tg-tvgbench-source",
        help="Pre-built TVGBench JSONL (e.g. tg_tvgbench_max256s_validated.jsonl)",
    )
    g.add_argument("--val-tg-n", type=int, default=150, help="TVGBench val sample size")


def setup_base(data_root: str, args: Namespace, force: bool, seed: int) -> None:
    base_dir = os.path.join(data_root, "base")
    val_dir = os.path.join(data_root, "val")
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # ── Train: copy pre-built TimeRFT ──
    tg_train = os.path.join(base_dir, _TRAIN_FILE)
    if force or not os.path.exists(tg_train):
        source = getattr(args, "tg_train_source", None)
        if not source or not os.path.exists(source):
            print(f"  [tg] WARN: train source not found: {source}")
            print("  请先运行: bash proxy_data/temporal_grounding/run_pipeline.sh")
            return
        print(f"\n>>> TG train: copy from {source}")
        shutil.copy2(source, tg_train)
        print(f"  {sum(1 for _ in open(tg_train))} samples")
    else:
        print(f"\n>>> TG train exists: {tg_train} — skip")

    # ── Val: sample from pre-built TVGBench ──
    val_n = args.val_tg_n
    tg_val = os.path.join(val_dir, f"{_VAL_PREFIX}_{val_n}.jsonl")
    if force or not os.path.exists(tg_val):
        tvg_source = getattr(args, "tg_tvgbench_source", None)
        if not tvg_source or not os.path.exists(tvg_source):
            print(f"  [tg] WARN: TVGBench source not found: {tvg_source}")
            print("  请先运行: TVGBENCH_JSON=... bash proxy_data/temporal_grounding/run_pipeline.sh")
            return
        print(f"\n>>> TG val: sample {val_n} from {tvg_source}")
        all_records = load_jsonl(tvg_source)
        sampled = random_sample(all_records, val_n, seed)
        write_jsonl(sampled, tg_val)
        print(f"  TVGBench: {len(all_records)} -> {len(sampled)}")
    else:
        print(f"\n>>> TG val exists: {tg_val} — skip")


def load_train(data_root: str, args: Namespace) -> list[dict]:
    return load_jsonl(os.path.join(data_root, "base", _TRAIN_FILE))


def sample_train(records: list[dict], target: int, seed: int) -> list[dict]:
    # TG 全量使用 (~2.2k)，不采样
    return list(records)


def load_val(data_root: str) -> list[dict]:
    val_dir = os.path.join(data_root, "val")
    for f in sorted(Path(val_dir).glob(f"{_VAL_PREFIX}_*.jsonl")):
        return load_jsonl(str(f))
    return []
