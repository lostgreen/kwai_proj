"""Temporal Grounding data handler.

Train: Pre-built by proxy_data/temporal_grounding/run_pipeline.sh, copied into base/
Val: TVGBench pre-built output → random sample

前置条件:
  数据已由 proxy_data/temporal_grounding/run_pipeline.sh 生成完毕，
  本模块只负责将生成好的数据 copy 到 base/ 并从 TVGBench 采样 val。
"""

from __future__ import annotations

import os
import shutil
from argparse import ArgumentParser, Namespace
from pathlib import Path

from .common import load_jsonl, random_sample, write_jsonl

NAME = "tg"
PROBLEM_TYPES = ["temporal_grounding"]

# ---- 文件命名约定 ----
_TRAIN_FILE = "tg_train_no_tvgbench.jsonl"
_VAL_PREFIX = "tvgbench_val"


def add_cli_args(parser: ArgumentParser) -> None:
    g = parser.add_argument_group("Temporal Grounding")
    g.add_argument(
        "--tg-train-source",
        help="Pre-built TG train JSONL (from run_pipeline.sh, e.g. tg_train_max256s_validated.jsonl)",
    )
    g.add_argument(
        "--tg-tvgbench-source",
        help="Pre-built TVGBench JSONL (full, for val sampling)",
    )
    g.add_argument("--val-tg-n", type=int, default=150, help="TVGBench val sample size")


def setup_base(data_root: str, args: Namespace, force: bool, seed: int) -> None:
    base_dir = os.path.join(data_root, "base")
    val_dir = os.path.join(data_root, "val")
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # ── TG train (copy pre-built) ──
    tg_train = os.path.join(base_dir, _TRAIN_FILE)
    if force or not os.path.exists(tg_train):
        source = getattr(args, "tg_train_source", None)
        if not source or not os.path.exists(source):
            print(f"  [tg] WARN: TG train source not found: {source}")
            print("  请先运行: bash proxy_data/temporal_grounding/run_pipeline.sh")
            return
        print(f"\n>>> TG train (copy from {source})...")
        shutil.copy2(source, tg_train)
        print(f"  TG train: {sum(1 for _ in open(tg_train))} samples")
    else:
        print(f"\n>>> TG train exists: {tg_train} — skip")

    # ── TVGBench val (sample from pre-built) ──
    val_n = args.val_tg_n
    tvg_val = os.path.join(val_dir, f"{_VAL_PREFIX}_{val_n}.jsonl")
    if force or not os.path.exists(tvg_val):
        source = getattr(args, "tg_tvgbench_source", None)
        if not source or not os.path.exists(source):
            print(f"  [tg] WARN: TVGBench source not found: {source}")
            print("  请先运行: TVGBENCH_JSON=... bash proxy_data/temporal_grounding/run_pipeline.sh")
            return
        print(f"\n>>> TVGBench val (sample {val_n} from {source})...")
        tvg_all = load_jsonl(source)
        sampled = random_sample(tvg_all, val_n, seed)
        write_jsonl(sampled, tvg_val)
        print(f"  TVGBench: {len(tvg_all)} -> {len(sampled)}")
    else:
        print(f"\n>>> TVGBench val exists: {tvg_val} — skip")


def load_train(data_root: str, args: Namespace) -> list[dict]:
    path = os.path.join(data_root, "base", _TRAIN_FILE)
    return load_jsonl(path)


def sample_train(records: list[dict], target: int, seed: int) -> list[dict]:
    # TG 全量使用 (~2.2k)，不采样
    return list(records)


def load_val(data_root: str) -> list[dict]:
    val_dir = os.path.join(data_root, "val")
    for f in sorted(Path(val_dir).glob(f"{_VAL_PREFIX}_*.jsonl")):
        return load_jsonl(str(f))
    return []
