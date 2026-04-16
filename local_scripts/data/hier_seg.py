"""Hierarchical Segmentation data handler.

Train: External train_all.jsonl (~20k) → stratified sample by problem_type
Val: External val_all.jsonl → stratified sample by problem_type
problem_types: temporal_seg_hier_L1, temporal_seg_hier_L2, temporal_seg_hier_L3_seg
"""

from __future__ import annotations

import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

from .common import load_jsonl, stratified_sample, write_jsonl

NAME = "hier_seg"
PROBLEM_TYPES = [
    "temporal_seg_hier_L1",
    "temporal_seg_hier_L2",
    "temporal_seg_hier_L3_seg",
]

# ---- 文件命名约定 ----
_VAL_PREFIX = "hier_seg_val"


def add_cli_args(parser: ArgumentParser) -> None:
    g = parser.add_argument_group("Hierarchical Segmentation")
    g.add_argument("--hier-train", help="Hier Seg train JSONL (full, e.g. train_all.jsonl)")
    g.add_argument("--hier-val-source", help="Hier Seg val source (e.g. val_all.jsonl)")
    g.add_argument("--hier-target", type=int, default=5000, help="Hier Seg train sample target")
    g.add_argument("--val-hier-n", type=int, default=150, help="Hier Seg val sample size")


def setup_base(data_root: str, args: Namespace, force: bool, seed: int) -> None:
    val_dir = os.path.join(data_root, "val")
    os.makedirs(val_dir, exist_ok=True)

    val_n = args.val_hier_n
    hier_val = os.path.join(val_dir, f"{_VAL_PREFIX}_{val_n}.jsonl")
    if force or not os.path.exists(hier_val):
        print(f"\n>>> Hier Seg val (sample {val_n}, stratified by problem_type)...")
        source = args.hier_val_source
        if not source or not os.path.exists(source):
            print(f"  [hier_seg] WARN: val source not found: {source}")
            return
        all_records = load_jsonl(source)
        sampled = stratified_sample(all_records, val_n, key="problem_type", seed=seed)
        write_jsonl(sampled, hier_val)
    else:
        print(f"\n>>> Hier Seg val exists: {hier_val} — skip")


def load_train(data_root: str, args: Namespace) -> list[dict]:
    path = args.hier_train
    if not path or not os.path.exists(path):
        print(f"  [hier_seg] WARN: train source not found: {path}")
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
