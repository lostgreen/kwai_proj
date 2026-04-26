"""Temporal AoT data handler.

Train: External AoT JSONL -> stratified sample by problem_type/domain_l1
Val: External val JSONL, or a stratified sample from train when no val source is provided
problem_types: seg_aot_*
"""

from __future__ import annotations

import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

from .common import load_jsonl, nested_stratified_sample, write_jsonl

NAME = "aot"

_VAL_PREFIX = "aot_val"


def add_cli_args(parser: ArgumentParser) -> None:
    g = parser.add_argument_group("Temporal AoT")
    g.add_argument("--aot-train", help="Temporal AoT train JSONL")
    g.add_argument("--aot-val-source", help="Temporal AoT val source JSONL")
    g.add_argument("--aot-target", type=int, default=10000, help="Temporal AoT train sample target")
    g.add_argument("--val-aot-n", type=int, default=300, help="Temporal AoT val sample size")


def _load_source(path: str | None, label: str) -> list[dict]:
    if not path or not os.path.exists(path):
        print(f"  [aot] WARN: {label} source not found: {path}")
        return []
    return load_jsonl(path)


def setup_base(data_root: str, args: Namespace, force: bool, seed: int) -> None:
    val_dir = os.path.join(data_root, "val")
    os.makedirs(val_dir, exist_ok=True)

    val_n = args.val_aot_n
    aot_val = os.path.join(val_dir, f"{_VAL_PREFIX}_{val_n}.jsonl")
    if force or not os.path.exists(aot_val):
        print(f"\n>>> Temporal AoT val (sample {val_n}, stratified by problem_type × domain_l1)...")
        source = getattr(args, "aot_val_source", None) or getattr(args, "aot_train", None)
        all_records = _load_source(source, "val/train")
        if not all_records:
            return
        sampled = nested_stratified_sample(
            all_records,
            val_n,
            key="problem_type",
            nested_key="domain_l1",
            seed=seed,
        )
        write_jsonl(sampled, aot_val)
    else:
        print(f"\n>>> Temporal AoT val exists: {aot_val} — skip")


def load_train(data_root: str, args: Namespace) -> list[dict]:
    return _load_source(getattr(args, "aot_train", None), "train")


def sample_train(records: list[dict], target: int, seed: int) -> list[dict]:
    if target <= 0:
        return list(records)
    return nested_stratified_sample(
        records,
        target,
        key="problem_type",
        nested_key="domain_l1",
        seed=seed,
    )


def load_val(data_root: str, args: Namespace | None = None) -> list[dict]:
    target_n = getattr(args, "val_aot_n", 300) if args is not None else 300
    seed = getattr(args, "seed", 42) if args is not None else 42

    records: list[dict] = []
    if args is not None:
        source = getattr(args, "aot_val_source", None)
        if source and os.path.exists(source):
            records = load_jsonl(source)
            print(f"  [aot] val from --aot-val-source: {source} ({len(records)})")

    if not records:
        val_dir = os.path.join(data_root, "val")
        exact = Path(val_dir) / f"{_VAL_PREFIX}_{target_n}.jsonl"
        if exact.exists():
            records = load_jsonl(str(exact))
        else:
            for f in sorted(Path(val_dir).glob(f"{_VAL_PREFIX}_*.jsonl")):
                records = load_jsonl(str(f))
                break

    if 0 < len(records) < target_n and args is not None:
        train_path = getattr(args, "aot_train", None)
        if train_path and os.path.exists(train_path):
            val_prompts = {r.get("prompt", "") for r in records}
            train_all = load_jsonl(train_path)
            candidates = [r for r in train_all if r.get("prompt", "") not in val_prompts]
            need = target_n - len(records)
            supplement = nested_stratified_sample(
                candidates,
                need,
                key="problem_type",
                nested_key="domain_l1",
                seed=seed,
            )
            print(f"  [aot] val supplemented from train: +{len(supplement)} (total {len(records) + len(supplement)})")
            records.extend(supplement)

    if not records and args is not None:
        train_path = getattr(args, "aot_train", None)
        train_all = _load_source(train_path, "train")
        if train_all:
            records = nested_stratified_sample(
                train_all,
                target_n,
                key="problem_type",
                nested_key="domain_l1",
                seed=seed,
            )

    return records
