#!/usr/bin/env python3
"""
多任务数据混合工具。

子命令:
    setup   — 生成基座数据 (各任务的 train + val)
    mix     — 混合实验训练数据 + 合并 val
    check   — 验证 base/val 文件是否存在

用法:
    # Step 1: 一键生成 base + val (只需运行一次)
    python3 local_scripts/data/mixer.py setup \\
        --data-root /path/to/data \\
        --tasks tg mcq hier_seg \\
        --tg-timerft-json ... --tg-tvgbench-json ... --tg-video-base ... \\
        --mcq-source ... \\
        --hier-val-source ...

    # Step 2: 为实验混合数据
    python3 local_scripts/data/mixer.py mix \\
        --data-root /path/to/data \\
        --tasks tg mcq hier_seg \\
        --hier-train ... --hier-target 5000 \\
        --exp-name R1_f1iou

    # 检查数据完整性
    python3 local_scripts/data/mixer.py check \\
        --data-root /path/to/data \\
        --tasks tg mcq hier_seg
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from types import ModuleType

from . import aot, event_logic, hier_seg, mcq, tg
from .common import print_summary, write_jsonl
from .frame_policy import (
    apply_frame_policy,
    default_frame_policy_cache_roots,
    parse_cache_roots,
    summarize_frame_policy_application,
)

# ---- 所有可用任务模块 ----
_ALL_MODULES: dict[str, ModuleType] = {
    tg.NAME: tg,
    mcq.NAME: mcq,
    hier_seg.NAME: hier_seg,
    event_logic.NAME: event_logic,
    aot.NAME: aot,
}

# ---- 需要 target 参数的任务 ----
_TARGET_ARGS: dict[str, str] = {
    "hier_seg": "hier_target",
    "event_logic": "el_target",
    "aot": "aot_target",
}


def _get_modules(task_names: list[str]) -> list[ModuleType]:
    modules = []
    for name in task_names:
        if name not in _ALL_MODULES:
            print(f"[mixer] ERROR: Unknown task '{name}'. Available: {list(_ALL_MODULES.keys())}")
            sys.exit(1)
        modules.append(_ALL_MODULES[name])
    return modules


def cmd_setup(args: argparse.Namespace) -> None:
    modules = _get_modules(args.tasks)
    for mod in modules:
        mod.setup_base(args.data_root, args, args.force, args.seed)

    # Summary
    print(f"\n{'='*50}")
    print("  Setup Complete!")
    print(f"{'='*50}")
    for sub in ["base", "val"]:
        d = Path(args.data_root) / sub
        if d.exists():
            print(f"  {sub}/:")
            for f in sorted(d.glob("*.jsonl")):
                count = sum(1 for _ in open(f))
                print(f"    {f.name}: {count}")


def cmd_mix(args: argparse.Namespace) -> None:
    exp_dir = Path(args.data_root) / "experiments" / args.exp_name
    train_out = str(exp_dir / "train.jsonl")
    val_out = str(exp_dir / "val.jsonl")
    cache_roots = default_frame_policy_cache_roots(args.data_root)
    cache_roots.extend(parse_cache_roots(getattr(args, "frame_sample_cache_roots", "")))

    modules = _get_modules(args.tasks)

    # ── Train ──
    if not args.force and Path(train_out).exists():
        print(f"Train already exists: {train_out} — skip (use --force to rebuild)")
    else:
        print("Loading + sampling train data...")
        all_train: list[dict] = []
        for mod in modules:
            records = mod.load_train(args.data_root, args)
            target_attr = _TARGET_ARGS.get(mod.NAME)
            target = getattr(args, target_attr, 0) if target_attr else 0
            if target > 0:
                print(f"  [{mod.NAME}]: {len(records)} (target: {target})")
                sampled = mod.sample_train(records, target, args.seed)
            else:
                print(f"  [{mod.NAME}]: {len(records)} (全量)")
                sampled = mod.sample_train(records, 0, args.seed)
            all_train.extend(sampled)

        random.Random(args.seed).shuffle(all_train)
        all_train = apply_frame_policy(
            all_train,
            policy=args.frame_sample_policy,
            max_frames=args.frame_sample_max_frames,
            cache_roots=cache_roots,
            progress_label=f"frame_policy train:{args.exp_name}",
            progress_interval=args.frame_sample_progress_interval,
        )
        if args.frame_sample_policy or args.frame_sample_max_frames > 0:
            frame_summary = summarize_frame_policy_application(all_train)
            source_counts = ",".join(
                f"{key}={value}" for key, value in sorted(frame_summary.get("sources", {}).items())
            )
            print(
                "  [frame_policy train]: "
                f"applied={frame_summary['applied']} skipped={frame_summary['skipped']}"
                f"{' sources=' + source_counts if source_counts else ''}"
            )
            print("  [frame_policy cache_roots]: " + "; ".join(cache_roots))
        write_jsonl(all_train, train_out)
        print_summary(all_train, f"Train -> {train_out}")

    # ── Val ──
    if not args.force and Path(val_out).exists():
        print(f"\nVal already exists: {val_out} — skip")
    else:
        print("\nLoading val data...")
        all_val: list[dict] = []
        for mod in modules:
            records = mod.load_val(args.data_root, args)
            print(f"  [{mod.NAME}]: {len(records)}")
            all_val.extend(records)

        random.Random(args.seed).shuffle(all_val)
        all_val = apply_frame_policy(
            all_val,
            policy=args.frame_sample_policy,
            max_frames=args.frame_sample_max_frames,
            cache_roots=cache_roots,
            progress_label=f"frame_policy val:{args.exp_name}",
            progress_interval=args.frame_sample_progress_interval,
        )
        if args.frame_sample_policy or args.frame_sample_max_frames > 0:
            frame_summary = summarize_frame_policy_application(all_val)
            source_counts = ",".join(
                f"{key}={value}" for key, value in sorted(frame_summary.get("sources", {}).items())
            )
            print(
                "  [frame_policy val]: "
                f"applied={frame_summary['applied']} skipped={frame_summary['skipped']}"
                f"{' sources=' + source_counts if source_counts else ''}"
            )
            print("  [frame_policy cache_roots]: " + "; ".join(cache_roots))
        write_jsonl(all_val, val_out)
        print_summary(all_val, f"Val -> {val_out}")


def cmd_check(args: argparse.Namespace) -> None:
    modules = _get_modules(args.tasks)
    missing = False

    for mod in modules:
        # 检查 val
        val_records = mod.load_val(args.data_root, args)
        if not val_records:
            print(f"[check] MISSING: {mod.NAME} val data")
            missing = True
        else:
            print(f"[check] OK: {mod.NAME} val ({len(val_records)} samples)")

        # 检查 train: tg/mcq 在 base/ 下，hier_seg/event_logic/aot 是外部路径
        target_attr = _TARGET_ARGS.get(mod.NAME)
        target = getattr(args, target_attr, 0) if target_attr else 0
        if not target_attr or target > 0:
            records = mod.load_train(args.data_root, args)
            if not records:
                print(f"[check] MISSING: {mod.NAME} train data")
                missing = True
            else:
                print(f"[check] OK: {mod.NAME} train ({len(records)} samples)")

    if missing:
        print("\n[check] FAIL: Missing data. Run 'setup' first.")
        sys.exit(1)
    else:
        print("\n[check] All base/val data present.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-task data management")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Force rebuild")
    parser.add_argument("--data-root", required=True, help="Data root directory")

    sub = parser.add_subparsers(dest="command", required=True)

    # 共享的 --tasks 参数 (注册到每个子命令，避免 nargs="+" 吞掉 subcommand)
    _tasks_kwargs = dict(
        nargs="+", default=list(_ALL_MODULES.keys()),
        help=f"Task modules to use (default: all). Available: {list(_ALL_MODULES.keys())}",
    )

    # ── setup ──
    p_setup = sub.add_parser("setup", help="Generate base data + val (one-time)")
    p_setup.add_argument("--tasks", **_tasks_kwargs)
    # ── mix ──
    p_mix = sub.add_parser("mix", help="Mix experiment training data")
    p_mix.add_argument("--tasks", **_tasks_kwargs)
    p_mix.add_argument("--exp-name", required=True, help="Experiment name (subdir)")
    p_mix.add_argument(
        "--frame-sample-policy",
        default="",
        help="Duration/fps rules for frame-list JSONL derivation, e.g. '0:60:2.0,60:inf:1.0'.",
    )
    p_mix.add_argument(
        "--frame-sample-max-frames",
        type=int,
        default=0,
        help="Uniform cap applied after fps downsampling. 0 disables the cap.",
    )
    p_mix.add_argument(
        "--frame-sample-cache-roots",
        default="",
        help=(
            "Extra trusted 2fps cache roots, separated by ':' or ','. Defaults are "
            "$data_root/offline_frames/base_cache_2fps and sibling "
            "hier_seg_annotation_v1/frame_cache/source_2fps."
        ),
    )
    p_mix.add_argument(
        "--frame-sample-progress-interval",
        type=int,
        default=1000,
        help="Print frame-policy progress every N records. Set <=0 to print every record.",
    )
    # ── check ──
    p_check = sub.add_parser("check", help="Verify base/val data exists")
    p_check.add_argument("--tasks", **_tasks_kwargs)

    # 注册各任务的 CLI 参数到每个子命令
    for p in [p_setup, p_mix, p_check]:
        for mod in _ALL_MODULES.values():
            mod.add_cli_args(p)

    args = parser.parse_args()

    if args.command == "setup":
        cmd_setup(args)
    elif args.command == "mix":
        cmd_mix(args)
    elif args.command == "check":
        cmd_check(args)


if __name__ == "__main__":
    main()
