#!/usr/bin/env python3
"""
build_event_shuffle.py — 从 hier seg annotation 构建 event shuffling (sort) 训练数据。

用 L1 phase 的 child L2 events 的原子 clips 构建排序任务：
给模型打乱顺序的 event clips，模型输出正确时间顺序排列。

可选通过 --filter-order 仅保留 _order_distinguishable=true 的 phase。

Reward: Jigsaw Displacement R = 1 - E_jigsaw / E_max（复用 hier_seg_reward 的 sort）。

用法:
    python proxy_data/youcook2_seg/event_logic/build_event_shuffle.py \\
        --annotation-dir /m2v_intern/.../annotations_fixed_gmn25 \\
        --clip-dir /m2v_intern/.../clips \\
        --output-dir /m2v_intern/.../event_shuffle \\
        --complete-only --seed 42

    # 仅保留 _order_distinguishable=true 的 phase:
    python ... --filter-order
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

# 添加 proxy_data 父目录到 sys.path 以便 import shared
_PROXY_DATA_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROXY_DATA_DIR not in sys.path:
    sys.path.insert(0, _PROXY_DATA_DIR)

# 添加脚本所在目录到 sys.path 以便 import prompts（与 build_l2_event_logic.py 对齐）
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from shared.seg_source import load_annotations, get_l2_event_atomic_path  # noqa: E402
from prompts import get_sort_prompt_generic  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# =====================================================================
# Phase 1: Collect phase-event groups
# =====================================================================

def collect_phase_groups(
    annotations: list[dict],
    clip_dir: str,
    min_events: int = 3,
    max_events: int = 8,
    complete_only: bool = False,
    filter_order: bool = False,
    order_confidence_threshold: float = 0.0,
) -> list[dict]:
    """Iterate annotations, collect child events per phase.

    Args:
        filter_order: If True, only keep phases with _order_distinguishable=True.
                      If False (default), keep all phases regardless.

    Returns list of group dicts with keys:
        clip_key, phase_id, events, domain_l1, domain_l2,
        order_confidence, order_cue
    """
    groups: list[dict] = []
    stats = Counter()

    for ann in annotations:
        l1 = ann.get("level1")
        l2 = ann.get("level2")
        if not l1 or not l2 or l2.get("_parse_error"):
            stats["skip_no_l1_l2"] += 1
            continue

        clip_key = ann.get("clip_key", "")
        events_all = l2.get("events", [])
        phases = l1.get("macro_phases", [])

        for phase in phases:
            if not isinstance(phase, dict):
                continue

            ph_id = phase.get("phase_id")
            stats["phases_total"] += 1

            # Filter: _order_distinguishable (only when --filter-order is set)
            if filter_order and not phase.get("_order_distinguishable", False):
                stats["phases_not_distinguishable"] += 1
                continue

            # Optional: filter by confidence (only meaningful with --filter-order)
            confidence = phase.get("_order_confidence", 0.0)
            if filter_order and confidence < order_confidence_threshold:
                stats["phases_low_confidence"] += 1
                continue

            # Gather child events sorted by start_time
            child_events = sorted(
                [ev for ev in events_all
                 if isinstance(ev, dict)
                 and ev.get("parent_phase_id") == ph_id
                 and ev.get("instruction", "").strip()
                 and isinstance(ev.get("start_time"), (int, float))],
                key=lambda ev: ev["start_time"],
            )
            if len(child_events) < min_events:
                stats["phases_too_few_events"] += 1
                continue
            if len(child_events) > max_events:
                stats["phases_too_many_events"] += 1
                continue

            # Resolve atomic clip paths
            event_records = []
            all_exist = True
            for ev in child_events:
                path = get_l2_event_atomic_path(
                    clip_key, ev["event_id"],
                    int(ev["start_time"]), int(ev["end_time"]),
                    clip_dir,
                )
                if complete_only and not os.path.exists(path):
                    all_exist = False
                    break
                event_records.append({
                    "event_id": ev["event_id"],
                    "start": int(ev["start_time"]),
                    "end": int(ev["end_time"]),
                    "instruction": ev["instruction"].strip(),
                    "clip_path": path,
                })

            if not all_exist:
                stats["phases_missing_clips"] += 1
                continue

            stats["phases_kept"] += 1
            groups.append({
                "clip_key": clip_key,
                "phase_id": ph_id,
                "events": event_records,
                "domain_l1": ann.get("domain_l1", "other"),
                "domain_l2": ann.get("domain_l2", "other"),
                "order_confidence": confidence,
                "order_cue": phase.get("_order_cue", ""),
            })

    log.info("Phase group collection stats: %s", dict(stats))
    return groups


# =====================================================================
# Phase 2: Build sort records
# =====================================================================

def build_sort_record(
    group: dict,
    seq_len: int,
    rng: random.Random,
) -> dict | None:
    """Build one sort record from a phase-event group.

    If the group has more events than seq_len, a random contiguous
    subsequence of seq_len events is selected.

    Returns EasyR1 record dict or None if shuffle fails.
    """
    events = group["events"]
    n = len(events)

    # Select contiguous subsequence if needed
    if n > seq_len:
        start = rng.randint(0, n - seq_len)
        selected = events[start:start + seq_len]
    else:
        selected = events
        seq_len = n

    # Shuffle (retry to avoid identity permutation)
    indices = list(range(seq_len))
    shuf_idx = indices[:]
    for _ in range(20):
        rng.shuffle(shuf_idx)
        if shuf_idx != indices:
            break
    if shuf_idx == indices:
        return None  # very unlikely, but safety check

    # Inverse permutation: answer[i] = 1-based clip number at position i
    inverse = [0] * seq_len
    for clip_idx, orig_idx in enumerate(shuf_idx):
        inverse[orig_idx] = clip_idx + 1
    answer = "".join(str(x) for x in inverse)

    shuffled_events = [selected[i] for i in shuf_idx]
    prompt = get_sort_prompt_generic(seq_len)

    return {
        "messages": [{"role": "user", "content": prompt}],
        "prompt": prompt,
        "answer": answer,
        "videos": [ev["clip_path"] for ev in shuffled_events],
        "data_type": "video",
        "problem_type": "sort",
        "metadata": {
            "clip_key": group["clip_key"],
            "phase_id": group["phase_id"],
            "original_event_ids": [ev["event_id"] for ev in selected],
            "shuffled_indices": shuf_idx,
            "event_instructions": [ev["instruction"] for ev in shuffled_events],
            "order_confidence": group["order_confidence"],
            "order_cue": group["order_cue"],
            "domain_l1": group["domain_l1"],
            "domain_l2": group["domain_l2"],
            "source": "event_shuffle",
        },
    }


# =====================================================================
# Balanced sampling (from build_aot_from_seg.py pattern)
# =====================================================================

def _balanced_sample_by_domain(
    records: list[dict],
    budget: int,
    rng: random.Random,
) -> list[dict]:
    """Two-tier balanced sampling by domain_l1 -> domain_l2."""
    if len(records) <= budget:
        rng.shuffle(records)
        return records

    by_l1: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        d = rec.get("metadata", {}).get("domain_l1", "other")
        by_l1[d].append(rec)

    n_l1 = len(by_l1)
    base_per_l1 = budget // max(n_l1, 1)

    sampled: list[dict] = []
    shortfall = 0
    overflows: list[list[dict]] = []

    for d in sorted(by_l1.keys()):
        pool = by_l1[d]
        rng.shuffle(pool)
        if len(pool) <= base_per_l1:
            sampled.extend(pool)
            shortfall += base_per_l1 - len(pool)
        else:
            # L2 balanced within this L1 domain
            l2_pool = _balanced_sample_l2(pool, base_per_l1, rng)
            sampled.extend(l2_pool)
            selected_ids = set(id(r) for r in l2_pool)
            leftover = [r for r in pool if id(r) not in selected_ids]
            if leftover:
                overflows.append(leftover)

    # Redistribute shortfall to overflow domains
    extra = shortfall
    if extra > 0 and overflows:
        per_overflow = extra // len(overflows)
        for leftovers in overflows:
            take = min(per_overflow, len(leftovers))
            sampled.extend(leftovers[:take])

    rng.shuffle(sampled)
    return sampled[:budget]


def _balanced_sample_l2(
    records: list[dict],
    budget: int,
    rng: random.Random,
) -> list[dict]:
    """Within one L1 domain, balanced sample by domain_l2."""
    by_l2: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        d = r.get("metadata", {}).get("domain_l2", "other")
        by_l2[d].append(r)
    n = len(by_l2)
    per = budget // max(n, 1)
    out: list[dict] = []
    for d in sorted(by_l2.keys()):
        pool = by_l2[d]
        rng.shuffle(pool)
        out.extend(pool[:per])
    rng.shuffle(out)
    return out[:budget]


# =====================================================================
# IO
# =====================================================================

def write_jsonl(records: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="从 hier seg annotation 构建 event shuffling (sort) 数据"
    )
    parser.add_argument("--annotation-dir", "-a", required=True,
                        help="标注 JSON 目录")
    parser.add_argument("--clip-dir", required=True,
                        help="原子 clips 根目录（含 L2/ 子目录）")
    parser.add_argument("--output-dir", "-o", required=True,
                        help="输出目录（生成 train.jsonl, val.jsonl, stats.json）")

    # Group filtering
    parser.add_argument("--min-events", type=int, default=3,
                        help="phase 至少包含的 event 数（默认 3）")
    parser.add_argument("--max-events", type=int, default=8,
                        help="phase 最多包含的 event 数（默认 8）")
    parser.add_argument("--order-confidence-threshold", type=float, default=0.0,
                        help="_order_confidence 最低阈值（默认 0.0，不过滤）")
    parser.add_argument("--filter-order", action="store_true",
                        help="仅保留 _order_distinguishable=true 的 phase（默认不筛选）")
    parser.add_argument("--complete-only", action="store_true",
                        help="仅保留所有 atomic clips 都存在的 group")

    # Sort task params
    parser.add_argument("--seq-len", type=int, default=5,
                        help="每条样本包含的 event clip 数（默认 5）")
    parser.add_argument("--samples-per-group", type=int, default=1,
                        help="每个 group 生成的样本数（默认 1）")

    # Train/val split
    parser.add_argument("--train-budget", type=int, default=-1,
                        help="训练集最大样本数（-1 = 不限）")
    parser.add_argument("--val-count", type=int, default=100,
                        help="验证集样本数（默认 100）")

    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # ---- 1. Load annotations ----
    log.info("Loading annotations from: %s", args.annotation_dir)
    annotations = load_annotations(args.annotation_dir, complete_only=False)
    log.info("Loaded %d annotations", len(annotations))

    # ---- 2. Collect phase groups ----
    groups = collect_phase_groups(
        annotations,
        clip_dir=args.clip_dir,
        min_events=args.min_events,
        max_events=args.max_events,
        complete_only=args.complete_only,
        filter_order=args.filter_order,
        order_confidence_threshold=args.order_confidence_threshold,
    )
    log.info("Collected %d phase groups (%s, %d-%d events)",
             len(groups),
             "filter_order=true" if args.filter_order else "all phases",
             args.min_events, args.max_events)

    if not groups:
        log.error("No qualifying phase groups found. Check --annotation-dir and --clip-dir.")
        sys.exit(1)

    # ---- 3. Build sort records ----
    records: list[dict] = []
    for group in groups:
        for _ in range(args.samples_per_group):
            rec = build_sort_record(group, args.seq_len, rng)
            if rec is not None:
                records.append(rec)
    log.info("Built %d sort records from %d groups", len(records), len(groups))

    # ---- 4. Train/val split ----
    rng.shuffle(records)
    n_val = min(args.val_count, max(1, len(records) // 5))
    val_records = records[:n_val]
    train_pool = records[n_val:]

    if args.train_budget > 0 and len(train_pool) > args.train_budget:
        train_records = _balanced_sample_by_domain(train_pool, args.train_budget, rng)
    else:
        train_records = train_pool

    rng.shuffle(train_records)
    rng.shuffle(val_records)

    # ---- 5. Write output ----
    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.jsonl")
    val_path = os.path.join(args.output_dir, "val.jsonl")
    write_jsonl(train_records, train_path)
    write_jsonl(val_records, val_path)

    # Stats
    stats = {
        "total_annotations": len(annotations),
        "total_groups": len(groups),
        "total_records": len(records),
        "train_count": len(train_records),
        "val_count": len(val_records),
        "train_budget": args.train_budget,
        "seq_len": args.seq_len,
        "min_events": args.min_events,
        "max_events": args.max_events,
        "order_confidence_threshold": args.order_confidence_threshold,
        "complete_only": args.complete_only,
        "events_per_group": dict(Counter(len(g["events"]) for g in groups)),
        "train_by_domain_l1": dict(Counter(
            r.get("metadata", {}).get("domain_l1", "other") for r in train_records
        )),
        "val_by_domain_l1": dict(Counter(
            r.get("metadata", {}).get("domain_l1", "other") for r in val_records
        )),
    }
    stats_path = os.path.join(args.output_dir, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # Summary
    log.info("=== Output ===")
    log.info("  Train: %d  ->  %s", len(train_records), train_path)
    log.info("  Val:   %d  ->  %s", len(val_records), val_path)
    log.info("  Stats: %s", stats_path)
    log.info("  Events per group distribution: %s", stats["events_per_group"])
    log.info("  Train domain_l1: %s", stats["train_by_domain_l1"])

    if train_records:
        ex = train_records[0]
        log.info("=== Example record ===")
        log.info("  answer: %s", ex["answer"])
        log.info("  videos (%d): %s", len(ex["videos"]), ex["videos"][0])
        log.info("  prompt (first 200 chars):\n  %s", ex["prompt"][:200])


if __name__ == "__main__":
    main()
