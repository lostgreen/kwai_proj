"""Shared utilities for multi-task data management."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path: str) -> list[dict]:
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  -> {path}: {len(records)} samples")


def random_sample(records: list[dict], n: int, seed: int = 42) -> list[dict]:
    if len(records) <= n:
        return list(records)
    return random.Random(seed).sample(records, n)


def stratified_sample(
    records: list[dict], target: int, key: str = "problem_type", seed: int = 42
) -> list[dict]:
    """按 key 字段等比例采样到 target 条。"""
    by_group: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_group[r.get(key, "unknown")].append(r)

    total = len(records)
    if total <= target:
        print(f"    {total} <= target {target}, keeping all")
        return list(records)

    rng = random.Random(seed)
    sampled: list[dict] = []
    remaining = target

    groups = sorted(by_group.keys())
    for i, group in enumerate(groups):
        pool = by_group[group]
        if i < len(groups) - 1:
            n = round(target * len(pool) / total)
            n = min(n, len(pool))
        else:
            n = min(remaining, len(pool))
        rng.shuffle(pool)
        sampled.extend(pool[:n])
        remaining -= n
        print(f"    [{group}]: {len(pool)} -> {n}")

    return sampled


def nested_stratified_sample(
    records: list[dict],
    target: int,
    key: str,
    nested_key: str,
    seed: int = 42,
) -> list[dict]:
    """两级分层采样: 先按 key 分配配额，再按 nested_key (metadata 字段) 均匀采样。"""
    by_group: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_group[r.get(key, "unknown")].append(r)

    total = len(records)
    if total <= target:
        print(f"    {total} <= target {target}, keeping all")
        return list(records)

    rng = random.Random(seed)
    sampled: list[dict] = []
    remaining = target

    groups = sorted(by_group.keys())
    for i, group in enumerate(groups):
        pool = by_group[group]
        if i < len(groups) - 1:
            quota = round(target * len(pool) / total)
            quota = min(quota, len(pool))
        else:
            quota = min(remaining, len(pool))

        # 二级分层
        by_sub: dict[str, list[dict]] = defaultdict(list)
        for r in pool:
            sub_val = (r.get("metadata") or {}).get(nested_key, "unknown")
            by_sub[str(sub_val)].append(r)

        sub_keys = sorted(by_sub.keys())
        sub_remaining = quota
        group_sampled: list[dict] = []
        for j, sk in enumerate(sub_keys):
            sub_pool = by_sub[sk]
            if j < len(sub_keys) - 1:
                n = round(quota * len(sub_pool) / len(pool))
                n = min(n, len(sub_pool))
            else:
                n = min(sub_remaining, len(sub_pool))
            rng.shuffle(sub_pool)
            group_sampled.extend(sub_pool[:n])
            sub_remaining -= n

        sampled.extend(group_sampled)
        remaining -= len(group_sampled)
        print(f"    [{group}]: {len(pool)} -> {len(group_sampled)} ({len(by_sub)} sub-groups)")

    return sampled


def print_summary(records: list[dict], label: str) -> None:
    counter = Counter(r.get("problem_type", "unknown") for r in records)
    print(f"\n{'='*50}")
    print(f"  {label}: {len(records)} total")
    print(f"{'='*50}")
    for pt in sorted(counter):
        print(f"  {pt:>30}: {counter[pt]:5d}")
