#!/usr/bin/env python3
"""
从 screen_keep.jsonl 中两层均衡采样: 先按 domain_l1 均衡，再在每个 L1 内按 domain_l2 均衡。
优先选择因果逻辑 order_dependency == "strict" 的样本。

策略:
  第一层: 将 total 均匀分配到各 domain_l1
          不足的 domain_l1 将余额重新分配给有余量的 L1
  第二层: 在每个 domain_l1 的 quota 内，再按 domain_l2 子类均衡分配
          每个 L2 内优先选 strict > loose > none, 同级按 l1_score*l2_score 降序
  最终输出 total 条 JSONL，保留原始字段

用法:
    python balanced_sample.py \
        --input screen_keep.jsonl \
        --output balanced_1200.jsonl \
        --total 1200 \
        --seed 42
"""

import argparse
import json
import random
from collections import defaultdict


def load_jsonl(path: str) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def get_domain_l1(item: dict) -> str:
    screen = item.get("_screen") or {}
    return screen.get("domain_l1") or "other"


def get_domain_l2(item: dict) -> str:
    screen = item.get("_screen") or {}
    return screen.get("domain_l2") or "other"


def get_order_dependency(item: dict) -> str:
    screen2 = item.get("_screen_2") or {}
    return screen2.get("order_dependency", "unknown")


def priority_sort_key(item: dict) -> tuple:
    """strict 在前, 然后 loose, 最后 none/unknown。同级按 score 降序。"""
    order = get_order_dependency(item)
    order_rank = {"strict": 0, "loose": 1, "none": 2}.get(order, 3)

    screen = item.get("_screen") or {}
    l1 = screen.get("l1_score") or 0
    l2 = screen.get("l2_score") or 0
    score = -(l1 * l2)

    return (order_rank, score)


def allocate_quota(
    pools: dict[str, list],
    total: int,
    rng: random.Random,
) -> dict[str, list[dict]]:
    """从多个 pool 中按均匀分配 + 溢出重分配选取样本。

    每个 pool 内部要求已经按 priority_sort_key 排好序。
    """
    keys = sorted(pools.keys())
    n_keys = len(keys)
    if n_keys == 0:
        return {}

    # 均匀分配
    quota = {k: total // n_keys for k in keys}
    remainder = total - sum(quota.values())
    sorted_by_size = sorted(keys, key=lambda k: len(pools[k]), reverse=True)
    for i in range(remainder):
        quota[sorted_by_size[i % n_keys]] += 1

    selected: dict[str, list[dict]] = {}
    overflow = 0

    # 第一轮
    for k in keys:
        pool = pools[k]
        n = min(quota[k], len(pool))
        selected[k] = pool[:n]
        if n < quota[k]:
            overflow += quota[k] - n

    # 溢出重分配
    if overflow > 0:
        for k in sorted_by_size:
            if overflow <= 0:
                break
            already = len(selected[k])
            available = len(pools[k]) - already
            if available > 0:
                extra = min(overflow, available)
                selected[k].extend(pools[k][already:already + extra])
                overflow -= extra

    return selected


def balanced_sample(
    items: list[dict],
    total: int,
    seed: int = 42,
) -> list[dict]:
    rng = random.Random(seed)

    # ── 第一层: 按 domain_l1 分组 ──
    by_l1: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        by_l1[get_domain_l1(item)].append(item)

    l1_domains = sorted(by_l1.keys())

    print(f"\n{'='*70}")
    print(f"共 {len(items)} 条候选, {len(l1_domains)} 个 L1 domain, 目标 {total} 条")
    print(f"{'='*70}")

    # 统计 L1 -> L2 分布
    print(f"\n候选分布 (L1 → L2):")
    for d1 in l1_domains:
        pool = by_l1[d1]
        strict_n = sum(1 for it in pool if get_order_dependency(it) == "strict")
        l2_counts: dict[str, int] = defaultdict(int)
        for it in pool:
            l2_counts[get_domain_l2(it)] += 1
        l2_str = ", ".join(f"{k}:{v}" for k, v in sorted(l2_counts.items(), key=lambda x: -x[1]))
        print(f"  {d1:<18} total={len(pool):>4}  strict={strict_n:>4}  L2=[{l2_str}]")

    # ── L1 quota 分配 ──
    n_l1 = len(l1_domains)
    l1_quota = {d: total // n_l1 for d in l1_domains}
    remainder = total - sum(l1_quota.values())
    sorted_l1_by_size = sorted(l1_domains, key=lambda d: len(by_l1[d]), reverse=True)
    for i in range(remainder):
        l1_quota[sorted_l1_by_size[i % n_l1]] += 1

    # 如果某 L1 候选不足，将余额重分配
    overflow = 0
    actual_l1_quota: dict[str, int] = {}
    for d in l1_domains:
        n = min(l1_quota[d], len(by_l1[d]))
        actual_l1_quota[d] = n
        if n < l1_quota[d]:
            overflow += l1_quota[d] - n

    if overflow > 0:
        for d in sorted_l1_by_size:
            if overflow <= 0:
                break
            available = len(by_l1[d]) - actual_l1_quota[d]
            if available > 0:
                extra = min(overflow, available)
                actual_l1_quota[d] += extra
                overflow -= extra

    # ── 第二层: 在每个 L1 内按 L2 均衡 ──
    result = []
    print(f"\n采样结果:")
    print(f"  {'L1 domain':<18} {'L1 quota':>8} {'actual':>7}  L2 detail")
    print("  " + "-" * 70)

    for d1 in l1_domains:
        l1_target = actual_l1_quota[d1]
        pool = by_l1[d1]

        # 按 L2 分组
        by_l2: dict[str, list[dict]] = defaultdict(list)
        for it in pool:
            by_l2[get_domain_l2(it)].append(it)

        # 在每个 L2 内排序
        for d2 in by_l2:
            rng.shuffle(by_l2[d2])
            by_l2[d2].sort(key=priority_sort_key)

        # L2 均衡分配
        selected_l2 = allocate_quota(dict(by_l2), l1_target, rng)

        l1_selected = []
        l2_detail_parts = []
        for d2 in sorted(selected_l2.keys()):
            sel = selected_l2[d2]
            strict_n = sum(1 for it in sel if get_order_dependency(it) == "strict")
            l2_detail_parts.append(f"{d2}:{len(sel)}(s{strict_n})")
            l1_selected.extend(sel)

        l2_detail = ", ".join(l2_detail_parts)
        print(f"  {d1:<18} {l1_quota[d1]:>8} {len(l1_selected):>7}  [{l2_detail}]")

        result.extend(l1_selected)

    # 最终 shuffle
    rng.shuffle(result)

    print(f"\n总计采样: {len(result)} 条")
    total_strict = sum(1 for it in result if get_order_dependency(it) == "strict")
    total_loose = sum(1 for it in result if get_order_dependency(it) == "loose")
    total_other = len(result) - total_strict - total_loose
    print(f"  strict: {total_strict} ({100*total_strict/len(result):.1f}%)")
    print(f"  loose:  {total_loose} ({100*total_loose/len(result):.1f}%)")
    print(f"  other:  {total_other} ({100*total_other/len(result):.1f}%)")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="两层均衡采样 (L1 → L2)，优先 strict 因果逻辑")
    parser.add_argument("--input", required=True,
                        help="screen_keep.jsonl 路径")
    parser.add_argument("--output", required=True,
                        help="输出 JSONL 路径")
    parser.add_argument("--total", type=int, default=1200,
                        help="总采样数 (默认 1200)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--stats", default="",
                        help="可选: 输出采样统计 JSON")
    args = parser.parse_args()

    items = load_jsonl(args.input)
    print(f"加载 {len(items)} 条 from {args.input}")

    if len(items) < args.total:
        print(f"⚠️  候选 ({len(items)}) < 目标 ({args.total})，将输出所有候选")

    result = balanced_sample(items, args.total, seed=args.seed)

    with open(args.output, "w", encoding="utf-8") as f:
        for item in result:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\n✅ 已写入 {args.output} ({len(result)} records)")

    # 可选统计输出
    if args.stats:
        stats = {
            "total_input": len(items),
            "total_sampled": len(result),
            "seed": args.seed,
            "per_l1": {},
            "per_l2": {},
        }
        for item in result:
            d1 = get_domain_l1(item)
            d2 = get_domain_l2(item)
            order = get_order_dependency(item)
            order_key = order if order in ("strict", "loose", "none") else "other"

            for level, key in [("per_l1", d1), ("per_l2", d2)]:
                if key not in stats[level]:
                    stats[level][key] = {"count": 0, "strict": 0, "loose": 0, "none": 0}
                stats[level][key]["count"] += 1
                stats[level][key][order_key] = stats[level][key].get(order_key, 0) + 1

        with open(args.stats, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"✅ 统计已保存到 {args.stats}")


if __name__ == "__main__":
    main()
