#!/usr/bin/env python3
"""
从 screen_keep.jsonl 中两层均衡采样: 先按 domain_l1 均衡，再在每个 L1 内按 domain_l2 均衡。
优先选择因果逻辑 order_dependency == "strict" 的样本。

策略:
  0. 前置筛选: min L1/L2 score, order_dependency, visual_diversity, prog_type
  1. 第一层: 将 total 均匀分配到各 domain_l1
          不足的 domain_l1 将余额重新分配给有余量的 L1
  2. 第二层: 在每个 domain_l1 的 quota 内，再按 domain_l2 子类均衡分配
          每个 L2 内优先选 strict > loose > none, 同级按 l1_score*l2_score 降序
  3. 输出 JSONL + 领域分布饼图

用法:
    # 基础
    python balanced_sample.py \
        --input screen_keep.jsonl \
        --output balanced_1200.jsonl \
        --total 1200 --seed 42

    # 严格筛选 + 饼图
    python balanced_sample.py \
        --input screen_keep.jsonl \
        --output balanced_1200.jsonl \
        --total 1200 --seed 42 \
        --min-l1-score 4 --min-l2-score 4 \
        --order-dep strict \
        --pie-chart distribution.png
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


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


def apply_filters(
    items: list[dict],
    min_l1_score: int = 0,
    min_l2_score: int = 0,
    order_deps: set[str] | None = None,
    visual_divs: set[str] | None = None,
    prog_types: set[str] | None = None,
) -> list[dict]:
    """前置筛选，返回满足所有条件的样本。"""
    filtered = []
    for item in items:
        screen = item.get("_screen") or {}
        screen2 = item.get("_screen_2") or {}

        l1 = screen.get("l1_score") or 0
        l2 = screen.get("l2_score") or 0
        order = screen2.get("order_dependency", "unknown")
        vis = screen2.get("visual_diversity", "unknown")
        prog = screen2.get("prog_type", "unknown")

        if l1 < min_l1_score:
            continue
        if l2 < min_l2_score:
            continue
        if order_deps and order not in order_deps:
            continue
        if visual_divs and vis not in visual_divs:
            continue
        if prog_types and prog not in prog_types:
            continue

        filtered.append(item)
    return filtered


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


def generate_pie_charts(result: list[dict], output_path: str):
    """生成 L1 + L2 领域分布饼图 (双子图)。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️  matplotlib 未安装，跳过饼图生成 (pip install matplotlib)")
        return

    # 统计
    l1_counts: dict[str, int] = defaultdict(int)
    l2_counts: dict[str, int] = defaultdict(int)
    order_counts: dict[str, int] = defaultdict(int)
    for item in result:
        l1_counts[get_domain_l1(item)] += 1
        l2_counts[get_domain_l2(item)] += 1
        order_counts[get_order_dependency(item)] += 1

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # L1 饼图
    l1_labels = sorted(l1_counts.keys())
    l1_sizes = [l1_counts[k] for k in l1_labels]
    l1_labels_pct = [f"{k}\n({v}, {100*v/len(result):.1f}%)" for k, v in zip(l1_labels, l1_sizes)]
    axes[0].pie(l1_sizes, labels=l1_labels_pct, autopct="", startangle=90)
    axes[0].set_title(f"L1 Domain Distribution (n={len(result)})", fontsize=14)

    # L2 饼图
    l2_sorted = sorted(l2_counts.items(), key=lambda x: -x[1])
    l2_labels = [k for k, _ in l2_sorted]
    l2_sizes = [v for _, v in l2_sorted]
    # 太小的合并为 "others"
    threshold = len(result) * 0.02
    main_labels, main_sizes = [], []
    other_size = 0
    for label, size in zip(l2_labels, l2_sizes):
        if size >= threshold:
            main_labels.append(f"{label}\n({size})")
            main_sizes.append(size)
        else:
            other_size += size
    if other_size > 0:
        main_labels.append(f"others\n({other_size})")
        main_sizes.append(other_size)
    axes[1].pie(main_sizes, labels=main_labels, autopct="", startangle=90)
    axes[1].set_title(f"L2 Sub-Domain Distribution ({len(l2_counts)} types)", fontsize=14)

    # Order dependency 饼图
    order_labels = sorted(order_counts.keys())
    order_sizes = [order_counts[k] for k in order_labels]
    order_labels_pct = [f"{k}\n({v}, {100*v/len(result):.1f}%)" for k, v in zip(order_labels, order_sizes)]
    colors = {"strict": "#2ecc71", "loose": "#f39c12", "none": "#e74c3c", "unknown": "#95a5a6"}
    order_colors = [colors.get(k, "#bdc3c7") for k in order_labels]
    axes[2].pie(order_sizes, labels=order_labels_pct, colors=order_colors, autopct="", startangle=90)
    axes[2].set_title("Causal Order Dependency", fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ 饼图已保存到 {output_path}")


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

    # 筛选参数
    parser.add_argument("--min-l1-score", type=int, default=0,
                        help="最低 L1 phase score (1-5, 默认 0 不筛)")
    parser.add_argument("--min-l2-score", type=int, default=0,
                        help="最低 L2 event score (1-5, 默认 0 不筛)")
    parser.add_argument("--order-dep", type=str, default="",
                        help="允许的 order_dependency, 逗号分隔 (strict,loose,none)")
    parser.add_argument("--visual-div", type=str, default="",
                        help="允许的 visual_diversity, 逗号分隔 (high,medium,low)")
    parser.add_argument("--prog-type", type=str, default="",
                        help="允许的 prog_type, 逗号分隔 (procedural,narrative)")

    # 可视化
    parser.add_argument("--pie-chart", default="",
                        help="输出领域分布饼图路径 (.png)")

    args = parser.parse_args()

    items = load_jsonl(args.input)
    print(f"加载 {len(items)} 条 from {args.input}")

    # 前置筛选
    order_deps = set(args.order_dep.split(",")) if args.order_dep else None
    visual_divs = set(args.visual_div.split(",")) if args.visual_div else None
    prog_types = set(args.prog_type.split(",")) if args.prog_type else None

    if any([args.min_l1_score, args.min_l2_score, order_deps, visual_divs, prog_types]):
        before = len(items)
        items = apply_filters(
            items,
            min_l1_score=args.min_l1_score,
            min_l2_score=args.min_l2_score,
            order_deps=order_deps,
            visual_divs=visual_divs,
            prog_types=prog_types,
        )
        print(f"筛选后: {len(items)} 条 (过滤掉 {before - len(items)} 条)")
        if order_deps:
            print(f"  order_dependency ∈ {order_deps}")
        if visual_divs:
            print(f"  visual_diversity ∈ {visual_divs}")
        if prog_types:
            print(f"  prog_type ∈ {prog_types}")
        if args.min_l1_score:
            print(f"  l1_score >= {args.min_l1_score}")
        if args.min_l2_score:
            print(f"  l2_score >= {args.min_l2_score}")

    if len(items) == 0:
        print("⚠️  筛选后无候选，请放宽条件")
        return

    if len(items) < args.total:
        print(f"⚠️  候选 ({len(items)}) < 目标 ({args.total})，将输出所有候选")

    result = balanced_sample(items, args.total, seed=args.seed)

    with open(args.output, "w", encoding="utf-8") as f:
        for item in result:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\n✅ 已写入 {args.output} ({len(result)} records)")

    # 饼图
    if args.pie_chart:
        Path(args.pie_chart).parent.mkdir(parents=True, exist_ok=True)
        generate_pie_charts(result, args.pie_chart)

    # 可选统计输出
    if args.stats:
        stats = {
            "total_input": len(items),
            "total_sampled": len(result),
            "seed": args.seed,
            "filters": {
                "min_l1_score": args.min_l1_score,
                "min_l2_score": args.min_l2_score,
                "order_dep": args.order_dep or "all",
                "visual_div": args.visual_div or "all",
                "prog_type": args.prog_type or "all",
            },
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
