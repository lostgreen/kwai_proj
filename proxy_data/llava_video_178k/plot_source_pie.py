#!/usr/bin/env python3
"""
LLaVA-Video-178K: 绘制 source 分布饼图（原始 vs 下采样后）。

生成两个子图:
  左: 参考数据集来源 source 分布（mcq_all.jsonl）
  右: 下采样过滤后 train 的 source 分布（train_final_combined.jsonl）

Usage:
    python plot_source_pie.py \
        --before results/mcq_all.jsonl \
        --after  results/train_final_combined.jsonl \
        --outdir results/figures

    # 仅画 after
    python plot_source_pie.py \
        --after results/train_final_combined.jsonl \
        --outdir results/figures
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_jsonl(path: str) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def extract_source(rec: dict) -> str:
    return rec.get("metadata", {}).get("source", "unknown")


def _make_pie(ax, counter: Counter, title: str, top_n: int = 12):
    """Draw a pie chart on the given axes. Merge small slices into 'others'."""
    total = sum(counter.values())
    # Sort by count descending
    ordered = counter.most_common()

    if len(ordered) > top_n:
        main = ordered[:top_n]
        others_count = sum(c for _, c in ordered[top_n:])
        main.append(("others", others_count))
    else:
        main = ordered

    labels = [f"{name}\n({cnt})" for name, cnt in main]
    sizes = [cnt for _, cnt in main]

    cmap = plt.cm.Set3
    colors = [cmap(i / max(len(main), 1)) for i in range(len(main))]

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.78,
        colors=colors,
    )
    for t in texts:
        t.set_fontsize(8)
    for t in autotexts:
        t.set_fontsize(7)
        t.set_color("dimgray")
    ax.set_title(f"{title}\n(N={total})", fontsize=12, fontweight="bold")


def main():
    parser = argparse.ArgumentParser(description="Draw source distribution pie charts")
    parser.add_argument("--before", default="", help="Original MCQ JSONL (before filtering)")
    parser.add_argument("--after", required=True, help="Final train JSONL (after downsampling)")
    parser.add_argument("--outdir", default="results/figures")
    parser.add_argument("--top-n", type=int, default=12,
                        help="Max number of source slices before merging into 'others'")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    has_before = args.before and os.path.isfile(args.before)
    ncols = 2 if has_before else 1
    fig, axes = plt.subplots(1, ncols, figsize=(8 * ncols, 7))
    if ncols == 1:
        axes = [axes]

    idx = 0
    if has_before:
        print(f"Loading before: {args.before}")
        before_recs = load_jsonl(args.before)
        print(f"  {len(before_recs)} records")
        before_counter = Counter(extract_source(r) for r in before_recs)
        _make_pie(axes[idx], before_counter, "Reference Dataset Source", top_n=args.top_n)
        idx += 1

    print(f"Loading after: {args.after}")
    after_recs = load_jsonl(args.after)
    print(f"  {len(after_recs)} records")
    after_counter = Counter(extract_source(r) for r in after_recs)
    _make_pie(axes[idx], after_counter, "Train (Downsampled) Source", top_n=args.top_n)

    fig.suptitle("LLaVA-Video-178K Source Distribution", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(args.outdir, "source_pie_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
