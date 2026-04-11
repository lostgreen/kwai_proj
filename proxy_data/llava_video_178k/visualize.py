#!/usr/bin/env python3
"""
Visualize LLaVA-Video-178K MCQ data distribution: before vs after filtering.

Generates comparison plots showing:
1. Source × Duration grid (heatmap) — before & after
2. Source bar chart — before & after
3. Duration bucket bar chart — before & after
4. Rollout accuracy distribution (if report available)

Usage:
    # Compare full MCQ vs final output
    python visualize.py \
        --before results/mcq_all.jsonl \
        --after results/train_final.jsonl \
        --outdir results/figures

    # Also show rollout accuracy distribution
    python visualize.py \
        --before results/mcq_all.jsonl \
        --after results/train_final.jsonl \
        --report results/rollout_report.jsonl \
        --outdir results/figures
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def load_jsonl(path: str) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def get_cell(rec: dict) -> tuple[str, str]:
    """Extract (duration_bucket, source) from record metadata."""
    meta = rec.get("metadata", {})
    return meta.get("duration_bucket", "unknown"), meta.get("source", "unknown")


def count_grid(records: list[dict]) -> dict[tuple[str, str], int]:
    grid: dict[tuple[str, str], int] = defaultdict(int)
    for rec in records:
        grid[get_cell(rec)] += 1
    return grid


def plot_comparison_heatmap(
    before: dict[tuple[str, str], int],
    after: dict[tuple[str, str], int],
    outdir: str,
):
    """Side-by-side heatmaps: before (full MCQ) vs after (filtered+downsampled)."""
    all_buckets = sorted({b for b, _ in list(before.keys()) + list(after.keys())})
    all_sources = sorted({s for _, s in list(before.keys()) + list(after.keys())})

    def build_matrix(grid):
        return np.array([
            [grid.get((b, s), 0) for s in all_sources]
            for b in all_buckets
        ])

    mat_before = build_matrix(before)
    mat_after = build_matrix(after)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(4, len(all_buckets) * 1.2)))

    for ax, mat, title in [(ax1, mat_before, "Before (Full MCQ)"), (ax2, mat_after, "After (Filtered + Downsampled)")]:
        vmax = max(mat.max(), 1)
        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(all_sources)))
        ax.set_xticklabels(all_sources, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(all_buckets)))
        ax.set_yticklabels(all_buckets, fontsize=9)
        ax.set_title(f"{title} (N={int(mat.sum())})", fontsize=11)

        for i in range(len(all_buckets)):
            for j in range(len(all_sources)):
                val = int(mat[i, j])
                if val > 0:
                    color = "white" if val > vmax * 0.6 else "black"
                    ax.text(j, i, str(val), ha="center", va="center", fontsize=8, color=color)

        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Source × Duration Distribution", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(outdir, "grid_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  -> {path}")


def plot_comparison_bars(
    before: dict[tuple[str, str], int],
    after: dict[tuple[str, str], int],
    outdir: str,
):
    """Bar chart comparison by source and by duration bucket."""
    # Aggregate by source
    src_before: dict[str, int] = defaultdict(int)
    src_after: dict[str, int] = defaultdict(int)
    bkt_before: dict[str, int] = defaultdict(int)
    bkt_after: dict[str, int] = defaultdict(int)

    for (b, s), c in before.items():
        src_before[s] += c
        bkt_before[b] += c
    for (b, s), c in after.items():
        src_after[s] += c
        bkt_after[b] += c

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- By source ---
    all_sources = sorted(set(list(src_before.keys()) + list(src_after.keys())))
    x = np.arange(len(all_sources))
    w = 0.35
    vals_b = [src_before.get(s, 0) for s in all_sources]
    vals_a = [src_after.get(s, 0) for s in all_sources]

    bars1 = ax1.bar(x - w/2, vals_b, w, label="Before", color="#4a90d9", alpha=0.8)
    bars2 = ax1.bar(x + w/2, vals_a, w, label="After", color="#e8524a", alpha=0.8)

    for bar, val in zip(bars1, vals_b):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     str(val), ha="center", va="bottom", fontsize=7)
    for bar, val in zip(bars2, vals_a):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     str(val), ha="center", va="bottom", fontsize=7)

    ax1.set_xticks(x)
    ax1.set_xticklabels(all_sources, rotation=45, ha="right")
    ax1.set_ylabel("Count")
    ax1.set_title("By Source")
    ax1.legend()
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # --- By duration bucket ---
    all_buckets = sorted(set(list(bkt_before.keys()) + list(bkt_after.keys())))
    x2 = np.arange(len(all_buckets))
    vals_b2 = [bkt_before.get(b, 0) for b in all_buckets]
    vals_a2 = [bkt_after.get(b, 0) for b in all_buckets]

    bars3 = ax2.bar(x2 - w/2, vals_b2, w, label="Before", color="#4a90d9", alpha=0.8)
    bars4 = ax2.bar(x2 + w/2, vals_a2, w, label="After", color="#e8524a", alpha=0.8)

    for bar, val in zip(bars3, vals_b2):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     str(val), ha="center", va="bottom", fontsize=7)
    for bar, val in zip(bars4, vals_a2):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     str(val), ha="center", va="bottom", fontsize=7)

    ax2.set_xticks(x2)
    ax2.set_xticklabels(all_buckets, rotation=45, ha="right")
    ax2.set_ylabel("Count")
    ax2.set_title("By Duration Bucket")
    ax2.legend()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("Before vs After Comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(outdir, "bar_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  -> {path}")


def plot_accuracy_distribution(report_path: str, outdir: str):
    """Plot rollout accuracy distribution from report JSONL."""
    reports = load_jsonl(report_path)

    mean_rewards = [r.get("mean_reward", 0.0) for r in reports]
    if not mean_rewards:
        print("  No report data")
        return

    arr = np.array(mean_rewards)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Overall histogram
    bins = np.linspace(0, 1, 21)
    ax1.hist(arr, bins=bins, color="#4a90d9", edgecolor="white", alpha=0.85)
    ax1.axvline(0.25, color="red", linestyle="--", linewidth=1.5, label="min_acc=0.25")
    ax1.axvline(0.50, color="red", linestyle="--", linewidth=1.5, label="max_acc=0.50")
    ax1.axvspan(0.25, 0.50, alpha=0.15, color="green", label="Keep zone")
    ax1.set_xlabel("Mean Rollout Accuracy")
    ax1.set_ylabel("Count")
    ax1.set_title(f"Accuracy Distribution (N={len(arr)})")
    ax1.legend(fontsize=8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Per-source accuracy
    src_accs: dict[str, list[float]] = defaultdict(list)
    for r in reports:
        idx = r.get("index", -1)
        src_accs["all"].append(r.get("mean_reward", 0.0))

    # Print stats
    total = len(arr)
    in_range = int(np.sum((arr > 0.25) & (arr < 0.5)))
    too_easy = int(np.sum(arr >= 0.5))
    too_hard = int(np.sum(arr <= 0.25))

    # Pie chart of categories
    labels = [f"Too easy\n(>0.5): {too_easy}", f"Keep zone\n(0.25-0.5): {in_range}", f"Too hard\n(<=0.25): {too_hard}"]
    sizes = [too_easy, in_range, too_hard]
    colors = ["#ff9999", "#99ff99", "#9999ff"]
    ax2.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    ax2.set_title("Accuracy Categories")

    fig.tight_layout()
    path = os.path.join(outdir, "accuracy_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  -> {path}")

    # Text summary
    print(f"\n  Accuracy Summary (N={total}):")
    print(f"    mean={arr.mean():.4f}, median={np.median(arr):.4f}")
    print(f"    Too easy (>=0.5): {too_easy} ({too_easy/total*100:.1f}%)")
    print(f"    Keep zone (0.25-0.5): {in_range} ({in_range/total*100:.1f}%)")
    print(f"    Too hard (<=0.25): {too_hard} ({too_hard/total*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="LLaVA MCQ distribution comparison: before vs after filtering"
    )
    parser.add_argument("--before", required=True, help="Full MCQ JSONL (before filtering)")
    parser.add_argument("--after", required=True, help="Final output JSONL (after filtering)")
    parser.add_argument("--report", default="",
                        help="Rollout report JSONL (optional, for accuracy plots)")
    parser.add_argument("--outdir", default="figures")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading before: {args.before}")
    before_recs = load_jsonl(args.before)
    print(f"  {len(before_recs)} records")

    print(f"Loading after: {args.after}")
    after_recs = load_jsonl(args.after)
    print(f"  {len(after_recs)} records")

    before_grid = count_grid(before_recs)
    after_grid = count_grid(after_recs)

    print("\n== Source × Duration Heatmap ==")
    plot_comparison_heatmap(before_grid, after_grid, args.outdir)

    print("\n== Bar Chart Comparison ==")
    plot_comparison_bars(before_grid, after_grid, args.outdir)

    if args.report and os.path.isfile(args.report):
        print("\n== Accuracy Distribution ==")
        plot_accuracy_distribution(args.report, args.outdir)

    print(f"\nDone! Figures saved to {args.outdir}/")


if __name__ == "__main__":
    main()
