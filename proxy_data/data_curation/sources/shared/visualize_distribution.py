"""
Vision Filter 数据分布可视化

对 vision_results_keep.jsonl (或任意 stage 的 JSONL) 做 source / duration 分布统计和图表。

用法:
    # 单文件
    python visualize_distribution.py --input results/vision_results_keep.jsonl

    # 多文件（合并统计）
    python visualize_distribution.py \
        --input ../et_instruct_164k/results/vision_results_keep.jsonl \
               ../timelens_100k/results/vision_results_keep.jsonl

    # 指定输出目录
    python visualize_distribution.py --input results/vision_results_keep.jsonl --outdir figures/

    # 按 _origin.dataset 分组而非 source 字段
    python visualize_distribution.py --input merged.jsonl --group-by origin
"""

import json
import argparse
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ── Data Loading ─────────────────────────────────────────

def load_jsonl(paths: list[str]) -> list[dict]:
    records = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def get_group_key(record: dict, group_by: str) -> str:
    if group_by == "origin":
        return record.get("_origin", {}).get("dataset", "unknown")
    return record.get("source", "unknown")


# ── Source Distribution ──────────────────────────────────

def plot_source_distribution(records: list[dict], group_by: str, outdir: str):
    counts: dict[str, int] = defaultdict(int)
    for r in records:
        counts[get_group_key(r, group_by)] += 1

    # Sort by count descending
    sorted_items = sorted(counts.items(), key=lambda x: -x[1])
    labels = [x[0] for x in sorted_items]
    values = [x[1] for x in sorted_items]
    total = sum(values)

    # --- Bar chart ---
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.6), 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
    bars = ax.bar(range(len(labels)), values, color=colors, edgecolor="white", linewidth=0.5)

    # Add count + percentage labels
    for bar, val in zip(bars, values):
        pct = val / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.005,
            f"{val}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=8,
        )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Sample Count")
    ax.set_title(f"Source Distribution (N={total})")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = os.path.join(outdir, "source_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  -> {path}")

    # --- Pie chart (top-15 + Others) ---
    max_slices = 15
    if len(sorted_items) > max_slices:
        top = sorted_items[:max_slices]
        others = sum(v for _, v in sorted_items[max_slices:])
        pie_labels = [x[0] for x in top] + ["others"]
        pie_values = [x[1] for x in top] + [others]
    else:
        pie_labels, pie_values = labels, values

    fig2, ax2 = plt.subplots(figsize=(9, 9))
    wedges, texts, autotexts = ax2.pie(
        pie_values, labels=pie_labels, autopct="%1.1f%%",
        pctdistance=0.8, startangle=90,
        colors=plt.cm.Set3(np.linspace(0, 1, len(pie_labels))),
    )
    for t in autotexts:
        t.set_fontsize(8)
    for t in texts:
        t.set_fontsize(8)
    ax2.set_title(f"Source Distribution (N={total})")
    fig2.tight_layout()
    path2 = os.path.join(outdir, "source_pie.png")
    fig2.savefig(path2, dpi=150)
    plt.close(fig2)
    print(f"  -> {path2}")

    # Print text summary
    print(f"\n  Source 分布 (共 {total} 条, {len(counts)} 个 domain):")
    for label, val in sorted_items:
        print(f"    {label:30s}: {val:5d} ({val/total*100:5.1f}%)")


# ── Duration Distribution ────────────────────────────────

def plot_duration_distribution(records: list[dict], group_by: str, outdir: str):
    durations = [r["duration"] for r in records if isinstance(r.get("duration"), (int, float))]
    if not durations:
        print("  无 duration 数据")
        return

    durations_arr = np.array(durations)

    # --- Overall histogram ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: histogram
    ax = axes[0]
    bins = np.arange(0, min(durations_arr.max() + 10, 300), 10)
    ax.hist(durations_arr, bins=bins, color="#4a90d9", edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.axvline(np.median(durations_arr), color="red", linestyle="--", linewidth=1.2, label=f"median={np.median(durations_arr):.0f}s")
    ax.axvline(np.mean(durations_arr), color="orange", linestyle="--", linewidth=1.2, label=f"mean={np.mean(durations_arr):.0f}s")
    ax.legend(fontsize=9)
    ax.set_xlabel("Duration (seconds)")
    ax.set_ylabel("Count")
    ax.set_title(f"Duration Distribution (N={len(durations)})")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Right: box plot by duration bins
    ax2 = axes[1]
    bin_edges = [0, 60, 90, 120, 150, 180, 240, float("inf")]
    bin_labels = ["<60", "60-90", "90-120", "120-150", "150-180", "180-240", ">240"]
    bin_counts = []
    for i in range(len(bin_edges) - 1):
        count = int(np.sum((durations_arr >= bin_edges[i]) & (durations_arr < bin_edges[i + 1])))
        bin_counts.append(count)
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(bin_labels)))
    bars = ax2.bar(range(len(bin_labels)), bin_counts, color=colors, edgecolor="white")
    for bar, val in zip(bars, bin_counts):
        if val > 0:
            pct = val / len(durations) * 100
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + len(durations) * 0.005,
                     f"{val}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(range(len(bin_labels)))
    ax2.set_xticklabels(bin_labels, fontsize=9)
    ax2.set_xlabel("Duration Range (seconds)")
    ax2.set_ylabel("Count")
    ax2.set_title("Duration Bins")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    path = os.path.join(outdir, "duration_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  -> {path}")

    # --- Per-source duration box plot ---
    source_durations: dict[str, list] = defaultdict(list)
    for r in records:
        if isinstance(r.get("duration"), (int, float)):
            source_durations[get_group_key(r, group_by)].append(r["duration"])

    # Sort by median duration
    sorted_sources = sorted(source_durations.keys(), key=lambda s: np.median(source_durations[s]))

    if len(sorted_sources) > 1:
        fig3, ax3 = plt.subplots(figsize=(max(10, len(sorted_sources) * 0.5), 6))
        data = [source_durations[s] for s in sorted_sources]
        bp = ax3.boxplot(data, vert=True, patch_artist=True, showfliers=False)
        colors3 = plt.cm.Set2(np.linspace(0, 1, len(sorted_sources)))
        for patch, color in zip(bp["boxes"], colors3):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax3.set_xticks(range(1, len(sorted_sources) + 1))
        ax3.set_xticklabels(sorted_sources, rotation=45, ha="right", fontsize=8)
        ax3.set_ylabel("Duration (seconds)")
        ax3.set_title("Duration by Source (box = IQR, no outliers)")
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)
        # Add sample count
        for i, s in enumerate(sorted_sources):
            n = len(source_durations[s])
            med = np.median(source_durations[s])
            ax3.text(i + 1, med, f"n={n}", ha="center", va="bottom", fontsize=7, color="gray")
        fig3.tight_layout()
        path3 = os.path.join(outdir, "duration_by_source.png")
        fig3.savefig(path3, dpi=150)
        plt.close(fig3)
        print(f"  -> {path3}")

    # Print summary
    print(f"\n  Duration 统计 (N={len(durations)}):")
    print(f"    min={durations_arr.min():.1f}s, max={durations_arr.max():.1f}s")
    print(f"    mean={durations_arr.mean():.1f}s, median={np.median(durations_arr):.1f}s")
    print(f"    std={durations_arr.std():.1f}s")
    print(f"\n  Duration 分段:")
    for label, count in zip(bin_labels, bin_counts):
        print(f"    {label:10s}: {count:5d} ({count/len(durations)*100:5.1f}%)")


# ── Combined: Source × Duration heatmap ──────────────────

def plot_source_duration_heatmap(records: list[dict], group_by: str, outdir: str):
    """Source x Duration-bin heatmap."""
    bin_edges = [0, 60, 90, 120, 150, 180, 240, float("inf")]
    bin_labels = ["<60", "60-90", "90-120", "120-150", "150-180", "180-240", ">240"]

    source_bin: dict[str, list[int]] = defaultdict(lambda: [0] * len(bin_labels))
    for r in records:
        dur = r.get("duration")
        if not isinstance(dur, (int, float)):
            continue
        src = get_group_key(r, group_by)
        for i in range(len(bin_edges) - 1):
            if bin_edges[i] <= dur < bin_edges[i + 1]:
                source_bin[src][i] += 1
                break

    if not source_bin:
        return

    # Sort sources by total count descending
    sorted_sources = sorted(source_bin.keys(), key=lambda s: -sum(source_bin[s]))

    matrix = np.array([source_bin[s] for s in sorted_sources])
    fig, ax = plt.subplots(figsize=(10, max(4, len(sorted_sources) * 0.4)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels, fontsize=9)
    ax.set_yticks(range(len(sorted_sources)))
    ax.set_yticklabels(sorted_sources, fontsize=8)
    ax.set_xlabel("Duration Range (seconds)")
    ax.set_title("Source × Duration Heatmap")

    # Annotate cells
    for i in range(len(sorted_sources)):
        for j in range(len(bin_labels)):
            val = matrix[i, j]
            if val > 0:
                color = "white" if val > matrix.max() * 0.6 else "black"
                ax.text(j, i, str(val), ha="center", va="center", fontsize=7, color=color)

    fig.colorbar(im, ax=ax, shrink=0.8, label="Count")
    fig.tight_layout()
    path = os.path.join(outdir, "source_duration_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  -> {path}")


# ── Main ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Vision filter 数据分布可视化 (source + duration)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", nargs="+", required=True, help="JSONL 文件路径（可多个）")
    parser.add_argument("--outdir", default="figures", help="图表输出目录")
    parser.add_argument("--group-by", choices=["source", "origin"], default="source",
                        help="分组字段: source (顶层 source 字段) 或 origin (_origin.dataset)")
    parser.add_argument("--filter-decision", default=None, choices=["keep", "reject"],
                        help="只分析指定 vision decision 的样本(默认全部)")
    args = parser.parse_args()

    records = load_jsonl(args.input)
    print(f"加载 {len(records)} 条记录 (来自 {len(args.input)} 个文件)")

    # Optional decision filter
    if args.filter_decision:
        records = [
            r for r in records
            if r.get("_vision", {}).get("decision") == args.filter_decision
        ]
        print(f"  筛选 _vision.decision={args.filter_decision}: {len(records)} 条")

    if not records:
        print("无数据，退出")
        return

    os.makedirs(args.outdir, exist_ok=True)
    print(f"输出目录: {args.outdir}\n")

    print("== Source 分布 ==")
    plot_source_distribution(records, args.group_by, args.outdir)

    print("\n== Duration 分布 ==")
    plot_duration_distribution(records, args.group_by, args.outdir)

    print("\n== Source × Duration 热力图 ==")
    plot_source_duration_heatmap(records, args.group_by, args.outdir)

    print(f"\n可视化完成，图表输出到 {args.outdir}/")


if __name__ == "__main__":
    main()
