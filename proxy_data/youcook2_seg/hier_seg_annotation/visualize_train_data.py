#!/usr/bin/env python3
"""
visualize_train_data.py — 从构建好的 train JSONL 可视化训练数据分布。

读取 build_v1_data.sh 产出的各层 train.jsonl，统计并绘制:
  Fig 1: 每层级 record 数量 (bar)
  Fig 2: 每层级输入时长分布 (overlaid histogram)
  Fig 3: 每层级 domain_l1 × domain_l2 嵌套饼图 (3 个 subplot)
  Fig 4: 每层级按 domain_l2 细分 (stacked bar)

用法:
    # 默认读取 train_dir 下 L1/L2/L3_seg 子目录的 train.jsonl
    python visualize_train_data.py \
        --train-dir /path/to/train

    # 或指定具体 JSONL 文件
    python visualize_train_data.py \
        --jsonl L1=/path/to/L1/train.jsonl \
               L2=/path/to/L2/train.jsonl \
               L3=/path/to/L3_seg/train.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# ── 样式配置 (与 visualize_annotations.py 对齐) ──
mpl.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})

LEVEL_COLORS = {
    "L1": "#4C72B0",
    "L2": "#55A868",
    "L3": "#C44E52",
}

DOMAIN_L1_COLORS = {
    "procedural": "#4C72B0",
    "physical": "#55A868",
    "lifestyle": "#C44E52",
    "educational": "#8172B2",
    "other": "#CCCCCC",
}

# 层级目录名 → 显示名
LEVEL_DIR_MAP = {"L1": "L1", "L2": "L2", "L3_seg": "L3"}


# =====================================================================
# 数据加载
# =====================================================================
def load_jsonl(path: str | Path) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def get_input_duration(rec: dict) -> float:
    """从 metadata 推导输入时长 (秒)。"""
    meta = rec.get("metadata", {})
    level = meta.get("level")

    # L1: 整个视频
    if level == 1:
        return float(meta.get("clip_duration_sec", 0))
    # L2 phase mode
    if level == 2 and "phase_start_sec" in meta:
        return float(meta["phase_end_sec"]) - float(meta["phase_start_sec"])
    # L2 window mode
    if level == 2 and "window_start_sec" in meta:
        return float(meta["window_end_sec"]) - float(meta["window_start_sec"])
    # L3 / L3_seg
    if "clip_start_sec" in meta:
        return float(meta["clip_end_sec"]) - float(meta["clip_start_sec"])
    if "event_start_sec" in meta:
        return float(meta["event_end_sec"]) - float(meta["event_start_sec"])

    return 0.0


# =====================================================================
# Fig 1: 每层级 record 数量
# =====================================================================
def plot_record_counts(per_level: dict[str, list[dict]], ax: plt.Axes):
    levels = [lv for lv in ("L1", "L2", "L3") if lv in per_level]
    counts = [len(per_level[lv]) for lv in levels]
    colors = [LEVEL_COLORS[lv] for lv in levels]

    bars = ax.bar(levels, counts, color=colors, edgecolor="white", linewidth=0.8, width=0.5)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, cnt + max(counts) * 0.02,
                str(cnt), ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Training Records")
    ax.set_title(f"Training Records per Level (total={sum(counts)})")
    ax.set_ylim(0, max(counts) * 1.2)


# =====================================================================
# Fig 2: 每层级输入时长分布
# =====================================================================
def plot_input_duration(per_level: dict[str, list[dict]], ax: plt.Axes):
    levels = [lv for lv in ("L1", "L2", "L3") if lv in per_level]
    for lv in levels:
        durations = [get_input_duration(r) for r in per_level[lv]]
        durations = [d for d in durations if d > 0]
        if durations:
            ax.hist(durations, bins=40, alpha=0.55, label=f"{lv} (n={len(durations)})",
                    color=LEVEL_COLORS[lv], edgecolor="white", linewidth=0.3)

    ax.set_xlabel("Input Duration (sec)")
    ax.set_ylabel("Count")
    ax.set_title("Input Duration Distribution per Level")
    ax.legend()

    # 统计量文字
    text_lines = []
    for lv in levels:
        durs = [get_input_duration(r) for r in per_level[lv]]
        durs = [d for d in durs if d > 0]
        if durs:
            text_lines.append(f"{lv}: avg={np.mean(durs):.0f}s, med={np.median(durs):.0f}s, "
                              f"[{min(durs):.0f}, {max(durs):.0f}]")
    ax.text(0.97, 0.95, "\n".join(text_lines),
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))


# =====================================================================
# Fig 3: 每层级 domain 嵌套饼图
# =====================================================================
def _draw_domain_donut(records: list[dict], level: str, ax: plt.Axes):
    if not records:
        ax.set_title(f"{level} (no data)")
        ax.axis("off")
        return

    l1_counter: Counter = Counter()
    l2_counter: Counter = Counter()
    for r in records:
        meta = r.get("metadata", {})
        d1 = meta.get("domain_l1", "other")
        d2 = meta.get("domain_l2", "other")
        l1_counter[d1] += 1
        l2_counter[(d1, d2)] += 1

    total = sum(l1_counter.values())
    l1_sorted = sorted(l1_counter.keys(), key=lambda x: -l1_counter[x])

    inner_sizes, inner_colors, inner_labels = [], [], []
    outer_sizes, outer_colors, outer_labels = [], [], []

    for d1 in l1_sorted:
        inner_sizes.append(l1_counter[d1])
        inner_colors.append(DOMAIN_L1_COLORS.get(d1, "#999"))
        pct = l1_counter[d1] / total * 100
        inner_labels.append(f"{d1}\n({pct:.0f}%)")

        subs = sorted(
            [(k, v) for k, v in l2_counter.items() if k[0] == d1],
            key=lambda x: -x[1],
        )
        base_color = mpl.colors.to_rgba(DOMAIN_L1_COLORS.get(d1, "#999"))
        for i, ((_, d2), cnt) in enumerate(subs):
            outer_sizes.append(cnt)
            alpha = 0.5 + 0.5 * (1 - i / max(len(subs), 1))
            c = (*base_color[:3], alpha)
            outer_colors.append(c)
            pct2 = cnt / total * 100
            outer_labels.append(f"{d2}\n({cnt})" if pct2 >= 3 else "")

    wedge_kwargs = dict(edgecolor="white", linewidth=1.5)
    ax.pie(inner_sizes, radius=0.65, colors=inner_colors,
           labels=inner_labels, labeldistance=0.35,
           textprops={"fontsize": 7, "fontweight": "bold"},
           wedgeprops=wedge_kwargs)
    ax.pie(outer_sizes, radius=1.0, colors=outer_colors,
           labels=outer_labels, labeldistance=1.12,
           textprops={"fontsize": 6},
           wedgeprops={**wedge_kwargs, "width": 0.3})
    ax.set_title(f"{level}  ({total} records)", fontsize=11, pad=15)


def plot_domain_per_level(per_level: dict[str, list[dict]], axes: list[plt.Axes]):
    levels = [lv for lv in ("L1", "L2", "L3") if lv in per_level]
    for ax, lv in zip(axes, levels):
        _draw_domain_donut(per_level[lv], lv, ax)
    # 剩余 axes 隐藏
    for ax in axes[len(levels):]:
        ax.axis("off")


# =====================================================================
# Fig 4: 按 domain_l2 细分 (stacked bar)
# =====================================================================
def plot_domain_l2_stacked(per_level: dict[str, list[dict]], ax: plt.Axes):
    levels = [lv for lv in ("L1", "L2", "L3") if lv in per_level]

    d2_yield: dict[str, dict[str, int]] = defaultdict(lambda: {lv: 0 for lv in levels})
    for lv in levels:
        for r in per_level[lv]:
            d2 = r.get("metadata", {}).get("domain_l2", "other")
            d2_yield[d2][lv] += 1

    d2_sorted = sorted(d2_yield.keys(), key=lambda d: -sum(d2_yield[d].values()))
    if len(d2_sorted) > 15:
        d2_sorted = d2_sorted[:15]

    x = np.arange(len(d2_sorted))
    bottom = np.zeros(len(d2_sorted))

    for lv in levels:
        vals = np.array([d2_yield[d][lv] for d in d2_sorted], dtype=float)
        ax.bar(x, vals, bottom=bottom, label=lv, color=LEVEL_COLORS[lv],
               edgecolor="white", linewidth=0.5)
        bottom += vals

    for i, d in enumerate(d2_sorted):
        total = sum(d2_yield[d].values())
        ax.text(i, bottom[i] + max(bottom) * 0.01, str(total),
                ha="center", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(d2_sorted, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Training Records")
    ax.set_title("Training Data by Domain (L2) — Stacked by Level")
    ax.legend(loc="upper right")


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="从构建好的 train JSONL 可视化训练数据分布",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--train-dir", default="",
                        help="训练数据根目录 (含 L1/L2/L3_seg 子目录)")
    parser.add_argument("--jsonl", nargs="*", default=[],
                        help="手动指定: LEVEL=path (如 L1=/path/to/train.jsonl)")
    parser.add_argument("--output-dir", default="./figures",
                        help="图片保存目录")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    # 加载数据
    per_level: dict[str, list[dict]] = {}

    if args.jsonl:
        for spec in args.jsonl:
            lv, path = spec.split("=", 1)
            lv_display = LEVEL_DIR_MAP.get(lv, lv)
            per_level[lv_display] = load_jsonl(path)
            print(f"  {lv_display}: loaded {len(per_level[lv_display])} records from {path}")
    elif args.train_dir:
        train_dir = Path(args.train_dir)
        for dir_name, display_name in LEVEL_DIR_MAP.items():
            jsonl_path = train_dir / dir_name / "train.jsonl"
            if jsonl_path.exists():
                per_level[display_name] = load_jsonl(jsonl_path)
                print(f"  {display_name}: loaded {len(per_level[display_name])} records from {jsonl_path}")
            else:
                print(f"  {display_name}: SKIP ({jsonl_path} not found)")
    else:
        parser.error("请指定 --train-dir 或 --jsonl")

    if not per_level:
        print("ERROR: 未找到任何训练数据")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Fig 1: Record counts ──
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    plot_record_counts(per_level, ax1)
    fig1.tight_layout()
    fig1.savefig(os.path.join(args.output_dir, "train_fig1_record_counts.png"), dpi=args.dpi)
    print(f"  Saved train_fig1_record_counts.png")

    # ── Fig 2: Input duration ──
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    plot_input_duration(per_level, ax2)
    fig2.tight_layout()
    fig2.savefig(os.path.join(args.output_dir, "train_fig2_input_duration.png"), dpi=args.dpi)
    print(f"  Saved train_fig2_input_duration.png")

    # ── Fig 3: Domain donut per level ──
    n_levels = len([lv for lv in ("L1", "L2", "L3") if lv in per_level])
    fig3, axes3 = plt.subplots(1, max(n_levels, 1), figsize=(6 * max(n_levels, 1), 6))
    if n_levels == 1:
        axes3 = [axes3]
    plot_domain_per_level(per_level, list(axes3))
    fig3.suptitle("Domain Distribution per Level (Train)", fontsize=13, y=1.02)
    fig3.tight_layout()
    fig3.savefig(os.path.join(args.output_dir, "train_fig3_domain_per_level.png"),
                 dpi=args.dpi, bbox_inches="tight")
    print(f"  Saved train_fig3_domain_per_level.png")

    # ── Fig 4: Domain L2 stacked bar ──
    fig4, ax4 = plt.subplots(figsize=(12, 5))
    plot_domain_l2_stacked(per_level, ax4)
    fig4.tight_layout()
    fig4.savefig(os.path.join(args.output_dir, "train_fig4_domain_l2_stacked.png"), dpi=args.dpi)
    print(f"  Saved train_fig4_domain_l2_stacked.png")

    plt.close("all")

    # ── 打印汇总 ──
    print(f"\n{'='*60}")
    print(f"  TRAIN DATA DISTRIBUTION SUMMARY")
    print(f"{'='*60}")
    for lv in ("L1", "L2", "L3"):
        if lv not in per_level:
            continue
        recs = per_level[lv]
        durs = [get_input_duration(r) for r in recs]
        durs = [d for d in durs if d > 0]
        d2_counter = Counter(r.get("metadata", {}).get("domain_l2", "other") for r in recs)
        top3 = d2_counter.most_common(3)
        top3_str = ", ".join(f"{d}({c})" for d, c in top3)
        print(f"  {lv}: {len(recs)} records"
              f"  | duration: avg={np.mean(durs):.0f}s med={np.median(durs):.0f}s"
              f"  | top domains: {top3_str}")
    print(f"{'='*60}")
    print(f"  Figures saved to: {os.path.abspath(args.output_dir)}/")


if __name__ == "__main__":
    main()
