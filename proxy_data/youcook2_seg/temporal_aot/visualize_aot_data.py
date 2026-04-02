#!/usr/bin/env python3
"""
visualize_aot_data.py — 可视化 AoT 训练数据的 domain / task 分布。

读取 build_aot_from_seg.py 产出的 train.jsonl，绘制:
  Fig 1: 每个 problem_type 的 record 数量 (bar)
  Fig 2: domain_l1 × domain_l2 嵌套饼图（所有 task 合并）
  Fig 3: 每个 problem_type 按 domain_l1 分布 (stacked bar)
  Fig 4: 按 level 分组的输入视频时长分布 (overlaid histogram)

用法:
    python visualize_aot_data.py --train-jsonl /path/to/train.jsonl
    python visualize_aot_data.py --train-dir /path/to/seg_aot_v2t  # 自动找 train.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# ── 样式 ──
mpl.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})

DOMAIN_L1_COLORS = {
    "procedural": "#4C72B0",
    "physical": "#55A868",
    "lifestyle": "#C44E52",
    "educational": "#8172B2",
    "other": "#CCCCCC",
}

LEVEL_COLORS = {
    "phase": "#4C72B0",
    "event": "#55A868",
    "action": "#C44E52",
}

TASK_COLORS = {
    "seg_aot_phase_v2t": "#4C72B0",
    "seg_aot_phase_t2v": "#7BA5D4",
    "seg_aot_event_v2t": "#55A868",
    "seg_aot_event_t2v": "#8DD49B",
    "seg_aot_action_v2t": "#C44E52",
    "seg_aot_action_t2v": "#E08185",
}


def load_jsonl(path: str | Path) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _get_meta(rec: dict, key: str, default: str = "other") -> str:
    """从 metadata 或顶层取字段。"""
    meta = rec.get("metadata", {})
    return meta.get(key, rec.get(key, default))


def _get_level(rec: dict) -> str:
    """从 problem_type 提取 level: phase/event/action。"""
    ptype = rec.get("problem_type", "")
    parts = ptype.replace("seg_aot_", "").split("_")
    return parts[0] if parts else "other"


def _get_duration(rec: dict) -> float:
    """从 metadata.total_duration_sec 获取时长。"""
    meta = rec.get("metadata", {})
    dur = meta.get("total_duration_sec")
    if dur is not None:
        return float(dur)
    return 0.0


# ── Fig 1: Task distribution bar chart ──
def plot_task_counts(records: list[dict], ax: plt.Axes):
    counts = Counter(r.get("problem_type", "unknown") for r in records)
    tasks = sorted(counts.keys())
    values = [counts[t] for t in tasks]
    colors = [TASK_COLORS.get(t, "#999") for t in tasks]

    # Shorten labels: seg_aot_phase_v2t → phase_v2t
    labels = [t.replace("seg_aot_", "") for t in tasks]

    bars = ax.barh(labels, values, color=colors, edgecolor="white")
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                str(v), va="center", fontsize=9)

    ax.set_xlabel("Record Count")
    ax.set_title(f"Task Distribution (total={len(records)})")
    ax.invert_yaxis()


# ── Fig 2: Domain donut (L1 inner, L2 outer) ──
def plot_domain_donut(records: list[dict], ax: plt.Axes):
    l1_counter: Counter = Counter()
    l2_counter: Counter = Counter()
    for r in records:
        d1 = _get_meta(r, "domain_l1")
        d2 = _get_meta(r, "domain_l2")
        l1_counter[d1] += 1
        l2_counter[(d1, d2)] += 1

    total = sum(l1_counter.values())
    if total == 0:
        ax.set_title("No data")
        ax.axis("off")
        return

    l1_sorted = sorted(l1_counter.keys(), key=lambda x: -l1_counter[x])

    inner_sizes, inner_colors, inner_labels = [], [], []
    outer_sizes, outer_colors, outer_labels = [], [], []

    for d1 in l1_sorted:
        inner_sizes.append(l1_counter[d1])
        inner_colors.append(DOMAIN_L1_COLORS.get(d1, "#999"))
        pct = l1_counter[d1] / total * 100
        inner_labels.append(f"{d1}\n({l1_counter[d1]}, {pct:.0f}%)")

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
           textprops={"fontsize": 8, "fontweight": "bold"},
           wedgeprops=wedge_kwargs)
    ax.pie(outer_sizes, radius=1.0, colors=outer_colors,
           labels=outer_labels, labeldistance=1.12,
           textprops={"fontsize": 7},
           wedgeprops={**wedge_kwargs, "width": 0.3})
    ax.set_title(f"Domain Distribution ({total} records)", fontsize=11, pad=15)


# ── Fig 3: Task × Domain stacked bar ──
def plot_task_domain_stacked(records: list[dict], ax: plt.Axes):
    tasks = sorted(set(r.get("problem_type", "unknown") for r in records))
    domains = sorted(set(_get_meta(r, "domain_l1") for r in records))

    # task → domain → count
    matrix = defaultdict(Counter)
    for r in records:
        t = r.get("problem_type", "unknown")
        d = _get_meta(r, "domain_l1")
        matrix[t][d] += 1

    labels = [t.replace("seg_aot_", "") for t in tasks]
    x = np.arange(len(tasks))
    bottom = np.zeros(len(tasks))

    for d in domains:
        values = [matrix[t][d] for t in tasks]
        color = DOMAIN_L1_COLORS.get(d, "#999")
        ax.bar(x, values, bottom=bottom, color=color, label=d, edgecolor="white")

        # Annotate each segment
        for i, v in enumerate(values):
            if v > 0:
                ax.text(x[i], bottom[i] + v / 2, str(v),
                        ha="center", va="center", fontsize=7, color="white", fontweight="bold")

        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Record Count")
    ax.set_title("Task × Domain_L1 Distribution")
    ax.legend(title="domain_l1", loc="upper right", fontsize=8)


# ── Fig 4: Duration distribution per level (overlaid histogram with stats) ──
def plot_duration_histogram(records: list[dict], ax: plt.Axes):
    by_level: dict[str, list[float]] = defaultdict(list)
    for r in records:
        dur = _get_duration(r)
        if dur <= 0:
            continue
        level = _get_level(r)
        by_level[level].append(dur)

    if not by_level:
        ax.set_title("Duration Distribution (no data)")
        ax.axis("off")
        return

    all_durs = [d for durs in by_level.values() for d in durs]
    bins = np.linspace(0, max(all_durs) * 1.05, 40)

    for level in ("phase", "event", "action"):
        durs = by_level.get(level, [])
        if durs:
            ax.hist(durs, bins=bins, alpha=0.55,
                    label=f"{level} (n={len(durs)})",
                    color=LEVEL_COLORS.get(level, "#999"),
                    edgecolor="white", linewidth=0.3)

    ax.set_xlabel("Input Duration (sec)")
    ax.set_ylabel("Count")
    ax.set_title("Input Duration Distribution per Level")
    ax.legend(fontsize=9)

    # 统计量文字
    text_lines = []
    for level in ("phase", "event", "action"):
        durs = by_level.get(level, [])
        if durs:
            text_lines.append(
                f"{level}: avg={np.mean(durs):.0f}s, med={np.median(durs):.0f}s, "
                f"[{min(durs):.0f}, {max(durs):.0f}]"
            )
    if text_lines:
        ax.text(0.97, 0.95, "\n".join(text_lines),
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))


def main():
    parser = argparse.ArgumentParser(description="可视化 AoT 训练数据分布")
    parser.add_argument("--train-jsonl", help="train.jsonl 文件路径")
    parser.add_argument("--train-dir", help="包含 train.jsonl 的目录")
    parser.add_argument("--output", "-o", help="输出图片路径 (default: aot_data_dist.png)")
    args = parser.parse_args()

    # Resolve train.jsonl path
    if args.train_jsonl:
        jsonl_path = args.train_jsonl
    elif args.train_dir:
        jsonl_path = os.path.join(args.train_dir, "train.jsonl")
    else:
        parser.error("Need --train-jsonl or --train-dir")
        return

    if not os.path.isfile(jsonl_path):
        print(f"ERROR: {jsonl_path} not found", file=sys.stderr)
        sys.exit(1)

    records = load_jsonl(jsonl_path)
    print(f"Loaded {len(records)} records from {jsonl_path}")

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(os.path.dirname(jsonl_path), "aot_data_dist.png")

    # ── Plot ──
    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    plot_task_counts(records, ax1)
    plot_domain_donut(records, ax2)
    plot_task_domain_stacked(records, ax3)
    plot_duration_histogram(records, ax4)

    fig.suptitle(f"AoT Training Data Distribution  ({len(records)} records)", fontsize=14, y=1.02)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
