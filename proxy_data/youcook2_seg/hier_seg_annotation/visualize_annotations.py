#!/usr/bin/env python3
"""
visualize_annotations.py — 标注数据可视化分析

训练数据层级结构:
  L1: 整个视频输入 → 分割出 macro phases  (每视频 1 条)
  L2: 每个 L1 phase 输入 → 分割出 events  (每 phase 1 条)
  L3: 每个 leaf node (event/empty-phase) 输入 → 分割出 micro-actions  (每 leaf 1 条)

生成图表:
  Fig 1: 每层级训练数据产出量 (bar + 倍率)
  Fig 2: domain_l1 × domain_l2 两层嵌套饼图
  Fig 3: 各层级输入时长分布 (overlaid histogram)
  Fig 4: 各层级产出按 domain_l1 分组 (grouped bar)
  Fig 5: 每层级按 domain_l2 细分产出 (stacked bar)
  Fig 6: Topology × Domain 热力图
  Fig 7: L3 完整率按 domain_l2 分组

用法:
    python visualize_annotations.py \
        --annotation-dir /path/to/annotations \
        [--output-dir ./figures]
        [--complete-only]
        [--min-actions 3]
        [--dpi 150]
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

# ── 项目导入 ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
PROXY_DATA_DIR = os.path.join(REPO_ROOT, "proxy_data")
for _p in (SCRIPT_DIR, PROXY_DATA_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── 样式配置 ──────────────────────────────────────────────────────────────
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

TOPO_COLORS = {
    "procedural": "#4C72B0",
    "periodic": "#55A868",
    "sequence": "#DD8452",
    "flat": "#CCCCCC",
}


# =====================================================================
# 数据加载
# =====================================================================
def load_annotations(ann_dir: Path, complete_only: bool = False) -> list[dict]:
    anns = []
    for p in sorted(ann_dir.glob("*.json")):
        try:
            with open(p, encoding="utf-8") as f:
                ann = json.load(f)
        except Exception:
            continue
        if complete_only and not ann.get("level3"):
            continue
        anns.append(ann)
    return anns


# =====================================================================
# 训练数据产出统计 (逐级父节点输入模式)
# =====================================================================
def compute_training_records(
    anns: list[dict],
    min_actions: int = 3,
) -> dict:
    """统计各层级训练记录数和输入时长。

    层级定义:
      L1: 整个视频 → phases  (1 个视频 = 1 条 L1 record)
      L2: 每个 phase → events  (1 个 phase = 1 条 L2 record)
      L3: 每个 leaf node → micro-actions  (1 个 leaf = 1 条 L3 record)
           leaf = event (有子事件的) or empty-phase (无子事件的phase)
    """
    records = {
        "per_video": [],           # list[dict] per-video info
        "per_level": Counter(),     # L1/L2/L3 → total records
        # 输入时长 (秒)
        "input_durations": {"L1": [], "L2": [], "L3": []},
        # 输出个数 (每条记录要产出多少个 segments)
        "output_counts": {"L1": [], "L2": [], "L3": []},
    }

    for ann in anns:
        clip_key = ann.get("clip_key", "?")
        domain_l1 = ann.get("domain_l1", "other")
        domain_l2 = ann.get("domain_l2", "other")
        topo = ann.get("topology_type", "unknown")
        clip_duration = float(ann.get("clip_duration_sec") or 0)

        l1 = ann.get("level1") or {}
        l2 = ann.get("level2") or {}
        l3 = ann.get("level3") or {}
        phases = l1.get("macro_phases") or []
        events = l2.get("events") or []
        all_actions = l3.get("grounding_results") or []

        video_yield = {
            "clip_key": clip_key,
            "domain_l1": domain_l1,
            "domain_l2": domain_l2,
            "topology": topo,
            "L1": 0, "L2": 0, "L3": 0,
        }

        # ── L1: 1 条 per video ──
        valid_phases = [
            p for p in phases
            if isinstance(p.get("start_time"), (int, float))
            and isinstance(p.get("end_time"), (int, float))
            and p["end_time"] > p["start_time"]
        ]
        if valid_phases and clip_duration > 0:
            video_yield["L1"] = 1
            records["input_durations"]["L1"].append(clip_duration)
            records["output_counts"]["L1"].append(len(valid_phases))

        # ── 构建 phase → events 映射 ──
        phase_event_map: dict[int, list[dict]] = defaultdict(list)
        for ev in events:
            if not isinstance(ev, dict):
                continue
            pid = ev.get("parent_phase_id")
            if pid is not None:
                phase_event_map[pid].append(ev)

        # ── L2: 1 条 per phase (输入 = phase 时长) ──
        for phase in valid_phases:
            pid = phase.get("phase_id")
            ph_start = float(phase.get("start_time", 0))
            ph_end = float(phase.get("end_time", 0))
            ph_dur = ph_end - ph_start
            children = phase_event_map.get(pid, [])

            if ph_dur > 0:
                video_yield["L2"] += 1
                records["input_durations"]["L2"].append(ph_dur)
                records["output_counts"]["L2"].append(len(children))

        # ── L3: 1 条 per leaf node ──
        # leaf = event (if phase has children) or phase (if no children)
        has_l3 = all_actions and not l3.get("_parse_error")

        for phase in valid_phases:
            pid = phase.get("phase_id")
            children = phase_event_map.get(pid, [])

            if children:
                # leaf nodes = events
                for ev in children:
                    ev_start = float(ev.get("start_time", 0))
                    ev_end = float(ev.get("end_time", 0))
                    ev_dur = ev_end - ev_start
                    eid = ev.get("event_id")

                    if ev_dur <= 0:
                        continue

                    if has_l3:
                        n_acts = sum(
                            1 for a in all_actions
                            if isinstance(a, dict) and a.get("parent_event_id") == eid
                        )
                        if n_acts >= min_actions:
                            video_yield["L3"] += 1
                            records["input_durations"]["L3"].append(ev_dur)
                            records["output_counts"]["L3"].append(n_acts)
            else:
                # leaf node = the phase itself (empty-event phase)
                ph_start = float(phase.get("start_time", 0))
                ph_end = float(phase.get("end_time", 0))
                ph_dur = ph_end - ph_start

                if ph_dur <= 0:
                    continue

                if has_l3:
                    # L3 actions may reference parent_phase_id directly
                    n_acts = sum(
                        1 for a in all_actions
                        if isinstance(a, dict) and a.get("parent_phase_id") == pid
                    )
                    if n_acts >= min_actions:
                        video_yield["L3"] += 1
                        records["input_durations"]["L3"].append(ph_dur)
                        records["output_counts"]["L3"].append(n_acts)

        for lv in ("L1", "L2", "L3"):
            records["per_level"][lv] += video_yield[lv]

        records["per_video"].append(video_yield)

    return records


# =====================================================================
# Fig 1: 每层级训练数据产出量
# =====================================================================
def plot_training_yield(results: dict, n_videos: int, ax: plt.Axes):
    levels = ["L1", "L2", "L3"]
    counts = [results["per_level"][lv] for lv in levels]
    colors = [LEVEL_COLORS[lv] for lv in levels]
    labels = [
        f"L1\n(1 per video)",
        f"L2\n(1 per phase)",
        f"L3\n(1 per leaf)",
    ]

    bars = ax.bar(labels, counts, color=colors, edgecolor="white", linewidth=0.8, width=0.5)

    for bar, cnt, lv in zip(bars, counts, levels):
        multiplier = cnt / n_videos if n_videos > 0 else 0
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.02,
            f"{cnt}\n({multiplier:.1f}x/video)",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_ylabel("Training Records")
    ax.set_title(f"Training Data Yield per Level ({n_videos} source videos)")
    ax.set_ylim(0, max(counts) * 1.3)


# =====================================================================
# Fig 2: 两层 Domain 嵌套饼图
# =====================================================================
def plot_domain_sunburst(anns: list[dict], ax: plt.Axes):
    l1_counter = Counter()
    l2_counter = Counter()
    for ann in anns:
        d1 = ann.get("domain_l1", "other")
        d2 = ann.get("domain_l2", "other")
        l1_counter[d1] += 1
        l2_counter[(d1, d2)] += 1

    l1_sorted = sorted(l1_counter.keys(), key=lambda x: -l1_counter[x])

    inner_sizes, inner_colors, inner_labels = [], [], []
    outer_sizes, outer_colors, outer_labels = [], [], []

    for d1 in l1_sorted:
        inner_sizes.append(l1_counter[d1])
        inner_colors.append(DOMAIN_L1_COLORS.get(d1, "#999"))
        pct = l1_counter[d1] / len(anns) * 100
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
            pct2 = cnt / len(anns) * 100
            outer_labels.append(f"{d2}\n({cnt})" if pct2 >= 2 else "")

    wedge_kwargs = dict(edgecolor="white", linewidth=1.5)
    ax.pie(
        inner_sizes, radius=0.65, colors=inner_colors,
        labels=inner_labels, labeldistance=0.35,
        textprops={"fontsize": 8, "fontweight": "bold"},
        wedgeprops=wedge_kwargs,
    )
    ax.pie(
        outer_sizes, radius=1.0, colors=outer_colors,
        labels=outer_labels, labeldistance=1.12,
        textprops={"fontsize": 7},
        wedgeprops={**wedge_kwargs, "width": 0.3},
    )
    ax.set_title(f"Domain Distribution (L1 inner / L2 outer)\n{len(anns)} videos", pad=20)


# =====================================================================
# Fig 3: 各层级输入时长分布 (overlaid histogram)
# =====================================================================
def plot_input_duration_distribution(results: dict, ax: plt.Axes):
    levels = ["L1", "L2", "L3"]
    for lv in levels:
        durations = results["input_durations"][lv]
        if durations:
            ax.hist(
                durations, bins=40, alpha=0.55, label=f"{lv} (n={len(durations)})",
                color=LEVEL_COLORS[lv], edgecolor="white", linewidth=0.3,
            )

    ax.set_xlabel("Input Duration (sec)")
    ax.set_ylabel("Count")
    ax.set_title("Input Duration Distribution per Level\n(L1=video, L2=phase, L3=leaf)")
    ax.legend()

    # 标注统计量
    text_lines = []
    for lv in levels:
        durs = results["input_durations"][lv]
        if durs:
            med = np.median(durs)
            avg = np.mean(durs)
            text_lines.append(f"{lv}: avg={avg:.0f}s, med={med:.0f}s")
    ax.text(
        0.97, 0.95, "\n".join(text_lines),
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )


# =====================================================================
# Fig 4: 各层级产出按 domain_l1 分组 (grouped bar)
# =====================================================================
def plot_yield_by_domain_l1(results: dict, ax: plt.Axes):
    per_video = results["per_video"]
    levels = ["L1", "L2", "L3"]

    domain_yield: dict[str, dict[str, int]] = defaultdict(lambda: {lv: 0 for lv in levels})
    for v in per_video:
        d1 = v["domain_l1"]
        for lv in levels:
            domain_yield[d1][lv] += v[lv]

    domains = sorted(domain_yield.keys(), key=lambda d: -sum(domain_yield[d].values()))
    x = np.arange(len(domains))
    width = 0.22

    for i, lv in enumerate(levels):
        vals = [domain_yield[d][lv] for d in domains]
        ax.bar(x + i * width, vals, width, label=lv, color=LEVEL_COLORS[lv],
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(x + width)
    ax.set_xticklabels(domains, rotation=15)
    ax.set_ylabel("Training Records")
    ax.set_title("Training Data Yield by Domain (L1)")
    ax.legend(loc="upper right")


# =====================================================================
# Fig 5: 每层级按 domain_l2 细分产出 (stacked bar)
# =====================================================================
def plot_yield_by_domain_l2(results: dict, ax: plt.Axes):
    per_video = results["per_video"]
    levels = ["L1", "L2", "L3"]

    # domain_l2 → {L1: n, L2: n, L3: n}
    d2_yield: dict[str, dict[str, int]] = defaultdict(lambda: {lv: 0 for lv in levels})
    for v in per_video:
        d2 = v["domain_l2"]
        for lv in levels:
            d2_yield[d2][lv] += v[lv]

    # 按总量排序, 取 top 12
    d2_sorted = sorted(d2_yield.keys(), key=lambda d: -sum(d2_yield[d].values()))
    if len(d2_sorted) > 12:
        d2_sorted = d2_sorted[:12]

    x = np.arange(len(d2_sorted))
    bottom = np.zeros(len(d2_sorted))

    for lv in levels:
        vals = np.array([d2_yield[d][lv] for d in d2_sorted], dtype=float)
        ax.bar(x, vals, bottom=bottom, label=lv, color=LEVEL_COLORS[lv],
               edgecolor="white", linewidth=0.5)
        bottom += vals

    # 标注总数
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
# Fig 6: Topology × Domain 热力图
# =====================================================================
def plot_topo_domain_heatmap(anns: list[dict], ax: plt.Axes):
    cross = Counter()
    for ann in anns:
        topo = ann.get("topology_type", "unknown")
        d1 = ann.get("domain_l1", "other")
        cross[(topo, d1)] += 1

    topos = sorted({k[0] for k in cross}, key=lambda t: -sum(v for k, v in cross.items() if k[0] == t))
    domains = sorted({k[1] for k in cross}, key=lambda d: -sum(v for k, v in cross.items() if k[1] == d))

    matrix = np.zeros((len(topos), len(domains)))
    for i, t in enumerate(topos):
        for j, d in enumerate(domains):
            matrix[i, j] = cross.get((t, d), 0)

    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels(domains, rotation=30, ha="right")
    ax.set_yticks(range(len(topos)))
    ax.set_yticklabels(topos)

    for i in range(len(topos)):
        for j in range(len(domains)):
            val = int(matrix[i, j])
            if val > 0:
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=9, color="white" if val > matrix.max() * 0.6 else "black")

    ax.set_title("Topology x Domain (L1) Distribution")
    plt.colorbar(im, ax=ax, shrink=0.8)


# =====================================================================
# Fig 7: L3 完整率按 domain_l2 分组
# =====================================================================
def plot_l3_completeness_by_domain(anns: list[dict], ax: plt.Axes):
    d2_total: Counter = Counter()
    d2_has_l3: Counter = Counter()
    for ann in anns:
        d2 = ann.get("domain_l2", "other")
        d2_total[d2] += 1
        l3 = ann.get("level3") or {}
        if l3.get("grounding_results"):
            d2_has_l3[d2] += 1

    domains = sorted(d2_total.keys(), key=lambda d: -d2_total[d])
    totals = [d2_total[d] for d in domains]
    has_l3 = [d2_has_l3.get(d, 0) for d in domains]
    no_l3 = [t - h for t, h in zip(totals, has_l3)]

    x = np.arange(len(domains))
    ax.bar(x, has_l3, label="Has L3", color="#55A868", edgecolor="white")
    ax.bar(x, no_l3, bottom=has_l3, label="No L3", color="#CCCCCC", edgecolor="white")

    for i, (h, t) in enumerate(zip(has_l3, totals)):
        if t > 0:
            pct = h / t * 100
            ax.text(i, t + max(totals) * 0.01, f"{pct:.0f}%", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Videos")
    ax.set_title("L3 Annotation Completeness by Domain (L2)")
    ax.legend(loc="upper right")


# =====================================================================
# Fig 8: 每层级输出 segments 数量分布 (3 subplot)
# =====================================================================
def plot_output_count_distribution(results: dict, axes: list[plt.Axes]):
    """每层级模型需要划分多少个 segments 的计数分布 (bar chart)。"""
    levels = ["L1", "L2", "L3"]
    titles = [
        "L1: # Phases per Video",
        "L2: # Events per Phase",
        "L3: # Actions per Leaf",
    ]
    for ax, lv, title in zip(axes, levels, titles):
        counts = results["output_counts"][lv]
        if not counts:
            ax.set_title(f"{title}\n(no data)")
            continue

        counter = Counter(counts)
        xs = sorted(counter.keys())
        ys = [counter[x] for x in xs]

        ax.bar(xs, ys, color=LEVEL_COLORS[lv], edgecolor="white", linewidth=0.5)
        ax.set_xlabel("# Segments")
        ax.set_ylabel("Count")
        ax.set_title(title)

        # 标注统计量
        avg = np.mean(counts)
        med = np.median(counts)
        ax.axvline(avg, color="red", linestyle="--", alpha=0.7, linewidth=1)
        ax.axvline(med, color="orange", linestyle="--", alpha=0.7, linewidth=1)
        ax.text(
            0.97, 0.95,
            f"n={len(counts)}\navg={avg:.1f}\nmed={med:.0f}\nmax={max(counts)}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        # x 轴整数刻度
        ax.set_xticks(xs if len(xs) <= 20 else np.linspace(min(xs), max(xs), 15, dtype=int))


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--annotation-dir", required=True,
                        help="标注 JSON 目录")
    parser.add_argument("--output-dir", default="./figures",
                        help="图片保存目录 (默认: ./figures)")
    parser.add_argument("--complete-only", action="store_true",
                        help="仅统计有 L3 标注的视频")
    parser.add_argument("--min-actions", type=int, default=3,
                        help="L3 每 leaf node 最少 micro-action 数")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    ann_dir = Path(args.annotation_dir)
    anns = load_annotations(ann_dir, args.complete_only)
    if not anns:
        print(f"No annotation files found in {ann_dir}")
        return

    print(f"Loaded {len(anns)} annotations from {ann_dir}")

    results = compute_training_records(anns, min_actions=args.min_actions)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Fig 1: Training yield per level ──
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    plot_training_yield(results, len(anns), ax1)
    fig1.tight_layout()
    fig1.savefig(os.path.join(args.output_dir, "fig1_training_yield.png"), dpi=args.dpi)
    print(f"  Saved fig1_training_yield.png")

    # ── Fig 2: Domain sunburst ──
    fig2, ax2 = plt.subplots(figsize=(9, 9))
    plot_domain_sunburst(anns, ax2)
    fig2.savefig(os.path.join(args.output_dir, "fig2_domain_sunburst.png"), dpi=args.dpi)
    print(f"  Saved fig2_domain_sunburst.png")

    # ── Fig 3: Input duration distribution ──
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    plot_input_duration_distribution(results, ax3)
    fig3.tight_layout()
    fig3.savefig(os.path.join(args.output_dir, "fig3_input_duration.png"), dpi=args.dpi)
    print(f"  Saved fig3_input_duration.png")

    # ── Fig 4: Yield by domain L1 ──
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    plot_yield_by_domain_l1(results, ax4)
    fig4.tight_layout()
    fig4.savefig(os.path.join(args.output_dir, "fig4_yield_by_domain_l1.png"), dpi=args.dpi)
    print(f"  Saved fig4_yield_by_domain_l1.png")

    # ── Fig 5: Yield by domain L2 ──
    fig5, ax5 = plt.subplots(figsize=(12, 5))
    plot_yield_by_domain_l2(results, ax5)
    fig5.tight_layout()
    fig5.savefig(os.path.join(args.output_dir, "fig5_yield_by_domain_l2.png"), dpi=args.dpi)
    print(f"  Saved fig5_yield_by_domain_l2.png")

    # ── Fig 6: Topology × Domain heatmap ──
    fig6, ax6 = plt.subplots(figsize=(8, 5))
    plot_topo_domain_heatmap(anns, ax6)
    fig6.tight_layout()
    fig6.savefig(os.path.join(args.output_dir, "fig6_topo_domain_heatmap.png"), dpi=args.dpi)
    print(f"  Saved fig6_topo_domain_heatmap.png")

    # ── Fig 7: L3 completeness by domain ──
    fig7, ax7 = plt.subplots(figsize=(10, 5))
    plot_l3_completeness_by_domain(anns, ax7)
    fig7.tight_layout()
    fig7.savefig(os.path.join(args.output_dir, "fig7_l3_completeness.png"), dpi=args.dpi)
    print(f"  Saved fig7_l3_completeness.png")

    # ── Fig 8: Output segments count distribution ──
    fig8, axes8 = plt.subplots(1, 3, figsize=(16, 5))
    plot_output_count_distribution(results, list(axes8))
    fig8.suptitle("Output Segments Count Distribution per Level", fontsize=13, y=1.02)
    fig8.tight_layout()
    fig8.savefig(os.path.join(args.output_dir, "fig8_output_counts.png"), dpi=args.dpi,
                 bbox_inches="tight")
    print(f"  Saved fig8_output_counts.png")

    plt.close("all")

    # ── 打印汇总 ──
    print(f"\n{'='*60}")
    print(f"  TRAINING DATA YIELD SUMMARY")
    print(f"{'='*60}")
    print(f"  Source videos: {len(anns)}")
    for lv in ("L1", "L2", "L3"):
        cnt = results["per_level"][lv]
        mult = cnt / len(anns) if len(anns) > 0 else 0
        durs = results["input_durations"][lv]
        dur_info = f"avg={np.mean(durs):.0f}s" if durs else "N/A"
        print(f"  {lv}: {cnt:5d} records  ({mult:.1f}x/video)  input {dur_info}")
    total = sum(results["per_level"].values())
    print(f"  TOTAL: {total:5d} records  ({total / len(anns):.1f}x/video)")
    print(f"\n  Figures saved to: {os.path.abspath(args.output_dir)}/")


if __name__ == "__main__":
    main()
