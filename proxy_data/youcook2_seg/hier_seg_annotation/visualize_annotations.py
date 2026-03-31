#!/usr/bin/env python3
"""
visualize_annotations.py — 标注数据可视化分析 (支持筛选参数)

训练数据层级结构:
  L1: 整个视频输入 → 分割出 macro phases  (每视频 1 条)
  L2: 每个 L1 phase 输入 → 分割出 events  (每 phase 1 条)
  L3: 每个 leaf node (event/empty-phase) 输入 → 分割出 micro-actions  (每 leaf 1 条)

筛选参数:
  每层级可设置输出 segments 数的 min/max 阈值 (--l1-min-phases, --l1-max-phases, ...)
  筛选后统计图表同时展示 before/after 对比

生成图表:
  Fig 1: 每层级训练数据产出量 (bar + 倍率)
  Fig 2: domain_l1 × domain_l2 两层嵌套饼图
  Fig 3: 各层级输入时长分布 (overlaid histogram)
  Fig 4: 各层级产出按 domain_l1 分组 (grouped bar)
  Fig 5: 每层级按 domain_l2 细分产出 (stacked bar)
  Fig 6: Topology × Domain 热力图
  Fig 7: L3 完整率按 domain_l2 分组
  Fig 8: 每层级输出 segments 数量分布 (before/after 筛选对比)

用法:
    # 不筛选, 只看原始分布
    python visualize_annotations.py \
        --annotation-dir /path/to/annotations

    # 设置筛选阈值, 看筛选后的分布
    python visualize_annotations.py \
        --annotation-dir /path/to/annotations \
        --l1-min-phases 2 --l1-max-phases 6 \
        --l2-min-events 2 --l2-max-events 8 \
        --l3-min-actions 2 --l3-max-actions 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
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
# 筛选参数
# =====================================================================
@dataclass
class FilterConfig:
    """每层级输出 segments 数的 min/max 筛选阈值。"""
    l1_min_phases: int = 0
    l1_max_phases: int = 999
    l2_min_events: int = 0
    l2_max_events: int = 999
    l3_min_actions: int = 0
    l3_max_actions: int = 999

    def active(self) -> bool:
        """是否有任何非默认筛选条件。"""
        return (self.l1_min_phases > 0 or self.l1_max_phases < 999
                or self.l2_min_events > 0 or self.l2_max_events < 999
                or self.l3_min_actions > 0 or self.l3_max_actions < 999)

    def summary(self) -> str:
        parts = []
        if self.l1_min_phases > 0 or self.l1_max_phases < 999:
            parts.append(f"L1: {self.l1_min_phases}-{self.l1_max_phases} phases")
        if self.l2_min_events > 0 or self.l2_max_events < 999:
            parts.append(f"L2: {self.l2_min_events}-{self.l2_max_events} events")
        if self.l3_min_actions > 0 or self.l3_max_actions < 999:
            parts.append(f"L3: {self.l3_min_actions}-{self.l3_max_actions} actions")
        return ", ".join(parts) if parts else "no filter"


# =====================================================================
# 每条训练记录的原始信息
# =====================================================================
@dataclass
class ContextExpansion:
    """L2/L3 上下文扩展参数: 先 grounding 再 segmentation。"""
    l2_target_dur: int = 0    # L2 目标最小输入时长 (秒), 0=不扩展
    l3_target_dur: int = 0    # L3 目标最小输入时长 (秒), 0=不扩展
    max_dur: int = 240        # 扩展后的最大时长上限 (秒)

    def active(self) -> bool:
        return self.l2_target_dur > 0 or self.l3_target_dur > 0

    def summary(self) -> str:
        parts = []
        if self.l2_target_dur > 0:
            parts.append(f"L2 target≥{self.l2_target_dur}s")
        if self.l3_target_dur > 0:
            parts.append(f"L3 target≥{self.l3_target_dur}s")
        return ", ".join(parts) if parts else "no expansion"


# =====================================================================
# 每条训练记录的原始信息
# =====================================================================
@dataclass
class TrainRecord:
    level: str          # "L1" / "L2" / "L3"
    clip_key: str
    domain_l1: str
    domain_l2: str
    topology: str
    input_duration: float   # 输入片段时长 (秒)
    output_count: int       # 要输出多少个 segments
    expanded_duration: float = 0.0  # 上下文扩展后的输入时长 (0=未扩展)


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
# 提取所有训练记录 (不做筛选)
# =====================================================================
def _expand_duration(
    seg_start: float, seg_end: float, video_dur: float, target: int, max_dur: int,
) -> float:
    """将 [seg_start, seg_end] 向两侧对称扩展到 target 秒, 不超过 video_dur 和 max_dur。"""
    seg_dur = seg_end - seg_start
    if target <= 0 or seg_dur >= target:
        return seg_dur
    need = target - seg_dur
    half = need / 2
    # 向两侧扩展, clamp 到视频边界
    new_start = max(0, seg_start - half)
    new_end = min(video_dur, seg_end + half)
    # 如果一侧碰到边界, 把剩余量加到另一侧
    actual = new_end - new_start
    if actual < target:
        shortfall = target - actual
        if new_start == 0:
            new_end = min(video_dur, new_end + shortfall)
        else:
            new_start = max(0, new_start - shortfall)
    expanded = min(new_end - new_start, max_dur)
    return max(expanded, seg_dur)  # 至少不缩短


def extract_all_records(
    anns: list[dict], ctx: ContextExpansion | None = None,
) -> list[TrainRecord]:
    """从标注中提取所有可能的训练记录。"""
    if ctx is None:
        ctx = ContextExpansion()
    records: list[TrainRecord] = []

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

        common = dict(clip_key=clip_key, domain_l1=domain_l1,
                      domain_l2=domain_l2, topology=topo)

        # ── L1: 1 条 per video ──
        valid_phases = [
            p for p in phases
            if isinstance(p.get("start_time"), (int, float))
            and isinstance(p.get("end_time"), (int, float))
            and p["end_time"] > p["start_time"]
        ]
        if valid_phases and clip_duration > 0:
            records.append(TrainRecord(
                level="L1", input_duration=clip_duration,
                output_count=len(valid_phases), **common,
            ))

        # ── phase → events 映射 ──
        phase_event_map: dict[int, list[dict]] = defaultdict(list)
        for ev in events:
            if isinstance(ev, dict) and ev.get("parent_phase_id") is not None:
                phase_event_map[ev["parent_phase_id"]].append(ev)

        # ── L2: 1 条 per phase ──
        for phase in valid_phases:
            pid = phase.get("phase_id")
            ph_start = float(phase.get("start_time", 0))
            ph_end = float(phase.get("end_time", 0))
            ph_dur = ph_end - ph_start
            children = phase_event_map.get(pid, [])
            if ph_dur > 0:
                exp_dur = _expand_duration(
                    ph_start, ph_end, clip_duration,
                    ctx.l2_target_dur, ctx.max_dur,
                )
                records.append(TrainRecord(
                    level="L2", input_duration=ph_dur,
                    output_count=len(children),
                    expanded_duration=exp_dur,
                    **common,
                ))

        # ── L3: 1 条 per leaf node ──
        has_l3 = all_actions and not l3.get("_parse_error")
        if not has_l3:
            continue

        for phase in valid_phases:
            pid = phase.get("phase_id")
            children = phase_event_map.get(pid, [])

            if children:
                for ev in children:
                    ev_start = float(ev.get("start_time", 0))
                    ev_end = float(ev.get("end_time", 0))
                    ev_dur = ev_end - ev_start
                    eid = ev.get("event_id")
                    if ev_dur <= 0:
                        continue
                    n_acts = sum(
                        1 for a in all_actions
                        if isinstance(a, dict) and a.get("parent_event_id") == eid
                    )
                    if n_acts > 0:
                        exp_dur = _expand_duration(
                            ev_start, ev_end, clip_duration,
                            ctx.l3_target_dur, ctx.max_dur,
                        )
                        records.append(TrainRecord(
                            level="L3", input_duration=ev_dur,
                            output_count=n_acts,
                            expanded_duration=exp_dur,
                            **common,
                        ))
            else:
                ph_start = float(phase.get("start_time", 0))
                ph_end = float(phase.get("end_time", 0))
                ph_dur = ph_end - ph_start
                if ph_dur <= 0:
                    continue
                n_acts = sum(
                    1 for a in all_actions
                    if isinstance(a, dict) and a.get("parent_phase_id") == pid
                )
                if n_acts > 0:
                    exp_dur = _expand_duration(
                        ph_start, ph_end, clip_duration,
                        ctx.l3_target_dur, ctx.max_dur,
                    )
                    records.append(TrainRecord(
                        level="L3", input_duration=ph_dur,
                        output_count=n_acts,
                        expanded_duration=exp_dur,
                        **common,
                    ))

    return records


# =====================================================================
# 筛选
# =====================================================================
def apply_filter(records: list[TrainRecord], cfg: FilterConfig) -> list[TrainRecord]:
    """按 output_count 的 min/max 区间筛选。"""
    filtered = []
    for r in records:
        if r.level == "L1":
            if not (cfg.l1_min_phases <= r.output_count <= cfg.l1_max_phases):
                continue
        elif r.level == "L2":
            if not (cfg.l2_min_events <= r.output_count <= cfg.l2_max_events):
                continue
        elif r.level == "L3":
            if not (cfg.l3_min_actions <= r.output_count <= cfg.l3_max_actions):
                continue
        filtered.append(r)
    return filtered


# =====================================================================
# 领域均衡采样
# =====================================================================
def balanced_sample(
    records: list[TrainRecord],
    target_per_level: int = 800,
    seed: int = 42,
) -> list[TrainRecord]:
    """按领域均衡采样, 超出的域优先砍 output_count 小的记录。

    策略 (每层级独立):
      1. 按 domain_l2 分组
      2. 计算每个域的 quota = target / n_domains (均匀分配)
      3. 不足 quota 的域全部保留, 多余名额重分配给其他域
      4. 需要裁剪的域内按 output_count 升序排列, 优先丢弃事件少的
    """
    import random
    rng = random.Random(seed)

    sampled: list[TrainRecord] = []
    for lv in ("L1", "L2", "L3"):
        lv_records = [r for r in records if r.level == lv]
        if not lv_records or target_per_level <= 0:
            sampled.extend(lv_records)
            continue

        if len(lv_records) <= target_per_level:
            sampled.extend(lv_records)
            continue

        # 按 domain_l2 分组
        by_domain: dict[str, list[TrainRecord]] = defaultdict(list)
        for r in lv_records:
            by_domain[r.domain_l2].append(r)

        n_domains = len(by_domain)
        base_quota = target_per_level // n_domains
        remaining = target_per_level

        # 第一轮: 不足 quota 的域全部保留
        small_domains = {}
        large_domains = {}
        for d, recs in by_domain.items():
            if len(recs) <= base_quota:
                small_domains[d] = recs
                remaining -= len(recs)
            else:
                large_domains[d] = recs

        # 第二轮: 大域的名额重分配
        if large_domains:
            quota_for_large = remaining // len(large_domains)
            extra = remaining - quota_for_large * len(large_domains)

            # 按域大小排序 (最大的域分到 extra)
            sorted_large = sorted(large_domains.items(), key=lambda x: -len(x[1]))

            for idx, (d, recs) in enumerate(sorted_large):
                q = quota_for_large + (1 if idx < extra else 0)
                # 按 output_count 降序排列, 保留事件多的
                recs_sorted = sorted(recs, key=lambda r: -r.output_count)
                sampled.extend(recs_sorted[:q])
        for d, recs in small_domains.items():
            sampled.extend(recs)

    return sampled


# =====================================================================
# 聚合统计
# =====================================================================
def aggregate(records: list[TrainRecord]) -> dict:
    """从记录列表聚合出绘图需要的统计数据。"""
    per_level = Counter()
    input_durations: dict[str, list[float]] = {"L1": [], "L2": [], "L3": []}
    expanded_durations: dict[str, list[float]] = {"L1": [], "L2": [], "L3": []}
    output_counts: dict[str, list[int]] = {"L1": [], "L2": [], "L3": []}
    per_video: dict[str, dict] = {}  # clip_key → {domain_l1, domain_l2, L1, L2, L3}
    # 按层级分组, 方便各层级独立做 domain 统计
    level_records: dict[str, list[TrainRecord]] = {"L1": [], "L2": [], "L3": []}

    for r in records:
        per_level[r.level] += 1
        input_durations[r.level].append(r.input_duration)
        expanded_durations[r.level].append(
            r.expanded_duration if r.expanded_duration > 0 else r.input_duration
        )
        output_counts[r.level].append(r.output_count)
        level_records[r.level].append(r)

        if r.clip_key not in per_video:
            per_video[r.clip_key] = {
                "clip_key": r.clip_key,
                "domain_l1": r.domain_l1,
                "domain_l2": r.domain_l2,
                "topology": r.topology,
                "L1": 0, "L2": 0, "L3": 0,
            }
        per_video[r.clip_key][r.level] += 1

    return {
        "per_level": per_level,
        "input_durations": input_durations,
        "expanded_durations": expanded_durations,
        "output_counts": output_counts,
        "per_video": list(per_video.values()),
        "level_records": level_records,
    }


# =====================================================================
# Fig 1: 每层级训练数据产出量
# =====================================================================
def plot_training_yield(stats: dict, stats_raw: dict, n_videos: int,
                        filt: FilterConfig, ax: plt.Axes):
    levels = ["L1", "L2", "L3"]
    counts = [stats["per_level"].get(lv, 0) for lv in levels]
    raw_counts = [stats_raw["per_level"].get(lv, 0) for lv in levels]
    colors = [LEVEL_COLORS[lv] for lv in levels]
    labels = ["L1\n(per video)", "L2\n(per phase)", "L3\n(per leaf)"]

    if filt.active():
        # raw bars (grey background)
        ax.bar(labels, raw_counts, color="#DDDDDD", edgecolor="white",
               linewidth=0.8, width=0.5, label="Before filter")
    bars = ax.bar(labels, counts, color=colors, edgecolor="white",
                  linewidth=0.8, width=0.5, alpha=0.85 if filt.active() else 1.0,
                  label="After filter" if filt.active() else None)

    for bar, cnt, raw in zip(bars, counts, raw_counts):
        mult = cnt / n_videos if n_videos > 0 else 0
        text = f"{cnt}\n({mult:.1f}x)"
        if filt.active() and raw > 0:
            text += f"\n[{cnt/raw*100:.0f}% kept]"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(cnt, raw if filt.active() else 0) + max(raw_counts) * 0.02,
            text, ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    title = f"Training Data Yield per Level ({n_videos} videos)"
    if filt.active():
        title += f"\nFilter: {filt.summary()}"
    ax.set_ylabel("Training Records")
    ax.set_title(title)
    ax.set_ylim(0, max(max(raw_counts), max(counts)) * 1.35)
    if filt.active():
        ax.legend()


# =====================================================================
# Fig 2: 筛选后三层各自的 domain 分布 (3 个嵌套饼图)
# =====================================================================
def _draw_one_domain_donut(records: list[TrainRecord], level: str, ax: plt.Axes):
    """为单个层级画 domain_l1(内环) + domain_l2(外环) 嵌套饼图。"""
    if not records:
        ax.set_title(f"{level} (no data)")
        ax.axis("off")
        return

    # 统计: 用去重的 clip_key 计数 (一个视频在某层可能贡献多条 record)
    l1_counter = Counter()
    l2_counter = Counter()

    # 按 record 数计  (而非 clip 数), 更准确反映训练数据分布
    for r in records:
        l1_counter[r.domain_l1] += 1
        l2_counter[(r.domain_l1, r.domain_l2)] += 1

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
    ax.pie(
        inner_sizes, radius=0.65, colors=inner_colors,
        labels=inner_labels, labeldistance=0.35,
        textprops={"fontsize": 7, "fontweight": "bold"},
        wedgeprops=wedge_kwargs,
    )
    ax.pie(
        outer_sizes, radius=1.0, colors=outer_colors,
        labels=outer_labels, labeldistance=1.12,
        textprops={"fontsize": 6},
        wedgeprops={**wedge_kwargs, "width": 0.3},
    )
    ax.set_title(f"{level}  ({total} records)", fontsize=11, pad=15)


def plot_domain_per_level(stats: dict, axes: list[plt.Axes]):
    """三个层级各画一个 domain 嵌套饼图。"""
    for ax, lv in zip(axes, ["L1", "L2", "L3"]):
        _draw_one_domain_donut(stats["level_records"][lv], lv, ax)


# =====================================================================
# Fig 3: 各层级输入时长分布 (overlaid histogram)
# =====================================================================
def plot_input_duration_distribution(stats: dict, ax: plt.Axes):
    levels = ["L1", "L2", "L3"]
    for lv in levels:
        durations = stats["input_durations"][lv]
        if durations:
            ax.hist(
                durations, bins=40, alpha=0.55, label=f"{lv} (n={len(durations)})",
                color=LEVEL_COLORS[lv], edgecolor="white", linewidth=0.3,
            )

    ax.set_xlabel("Input Duration (sec)")
    ax.set_ylabel("Count")
    ax.set_title("Input Duration Distribution per Level\n(L1=video, L2=phase, L3=leaf)")
    ax.legend()

    text_lines = []
    for lv in levels:
        durs = stats["input_durations"][lv]
        if durs:
            text_lines.append(f"{lv}: avg={np.mean(durs):.0f}s, med={np.median(durs):.0f}s")
    ax.text(
        0.97, 0.95, "\n".join(text_lines),
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )


# =====================================================================
# Fig 4: 各层级产出按 domain_l1 分组 (grouped bar)
# =====================================================================
def plot_yield_by_domain_l1(stats: dict, ax: plt.Axes):
    per_video = stats["per_video"]
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
def plot_yield_by_domain_l2(stats: dict, ax: plt.Axes):
    per_video = stats["per_video"]
    levels = ["L1", "L2", "L3"]

    d2_yield: dict[str, dict[str, int]] = defaultdict(lambda: {lv: 0 for lv in levels})
    for v in per_video:
        d2 = v["domain_l2"]
        for lv in levels:
            d2_yield[d2][lv] += v[lv]

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

    topos = sorted({k[0] for k in cross},
                   key=lambda t: -sum(v for k, v in cross.items() if k[0] == t))
    domains = sorted({k[1] for k in cross},
                     key=lambda d: -sum(v for k, v in cross.items() if k[1] == d))

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
# Fig 8: 每层级输出 segments 数量分布 (before / after 对比)
# =====================================================================
def plot_output_count_distribution(
    stats: dict, stats_raw: dict, filt: FilterConfig, axes: list[plt.Axes],
):
    """每层级模型要输出多少 segments, 筛选前(灰) vs 筛选后(彩) 对比。"""
    levels = ["L1", "L2", "L3"]
    titles = [
        "L1: # Phases per Video",
        "L2: # Events per Phase",
        "L3: # Actions per Leaf",
    ]
    filter_ranges = [
        (filt.l1_min_phases, filt.l1_max_phases),
        (filt.l2_min_events, filt.l2_max_events),
        (filt.l3_min_actions, filt.l3_max_actions),
    ]

    for ax, lv, title, (fmin, fmax) in zip(axes, levels, titles, filter_ranges):
        raw_counts = stats_raw["output_counts"][lv]
        filt_counts = stats["output_counts"][lv]

        if not raw_counts:
            ax.set_title(f"{title}\n(no data)")
            continue

        # 计算 bin range
        all_vals = raw_counts
        counter_raw = Counter(all_vals)
        counter_filt = Counter(filt_counts)
        xs = sorted(counter_raw.keys())
        ys_raw = [counter_raw[x] for x in xs]
        ys_filt = [counter_filt.get(x, 0) for x in xs]

        # 灰色: 原始分布
        ax.bar(xs, ys_raw, color="#DDDDDD", edgecolor="white", linewidth=0.5,
               label=f"Raw (n={len(raw_counts)})")
        # 彩色: 筛选后
        ax.bar(xs, ys_filt, color=LEVEL_COLORS[lv], edgecolor="white", linewidth=0.5,
               alpha=0.85, label=f"Filtered (n={len(filt_counts)})")

        # 筛选区间标注
        if filt.active():
            ymax = max(max(ys_raw), 1)
            if fmin > 0:
                ax.axvline(fmin - 0.5, color="red", linestyle="--", alpha=0.7, linewidth=1.2)
                ax.text(fmin - 0.5, ymax * 0.95, f"min={fmin}", color="red",
                        fontsize=8, ha="right", rotation=90, va="top")
            if fmax < 999:
                ax.axvline(fmax + 0.5, color="red", linestyle="--", alpha=0.7, linewidth=1.2)
                ax.text(fmax + 0.5, ymax * 0.95, f"max={fmax}", color="red",
                        fontsize=8, ha="left", rotation=90, va="top")

        ax.set_xlabel("# Segments")
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.legend(fontsize=8)

        # 统计量 (筛选后)
        if filt_counts:
            avg = np.mean(filt_counts)
            med = np.median(filt_counts)
            kept_pct = len(filt_counts) / len(raw_counts) * 100 if raw_counts else 0
            info = f"kept: {len(filt_counts)}/{len(raw_counts)} ({kept_pct:.0f}%)"
            info += f"\navg={avg:.1f}  med={med:.0f}"
            info += f"\nmin={min(filt_counts)}  max={max(filt_counts)}"
            ax.text(
                0.97, 0.95, info,
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

        ax.set_xticks(xs if len(xs) <= 20 else
                       np.linspace(min(xs), max(xs), 15, dtype=int))


# =====================================================================
# Fig 9: 上下文扩展前后输入时长对比 (L2/L3 各一个 subplot)
# =====================================================================
def plot_context_expansion_comparison(
    stats: dict, ctx: ContextExpansion, axes: list[plt.Axes],
):
    """L2/L3 的原始时长 vs 扩展后时长对比直方图。"""
    configs = [
        ("L2", ctx.l2_target_dur),
        ("L3", ctx.l3_target_dur),
    ]
    for ax, (lv, target) in zip(axes, configs):
        orig = stats["input_durations"][lv]
        expanded = stats["expanded_durations"][lv]
        if not orig:
            ax.set_title(f"{lv}: no data")
            continue

        bins = np.linspace(0, max(max(orig), max(expanded)) + 5, 50)
        ax.hist(orig, bins=bins, alpha=0.5, color="#BBBBBB", edgecolor="white",
                linewidth=0.3, label=f"Original (avg={np.mean(orig):.0f}s)")
        ax.hist(expanded, bins=bins, alpha=0.6, color=LEVEL_COLORS[lv], edgecolor="white",
                linewidth=0.3, label=f"Expanded (avg={np.mean(expanded):.0f}s)")

        if target > 0:
            ax.axvline(target, color="red", linestyle="--", alpha=0.8, linewidth=1.5)
            ax.text(target + 1, ax.get_ylim()[1] * 0.9, f"target={target}s",
                    color="red", fontsize=9, va="top")

        ax.set_xlabel("Input Duration (sec)")
        ax.set_ylabel("Count")
        ax.set_title(f"{lv}: Original vs Context-Expanded Duration")
        ax.legend(fontsize=9)

        # 统计量对比
        info_lines = [
            f"Original:  avg={np.mean(orig):.0f}s  med={np.median(orig):.0f}s",
            f"Expanded:  avg={np.mean(expanded):.0f}s  med={np.median(expanded):.0f}s",
        ]
        if target > 0:
            below_orig = sum(1 for d in orig if d < target)
            below_exp = sum(1 for d in expanded if d < target)
            info_lines.append(f"<{target}s:  {below_orig}→{below_exp} "
                              f"({below_orig/len(orig)*100:.0f}%→{below_exp/len(expanded)*100:.0f}%)")
        ax.text(
            0.97, 0.72, "\n".join(info_lines),
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
        )


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
    parser.add_argument("--dpi", type=int, default=150)

    # ── 筛选参数 ──
    g = parser.add_argument_group("filter", "每层级输出 segments 数的 min/max 筛选阈值")
    g.add_argument("--l1-min-phases", type=int, default=0,
                   help="L1 每视频最少 phases 数 (默认: 0, 不限)")
    g.add_argument("--l1-max-phases", type=int, default=999,
                   help="L1 每视频最多 phases 数 (默认: 999, 不限)")
    g.add_argument("--l2-min-events", type=int, default=0,
                   help="L2 每 phase 最少 events 数 (默认: 0, 不限)")
    g.add_argument("--l2-max-events", type=int, default=999,
                   help="L2 每 phase 最多 events 数 (默认: 999, 不限)")
    g.add_argument("--l3-min-actions", type=int, default=0,
                   help="L3 每 leaf 最少 actions 数 (默认: 0, 不限)")
    g.add_argument("--l3-max-actions", type=int, default=999,
                   help="L3 每 leaf 最多 actions 数 (默认: 999, 不限)")

    # ── 上下文扩展参数 ──
    e = parser.add_argument_group("expansion",
        "L2/L3 上下文扩展: 向两侧 padding 到目标时长 (先 grounding 再 segmentation)")
    e.add_argument("--l2-target-dur", type=int, default=0,
                   help="L2 目标最小输入时长(秒), 不足则向两侧扩展 (默认: 0, 不扩展)")
    e.add_argument("--l3-target-dur", type=int, default=0,
                   help="L3 目标最小输入时长(秒), 不足则向两侧扩展 (默认: 0, 不扩展)")
    e.add_argument("--max-expansion-dur", type=int, default=240,
                   help="扩展后的最大时长上限(秒) (默认: 240)")

    # ── 均衡采样参数 ──
    s = parser.add_argument_group("sampling", "领域均衡采样")
    s.add_argument("--balance-per-level", type=int, default=0,
                   help="每层级目标保留条数 (默认: 0, 不做均衡采样). "
                        "超出的域优先砍 output_count 小的记录")
    s.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    filt = FilterConfig(
        l1_min_phases=args.l1_min_phases, l1_max_phases=args.l1_max_phases,
        l2_min_events=args.l2_min_events, l2_max_events=args.l2_max_events,
        l3_min_actions=args.l3_min_actions, l3_max_actions=args.l3_max_actions,
    )
    ctx = ContextExpansion(
        l2_target_dur=args.l2_target_dur,
        l3_target_dur=args.l3_target_dur,
        max_dur=args.max_expansion_dur,
    )

    ann_dir = Path(args.annotation_dir)
    anns = load_annotations(ann_dir, args.complete_only)
    if not anns:
        print(f"No annotation files found in {ann_dir}")
        return

    print(f"Loaded {len(anns)} annotations from {ann_dir}")
    print(f"Filter: {filt.summary()}")
    print(f"Context expansion: {ctx.summary()}")

    # 提取 & 筛选 & 均衡采样
    all_records = extract_all_records(anns, ctx)
    filtered_records = apply_filter(all_records, filt) if filt.active() else all_records
    if args.balance_per_level > 0:
        filtered_records = balanced_sample(
            filtered_records, target_per_level=args.balance_per_level, seed=args.seed,
        )
        print(f"Balanced sampling: {args.balance_per_level} per level")

    stats_raw = aggregate(all_records)
    stats = aggregate(filtered_records)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Fig 1 ──
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    plot_training_yield(stats, stats_raw, len(anns), filt, ax1)
    fig1.tight_layout()
    fig1.savefig(os.path.join(args.output_dir, "fig1_training_yield.png"), dpi=args.dpi)
    print(f"  Saved fig1_training_yield.png")

    # ── Fig 2: Domain distribution per level ──
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    plot_domain_per_level(stats, list(axes2))
    filter_label = f"  (Filter: {filt.summary()})" if filt.active() else ""
    fig2.suptitle(f"Domain Distribution per Level{filter_label}",
                  fontsize=13, y=1.02)
    fig2.tight_layout()
    fig2.savefig(os.path.join(args.output_dir, "fig2_domain_per_level.png"), dpi=args.dpi,
                 bbox_inches="tight")
    print(f"  Saved fig2_domain_per_level.png")

    # ── Fig 3 ──
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    plot_input_duration_distribution(stats, ax3)
    fig3.tight_layout()
    fig3.savefig(os.path.join(args.output_dir, "fig3_input_duration.png"), dpi=args.dpi)
    print(f"  Saved fig3_input_duration.png")

    # ── Fig 4 ──
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    plot_yield_by_domain_l1(stats, ax4)
    fig4.tight_layout()
    fig4.savefig(os.path.join(args.output_dir, "fig4_yield_by_domain_l1.png"), dpi=args.dpi)
    print(f"  Saved fig4_yield_by_domain_l1.png")

    # ── Fig 5 ──
    fig5, ax5 = plt.subplots(figsize=(12, 5))
    plot_yield_by_domain_l2(stats, ax5)
    fig5.tight_layout()
    fig5.savefig(os.path.join(args.output_dir, "fig5_yield_by_domain_l2.png"), dpi=args.dpi)
    print(f"  Saved fig5_yield_by_domain_l2.png")

    # ── Fig 6 ──
    fig6, ax6 = plt.subplots(figsize=(8, 5))
    plot_topo_domain_heatmap(anns, ax6)
    fig6.tight_layout()
    fig6.savefig(os.path.join(args.output_dir, "fig6_topo_domain_heatmap.png"), dpi=args.dpi)
    print(f"  Saved fig6_topo_domain_heatmap.png")

    # ── Fig 7 ──
    fig7, ax7 = plt.subplots(figsize=(10, 5))
    plot_l3_completeness_by_domain(anns, ax7)
    fig7.tight_layout()
    fig7.savefig(os.path.join(args.output_dir, "fig7_l3_completeness.png"), dpi=args.dpi)
    print(f"  Saved fig7_l3_completeness.png")

    # ── Fig 8 ──
    fig8, axes8 = plt.subplots(1, 3, figsize=(16, 5))
    plot_output_count_distribution(stats, stats_raw, filt, list(axes8))
    fig8.suptitle("Output Segments Count Distribution per Level", fontsize=13, y=1.02)
    fig8.tight_layout()
    fig8.savefig(os.path.join(args.output_dir, "fig8_output_counts.png"), dpi=args.dpi,
                 bbox_inches="tight")
    print(f"  Saved fig8_output_counts.png")

    # ── Fig 9: Context expansion comparison (only if expansion is active) ──
    if ctx.active():
        fig9, axes9 = plt.subplots(1, 2, figsize=(14, 5))
        plot_context_expansion_comparison(stats, ctx, list(axes9))
        fig9.suptitle(
            f"Context Expansion: Input Duration Before vs After\n({ctx.summary()})",
            fontsize=13, y=1.03,
        )
        fig9.tight_layout()
        fig9.savefig(os.path.join(args.output_dir, "fig9_context_expansion.png"),
                     dpi=args.dpi, bbox_inches="tight")
        print(f"  Saved fig9_context_expansion.png")

    plt.close("all")

    # ── 打印汇总 ──
    print(f"\n{'='*60}")
    print(f"  TRAINING DATA YIELD SUMMARY")
    print(f"{'='*60}")
    print(f"  Source videos: {len(anns)}")
    print(f"  Filter: {filt.summary()}")
    print(f"  Context expansion: {ctx.summary()}")
    print(f"{'─'*70}")
    print(f"  {'Level':<6} {'Raw':>6} {'Filt':>6} {'Kept%':>6}  {'Orig Dur':>12}  {'Expanded Dur':>14}")
    print(f"{'─'*70}")
    for lv in ("L1", "L2", "L3"):
        raw = stats_raw["per_level"].get(lv, 0)
        flt = stats["per_level"].get(lv, 0)
        pct = flt / raw * 100 if raw > 0 else 0
        durs = stats["input_durations"][lv]
        exp_durs = stats["expanded_durations"][lv]
        dur_info = f"avg={np.mean(durs):.0f}s" if durs else "N/A"
        exp_info = f"avg={np.mean(exp_durs):.0f}s" if exp_durs else "N/A"
        print(f"  {lv:<6} {raw:>6} {flt:>6} {pct:>5.1f}%  {dur_info:>12}  {exp_info:>14}")
    raw_total = sum(stats_raw["per_level"].values())
    flt_total = sum(stats["per_level"].values())
    pct_total = flt_total / raw_total * 100 if raw_total > 0 else 0
    print(f"{'─'*70}")
    print(f"  {'TOTAL':<6} {raw_total:>6} {flt_total:>6} {pct_total:>5.1f}%")
    print(f"\n  Figures saved to: {os.path.abspath(args.output_dir)}/")


if __name__ == "__main__":
    main()
