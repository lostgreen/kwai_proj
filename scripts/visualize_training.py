#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# -*- coding: utf-8 -*-
"""
多任务融合训练日志可视化脚本

用法:
    python scripts/visualize_training.py \
        -i /m2v_intern/xuboshen/zgw/RL-Models/qwen3_vl_mixed_proxy_dapo_2gpu/experiment_log.jsonl \
        -o checkpoints/training_curves.png

    # 交互模式（弹窗预览）
    python scripts/visualize_training.py \
        -i checkpoints/.../experiment_log.jsonl \
        --show
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


# ─── 样式 ────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":     "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid":       True,
    "grid.alpha":      0.3,
    "grid.linestyle":  "--",
    "lines.linewidth": 1.6,
    "figure.dpi":      150,
})

TASK_COLORS = {
    "overall":      "#2c3e50",
    "temporal_seg": "#e74c3c",
    "add":          "#3498db",
    "delete":       "#2ecc71",
    "replace":      "#f39c12",
    "sort":         "#9b59b6",
    "format":       "#95a5a6",
    "accuracy":     "#1abc9c",
}

SMOOTH_COEF = 0.8  # TensorBoard-style smoothing coefficient in [0, 1)


# ─── 数据加载 ─────────────────────────────────────

def load_log(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r["step"])
    return rows


def extract(rows: list[dict], *key_path: str, default=float("nan")) -> np.ndarray:
    """按 key_path 路径取嵌套字段，返回 ndarray。"""
    vals = []
    for r in rows:
        v = r
        for k in key_path:
            if isinstance(v, dict):
                v = v.get(k, None)
            else:
                v = None
                break
        vals.append(v if v is not None else default)
    return np.array(vals, dtype=float)


def ema_smooth(x: np.ndarray, smooth: float = SMOOTH_COEF) -> np.ndarray:
    """
    TensorBoard-style smoothing.

    y_t = smooth * y_{t-1} + (1 - smooth) * x_t
    where smooth in [0, 1). Larger smooth -> smoother curve.
    """
    if len(x) == 0:
        return x

    smooth = float(np.clip(smooth, 0.0, 0.9999))
    if smooth <= 0:
        return x.copy()

    out = np.empty_like(x)
    first_valid_idx = None
    for i, v in enumerate(x):
        if not np.isnan(v):
            first_valid_idx = i
            break

    if first_valid_idx is None:
        out[:] = np.nan
        return out

    out[:first_valid_idx] = np.nan
    out[first_valid_idx] = x[first_valid_idx]

    for i in range(first_valid_idx + 1, len(x)):
        if np.isnan(x[i]):
            out[i] = out[i - 1]
        else:
            out[i] = smooth * out[i - 1] + (1 - smooth) * x[i]
    return out


# ─── 绘图单元 ────────────────────────────────────

def plot_lines(ax, steps, series: dict, smooth=True, alpha_raw=0.25, ylabel="", title=""):
    """Plot multiple lines with optional EMA smoothing."""
    for name, vals in series.items():
        color = TASK_COLORS.get(name, None)
        if smooth and np.any(~np.isnan(vals)):
            ax.plot(steps, vals, color=color, alpha=alpha_raw)
            ax.plot(steps, ema_smooth(vals), color=color, label=name, linewidth=2.0)
        else:
            ax.plot(steps, vals, color=color, label=name)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, ncol=2, loc="best")


def plot_dense_reward(ax, steps, vals, name, color, ylabel="Reward", title=""):
    """Dense task (many steps): solid line + light fill under trend."""
    mask = ~np.isnan(vals)
    if not mask.any():
        return
    ax.plot(steps[mask], vals[mask], color=color, alpha=0.2, linewidth=0.8)
    smooth = ema_smooth(vals[mask])
    ax.plot(steps[mask], smooth, color=color, linewidth=2.2, label=name)
    ax.fill_between(steps[mask], smooth.min(), smooth, alpha=0.08, color=color)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)


def plot_proxy_task_reward(ax, steps, task_series: dict, ylabel="Reward", title=""):
    """
    Sparse proxy tasks (~25 active steps each).
    Draws: scatter dots (raw) + thick EMA trend line only.
    No connecting lines between non-adjacent points to avoid visual noise.
    """
    for name, vals in task_series.items():
        color = TASK_COLORS.get(name, None)
        mask = ~np.isnan(vals)
        if not mask.any():
            continue
        sx, sy = steps[mask], vals[mask]
        # raw scatter dots
        ax.scatter(sx, sy, color=color, s=22, zorder=4, alpha=0.5, linewidths=0)
        # EMA trend — the only line drawn
        smooth_y = ema_smooth(sy, smooth=SMOOTH_COEF)
        ax.plot(sx, smooth_y, color=color, linewidth=2.4,
                label=f"{name}  (n={mask.sum()})", solid_capstyle="round")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, ncol=1, loc="lower right",
              framealpha=0.85, edgecolor="#cccccc")


def fill_timing(ax, steps, timing: dict):
    """Stacked area chart showing time breakdown per step."""
    STAGE_ORDER = ["gen", "old", "ref", "update_actor", "reward", "adv"]
    STAGE_COLORS = {
        "gen":          "#3498db",
        "old":          "#e74c3c",
        "ref":          "#2ecc71",
        "update_actor": "#f39c12",
        "reward":       "#9b59b6",
        "adv":          "#95a5a6",
    }
    bottoms = np.zeros(len(steps))
    for stage in STAGE_ORDER:
        vals = timing.get(stage, np.full(len(steps), np.nan))
        mask = ~np.isnan(vals)
        if mask.any():
            ax.fill_between(steps, bottoms, bottoms + np.where(mask, vals, 0),
                            alpha=0.75, color=STAGE_COLORS.get(stage, None),
                            label=stage)
            bottoms += np.where(mask, vals, 0)
    ax.set_ylabel("Seconds / step", fontsize=9)
    ax.set_title("Step Time Breakdown", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, ncol=3, loc="upper right")


# ─── 主绘图函数 ───────────────────────────────────

def build_figure(rows: list[dict]) -> plt.Figure:
    steps = np.array([r["step"] for r in rows])

    # ── 收集数据 ──
    # Reward
    task_names = ["temporal_seg", "add", "delete", "replace", "sort"]
    reward_overall  = extract(rows, "reward", "overall")
    reward_accuracy = extract(rows, "reward", "accuracy")
    reward_format   = extract(rows, "reward", "format")
    reward_tasks    = {t: extract(rows, "reward", t) for t in task_names}

    # Actor
    kl_loss    = extract(rows, "actor", "kl_loss")
    ppo_kl     = extract(rows, "actor", "ppo_kl")
    pg_loss    = extract(rows, "actor", "pg_loss")
    entropy    = extract(rows, "actor", "entropy_loss")
    grad_norm  = extract(rows, "actor", "grad_norm")
    lr         = extract(rows, "actor", "lr")
    clip_hi    = extract(rows, "actor", "pg_clipfrac_higher")
    clip_lo    = extract(rows, "actor", "pg_clipfrac_lower")

    # Response / prompt length
    resp_mean = extract(rows, "response_length", "mean")
    resp_max  = extract(rows, "response_length", "max")
    resp_clip = extract(rows, "response_length", "clip_ratio")
    prom_mean = extract(rows, "prompt_length", "mean")

    # Advantages / returns
    adv_mean = extract(rows, "critic", "advantages", "mean")
    adv_max  = extract(rows, "critic", "advantages", "max")
    adv_min  = extract(rows, "critic", "advantages", "min")

    # Timing
    timing_stages = {}
    stage_keys = ["gen", "old", "ref", "update_actor", "reward", "adv"]
    for sk in stage_keys:
        timing_stages[sk] = extract(rows, "timing_s", sk)

    # Perf
    throughput = extract(rows, "perf", "throughput")
    mfu        = extract(rows, "perf", "mfu_actor")
    mem_alloc  = extract(rows, "perf", "max_memory_allocated_gb")

    # ── 布局：4 行 × 4 列（Row 0 用全部 4 列；其余行用前 3 列 3 格）──
    fig = plt.figure(figsize=(22, 22))
    fig.suptitle("Multi-Task Mixed Training — Qwen3-VL", fontsize=14, fontweight="bold", y=0.998)
    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.52, wspace=0.35)

    # ── Row 0: Reward（4 列全用）──
    # col 0: temporal_seg reward (dense, 81 steps)
    ax_seg = fig.add_subplot(gs[0, 0])
    plot_dense_reward(ax_seg, steps,
                      reward_tasks["temporal_seg"],
                      name="temporal_seg",
                      color=TASK_COLORS["temporal_seg"],
                      ylabel="Reward",
                      title="Temporal-Seg Reward")

    # col 1: proxy task rewards (sparse, ~25 steps each)
    proxy_tasks = {k: v for k, v in reward_tasks.items() if k != "temporal_seg"}
    ax_proxy = fig.add_subplot(gs[0, 1])
    plot_proxy_task_reward(ax_proxy, steps, proxy_tasks,
                           ylabel="Reward",
                           title="Proxy Task Rewards  (add / delete / replace / sort)")

    # col 2: overall / accuracy / format
    ax_rew_overall = fig.add_subplot(gs[0, 2])
    plot_lines(ax_rew_overall, steps, {
        "overall": reward_overall,
        "accuracy": reward_accuracy,
        "format": reward_format,
    }, ylabel="Reward", title="Overall Reward")

    # col 3: advantage distribution
    ax_adv = fig.add_subplot(gs[0, 3])
    ax_adv.plot(steps, adv_mean, color="#2c3e50", label="mean")
    ax_adv.fill_between(steps, adv_min, adv_max, alpha=0.12, color="#2c3e50", label="min/max range")
    ax_adv.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax_adv.set_ylabel("Advantage", fontsize=9)
    ax_adv.set_title("Advantage Distribution", fontsize=10, fontweight="bold")
    ax_adv.legend(fontsize=7)

    # ── Row 1: Loss / KL ──
    ax_kl = fig.add_subplot(gs[1, 0])
    plot_lines(ax_kl, steps, {
        "kl_loss": kl_loss,
        "ppo_kl":  ppo_kl,
    }, ylabel="KL", title="KL Divergence (kl_loss / ppo_kl)")

    ax_pg = fig.add_subplot(gs[1, 1])
    ax_pg2 = ax_pg.twinx()
    ax_pg.plot(steps, ema_smooth(pg_loss), color="#e74c3c", label="pg_loss")
    ax_pg2.plot(steps, ema_smooth(entropy), color="#3498db", label="entropy", linestyle="--")
    ax_pg.set_ylabel("PG Loss", fontsize=9, color="#e74c3c")
    ax_pg2.set_ylabel("Entropy", fontsize=9, color="#3498db")
    ax_pg.set_title("PG Loss & Entropy", fontsize=10, fontweight="bold")
    lines1, labels1 = ax_pg.get_legend_handles_labels()
    lines2, labels2 = ax_pg2.get_legend_handles_labels()
    ax_pg.legend(lines1 + lines2, labels1 + labels2, fontsize=7)

    ax_clip = fig.add_subplot(gs[1, 2])
    plot_lines(ax_clip, steps, {
        "clip_higher": clip_hi,
        "clip_lower":  clip_lo,
    }, ylabel="Clip Fraction", title="PG Clip Fraction (higher = more aggressive update)")

    # col 3 row 1: empty (visual breathing room) – skip

    # ── Row 2: 长度 / Grad / LR ──
    ax_len = fig.add_subplot(gs[2, 0])
    ax_len.plot(steps, ema_smooth(resp_mean), color="#3498db", label="resp mean")
    ax_len.plot(steps, ema_smooth(resp_max),  color="#3498db", alpha=0.35, linestyle="--", label="resp max")
    ax_len.plot(steps, ema_smooth(prom_mean), color="#e74c3c", label="prompt mean")
    ax_len2 = ax_len.twinx()
    ax_len2.plot(steps, resp_clip, color="#2ecc71", linewidth=1.2, label="resp clip%", alpha=0.6)
    ax_len2.set_ylabel("Clip Ratio", fontsize=9, color="#2ecc71")
    ax_len.set_ylabel("Tokens", fontsize=9)
    ax_len.set_title("Response / Prompt Length", fontsize=10, fontweight="bold")
    lines1, labels1 = ax_len.get_legend_handles_labels()
    lines2, labels2 = ax_len2.get_legend_handles_labels()
    ax_len.legend(lines1 + lines2, labels1 + labels2, fontsize=7)

    ax_grad = fig.add_subplot(gs[2, 1])
    ax_grad.plot(steps, grad_norm, color="#e74c3c", alpha=0.4)
    ax_grad.plot(steps, ema_smooth(grad_norm), color="#e74c3c", linewidth=2.0, label="grad_norm")
    ax_grad.set_ylabel("Grad Norm", fontsize=9)
    ax_grad.set_title("Gradient Norm (with spike detection)", fontsize=10, fontweight="bold")
    # mark spikes beyond 2σ
    mu, sigma = np.nanmean(grad_norm), np.nanstd(grad_norm)
    spike_mask = grad_norm > mu + 2 * sigma
    if spike_mask.any():
        ax_grad.scatter(steps[spike_mask], grad_norm[spike_mask],
                        color="red", s=30, zorder=5, label=f"spike (>{mu+2*sigma:.1f})")
    ax_grad.axhline(mu + 2 * sigma, color="red", linestyle=":", linewidth=0.8, alpha=0.5)
    ax_grad.legend(fontsize=7)

    ax_lr = fig.add_subplot(gs[2, 2])
    ax_lr.plot(steps, lr, color="#9b59b6")
    ax_lr.set_ylabel("Learning Rate", fontsize=9)
    ax_lr.set_title("Learning Rate Schedule", fontsize=10, fontweight="bold")
    ax_lr.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    # ── Row 3: Timing / 吞吐 / 显存 ──
    ax_timing = fig.add_subplot(gs[3, 0])
    fill_timing(ax_timing, steps, timing_stages)

    ax_thru = fig.add_subplot(gs[3, 1])
    ax_thru.plot(steps, ema_smooth(throughput), color="#3498db", label="throughput (tok/s)")
    ax_thru.set_ylabel("Tokens/s", fontsize=9)
    ax_thru.set_title("Training Throughput", fontsize=10, fontweight="bold")
    ax_thru2 = ax_thru.twinx()
    ax_thru2.plot(steps, ema_smooth(mfu), color="#f39c12", linestyle="--", label="MFU", linewidth=1.4)
    ax_thru2.set_ylabel("MFU", fontsize=9, color="#f39c12")
    lines1, labels1 = ax_thru.get_legend_handles_labels()
    lines2, labels2 = ax_thru2.get_legend_handles_labels()
    ax_thru.legend(lines1 + lines2, labels1 + labels2, fontsize=7)

    ax_mem = fig.add_subplot(gs[3, 2])
    ax_mem.plot(steps, mem_alloc, color="#e74c3c", label="allocated (GB)")
    ax_mem.fill_between(steps, 0, mem_alloc, alpha=0.15, color="#e74c3c")
    ax_mem.set_ylabel("GB", fontsize=9)
    ax_mem.set_title("GPU Memory (allocated)", fontsize=10, fontweight="bold")
    ax_mem.legend(fontsize=7)

    # ── X 轴统一标签 ──
    for ax in fig.get_axes():
        if not hasattr(ax, "_is_twin"):
            ax.set_xlabel("Training Step", fontsize=8)

    return fig


# ─── 入口 ────────────────────────────────────────

def main():
    global SMOOTH_COEF
    parser = argparse.ArgumentParser(description="Training log visualization")
    parser.add_argument("-i", "--input", required=True, help="path to experiment_log.jsonl")
    parser.add_argument("-o", "--output", default=None,
                        help="output image path (default: training_curves.png next to input)")
    parser.add_argument("--show", action="store_true", help="show interactive preview")
    parser.add_argument("--smooth", type=float, default=SMOOTH_COEF,
                        help=(
                            f"TensorBoard-style smoothing coefficient in [0, 1). "
                            f"Default {SMOOTH_COEF}. 0 = no smoothing, closer to 1 = smoother"
                        ))
    args = parser.parse_args()

    if not (0.0 <= args.smooth < 1.0):
        raise ValueError(f"--smooth must be in [0, 1), got {args.smooth}")

    SMOOTH_COEF = args.smooth

    rows = load_log(args.input)
    print(f"Loaded {len(rows)} steps, step {rows[0]['step']} → {rows[-1]['step']}")

    fig = build_figure(rows)

    # output
    if args.output is None:
        args.output = str(Path(args.input).parent / "training_curves.png")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", dpi=150)
    print(f"Saved → {args.output}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
