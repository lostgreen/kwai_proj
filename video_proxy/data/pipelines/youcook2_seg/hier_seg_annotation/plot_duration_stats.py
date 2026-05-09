#!/usr/bin/env python3
"""
plot_duration_stats.py — 学术汇报用的输入时长分布图。

布局: 1行3列子图，每图对应一个层级 (L1 / L2 / L3)。
每图: 直方图 (密度归一化) + KDE曲线 + 中位线 + p90线 + 统计文字框。

用法:
    python plot_duration_stats.py \\
        --output-dir /path/to/train_dir \\
        --save-path duration_distribution.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.stats import gaussian_kde


# ─── 配色 (适合打印 + 投影) ───────────────────────────────────────────
PALETTE = {
    "L1":     {"color": "#2E86AB", "label": "L1 — Phase Segmentation\n(full video, 1 fps)"},
    "L2":     {"color": "#E87B30", "label": "L2 — Event Detection\n(full video, 1 fps)"},
    "L3_seg": {"color": "#3BB273", "label": "L3 — Action Segmentation\n(event clip, 2 fps)"},
}


def _load_durations(jsonl_path: Path, level: str) -> np.ndarray:
    """Extract input video duration (seconds) for each record."""
    if not jsonl_path.exists():
        return np.array([])
    durations = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            meta = rec.get("metadata", {})
            if level in ("L1", "L2"):
                d = meta.get("clip_duration_sec")
            else:
                cs = meta.get("clip_start_sec")
                ce = meta.get("clip_end_sec")
                if cs is not None and ce is not None:
                    d = ce - cs
                else:
                    es = meta.get("event_start_sec")
                    ee = meta.get("event_end_sec")
                    d = (ee - es) if (es is not None and ee is not None) else None
            if d is not None and float(d) > 0:
                durations.append(float(d))
    return np.array(durations)


def _draw_level(ax: plt.Axes, data: np.ndarray, level: str, bins: int) -> None:
    """Draw histogram + KDE + quantile annotations for one level."""
    color = PALETTE[level]["color"]
    label = PALETTE[level]["label"]

    if data.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=13, color="gray")
        ax.set_title(label, fontsize=12, fontweight="bold", pad=10)
        return

    # ── 直方图 (密度归一化) ──
    ax.hist(data, bins=bins, density=True,
            color=color, alpha=0.35, edgecolor="white", linewidth=0.4)

    # ── KDE 曲线 ──
    x_max = np.percentile(data, 99)          # 截到 p99，避免长尾压缩图形
    x = np.linspace(0, x_max * 1.05, 400)
    try:
        kde = gaussian_kde(data, bw_method="scott")
        ax.plot(x, kde(x), color=color, linewidth=2.2)
    except np.linalg.LinAlgError:
        pass  # 数据过少时 KDE 失败

    # ── 分位线 ──
    p50 = np.median(data)
    p75 = np.percentile(data, 75)
    p90 = np.percentile(data, 90)

    y_top = ax.get_ylim()[1] or 1
    ax.axvline(p50, color=color, linestyle="-",  linewidth=1.8, alpha=0.9,
               label=f"Median = {p50:.0f}s")
    ax.axvline(p90, color=color, linestyle="--", linewidth=1.5, alpha=0.85,
               label=f"P90 = {p90:.0f}s")

    # ── 统计文字框 ──
    stats_lines = [
        f"$n$ = {len(data):,}",
        f"Median = {p50:.0f} s",
        f"P75 = {p75:.0f} s",
        f"P90 = {p90:.0f} s",
        f"Max = {data.max():.0f} s",
    ]
    stats_text = "\n".join(stats_lines)
    ax.text(0.97, 0.97, stats_text,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9.5, family="monospace",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                      edgecolor=color, linewidth=1.2, alpha=0.92))

    # ── 标题 & 轴 ──
    ax.set_title(label, fontsize=11.5, fontweight="bold", pad=9, color="#222222")
    ax.set_xlabel("Input duration (seconds)", fontsize=10.5)
    ax.set_ylabel("Density", fontsize=10.5)
    ax.set_xlim(left=0, right=min(x_max * 1.1, data.max() * 1.05))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.6,
              handlelength=1.8, borderpad=0.5)

    # 去掉上/右边框
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)


def main():
    parser = argparse.ArgumentParser(
        description="Plot input video duration distributions (academic style)"
    )
    parser.add_argument("--output-dir", required=True,
                        help="build_v1_data.sh 生成的输出目录")
    parser.add_argument("--suffix", default="",
                        help="JSONL 文件后缀 (如 '_hint'，默认为空)")
    parser.add_argument("--split", default="train",
                        choices=["train", "val"],
                        help="统计哪个 split (默认: train)")
    parser.add_argument("--bins", type=int, default=40,
                        help="直方图 bin 数 (默认: 40)")
    parser.add_argument("--save-path", default="",
                        help="输出图路径 (默认: <output-dir>/duration_distribution.png)")
    parser.add_argument("--dpi", type=int, default=180,
                        help="输出分辨率 (默认: 180 dpi，学术用 300)")
    args = parser.parse_args()

    out_dir  = Path(args.output_dir)
    suffix   = args.suffix
    split    = args.split
    save_path = Path(args.save_path) if args.save_path \
        else out_dir / f"duration_distribution{suffix}.png"

    levels = ["L1", "L2", "L3_seg"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(
        f"Hier-Seg Training Data — Input Video Duration Distribution  ({split})",
        fontsize=13.5, fontweight="bold", y=1.01
    )

    for ax, level in zip(axes, levels):
        jsonl = out_dir / level / f"{split}{suffix}.jsonl"
        data = _load_durations(jsonl, level)
        _draw_level(ax, data, level, bins=args.bins)

    fig.tight_layout(pad=1.8)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved → {save_path}")

    # ── 文字汇总 ──
    print(f"\n{'Level':<10} {'n':>7}  {'Mean':>7}  {'Median':>7}  "
          f"{'P75':>7}  {'P90':>7}  {'Max':>7}")
    print("-" * 60)
    for level in levels:
        jsonl = out_dir / level / f"{split}{suffix}.jsonl"
        data  = _load_durations(jsonl, level)
        if data.size == 0:
            print(f"{level:<10} {'—':>7}")
            continue
        print(f"{level:<10} {len(data):>7,}  {data.mean():>7.1f}  "
              f"{np.median(data):>7.1f}  {np.percentile(data,75):>7.1f}  "
              f"{np.percentile(data,90):>7.1f}  {data.max():>7.1f}")


if __name__ == "__main__":
    main()
