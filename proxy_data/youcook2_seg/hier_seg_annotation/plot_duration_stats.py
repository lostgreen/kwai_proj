#!/usr/bin/env python3
"""
plot_duration_stats.py — 绘制 hier-seg 训练数据的输入时长分布图。

读取 build_v1_data.sh 生成的各层 train.jsonl，从 metadata.clip_duration_sec
(L1/L2) 和 metadata.clip_end_sec - metadata.clip_start_sec (L3) 提取输入视频时长，
绘制三层堆叠直方图。

用法:
    python plot_duration_stats.py --output-dir /path/to/train_dir \
        --save-path /path/to/duration_distribution.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无 display 环境
import matplotlib.pyplot as plt
import numpy as np


def _load_durations(jsonl_path: Path, level: str) -> list[float]:
    """从 JSONL 提取每条记录的 *输入视频* 时长 (秒)。"""
    if not jsonl_path.exists():
        return []
    durations = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            meta = rec.get("metadata", {})
            if level in ("L1", "L2"):
                # 全视频输入 → clip_duration_sec
                d = meta.get("clip_duration_sec")
            else:
                # L3: event clip 时长 = clip_end - clip_start
                cs = meta.get("clip_start_sec")
                ce = meta.get("clip_end_sec")
                if cs is not None and ce is not None:
                    d = ce - cs
                else:
                    # fallback: event bounds
                    es = meta.get("event_start_sec")
                    ee = meta.get("event_end_sec")
                    d = (ee - es) if (es is not None and ee is not None) else None
            if d is not None and d > 0:
                durations.append(float(d))
    return durations


def main():
    parser = argparse.ArgumentParser(
        description="Plot input video duration distributions for hier-seg train data"
    )
    parser.add_argument("--output-dir", required=True,
                        help="build_v1_data.sh 生成的 train/ 目录")
    parser.add_argument("--suffix", default="",
                        help="文件后缀 (如 '_hint'，默认为空)")
    parser.add_argument("--save-path", default="",
                        help="输出图片路径 (默认: <output-dir>/duration_distribution.png)")
    parser.add_argument("--split", choices=["train", "val", "both"], default="train",
                        help="统计 train / val / both (默认: train)")
    parser.add_argument("--bins", type=int, default=40,
                        help="直方图 bin 数 (默认: 40)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    suffix = args.suffix
    save_path = Path(args.save_path) if args.save_path else out_dir / f"duration_distribution{suffix}.png"

    LEVEL_COLORS = {
        "L1":     ("#4C9BE8", "L1 (full video, phases)"),
        "L2":     ("#F5A623", "L2 (full video, events)"),
        "L3_seg": ("#7ED321", "L3 (event clip, actions)"),
    }

    splits = ["train", "val"] if args.split == "both" else [args.split]
    n_plots = len(splits)
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5), squeeze=False)

    for ax, split in zip(axes[0], splits):
        all_max = 0
        level_data: dict[str, list[float]] = {}
        for level in ("L1", "L2", "L3_seg"):
            jsonl = out_dir / level / f"{split}{suffix}.jsonl"
            durs = _load_durations(jsonl, level)
            level_data[level] = durs
            if durs:
                all_max = max(all_max, max(durs))

        if all_max == 0:
            ax.text(0.5, 0.5, f"No data ({split})", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        bins = np.linspace(0, min(all_max, 600), args.bins + 1)

        for level, (color, label) in LEVEL_COLORS.items():
            durs = level_data.get(level, [])
            if not durs:
                continue
            ax.hist(durs, bins=bins, alpha=0.6, color=color, label=label, edgecolor="none")

            # 统计信息注释
            arr = np.array(durs)
            p50, p90 = np.percentile(arr, [50, 90])

        # 每层单独文字统计
        y_top = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1
        for i, (level, (color, label)) in enumerate(LEVEL_COLORS.items()):
            durs = level_data.get(level, [])
            if not durs:
                continue
            arr = np.array(durs)
            stats_text = (
                f"{label}\n"
                f"  n={len(arr):,}  median={np.median(arr):.0f}s\n"
                f"  p90={np.percentile(arr, 90):.0f}s  max={arr.max():.0f}s"
            )
            ax.text(0.98, 0.97 - i * 0.22, stats_text,
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=8, color=color,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

        ax.set_xlabel("Input video duration (seconds)", fontsize=11)
        ax.set_ylabel("Record count", fontsize=11)
        ax.set_title(f"Input Duration Distribution — {split}", fontsize=13)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.set_xlim(0, min(all_max * 1.05, 620))

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved duration distribution plot → {save_path}")

    # 打印文字汇总
    print("\n=== Duration Statistics ===")
    for split in splits:
        print(f"\n[{split}]")
        for level in ("L1", "L2", "L3_seg"):
            durs = []
            jsonl = out_dir / level / f"{split}{suffix}.jsonl"
            durs = _load_durations(jsonl, level)
            if not durs:
                print(f"  {level}: no data")
                continue
            arr = np.array(durs)
            print(f"  {level}: n={len(arr):,}  mean={arr.mean():.1f}s  "
                  f"median={np.median(arr):.1f}s  p90={np.percentile(arr, 90):.1f}s  "
                  f"max={arr.max():.1f}s")


if __name__ == "__main__":
    main()
