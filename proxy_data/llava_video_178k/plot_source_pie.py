#!/usr/bin/env python3
"""
LLaVA-Video-178K: Sunburst chart — inner ring = source, outer ring = duration_bucket.

Generates two side-by-side sunburst charts:
  Left:  Reference dataset (mcq_all.jsonl)
  Right: Downsampled train (train_final_combined.jsonl)

Usage:
    python plot_source_pie.py \
        --before results/mcq_all.jsonl \
        --after  results/train_final_combined.jsonl \
        --outdir results/figures

    # Only draw after
    python plot_source_pie.py \
        --after results/train_final_combined.jsonl \
        --outdir results/figures
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# Font — Times New Roman globally
# ============================================================
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]

FONT_SIZE_INNER = 11.0
FONT_SIZE_OUTER = 9.0

# Pastel colors for sources (cycle if more)
SOURCE_PALETTE = [
    "#F4978E", "#FFD166", "#89C2F5", "#86CFA3", "#C8A4D8",
    "#F8AD9D", "#FFD97D", "#9ECEF8", "#9DDBB5", "#D4B5E0",
    "#FBC4AB", "#FFE3A0", "#AED8FA", "#B2E5C8", "#DFC6E8",
    "#E8E8E8", "#FDDBC5", "#FFECC2", "#C4E3FB", "#CCF0DB",
]


def load_jsonl(path: str) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def extract_fields(rec: dict) -> tuple[str, str]:
    meta = rec.get("metadata", {})
    return meta.get("source", "unknown"), meta.get("duration_bucket", "unknown")


def _lighten(hex_color: str, factor: float = 0.35) -> str:
    """Lighten a hex color by mixing with white."""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


def _tangential_rotation(angle_mid: float) -> float:
    a = angle_mid % 360
    rot = a - 90
    rot_norm = rot % 360
    if 90 < rot_norm < 270:
        rot += 180
    return rot


def _radial_rotation(angle_mid: float) -> float:
    a = angle_mid % 360
    if 90 < a < 270:
        return a + 180
    return a


def draw_wedge(ax, r_in, r_out, theta1, theta2, color, alpha=1.0):
    n = max(50, int(abs(theta2 - theta1)))
    angles = np.linspace(np.radians(theta1), np.radians(theta2), n)
    xs_out = r_out * np.cos(angles)
    ys_out = r_out * np.sin(angles)
    xs_in = r_in * np.cos(angles[::-1])
    ys_in = r_in * np.sin(angles[::-1])
    xs = np.concatenate([xs_out, xs_in])
    ys = np.concatenate([ys_out, ys_in])
    poly = plt.Polygon(
        np.column_stack([xs, ys]),
        facecolor=color, edgecolor="white", linewidth=1.5,
        alpha=alpha, zorder=2,
    )
    ax.add_patch(poly)


def add_text(ax, angle_mid, r_mid, text, fontsize, bold=False, radial=False):
    rad = np.radians(angle_mid)
    x = r_mid * np.cos(rad)
    y = r_mid * np.sin(rad)
    rot = _radial_rotation(angle_mid) if radial else _tangential_rotation(angle_mid)
    weight = "bold" if bold else "normal"
    ax.text(
        x, y, text,
        ha="center", va="center",
        fontsize=fontsize, fontweight=weight, fontfamily="serif",
        color="#222222", rotation=rot, zorder=5,
    )


def build_hierarchy(records: list[dict]) -> dict[str, Counter]:
    """Returns {source: Counter({duration_bucket: count})}."""
    hierarchy: dict[str, Counter] = defaultdict(Counter)
    for rec in records:
        source, bucket = extract_fields(rec)
        hierarchy[source][bucket] += 1
    return dict(hierarchy)


def draw_sunburst(
    ax,
    records: list[dict],
    title: str,
    src_order: list[str],
    src_colors: dict[str, str],
):
    """Draw a sunburst: inner = source, outer = duration_bucket.

    Args:
        src_order: Globally-consistent source order (same across both charts).
        src_colors: Globally-consistent {source: hex_color} mapping.
    """
    hierarchy = build_hierarchy(records)
    total = len(records)

    # Only draw sources that actually appear in this dataset
    active_sources = [s for s in src_order if s in hierarchy]

    # Geometry
    INNER_R = 0.40
    MID_R = 0.72
    OUTER_R = 1.25
    GAP = 1.0  # degrees between source groups

    available = 360 - len(active_sources) * GAP
    theta = 90  # start from top

    for src in active_sources:
        src_total = sum(hierarchy[src].values())
        src_span = src_total / total * available

        # Inner ring: source wedge
        draw_wedge(ax, INNER_R, MID_R, theta, theta - src_span, src_colors[src])

        # Label inner ring (only if span large enough)
        if src_span > 8:
            mid_angle = theta - src_span / 2
            label = f"{src}\n({src_total})"
            add_text(ax, mid_angle, (INNER_R + MID_R) / 2, label,
                     fontsize=FONT_SIZE_INNER, bold=True, radial=False)

        # Outer ring: duration buckets within this source
        buckets = sorted(hierarchy[src].keys())
        sub_theta = theta
        for bkt in buckets:
            bkt_count = hierarchy[src][bkt]
            bkt_span = bkt_count / total * available
            bkt_color = _lighten(src_colors[src], factor=0.3)
            draw_wedge(ax, MID_R, OUTER_R, sub_theta, sub_theta - bkt_span, bkt_color)

            # Label outer ring (only if span large enough)
            if bkt_span > 5:
                sub_mid = sub_theta - bkt_span / 2
                bkt_label = f"{bkt}\n({bkt_count})"
                add_text(ax, sub_mid, (MID_R + OUTER_R) / 2, bkt_label,
                         fontsize=FONT_SIZE_OUTER, bold=False, radial=True)

            sub_theta -= bkt_span

        theta -= src_span + GAP

    # Center circle
    center = plt.Circle((0, 0), INNER_R, color="white", ec="white", linewidth=2, zorder=3)
    ax.add_patch(center)

    # Center text
    ax.text(0, 0, f"N={total}", ha="center", va="center",
            fontsize=14, fontweight="bold", fontfamily="serif", color="#555555", zorder=6)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)


def main():
    parser = argparse.ArgumentParser(description="Sunburst: source × duration_bucket")
    parser.add_argument("--before", default="", help="Original MCQ JSONL (before filtering)")
    parser.add_argument("--after", required=True, help="Final train JSONL (after downsampling)")
    parser.add_argument("--outdir", default="results/figures")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load data
    has_before = args.before and os.path.isfile(args.before)
    before_recs = []
    if has_before:
        print(f"Loading before: {args.before}")
        before_recs = load_jsonl(args.before)
        print(f"  {len(before_recs)} records")

    print(f"Loading after: {args.after}")
    after_recs = load_jsonl(args.after)
    print(f"  {len(after_recs)} records")

    # Build global source order & color mapping from union of both datasets
    all_recs = before_recs + after_recs
    global_src_counter: Counter = Counter()
    for rec in all_recs:
        src, _ = extract_fields(rec)
        global_src_counter[src] += 1
    src_order = [s for s, _ in global_src_counter.most_common()]
    src_colors = {s: SOURCE_PALETTE[i % len(SOURCE_PALETTE)] for i, s in enumerate(src_order)}

    # Draw
    ncols = 2 if has_before else 1
    fig, axes = plt.subplots(1, ncols, figsize=(11 * ncols, 11))
    fig.patch.set_facecolor("white")
    if ncols == 1:
        axes = [axes]

    idx = 0
    if has_before:
        draw_sunburst(axes[idx], before_recs, "Reference Dataset", src_order, src_colors)
        idx += 1

    draw_sunburst(axes[idx], after_recs, "Train (Downsampled)", src_order, src_colors)

    fig.suptitle("LLaVA-Video-178K: Source × Duration", fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        path = os.path.join(args.outdir, f"source_sunburst.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
