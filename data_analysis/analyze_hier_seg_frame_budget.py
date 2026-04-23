#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("XDG_CACHE_HOME", str(Path("/tmp") / "codex-xdg-cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "codex-matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


REPO_ROOT = Path(__file__).resolve().parents[1]
PROXY_DATA_DIR = REPO_ROOT / "proxy_data"
if str(PROXY_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(PROXY_DATA_DIR))

from shared.seg_source import L3_MAX_CLIP_SEC, L3_PADDING, compute_l3_clip, load_annotations


DEFAULT_BUDGETS = [48, 64, 96, 128, 256]
DEFAULT_THRESHOLDS = [2.0, 4.0, 8.0]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linestyle": "--",
        "figure.dpi": 160,
        "savefig.bbox": "tight",
    }
)


@dataclass
class LevelBundle:
    key: str
    label: str
    color: str
    base_fps: float
    input_title: str
    input_xlabel: str
    unit_title: str
    unit_xlabel: str
    count_title: str
    count_xlabel: str
    input_durations: list[float] = field(default_factory=list)
    unit_durations: list[float] = field(default_factory=list)
    units_per_input: list[int] = field(default_factory=list)
    unit_to_input_ratios: list[float] = field(default_factory=list)
    unit_durations_by_input: list[list[float]] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze hierarchical segmentation annotation duration distributions and "
            "estimate frame-budget pressure for L1/L2/L3 training inputs."
        )
    )
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        required=True,
        help="Directory containing hier-seg annotation JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_analysis/outputs/hier_seg_frame_budget"),
        help="Directory where plots and summaries will be written.",
    )
    parser.add_argument(
        "--complete-only",
        action="store_true",
        help="Use the same complete-only filtering as build_hier_data.py.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of annotation JSONs to load. 0 means no limit.",
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=int,
        default=DEFAULT_BUDGETS,
        help="Frame budgets to compare, e.g. 48 64 96 128 256.",
    )
    parser.add_argument(
        "--frame-thresholds",
        nargs="+",
        type=float,
        default=DEFAULT_THRESHOLDS,
        help="GT-unit frame thresholds used in summaries and heatmaps.",
    )
    parser.add_argument(
        "--focus-threshold",
        type=float,
        default=4.0,
        help="Threshold shown in the overview line chart (pct GT units below N frames).",
    )
    parser.add_argument(
        "--l1-fps",
        type=float,
        default=1.0,
        help="Effective training fps for L1 inputs.",
    )
    parser.add_argument(
        "--l2-fps",
        type=float,
        default=2.0,
        help="Effective training fps for L2 inputs.",
    )
    parser.add_argument(
        "--l3-fps",
        type=float,
        default=2.0,
        help="Effective training fps for L3 inputs.",
    )
    parser.add_argument(
        "--l1-min-phases",
        type=int,
        default=1,
        help="Minimum number of valid phases required for an L1 sample.",
    )
    parser.add_argument(
        "--l1-max-phases",
        type=int,
        default=999,
        help="Maximum number of valid phases allowed for an L1 sample.",
    )
    parser.add_argument(
        "--l2-min-events",
        type=int,
        default=2,
        help="Minimum number of valid events per phase for an L2 sample.",
    )
    parser.add_argument(
        "--l2-max-events",
        type=int,
        default=999,
        help="Maximum number of valid events per phase for an L2 sample.",
    )
    parser.add_argument(
        "--l3-min-actions",
        type=int,
        default=3,
        help="Minimum number of valid actions per event for an L3 sample.",
    )
    parser.add_argument(
        "--l3-max-actions",
        type=int,
        default=999,
        help="Maximum number of valid actions per event for an L3 sample.",
    )
    parser.add_argument(
        "--l3-padding",
        type=int,
        default=L3_PADDING,
        help="Padding used to build L3 clips, in seconds.",
    )
    parser.add_argument(
        "--l3-max-clip-sec",
        type=int,
        default=L3_MAX_CLIP_SEC,
        help="Maximum allowed L3 clip duration, in seconds.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output plot DPI.",
    )
    return parser.parse_args()


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _sorted_unique_ints(values: list[int]) -> list[int]:
    cleaned = sorted({int(v) for v in values if int(v) > 0})
    if not cleaned:
        raise ValueError("At least one positive frame budget is required.")
    return cleaned


def _sorted_unique_floats(values: list[float]) -> list[float]:
    cleaned = sorted({float(v) for v in values if float(v) > 0})
    if not cleaned:
        raise ValueError("At least one positive frame threshold is required.")
    return cleaned


def _as_array(values: list[float]) -> np.ndarray:
    if not values:
        return np.asarray([], dtype=float)
    return np.asarray(values, dtype=float)


def _summary_stats(values: list[float]) -> dict[str, float | int | None]:
    arr = _as_array(values)
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p10": None,
            "p25": None,
            "p75": None,
            "p90": None,
            "max": None,
        }
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
    }


def _fmt_stat(value: float | int | None, digits: int = 1) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:.{digits}f}"


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _valid_spans(items: list[dict[str, Any]], start_key: str, end_key: str) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        start = _to_float(item.get(start_key))
        end = _to_float(item.get(end_key))
        if start is None or end is None:
            continue
        start_i = int(start)
        end_i = int(end)
        if end_i <= start_i:
            continue
        payload = dict(item)
        payload["_index"] = idx
        payload["_start"] = start_i
        payload["_end"] = end_i
        payload["_duration"] = end_i - start_i
        spans.append(payload)
    return spans


def _phase_id(phase: dict[str, Any]) -> int:
    phase_id = phase.get("phase_id")
    if isinstance(phase_id, int):
        return phase_id
    return int(phase["_index"]) + 1


def _find_parent_phase(
    event: dict[str, Any],
    phases: list[dict[str, Any]],
    phase_by_id: dict[int, dict[str, Any]],
) -> dict[str, Any] | None:
    parent_phase_id = event.get("parent_phase_id")
    if isinstance(parent_phase_id, int) and parent_phase_id in phase_by_id:
        return phase_by_id[parent_phase_id]

    event_start = event["_start"]
    event_end = event["_end"]
    midpoint = 0.5 * (event_start + event_end)
    best_phase = None
    best_overlap = -1
    best_midpoint_distance = float("inf")
    for phase in phases:
        phase_start = phase["_start"]
        phase_end = phase["_end"]
        overlap = max(0, min(event_end, phase_end) - max(event_start, phase_start))
        midpoint_distance = 0.0
        if midpoint < phase_start:
            midpoint_distance = phase_start - midpoint
        elif midpoint > phase_end:
            midpoint_distance = midpoint - phase_end
        if overlap > best_overlap:
            best_phase = phase
            best_overlap = overlap
            best_midpoint_distance = midpoint_distance
            continue
        if overlap == best_overlap and midpoint_distance < best_midpoint_distance:
            best_phase = phase
            best_midpoint_distance = midpoint_distance
    if best_overlap <= 0:
        return None
    return best_phase


def collect_level_bundles(args: argparse.Namespace) -> tuple[dict[str, LevelBundle], dict[str, Any]]:
    bundles = {
        "L1": LevelBundle(
            key="L1",
            label="L1",
            color="#2563EB",
            base_fps=args.l1_fps,
            input_title="L1 input duration",
            input_xlabel="Full-video input duration (s)",
            unit_title="L1 GT phase duration",
            unit_xlabel="Phase duration (s)",
            count_title="Phases per video",
            count_xlabel="Valid phases / video",
        ),
        "L2": LevelBundle(
            key="L2",
            label="L2",
            color="#EA580C",
            base_fps=args.l2_fps,
            input_title="L2 input duration",
            input_xlabel="Per-phase input duration (s)",
            unit_title="L2 GT event duration",
            unit_xlabel="Event duration (s)",
            count_title="Events per phase",
            count_xlabel="Valid events / phase",
        ),
        "L3": LevelBundle(
            key="L3",
            label="L3",
            color="#059669",
            base_fps=args.l3_fps,
            input_title="L3 input duration",
            input_xlabel="Padded event-clip duration (s)",
            unit_title="L3 GT action duration",
            unit_xlabel="Action duration (s)",
            count_title="Actions per event",
            count_xlabel="Valid actions / event",
        ),
    }

    annotations = load_annotations(
        args.annotation_dir,
        complete_only=args.complete_only,
        limit=args.limit,
    )
    meta = {
        "n_annotations": len(annotations),
        "video_duration_sec": [],
        "filters": {
            "complete_only": bool(args.complete_only),
            "limit": int(args.limit),
            "l1_min_phases": int(args.l1_min_phases),
            "l1_max_phases": int(args.l1_max_phases),
            "l2_min_events": int(args.l2_min_events),
            "l2_max_events": int(args.l2_max_events),
            "l3_min_actions": int(args.l3_min_actions),
            "l3_max_actions": int(args.l3_max_actions),
            "l3_padding": int(args.l3_padding),
            "l3_max_clip_sec": int(args.l3_max_clip_sec),
        },
    }

    for ann in annotations:
        clip_duration = _to_float(ann.get("clip_duration_sec"))
        if clip_duration is None or clip_duration <= 0:
            continue
        clip_duration_i = int(clip_duration)
        meta["video_duration_sec"].append(clip_duration)

        phases = _valid_spans(
            (ann.get("level1") or {}).get("macro_phases", []),
            "start_time",
            "end_time",
        )
        events = _valid_spans(
            (ann.get("level2") or {}).get("events", []),
            "start_time",
            "end_time",
        )

        valid_phase_durations = [float(phase["_duration"]) for phase in phases]
        if args.l1_min_phases <= len(valid_phase_durations) <= args.l1_max_phases:
            l1 = bundles["L1"]
            l1.input_durations.append(float(clip_duration_i))
            l1.units_per_input.append(len(valid_phase_durations))
            l1.unit_durations.extend(valid_phase_durations)
            l1.unit_durations_by_input.append(list(valid_phase_durations))
            l1.unit_to_input_ratios.extend(
                duration / clip_duration_i for duration in valid_phase_durations if clip_duration_i > 0
            )

        phase_by_id = {_phase_id(phase): phase for phase in phases}
        events_by_phase: dict[int, list[float]] = defaultdict(list)
        for event in events:
            phase = _find_parent_phase(event, phases, phase_by_id)
            if phase is None:
                continue
            phase_start = phase["_start"]
            phase_end = phase["_end"]
            clipped_start = max(event["_start"], phase_start)
            clipped_end = min(event["_end"], phase_end)
            if clipped_end <= clipped_start:
                continue
            events_by_phase[_phase_id(phase)].append(float(clipped_end - clipped_start))

        l2 = bundles["L2"]
        for phase in phases:
            phase_duration = float(phase["_duration"])
            event_durations = events_by_phase.get(_phase_id(phase), [])
            if not (args.l2_min_events <= len(event_durations) <= args.l2_max_events):
                continue
            l2.input_durations.append(phase_duration)
            l2.units_per_input.append(len(event_durations))
            l2.unit_durations.extend(event_durations)
            l2.unit_durations_by_input.append(list(event_durations))
            if phase_duration > 0:
                l2.unit_to_input_ratios.extend(duration / phase_duration for duration in event_durations)

        grounding_results = (ann.get("level3") or {}).get("grounding_results", [])
        l3 = bundles["L3"]
        for grounding in grounding_results:
            if not isinstance(grounding, dict):
                continue
            event_start = _to_float(grounding.get("event_start"))
            event_end = _to_float(grounding.get("event_end"))
            if event_start is None or event_end is None:
                continue
            event_start_i = int(event_start)
            event_end_i = int(event_end)
            if event_end_i <= event_start_i:
                continue

            sub_actions = _valid_spans(
                grounding.get("sub_actions", []),
                "start_time",
                "end_time",
            )
            action_durations = [float(action["_duration"]) for action in sub_actions]
            if not (args.l3_min_actions <= len(action_durations) <= args.l3_max_actions):
                continue

            _, _, clip_duration_l3 = compute_l3_clip(
                event_start_i,
                event_end_i,
                clip_duration_i,
                padding=args.l3_padding,
                max_clip=args.l3_max_clip_sec,
            )
            clip_duration_l3 = float(clip_duration_l3)
            if clip_duration_l3 <= 0:
                continue
            l3.input_durations.append(clip_duration_l3)
            l3.units_per_input.append(len(action_durations))
            l3.unit_durations.extend(action_durations)
            l3.unit_durations_by_input.append(list(action_durations))
            l3.unit_to_input_ratios.extend(
                duration / clip_duration_l3 for duration in action_durations if clip_duration_l3 > 0
            )

    return bundles, meta


def build_budget_rows(
    bundles: dict[str, LevelBundle],
    budgets: list[int],
    thresholds: list[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for level_key, bundle in bundles.items():
        for budget in budgets:
            effective_fps: list[float] = []
            downsample_flags: list[float] = []
            gt_unit_frames: list[float] = []
            for input_duration, unit_durations in zip(bundle.input_durations, bundle.unit_durations_by_input):
                if input_duration <= 0:
                    continue
                sample_fps = min(bundle.base_fps, budget / input_duration)
                effective_fps.append(sample_fps)
                downsample_flags.append(1.0 if (input_duration * bundle.base_fps) > budget else 0.0)
                for unit_duration in unit_durations:
                    gt_unit_frames.append(unit_duration * sample_fps)

            eff_arr = _as_array(effective_fps)
            unit_arr = _as_array(gt_unit_frames)
            row: dict[str, Any] = {
                "level": level_key,
                "label": bundle.label,
                "base_fps": bundle.base_fps,
                "budget_frames": budget,
                "n_inputs": len(bundle.input_durations),
                "n_gt_units": int(unit_arr.size),
                "pct_inputs_downsampled": float(100.0 * np.mean(downsample_flags)) if downsample_flags else None,
                "median_effective_fps": float(np.median(eff_arr)) if eff_arr.size else None,
                "p10_effective_fps": float(np.percentile(eff_arr, 10)) if eff_arr.size else None,
                "median_seconds_per_frame": float(np.median(1.0 / eff_arr)) if eff_arr.size else None,
                "median_gt_unit_frames": float(np.median(unit_arr)) if unit_arr.size else None,
                "p10_gt_unit_frames": float(np.percentile(unit_arr, 10)) if unit_arr.size else None,
                "p25_gt_unit_frames": float(np.percentile(unit_arr, 25)) if unit_arr.size else None,
                "p75_gt_unit_frames": float(np.percentile(unit_arr, 75)) if unit_arr.size else None,
            }
            for threshold in thresholds:
                key = f"pct_gt_units_lt_{str(threshold).replace('.', '_')}_frames"
                row[key] = float(100.0 * np.mean(unit_arr < threshold)) if unit_arr.size else None
            rows.append(row)
    return rows


def _plot_hist_panel(
    ax: plt.Axes,
    data: list[float],
    color: str,
    title: str,
    xlabel: str,
    discrete: bool = False,
) -> None:
    arr = _as_array(data)
    if arr.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        return

    if discrete:
        min_v = int(arr.min())
        max_v = int(arr.max())
        bins = np.arange(min_v - 0.5, max_v + 1.5, 1.0)
    else:
        x_hi = float(np.percentile(arr, 99)) if arr.size > 8 else float(arr.max())
        bins = np.linspace(0, max(x_hi, 1.0), 28)

    ax.hist(arr, bins=bins, color=color, alpha=0.82, edgecolor="white", linewidth=0.8)

    median = float(np.median(arr))
    p90 = float(np.percentile(arr, 90))
    ax.axvline(median, color="#111827", linewidth=1.7, linestyle="-", label=f"median {median:.1f}")
    ax.axvline(p90, color="#374151", linewidth=1.5, linestyle="--", label=f"p90 {p90:.1f}")
    stats_text = "\n".join(
        [
            f"n={arr.size:,}",
            f"mean={arr.mean():.1f}",
            f"median={median:.1f}",
            f"p90={p90:.1f}",
            f"max={arr.max():.1f}",
        ]
    )
    ax.text(
        0.97,
        0.97,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": color, "alpha": 0.92},
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)


def plot_duration_overview(
    bundles: dict[str, LevelBundle],
    output_path: Path,
    dpi: int,
    n_annotations: int,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle(
        f"Hier-Seg Duration Overview ({n_annotations:,} annotations)",
        fontsize=15,
        fontweight="bold",
    )

    ordered = [bundles["L1"], bundles["L2"], bundles["L3"]]
    for idx, bundle in enumerate(ordered):
        _plot_hist_panel(
            axes[0, idx],
            bundle.input_durations,
            bundle.color,
            bundle.input_title,
            bundle.input_xlabel,
        )
        _plot_hist_panel(
            axes[1, idx],
            bundle.unit_durations,
            bundle.color,
            bundle.unit_title,
            bundle.unit_xlabel,
        )

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_count_overview(
    bundles: dict[str, LevelBundle],
    output_path: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    fig.suptitle("Hier-Seg Structure Counts", fontsize=15, fontweight="bold")

    ordered = [bundles["L1"], bundles["L2"], bundles["L3"]]
    for ax, bundle in zip(axes, ordered):
        _plot_hist_panel(
            ax,
            [float(v) for v in bundle.units_per_input],
            bundle.color,
            bundle.count_title,
            bundle.count_xlabel,
            discrete=True,
        )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_budget_overview(
    bundles: dict[str, LevelBundle],
    budget_rows: list[dict[str, Any]],
    budgets: list[int],
    focus_threshold: float,
    output_path: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    ordered = [bundles["L1"], bundles["L2"], bundles["L3"]]
    focus_key = f"pct_gt_units_lt_{str(focus_threshold).replace('.', '_')}_frames"

    for ax, bundle in zip(axes, ordered):
        rows = [row for row in budget_rows if row["level"] == bundle.key]
        downsample = [row["pct_inputs_downsampled"] or 0.0 for row in rows]
        median_frames = [row["median_gt_unit_frames"] or 0.0 for row in rows]
        under_focus = [row.get(focus_key) or 0.0 for row in rows]

        ax2 = ax.twinx()
        ax.plot(budgets, downsample, color=bundle.color, marker="o", linewidth=2.1, label="% inputs downsampled")
        ax.plot(
            budgets,
            under_focus,
            color="#111827",
            marker="s",
            linewidth=1.9,
            linestyle="--",
            label=f"% GT units < {focus_threshold:g}f",
        )
        ax2.plot(
            budgets,
            median_frames,
            color="#9A3412",
            marker="D",
            linewidth=1.8,
            alpha=0.9,
            label="median GT-unit frames",
        )

        ax.set_title(f"{bundle.label} budget trade-off")
        ax.set_xlabel("Frame budget")
        ax.set_ylabel("Percent (%)")
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        ax2.set_ylabel("Median GT-unit frames")
        ax.set_xticks(budgets)

        lines = ax.get_lines() + ax2.get_lines()
        labels = [line.get_label() for line in lines]
        ax.legend(lines, labels, fontsize=8, loc="upper right", framealpha=0.85)

    fig.suptitle("Frame Budget Overview", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_threshold_heatmaps(
    bundles: dict[str, LevelBundle],
    budget_rows: list[dict[str, Any]],
    budgets: list[int],
    thresholds: list[float],
    output_path: Path,
    dpi: int,
) -> None:
    ordered = [bundles["L1"], bundles["L2"], bundles["L3"]]
    fig, axes = plt.subplots(1, len(thresholds), figsize=(5.2 * len(thresholds), 5.2), squeeze=False)
    axes_list = axes[0]

    for ax, threshold in zip(axes_list, thresholds):
        key = f"pct_gt_units_lt_{str(threshold).replace('.', '_')}_frames"
        matrix = np.zeros((len(ordered), len(budgets)), dtype=float)
        for row_idx, bundle in enumerate(ordered):
            rows = [row for row in budget_rows if row["level"] == bundle.key]
            for col_idx, budget in enumerate(budgets):
                match = next((row for row in rows if row["budget_frames"] == budget), None)
                matrix[row_idx, col_idx] = float(match.get(key) or 0.0) if match else 0.0

        im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=100, aspect="auto")
        ax.set_title(f"Pct GT units < {threshold:g} frames")
        ax.set_xticks(range(len(budgets)))
        ax.set_xticklabels([str(b) for b in budgets])
        ax.set_yticks(range(len(ordered)))
        ax.set_yticklabels([bundle.label for bundle in ordered])
        ax.set_xlabel("Frame budget")
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                value = matrix[row_idx, col_idx]
                text_color = "white" if value >= 55 else "#111827"
                ax.text(col_idx, row_idx, f"{value:.1f}%", ha="center", va="center", fontsize=9, color=text_color)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, format=PercentFormatter(xmax=100))

    fig.suptitle("Frame Budget Threshold Heatmaps", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def build_duration_summary_rows(
    bundles: dict[str, LevelBundle],
    meta: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    video_stats = _summary_stats(meta["video_duration_sec"])
    rows.append({"metric": "video_input_duration_sec", **video_stats})
    for bundle in [bundles["L1"], bundles["L2"], bundles["L3"]]:
        rows.append({"metric": f"{bundle.key.lower()}_input_duration_sec", **_summary_stats(bundle.input_durations)})
        rows.append({"metric": f"{bundle.key.lower()}_gt_unit_duration_sec", **_summary_stats(bundle.unit_durations)})
        rows.append({"metric": f"{bundle.key.lower()}_units_per_input", **_summary_stats([float(v) for v in bundle.units_per_input])})
        rows.append({"metric": f"{bundle.key.lower()}_gt_unit_to_input_ratio", **_summary_stats(bundle.unit_to_input_ratios)})
    return rows


def build_summary_payload(
    bundles: dict[str, LevelBundle],
    meta: dict[str, Any],
    budget_rows: list[dict[str, Any]],
    budgets: list[int],
    thresholds: list[float],
) -> dict[str, Any]:
    per_level = {}
    for bundle in bundles.values():
        per_level[bundle.key] = {
            "base_fps": bundle.base_fps,
            "input_duration_sec": _summary_stats(bundle.input_durations),
            "gt_unit_duration_sec": _summary_stats(bundle.unit_durations),
            "units_per_input": _summary_stats([float(v) for v in bundle.units_per_input]),
            "gt_unit_to_input_ratio": _summary_stats(bundle.unit_to_input_ratios),
            "budgets": [row for row in budget_rows if row["level"] == bundle.key],
        }
    return {
        "annotation_count": meta["n_annotations"],
        "filters": meta["filters"],
        "budgets": budgets,
        "frame_thresholds": thresholds,
        "video_input_duration_sec": _summary_stats(meta["video_duration_sec"]),
        "levels": per_level,
    }


def write_markdown_summary(
    path: Path,
    bundles: dict[str, LevelBundle],
    meta: dict[str, Any],
    budget_rows: list[dict[str, Any]],
    budgets: list[int],
    thresholds: list[float],
) -> None:
    focus_budgets = [budget for budget in budgets if budget in {48, 64, 128, 256}]
    if not focus_budgets:
        focus_budgets = budgets[: min(4, len(budgets))]

    lines: list[str] = []
    lines.append("# Hier-Seg Frame Budget Analysis")
    lines.append("")
    lines.append(f"- Annotation files loaded: **{meta['n_annotations']:,}**")
    lines.append(f"- Complete-only: **{meta['filters']['complete_only']}**")
    lines.append(
        "- Base fps: "
        + ", ".join(f"**{bundle.key}={bundle.base_fps:g}fps**" for bundle in bundles.values())
    )
    lines.append(f"- Compared frame budgets: **{', '.join(str(v) for v in budgets)}**")
    lines.append("")
    lines.append("## Duration Summary")
    lines.append("")
    lines.append("| Metric | n | mean | median | p90 | max |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    duration_rows = build_duration_summary_rows(bundles, meta)
    for row in duration_rows:
        lines.append(
            f"| {row['metric']} | {_fmt_stat(row['count'], 0)} | {_fmt_stat(row['mean'])} | "
            f"{_fmt_stat(row['median'])} | {_fmt_stat(row['p90'])} | {_fmt_stat(row['max'])} |"
        )

    for bundle in [bundles["L1"], bundles["L2"], bundles["L3"]]:
        lines.append("")
        lines.append(f"## {bundle.key} Budget View")
        lines.append("")
        lines.append(
            "| budget | % inputs downsampled | median GT-unit frames | p10 GT-unit frames | "
            + " | ".join(f"% GT < {threshold:g}f" for threshold in thresholds)
            + " |"
        )
        lines.append(
            "|---:|---:|---:|---:|"
            + "".join("---:|" for _ in thresholds)
        )
        for row in [r for r in budget_rows if r["level"] == bundle.key and r["budget_frames"] in focus_budgets]:
            threshold_cells = []
            for threshold in thresholds:
                threshold_key = f"pct_gt_units_lt_{str(threshold).replace('.', '_')}_frames"
                threshold_cells.append(f"| {_fmt_stat(row.get(threshold_key))}")
            threshold_cells_text = " ".join(threshold_cells)
            lines.append(
                f"| {row['budget_frames']} | {_fmt_stat(row['pct_inputs_downsampled'])} | "
                f"{_fmt_stat(row['median_gt_unit_frames'])} | {_fmt_stat(row['p10_gt_unit_frames'])} "
                f"{threshold_cells_text} |"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.budgets = _sorted_unique_ints(args.budgets)
    args.frame_thresholds = _sorted_unique_floats(args.frame_thresholds)
    if args.focus_threshold not in args.frame_thresholds:
        args.frame_thresholds = _sorted_unique_floats(args.frame_thresholds + [args.focus_threshold])

    if not args.annotation_dir.exists():
        raise SystemExit(f"Annotation directory not found: {args.annotation_dir}")

    bundles, meta = collect_level_bundles(args)
    budget_rows = build_budget_rows(bundles, args.budgets, args.frame_thresholds)
    summary_payload = build_summary_payload(
        bundles,
        meta,
        budget_rows,
        args.budgets,
        args.frame_thresholds,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_duration_overview(
        bundles,
        args.output_dir / "duration_overview.png",
        args.dpi,
        meta["n_annotations"],
    )
    plot_count_overview(
        bundles,
        args.output_dir / "count_overview.png",
        args.dpi,
    )
    plot_budget_overview(
        bundles,
        budget_rows,
        args.budgets,
        args.focus_threshold,
        args.output_dir / "frame_budget_overview.png",
        args.dpi,
    )
    plot_threshold_heatmaps(
        bundles,
        budget_rows,
        args.budgets,
        args.frame_thresholds,
        args.output_dir / "frame_budget_threshold_heatmaps.png",
        args.dpi,
    )

    duration_rows = build_duration_summary_rows(bundles, meta)
    _write_csv(
        args.output_dir / "duration_summary.csv",
        ["metric", "count", "mean", "median", "p10", "p25", "p75", "p90", "max"],
        duration_rows,
    )

    budget_fieldnames = list(budget_rows[0].keys()) if budget_rows else ["level", "budget_frames"]
    _write_csv(args.output_dir / "budget_summary.csv", budget_fieldnames, budget_rows)

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, ensure_ascii=False, indent=2)

    write_markdown_summary(
        args.output_dir / "README.md",
        bundles,
        meta,
        budget_rows,
        args.budgets,
        args.frame_thresholds,
    )

    print(f"Saved analysis to: {args.output_dir}")
    print("Generated files:")
    for name in [
        "duration_overview.png",
        "count_overview.png",
        "frame_budget_overview.png",
        "frame_budget_threshold_heatmaps.png",
        "duration_summary.csv",
        "budget_summary.csv",
        "summary.json",
        "README.md",
    ]:
        print(f"  - {args.output_dir / name}")


if __name__ == "__main__":
    main()
