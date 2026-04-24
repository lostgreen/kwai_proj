#!/usr/bin/env python3
"""
Analyze TimeLens temporal-grounding rollout reports.

The rollout report keeps scores and responses, while duration/source metadata
lives in the query-level rollout input. This script joins the two files and
produces query/video-level score summaries, duration summaries, threshold
sweeps, and plots for choosing an IoU cutoff.

Usage from train/:
    python proxy_data/data_curation/timelens_100k/analyze_tg_rollout.py \
        --report proxy_data/data_curation/results/timelens_100k_short/tg_rollout_qwen3_vl_8b_roll8/rollout_report.jsonl \
        --input-jsonl proxy_data/data_curation/results/timelens_100k_short/tg_rollout_qwen3_vl_8b_roll8/tg_rollout_input.jsonl \
        --output-dir proxy_data/data_curation/results/timelens_100k_short/tg_rollout_qwen3_vl_8b_roll8/analysis

    # If final merge failed, analyze shard reports directly:
    python proxy_data/data_curation/timelens_100k/analyze_tg_rollout.py \
        --report-glob 'proxy_data/data_curation/results/timelens_100k_short/tg_rollout_qwen3_vl_8b_roll8/_shard*_report.jsonl' \
        --input-jsonl proxy_data/data_curation/results/timelens_100k_short/tg_rollout_qwen3_vl_8b_roll8/tg_rollout_input.jsonl \
        --output-dir proxy_data/data_curation/results/timelens_100k_short/tg_rollout_qwen3_vl_8b_roll8/analysis
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

TMP_ROOT = Path(os.environ.get("TMPDIR", "/tmp"))
os.environ.setdefault("MPLCONFIGDIR", str(TMP_ROOT / "matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(TMP_ROOT / "xdg-cache"))

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:  # pragma: no cover - handled at runtime on cluster images
    plt = None
    np = None


DURATION_BUCKETS = ["[0,15)", "[15,30)", "[30,45)", "[45,60]"]
THRESHOLDS = [round(i / 100, 2) for i in range(0, 101, 5)]


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def mean(values: list[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    return sum(vals) / len(vals) if vals else math.nan


def std(values: list[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    if len(vals) < 2:
        return 0.0
    mu = mean(vals)
    return math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals))


def percentile(values: list[float], q: float) -> float:
    vals = sorted(v for v in values if not math.isnan(v))
    if not vals:
        return math.nan
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def fmt_float(value: float, digits: int = 4) -> str:
    if value is None or math.isnan(value):
        return ""
    return f"{value:.{digits}f}"


def duration_bucket(duration: float) -> str:
    if math.isnan(duration):
        return "unknown"
    if duration < 15:
        return "[0,15)"
    if duration < 30:
        return "[15,30)"
    if duration < 45:
        return "[30,45)"
    return "[45,60]"


def score_bin(score: float, step: float = 0.1) -> str:
    if math.isnan(score):
        return "unknown"
    lo = math.floor(min(score, 0.999999) / step) * step
    hi = min(1.0, lo + step)
    close = "]" if hi >= 1.0 else ")"
    return f"[{lo:.1f},{hi:.1f}{close}"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Failed to parse {path}:{line_no}: {exc}") from exc
    return records


def build_query_index(input_jsonl: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for rec in load_jsonl(input_jsonl):
        meta = rec.get("metadata") or {}
        query_id = str(meta.get("id") or "")
        if query_id:
            index[query_id] = rec
    return index


def resolve_report_paths(report_args: list[str], report_globs: list[str]) -> list[Path]:
    paths: list[Path] = []
    patterns = list(report_args or []) + list(report_globs or [])
    for pattern in patterns:
        if any(ch in pattern for ch in "*?["):
            paths.extend(Path(p) for p in sorted(glob.glob(pattern)))
        else:
            paths.append(Path(pattern))

    unique: dict[str, Path] = {}
    for path in paths:
        unique[str(path)] = path
    return list(unique.values())


def shard_id_from_path(path: Path) -> str:
    match = re.search(r"_shard(\d+)_report\.jsonl$", path.name)
    return match.group(1) if match else ""


def build_query_stats(
    report_paths: list[Path],
    query_index: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    stats: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    seen_query_ids: set[str] = set()

    counters["report_files"] = len(report_paths)
    for report_path in report_paths:
        shard_id = shard_id_from_path(report_path)
        for report in load_jsonl(report_path):
            counters["reports"] += 1
            query_id = str(report.get("metadata_id") or "")
            if query_id and query_id in seen_query_ids:
                counters["duplicate_query_id"] += 1
                continue
            if query_id:
                seen_query_ids.add(query_id)
            if report.get("error"):
                counters["errors"] += 1
                continue

            raw_rewards = report.get("rewards") or []
            rewards = [safe_float(v) for v in raw_rewards]
            rewards = [v for v in rewards if not math.isnan(v)]
            mean_iou = safe_float(report.get("mean_reward"), mean(rewards))
            if math.isnan(mean_iou):
                counters["missing_score"] += 1
                continue

            item = query_index.get(query_id, {})
            if not item:
                counters["missing_input_metadata"] += 1
            meta = item.get("metadata") or {}

            duration = safe_float(meta.get("duration"))
            bucket = str(meta.get("duration_bucket") or duration_bucket(duration))
            video_uid = str(
                meta.get("video_uid")
                or meta.get("video_id")
                or meta.get("clip_key")
                or query_id.split("::", 1)[0]
                or query_id
            )
            source = str(meta.get("source") or "unknown")
            query_text = str(meta.get("query") or meta.get("sentence") or "")

            stats.append(
                {
                    "query_id": query_id,
                    "video_uid": video_uid,
                    "source": source,
                    "duration": duration,
                    "duration_bucket": bucket,
                    "query_idx": meta.get("query_idx"),
                    "span_idx": meta.get("span_idx"),
                    "query": query_text,
                    "answer": report.get("answer", item.get("answer", "")),
                    "mean_iou": mean_iou,
                    "std_iou": std(rewards),
                    "min_iou": min(rewards) if rewards else mean_iou,
                    "max_iou": max(rewards) if rewards else mean_iou,
                    "num_rollouts": len(rewards),
                    "rewards": rewards,
                    "keep": bool(report.get("keep", False)),
                    "has_diversity": bool(report.get("has_diversity", False)),
                    "report_file": str(report_path),
                    "shard_id": shard_id,
                }
            )

    counters["scored_queries"] = len(stats)
    counters["input_queries"] = len(query_index)
    return stats, dict(counters)


def build_video_stats(query_stats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in query_stats:
        grouped[row["video_uid"]].append(row)

    videos: list[dict[str, Any]] = []
    for video_uid, rows in grouped.items():
        query_scores = [r["mean_iou"] for r in rows]
        rollout_scores = [v for r in rows for v in r.get("rewards", [])]
        duration_values = [r["duration"] for r in rows if not math.isnan(r["duration"])]
        duration = duration_values[0] if duration_values else math.nan
        source = rows[0]["source"] if rows else "unknown"
        bucket = rows[0]["duration_bucket"] if rows else "unknown"
        videos.append(
            {
                "video_uid": video_uid,
                "source": source,
                "duration": duration,
                "duration_bucket": bucket,
                "num_queries": len(rows),
                "mean_iou": mean(query_scores),
                "median_query_iou": percentile(query_scores, 0.5),
                "min_query_iou": min(query_scores) if query_scores else math.nan,
                "max_query_iou": max(query_scores) if query_scores else math.nan,
                "rollout_mean_iou": mean(rollout_scores),
                "rollout_std_iou": std(rollout_scores),
            }
        )
    videos.sort(key=lambda x: (x["mean_iou"], x["video_uid"]))
    return videos


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            clean = dict(row)
            for key, value in list(clean.items()):
                if isinstance(value, float):
                    clean[key] = fmt_float(value)
                elif isinstance(value, list):
                    clean[key] = json.dumps(value, ensure_ascii=False)
            writer.writerow(clean)


def summarize_scores(values: list[float]) -> dict[str, Any]:
    vals = [v for v in values if not math.isnan(v)]
    return {
        "count": len(vals),
        "mean": mean(vals),
        "std": std(vals),
        "min": min(vals) if vals else math.nan,
        "p10": percentile(vals, 0.10),
        "p25": percentile(vals, 0.25),
        "p50": percentile(vals, 0.50),
        "p75": percentile(vals, 0.75),
        "p90": percentile(vals, 0.90),
        "max": max(vals) if vals else math.nan,
    }


def group_summary(rows: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_key) or "unknown")].append(row)

    out: list[dict[str, Any]] = []
    for group, vals in grouped.items():
        scores = [v["mean_iou"] for v in vals]
        videos = {v["video_uid"] for v in vals}
        durations = [v["duration"] for v in vals if not math.isnan(v["duration"])]
        out.append(
            {
                group_key: group,
                "num_queries": len(vals),
                "num_videos": len(videos),
                "mean_iou": mean(scores),
                "median_iou": percentile(scores, 0.5),
                "p25_iou": percentile(scores, 0.25),
                "p75_iou": percentile(scores, 0.75),
                "mean_duration": mean(durations),
            }
        )
    out.sort(key=lambda x: (-x["num_queries"], str(x[group_key])))
    return out


def build_threshold_sweeps(
    query_stats: list[dict[str, Any]],
    video_stats: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    total_queries = len(query_stats)
    total_videos = len(video_stats)
    sweep: list[dict[str, Any]] = []
    by_source: list[dict[str, Any]] = []
    by_duration: list[dict[str, Any]] = []
    sources = sorted({r["source"] for r in query_stats})
    buckets = [b for b in DURATION_BUCKETS if any(r["duration_bucket"] == b for r in query_stats)]

    for threshold in THRESHOLDS:
        kept_queries = [r for r in query_stats if r["mean_iou"] >= threshold]
        kept_videos = [r for r in video_stats if r["mean_iou"] >= threshold]
        kept_query_videos = {r["video_uid"] for r in kept_queries}
        durations = [r["duration"] for r in kept_queries if not math.isnan(r["duration"])]
        sweep.append(
            {
                "threshold": threshold,
                "query_count": len(kept_queries),
                "query_pct": len(kept_queries) / total_queries if total_queries else 0.0,
                "video_count_by_query": len(kept_query_videos),
                "video_count": len(kept_videos),
                "video_pct": len(kept_videos) / total_videos if total_videos else 0.0,
                "mean_duration": mean(durations),
                "median_duration": percentile(durations, 0.5),
            }
        )

        for source in sources:
            source_queries = [r for r in kept_queries if r["source"] == source]
            by_source.append(
                {
                    "threshold": threshold,
                    "source": source,
                    "query_count": len(source_queries),
                    "video_count_by_query": len({r["video_uid"] for r in source_queries}),
                }
            )
        for bucket in buckets:
            bucket_queries = [r for r in kept_queries if r["duration_bucket"] == bucket]
            by_duration.append(
                {
                    "threshold": threshold,
                    "duration_bucket": bucket,
                    "query_count": len(bucket_queries),
                    "video_count_by_query": len({r["video_uid"] for r in bucket_queries}),
                }
            )
    return sweep, by_source, by_duration


def build_score_bin_table(query_stats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in query_stats:
        grouped[(row["duration_bucket"], score_bin(row["mean_iou"]))].append(row)
    out = []
    for (bucket, bin_name), rows in grouped.items():
        out.append(
            {
                "duration_bucket": bucket,
                "score_bin": bin_name,
                "query_count": len(rows),
                "video_count_by_query": len({r["video_uid"] for r in rows}),
            }
        )
    out.sort(key=lambda r: (DURATION_BUCKETS.index(r["duration_bucket"]) if r["duration_bucket"] in DURATION_BUCKETS else 99, r["score_bin"]))
    return out


def plot_score_distribution(
    query_stats: list[dict[str, Any]],
    video_stats: list[dict[str, Any]],
    outdir: Path,
) -> None:
    if plt is None or np is None:
        return
    query_scores = np.array([r["mean_iou"] for r in query_stats], dtype=float)
    video_scores = np.array([r["mean_iou"] for r in video_stats], dtype=float)
    bins = np.linspace(0, 1, 21)
    thresholds = np.array(THRESHOLDS)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, scores, title in [
        (axes[0, 0], query_scores, "Query mean IoU"),
        (axes[0, 1], video_scores, "Video mean IoU"),
    ]:
        ax.hist(scores, bins=bins, color="#3b82c4", edgecolor="white", alpha=0.9)
        if len(scores):
            ax.axvline(float(np.mean(scores)), color="#d9480f", linestyle="--", linewidth=1.2, label=f"mean={np.mean(scores):.3f}")
            ax.axvline(float(np.median(scores)), color="#2b8a3e", linestyle="--", linewidth=1.2, label=f"median={np.median(scores):.3f}")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Mean IoU")
        ax.set_ylabel("Count")
        ax.set_title(f"{title} distribution (N={len(scores)})")
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    query_keep = [int(np.sum(query_scores >= t)) for t in thresholds]
    video_keep = [int(np.sum(video_scores >= t)) for t in thresholds]
    axes[1, 0].plot(thresholds, query_keep, marker="o", color="#3b82c4")
    axes[1, 0].set_title("Queries retained by IoU threshold")
    axes[1, 0].set_xlabel("Threshold")
    axes[1, 0].set_ylabel("Query count")
    axes[1, 0].grid(alpha=0.2)
    axes[1, 1].plot(thresholds, video_keep, marker="o", color="#845ef7")
    axes[1, 1].set_title("Videos retained by IoU threshold")
    axes[1, 1].set_xlabel("Threshold")
    axes[1, 1].set_ylabel("Video count")
    axes[1, 1].grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(outdir / "score_distribution.png", dpi=160)
    plt.close(fig)


def plot_duration_distribution(query_stats: list[dict[str, Any]], outdir: Path) -> None:
    if plt is None or np is None:
        return
    durations = np.array([r["duration"] for r in query_stats if not math.isnan(r["duration"])], dtype=float)
    if len(durations) == 0:
        return
    bucket_counts = Counter(r["duration_bucket"] for r in query_stats)
    buckets = [b for b in DURATION_BUCKETS if b in bucket_counts]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(durations, bins=np.arange(0, max(65, durations.max() + 5), 5), color="#0b7285", edgecolor="white", alpha=0.9)
    axes[0].axvline(float(np.mean(durations)), color="#d9480f", linestyle="--", linewidth=1.2, label=f"mean={np.mean(durations):.1f}s")
    axes[0].axvline(float(np.median(durations)), color="#2b8a3e", linestyle="--", linewidth=1.2, label=f"median={np.median(durations):.1f}s")
    axes[0].set_xlabel("Duration (sec)")
    axes[0].set_ylabel("Query count")
    axes[0].set_title(f"Duration distribution (query-level, N={len(durations)})")
    axes[0].legend(fontsize=8)

    values = [bucket_counts[b] for b in buckets]
    bars = axes[1].bar(buckets, values, color="#74b816", edgecolor="white")
    total = sum(values)
    for bar, val in zip(bars, values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val}\n{val / total * 100:.1f}%", ha="center", va="bottom", fontsize=8)
    axes[1].set_xlabel("Duration bucket")
    axes[1].set_ylabel("Query count")
    axes[1].set_title("Duration buckets")
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(outdir / "duration_distribution.png", dpi=160)
    plt.close(fig)


def plot_score_by_duration(query_stats: list[dict[str, Any]], outdir: Path) -> None:
    if plt is None or np is None:
        return
    buckets = [b for b in DURATION_BUCKETS if any(r["duration_bucket"] == b for r in query_stats)]
    if not buckets:
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    box_data = [[r["mean_iou"] for r in query_stats if r["duration_bucket"] == b] for b in buckets]
    axes[0].boxplot(box_data, labels=buckets, showfliers=False, patch_artist=True)
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("Duration bucket")
    axes[0].set_ylabel("Query mean IoU")
    axes[0].set_title("Mean IoU by duration bucket")
    for i, vals in enumerate(box_data, start=1):
        if vals:
            axes[0].text(i, min(0.98, max(vals) + 0.02), f"n={len(vals)}", ha="center", va="bottom", fontsize=8)

    score_bins = [f"[{i/10:.1f},{(i+1)/10:.1f}{']' if i == 9 else ')'}" for i in range(10)]
    matrix = np.zeros((len(buckets), len(score_bins)), dtype=int)
    bin_to_idx = {b: i for i, b in enumerate(score_bins)}
    bucket_to_idx = {b: i for i, b in enumerate(buckets)}
    for row in query_stats:
        bucket = row["duration_bucket"]
        if bucket not in bucket_to_idx:
            continue
        label = score_bin(row["mean_iou"])
        if label in bin_to_idx:
            matrix[bucket_to_idx[bucket], bin_to_idx[label]] += 1

    im = axes[1].imshow(matrix, aspect="auto", cmap="YlGnBu")
    axes[1].set_xticks(range(len(score_bins)))
    axes[1].set_xticklabels(score_bins, rotation=45, ha="right", fontsize=8)
    axes[1].set_yticks(range(len(buckets)))
    axes[1].set_yticklabels(buckets)
    axes[1].set_xlabel("Query mean IoU bin")
    axes[1].set_title("Duration bucket x score bin")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] > 0:
                axes[1].text(j, i, str(int(matrix[i, j])), ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=axes[1], shrink=0.8)

    fig.tight_layout()
    fig.savefig(outdir / "score_by_duration.png", dpi=160)
    plt.close(fig)


def plot_source_summary(query_stats: list[dict[str, Any]], outdir: Path, top_k: int = 20) -> None:
    if plt is None or np is None:
        return
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in query_stats:
        grouped[row["source"]].append(row["mean_iou"])
    if not grouped:
        return
    items = sorted(grouped.items(), key=lambda kv: -len(kv[1]))[:top_k]
    labels = [k for k, _ in items]
    means = [mean(v) for _, v in items]
    counts = [len(v) for _, v in items]

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.55), 5.5))
    bars = ax.bar(range(len(labels)), means, color="#f59f00", edgecolor="white")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"n={count}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean IoU")
    ax.set_title("Mean IoU by source")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(outdir / "source_mean_iou.png", dpi=160)
    plt.close(fig)


def write_summary_md(
    path: Path,
    counters: dict[str, int],
    query_stats: list[dict[str, Any]],
    video_stats: list[dict[str, Any]],
    sweep: list[dict[str, Any]],
) -> None:
    query_scores = [r["mean_iou"] for r in query_stats]
    video_scores = [r["mean_iou"] for r in video_stats]
    durations = [r["duration"] for r in query_stats if not math.isnan(r["duration"])]
    qs = summarize_scores(query_scores)
    vs = summarize_scores(video_scores)
    ds = summarize_scores(durations)

    selected_thresholds = {0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80}
    lines = [
        "# TimeLens TG Rollout Analysis",
        "",
        "## Overview",
        "",
        f"- Report files: {counters.get('report_files', 0)}",
        f"- Reports: {counters.get('reports', 0)}",
        f"- Scored queries: {len(query_stats)}",
        f"- Unique videos: {len(video_stats)}",
        f"- Error reports: {counters.get('errors', 0)}",
        f"- Duplicate query ids skipped: {counters.get('duplicate_query_id', 0)}",
        f"- Missing input metadata: {counters.get('missing_input_metadata', 0)}",
        "",
        "## Mean IoU",
        "",
        "| level | mean | p25 | p50 | p75 | p90 | min | max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| query | {fmt_float(qs['mean'], 3)} | {fmt_float(qs['p25'], 3)} | {fmt_float(qs['p50'], 3)} | {fmt_float(qs['p75'], 3)} | {fmt_float(qs['p90'], 3)} | {fmt_float(qs['min'], 3)} | {fmt_float(qs['max'], 3)} |",
        f"| video | {fmt_float(vs['mean'], 3)} | {fmt_float(vs['p25'], 3)} | {fmt_float(vs['p50'], 3)} | {fmt_float(vs['p75'], 3)} | {fmt_float(vs['p90'], 3)} | {fmt_float(vs['min'], 3)} | {fmt_float(vs['max'], 3)} |",
        "",
        "## Duration",
        "",
        f"- Query-level duration mean: {fmt_float(ds['mean'], 2)} sec",
        f"- Query-level duration median: {fmt_float(ds['p50'], 2)} sec",
        f"- Query-level duration p25/p75: {fmt_float(ds['p25'], 2)} / {fmt_float(ds['p75'], 2)} sec",
        "",
        "## Threshold Sweep",
        "",
        "| IoU >= | queries | videos | query pct | video pct |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sweep:
        if row["threshold"] in selected_thresholds:
            lines.append(
                f"| {row['threshold']:.2f} | {row['query_count']} | {row['video_count']} | "
                f"{row['query_pct'] * 100:.1f}% | {row['video_pct'] * 100:.1f}% |"
            )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `score_distribution.png`: query/video score histograms and retention curves",
            "- `duration_distribution.png`: duration histogram and bucket counts",
            "- `score_by_duration.png`: score distribution split by duration",
            "- `source_mean_iou.png`: source-level mean IoU",
            "- `threshold_sweep.csv`: counts retained by IoU cutoff",
            "- `query_stats.jsonl` and `video_stats.jsonl`: detailed joined stats",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze TimeLens TG rollout score/duration distribution")
    parser.add_argument(
        "--report",
        nargs="*",
        default=[],
        help="One or more report JSONL files. Shell-expanded _shard*_report.jsonl is supported.",
    )
    parser.add_argument(
        "--report-glob",
        action="append",
        default=[],
        help="Glob pattern for report JSONL files, useful when final rollout_report.jsonl was not merged.",
    )
    parser.add_argument("--input-jsonl", required=True, help="tg_rollout_input.jsonl used by rollout")
    parser.add_argument("--output-dir", required=True, help="Directory for analysis outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_paths = resolve_report_paths(args.report, args.report_glob)
    input_path = Path(args.input_jsonl)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not report_paths:
        raise SystemExit("No report files provided. Use --report or --report-glob.")
    missing_reports = [str(path) for path in report_paths if not path.is_file()]
    if missing_reports:
        raise SystemExit("Report file(s) not found:\n" + "\n".join(missing_reports))
    if not input_path.is_file():
        raise SystemExit(f"Input JSONL not found: {input_path}")

    query_index = build_query_index(input_path)
    query_stats, counters = build_query_stats(report_paths, query_index)
    video_stats = build_video_stats(query_stats)

    source_summary = group_summary(query_stats, "source")
    duration_summary = group_summary(query_stats, "duration_bucket")
    sweep, sweep_source, sweep_duration = build_threshold_sweeps(query_stats, video_stats)
    score_bins = build_score_bin_table(query_stats)

    write_jsonl(outdir / "query_stats.jsonl", query_stats)
    write_jsonl(outdir / "video_stats.jsonl", video_stats)
    write_csv(
        outdir / "query_stats.csv",
        query_stats,
        [
            "query_id",
            "video_uid",
            "source",
            "duration",
            "duration_bucket",
            "mean_iou",
            "std_iou",
            "min_iou",
            "max_iou",
            "num_rollouts",
            "shard_id",
            "report_file",
            "query",
            "answer",
        ],
    )
    write_csv(
        outdir / "video_stats.csv",
        video_stats,
        [
            "video_uid",
            "source",
            "duration",
            "duration_bucket",
            "num_queries",
            "mean_iou",
            "median_query_iou",
            "min_query_iou",
            "max_query_iou",
            "rollout_mean_iou",
            "rollout_std_iou",
        ],
    )
    write_csv(outdir / "source_summary.csv", source_summary, ["source", "num_queries", "num_videos", "mean_iou", "median_iou", "p25_iou", "p75_iou", "mean_duration"])
    write_csv(outdir / "duration_summary.csv", duration_summary, ["duration_bucket", "num_queries", "num_videos", "mean_iou", "median_iou", "p25_iou", "p75_iou", "mean_duration"])
    write_csv(outdir / "threshold_sweep.csv", sweep, ["threshold", "query_count", "query_pct", "video_count_by_query", "video_count", "video_pct", "mean_duration", "median_duration"])
    write_csv(outdir / "threshold_sweep_by_source.csv", sweep_source, ["threshold", "source", "query_count", "video_count_by_query"])
    write_csv(outdir / "threshold_sweep_by_duration.csv", sweep_duration, ["threshold", "duration_bucket", "query_count", "video_count_by_query"])
    write_csv(outdir / "score_duration_bins.csv", score_bins, ["duration_bucket", "score_bin", "query_count", "video_count_by_query"])

    summary = {
        "counters": counters,
        "report_files": [str(path) for path in report_paths],
        "query_iou": summarize_scores([r["mean_iou"] for r in query_stats]),
        "video_iou": summarize_scores([r["mean_iou"] for r in video_stats]),
        "duration_sec": summarize_scores([r["duration"] for r in query_stats if not math.isnan(r["duration"])]),
    }
    write_json(outdir / "summary.json", summary)
    write_summary_md(outdir / "summary.md", counters, query_stats, video_stats, sweep)

    plot_score_distribution(query_stats, video_stats, outdir)
    plot_duration_distribution(query_stats, outdir)
    plot_score_by_duration(query_stats, outdir)
    plot_source_summary(query_stats, outdir)

    print("==========================================")
    print(" TimeLens TG analysis done")
    print(f" Report files:   {len(report_paths)}")
    print(f" Reports:        {counters.get('reports', 0)}")
    if counters.get("duplicate_query_id", 0):
        print(f" Duplicates:     {counters['duplicate_query_id']} skipped by metadata_id")
    print(f" Scored queries: {len(query_stats)}")
    print(f" Unique videos:  {len(video_stats)}")
    print(f" Output:         {outdir}")
    if plt is None:
        print(" Plots skipped: matplotlib/numpy not available")
    print("==========================================")


if __name__ == "__main__":
    main()
