#!/usr/bin/env python3
"""Merge sharded offline-rollout resume outputs and plot distribution charts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("XDG_CACHE_HOME", str(Path("/tmp") / "codex-xdg-cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "codex-matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from proxy_data.youcook2_seg.temporal_aot.filter_rollout_hard_cases import count_successes  # noqa: E402
from proxy_data.youcook2_seg.temporal_aot.hard_qa_pipeline import (  # noqa: E402
    build_raw_record_dedupe_key,
    summarize_raw_records,
    write_jsonl_rows,
    write_stats_output,
)


RUN_ORDER = {"base": 0, "resume": 1, "resume2": 2}
REPORT_RE = re.compile(r"^_shard(?P<shard>\d+)(?P<resume>_resume\d*)?_report\.jsonl$")
KEPT_RE = re.compile(r"^_shard(?P<shard>\d+)(?P<resume>_resume\d*)?_kept\.jsonl$")


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 160,
        "savefig.bbox": "tight",
    }
)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"failed to parse {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"expected object in {path}:{line_no}, got {type(row).__name__}")
            yield row


def _run_label(match: re.Match[str]) -> str:
    raw = match.group("resume")
    if not raw:
        return "base"
    return raw.lstrip("_")


def _run_rank(run_label: str) -> int:
    if run_label in RUN_ORDER:
        return RUN_ORDER[run_label]
    match = re.fullmatch(r"resume(\d+)", run_label)
    if match:
        return int(match.group(1))
    return 99


def discover_files(rollout_dir: Path, pattern: re.Pattern[str]) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in rollout_dir.glob("_shard*.jsonl"):
        match = pattern.match(path.name)
        if not match:
            continue
        run_label = _run_label(match)
        files.append(
            {
                "path": path,
                "shard": int(match.group("shard")),
                "run_label": run_label,
                "run_rank": _run_rank(run_label),
            }
        )
    return sorted(files, key=lambda item: (item["shard"], item["run_rank"], item["path"].name))


def report_identity(row: dict[str, Any], shard: int) -> tuple[Any, ...]:
    metadata_id = str(row.get("metadata_id") or "").strip()
    if metadata_id:
        return ("metadata_id", metadata_id)

    prompt = str(row.get("prompt") or "")
    answer = str(row.get("answer") or "")
    problem_type = str(row.get("problem_type") or "")
    if prompt or answer:
        return ("content", problem_type, prompt, answer)

    index = row.get("index")
    if isinstance(index, int):
        return ("shard_index", shard, index)
    return ("row", shard, json.dumps(row, ensure_ascii=False, sort_keys=True))


def merge_reports(report_files: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    attempts = 0
    by_identity: dict[tuple[Any, ...], tuple[tuple[int, int, int], dict[str, Any]]] = {}
    duplicate_attempts = 0

    for file_order, spec in enumerate(report_files):
        path = spec["path"]
        for row_order, row in enumerate(iter_jsonl(path)):
            attempts += 1
            merged_row = dict(row)
            merged_row["_merge_source"] = {
                "path": str(path),
                "shard": spec["shard"],
                "run": spec["run_label"],
            }
            key = report_identity(merged_row, shard=spec["shard"])
            order = (int(spec["run_rank"]), file_order, row_order)
            if key in by_identity:
                duplicate_attempts += 1
            by_identity[key] = (order, merged_row)

    merged = [item[1] for _, item in sorted(by_identity.items(), key=lambda kv: kv[1][0])]
    summary = {
        "report_input_files": [str(spec["path"]) for spec in report_files],
        "report_attempt_count": attempts,
        "report_unique_count": len(merged),
        "report_duplicate_attempt_count": duplicate_attempts,
    }
    return merged, summary


def merge_kept_records(kept_files: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    attempts = 0
    seen: set[tuple[Any, ...]] = set()
    merged: list[dict[str, Any]] = []
    duplicate_records = 0

    for spec in kept_files:
        path = spec["path"]
        for row in iter_jsonl(path):
            attempts += 1
            key = build_raw_record_dedupe_key(row)
            if key in seen:
                duplicate_records += 1
                continue
            seen.add(key)
            merged.append(row)

    merged.sort(key=lambda row: json.dumps(build_raw_record_dedupe_key(row), ensure_ascii=False, sort_keys=False))
    summary = {
        "kept_input_files": [str(spec["path"]) for spec in kept_files],
        "kept_input_record_count": attempts,
        "kept_unique_record_count": len(merged),
        "kept_duplicate_record_count": duplicate_records,
    }
    return merged, summary


def _ratio(count: int, total: int) -> float:
    return count / total if total else 0.0


def summarize_reports(reports: list[dict[str, Any]], success_threshold: float) -> dict[str, Any]:
    total = len(reports)
    kept = [row for row in reports if row.get("keep") is True]
    dropped = [row for row in reports if row.get("keep") is not True]
    errors = [row for row in reports if row.get("error")]
    by_type = Counter(str(row.get("problem_type") or "unknown") for row in reports)
    kept_by_type = Counter(str(row.get("problem_type") or "unknown") for row in kept)
    dropped_by_type = Counter(str(row.get("problem_type") or "unknown") for row in dropped)
    success_by_type: dict[str, Counter[int]] = defaultdict(Counter)
    for row in reports:
        success_by_type[str(row.get("problem_type") or "unknown")][count_successes(row, success_threshold)] += 1

    per_problem_type = {}
    for problem_type in sorted(by_type):
        type_total = by_type[problem_type]
        per_problem_type[problem_type] = {
            "total": type_total,
            "kept": kept_by_type.get(problem_type, 0),
            "dropped": dropped_by_type.get(problem_type, 0),
            "keep_ratio": _ratio(kept_by_type.get(problem_type, 0), type_total),
            "success_count_histogram": dict(sorted(success_by_type[problem_type].items())),
        }

    return {
        "report_total": total,
        "report_kept": len(kept),
        "report_dropped": len(dropped),
        "report_error_count": len(errors),
        "report_keep_ratio": _ratio(len(kept), total),
        "by_problem_type": per_problem_type,
    }


def write_report_csv(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["problem_type", "total", "kept", "dropped", "keep_ratio"],
        )
        writer.writeheader()
        for problem_type, stats in summary["by_problem_type"].items():
            writer.writerow(
                {
                    "problem_type": problem_type,
                    "total": stats["total"],
                    "kept": stats["kept"],
                    "dropped": stats["dropped"],
                    "keep_ratio": stats["keep_ratio"],
                }
            )


def _save_bar_chart(path: Path, title: str, labels: list[str], values: list[int], ylabel: str) -> None:
    if not labels:
        path.unlink(missing_ok=True)
        return
    width = max(7.0, min(18.0, 0.55 * len(labels) + 3.0))
    fig, ax = plt.subplots(figsize=(width, 5.0))
    bars = ax.bar(range(len(labels)), values, color="#3b82f6")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.bar_label(bars, padding=2, fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def _save_stacked_report_chart(path: Path, summary: dict[str, Any]) -> None:
    labels = list(summary["by_problem_type"])
    if not labels:
        path.unlink(missing_ok=True)
        return
    kept = [summary["by_problem_type"][label]["kept"] for label in labels]
    dropped = [summary["by_problem_type"][label]["dropped"] for label in labels]
    width = max(8.0, min(18.0, 0.65 * len(labels) + 3.0))
    fig, ax = plt.subplots(figsize=(width, 5.2))
    x = list(range(len(labels)))
    ax.bar(x, kept, color="#16a34a", label="kept")
    ax.bar(x, dropped, bottom=kept, color="#dc2626", label="dropped")
    ax.set_title("rollout report distribution by problem_type")
    ax.set_ylabel("samples")
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.legend(frameon=False)
    fig.savefig(path)
    plt.close(fig)


def _save_mean_reward_hist(path: Path, reports: list[dict[str, Any]]) -> None:
    values = [float(row["mean_reward"]) for row in reports if isinstance(row.get("mean_reward"), (int, float))]
    if not values:
        path.unlink(missing_ok=True)
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.hist(values, bins=20, range=(0.0, 1.0), color="#2563eb", edgecolor="white")
    ax.set_title("mean_reward distribution")
    ax.set_xlabel("mean_reward")
    ax.set_ylabel("samples")
    fig.savefig(path)
    plt.close(fig)


def write_plots(output_dir: Path, reports: list[dict[str, Any]], report_summary: dict[str, Any], kept_records: list[dict[str, Any]]) -> dict[str, str]:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    report_dist = plot_dir / "report_problem_type_distribution.png"
    _save_stacked_report_chart(report_dist, report_summary)

    mean_reward = plot_dir / "mean_reward_histogram.png"
    _save_mean_reward_hist(mean_reward, reports)

    kept_dist = plot_dir / "kept_problem_type_distribution.png"
    kept_counter = Counter(str(row.get("problem_type") or "unknown") for row in kept_records)
    _save_bar_chart(
        kept_dist,
        "merged kept records by problem_type",
        [label for label, _ in kept_counter.most_common()],
        [count for _, count in kept_counter.most_common()],
        "records",
    )

    domain_dist = plot_dir / "kept_domain_l1_distribution.png"
    domain_counter = Counter(str((row.get("metadata") or {}).get("domain_l1") or row.get("domain_l1") or "other") for row in kept_records)
    _save_bar_chart(
        domain_dist,
        "merged kept records by domain_l1",
        [label for label, _ in domain_counter.most_common()],
        [count for _, count in domain_counter.most_common()],
        "records",
    )

    return {
        "report_problem_type_distribution": str(report_dist),
        "mean_reward_histogram": str(mean_reward),
        "kept_problem_type_distribution": str(kept_dist),
        "kept_domain_l1_distribution": str(domain_dist),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge _shard*/resume offline rollout outputs and draw distribution charts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--rollout-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, help="Defaults to --rollout-dir")
    parser.add_argument("--output-report", default="rollout_report.merged.jsonl")
    parser.add_argument("--output-kept", default="rollout_output.merged.jsonl")
    parser.add_argument("--stats-output", default="rollout_merge_stats.json")
    parser.add_argument("--success-threshold", type=float, default=1.0)
    return parser.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    rollout_dir = args.rollout_dir
    output_dir = args.output_dir or rollout_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    report_files = discover_files(rollout_dir, REPORT_RE)
    kept_files = discover_files(rollout_dir, KEPT_RE)
    if not report_files and not kept_files:
        raise SystemExit(f"no shard report/kept files found under {rollout_dir}")

    merged_reports, report_merge_summary = merge_reports(report_files)
    merged_kept, kept_merge_summary = merge_kept_records(kept_files)
    report_summary = summarize_reports(merged_reports, success_threshold=args.success_threshold)
    raw_summary = summarize_raw_records(merged_kept) if merged_kept else {}

    output_report = output_dir / args.output_report
    output_kept = output_dir / args.output_kept
    stats_output = output_dir / args.stats_output
    write_jsonl_rows(merged_reports, output_report)
    write_jsonl_rows(merged_kept, output_kept)
    write_report_csv(output_dir / "rollout_report_by_problem_type.csv", report_summary)
    plot_paths = write_plots(output_dir, merged_reports, report_summary, merged_kept)

    summary = {
        "stage": "merge-rollout-resume-outputs",
        "rollout_dir": str(rollout_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "output_report": str(output_report.resolve()),
        "output_kept": str(output_kept.resolve()),
        "stats_output": str(stats_output.resolve()),
        "success_threshold": args.success_threshold,
        **report_merge_summary,
        **kept_merge_summary,
        "report_summary": report_summary,
        "merged_kept_summary": raw_summary,
        "plots": plot_paths,
    }
    write_stats_output(summary, stats_output)

    print(f"[merge-rollout] report files: {len(report_files)} unique_reports={len(merged_reports)}")
    print(f"[merge-rollout] kept files: {len(kept_files)} unique_kept={len(merged_kept)}")
    print(f"[merge-rollout] output report: {output_report}")
    print(f"[merge-rollout] output kept: {output_kept}")
    print(f"[merge-rollout] stats: {stats_output}")
    print(f"[merge-rollout] plots: {output_dir / 'plots'}")
    return summary


if __name__ == "__main__":
    main()
