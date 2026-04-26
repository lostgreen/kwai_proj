#!/usr/bin/env python3
"""Merge sharded offline-rollout resume outputs and plot distribution charts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
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
    extract_duration_sec,
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


def content_identity(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("problem_type") or ""),
        str(row.get("prompt") or ""),
        str(row.get("answer") or ""),
    )


def _extract_domain(record: dict[str, Any], domain_key: str) -> str:
    metadata = record.get("metadata")
    value = None
    if isinstance(metadata, dict):
        value = metadata.get(domain_key)
    if value is None:
        value = record.get(domain_key)
    text = str(value or "other").strip()
    return text or "other"


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


def filter_reports_by_mean_reward(
    reports: list[dict[str, Any]],
    *,
    min_mean_reward: float,
    max_mean_reward: float,
    kept_only: bool,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in reports:
        if kept_only and row.get("keep") is not True:
            continue
        mean_reward = row.get("mean_reward")
        if not isinstance(mean_reward, (int, float)):
            continue
        if min_mean_reward <= float(mean_reward) <= max_mean_reward:
            filtered.append(row)
    return filtered


def filter_kept_records_by_reports(kept_records: list[dict[str, Any]], reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = {content_identity(row) for row in reports}
    filtered: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for record in kept_records:
        if content_identity(record) not in wanted:
            continue
        key = build_raw_record_dedupe_key(record)
        if key in seen:
            continue
        seen.add(key)
        filtered.append(record)
    return filtered


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


def _save_problem_type_pie(path: Path, records: list[dict[str, Any]], title: str) -> None:
    counter = Counter(str(row.get("problem_type") or "unknown") for row in records)
    if not counter:
        path.unlink(missing_ok=True)
        return
    labels = [f"{problem_type}\n{count}" for problem_type, count in counter.most_common()]
    values = [count for _, count in counter.most_common()]
    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    _, _, autotexts = ax.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        counterclock=False,
        wedgeprops={"edgecolor": "white"},
        textprops={"fontsize": 9},
    )
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_weight("bold")
        autotext.set_fontsize(9)
    ax.set_title(title)
    fig.savefig(path)
    plt.close(fig)


def _save_input_duration_histogram(path: Path, records: list[dict[str, Any]], title: str) -> None:
    durations = [
        duration
        for duration in (extract_duration_sec(record) for record in records)
        if duration is not None and duration > 0
    ]
    if not durations:
        path.unlink(missing_ok=True)
        return
    bins = min(30, max(5, len(durations)))
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.hist(durations, bins=bins, color="#0f766e", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel("input duration (sec)")
    ax.set_ylabel("samples")
    fig.savefig(path)
    plt.close(fig)


def _write_counter_csv(path: Path, key_name: str, counter: Counter[str]) -> None:
    total = sum(counter.values())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[key_name, "count", "ratio", "total"])
        writer.writeheader()
        for key, count in counter.most_common():
            writer.writerow(
                {
                    key_name: key,
                    "count": count,
                    "ratio": _ratio(count, total),
                    "total": total,
                }
            )


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


def write_filtered_report_outputs(
    *,
    output_dir: Path,
    reports: list[dict[str, Any]],
    success_threshold: float,
    min_mean_reward: float,
    max_mean_reward: float,
    kept_only: bool,
    output_report_name: str,
    kept_records: list[dict[str, Any]] | None = None,
    output_kept_name: str | None = None,
    balance_largest_fraction: float | None = None,
    balance_domain_key: str = "domain_l1",
    balance_seed: int = 42,
    balanced_kept_name: str | None = None,
) -> dict[str, Any]:
    filtered_reports = filter_reports_by_mean_reward(
        reports,
        min_mean_reward=min_mean_reward,
        max_mean_reward=max_mean_reward,
        kept_only=kept_only,
    )
    filtered_summary = summarize_reports(filtered_reports, success_threshold=success_threshold)
    output_path = output_dir / output_report_name
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "filtered_report_by_problem_type.csv"
    plot_path = plot_dir / "filtered_report_problem_type_distribution.png"
    problem_type_pie_path = plot_dir / "filtered_problem_type_pie.png"
    input_duration_histogram_path = plot_dir / "filtered_input_duration_histogram.png"
    write_jsonl_rows(filtered_reports, output_path)
    write_report_csv(csv_path, filtered_summary)
    _save_stacked_report_chart(plot_path, filtered_summary)
    filtered_kept_output = None
    filtered_kept_summary = None
    filtered_kept_records: list[dict[str, Any]] = []
    balanced_outputs = None
    if kept_records is not None and output_kept_name is not None:
        filtered_kept_records = filter_kept_records_by_reports(kept_records, filtered_reports)
        filtered_kept_path = output_dir / output_kept_name
        write_jsonl_rows(filtered_kept_records, filtered_kept_path)
        filtered_kept_output = str(filtered_kept_path)
        filtered_kept_summary = summarize_raw_records(filtered_kept_records)
        _save_problem_type_pie(
            problem_type_pie_path,
            filtered_kept_records,
            "filtered samples by problem_type",
        )
        _save_input_duration_histogram(
            input_duration_histogram_path,
            filtered_kept_records,
            "filtered sample input duration",
        )
        if balance_largest_fraction is not None and balanced_kept_name is not None:
            balanced_outputs = write_balanced_filtered_kept_outputs(
                output_dir=output_dir,
                records=filtered_kept_records,
                largest_fraction=balance_largest_fraction,
                domain_key=balance_domain_key,
                seed=balance_seed,
                output_name=balanced_kept_name,
            )
    return {
        "filter": {
            "min_mean_reward": min_mean_reward,
            "max_mean_reward": max_mean_reward,
            "kept_only": kept_only,
        },
        "output_report": str(output_path),
        "csv": str(csv_path),
        "plot": str(plot_path),
        "summary": filtered_summary,
        "output_kept": filtered_kept_output,
        "kept_summary": filtered_kept_summary,
        "problem_type_pie": str(problem_type_pie_path),
        "input_duration_histogram": str(input_duration_histogram_path),
        "balanced_outputs": balanced_outputs,
    }


def _record_stable_key(record: dict[str, Any]) -> str:
    return json.dumps(build_raw_record_dedupe_key(record), ensure_ascii=False, sort_keys=False)


def sample_records_by_domain_balance(
    records: list[dict[str, Any]],
    *,
    target_count: int,
    domain_key: str,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if target_count <= 0:
        return []
    if target_count >= len(records):
        selected = list(records)
        rng.shuffle(selected)
        return selected

    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_domain[_extract_domain(record, domain_key)].append(record)

    pools: dict[str, list[dict[str, Any]]] = {}
    for domain, domain_records in sorted(by_domain.items()):
        pool = sorted(domain_records, key=_record_stable_key)
        rng.shuffle(pool)
        pools[domain] = pool

    selected: list[dict[str, Any]] = []
    domains = sorted(pools)
    while len(selected) < target_count:
        made_progress = False
        for domain in domains:
            pool = pools[domain]
            if not pool:
                continue
            selected.append(pool.pop())
            made_progress = True
            if len(selected) >= target_count:
                break
        if not made_progress:
            break

    rng.shuffle(selected)
    return selected


def balance_largest_problem_type_by_domain(
    records: list[dict[str, Any]],
    *,
    largest_fraction: float,
    domain_key: str,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not 0.0 < largest_fraction <= 1.0:
        raise ValueError(f"largest_fraction must be in (0, 1], got {largest_fraction!r}")
    if not records:
        return [], {
            "largest_fraction": largest_fraction,
            "domain_key": domain_key,
            "original_total_count": 0,
            "balanced_total_count": 0,
        }

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_type[str(record.get("problem_type") or "unknown")].append(record)

    largest_problem_type, largest_records = max(
        sorted(by_type.items()),
        key=lambda item: len(item[1]),
    )
    target_largest_count = max(1, int(len(largest_records) * largest_fraction))
    rng = random.Random(seed)
    balanced_largest = sample_records_by_domain_balance(
        largest_records,
        target_count=target_largest_count,
        domain_key=domain_key,
        rng=rng,
    )

    balanced_records: list[dict[str, Any]] = []
    for problem_type in sorted(by_type):
        if problem_type == largest_problem_type:
            balanced_records.extend(balanced_largest)
        else:
            balanced_records.extend(by_type[problem_type])
    rng.shuffle(balanced_records)

    original_by_type = Counter(str(record.get("problem_type") or "unknown") for record in records)
    balanced_by_type = Counter(str(record.get("problem_type") or "unknown") for record in balanced_records)
    balance_summary = {
        "largest_fraction": largest_fraction,
        "domain_key": domain_key,
        "seed": seed,
        "largest_problem_type": largest_problem_type,
        "original_largest_count": len(largest_records),
        "balanced_largest_count": len(balanced_largest),
        "original_total_count": len(records),
        "balanced_total_count": len(balanced_records),
        "original_by_problem_type": dict(sorted(original_by_type.items())),
        "balanced_by_problem_type": dict(sorted(balanced_by_type.items())),
        "largest_original_by_domain": dict(sorted(Counter(_extract_domain(row, domain_key) for row in largest_records).items())),
        "largest_balanced_by_domain": dict(sorted(Counter(_extract_domain(row, domain_key) for row in balanced_largest).items())),
    }
    return balanced_records, balance_summary


def write_balanced_filtered_kept_outputs(
    *,
    output_dir: Path,
    records: list[dict[str, Any]],
    largest_fraction: float,
    domain_key: str,
    seed: int,
    output_name: str,
) -> dict[str, Any]:
    balanced_records, balance_summary = balance_largest_problem_type_by_domain(
        records,
        largest_fraction=largest_fraction,
        domain_key=domain_key,
        seed=seed,
    )
    output_path = output_dir / output_name
    stats_path = output_dir / "balanced_filtered_kept_stats.json"
    problem_type_csv = output_dir / "balanced_filtered_kept_by_problem_type.csv"
    domain_csv = output_dir / f"balanced_filtered_kept_by_{domain_key}.csv"
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    problem_type_pie = plot_dir / "balanced_filtered_problem_type_pie.png"
    input_duration_histogram = plot_dir / "balanced_filtered_input_duration_histogram.png"

    write_jsonl_rows(balanced_records, output_path)
    summary = summarize_raw_records(balanced_records)
    write_stats_output(
        {
            "stage": "balance-filtered-kept",
            "output_kept": str(output_path),
            "summary": summary,
            "balance": balance_summary,
        },
        stats_path,
    )
    _write_counter_csv(problem_type_csv, "problem_type", Counter(str(row.get("problem_type") or "unknown") for row in balanced_records))
    _write_counter_csv(domain_csv, domain_key, Counter(_extract_domain(row, domain_key) for row in balanced_records))
    _save_problem_type_pie(problem_type_pie, balanced_records, "balanced filtered samples by problem_type")
    _save_input_duration_histogram(
        input_duration_histogram,
        balanced_records,
        "balanced filtered sample input duration",
    )

    return {
        "output_kept": str(output_path),
        "stats": str(stats_path),
        "problem_type_csv": str(problem_type_csv),
        "domain_csv": str(domain_csv),
        "problem_type_pie": str(problem_type_pie),
        "input_duration_histogram": str(input_duration_histogram),
        "summary": summary,
        "balance": balance_summary,
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
    parser.add_argument(
        "--filter-min-mean-reward",
        type=float,
        help="Optionally write a filtered report with mean_reward >= this value.",
    )
    parser.add_argument(
        "--filter-max-mean-reward",
        type=float,
        help="Optionally write a filtered report with mean_reward <= this value.",
    )
    parser.add_argument(
        "--filter-include-dropped",
        action="store_true",
        help="Include keep=False rows in the optional filtered report.",
    )
    parser.add_argument("--filtered-report", default="rollout_report.filtered.jsonl")
    parser.add_argument("--filtered-kept", default="rollout_output.filtered.jsonl")
    parser.add_argument(
        "--balance-largest-fraction",
        type=float,
        default=0.0,
        help="If >0, downsample the largest problem_type in the filtered kept set by this fraction.",
    )
    parser.add_argument(
        "--balance-domain-key",
        default="domain_l1",
        help="Metadata key used for balanced sampling within the largest problem_type.",
    )
    parser.add_argument("--balance-seed", type=int, default=42)
    parser.add_argument("--balanced-kept", default="rollout_output.filtered.balanced.jsonl")
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
    filtered_outputs = None
    if args.filter_min_mean_reward is not None or args.filter_max_mean_reward is not None:
        if args.filter_min_mean_reward is None or args.filter_max_mean_reward is None:
            raise SystemExit("--filter-min-mean-reward and --filter-max-mean-reward must be provided together")
        filtered_outputs = write_filtered_report_outputs(
            output_dir=output_dir,
            reports=merged_reports,
            success_threshold=args.success_threshold,
            min_mean_reward=args.filter_min_mean_reward,
            max_mean_reward=args.filter_max_mean_reward,
            kept_only=not args.filter_include_dropped,
            output_report_name=args.filtered_report,
            kept_records=merged_kept,
            output_kept_name=args.filtered_kept,
            balance_largest_fraction=args.balance_largest_fraction if args.balance_largest_fraction > 0 else None,
            balance_domain_key=args.balance_domain_key,
            balance_seed=args.balance_seed,
            balanced_kept_name=args.balanced_kept,
        )

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
        "filtered_outputs": filtered_outputs,
    }
    write_stats_output(summary, stats_output)

    print(f"[merge-rollout] report files: {len(report_files)} unique_reports={len(merged_reports)}")
    print(f"[merge-rollout] kept files: {len(kept_files)} unique_kept={len(merged_kept)}")
    print(f"[merge-rollout] output report: {output_report}")
    print(f"[merge-rollout] output kept: {output_kept}")
    print(f"[merge-rollout] stats: {stats_output}")
    print(f"[merge-rollout] plots: {output_dir / 'plots'}")
    if filtered_outputs is not None:
        filtered_summary = filtered_outputs["summary"]
        print(
            "[merge-rollout] filtered "
            f"mean_reward=[{args.filter_min_mean_reward}, {args.filter_max_mean_reward}] "
            f"kept_only={not args.filter_include_dropped}: {filtered_summary['report_total']} rows"
        )
        print("[merge-rollout] filtered by_problem_type:")
        for problem_type, stats in filtered_summary["by_problem_type"].items():
            print(f"  {problem_type}: {stats['total']}")
        if filtered_outputs.get("kept_summary") is not None:
            kept_summary = filtered_outputs["kept_summary"]
            print(f"[merge-rollout] filtered raw kept records: {kept_summary['total_count']}")
            print("[merge-rollout] filtered raw kept by_problem_type:")
            for problem_type, count in kept_summary["by_problem_type"].items():
                print(f"  {problem_type}: {count}")
        if filtered_outputs.get("balanced_outputs") is not None:
            balanced_outputs = filtered_outputs["balanced_outputs"]
            balance = balanced_outputs["balance"]
            print(
                "[merge-rollout] balanced filtered raw kept "
                f"largest={balance['largest_problem_type']} "
                f"{balance['original_largest_count']} -> {balance['balanced_largest_count']} "
                f"total={balance['balanced_total_count']}"
            )
            print(f"[merge-rollout] balanced output: {balanced_outputs['output_kept']}")
    return summary


if __name__ == "__main__":
    main()
