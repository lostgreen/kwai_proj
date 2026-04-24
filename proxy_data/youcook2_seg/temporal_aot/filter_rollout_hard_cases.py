#!/usr/bin/env python3
"""Filter rollout reports into a hard-case training set."""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from proxy_data.youcook2_seg.temporal_aot.hard_qa_pipeline import (  # noqa: E402
    build_raw_record_dedupe_key,
    load_jsonl_rows,
    summarize_raw_records,
    write_jsonl_rows,
    write_stats_output,
)


def count_successes(report_entry: dict, success_threshold: float) -> int:
    rewards = report_entry.get("rewards")
    if isinstance(rewards, list):
        return sum(1 for reward in rewards if isinstance(reward, (int, float)) and reward >= success_threshold)

    max_reward = report_entry.get("max_reward")
    if isinstance(max_reward, (int, float)):
        return 1 if float(max_reward) >= success_threshold else 0

    mean_reward = report_entry.get("mean_reward")
    if isinstance(mean_reward, (int, float)):
        return 1 if float(mean_reward) >= success_threshold else 0

    return 0


def _record_identity(record: dict) -> tuple[str, str]:
    return str(record.get("prompt", "")), str(record.get("answer", ""))


def _report_identity(report_entry: dict) -> tuple[str, int] | tuple[str, str, str]:
    idx = report_entry.get("index", -1)
    if isinstance(idx, int) and idx >= 0:
        return ("index", idx)
    prompt, answer = _record_identity(report_entry)
    return ("content", prompt, answer)


def _sample_nested(records: list[dict], target: int, nested_key: str, rng: random.Random) -> list[dict]:
    if len(records) <= target or target <= 0 or not nested_key:
        picked = list(records)
        rng.shuffle(picked)
        return picked[:target] if target > 0 else picked

    by_nested: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        meta = rec.get("metadata") or {}
        by_nested[str(meta.get(nested_key, "unknown"))].append(rec)

    groups = sorted(by_nested)
    per_group = max(1, target // max(len(groups), 1))
    remainder = target - per_group * len(groups)

    sampled: list[dict] = []
    leftovers: list[dict] = []
    for group in groups:
        pool = list(by_nested[group])
        rng.shuffle(pool)
        take = min(per_group, len(pool))
        sampled.extend(pool[:take])
        leftovers.extend(pool[take:])

    if remainder > 0 and leftovers:
        rng.shuffle(leftovers)
        sampled.extend(leftovers[:remainder])

    rng.shuffle(sampled)
    return sampled[:target]


def lookup_original_record(
    report_entry: dict,
    originals: list[dict],
    originals_by_content: dict[tuple[str, str], list[tuple[int, dict]]],
) -> tuple[int, dict] | None:
    idx = report_entry.get("index", -1)
    if isinstance(idx, int) and 0 <= idx < len(originals):
        indexed_record = originals[idx]
        if _record_identity(indexed_record) == _record_identity(report_entry):
            return idx, indexed_record

    key = _record_identity(report_entry)
    matches = originals_by_content.get(key, [])
    if len(matches) == 1:
        return matches[0]
    return None


def collect_hard_case_candidates(
    reports: list[dict],
    originals: list[dict],
    *,
    min_mean_reward: float,
    max_mean_reward: float,
    min_success_count: int,
    success_threshold: float,
) -> tuple[list[dict], dict]:
    originals_by_content: dict[tuple[str, str], list[tuple[int, dict]]] = defaultdict(list)
    for idx, record in enumerate(originals):
        originals_by_content[_record_identity(record)].append((idx, record))

    candidates: list[dict] = []
    seen_keys: set[tuple[str, int] | tuple[str, str, str]] = set()
    filtered_out_by_upstream_keep = 0
    filtered_out_by_mean_reward = 0
    filtered_out_by_success_count = 0
    unmatched_report_count = 0
    duplicate_report_matches = 0

    for report in reports:
        if report.get("keep") is False:
            filtered_out_by_upstream_keep += 1
            continue

        mean_reward = float(report.get("mean_reward", 0.0))
        if not (min_mean_reward <= mean_reward <= max_mean_reward):
            filtered_out_by_mean_reward += 1
            continue

        success_count = count_successes(report, success_threshold=success_threshold)
        if success_count < min_success_count:
            filtered_out_by_success_count += 1
            continue

        matched = lookup_original_record(report, originals=originals, originals_by_content=originals_by_content)
        if matched is None:
            unmatched_report_count += 1
            continue
        original_index, record = matched

        key = ("index", original_index)
        if key in seen_keys:
            duplicate_report_matches += 1
            continue
        seen_keys.add(key)
        candidates.append(record)

    summary = {
        "candidate_count": len(candidates),
        "filtered_out_by_upstream_keep": filtered_out_by_upstream_keep,
        "filtered_out_by_mean_reward": filtered_out_by_mean_reward,
        "filtered_out_by_success_count": filtered_out_by_success_count,
        "unmatched_report_count": unmatched_report_count,
        "duplicate_report_matches": duplicate_report_matches,
    }
    return candidates, summary


def balance_hard_cases(
    records: list[dict],
    *,
    target_total: int,
    nested_balance_key: str,
    seed: int,
) -> tuple[list[dict], dict[str, Any]]:
    rng = random.Random(seed)
    if target_total <= 0 or len(records) <= target_total:
        final_records = list(records)
        rng.shuffle(final_records)
        return final_records, {
            "applied": False,
            "target_total": target_total,
            "nested_balance_key": nested_balance_key,
        }

    by_type: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_type[str(record.get("problem_type", "unknown"))].append(record)

    task_types = sorted(by_type)
    per_type = max(1, target_total // max(len(task_types), 1))
    remainder = target_total - per_type * len(task_types)

    final_records: list[dict] = []
    leftovers: list[dict] = []
    for task_type in task_types:
        pool = by_type[task_type]
        picked = _sample_nested(pool, per_type, nested_balance_key, rng)
        final_records.extend(picked)
        picked_keys = {build_raw_record_dedupe_key(record) for record in picked}
        leftovers.extend([record for record in pool if build_raw_record_dedupe_key(record) not in picked_keys])

    if remainder > 0 and leftovers:
        rng.shuffle(leftovers)
        final_records.extend(leftovers[:remainder])

    rng.shuffle(final_records)
    return final_records[:target_total], {
        "applied": True,
        "target_total": target_total,
        "nested_balance_key": nested_balance_key,
        "per_problem_type_target": per_type,
        "remainder": remainder,
        "problem_type_count": len(task_types),
    }


def summarize_hard_cases(
    final_records: list[dict],
    *,
    report_path: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    stats_output_path: str | Path,
    report_count: int,
    input_count: int,
    candidate_summary: dict,
    balancing_summary: dict,
    min_mean_reward: float,
    max_mean_reward: float,
    min_success_count: int,
    success_threshold: float,
    seed: int,
) -> dict:
    summary = summarize_raw_records(final_records)
    summary.update(
        {
            "stage": "filter-rollout-hard-cases",
            "report_path": str(Path(report_path).resolve()),
            "input_path": str(Path(input_path).resolve()),
            "output_path": str(Path(output_path).resolve()),
            "stats_output_path": str(Path(stats_output_path).resolve()),
            "report_record_count": report_count,
            "input_record_count": input_count,
            "seed": seed,
            "min_mean_reward": min_mean_reward,
            "max_mean_reward": max_mean_reward,
            "min_success_count": min_success_count,
            "success_threshold": success_threshold,
            "balancing": balancing_summary,
        }
    )
    summary.update(candidate_summary)
    summary["selected_by_problem_type"] = dict(sorted(Counter(str(record.get("problem_type", "unknown")) for record in final_records).items()))
    return summary


def filter_rollout_hard_cases(
    *,
    report_path: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    stats_output_path: str | Path | None = None,
    min_mean_reward: float = 0.125,
    max_mean_reward: float = 0.625,
    min_success_count: int = 1,
    success_threshold: float = 1.0,
    target_total: int = 5000,
    nested_balance_key: str = "domain_l1",
    seed: int = 42,
) -> dict:
    report_path = Path(report_path)
    input_path = Path(input_path)
    output_path = Path(output_path)
    if stats_output_path is None:
        stats_output_path = output_path.with_name(f"{output_path.stem}.stats.json")
    stats_output_path = Path(stats_output_path)

    reports = load_jsonl_rows([report_path])
    originals = load_jsonl_rows([input_path])
    candidates, candidate_summary = collect_hard_case_candidates(
        reports,
        originals,
        min_mean_reward=min_mean_reward,
        max_mean_reward=max_mean_reward,
        min_success_count=min_success_count,
        success_threshold=success_threshold,
    )
    final_records, balancing_summary = balance_hard_cases(
        candidates,
        target_total=target_total,
        nested_balance_key=nested_balance_key,
        seed=seed,
    )

    write_jsonl_rows(final_records, output_path)
    summary = summarize_hard_cases(
        final_records,
        report_path=report_path,
        input_path=input_path,
        output_path=output_path,
        stats_output_path=stats_output_path,
        report_count=len(reports),
        input_count=len(originals),
        candidate_summary=candidate_summary,
        balancing_summary=balancing_summary,
        min_mean_reward=min_mean_reward,
        max_mean_reward=max_mean_reward,
        min_success_count=min_success_count,
        success_threshold=success_threshold,
        seed=seed,
    )
    write_stats_output(summary, stats_output_path)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Filter rollout reports into hard-but-solvable cases",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--report", required=True, help="Rollout report JSONL from offline_rollout_filter.py")
    parser.add_argument("--input", required=True, help="Original raw JSONL used for rollout")
    parser.add_argument("--output", required=True, help="Filtered hard-case JSONL output path")
    parser.add_argument(
        "--stats-output",
        help="Optional hard-case stats JSON path (defaults to <output stem>.stats.json)",
    )
    parser.add_argument("--min-mean-reward", type=float, default=0.125)
    parser.add_argument("--max-mean-reward", type=float, default=0.625)
    parser.add_argument(
        "--min-success-count",
        type=int,
        default=1,
        help="Keep only reports with at least this many rollout rewards >= success-threshold",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=1.0,
        help="Reward threshold treated as a successful rollout",
    )
    parser.add_argument(
        "--target-total",
        type=int,
        default=5000,
        help="0 or negative means keep all filtered samples",
    )
    parser.add_argument(
        "--nested-balance-key",
        default="domain_l1",
        help="metadata key used within each problem_type bucket",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> dict:
    parser = _build_parser()
    args = parser.parse_args(argv)
    summary = filter_rollout_hard_cases(
        report_path=args.report,
        input_path=args.input,
        output_path=args.output,
        stats_output_path=args.stats_output,
        min_mean_reward=args.min_mean_reward,
        max_mean_reward=args.max_mean_reward,
        min_success_count=args.min_success_count,
        success_threshold=args.success_threshold,
        target_total=args.target_total,
        nested_balance_key=args.nested_balance_key,
        seed=args.seed,
    )

    print(
        f"[filter-rollout-hard-cases] selected {summary['total_count']} / {summary['report_record_count']} reports "
        f"(candidates={summary['candidate_count']}, min_success_count={summary['min_success_count']}, "
        f"mean_reward in [{summary['min_mean_reward']}, {summary['max_mean_reward']}])"
    )
    print(f"[filter-rollout-hard-cases] output: {summary['output_path']}")
    print(f"[filter-rollout-hard-cases] stats: {summary['stats_output_path']}")
    print("[filter-rollout-hard-cases] by_problem_type:")
    for task_type, count in summary["by_problem_type"].items():
        print(f"  {task_type}: {count}")
    return summary


if __name__ == "__main__":
    main()
