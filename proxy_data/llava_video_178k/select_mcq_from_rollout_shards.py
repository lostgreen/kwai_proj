#!/usr/bin/env python3
"""Select low-reward LLaVA MCQ samples from rollout shard reports.

The rollout report index is local after data-parallel sharding, so this script
matches reports back to the original MCQ JSONL by metadata id first and then by
the (prompt, answer) content pair.
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from proxy_data.llava_video_178k.convert_mcq_to_direct import (  # noqa: E402
    ensure_messages,
    rewrite_prompt,
)


IdentityKey = tuple[str, ...]


def expand_paths(paths: list[str], globs: list[str]) -> list[Path]:
    expanded: list[Path] = []
    for value in list(paths or []) + list(globs or []):
        if any(ch in value for ch in "*?["):
            expanded.extend(Path(p) for p in sorted(glob.glob(value)))
        else:
            expanded.append(Path(value))
    unique: dict[str, Path] = {}
    for path in expanded:
        unique[str(path)] = path
    return list(unique.values())


def load_jsonl(path: Path, *, skip_bad_lines: bool = False) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    bad_lines = 0
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                if skip_bad_lines:
                    bad_lines += 1
                    continue
                raise SystemExit(f"Failed to parse {path}:{line_no}: {exc}") from exc
    return rows, bad_lines


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _clean_str(value: Any) -> str:
    return str(value or "").strip()


def _metadata_id(row: dict[str, Any]) -> str:
    meta = row.get("metadata") or {}
    return _clean_str(row.get("metadata_id") or meta.get("id"))


def content_key(row: dict[str, Any]) -> IdentityKey:
    return ("content", _clean_str(row.get("prompt")), _clean_str(row.get("answer")))


def primary_key(row: dict[str, Any]) -> IdentityKey:
    mid = _metadata_id(row)
    if mid:
        return ("metadata_id", mid)
    return content_key(row)


def lookup_keys(row: dict[str, Any]) -> list[IdentityKey]:
    keys = [primary_key(row)]
    ckey = content_key(row)
    if ckey not in keys:
        keys.append(ckey)
    return keys


def safe_float(value: Any) -> float:
    if isinstance(value, dict):
        value = value.get("overall", value.get("reward", value.get("score")))
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result


def reward_values(report: dict[str, Any]) -> list[float]:
    raw_values = report.get("rewards")
    if raw_values is None:
        raw_values = report.get("Rewards")
    if not isinstance(raw_values, list):
        return []
    values = [safe_float(value) for value in raw_values]
    return [value for value in values if not math.isnan(value)]


def report_mean_reward(report: dict[str, Any]) -> float:
    for key in ("mean_reward", "mean_acc", "reward", "Reward", "score"):
        if key in report:
            value = safe_float(report.get(key))
            if not math.isnan(value):
                return value
    rewards = reward_values(report)
    if rewards:
        return sum(rewards) / len(rewards)
    return math.nan


def build_input_lookup(records: list[dict[str, Any]]) -> dict[IdentityKey, list[dict[str, Any]]]:
    lookup: dict[IdentityKey, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        for key in lookup_keys(record):
            lookup[key].append(record)
    return lookup


def _direct_record(record: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(record)
    prompt, _changed = rewrite_prompt(_clean_str(result.get("prompt")))
    result["prompt"] = prompt
    ensure_messages(result, prompt)
    return result


def _record_key(record: dict[str, Any]) -> IdentityKey:
    return primary_key(record)


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_source: Counter[str] = Counter()
    by_duration_bucket: Counter[str] = Counter()
    by_data_source: Counter[str] = Counter()
    for record in records:
        meta = record.get("metadata") or {}
        by_source[_clean_str(meta.get("source")) or "unknown"] += 1
        by_duration_bucket[_clean_str(meta.get("duration_bucket")) or "unknown"] += 1
        by_data_source[_clean_str(meta.get("data_source")) or "unknown"] += 1
    return {
        "by_source": dict(by_source.most_common()),
        "by_duration_bucket": dict(by_duration_bucket.most_common()),
        "by_data_source": dict(by_data_source.most_common()),
    }


def select_records_from_reports(
    *,
    input_paths: list[Path],
    report_paths: list[Path],
    min_mean_reward: float = 0.0,
    max_mean_reward: float = 0.375,
    target_total: int = 0,
    seed: int = 42,
    skip_bad_report_lines: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    input_rows: list[dict[str, Any]] = []
    for path in input_paths:
        rows, _bad = load_jsonl(path)
        input_rows.extend(rows)
    lookup = build_input_lookup(input_rows)

    report_rows: list[tuple[Path, dict[str, Any]]] = []
    bad_report_lines = 0
    for path in report_paths:
        rows, bad = load_jsonl(path, skip_bad_lines=skip_bad_report_lines)
        bad_report_lines += bad
        report_rows.extend((path, row) for row in rows)

    selected: list[dict[str, Any]] = []
    selected_keys: set[IdentityKey] = set()
    candidate_count = 0
    filtered_out_by_reward = 0
    skipped_missing_reward = 0
    missing_input_count = 0
    deduped_count = 0

    for report_path, report in report_rows:
        mean_reward = report_mean_reward(report)
        if math.isnan(mean_reward):
            skipped_missing_reward += 1
            continue
        if mean_reward < min_mean_reward or mean_reward > max_mean_reward:
            filtered_out_by_reward += 1
            continue

        candidate_count += 1
        matches = lookup.get(primary_key(report)) or lookup.get(content_key(report)) or []
        if not matches:
            missing_input_count += 1
            continue

        record = matches[0]
        record_key = _record_key(record)
        if record_key in selected_keys:
            deduped_count += 1
            continue

        out = _direct_record(record)
        meta = out.setdefault("metadata", {})
        rewards = reward_values(report)
        meta["llava_rollout_mean_reward"] = round(mean_reward, 6)
        if rewards:
            meta["llava_rollout_rewards"] = rewards
        meta["llava_rollout_report"] = str(report_path)
        meta["llava_rollout_filter"] = {
            "min_mean_reward": min_mean_reward,
            "max_mean_reward": max_mean_reward,
        }
        meta["mcq_base_training_source"] = "llava_rollout_low_reward"
        selected.append(out)
        selected_keys.add(record_key)

    before_target = len(selected)
    if target_total > 0 and len(selected) > target_total:
        selected = random.Random(seed).sample(selected, target_total)

    summary: dict[str, Any] = {
        "input_files": [str(path) for path in input_paths],
        "report_files": [str(path) for path in report_paths],
        "input_count": len(input_rows),
        "report_count": len(report_rows),
        "candidate_count": candidate_count,
        "selected_before_target": before_target,
        "selected_count": len(selected),
        "target_total": target_total,
        "deduped_count": deduped_count,
        "missing_input_count": missing_input_count,
        "filtered_out_by_reward": filtered_out_by_reward,
        "skipped_missing_reward": skipped_missing_reward,
        "bad_report_lines": bad_report_lines,
        "min_mean_reward": min_mean_reward,
        "max_mean_reward": max_mean_reward,
        **_summarize_records(selected),
    }
    return selected, summary


def merge_base_and_selected(
    *,
    base_paths: list[Path],
    selected_rows: list[dict[str, Any]],
    skip_bad_lines: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[IdentityKey] = set()
    base_count = 0
    bad_base_lines = 0
    skipped_selected_duplicates = 0

    for path in base_paths:
        rows, bad = load_jsonl(path, skip_bad_lines=skip_bad_lines)
        bad_base_lines += bad
        for row in rows:
            base_count += 1
            key = _record_key(row)
            if key in seen:
                continue
            merged.append(row)
            seen.add(key)

    for row in selected_rows:
        key = _record_key(row)
        if key in seen:
            skipped_selected_duplicates += 1
            continue
        merged.append(row)
        seen.add(key)

    summary: dict[str, Any] = {
        "base_files": [str(path) for path in base_paths],
        "base_count": base_count,
        "selected_input_count": len(selected_rows),
        "merged_count": len(merged),
        "skipped_selected_duplicates": skipped_selected_duplicates,
        "bad_base_lines": bad_base_lines,
        **_summarize_records(merged),
    }
    return merged, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select low-reward LLaVA MCQ samples from shard reports")
    parser.add_argument("--input", action="append", default=[], help="Original MCQ JSONL; repeatable")
    parser.add_argument("--input-glob", action="append", default=[], help="Glob for original MCQ JSONL files")
    parser.add_argument("--report", action="append", default=[], help="Rollout report JSONL; repeatable")
    parser.add_argument("--report-glob", action="append", default=[], help="Glob for shard report JSONL files")
    parser.add_argument("--output", required=True, help="Selected low-reward MCQ JSONL")
    parser.add_argument("--summary-json", default="", help="Optional summary JSON path")
    parser.add_argument("--base-jsonl", action="append", default=[], help="Existing base/source MCQ JSONL to merge")
    parser.add_argument("--merged-output", default="", help="Optional merged base + selected MCQ JSONL")
    parser.add_argument("--min-mean-reward", type=float, default=0.0)
    parser.add_argument("--max-mean-reward", type=float, default=0.375)
    parser.add_argument("--target-total", type=int, default=0, help="0 keeps all selected records")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-bad-report-lines",
        action="store_true",
        help="Skip malformed report lines, useful when reading live shard files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = expand_paths(args.input, args.input_glob)
    report_paths = expand_paths(args.report, args.report_glob)
    if not input_paths:
        raise SystemExit("No MCQ input files. Pass --input or --input-glob.")
    if not report_paths:
        raise SystemExit("No report files. Pass --report or --report-glob.")

    missing = [str(path) for path in input_paths + report_paths if not path.is_file()]
    if missing:
        raise SystemExit("Missing input file(s):\n" + "\n".join(missing))

    selected, summary = select_records_from_reports(
        input_paths=input_paths,
        report_paths=report_paths,
        min_mean_reward=args.min_mean_reward,
        max_mean_reward=args.max_mean_reward,
        target_total=args.target_total,
        seed=args.seed,
        skip_bad_report_lines=args.skip_bad_report_lines,
    )
    write_jsonl(Path(args.output), selected)

    merged_summary: dict[str, Any] | None = None
    if args.merged_output:
        base_paths = [Path(p) for p in args.base_jsonl]
        missing_base = [str(path) for path in base_paths if not path.is_file()]
        if missing_base:
            raise SystemExit("Missing base file(s):\n" + "\n".join(missing_base))
        merged, merged_summary = merge_base_and_selected(
            base_paths=base_paths,
            selected_rows=selected,
        )
        write_jsonl(Path(args.merged_output), merged)

    full_summary = dict(summary)
    if merged_summary is not None:
        full_summary["merge"] = merged_summary
    if args.summary_json:
        out_path = Path(args.summary_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(full_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("==========================================")
    print(" LLaVA MCQ low-reward selection")
    print(f" Inputs:      {len(input_paths)}")
    print(f" Reports:     {len(report_paths)}")
    print(f" Range:       [{args.min_mean_reward}, {args.max_mean_reward}]")
    print(f" Candidates:  {summary['candidate_count']}")
    print(f" Selected:    {summary['selected_count']}")
    print(f" Output:      {args.output}")
    if args.merged_output and merged_summary is not None:
        print(f" Merged:      {merged_summary['merged_count']} -> {args.merged_output}")
    if args.summary_json:
        print(f" Summary:     {args.summary_json}")
    print("==========================================")


if __name__ == "__main__":
    main()
