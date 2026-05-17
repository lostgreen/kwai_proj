#!/usr/bin/env python3
"""Summarize 100-step CoT vs no-CoT learning efficiency experiments."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


DEFAULT_TRAIN_KEY = "reward/overall"
DEFAULT_VAL_KEY = "val/reward_score"
THRESHOLDS = (0.3, 0.5, 0.7)
LOG_COT_KEYS = {
    "cot_budget/end_detected_ratio": "log_cot_end_ratio",
    "cot_budget/repaired_ratio": "log_cot_repaired_ratio",
    "cot_budget/text_fallback_ratio": "log_cot_text_fallback_ratio",
    "cot_budget/final_token_len_mean": "log_cot_final_token_len_mean",
    "cot_budget/remaining_tokens_mean": "log_cot_remaining_tokens_mean",
}


def flatten(obj: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in obj.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(flatten(value, name))
        else:
            out[name] = value
    return out


def as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def read_log(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        path = path / "experiment_log.jsonl"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def metric_series(
    rows: list[dict[str, Any]],
    key: str,
    *,
    max_step: int | None = None,
) -> list[tuple[int, float]]:
    series: list[tuple[int, float]] = []
    for row in rows:
        step = int(row.get("step", -1))
        if max_step is not None and step > max_step:
            continue
        value = as_float(flatten(row).get(key))
        if value is not None:
            series.append((step, value))
    return series


def latest_metric(
    rows: list[dict[str, Any]],
    key: str,
    *,
    max_step: int | None = None,
) -> float | None:
    series = metric_series(rows, key, max_step=max_step)
    return series[-1][1] if series else None


def collect_val_task_scores(
    rows: list[dict[str, Any]],
    *,
    max_step: int | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    scores: dict[str, float] = {}
    counts: dict[str, float] = {}
    for row in rows:
        step = int(row.get("step", -1))
        if max_step is not None and step > max_step:
            continue
        flat = flatten(row)
        for key, value in flat.items():
            score_match = re.match(r"val/(.+)/overall_reward$", key)
            count_match = re.match(r"val/(.+)/count$", key)
            if score_match:
                score = as_float(value)
                if score is not None:
                    scores[score_match.group(1)] = score
            elif count_match:
                count = as_float(value)
                if count is not None:
                    counts[count_match.group(1)] = count
    return scores, counts


def is_proxy_task(family: str, task: str) -> bool:
    task_l = task.lower()
    if family == "aot":
        return "aot" in task_l
    if family == "seg":
        return "hier" in task_l or ("seg" in task_l and "aot" not in task_l)
    if family == "logic":
        return any(part in task_l for part in ("logic", "fill_blank", "predict_next", "sort"))
    return False


def weighted_mean_for_tasks(
    scores: dict[str, float],
    counts: dict[str, float],
    predicate: Any,
) -> float | None:
    selected = [(score, counts.get(task, 1.0)) for task, score in scores.items() if predicate(task)]
    denom = sum(max(count, 0.0) for _, count in selected)
    if denom <= 0.0:
        return None
    return sum(score * max(count, 0.0) for score, count in selected) / denom


def delta(series: list[tuple[int, float]]) -> float | None:
    if len(series) < 2:
        return None
    return series[-1][1] - series[0][1]


def slope(series: list[tuple[int, float]]) -> float | None:
    if len(series) < 2:
        return None
    start_step, start_value = series[0]
    end_step, end_value = series[-1]
    if end_step == start_step:
        return None
    return (end_value - start_value) / (end_step - start_step)


def step_to_threshold(series: list[tuple[int, float]], threshold: float) -> int | None:
    for step, value in series:
        if value >= threshold:
            return step
    return None


def rollout_step(path: Path) -> int | None:
    match = re.match(r"step_(\d+)\.jsonl$", path.name)
    if not match:
        return None
    return int(match.group(1))


def select_rollout_file(exp_dir: Path, max_step: int | None) -> Path | None:
    rollout_dir = exp_dir / "rollouts"
    if not rollout_dir.exists():
        return None
    candidates: list[tuple[int, Path]] = []
    for path in rollout_dir.glob("step_*.jsonl"):
        step = rollout_step(path)
        if step is None:
            continue
        if max_step is not None and step > max_step:
            continue
        candidates.append((step, path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_rollout(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}

    rewards: list[float] = []
    response_words: list[float] = []
    cot_records: list[dict[str, Any]] = []
    problem_counts: dict[str, int] = {}

    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            reward = as_float(row.get("reward"))
            if reward is not None:
                rewards.append(reward)
            response_words.append(float(len(str(row.get("response") or "").split())))
            problem_type = str(row.get("problem_type") or "unknown")
            problem_counts[problem_type] = problem_counts.get(problem_type, 0) + 1
            debug = row.get("cot_budget_debug")
            if isinstance(debug, dict) and debug.get("cot_budget_enabled"):
                cot_records.append(debug)

    summary: dict[str, Any] = {
        "rollout_step": rollout_step(path),
        "rollout_records": sum(problem_counts.values()),
        "rollout_reward_mean": _mean(rewards),
        "rollout_response_words_mean": _mean(response_words),
        "rollout_problem_types": problem_counts,
    }
    if cot_records:
        summary.update(
            {
                "cot_start_ratio": _mean(
                    [1.0 if item.get("cot_start_detected") else 0.0 for item in cot_records]
                ),
                "cot_end_ratio": _mean(
                    [1.0 if item.get("cot_end_detected") else 0.0 for item in cot_records]
                ),
                "cot_repaired_ratio": _mean(
                    [1.0 if item.get("cot_repaired") else 0.0 for item in cot_records]
                ),
                "cot_text_fallback_ratio": _mean(
                    [1.0 if item.get("cot_text_fallback_used") else 0.0 for item in cot_records]
                ),
                "cot_remaining_tokens_mean": _mean(
                    [
                        value
                        for value in (as_float(item.get("remaining_tokens")) for item in cot_records)
                        if value is not None
                    ]
                ),
                "cot_final_token_len_mean": _mean(
                    [
                        value
                        for value in (as_float(item.get("final_token_len")) for item in cot_records)
                        if value is not None
                    ]
                ),
            }
        )
    return summary


def infer_family_mode(label: str, exp_dir: Path) -> tuple[str, str]:
    text = f"{label} {exp_dir.name}".lower()
    family = "unknown"
    for candidate in ("aot", "seg", "logic"):
        if candidate in text:
            family = candidate
            break
    mode = "cot" if "cot" in text and "nocot" not in text else "nocot"
    return family, mode


def summarize_experiment(
    exp_dir: Path,
    *,
    label: str | None = None,
    max_step: int | None = None,
    train_key: str = DEFAULT_TRAIN_KEY,
    val_key: str = DEFAULT_VAL_KEY,
) -> dict[str, Any]:
    label = label or exp_dir.name
    family, mode = infer_family_mode(label, exp_dir)
    rows = read_log(exp_dir)
    train = metric_series(rows, train_key, max_step=max_step)
    val = metric_series(rows, val_key, max_step=max_step)

    summary: dict[str, Any] = {
        "label": label,
        "path": str(exp_dir),
        "family": family,
        "mode": mode,
        "train_key": train_key,
        "val_key": val_key,
        "train_first": train[0][1] if train else None,
        "train_final": train[-1][1] if train else None,
        "train_delta": delta(train),
        "train_slope": slope(train),
        "val_first": val[0][1] if val else None,
        "val_final": val[-1][1] if val else None,
        "val_delta": delta(val),
        "val_slope": slope(val),
    }
    for threshold in THRESHOLDS:
        summary[f"step_to_train_{threshold}"] = step_to_threshold(train, threshold)
        summary[f"step_to_val_{threshold}"] = step_to_threshold(val, threshold)

    for log_key, summary_key in LOG_COT_KEYS.items():
        summary[summary_key] = latest_metric(rows, log_key, max_step=max_step)

    task_scores, task_counts = collect_val_task_scores(rows, max_step=max_step)
    summary["base_tg_val_final"] = task_scores.get("temporal_grounding")
    summary["base_mcq_val_final"] = task_scores.get("llava_mcq")
    summary["base_val_final"] = weighted_mean_for_tasks(
        task_scores,
        task_counts,
        lambda task: task in {"temporal_grounding", "llava_mcq"},
    )
    summary["proxy_val_final"] = weighted_mean_for_tasks(
        task_scores,
        task_counts,
        lambda task: is_proxy_task(family, task),
    )
    summary["val_task_finals"] = task_scores

    summary.update(summarize_rollout(select_rollout_file(exp_dir, max_step)))
    return summary


def add_pairwise_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baselines = {
        row.get("family"): row
        for row in rows
        if row.get("mode") == "nocot" and row.get("family") != "unknown"
    }
    out: list[dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        baseline = baselines.get(row.get("family"))
        if baseline is not None and row.get("mode") == "cot":
            for key in ("train_final", "val_final", "train_delta", "val_delta", "base_val_final", "proxy_val_final"):
                current = as_float(row.get(key))
                base = as_float(baseline.get(key))
                if current is not None and base is not None:
                    enriched[f"delta_vs_nocot_{key}"] = current - base
        out.append(enriched)
    return out


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def print_markdown(rows: list[dict[str, Any]]) -> None:
    keys = [
        "label",
        "family",
        "mode",
        "train_final",
        "train_delta",
        "val_final",
        "val_delta",
        "delta_vs_nocot_val_final",
        "base_val_final",
        "proxy_val_final",
        "delta_vs_nocot_proxy_val_final",
        "step_to_train_0.5",
        "step_to_val_0.5",
        "log_cot_end_ratio",
        "log_cot_repaired_ratio",
        "log_cot_final_token_len_mean",
        "cot_end_ratio",
        "cot_repaired_ratio",
        "cot_final_token_len_mean",
        "rollout_reward_mean",
    ]
    print("| " + " | ".join(keys) + " |")
    print("| " + " | ".join(["---"] * len(keys)) + " |")
    for row in rows:
        print("| " + " | ".join(format_value(row.get(key)) for key in keys) + " |")


def parse_exp_arg(raw: str) -> tuple[str | None, Path]:
    if "=" in raw:
        label, path = raw.split("=", 1)
        return label, Path(path)
    return None, Path(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiments", nargs="+", help="Experiment dirs, or label=/path/to/dir")
    parser.add_argument("--max-step", type=int, default=100)
    parser.add_argument("--train-key", default=DEFAULT_TRAIN_KEY)
    parser.add_argument("--val-key", default=DEFAULT_VAL_KEY)
    parser.add_argument("--format", choices=("markdown", "jsonl"), default="markdown")
    args = parser.parse_args()

    rows = []
    for item in args.experiments:
        label, exp_dir = parse_exp_arg(item)
        rows.append(
            summarize_experiment(
                exp_dir,
                label=label,
                max_step=args.max_step,
                train_key=args.train_key,
                val_key=args.val_key,
            )
        )
    rows = add_pairwise_deltas(rows)

    if args.format == "jsonl":
        for row in rows:
            print(json.dumps(row, ensure_ascii=False, sort_keys=True))
    else:
        print_markdown(rows)


if __name__ == "__main__":
    main()
