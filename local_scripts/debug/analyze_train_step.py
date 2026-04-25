#!/usr/bin/env python3
"""Analyze training logs and rollouts from an EasyR1 experiment directory.

Reads:
  - experiment_log.jsonl
  - rollouts/step_XXXXXX.jsonl over a step range

The script is intentionally read-only. It summarizes KL/reward trends and
rollout diversity so low-frame ablations can be judged quickly.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TG_PATTERNS = (
    re.compile(
        r"The\s+event(?:\s+['\"].*?['\"])?\s+happens\s+in\s+(?:the\s+)?"
        r"([0-9]*\.?[0-9]+)\s*-\s*([0-9]*\.?[0-9]+)\s*seconds?",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(r"([0-9]*\.?[0-9]+)\s*-\s*([0-9]*\.?[0-9]+)\s*seconds?", re.IGNORECASE),
    re.compile(r"\[\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\]"),
)

WATCH_KEYS = (
    "actor/kl_loss",
    "actor/ppo_kl",
    "actor/entropy_loss",
    "actor/kl_coef",
    "reward/temporal_grounding",
    "reward_filtered/temporal_grounding",
    "reward/overall",
    "reward_filtered/overall",
    "debug/online_filtering_keep_ratio",
    "response_length/mean",
    "response_length/max",
    "response_length/clip_ratio",
)


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
        if math.isfinite(result):
            return result
    except (TypeError, ValueError):
        return None
    return None


def read_log(exp_dir: Path) -> list[dict[str, Any]]:
    path = exp_dir / "experiment_log.jsonl"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def scalar_at_or_before(rows: list[dict[str, Any]], step: int, key: str) -> tuple[int, float] | None:
    best: tuple[int, float] | None = None
    for row in rows:
        row_step = int(row.get("step", -1))
        if row_step > step:
            continue
        value = as_float(flatten(row).get(key))
        if value is not None:
            best = (row_step, value)
    return best


def series(
    rows: list[dict[str, Any]],
    key: str,
    min_step: int | None = None,
    max_step: int | None = None,
) -> list[tuple[int, float]]:
    values: list[tuple[int, float]] = []
    for row in rows:
        step = int(row.get("step", -1))
        if min_step is not None and step < min_step:
            continue
        if max_step is not None and step > max_step:
            continue
        value = as_float(flatten(row).get(key))
        if value is not None:
            values.append((step, value))
    return values


def slope(values: list[tuple[int, float]], window: int = 20) -> float | None:
    tail = values[-window:]
    if len(tail) < 2:
        return None
    x0, y0 = tail[0]
    x1, y1 = tail[-1]
    if x1 == x0:
        return None
    return (y1 - y0) / (x1 - x0)


def normalize_response(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def parse_tg_pair(text: str) -> tuple[float, float] | None:
    for pattern in TG_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        start, end = float(match.group(1)), float(match.group(2))
        if start >= 0 and end > start:
            return (round(start, 2), round(end, 2))
    return None


def rollout_path(exp_dir: Path, step: int) -> Path:
    return exp_dir / "rollouts" / f"step_{step:06d}.jsonl"


def read_rollouts(exp_dir: Path, step: int) -> list[dict[str, Any]]:
    path = rollout_path(exp_dir, step)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_rollout_step(path: Path) -> int | None:
    match = re.match(r"step_(\d+)\.jsonl$", path.name)
    if not match:
        return None
    return int(match.group(1))


def available_rollout_steps(exp_dir: Path, start: int, end: int) -> list[int]:
    rollout_dir = exp_dir / "rollouts"
    if not rollout_dir.exists():
        return []
    steps = []
    for path in rollout_dir.glob("step_*.jsonl"):
        step = parse_rollout_step(path)
        if step is not None and start <= step <= end:
            steps.append(step)
    return sorted(set(steps))


def select_rollout_steps(
    exp_dir: Path,
    start: int,
    end: int,
    stride: int,
    explicit_steps: str | None,
) -> list[int]:
    available = available_rollout_steps(exp_dir, start, end)
    if explicit_steps:
        wanted = sorted({int(part) for part in explicit_steps.split(",") if part.strip()})
        return [step for step in wanted if step in set(available)]
    if not available:
        return []
    if stride <= 1:
        return available
    selected = [step for step in available if (step - start) % stride == 0]
    if end in available and end not in selected:
        selected.append(end)
    if not selected:
        selected = [available[-1]]
    return sorted(set(selected))


def summarize_rollouts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_task_uid: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        uid = str(row.get("uid") or "")
        task = str(row.get("problem_type") or "unknown")
        by_task_uid[(task, uid)].append(row)

    task_summaries: dict[str, dict[str, Any]] = {}
    by_task: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for (task, _uid), group in by_task_uid.items():
        by_task[task].append(group)

    for task, groups in sorted(by_task.items()):
        unique_counts = []
        reward_stds = []
        group_sizes = []
        invalid_tg = 0
        same_groups = 0
        pair_unique_counts = []
        response_lens = []
        examples: list[dict[str, Any]] = []

        for group in groups:
            responses = [normalize_response(row.get("response")) for row in group]
            rewards = [float(row.get("reward") or 0.0) for row in group]
            unique_responses = set(responses)
            unique_counts.append(len(unique_responses))
            group_sizes.append(len(group))
            response_lens.extend(len(resp.split()) for resp in responses)
            if len(unique_responses) == 1:
                same_groups += 1
                if len(examples) < 3:
                    examples.append(
                        {
                            "uid": group[0].get("uid"),
                            "reward": rewards[0] if rewards else None,
                            "response": responses[0][:240],
                        }
                    )

            if len(rewards) > 1:
                reward_stds.append(statistics.pstdev(rewards))
            else:
                reward_stds.append(0.0)

            if task == "temporal_grounding":
                pairs = [parse_tg_pair(resp) for resp in responses]
                invalid_tg += sum(pair is None for pair in pairs)
                pair_unique_counts.append(len({pair for pair in pairs if pair is not None}))

        task_summaries[task] = {
            "groups": len(groups),
            "records": sum(group_sizes),
            "group_size_hist": dict(Counter(group_sizes)),
            "all_same": same_groups,
            "all_same_ratio": same_groups / max(len(groups), 1),
            "unique_response_hist": dict(Counter(unique_counts)),
            "reward_std_avg": statistics.mean(reward_stds) if reward_stds else 0.0,
            "reward_std_median": statistics.median(reward_stds) if reward_stds else 0.0,
            "response_words_avg": statistics.mean(response_lens) if response_lens else 0.0,
            "tg_invalid_ratio": invalid_tg / max(sum(group_sizes), 1) if task == "temporal_grounding" else None,
            "tg_unique_pair_hist": dict(Counter(pair_unique_counts)) if task == "temporal_grounding" else None,
            "same_examples": examples,
        }

    return task_summaries


def print_log_summary(exp_dir: Path, rows: list[dict[str, Any]], start: int, step: int) -> dict[str, float]:
    print(f"\n=== Scalars: {exp_dir} @ steps {start}..{step} ===")
    scalars: dict[str, float] = {}
    if not rows:
        print("experiment_log.jsonl: missing")
        return scalars

    for key in WATCH_KEYS:
        point = scalar_at_or_before(rows, step, key)
        if point is None:
            continue
        row_step, value = point
        scalars[key] = value
        values_all = series(rows, key, max_step=step)
        values_range = series(rows, key, min_step=start, max_step=step)
        key_slope20 = slope(values_all)
        key_slope_range = slope(values_range, window=len(values_range)) if values_range else None
        if values_range:
            first_step, first_value = values_range[0]
            last_step, last_value = values_range[-1]
            delta = last_value - first_value
            min_value = min(v for _, v in values_range)
            max_value = max(v for _, v in values_range)
            scalars[f"trend/{key}/slope_range"] = key_slope_range if key_slope_range is not None else 0.0
            scalars[f"trend/{key}/delta_range"] = delta
            trend = (
                f" first={first_value:.6f}@{first_step} last={last_value:.6f}@{last_step} "
                f"delta={delta:+.6f} min={min_value:.6f} max={max_value:.6f}"
            )
        else:
            trend = ""
        slope20_txt = f" slope20={key_slope20:+.5f}/step" if key_slope20 is not None else ""
        slope_range_txt = f" slope_range={key_slope_range:+.5f}/step" if key_slope_range is not None else ""
        print(f"{key:40s} step={row_step:<5d} value={value:.6f}{trend}{slope_range_txt}{slope20_txt}")
    return scalars


def print_task_summaries(summaries: dict[str, Any]) -> None:
    for task, summary in summaries.items():
        print(
            f"{task:28s} groups={summary['groups']:<4d} records={summary['records']:<4d} "
            f"all_same={summary['all_same']:<4d} ratio={summary['all_same_ratio']:.3f} "
            f"uniq_hist={summary['unique_response_hist']} "
            f"reward_std_avg={summary['reward_std_avg']:.5f} "
            f"reward_std_med={summary['reward_std_median']:.5f}"
        )
        if task == "temporal_grounding":
            print(
                f"{'':28s} tg_invalid_ratio={summary['tg_invalid_ratio']:.3f} "
                f"tg_unique_pair_hist={summary['tg_unique_pair_hist']}"
            )
        for example in summary["same_examples"]:
            print(
                f"{'':28s} same_uid={example['uid']} reward={example['reward']} "
                f"resp={example['response']!r}"
            )


def print_rollout_summary(exp_dir: Path, rows: list[dict[str, Any]], step: int) -> dict[str, Any]:
    path = rollout_path(exp_dir, step)
    print(f"\n=== Rollouts: {path} ===")
    if not rows:
        print("rollout file: missing")
        return {}
    summaries = summarize_rollouts(rows)
    print_task_summaries(summaries)
    return summaries


def print_rollout_trend(exp_dir: Path, steps: list[int]) -> dict[str, Any]:
    print(f"\n=== Rollout Diversity Trend: {exp_dir} ===")
    if not steps:
        print("no rollout files selected")
        return {}

    aggregate_rows: list[dict[str, Any]] = []
    for step in steps:
        rows = read_rollouts(exp_dir, step)
        aggregate_rows.extend(rows)
        summary = summarize_rollouts(rows).get("temporal_grounding")
        if summary is None:
            print(f"step={step:<5d} temporal_grounding: missing")
            continue
        print(
            f"step={step:<5d} tg_groups={summary['groups']:<4d} "
            f"all_same_ratio={summary['all_same_ratio']:.3f} "
            f"uniq_hist={summary['unique_response_hist']} "
            f"reward_std_avg={summary['reward_std_avg']:.5f} "
            f"invalid={summary['tg_invalid_ratio']:.3f}"
        )

    print(f"\n=== Rollout Aggregate: steps={steps[0]}..{steps[-1]} selected={len(steps)} ===")
    aggregate = summarize_rollouts(aggregate_rows)
    print_task_summaries(aggregate)
    return aggregate


def decide(scalars: dict[str, float], rollouts: dict[str, Any]) -> str:
    kl = scalars.get("actor/kl_loss")
    kl_slope = scalars.get("trend/actor/kl_loss/slope_range")
    reward_delta = scalars.get("trend/reward/overall/delta_range")
    keep_ratio = scalars.get("debug/online_filtering_keep_ratio")
    tg = rollouts.get("temporal_grounding", {})
    tg_same = float(tg.get("all_same_ratio", 0.0) or 0.0)
    tg_std = float(tg.get("reward_std_avg", 0.0) or 0.0)

    reasons = []
    severe = False
    caution = False

    if kl is not None:
        if kl >= 0.5:
            severe = True
            reasons.append(f"actor/kl_loss={kl:.3f} is high")
        elif kl >= 0.3:
            caution = True
            reasons.append(f"actor/kl_loss={kl:.3f} is elevated")
    if kl_slope is not None and kl_slope > 0.0025:
        caution = True
        reasons.append(f"actor/kl_loss slope={kl_slope:.5f}/step is still rising")
    if reward_delta is not None and reward_delta < -0.05:
        caution = True
        reasons.append(f"reward/overall delta={reward_delta:.3f} over range")

    if tg:
        if tg_same >= 0.4:
            severe = True
            reasons.append(f"TG all-same rollout ratio={tg_same:.3f}")
        elif tg_same >= 0.2:
            caution = True
            reasons.append(f"TG all-same rollout ratio={tg_same:.3f}")
        if tg_std < 0.02:
            caution = True
            reasons.append(f"TG reward std avg={tg_std:.4f} is low")

    if keep_ratio is not None and keep_ratio < 0.25:
        caution = True
        reasons.append(f"online filtering keep ratio={keep_ratio:.3f} is low")

    if severe:
        verdict = "RESTART_OR_STABLE_RERUN"
    elif caution:
        verdict = "CONTINUE_WITH_CAUTION"
    else:
        verdict = "CONTINUE"

    detail = "; ".join(reasons) if reasons else "no stop signal from checked metrics"
    return f"{verdict}: {detail}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exp_dir", nargs="+", type=Path, help="Experiment checkpoint directory")
    parser.add_argument("--start", type=int, default=1, help="First training step for trend analysis")
    parser.add_argument("--step", type=int, default=200, help="Last training step for trend analysis")
    parser.add_argument(
        "--rollout-stride",
        type=int,
        default=25,
        help="Analyze one rollout file every N steps; use 1 for every rollout file",
    )
    parser.add_argument(
        "--rollout-steps",
        default=None,
        help="Comma-separated explicit rollout steps, e.g. 50,100,150,200",
    )
    args = parser.parse_args()

    for exp_dir in args.exp_dir:
        print("\n" + "=" * 96)
        print(f"Experiment: {exp_dir}")
        log_rows = read_log(exp_dir)
        selected_steps = select_rollout_steps(
            exp_dir,
            start=args.start,
            end=args.step,
            stride=args.rollout_stride,
            explicit_steps=args.rollout_steps,
        )
        scalars = print_log_summary(exp_dir, log_rows, args.start, args.step)
        rollouts = print_rollout_trend(exp_dir, selected_steps)
        print(f"\nAUTO_DECISION: {decide(scalars, rollouts)}")


if __name__ == "__main__":
    main()
