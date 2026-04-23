#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("XDG_CACHE_HOME", str(Path("/tmp") / "codex-xdg-cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "codex-matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


DEFAULT_DATA_ROOT = Path("/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/multi_task")
DEFAULT_CHECKPOINT_ROOT = Path("/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/multi_task")
DEFAULT_OUTPUT_ROOT = Path("data_analysis/outputs/event_logic_ablations")
DEFAULT_BATCH_SIZE = 16
DEFAULT_SEED = 1


@dataclass(frozen=True)
class ExperimentSpec:
    key: str
    exp_name: str
    event_problem_type: str
    label: str


EXPERIMENTS = {
    "PN": ExperimentSpec(
        key="PN",
        exp_name="el_ablation_predict_next",
        event_problem_type="event_logic_predict_next",
        label="Predict Next",
    ),
    "FB": ExperimentSpec(
        key="FB",
        exp_name="el_ablation_fill_blank",
        event_problem_type="event_logic_fill_blank",
        label="Fill Blank",
    ),
    "SORT": ExperimentSpec(
        key="SORT",
        exp_name="el_ablation_sort",
        event_problem_type="event_logic_sort",
        label="Sort",
    ),
}


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "grid.linestyle": "--",
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze planned-vs-actual task ratios for event logic ablation experiments."
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["PN", "FB", "SORT"],
        help="Experiment keys to analyze. Choices: PN FB SORT",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory containing per-experiment train.jsonl / val.jsonl",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=DEFAULT_CHECKPOINT_ROOT,
        help="Root directory containing per-experiment checkpoints, experiment_log.jsonl, and rollouts/",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where analysis artifacts will be written.",
    )
    parser.add_argument(
        "--batch-size-default",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Fallback batch size when experiment_config.json is missing.",
    )
    parser.add_argument(
        "--seed-default",
        type=int,
        default=DEFAULT_SEED,
        help="Fallback data seed when experiment_config.json is missing.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=0,
        help="Rolling window size for actual task ratio plot. 0 means auto.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any requested experiment is missing required inputs.",
    )
    return parser.parse_args()


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Failed to parse {path}:{line_no}: {exc}") from exc


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def flatten_items(obj: Any, prefix: str = ""):
    if isinstance(obj, dict):
        for key, value in obj.items():
            next_prefix = f"{prefix}/{key}" if prefix else str(key)
            yield from flatten_items(value, next_prefix)
    else:
        yield prefix, obj


def safe_pct(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def normalize_counter(counter: Counter[str]) -> dict[str, float]:
    total = sum(counter.values())
    return {key: safe_pct(value, total) for key, value in sorted(counter.items())}


def write_rows_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def counter_rows(counter: Counter[str], ratio_map: dict[str, float], label: str) -> list[dict[str, Any]]:
    rows = []
    total = sum(counter.values())
    for task in sorted(counter):
        rows.append(
            {
                "task": task,
                "count": counter[task],
                "ratio": ratio_map.get(task, 0.0),
                "total": total,
                "source": label,
            }
        )
    return rows


def plot_pie(counter: Counter[str], title: str, output_path: Path) -> None:
    if not counter:
        output_path.unlink(missing_ok=True)
        return
    labels = [f"{task} ({count})" for task, count in counter.most_common()]
    values = [count for _, count in counter.most_common()]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90, counterclock=False)
    ax.set_title(title)
    fig.savefig(output_path)
    plt.close(fig)


def plot_grouped_ratio_bars(
    ratio_series: list[tuple[str, dict[str, float]]],
    title: str,
    output_path: Path,
    category_order: list[str] | None = None,
) -> None:
    categories = set()
    for _, ratios in ratio_series:
        categories.update(ratios.keys())
    if not categories:
        output_path.unlink(missing_ok=True)
        return
    categories = sorted(categories) if category_order is None else category_order

    width = 0.8 / max(1, len(ratio_series))
    positions = list(range(len(categories)))
    fig, ax = plt.subplots(figsize=(max(8, len(categories) * 1.3), 5))
    for idx, (label, ratios) in enumerate(ratio_series):
        xs = [pos - 0.4 + width / 2 + idx * width for pos in positions]
        ys = [ratios.get(task, 0.0) for task in categories]
        ax.bar(xs, ys, width=width, label=label)
    ax.set_xticks(positions)
    ax.set_xticklabels(categories, rotation=25, ha="right")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("Share")
    ax.set_title(title)
    ax.legend()
    fig.savefig(output_path)
    plt.close(fig)


def plot_rolling_task_share(
    step_sequence: list[tuple[int, str]],
    output_path: Path,
    title: str,
    window_size: int = 0,
) -> None:
    if not step_sequence:
        output_path.unlink(missing_ok=True)
        return

    ordered_steps = [step for step, _ in step_sequence]
    ordered_tasks = [task for _, task in step_sequence]
    tasks = sorted(set(ordered_tasks))
    if window_size <= 0:
        window_size = max(5, int(math.sqrt(len(step_sequence))))
    window_size = min(window_size, len(step_sequence))

    ys: dict[str, list[float]] = {task: [] for task in tasks}
    for idx in range(len(step_sequence)):
        window = ordered_tasks[max(0, idx - window_size + 1) : idx + 1]
        counts = Counter(window)
        total = len(window)
        for task in tasks:
            ys[task].append(safe_pct(counts[task], total))

    fig, ax = plt.subplots(figsize=(9, 5))
    for task in tasks:
        ax.plot(ordered_steps, ys[task], label=task, linewidth=2.0)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Train step")
    ax.set_ylabel(f"Rolling share (window={window_size})")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2, loc="best")
    fig.savefig(output_path)
    plt.close(fig)


def plot_val_lines(val_series: dict[str, list[tuple[int, float]]], output_path: Path, title: str) -> None:
    if not val_series:
        output_path.unlink(missing_ok=True)
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for task in sorted(val_series):
        points = sorted(val_series[task])
        xs = [step for step, _ in points]
        ys = [value for _, value in points]
        ax.plot(xs, ys, marker="o", linewidth=2.0, markersize=3, label=task)
    ax.set_xlabel("Validation step")
    ax.set_ylabel("Validation score")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2, loc="best")
    fig.savefig(output_path)
    plt.close(fig)


def count_problem_types(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in iter_jsonl(path):
        task = str(record.get("problem_type") or "unknown")
        counts[task] += 1
    return counts


def infer_task_weights(train_counts: Counter[str], config: dict[str, Any] | None) -> dict[str, float]:
    if config:
        raw_weights = (config.get("data") or {}).get("task_weights")
        if isinstance(raw_weights, dict) and raw_weights:
            weights = {str(task): float(weight) for task, weight in raw_weights.items() if task in train_counts}
            if weights:
                total = sum(weights.values())
                return {task: safe_pct(weight, total) for task, weight in sorted(weights.items())}
    total = sum(train_counts.values())
    return {task: safe_pct(count, total) for task, count in sorted(train_counts.items())}


def infer_batch_size(config: dict[str, Any] | None, fallback: int) -> int:
    if not config:
        return fallback
    data_cfg = config.get("data") or {}
    batch_size = data_cfg.get("mini_rollout_batch_size") or data_cfg.get("rollout_batch_size")
    return int(batch_size or fallback)


def infer_seed(config: dict[str, Any] | None, fallback: int) -> int:
    if not config:
        return fallback
    data_cfg = config.get("data") or {}
    return int(data_cfg.get("seed", fallback))


def compute_sampler_batch_counts(
    train_counts: Counter[str],
    batch_size: int,
    task_weights: dict[str, float],
) -> dict[str, int]:
    total_batches = sum(count // batch_size for count in train_counts.values())
    task_batch_counts: dict[str, int] = {}
    for task in sorted(train_counts):
        max_batches = train_counts[task] // batch_size
        ideal = int(total_batches * task_weights.get(task, 0.0))
        task_batch_counts[task] = min(max(1, ideal), max_batches) if max_batches > 0 else 0
    return task_batch_counts


def simulate_sampler_step_sequence(
    task_batch_counts: dict[str, int],
    seed: int,
    total_steps: int,
) -> list[str]:
    if total_steps <= 0:
        return []
    base_epoch_steps = sum(task_batch_counts.values())
    if base_epoch_steps == 0:
        return []

    generated: list[str] = []
    epoch_seed = seed
    while len(generated) < total_steps:
        rng = random.Random(epoch_seed)
        positioned: list[tuple[float, str]] = []
        total = sum(task_batch_counts.values())
        for task, count in sorted(task_batch_counts.items()):
            if count <= 0:
                continue
            stride = total / count
            for idx in range(count):
                position = idx * stride + rng.uniform(0.0, stride * 0.3)
                positioned.append((position, task))
        positioned.sort(key=lambda item: item[0])
        generated.extend(task for _, task in positioned)
        epoch_seed += 1
    return generated[:total_steps]


def parse_step_from_name(path: Path, fallback: int = 0) -> int:
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else fallback


def summarize_train_rollouts(rollout_dir: Path) -> dict[str, Any]:
    sample_counts: Counter[str] = Counter()
    per_step_task_counts: dict[int, Counter[str]] = defaultdict(Counter)

    for path in sorted(rollout_dir.glob("step_*.jsonl")):
        default_step = parse_step_from_name(path)
        for record in iter_jsonl(path):
            step = int(record.get("step") or default_step)
            task = str(record.get("problem_type") or "unknown")
            sample_counts[task] += 1
            per_step_task_counts[step][task] += 1

    step_sequence: list[tuple[int, str]] = []
    step_counts: Counter[str] = Counter()
    impure_steps: list[dict[str, Any]] = []
    for step in sorted(per_step_task_counts):
        counter = per_step_task_counts[step]
        top_task, top_count = counter.most_common(1)[0]
        purity = safe_pct(top_count, sum(counter.values()))
        step_sequence.append((step, top_task))
        step_counts[top_task] += 1
        if len(counter) > 1:
            impure_steps.append(
                {
                    "step": step,
                    "purity": purity,
                    "task_counts": dict(counter),
                }
            )

    return {
        "sample_counts": sample_counts,
        "sample_ratios": normalize_counter(sample_counts),
        "step_counts": step_counts,
        "step_ratios": normalize_counter(step_counts),
        "step_sequence": step_sequence,
        "impure_steps": impure_steps,
    }


def summarize_val_from_log(log_path: Path) -> dict[str, list[tuple[int, float]]]:
    val_series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for record in iter_jsonl(log_path):
        step = int(record.get("step", 0))
        task_metrics: dict[str, dict[str, float]] = defaultdict(dict)
        for path, value in flatten_items(record):
            parts = path.split("/")
            if len(parts) < 3 or parts[0] != "val":
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            task = parts[1]
            metric = "/".join(parts[2:])
            task_metrics[task][metric] = float(value)
        for task, metrics in task_metrics.items():
            chosen = choose_primary_val_metric(metrics)
            if chosen is not None:
                val_series[task].append((step, chosen))
    return dict(val_series)


def summarize_val_from_rollouts(rollout_dir: Path) -> dict[str, list[tuple[int, float]]]:
    per_step_task_values: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(rollout_dir.glob("val_step_*.jsonl")):
        default_step = parse_step_from_name(path)
        for record in iter_jsonl(path):
            step = int(record.get("step") or default_step)
            task = str(record.get("problem_type") or "unknown")
            reward = record.get("reward")
            if isinstance(reward, dict):
                reward = reward.get("overall", reward.get("reward"))
            if reward is None:
                continue
            per_step_task_values[step][task].append(float(reward))

    val_series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for step in sorted(per_step_task_values):
        for task, values in per_step_task_values[step].items():
            if values:
                val_series[task].append((step, sum(values) / len(values)))
    return dict(val_series)


def choose_primary_val_metric(metrics: dict[str, float]) -> float | None:
    priority = [
        "overall_reward",
        "accuracy_reward",
        "reward",
        "score_reward",
        "structure_reward",
        "format_reward",
    ]
    for key in priority:
        if key in metrics:
            return metrics[key]
    reward_keys = [key for key in sorted(metrics) if key.endswith("_reward")]
    for key in reward_keys:
        if key != "count":
            return metrics[key]
    return None


def write_summary_markdown(
    output_path: Path,
    spec: ExperimentSpec,
    paths: dict[str, Path],
    train_counts: Counter[str] | None,
    val_counts: Counter[str] | None,
    task_weights: dict[str, float] | None,
    sampler_batch_counts: dict[str, int] | None,
    planned_prefix_ratios: dict[str, float] | None,
    rollout_summary: dict[str, Any] | None,
    val_series: dict[str, list[tuple[int, float]]] | None,
    missing: list[str],
) -> None:
    lines: list[str] = []
    lines.append(f"# {spec.exp_name}")
    lines.append("")
    lines.append(f"- Experiment key: `{spec.key}`")
    lines.append(f"- Event problem_type: `{spec.event_problem_type}`")
    lines.append(f"- Data dir: `{paths['data_dir']}`")
    lines.append(f"- Checkpoint dir: `{paths['checkpoint_dir']}`")
    lines.append("")

    if missing:
        lines.append("## Missing Inputs")
        lines.append("")
        for item in missing:
            lines.append(f"- {item}")
        lines.append("")

    if train_counts:
        lines.append("## Planned Dataset Mix")
        lines.append("")
        total = sum(train_counts.values())
        for task, count in train_counts.most_common():
            lines.append(f"- `{task}`: {count} ({safe_pct(count, total):.2%})")
        lines.append("")

    if task_weights:
        lines.append("## Sampler Plan")
        lines.append("")
        for task in sorted(task_weights):
            weight = task_weights[task]
            batch_count = sampler_batch_counts.get(task, 0) if sampler_batch_counts else 0
            prefix_ratio = planned_prefix_ratios.get(task, 0.0) if planned_prefix_ratios else 0.0
            lines.append(
                f"- `{task}`: weight={weight:.4f}, epoch_batches={batch_count}, "
                f"prefix_share={prefix_ratio:.2%}"
            )
        lines.append("")

    if rollout_summary:
        lines.append("## Actual Training Mix")
        lines.append("")
        for task, count in rollout_summary["step_counts"].most_common():
            lines.append(
                f"- `{task}`: steps={count} ({rollout_summary['step_ratios'].get(task, 0.0):.2%}), "
                f"samples={rollout_summary['sample_counts'].get(task, 0)} "
                f"({rollout_summary['sample_ratios'].get(task, 0.0):.2%})"
            )
        if rollout_summary["impure_steps"]:
            lines.append("")
            lines.append(f"- Mixed-task train steps detected: {len(rollout_summary['impure_steps'])}")
        lines.append("")

    if val_counts:
        lines.append("## Validation Dataset Mix")
        lines.append("")
        total = sum(val_counts.values())
        for task, count in val_counts.most_common():
            lines.append(f"- `{task}`: {count} ({safe_pct(count, total):.2%})")
        lines.append("")

    if val_series:
        lines.append("## Validation Scores")
        lines.append("")
        for task in sorted(val_series):
            final_step, final_score = sorted(val_series[task])[-1]
            lines.append(f"- `{task}`: final={final_score:.4f} at step {final_step}")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def analyze_experiment(
    spec: ExperimentSpec,
    data_root: Path,
    checkpoint_root: Path,
    output_root: Path,
    batch_size_default: int,
    seed_default: int,
    rolling_window: int,
) -> dict[str, Any]:
    data_dir = data_root / spec.exp_name
    checkpoint_dir = checkpoint_root / spec.exp_name
    rollout_dir = checkpoint_dir / "rollouts"
    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"
    config_path = checkpoint_dir / "experiment_config.json"
    log_path = checkpoint_dir / "experiment_log.jsonl"
    experiment_output_dir = output_root / spec.exp_name
    experiment_output_dir.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []
    for path in [train_path, val_path]:
        if not path.exists():
            missing.append(f"Missing data file: {path}")
    if not checkpoint_dir.exists():
        missing.append(f"Missing checkpoint dir: {checkpoint_dir}")

    config = read_json(config_path) if config_path.exists() else None
    train_counts = count_problem_types(train_path) if train_path.exists() else Counter()
    val_counts = count_problem_types(val_path) if val_path.exists() else Counter()

    task_weights = infer_task_weights(train_counts, config) if train_counts else {}
    batch_size = infer_batch_size(config, batch_size_default)
    seed = infer_seed(config, seed_default)
    sampler_batch_counts = (
        compute_sampler_batch_counts(train_counts, batch_size, task_weights) if train_counts else {}
    )
    sampler_epoch_ratios = normalize_counter(Counter(sampler_batch_counts))

    rollout_summary = summarize_train_rollouts(rollout_dir) if rollout_dir.exists() else None
    actual_train_steps = len(rollout_summary["step_sequence"]) if rollout_summary else 0
    planned_prefix_sequence = (
        simulate_sampler_step_sequence(sampler_batch_counts, seed, actual_train_steps)
        if actual_train_steps > 0
        else []
    )
    planned_prefix_ratios = normalize_counter(Counter(planned_prefix_sequence)) if planned_prefix_sequence else {}

    val_series = {}
    if log_path.exists():
        val_series = summarize_val_from_log(log_path)
    elif rollout_dir.exists():
        val_series = summarize_val_from_rollouts(rollout_dir)

    write_rows_csv(
        experiment_output_dir / "train_problem_type_counts.csv",
        ["task", "count", "ratio", "total", "source"],
        counter_rows(train_counts, normalize_counter(train_counts), "train"),
    )
    write_rows_csv(
        experiment_output_dir / "val_problem_type_counts.csv",
        ["task", "count", "ratio", "total", "source"],
        counter_rows(val_counts, normalize_counter(val_counts), "val"),
    )
    write_rows_csv(
        experiment_output_dir / "sampler_plan.csv",
        ["task", "weight", "epoch_batch_count", "epoch_ratio", "planned_prefix_ratio", "batch_size", "seed"],
        [
            {
                "task": task,
                "weight": task_weights.get(task, 0.0),
                "epoch_batch_count": sampler_batch_counts.get(task, 0),
                "epoch_ratio": sampler_epoch_ratios.get(task, 0.0),
                "planned_prefix_ratio": planned_prefix_ratios.get(task, 0.0),
                "batch_size": batch_size,
                "seed": seed,
            }
            for task in sorted(task_weights)
        ],
    )
    if rollout_summary:
        write_rows_csv(
            experiment_output_dir / "actual_train_step_counts.csv",
            ["task", "count", "ratio", "total", "source"],
            counter_rows(rollout_summary["step_counts"], rollout_summary["step_ratios"], "actual_steps"),
        )
        write_rows_csv(
            experiment_output_dir / "actual_train_sample_counts.csv",
            ["task", "count", "ratio", "total", "source"],
            counter_rows(rollout_summary["sample_counts"], rollout_summary["sample_ratios"], "actual_samples"),
        )
        write_rows_csv(
            experiment_output_dir / "actual_train_step_sequence.csv",
            ["step", "task"],
            [{"step": step, "task": task} for step, task in rollout_summary["step_sequence"]],
        )

    if val_series:
        val_rows: list[dict[str, Any]] = []
        for task in sorted(val_series):
            for step, score in sorted(val_series[task]):
                val_rows.append({"task": task, "step": step, "score": score})
        write_rows_csv(experiment_output_dir / "val_scores.csv", ["task", "step", "score"], val_rows)

    plot_pie(train_counts, f"{spec.label}: planned train mix", experiment_output_dir / "train_mix_pie.png")
    plot_pie(val_counts, f"{spec.label}: val mix", experiment_output_dir / "val_mix_pie.png")
    ratio_series = [
        ("dataset share", normalize_counter(train_counts)),
        ("sampler epoch share", sampler_epoch_ratios),
    ]
    if planned_prefix_ratios:
        ratio_series.append(("sampler prefix share", planned_prefix_ratios))
    if rollout_summary:
        ratio_series.append(("actual step share", rollout_summary["step_ratios"]))
        ratio_series.append(("actual sample share", rollout_summary["sample_ratios"]))
    plot_grouped_ratio_bars(
        ratio_series,
        f"{spec.label}: planned vs actual train mix",
        experiment_output_dir / "planned_vs_actual_train_mix.png",
    )
    if rollout_summary:
        plot_rolling_task_share(
            rollout_summary["step_sequence"],
            experiment_output_dir / "actual_train_mix_over_steps.png",
            f"{spec.label}: actual train mix over steps",
            window_size=rolling_window,
        )
    if val_series:
        plot_val_lines(
            val_series,
            experiment_output_dir / "val_score_by_problem_type.png",
            f"{spec.label}: val score by problem_type",
        )

    write_summary_markdown(
        experiment_output_dir / "summary.md",
        spec,
        {
            "data_dir": data_dir,
            "checkpoint_dir": checkpoint_dir,
        },
        train_counts if train_counts else None,
        val_counts if val_counts else None,
        task_weights if task_weights else None,
        sampler_batch_counts if sampler_batch_counts else None,
        planned_prefix_ratios if planned_prefix_ratios else None,
        rollout_summary,
        val_series if val_series else None,
        missing,
    )

    final_event_val = None
    if spec.event_problem_type in val_series and val_series[spec.event_problem_type]:
        _, final_event_val = sorted(val_series[spec.event_problem_type])[-1]

    return {
        "spec": spec,
        "data_dir": data_dir,
        "checkpoint_dir": checkpoint_dir,
        "output_dir": experiment_output_dir,
        "missing": missing,
        "train_counts": train_counts,
        "train_ratios": normalize_counter(train_counts),
        "task_weights": task_weights,
        "sampler_epoch_ratios": sampler_epoch_ratios,
        "planned_prefix_ratios": planned_prefix_ratios,
        "rollout_summary": rollout_summary,
        "final_event_val": final_event_val,
    }


def write_overview(output_root: Path, results: list[dict[str, Any]]) -> None:
    if not results:
        return

    overview_rows: list[dict[str, Any]] = []

    event_share_rows: list[dict[str, Any]] = []
    for result in results:
        spec: ExperimentSpec = result["spec"]
        event_task = spec.event_problem_type
        planned_dataset = None
        if result["train_counts"]:
            planned_dataset = result["train_ratios"].get(event_task, 0.0)

        planned_sampler = None
        if result["task_weights"]:
            planned_sampler = result["planned_prefix_ratios"].get(event_task)
            if planned_sampler is None:
                planned_sampler = result["sampler_epoch_ratios"].get(event_task, 0.0)

        actual_steps = None
        actual_samples = None
        rollout_summary = result["rollout_summary"]
        if rollout_summary:
            actual_steps = rollout_summary["step_ratios"].get(event_task, 0.0)
            actual_samples = rollout_summary["sample_ratios"].get(event_task, 0.0)

        overview_rows.append(
            {
                "experiment": spec.exp_name,
                "label": spec.label,
                "event_problem_type": event_task,
                "planned_dataset_share": planned_dataset,
                "planned_sampler_share": planned_sampler,
                "actual_step_share": actual_steps,
                "actual_sample_share": actual_samples,
                "final_event_val": result["final_event_val"],
                "missing_count": len(result["missing"]),
            }
        )

        for series, share in [
            ("dataset", planned_dataset),
            ("sampler", planned_sampler),
            ("actual_steps", actual_steps),
            ("actual_samples", actual_samples),
        ]:
            if share is not None:
                event_share_rows.append({"experiment": spec.label, "series": series, "share": share})

    write_rows_csv(
        output_root / "overview.csv",
        [
            "experiment",
            "label",
            "event_problem_type",
            "planned_dataset_share",
            "planned_sampler_share",
            "actual_step_share",
            "actual_sample_share",
            "final_event_val",
            "missing_count",
        ],
        overview_rows,
    )
    write_rows_csv(output_root / "event_share_overview.csv", ["experiment", "series", "share"], event_share_rows)

    ratio_plot_rows = []
    for row in overview_rows:
        ratios = {}
        if row["planned_dataset_share"] is not None:
            ratios["planned_dataset_share"] = row["planned_dataset_share"]
        if row["planned_sampler_share"] is not None:
            ratios["planned_sampler_share"] = row["planned_sampler_share"]
        if row["actual_step_share"] is not None:
            ratios["actual_step_share"] = row["actual_step_share"]
        if row["actual_sample_share"] is not None:
            ratios["actual_sample_share"] = row["actual_sample_share"]
        if ratios:
            ratio_plot_rows.append((row["label"], ratios))

    if ratio_plot_rows:
        plot_grouped_ratio_bars(
            ratio_plot_rows,
            "Event task share across experiments",
            output_root / "event_share_overview.png",
            category_order=[
                "planned_dataset_share",
                "planned_sampler_share",
                "actual_step_share",
                "actual_sample_share",
            ],
        )
    else:
        (output_root / "event_share_overview.png").unlink(missing_ok=True)

    lines = ["# Event Logic Ablation Overview", ""]
    for row in overview_rows:
        lines.append(
            f"- `{row['experiment']}` ({row['event_problem_type']}): "
            f"dataset={fmt_pct(row['planned_dataset_share'])}, "
            f"sampler={fmt_pct(row['planned_sampler_share'])}, "
            f"actual_steps={fmt_pct(row['actual_step_share'])}, "
            f"actual_samples={fmt_pct(row['actual_sample_share'])}, "
            f"final_event_val={row['final_event_val'] if row['final_event_val'] is not None else 'NA'}"
        )
    (output_root / "overview.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.2%}"


def main() -> None:
    args = parse_args()
    selected_specs: list[ExperimentSpec] = []
    unknown = [name for name in args.experiments if name not in EXPERIMENTS]
    if unknown:
        raise SystemExit(f"Unknown experiment keys: {unknown}. Choices: {sorted(EXPERIMENTS)}")
    for key in args.experiments:
        selected_specs.append(EXPERIMENTS[key])

    args.output_root.mkdir(parents=True, exist_ok=True)

    results = []
    for spec in selected_specs:
        result = analyze_experiment(
            spec,
            data_root=args.data_root,
            checkpoint_root=args.checkpoint_root,
            output_root=args.output_root,
            batch_size_default=args.batch_size_default,
            seed_default=args.seed_default,
            rolling_window=args.rolling_window,
        )
        results.append(result)

    write_overview(args.output_root, results)

    missing_count = sum(1 for result in results if result["missing"])
    print(f"[event-analysis] analyzed {len(results)} experiments -> {args.output_root}")
    if missing_count:
        print(f"[event-analysis] experiments with missing inputs: {missing_count}")
        for result in results:
            if result["missing"]:
                print(f"  - {result['spec'].exp_name}")
                for item in result["missing"]:
                    print(f"      {item}")
        if args.strict:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
