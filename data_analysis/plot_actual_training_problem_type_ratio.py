#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("XDG_CACHE_HOME", str(Path("/tmp") / "codex-xdg-cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "codex-matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_EXPERIMENT_ROOT = Path("/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/multi_task")
DEFAULT_OUTPUT_ROOT = Path("data_analysis/outputs/actual_training_problem_type_ratio")


@dataclass(frozen=True)
class ExperimentSpec:
    key: str
    exp_name: str
    label: str


EXPERIMENTS = {
    "PN": ExperimentSpec("PN", "el_ablation_predict_next", "Predict Next"),
    "FB": ExperimentSpec("FB", "el_ablation_fill_blank", "Fill Blank"),
    "SORT": ExperimentSpec("SORT", "el_ablation_sort", "Sort"),
}


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 160,
        "savefig.bbox": "tight",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the actual training sample ratio by problem_type from saved train rollouts."
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["PN", "FB", "SORT"],
        help="Experiment keys to analyze. Choices: PN FB SORT",
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT,
        help="Root directory containing experiment folders such as el_ablation_predict_next/",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory for output charts and csv files.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any requested experiment is missing its rollout directory.",
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


def normalize(counter: Counter[str]) -> dict[str, float]:
    total = sum(counter.values())
    if total == 0:
        return {}
    return {key: counter[key] / total for key in sorted(counter)}


def count_actual_training_problem_types(rollout_dir: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for path in sorted(rollout_dir.glob("step_*.jsonl")):
        for record in iter_jsonl(path):
            task = str(record.get("problem_type") or "unknown")
            counts[task] += 1
    return counts


def write_csv(path: Path, counter: Counter[str]) -> None:
    ratios = normalize(counter)
    total = sum(counter.values())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["problem_type", "count", "ratio", "total"])
        writer.writeheader()
        for task in sorted(counter):
            writer.writerow(
                {
                    "problem_type": task,
                    "count": counter[task],
                    "ratio": ratios.get(task, 0.0),
                    "total": total,
                }
            )


def plot_pie(path: Path, title: str, counter: Counter[str]) -> None:
    if not counter:
        path.unlink(missing_ok=True)
        return
    labels = [f"{task}\n{count}" for task, count in counter.most_common()]
    values = [count for _, count in counter.most_common()]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.9, "edgecolor": "white"},
        textprops={"fontsize": 10},
    )
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_color("white")
        autotext.set_weight("bold")
    ax.set_title(title)
    fig.savefig(path)
    plt.close(fig)


def write_summary(path: Path, spec: ExperimentSpec, counter: Counter[str], rollout_dir: Path) -> None:
    ratios = normalize(counter)
    total = sum(counter.values())
    lines = [
        f"# {spec.exp_name}",
        "",
        f"- rollout dir: `{rollout_dir}`",
        f"- total saved train samples: {total}",
        "",
        "## Actual Training Sample Ratio",
        "",
    ]
    for task, count in counter.most_common():
        lines.append(f"- `{task}`: {count} ({ratios.get(task, 0.0):.2%})")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_one(spec: ExperimentSpec, experiment_root: Path, output_root: Path) -> dict[str, Any]:
    experiment_dir = experiment_root / spec.exp_name
    rollout_dir = experiment_dir / "rollouts"
    output_dir = output_root / spec.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if not rollout_dir.exists():
        return {
            "spec": spec,
            "missing": f"Missing rollout dir: {rollout_dir}",
            "counter": Counter(),
        }

    counter = count_actual_training_problem_types(rollout_dir)
    write_csv(output_dir / "actual_training_problem_type_ratio.csv", counter)
    plot_pie(output_dir / "actual_training_problem_type_ratio.png", f"{spec.label}: actual training samples", counter)
    write_summary(output_dir / "README.md", spec, counter, rollout_dir)
    return {
        "spec": spec,
        "missing": None,
        "counter": counter,
    }


def write_overview(output_root: Path, results: list[dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    lines = ["# Actual Training Sample Ratio Overview", ""]
    for result in results:
        spec: ExperimentSpec = result["spec"]
        counter: Counter[str] = result["counter"]
        missing = result["missing"]
        if missing:
            lines.append(f"- `{spec.exp_name}`: {missing}")
            continue
        total = sum(counter.values())
        major_task, major_count = counter.most_common(1)[0]
        lines.append(f"- `{spec.exp_name}`: total={total}, top=`{major_task}` ({major_count / total:.2%})")
        for task in sorted(counter):
            rows.append(
                {
                    "experiment": spec.exp_name,
                    "problem_type": task,
                    "count": counter[task],
                    "ratio": counter[task] / total if total else 0.0,
                    "total": total,
                }
            )
    with (output_root / "overview.md").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    with (output_root / "overview.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["experiment", "problem_type", "count", "ratio", "total"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    unknown = [key for key in args.experiments if key not in EXPERIMENTS]
    if unknown:
        raise SystemExit(f"Unknown experiment keys: {unknown}. Choices: {sorted(EXPERIMENTS)}")

    results = []
    missing = []
    for key in args.experiments:
        result = analyze_one(EXPERIMENTS[key], args.experiment_root, args.output_root)
        results.append(result)
        if result["missing"]:
            missing.append(result["missing"])

    write_overview(args.output_root, results)
    print(f"[actual-problem-type-ratio] wrote outputs to {args.output_root}")
    if missing:
        print(f"[actual-problem-type-ratio] missing inputs: {len(missing)}")
        for item in missing:
            print(f"  - {item}")
        if args.strict:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
