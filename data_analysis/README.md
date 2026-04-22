# Event Logic Ablation Analysis

This folder contains an offline analysis tool for the three event logic ablation
experiments:

- `PN` -> `el_ablation_predict_next`
- `FB` -> `el_ablation_fill_blank`
- `SORT` -> `el_ablation_sort`

The analysis compares:

1. Planned dataset mix from each experiment's `train.jsonl`
2. Planned sampler mix from `task_weights` and the task-homogeneous sampler
3. Actual training mix from saved `rollouts/step_*.jsonl`
4. Validation score by `problem_type` from `experiment_log.jsonl`

## Run

Run from the repo root:

```bash
bash data_analysis/run_event_logic_ablation_analysis.sh
```

This uses the default paths hard-coded by the event ablation scripts:

```text
data root:
/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/multi_task/experiments

checkpoint root:
/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/event_logic/ablations
```

Override roots if your experiment artifacts live somewhere else:

```bash
bash data_analysis/run_event_logic_ablation_analysis.sh \
  --data-root /path/to/multi_task/experiments \
  --checkpoint-root /path/to/event_logic/ablations \
  --output-root data_analysis/outputs/my_run
```

Analyze only one experiment:

```bash
bash data_analysis/run_event_logic_ablation_analysis.sh --experiments PN
```

Analyze all three and write to a custom output directory:

```bash
bash data_analysis/run_event_logic_ablation_analysis.sh \
  --experiments PN FB SORT \
  --output-root data_analysis/outputs/event_logic_ablations
```

Fail fast if any expected input is missing:

```bash
bash data_analysis/run_event_logic_ablation_analysis.sh --strict
```

## Outputs

For each experiment the script writes:

- `summary.md`
- `train_problem_type_counts.csv`
- `val_problem_type_counts.csv`
- `sampler_plan.csv`
- `actual_train_step_counts.csv`
- `actual_train_sample_counts.csv`
- `actual_train_step_sequence.csv`
- `val_scores.csv`
- `train_mix_pie.png`
- `val_mix_pie.png`
- `planned_vs_actual_train_mix.png`
- `actual_train_mix_over_steps.png`
- `val_score_by_problem_type.png`

The script also writes a top-level overview:

- `overview.csv`
- `event_share_overview.csv`
- `overview.md`
- `event_share_overview.png`

## Notes

- The script compares three different notions of ratio:
  - dataset ratio from `train.jsonl`
  - sampler-planned ratio from `task_weights` plus the task-homogeneous sampler
  - actual training ratio from saved train rollouts
- Validation score is read from `experiment_log.jsonl` when available, otherwise it
  falls back to aggregating `rollouts/val_step_*.jsonl`.
- Generated outputs are ignored by git via `data_analysis/.gitignore`.
