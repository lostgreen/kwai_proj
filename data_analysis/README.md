# Data Analysis

This folder currently contains two analysis entry points:

- `analyze_event_ablations.py`: read saved train rollouts, count actual `problem_type` ratios, and draw one pie chart per experiment.
- `analyze_hier_seg_frame_budget.py`: analyze hier-seg annotation durations and estimate how much GT phase/event/action resolution remains under frame budgets like `48 / 64 / 128 / 256`.

## Hier-Seg Frame Budget Analysis

Run from the repo root:

```bash
python data_analysis/analyze_hier_seg_frame_budget.py \
  --annotation-dir /path/to/youcook2_seg/hier_seg_annotation/annotations \
  --output-dir data_analysis/outputs/hier_seg_frame_budget
```

Default assumptions:

- `L1` uses `1 fps`
- `L2` uses `2 fps`
- `L3` uses `2 fps`
- compared budgets are `48 64 96 128 256`

Outputs include:

- `duration_overview.png`
- `count_overview.png`
- `frame_budget_overview.png`
- `frame_budget_threshold_heatmaps.png`
- `duration_summary.csv`
- `budget_summary.csv`
- `summary.json`
- `README.md`

## Event Logic Ratio Analysis

Supported experiments:

- `PN` -> `el_ablation_predict_next`
- `FB` -> `el_ablation_fill_blank`
- `SORT` -> `el_ablation_sort`

## Run

Run from the repo root:

```bash
bash data_analysis/run_event_logic_ablation_analysis.sh
```

The default root is:

```text
/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/multi_task
```

If your experiment directories live somewhere else:

```bash
bash data_analysis/run_event_logic_ablation_analysis.sh \
  --experiment-root /path/to/multi_task \
  --output-root data_analysis/outputs/my_run
```

Analyze only one experiment:

```bash
bash data_analysis/run_event_logic_ablation_analysis.sh --experiments PN
```

Analyze all three:

```bash
bash data_analysis/run_event_logic_ablation_analysis.sh \
  --experiments PN FB SORT \
  --output-root data_analysis/outputs/actual_training_problem_type_ratio
```

Fail fast if any expected input is missing:

```bash
bash data_analysis/run_event_logic_ablation_analysis.sh --strict
```

## Outputs

For each experiment the script writes only:

- `actual_training_problem_type_ratio.csv`
- `actual_training_problem_type_ratio.png`
- `README.md`

At the top level it also writes:

- `overview.csv`
- `overview.md`

## Notes

- It only reads `rollouts/step_*.jsonl`.
- It ignores `val_step_*.jsonl`.
- It does not try to compare planned ratio, sampler ratio, or validation metrics anymore.
- Generated outputs are ignored by git via `data_analysis/.gitignore`.
