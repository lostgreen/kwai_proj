# Actual Training Problem Type Ratio

This folder now focuses on one thing only:

- read saved train rollouts
- count how many actual training samples belong to each `problem_type`
- draw one pie chart per experiment

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
