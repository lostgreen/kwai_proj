# VideoProxy

VideoProxy-specific data, experiment, training, and inspection utilities live here.
The surrounding repository still contains the underlying EasyR1/verl framework.

## Directory Map

- `data/`: build, filter, convert, and mix datasets.
- `experiments/`: user-facing run entrypoints for teacher training, OPD, and baselines.
- `training/`: shared recipes, launchers, common shell defaults, and tools used by experiments.
- `insights/`: offline reports and browser tools for checking data and rollouts.

## Common Flow

```bash
# Prepare or refresh data.
bash video_proxy/data/scripts/setup_base_data.sh

# Train a teacher.
bash video_proxy/experiments/teacher_train/qwen3_vl_4b/run.sh

# Run OPD with a teacher checkpoint.
TEACHER_MODEL_PATH=/path/to/teacher \
  bash video_proxy/experiments/opd/qwen3_vl_4b/run.sh

# Inspect rollouts.
ROLLOUT_DIR=/path/to/rollouts bash video_proxy/insights/rollout_viewer/run.sh
```

Do not add new top-level directories unless they introduce a new user-facing
concept. Most additions should fit under one of the four directories above.
