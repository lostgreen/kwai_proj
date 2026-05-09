# VideoProxy Training Infrastructure

This directory is shared infrastructure for `video_proxy/experiments/`.
Browse `video_proxy/experiments/` when you want to choose a run to launch.
Keep reusable training recipes, low-level launchers, debug helpers, and tools here.

## What Belongs Here

- `recipes/`: shared teacher-training and OPD recipes.
- `launchers/`: low-level launchers that call verl/EasyR1.
- `common/`: shell defaults and shared helper functions.
- `tools/`: checkpoint, rollout, GPU filler, and inspection utilities.
- `debug/`: scripts for diagnosing a running or saved training job.

Model-specific user entrypoints do not belong here. Put them under
`video_proxy/experiments/teacher_train/`, `video_proxy/experiments/opd/`, or
`video_proxy/experiments/baselines/`.

## Common Entrypoints

Use these through experiment wrappers unless you are debugging the runner itself:

```bash
bash -n video_proxy/training/launchers/run_multi_task.sh
bash -n video_proxy/training/recipes/teacher_train_ema_grpo.sh
bash -n video_proxy/training/recipes/opd_train.sh
```

Example user-facing runs live in:

```bash
bash video_proxy/experiments/teacher_train/qwen3_vl_4b/run.sh
TEACHER_MODEL_PATH=/path/to/teacher bash video_proxy/experiments/opd/qwen3_vl_4b/run.sh
bash video_proxy/experiments/baselines/grpo/qwen3_vl_4b/run.sh
```
