# VideoProxy Experiments

This is the user-facing place to choose and launch training runs.

## Layout

- `teacher_train/`: train teacher models with EMA-GRPO.
- `opd/`: run OPD using single-teacher or multi-teacher checkpoints.
- `baselines/`: non-OPD baselines, currently GRPO.

Shared mechanics live in `video_proxy/training/`. Experiment scripts should be
thin wrappers that set model paths, defaults, and run names before sourcing a
shared recipe or launcher.

## Examples

```bash
bash video_proxy/experiments/teacher_train/qwen3_vl_4b/run.sh
TEACHER_MODEL_PATH=/path/to/teacher bash video_proxy/experiments/opd/qwen3_vl_4b/run.sh
bash video_proxy/experiments/baselines/grpo/qwen3_vl_4b/run.sh
```
