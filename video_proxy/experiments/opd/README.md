# OPD Experiments

Run OPD training using a teacher checkpoint or task-specific teacher checkpoints.

## Models

- `qwen3_vl_4b/run.sh`
- `qwen3_vl_8b/run.sh`
- `qwen2_5_vl_3b/run.sh`
- `qwen2_5_vl_7b/run.sh`

Each script sets model defaults and sources `video_proxy/training/recipes/opd_train.sh`.

## Single-Teacher Example

```bash
TEACHER_MODEL_PATH=/path/to/teacher \
  bash video_proxy/experiments/opd/qwen3_vl_4b/run.sh
```

## Multi-Teacher Example

```bash
AOT_TEACHER_MODEL_PATH=/path/to/aot \
SEG_TEACHER_MODEL_PATH=/path/to/seg \
EVENTLOGIC_TEACHER_MODEL_PATH=/path/to/eventlogic \
OPD_TEACHER_KEY=problem_type \
bash video_proxy/experiments/opd/qwen3_vl_4b/run.sh
```
