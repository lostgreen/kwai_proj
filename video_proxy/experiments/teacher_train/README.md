# Teacher Training Experiments

Train teacher models with the shared EMA-GRPO recipe.

## Models

- `qwen3_vl_4b/run.sh`
- `qwen3_vl_8b/run.sh`
- `qwen2_5_vl_3b/run.sh`
- `qwen2_5_vl_7b/run.sh`

Each script sets `MODEL_FAMILY`, `MODEL_SIZE`, and `MODEL_PATH`, then sources
`video_proxy/training/recipes/teacher_train_ema_grpo.sh`.

## Example

```bash
EXP_NAME=my_teacher_run \
TASKS="tg mcq hier_seg" \
bash video_proxy/experiments/teacher_train/qwen3_vl_4b/run.sh
```
