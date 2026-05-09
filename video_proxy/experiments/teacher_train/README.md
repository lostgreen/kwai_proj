# Teacher Training Experiments

Train teacher models with the shared EMA-GRPO recipe.

## Models

- `qwen3_vl_4b/run.sh`
- `qwen3_vl_4b/run_cot_2gpu.sh`
- `qwen3_vl_8b/run.sh`
- `qwen3_vl_8b/run_cot_2gpu.sh`
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

## Qwen3-VL CoT Smoke

The `run_cot_2gpu.sh` wrappers reuse an existing no-CoT experiment JSONL,
convert `prompt`/`messages` to tagged CoT, print prompt samples grouped by
`problem_type`, then run a 2-GPU one-step teacher-training smoke with CoT budget
control enabled.

```bash
SOURCE_EXP_NAME=composition_base_seg_logic_aot_hier10k_el10k_aot10k_mf256_ema \
SAMPLE_PROMPTS_PER_TYPE=3 \
COT_BUDGET_MAX_TOKENS=128 \
bash video_proxy/experiments/teacher_train/qwen3_vl_4b/run_cot_2gpu.sh
```

Use `SOURCE_TRAIN_FILE` and `SOURCE_TEST_FILE` to point at arbitrary old JSONL
files. The smoke writes converted data under
`$MULTI_TASK_DATA_ROOT/experiments/$EXP_NAME/` and checks
`$CHECKPOINT_ROOT/$EXP_NAME/rollouts/step_000001.jsonl` after training.
