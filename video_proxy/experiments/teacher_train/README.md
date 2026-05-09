# Teacher Training Experiments

Train teacher models with the shared EMA-GRPO recipe.

## Models

- `qwen3_vl_4b/run.sh`
- `qwen3_vl_4b/run_aot_nocot.sh`
- `qwen3_vl_4b/run_aot_cot.sh`
- `qwen3_vl_4b/run_seg_nocot.sh`
- `qwen3_vl_4b/run_seg_cot.sh`
- `qwen3_vl_4b/run_logic_nocot.sh`
- `qwen3_vl_4b/run_logic_cot.sh`
- `qwen3_vl_8b/run.sh`
- `qwen3_vl_8b/run_aot_nocot.sh`
- `qwen3_vl_8b/run_aot_cot.sh`
- `qwen3_vl_8b/run_seg_nocot.sh`
- `qwen3_vl_8b/run_seg_cot.sh`
- `qwen3_vl_8b/run_logic_nocot.sh`
- `qwen3_vl_8b/run_logic_cot.sh`
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

## Qwen3-VL Single Teachers

The Qwen3-VL single-teacher wrappers train one teacher at a time from existing
`mf256` experiment JSONL directories:

| Teacher | Default source experiment | Default tasks |
| --- | --- | --- |
| AOT | `composition_base_aot_aot10k_mf256_ema` | `tg mcq aot` |
| Seg | `composition_base_seg_hier10k_mf256_ema` | `tg mcq hier_seg` |
| Logic | `composition_base_logic_el10k_mf256_ema` | `tg mcq event_logic` |

The no-CoT scripts read the source `train.jsonl`/`val.jsonl` directly. The CoT
scripts convert `prompt`/`messages` to tagged CoT under a new experiment data
directory, print prompt samples grouped by `problem_type`, run the 2-GPU smoke
defaults, and check saved rollouts for the CoT budget. Qwen3-VL CoT scripts
default to `<thought></thought>` tags; override `REASONING_TAG` only if a
different tag pair is desired.

```bash
SAMPLE_PROMPTS_PER_TYPE=3 \
COT_BUDGET_MAX_TOKENS=128 \
bash video_proxy/experiments/teacher_train/qwen3_vl_4b/run_aot_cot.sh
```

Use `SOURCE_TRAIN_FILE` and `SOURCE_TEST_FILE` to point at arbitrary old JSONL
files. CoT scripts write converted data under
`$MULTI_TASK_DATA_ROOT/experiments/$EXP_NAME/`.
