# Training Tools

Utilities for checkpoint management, rollout inspection, GPU filler behavior,
and training diagnostics.

## Examples

```bash
bash video_proxy/training/tools/cleanup_checkpoints.sh /path/to/checkpoints
python video_proxy/training/tools/probe_cot_tags.py --help
python video_proxy/training/tools/sample_rollout_analysis.py --help
python video_proxy/training/tools/cot_efficiency_report.py --help
```

Long-running training entrypoints belong in `video_proxy/experiments/`, not here.

## CoT Efficiency Smoke Runs

For 100-step CoT/no-CoT comparisons on the existing `base + proxy` JSONL
mixtures, use the Qwen3-VL-4B single-teacher 8-GPU preset. The old source
mixtures are:

- AOT no-CoT: `composition_base_aot_aot10k_mf256_ema`
- Seg no-CoT: `composition_base_seg_hier10k_mf256_ema`
- Logic no-CoT: `composition_base_logic_el10k_mf256_ema`

The CoT preset converts the matching source JSONL into `<thought>...</thought>`
format under the new experiment data directory when needed.

```bash
# AOT no-CoT / CoT.
MAX_STEPS=100 VAL_FREQ=25 SAVE_FREQ=100 VAL_BEFORE_TRAIN=true MAX_RESPONSE_LEN=256 \
EXP_NAME=qwen3_vl_4b_aot_100step_nocot \
bash video_proxy/experiments/teacher_train/qwen3_vl_4b/run_train_8gpu.sh aot_nocot

MAX_STEPS=100 VAL_FREQ=25 SAVE_FREQ=100 VAL_BEFORE_TRAIN=true MAX_RESPONSE_LEN=256 \
EXP_NAME=qwen3_vl_4b_aot_100step_cot \
COT_BUDGET_MAX_TOKENS=128 \
bash video_proxy/experiments/teacher_train/qwen3_vl_4b/run_train_8gpu.sh aot_cot

# Seg no-CoT / CoT.
MAX_STEPS=100 VAL_FREQ=25 SAVE_FREQ=100 VAL_BEFORE_TRAIN=true MAX_RESPONSE_LEN=256 \
EXP_NAME=qwen3_vl_4b_seg_100step_nocot \
bash video_proxy/experiments/teacher_train/qwen3_vl_4b/run_train_8gpu.sh seg_nocot

MAX_STEPS=100 VAL_FREQ=25 SAVE_FREQ=100 VAL_BEFORE_TRAIN=true MAX_RESPONSE_LEN=256 \
EXP_NAME=qwen3_vl_4b_seg_100step_cot \
COT_BUDGET_MAX_TOKENS=128 \
bash video_proxy/experiments/teacher_train/qwen3_vl_4b/run_train_8gpu.sh seg_cot

# Logic no-CoT / CoT.
MAX_STEPS=100 VAL_FREQ=25 SAVE_FREQ=100 VAL_BEFORE_TRAIN=true MAX_RESPONSE_LEN=256 \
EXP_NAME=qwen3_vl_4b_logic_100step_nocot \
bash video_proxy/experiments/teacher_train/qwen3_vl_4b/run_train_8gpu.sh logic_nocot

MAX_STEPS=100 VAL_FREQ=25 SAVE_FREQ=100 VAL_BEFORE_TRAIN=true MAX_RESPONSE_LEN=256 \
EXP_NAME=qwen3_vl_4b_logic_100step_cot \
COT_BUDGET_MAX_TOKENS=128 \
bash video_proxy/experiments/teacher_train/qwen3_vl_4b/run_train_8gpu.sh logic_cot
```

Then compare learning speed and CoT health:

```bash
python video_proxy/training/tools/cot_efficiency_report.py \
  aot_nocot=/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/multi_task/qwen3_vl_4b_aot_100step_nocot \
  aot_cot=/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/multi_task/qwen3_vl_4b_aot_100step_cot \
  seg_nocot=/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/multi_task/qwen3_vl_4b_seg_100step_nocot \
  seg_cot=/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/multi_task/qwen3_vl_4b_seg_100step_cot \
  logic_nocot=/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/multi_task/qwen3_vl_4b_logic_100step_nocot \
  logic_cot=/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/multi_task/qwen3_vl_4b_logic_100step_cot \
  --max-step 100
```

Read the table as a diagnostic, not a final benchmark:

- `train_delta` and `val_delta`: how much reward improved within 100 steps.
- `step_to_*_0.5`: sample efficiency proxy.
- `delta_vs_nocot_val_final`: whether CoT helps or hurts the same family.
- `base_val_final` and `proxy_val_final`: whether the gain/loss comes from
  TG/MCQ base tasks or the added AOT/Seg/Logic proxy task.
- `delta_vs_nocot_proxy_val_final`: CoT's direct effect on the target proxy
  family.
- `log_cot_end_ratio`: whether generated CoT closes inside the budget.
- `log_cot_repaired_ratio` and `log_cot_final_token_len_mean`: whether the
  256-token response budget is being consumed by repairs or overlong thoughts.
