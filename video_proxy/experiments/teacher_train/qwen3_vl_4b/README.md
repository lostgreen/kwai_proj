# Qwen3-VL-4B Teacher Training

Run EMA-GRPO teacher training for Qwen3-VL-4B.

```bash
bash video_proxy/experiments/teacher_train/qwen3_vl_4b/run.sh
```

Override common settings with environment variables such as `EXP_NAME`, `TASKS`,
`ROLLOUT_BS`, `GLOBAL_BS`, and `N_GPUS_PER_NODE`.

For the CoT teacher-training smoke that reuses previous experiment data:

```bash
SOURCE_EXP_NAME=composition_base_seg_logic_aot_hier10k_el10k_aot10k_mf256_ema \
SAMPLE_PROMPTS_PER_TYPE=3 \
COT_BUDGET_MAX_TOKENS=128 \
bash video_proxy/experiments/teacher_train/qwen3_vl_4b/run_cot_2gpu.sh
```
