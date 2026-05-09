# Qwen3-VL-8B Teacher Training

Run EMA-GRPO teacher training for Qwen3-VL-8B.

```bash
bash video_proxy/experiments/teacher_train/qwen3_vl_8b/run.sh
```

Override common settings with environment variables such as `EXP_NAME`, `TASKS`,
`ROLLOUT_BS`, `GLOBAL_BS`, and `N_GPUS_PER_NODE`.

Single-teacher entrypoints:

```bash
bash video_proxy/experiments/teacher_train/qwen3_vl_8b/run_aot_nocot.sh
bash video_proxy/experiments/teacher_train/qwen3_vl_8b/run_aot_cot.sh
bash video_proxy/experiments/teacher_train/qwen3_vl_8b/run_seg_nocot.sh
bash video_proxy/experiments/teacher_train/qwen3_vl_8b/run_seg_cot.sh
bash video_proxy/experiments/teacher_train/qwen3_vl_8b/run_logic_nocot.sh
bash video_proxy/experiments/teacher_train/qwen3_vl_8b/run_logic_cot.sh
```

The default sources are the existing `mf256` experiment directories:
`composition_base_aot_aot10k_mf256_ema`,
`composition_base_seg_hier10k_mf256_ema`, and
`composition_base_logic_el10k_mf256_ema`.
