# Qwen3-VL-4B Teacher Training

Run EMA-GRPO teacher training for Qwen3-VL-4B.

```bash
bash video_proxy/experiments/teacher_train/qwen3_vl_4b/run.sh
```

Override common settings with environment variables such as `EXP_NAME`, `TASKS`,
`ROLLOUT_BS`, `GLOBAL_BS`, and `N_GPUS_PER_NODE`.

Preset launchers:

```bash
bash video_proxy/experiments/teacher_train/qwen3_vl_4b/run_debug_2gpu.sh aot_nocot
bash video_proxy/experiments/teacher_train/qwen3_vl_4b/run_train_8gpu.sh aot_nocot
```

The preset argument can be one of `aot_nocot`, `aot_cot`, `seg_nocot`,
`seg_cot`, `logic_nocot`, or `logic_cot`. The debug preset defaults to
`N_GPUS_PER_NODE=2`, `ROLLOUT_BS=8`, `GLOBAL_BS=8`, and `VAL_BATCH_SIZE=32`.
The train preset defaults to `N_GPUS_PER_NODE=8`, `ROLLOUT_BS=64`,
`GLOBAL_BS=64`, and `VAL_BATCH_SIZE=128`.

Single-teacher entrypoints:

```bash
bash video_proxy/experiments/teacher_train/qwen3_vl_4b/run_aot_nocot.sh
bash video_proxy/experiments/teacher_train/qwen3_vl_4b/run_aot_cot.sh
bash video_proxy/experiments/teacher_train/qwen3_vl_4b/run_seg_nocot.sh
bash video_proxy/experiments/teacher_train/qwen3_vl_4b/run_seg_cot.sh
bash video_proxy/experiments/teacher_train/qwen3_vl_4b/run_logic_nocot.sh
bash video_proxy/experiments/teacher_train/qwen3_vl_4b/run_logic_cot.sh
```

The default sources are the existing `mf256` experiment directories:
`composition_base_aot_aot10k_mf256_ema`,
`composition_base_seg_hier10k_mf256_ema`, and
`composition_base_logic_el10k_mf256_ema`.
