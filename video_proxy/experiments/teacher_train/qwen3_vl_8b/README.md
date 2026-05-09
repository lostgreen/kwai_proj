# Qwen3-VL-8B Teacher Training

Run EMA-GRPO teacher training for Qwen3-VL-8B.

```bash
bash video_proxy/experiments/teacher_train/qwen3_vl_8b/run.sh
```

Override common settings with environment variables such as `EXP_NAME`, `TASKS`,
`ROLLOUT_BS`, `GLOBAL_BS`, and `N_GPUS_PER_NODE`.
