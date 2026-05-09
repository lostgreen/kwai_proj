# Qwen2.5-VL-3B Teacher Training

Run EMA-GRPO teacher training for Qwen2.5-VL-3B.

```bash
bash video_proxy/experiments/teacher_train/qwen2_5_vl_3b/run.sh
```

Override common settings with environment variables such as `EXP_NAME`, `TASKS`,
`ROLLOUT_BS`, `GLOBAL_BS`, and `N_GPUS_PER_NODE`.
