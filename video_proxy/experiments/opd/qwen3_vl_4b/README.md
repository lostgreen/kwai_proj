# Qwen3-VL-4B OPD

Run OPD for Qwen3-VL-4B.

```bash
TEACHER_MODEL_PATH=/path/to/teacher \
  bash video_proxy/experiments/opd/qwen3_vl_4b/run.sh
```

This launcher defaults to batch-64 and save-every-50 settings for OPD-vs-GRPO
comparisons. Override `SAVE_LIMIT`, `SAVE_FREQ`, or `CHECKPOINT_ROOT` when needed.
