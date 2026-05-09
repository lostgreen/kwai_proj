# Qwen3-VL-8B OPD

Run OPD for Qwen3-VL-8B.

```bash
TEACHER_MODEL_PATH=/path/to/teacher \
  bash video_proxy/experiments/opd/qwen3_vl_8b/run.sh
```

This launcher keeps only the latest regular checkpoint by default
(`SAVE_LIMIT=1`) and disables the training GPU filler unless overridden.
