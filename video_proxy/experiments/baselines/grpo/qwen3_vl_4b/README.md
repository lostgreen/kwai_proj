# Qwen3-VL-4B GRPO Baseline

Run the Qwen3-VL-4B EMA-GRPO baseline used for OPD comparisons.

```bash
bash video_proxy/experiments/baselines/grpo/qwen3_vl_4b/run.sh
```

This launcher intentionally clears OPD teacher paths and calls
`video_proxy/training/launchers/run_multi_task.sh` directly.
