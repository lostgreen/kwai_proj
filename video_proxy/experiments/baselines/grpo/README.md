# GRPO Baselines

EMA-GRPO baseline launchers used when comparing against OPD.

## Available Runs

- `qwen3_vl_4b/run.sh`: Qwen3-VL-4B full-composition GRPO baseline.

## Example

```bash
EXP_NAME=my_grpo_baseline \
bash video_proxy/experiments/baselines/grpo/qwen3_vl_4b/run.sh
```

These scripts do not set OPD teacher paths.
