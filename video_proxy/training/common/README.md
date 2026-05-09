# Training Common

Shared shell defaults and helper functions for VideoProxy training.

## Files

- `multi_task_common.sh`: model, data, hardware, algorithm, checkpoint, and reward defaults.
- `gpu_filler_common.sh`: GPU filler lifecycle helpers used by launchers.

Do not put model-specific experiment defaults here unless they are shared across
multiple experiment families.
