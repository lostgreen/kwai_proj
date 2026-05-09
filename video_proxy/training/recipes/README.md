# Training Recipes

Shared recipes used by experiment entrypoints.

## Files

- `teacher_train_ema_grpo.sh`: shared EMA-GRPO teacher training defaults.
- `opd_train.sh`: shared OPD defaults and teacher-path validation.

Recipes expect model-specific environment variables to be set by an experiment
wrapper before sourcing.
