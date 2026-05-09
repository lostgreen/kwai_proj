# Training Launchers

Low-level launchers that call the underlying verl/EasyR1 training stack.

## Main Launcher

- `run_multi_task.sh`: builds or validates mixed train/val JSONL files, starts GPU filler when enabled, and launches training.

User-facing scripts should live under `video_proxy/experiments/` and source this
launcher through a recipe.
