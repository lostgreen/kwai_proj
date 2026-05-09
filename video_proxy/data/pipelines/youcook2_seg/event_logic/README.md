# Event Logic Pipeline

Build event-logic training and validation data from segmented videos.

## Main Files

- `build_event_logic_vlm.py`: construct VLM-style event logic data.
- `build_event_shuffle.py`: build shuffled event-order examples.
- `build_harder_training_mix.py`: assemble harder event-logic mixes.
- `prompts.py` and `vlm_task_prompts.py`: prompt templates.
- `run_event_logic_rollout.sh` and `run_event_logic_vlm.sh`: rollout/data generation helpers.

Use `video_proxy/insights/event_logic_ablations/` for analyzing saved rollout
ratios after training.
