# Temporal Grounding Pipeline

Build and validate temporal grounding data.

## Main Files

- `build_dataset.py`: construct temporal grounding datasets.
- `convert_nocot_to_cot.py`: convert direct prompts to CoT-style prompts.
- `merge_tg_train_with_timelens.py`: merge training data with TimeLens-derived data.
- `trim_videos.py` and `validate_tg_videos.py`: media preparation and validation.
- `run_pipeline.sh`: end-to-end helper.

Prepared data can be mixed into training with `video_proxy/data/mixing/`.
