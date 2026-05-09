# LLaVA Video 178K Pipeline

Utilities for preparing and filtering LLaVA Video 178K multiple-choice data.

## Main Files

- `prepare_mcq.py`: build MCQ training examples.
- `filter_and_downsample.py`: filter and sample data.
- `convert_mcq_to_direct.py`: convert MCQ format when needed.
- `mcq_reward.py`: MCQ reward helper.
- `run_pipeline.sh`: end-to-end helper.

Use `video_proxy/insights/data_browser/` for visual inspection of generated samples.
