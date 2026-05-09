# Hierarchical Segmentation Annotation

Tools for annotating, validating, and converting YouCook2 videos into
hierarchical segmentation data.

## Main Files

- `annotate.py` and `annotate_gemini_native.py`: annotation entrypoints.
- `extract_frames.py`: frame extraction.
- `build_hier_data.py`: build training JSONL from annotations.
- `prompt_variants_v4.py`: prompt variants for annotation/data construction.
- `run_pipeline.sh` and `run_pipeline_gemini.sh`: end-to-end helpers.

Use `video_proxy/insights/data_browser/` to inspect generated data and
`video_proxy/insights/hier_seg_frame_budget/` to analyze frame budgets.
