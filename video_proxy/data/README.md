# VideoProxy Data

This directory owns dataset creation and transformation. It does not launch
training jobs; training entrypoints live under `video_proxy/experiments/`.

## Contents

- `pipelines/`: source-specific dataset construction and curation pipelines.
- `mixing/`: reusable Python helpers for building multi-task train/val JSONL files.
- `scripts/`: command-line data preparation entrypoints.

## Common Commands

```bash
bash video_proxy/data/scripts/setup_base_data.sh
bash video_proxy/data/scripts/prepare_offline_frames.sh
python video_proxy/data/scripts/convert_jsonl_to_cot.py --help
```

Generated large datasets should stay outside git-managed source files unless
they are intentionally small fixtures.
