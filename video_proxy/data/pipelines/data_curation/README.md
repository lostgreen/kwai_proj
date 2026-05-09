# Data Curation

This directory contains the active candidate-building pipeline for videos that
may enter hierarchical segmentation annotation.

The current pipeline is intentionally small:

```text
raw dataset
  -> duration filter + unified schema
  -> optional local model score
  -> screen_keep.jsonl
```

## Active Layout

```text
data_curation/
├── README.md
├── run.sh
├── configs/
│   ├── et_instruct_164k.yaml
│   └── timelens_100k.yaml
├── curation/
│   ├── sources.py
│   ├── duration_filter.py
│   ├── local_score.py
│   └── io.py
└── shared/
    └── local_screen.py
```

## Outputs

`run.sh` writes outputs under `results/<dataset>/` by default:

| File | Meaning |
| --- | --- |
| `duration_keep.jsonl` | Records that passed duration filtering and were converted to the unified schema. |
| `duration_summary.json` | Input count, kept count, duration settings, sampling settings, and source distribution. |
| `screen_keep.jsonl` | Downstream-compatible keep file. Without local scoring, this equals `duration_keep.jsonl`. |
| `screen_results.jsonl` | Local model score output, only when `LOCAL_SCORE=1`. |
| `screen_reject.jsonl` | Local model rejects, only when `LOCAL_SCORE=1`. |

## Run

```bash
# ET-Instruct duration-only curation
DATASET=et_instruct_164k \
INPUT=/path/to/et_instruct_164k_txt.json \
VIDEO_ROOT=/path/to/ET-Instruct-164K/videos \
bash video_proxy/data/pipelines/data_curation/run.sh

# TimeLens short-video pool, balanced to 3k
DATASET=timelens_100k \
INPUT=/path/to/timelens-100k.jsonl \
VIDEO_ROOT=/path/to/TimeLens-100K/video_shards \
MIN_DURATION=0 MAX_DURATION=60 TARGET_TOTAL=3000 BALANCED_TOTAL=1 \
bash video_proxy/data/pipelines/data_curation/run.sh

# Add optional local model scoring
LOCAL_SCORE=1 LOCAL_MODEL=/path/to/Qwen3-VL-4B-Instruct NUM_GPUS=2 \
bash video_proxy/data/pipelines/data_curation/run.sh
```

## Environment Variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `DATASET` | `et_instruct_164k` | `et_instruct_164k` or `timelens_100k`. |
| `INPUT` | dataset-specific cluster path | Raw ET JSON or TimeLens JSONL. |
| `VIDEO_ROOT` | dataset-specific cluster path | Root prepended to relative video paths. |
| `OUTPUT_ROOT` | `results/<dataset>` | Output directory. |
| `MIN_DURATION` | `60` | Minimum duration in seconds. |
| `MAX_DURATION` | `240` | Maximum duration in seconds. |
| `PER_SOURCE` | `0` | Per-source cap before total sampling; `0` means no cap. |
| `TARGET_TOTAL` | `0` | Total sample cap; `0` means no cap. |
| `BALANCED_TOTAL` | `0` | If `1`, distribute `TARGET_TOTAL` across sources. |
| `LOCAL_SCORE` | `0` | If `1`, run local VLM scoring after duration filtering. |
| `LOCAL_MODEL` | Qwen3-VL-4B cluster path | Model path used by `shared/local_screen.py`. |
| `NUM_GPUS` | `1` | Data-parallel local scoring GPUs when `TP_SIZE=1`. |
| `TP_SIZE` | `1` | Tensor parallel size for local scoring. |

## Unified Record Contract

Downstream annotation expects each JSONL record to contain:

```json
{
  "videos": ["/abs/path/to/video.mp4"],
  "metadata": {
    "clip_key": "video_stem",
    "video_id": "video_stem",
    "clip_start": 0,
    "clip_end": 120.0,
    "clip_duration": 120.0,
    "original_duration": 120.0,
    "is_full_video": true,
    "source": "coin"
  },
  "source": "coin",
  "dataset": "ET-Instruct-164K",
  "duration": 120.0
}
```

Source-specific raw metadata is preserved in `_et_raw` or `_tl_raw`.
