# YouCook2 Seg Pipelines

Pipelines for building VideoProxy tasks from YouCook2-style segmentation data.

## Contents

- `hier_seg_annotation/`: hierarchical segmentation annotation and data building.
- `temporal_aot/`: action-ordering / temporal AoT task construction.
- `event_logic/`: event logic task generation.

Shared helpers for all data pipelines live in `video_proxy/data/pipelines/shared/`.
Training data mixing happens later in `video_proxy/data/mixing/`.
