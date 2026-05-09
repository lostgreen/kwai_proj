# Pipeline Shared Helpers

Reusable helpers shared by source-specific data pipelines.

## Files

- `frame_cache.py`: frame cache handling.
- `seg_source.py`: segmentation source loading and derived clip helpers.

Keep this directory small and dependency-light. Pipeline-specific behavior
belongs in the pipeline that uses it.
