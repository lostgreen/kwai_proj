# Curation Helpers

This package contains the active data curation pipeline.

## Files

- `sources.py`: adapters for ET-Instruct-164K and TimeLens-100K raw records.
- `duration_filter.py`: duration filtering, sampling, unified JSONL writing.
- `local_score.py`: optional wrapper around `../shared/local_screen.py`.
- `io.py`: JSON and JSONL helpers.

The default path is duration-only. When local scoring is disabled,
`screen_keep.jsonl` is copied from `duration_keep.jsonl` so downstream
annotation scripts can keep using the same input path.
