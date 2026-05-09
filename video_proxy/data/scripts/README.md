# Data Scripts

Shell and Python entrypoints for preparing VideoProxy data.

## What Belongs Here

- One-off or reusable commands that prepare base data, frame caches, or prompt formats.
- Thin wrappers around modules in `video_proxy/data/mixing` or `video_proxy/data/pipelines`.

## What Does Not Belong Here

- Training launchers. Put those in `video_proxy/experiments/`.
- Browser tools or plotting reports. Put those in `video_proxy/insights/`.

## Examples

```bash
bash video_proxy/data/scripts/setup_base_data.sh
bash video_proxy/data/scripts/refresh_tg_base_with_timelens.sh
python video_proxy/data/scripts/convert_jsonl_to_cot.py --help
```
