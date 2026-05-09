# TimeLens Rollout

Temporal-grounding rollout helpers for TimeLens short-video queries.

These scripts are not part of data curation. They take an already curated
TimeLens pool, expand event queries into TG prompts, run offline rollout, and
select samples by rollout IoU.

## Files

- `build_tg_rollout_dataset.py`: expand TimeLens records into query-level TG JSONL.
- `run_tg_rollout.sh`: run model rollout and optional analysis.
- `analyze_tg_rollout.py`: summarize rollout reports into query statistics.
- `select_tg_iou_range.py`: select TG samples by mean IoU range.
