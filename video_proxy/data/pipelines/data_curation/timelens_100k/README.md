# TimeLens Source Notes

TimeLens-specific active curation code now lives in the shared `../curation/`
package. Use the top-level launcher:

```bash
DATASET=timelens_100k bash video_proxy/data/pipelines/data_curation/run.sh
```

Temporal-grounding rollout helpers moved to
`../../temporal_grounding/timelens_rollout/` because they are TG experiment
utilities, not curation steps.
