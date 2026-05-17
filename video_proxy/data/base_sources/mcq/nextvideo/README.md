# NeXTVideo MCQ Base Refresh

This directory adapts NeXTVideo MCQ records into the shared MCQ format used by
the existing LLaVA rollout, reward, selection, and base setup utilities.

Remote data root verified on KML:

```text
/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/NeXTVideo/
  train.jsonl     # 34,132 records
  val.jsonl       # 4,996 records
  NExTVideo/      # video files referenced by ./NExTVideo/... paths
```

`/m2v_intern/xuboshen/zgw/VideoProxyMixed` is an eval-output root on the current
server. Use the `data/VideoProxyMixed/NeXTVideo` root above for training data.

## Files

- `prepare_nextvideo.py` converts NeXTVideo chat JSONL into shared MCQ JSONL.
- `run_pipeline.sh` converts train data, runs 8-GPU rollout, selects low-reward
  samples, and writes `train_final_direct.jsonl`.
- Existing shared utilities reused by this source:
  - `../rollout/reward.py`
  - `../rollout/select_from_shards.py`
  - `../check_format.py`

## Recommended Run

From the repo root:

```bash
bash video_proxy/data/base_sources/mcq/nextvideo/run_pipeline.sh
```

By default `TARGET_TOTAL=0`, so the pipeline keeps all low-reward candidates
after rollout. Once the report is complete, set `TARGET_TOTAL` only if you want
to downsample to a specific comparison size.

The default rollout model is Qwen3-VL-4B. Override `MODEL_PATH` if you need a
different teacher. The default vLLM batch size is 32, matching the current
smoke/full rollout setting.

Then refresh the multi-task MCQ base:

```bash
bash video_proxy/data/scripts/refresh_mcq_base_with_nextvideo.sh
```

Useful dry-run style switches:

```bash
RUN_NEXTVIDEO_PIPELINE=false \
RUN_FRAME_EXTRACTION=false \
bash video_proxy/data/scripts/refresh_mcq_base_with_nextvideo.sh
```

## Output Layout

```text
/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/rollouts/mcq_nextvideo_qwen3_vl_4b_roll8_leq3of8/
  nextvideo_mcq_all.jsonl
  _shard*_report.jsonl
  rollout_report.jsonl
  nextvideo_selection_summary.json
  train_final.jsonl
  train_final_direct.jsonl
```

`rollout_kept.jsonl` is no longer a default output; it can be enabled for legacy
debugging with `WRITE_KEPT_JSONL=true`. Treat `rollout_report.jsonl` as the
canonical rollout artifact and derive training subsets from it.

Base-refresh bookkeeping defaults to:

```text
video_proxy/data/base_sources/mcq/results/base_refresh/nextvideo_reward_0p0_0p375_n3237/
```
