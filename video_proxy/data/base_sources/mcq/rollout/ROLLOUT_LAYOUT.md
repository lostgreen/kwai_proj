# Rollout Artifact Layout

Use this directory convention for offline rollout artifacts under the shared data
root:

```text
/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/
  rollouts/
    mcq_llava_qwen3_vl_8b_roll8_leq3of8/
    mcq_nextvideo_qwen3_vl_4b_roll8_leq3of8/
    aot_nocot_qwen3_vl_8b_roll8/
```

Each rollout directory should keep:

```text
input.jsonl or *_all.jsonl
rollout_report.jsonl
*_selection_summary.json
train_final.jsonl
train_final_direct.jsonl
_shard*_report.jsonl
```

`rollout_report.jsonl` is the canonical artifact. It keeps all samples and the
raw rollout evidence (`responses`, `rewards`, `mean_reward`). Selection scripts
should derive final training data from this report.

`rollout_kept.jsonl` is a legacy convenience output for samples whose rewards
are neither all-zero nor all-one. It is optional and should not be treated as the
source of truth.

For existing scattered results, prefer a symlink or manifest migration first:

```bash
mkdir -p /m2v_intern/xuboshen/zgw/data/VideoProxyMixed/rollouts
ln -s ../results_qwen3_vl_8b_roll8_leq3of8 \
  /m2v_intern/xuboshen/zgw/data/VideoProxyMixed/rollouts/mcq_llava_qwen3_vl_8b_roll8_leq3of8
```

After all scripts point to `rollouts/`, the old top-level result directories can
be archived or removed by a separate, explicit cleanup step.
