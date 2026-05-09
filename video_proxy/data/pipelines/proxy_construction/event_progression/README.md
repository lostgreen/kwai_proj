# Event Progression：事件进展数据构造

本目录构造 event-progression proxy，用于训练模型理解事件、动作的自然进展顺序。当前主路径是 `2fps source frame cache + frame-list JSONL`，不再把拼接 mp4 clips 作为主数据产物。

## 任务类型

- `action_v2t_3way`
- `action_t2v_binary`
- `event_v2t_3way`
- `event_t2v_binary`
- `event_forward_reverse_binary`

## 文件说明

- `build_progression_manifests.py`：把 annotation JSON 转成 action/event/event_dir manifests。
- `build_progression_frame_list_data.py`：基于 manifests 和帧缓存构造 action/event progression 记录。
- `build_forward_reverse_frame_list_data.py`：构造 forward/reverse event 记录。
- `run_progression_pipeline.py`：编排源帧缓存构建、raw 合并和 rollout 筛选。
- `filter_progression_rollouts.py`：从 rollout 中筛选 hard-but-solvable 样本。
- `merge_progression_rollout_shards.py`：合并断点续跑产生的 rollout shards。

## 运行命令

生成 manifests：

```bash
python video_proxy/data/pipelines/proxy_construction/event_progression/build_progression_manifests.py \
  --annotation-dir /path/to/annotations \
  --action-output /path/to/manifests/action.jsonl \
  --event-output /path/to/manifests/event.jsonl \
  --event-dir-output /path/to/manifests/event_dir.jsonl \
  --complete-only \
  --filter-order
```

构建共享源帧缓存：

```bash
python video_proxy/data/pipelines/proxy_construction/event_progression/run_progression_pipeline.py build-source-cache \
  --manifest /path/to/manifests/action.jsonl \
  --manifest /path/to/manifests/event.jsonl \
  --manifest /path/to/manifests/event_dir.jsonl \
  --frames-root /path/to/source_frame_cache \
  --workers 8
```

构造 action/event progression 数据：

```bash
python video_proxy/data/pipelines/proxy_construction/event_progression/build_progression_frame_list_data.py \
  --frames-root /path/to/source_frame_cache \
  --action-manifest /path/to/manifests/action.jsonl \
  --event-manifest /path/to/manifests/event.jsonl \
  --action-v2t-output /path/to/raw/action_v2t.jsonl \
  --action-t2v-output /path/to/raw/action_t2v.jsonl \
  --event-v2t-output /path/to/raw/event_v2t.jsonl \
  --event-t2v-output /path/to/raw/event_t2v.jsonl
```

构造 forward/reverse 数据：

```bash
python video_proxy/data/pipelines/proxy_construction/event_progression/build_forward_reverse_frame_list_data.py \
  --event-manifest /path/to/manifests/event_dir.jsonl \
  --frames-root /path/to/source_frame_cache \
  --output /path/to/raw/event_forward_reverse.jsonl
```

合并 raw pool：

```bash
python video_proxy/data/pipelines/proxy_construction/event_progression/run_progression_pipeline.py merge-raw \
  --input /path/to/raw/action_v2t.jsonl \
  --input /path/to/raw/action_t2v.jsonl \
  --input /path/to/raw/event_v2t.jsonl \
  --input /path/to/raw/event_t2v.jsonl \
  --input /path/to/raw/event_forward_reverse.jsonl \
  --output-dir /path/to/merged_raw \
  --val-ratio 0.1
```

## Rollout 筛选

先 dry-run 检查 rollout 命令：

```bash
python video_proxy/data/pipelines/proxy_construction/event_progression/run_progression_pipeline.py rollout-filter \
  --input /path/to/merged_raw/train.jsonl \
  --output-dir /path/to/rollout \
  --dry-run
```

已有 rollout report 时直接筛 hard cases：

```bash
python video_proxy/data/pipelines/proxy_construction/event_progression/filter_progression_rollouts.py \
  --report /path/to/rollout/rollout_report.jsonl \
  --input /path/to/merged_raw/train.jsonl \
  --output /path/to/rollout/hard_cases.jsonl \
  --stats-output /path/to/rollout/hard_cases.stats.json
```

合并断点续跑 shards：

```bash
python video_proxy/data/pipelines/proxy_construction/event_progression/merge_progression_rollout_shards.py \
  --rollout-dir /path/to/rollout \
  --output-dir /path/to/rollout
```

## 输出

- `manifests/*.jsonl`：从标注抽出的 action/event 描述。
- `source_frame_cache/`：按源视频构建的共享帧缓存。
- `raw/*.jsonl`：未经 rollout 筛选的代理样本。
- `train_filtered.jsonl`：rollout 过滤后的训练样本。

## 注意

旧版 mp4 clip 实验、答案重平衡和可视化脚本只作为本地历史材料保留。新工作请优先使用 frame-list 路径。
