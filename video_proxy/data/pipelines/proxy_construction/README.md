# Proxy Construction 总览

`proxy_construction/` 负责把候选视频和层级标注转换成论文里的三类 proxy 训练数据。当前主路径是“共享源视频帧缓存 + frame-list JSONL”，不再把大量中间 mp4 clip 当成主要数据产物。

## 目录功能

- `annotation/`：从候选视频抽帧、检测场景边界，并生成层级标注 JSON。
- `event_boundary/`：从层级标注构造 L1/L2/L3_seg 事件边界任务。
- `event_progression/`：构造事件进展任务，包括 action/event 顺序判断和 forward/reverse 判断。
- `event_relation/`：构造事件关系任务，包括 predict-next、fill-blank、sort。

## 推荐运行顺序

```bash
# 1. 生成层级标注
JSONL=/path/to/screen_keep.jsonl \
DATA_ROOT=/path/to/proxy_annotation_root \
LIMIT=5 \
bash video_proxy/data/pipelines/proxy_construction/annotation/run_annotation_pipeline.sh

# 2. 构造事件边界 proxy
python video_proxy/data/pipelines/proxy_construction/event_boundary/build_boundary_frame_list_data.py \
  --annotation-dir /path/to/proxy_annotation_root/annotations \
  --output-dir /path/to/event_boundary \
  --levels L1 L2 L3_seg \
  --complete-only

# 3. 构造事件进展 manifests
python video_proxy/data/pipelines/proxy_construction/event_progression/build_progression_manifests.py \
  --annotation-dir /path/to/proxy_annotation_root/annotations \
  --action-output /path/to/event_progression/manifests/action.jsonl \
  --event-output /path/to/event_progression/manifests/event.jsonl \
  --event-dir-output /path/to/event_progression/manifests/event_dir.jsonl \
  --complete-only \
  --filter-order

# 4. 构造事件关系 proxy
DATA_ROOT=/path/to/proxy_annotation_root \
OUTPUT_DIR=/path/to/event_relation \
LIMIT=5 \
bash video_proxy/data/pipelines/proxy_construction/event_relation/run_event_relation_vlm.sh
```

## 常见产物

- `annotations/*.json`：层级标注结果。
- `frame_cache/source_2fps/`：共享源视频帧缓存。
- `manifests/*.jsonl`：从标注抽出的 action/event/event_dir 描述。
- `raw/*.jsonl`：rollout 前的 proxy 样本。
- `train.jsonl`、`val.jsonl`：训练/验证数据。
- `logs/`、`cache/`：运行日志和 LLM 响应缓存。

大体量产物建议写到外部 `$DATA_ROOT` 或实验数据盘，不要放进 git 仓库。
