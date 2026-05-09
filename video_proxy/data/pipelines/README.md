# 数据流水线

`pipelines/` 放 VideoProxy 自己的数据生产流水线。它和 `base_sources/` 的区别是：这里主要构造代理任务、筛选候选视频和复用标注产物；外部基础数据集的清洗请放到 `video_proxy/data/base_sources/`。

## 当前布局

- `data_curation/`：候选视频筛选，先按时长和来源过滤，再可选本地 VLM 打分。
- `proxy_construction/`：从筛选视频和层级标注构造代理任务。
- `shared/`：pipeline 之间复用的帧缓存、标注加载和路径工具。
- `youcook2_seg/`：旧路径兼容层，不建议新增代码继续放这里。

## 推荐流程

```text
raw video dataset
  -> data_curation/
  -> proxy_construction/annotation/
  -> proxy_construction/event_boundary/
  -> proxy_construction/event_progression/
  -> proxy_construction/event_relation/
  -> video_proxy/data/mixing/
```

## 常用命令

```bash
# 1. 候选视频筛选
DATASET=et_instruct_164k \
INPUT=/path/to/et_instruct_164k_txt.json \
VIDEO_ROOT=/path/to/videos \
bash video_proxy/data/pipelines/data_curation/run.sh

# 2. 层级标注
JSONL=video_proxy/data/pipelines/data_curation/results/et_instruct_164k/screen_keep.jsonl \
LIMIT=100 \
bash video_proxy/data/pipelines/proxy_construction/annotation/run_annotation_pipeline.sh

# 3. 事件边界 / HierSeg 数据
python video_proxy/data/pipelines/proxy_construction/event_boundary/build_boundary_frame_list_data.py \
  --annotation-dir /path/to/annotations \
  --output-dir /path/to/hier_seg \
  --levels L1 L2 L3_seg

# 4. 事件进展 / Temporal AoT 数据
python video_proxy/data/pipelines/proxy_construction/event_progression/build_progression_manifests.py \
  --annotation-dir /path/to/annotations \
  --action-output /path/to/manifests/action.jsonl \
  --event-output /path/to/manifests/event.jsonl \
  --event-dir-output /path/to/manifests/event_dir.jsonl \
  --complete-only \
  --filter-order

# 5. 事件关系数据
bash video_proxy/data/pipelines/proxy_construction/event_relation/run_event_relation_vlm.sh
```

## 输出约定

- `data_curation/run.sh` 默认写到 `video_proxy/data/pipelines/data_curation/results/<dataset>/`。
- `annotation/run_annotation_pipeline.sh` 默认写到外部 `$DATA_ROOT/frames`、`$DATA_ROOT/annotations` 和 `$DATA_ROOT/logs`。
- 事件边界、事件进展、事件关系建议写入外部实验数据目录，避免把大文件放进仓库。

## 迁移说明

原 `llava_video_178k/` 和 `temporal_grounding/` 的基础数据逻辑已经迁移到 `video_proxy/data/base_sources/`。原 `youcook2_seg/` 的代理任务逻辑已经迁移到 `proxy_construction/`。旧目录只用于兼容和历史追踪。
