# Event Relation：事件关系数据构造

本目录构造 event-relation proxy，用于训练模型理解事件之间的时序和上下文关系。当前推荐使用共享源视频帧缓存，避免预先切大量 mp4 clips。

## 任务类型

- `predict_next`：给定上下文预测下一步，MCQ。
- `fill_blank`：补全缺失事件，MCQ。
- `sort`：事件顺序排序，输出数字序列。

## 文件说明

- `run_event_relation_vlm.sh`：一键入口。
- `build_relation_frame_list_data.py`：调用 LLM 设计问题，并组装 frame-list 训练记录。
- `build_relation_training_mix.py`：构造更难的训练混合。
- `relation_prompts.py`：任务构造 prompt。
- `relation_task_prompts.py`：各事件关系任务 prompt。
- `run_event_relation_rollout.sh`：事件关系 rollout 入口。

## 运行命令

冒烟测试：

```bash
DATA_ROOT=/path/to/proxy_annotation_root \
ANN_DIR=/path/to/proxy_annotation_root/annotations \
OUTPUT_DIR=/path/to/event_relation \
LIMIT=5 \
TASKS="predict_next" \
bash video_proxy/data/pipelines/proxy_construction/event_relation/run_event_relation_vlm.sh
```

完整构造：

```bash
DATA_ROOT=/path/to/proxy_annotation_root \
ANN_DIR=/path/to/annotations_reclassified \
OUTPUT_DIR=/path/to/event_relation \
MODEL=pa/gemini-3.1-pro-preview \
TASKS="predict_next fill_blank sort" \
WORKERS=8 \
bash video_proxy/data/pipelines/proxy_construction/event_relation/run_event_relation_vlm.sh
```

只生成脚本文本、不调用 API：

```bash
DRY_RUN=1 LIMIT=10 \
bash video_proxy/data/pipelines/proxy_construction/event_relation/run_event_relation_vlm.sh
```

直接调用 Python 构造脚本：

```bash
python video_proxy/data/pipelines/proxy_construction/event_relation/build_relation_frame_list_data.py \
  --annotation-dir /path/to/annotations \
  --clip-dir /path/to/clips \
  --output-dir /path/to/event_relation \
  --tasks predict_next fill_blank sort \
  --frame-cache-root /path/to/frame_cache/source_2fps \
  --complete-only \
  --limit 5
```

## 常用环境变量

| 变量 | 说明 |
| --- | --- |
| `DATA_ROOT` | 标注、帧缓存和输出的外部根目录 |
| `ANN_DIR` | annotation JSON 目录 |
| `OUTPUT_DIR` | `train.jsonl`、`val.jsonl`、`stats.json` 输出目录 |
| `SHARED_FRAME_ROOT` | 共享源帧缓存目录 |
| `USE_SHARED_FRAMES=1` | 使用 frame-list 路径 |
| `BUILD_SOURCE_FRAME_CACHE=1` | 运行前先构建源帧缓存 |
| `TASKS` | 任务集合，例如 `predict_next fill_blank sort` |
| `TRAIN_BUDGET` | train 采样预算，`-1` 表示不限 |
| `VAL_COUNT` | val 样本数 |
| `LIMIT` | 调试时限制处理数量 |
| `DRY_RUN=1` | 只生成文本和检查流程，不调用 API |

## 输出

```text
$OUTPUT_DIR/
├── train.jsonl
├── val.jsonl
├── stats.json
├── cache/
└── logs/
```

重复运行会复用 `cache/` 中的 LLM 响应。
