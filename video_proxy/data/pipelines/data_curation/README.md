# 候选数据筛选

`data_curation/` 是进入层级标注和代理任务构造前的候选视频筛选流水线。当前流程故意保持简单：

```text
raw dataset
  -> 时长过滤 + 统一字段
  -> 可选本地 VLM 打分
  -> screen_keep.jsonl
```

## 目录结构

```text
data_curation/
├── run.sh
├── configs/
│   ├── et_instruct_164k.yaml
│   └── timelens_100k.yaml
├── curation/
│   ├── sources.py
│   ├── duration_filter.py
│   ├── local_score.py
│   └── io.py
└── shared/
    └── local_screen.py
```

## 一键运行

```bash
# ET-Instruct：只做时长过滤
DATASET=et_instruct_164k \
INPUT=/path/to/et_instruct_164k_txt.json \
VIDEO_ROOT=/path/to/ET-Instruct-164K/videos \
bash video_proxy/data/pipelines/data_curation/run.sh

# TimeLens：短视频池，平衡采样到 3000 条
DATASET=timelens_100k \
INPUT=/path/to/timelens-100k.jsonl \
VIDEO_ROOT=/path/to/TimeLens-100K/video_shards \
MIN_DURATION=0 \
MAX_DURATION=60 \
TARGET_TOTAL=3000 \
BALANCED_TOTAL=1 \
bash video_proxy/data/pipelines/data_curation/run.sh

# 增加本地 VLM 打分
LOCAL_SCORE=1 \
LOCAL_MODEL=/path/to/Qwen3-VL-4B-Instruct \
NUM_GPUS=2 \
bash video_proxy/data/pipelines/data_curation/run.sh
```

## 关键环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DATASET` | `et_instruct_164k` | 支持 `et_instruct_164k` 或 `timelens_100k` |
| `INPUT` | 按数据集选择默认集群路径 | 原始 JSON/JSONL |
| `VIDEO_ROOT` | 按数据集选择默认集群路径 | 相对视频路径的根目录 |
| `OUTPUT_ROOT` | `results/<dataset>` | 输出目录 |
| `MIN_DURATION` | `60` | 最短时长，单位秒 |
| `MAX_DURATION` | `240` | 最长时长，单位秒 |
| `PER_SOURCE` | `0` | 每个来源的上限，`0` 表示不限 |
| `TARGET_TOTAL` | `0` | 总采样上限，`0` 表示不限 |
| `BALANCED_TOTAL` | `0` | 为 `1` 时按来源均衡分配 `TARGET_TOTAL` |
| `LOCAL_SCORE` | `0` | 为 `1` 时启用本地 VLM 打分 |
| `LOCAL_MODEL` | 集群默认 Qwen3-VL-4B | 本地打分模型路径 |
| `NUM_GPUS` | `1` | 多卡 shard 打分数量 |
| `TP_SIZE` | `1` | 本地打分 tensor parallel size |

## 输出文件

默认写到 `results/<dataset>/`：

| 文件 | 说明 |
| --- | --- |
| `duration_keep.jsonl` | 通过时长过滤并转成统一 schema 的样本 |
| `duration_summary.json` | 输入数、保留数、时长阈值、采样设置和来源分布 |
| `screen_keep.jsonl` | 下游使用的保留文件；未启用本地打分时等同于 `duration_keep.jsonl` |
| `screen_results.jsonl` | 本地 VLM 打分结果，仅 `LOCAL_SCORE=1` 时生成 |
| `screen_reject.jsonl` | 本地 VLM 拒绝样本，仅 `LOCAL_SCORE=1` 时生成 |

## 统一记录格式

下游标注至少依赖这些字段：

```json
{
  "videos": ["/abs/path/to/video.mp4"],
  "metadata": {
    "clip_key": "video_stem",
    "video_id": "video_stem",
    "clip_start": 0,
    "clip_end": 120.0,
    "clip_duration": 120.0,
    "original_duration": 120.0,
    "is_full_video": true,
    "source": "coin"
  },
  "source": "coin",
  "dataset": "ET-Instruct-164K",
  "duration": 120.0
}
```

原始来源字段会保留在 `_et_raw` 或 `_tl_raw` 中，方便后续追踪。
