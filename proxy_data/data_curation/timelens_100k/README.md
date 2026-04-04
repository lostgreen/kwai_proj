# TimeLens-100K 数据源

## 基本信息

- **来源**: TimeLens-100K
- **路径**: `/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/TimeLens-100K/`
- **规模**: 19,466 条
- **标注文件**: `timelens-100k.jsonl`
- **视频目录**: `video_shards/{source}/{video_id}.mp4`

## 数据格式

```json
{
  "source": "cosmo_cap",
  "video_path": "cosmo_cap/BVs52yd-RUQ.mp4",
  "duration": 117.4,
  "events": [
    {
      "query": "When does the speaker introduce himself?",
      "span": [[0.0, 5.0]]
    }
  ]
}
```

### 域分布

| Domain | 数量 | 特点 |
|--------|------|------|
| cosmo_cap | 9,549 | 产品演示/YouTube 教程 |
| internvid_vtime | 4,651 | InternVid 时序标注 |
| didemo | 2,955 | DiDeMo 短视频检索 |
| queryd | 1,518 | QuerYD 时序定位 |
| **hirest_step** | **387** | **HiREST 层次步骤标注** (最高优先级) |
| hirest_grounding | 249 | HiREST grounding |
| hirest | 157 | HiREST 基础 |

## 筛选流程

```
text_filter.py (时长 + 事件过滤)
    ↓ passed_timelens.jsonl
sample_per_source.py (格式转换 + 可选采样)
    ↓ sample_dev.jsonl
local_screen.py (本地 Qwen3-VL-4B 视觉筛选)
    ├─ [Stage 1] L1/L2 score + domain + quality → keep/reject
    └─ [Stage 2, 可选] prog_type + visual_diversity + order_dependency
    ↓ screen_keep.jsonl / screen_reject.jsonl
```

## 运行指令

### 一键运行 (推荐)

```bash
cd proxy_data/data_curation/timelens_100k/

# 全量筛选 (默认 2 GPU, 自动断点续跑)
bash run_pipeline.sh

# 多 GPU 数据并行
NUM_GPUS=8 bash run_pipeline.sh

# 先抽样试跑
PER_SOURCE=5 bash run_pipeline.sh

# 开启二阶段筛选
SECONDARY=1 bash run_pipeline.sh

# 全量重跑
RESUME=0 bash run_pipeline.sh
```

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TL_INPUT` | `/m2v_intern/.../timelens-100k.jsonl` | 源数据 JSONL |
| `VIDEO_ROOT` | `/m2v_intern/.../TimeLens-100K/video_shards` | 视频根目录 |
| `LOCAL_MODEL` | `/home/xuboshen/models/Qwen3-VL-4B-Instruct` | 本地 VLM 路径 |
| `NUM_GPUS` | `2` | 数据并行 GPU 数 |
| `PER_SOURCE` | `0` | 每 source 采样条数 (0 = 全量) |
| `SECONDARY` | `0` | 二阶段筛选 (1 = 开启) |
| `RESUME` | `1` | 断点续跑 (1 = 跳过已有结果) |
| `OUTPUT_ROOT` | `../results/timelens_100k` | 输出目录 |

### 单步运行

```bash
# Step 1: 文本筛选
python text_filter.py \
    --input /path/to/timelens-100k.jsonl \
    --output results/passed_timelens.jsonl \
    --config ../configs/timelens_100k.yaml

# Step 2: 采样 + 格式转换
python sample_per_source.py \
    --input results/passed_timelens.jsonl \
    --output results/sample_dev.jsonl \
    --video-root /path/to/video_shards \
    --per-source 5

# Step 3: 本地 VLM 筛选 (单 GPU, 自动续跑)
python ../shared/local_screen.py \
    --input_jsonl results/sample_dev.jsonl \
    --output_jsonl results/screen_results.jsonl \
    --keep_jsonl results/screen_keep.jsonl \
    --reject_jsonl results/screen_reject.jsonl \
    --model_path /path/to/Qwen3-VL-4B-Instruct \
    --resume
```

## 产出文件

| 文件 | 说明 |
|------|------|
| `passed_timelens.jsonl` | 文本过滤后的候选 |
| `sample_dev.jsonl` | 统一格式 (local_screen 输入) |
| `screen_keep.jsonl` | **最终候选** → 下游标注 |
| `screen_reject.jsonl` | 被拒绝的记录 |
| `screen_results.jsonl` | 全部结果 (含 `_screen` / `_screen_2` 字段) |
