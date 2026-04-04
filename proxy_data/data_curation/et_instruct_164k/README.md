# ET-Instruct-164K 数据源

## 基本信息

- **来源**: ET-Instruct-164K
- **路径**: `/m2v_intern/xuboshen/zgw/data/ET-Instruct-164K/`
- **规模**: ~164K 样本
- **文件**: `et_instruct_164k_txt.json`
- **视频域**: ActivityNet, COIN, DiDeMo, Ego4D-NAQ, Ego-TimeQA, HACS, HowToCaption, HowToStep, MR-HiSum, QuerYD, TACoS, ViTT

## 数据格式

来源论文: [E.T. Bench (NeurIPS 2024)](https://arxiv.org/abs/2409.18111)

```json
{
  "task": "slc",
  "source": "how_to_step",
  "video": "how_to_step/PJi8ZEHAFcI.mp4",
  "duration": 200.767,
  "tgt": [36, 44, 49, 57],
  "conversations": [...]
}
```

## 筛选流程

```
text_filter.py --no_event_filter (仅时长过滤)
    ↓ passed.jsonl
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
cd proxy_data/data_curation/et_instruct_164k/

# 全量筛选 (默认 2 GPU, 自动断点续跑)
bash run_pipeline.sh

# 多 GPU 数据并行
NUM_GPUS=8 bash run_pipeline.sh

# 先抽样试跑 (每 source 5 条)
PER_SOURCE=5 bash run_pipeline.sh

# 开启二阶段筛选
SECONDARY=1 bash run_pipeline.sh

# 全量重跑 (关闭 resume)
RESUME=0 bash run_pipeline.sh
```

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ET_JSON_PATH` | `/m2v_intern/.../et_instruct_164k_txt.json` | 源数据 JSON |
| `VIDEO_ROOT` | `/m2v_intern/.../ET-Instruct-164K/videos` | 视频根目录 |
| `LOCAL_MODEL` | `/home/xuboshen/models/Qwen3-VL-4B-Instruct` | 本地 VLM 路径 |
| `NUM_GPUS` | `2` | 数据并行 GPU 数 |
| `PER_SOURCE` | `0` | 每 source 采样条数 (0 = 全量) |
| `SECONDARY` | `0` | 二阶段筛选 (1 = 开启) |
| `RESUME` | `1` | 断点续跑 (1 = 跳过已有结果) |
| `OUTPUT_ROOT` | `../results/et_instruct_164k` | 输出目录 |

### 单步运行

```bash
# Step 1: 时长过滤
python text_filter.py \
    --json_path /path/to/et_instruct_164k_txt.json \
    --output_dir results/ \
    --config ../configs/et_instruct_164k.yaml \
    --no_event_filter

# Step 2: 采样 + 格式转换
python sample_per_source.py \
    --input results/passed.jsonl \
    --output results/sample_dev.jsonl \
    --video-root /path/to/videos \
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
| `passed.jsonl` | 时长过滤后的候选 |
| `sample_dev.jsonl` | 统一格式 (local_screen 输入) |
| `screen_keep.jsonl` | **最终候选** → 下游标注 |
| `screen_reject.jsonl` | 被拒绝的记录 |
| `screen_results.jsonl` | 全部结果 (含 `_screen` / `_screen_2` 字段) |

## 文件说明

| 文件 | 用途 |
|------|------|
| `text_filter.py` | Step 1: 文本/元数据筛选 (时长、去重、域均衡) |
| `sample_per_source.py` | Step 2: 格式转换 + 每源采样 |
| `../shared/local_screen.py` | Step 3: 本地 VLM 视觉预筛选 (vLLM batch inference) |
| `run_pipeline.sh` | 一键运行三步流程 |
| `explore_data.py` | 数据探索工具 (查看分布、统计) |
