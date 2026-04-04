# ET-Instruct-164K 数据源

## 基本信息

- **来源**: ET-Instruct-164K
- **路径**: `/m2v_intern/xuboshen/zgw/data/ET-Instruct-164K/`
- **规模**: ~164K 样本
- **文件**:
  - `et_instruct_164k_vid.json` — 视频相关样本
  - `et_instruct_164k_txt.json` — 纯文本样本
- **视频域**: ActivityNet, COIN, DiDeMo, Ego4D-NAQ, Ego-TimeQA, HACS, HowToCaption, HowToStep, MR-HiSum, QuerYD, TACoS, ViTT

## 数据格式

来源论文: [E.T. Bench (NeurIPS 2024)](https://arxiv.org/abs/2409.18111)

使用 `et_instruct_164k_txt.json`（纯文本时间戳，非 `<vid>` token 版本）。

```json
{
  "task": "slc",                           // 任务类型（9 种）
  "source": "how_to_step",                 // 来源数据集 = domain
  "video": "how_to_step/PJi8ZEHAFcI.mp4",  // 视频相对路径
  "duration": 200.767,                     // 视频时长（秒）
  "src": [12, 18],                         // [可选] 输入时间戳（秒）
  "tgt": [36, 44, 49, 57],                 // [可选] 输出时间戳（秒），成对出现 [start, end, ...]
  "conversations": [                       // 对话对
    {"from": "human", "value": "<image>\n..."},
    {"from": "gpt", "value": "36.0 - 44.0 seconds, clean the bananas. ..."}
  ]
}
```

### Domains & 视频包大小

| Domain | tar.gz 大小 |
|--------|------------|
| how_to_step | 45G |
| how_to_caption | 40G |
| hacs | 39G |
| ego_timeqa | 34G |
| mr_hisum | 25G |
| activitynet | 20G |
| coin | 14G |
| didemo | 8.1G |
| vitt | 7.4G |
| ego4d_naq | 2.9G |
| queryd | 1.3G |
| tacos | 24M |

## 筛选流程

```
text_filter.py (仅时长过滤)
    ↓ passed.jsonl
sample_per_source.py (格式转换 + 可选采样)
    ↓ sample_dev.jsonl
local_screen.py (本地 Qwen3-VL-4B 视觉筛选)
    ↓ screen_keep.jsonl / screen_reject.jsonl
```

### 三步说明

| Step | 脚本 | 作用 |
|------|------|------|
| 1 | `text_filter.py --no_event_filter` | 按时长 (60-240s) 过滤，去重，域均衡上限 |
| 2 | `sample_per_source.py` | 将原始格式转为统一 JSONL (`videos[]`, `metadata{}`); 可选每 source 抽 N 条 |
| 3 | `local_screen.py` | 本地 VLM (vLLM) 评估: 层次潜力(1-5)、域分类、视觉质量; 多卡并行 |

### local_screen 筛选维度

- **HIER_SCORE (1-5)**: 层次分割潜力 — 视频是否有多层时序结构
- **DOMAIN_L1 / L2**: 域分类 — procedural, physical, lifestyle, entertainment, narrative, educational
- **QUALITY**: 视觉质量 — good (清晰物理动作) / bad (模糊/静态/纯语音/游戏画面)

**决策规则**: `hier_score >= 3` 且 `quality == good` → keep

## 运行指令

### 一键运行 (推荐)

```bash
cd proxy_data/data_curation/et_instruct_164k/

# 全量筛选 (单 GPU)
bash run_pipeline.sh

# 多 GPU 数据并行
NUM_GPUS=8 bash run_pipeline.sh

# 先抽样试跑 (每 source 5 条)
PER_SOURCE=5 bash run_pipeline.sh
```

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ET_JSON_PATH` | `/m2v_intern/.../et_instruct_164k_txt.json` | 源数据 JSON |
| `VIDEO_ROOT` | `/m2v_intern/.../ET-Instruct-164K/videos` | 视频根目录 |
| `LOCAL_MODEL` | `/home/xuboshen/models/Qwen3-VL-4B-Instruct` | 本地 VLM 路径 |
| `NUM_GPUS` | `1` | 数据并行 GPU 数 |
| `PER_SOURCE` | `0` | 每 source 采样条数 (0 = 全量) |
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

# Step 3: 本地 VLM 筛选 (单 GPU)
python local_screen.py \
    --input_jsonl results/sample_dev.jsonl \
    --output_jsonl results/screen_results.jsonl \
    --keep_jsonl results/screen_keep.jsonl \
    --reject_jsonl results/screen_reject.jsonl \
    --model_path /path/to/Qwen3-VL-4B-Instruct

# Step 3: 多 GPU 数据并行 (8 卡)
for i in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=$i python local_screen.py \
        --input_jsonl results/sample_dev.jsonl \
        --output_jsonl results/screen_shard${i}.jsonl \
        --keep_jsonl results/keep_shard${i}.jsonl \
        --reject_jsonl results/reject_shard${i}.jsonl \
        --model_path /path/to/Qwen3-VL-4B-Instruct \
        --shard_id $i --num_shards 8 &
done
wait
cat results/keep_shard*.jsonl > results/screen_keep.jsonl
```

### 产出文件

| 文件 | 说明 |
|------|------|
| `passed.jsonl` | 时长过滤后的候选 |
| `sample_dev.jsonl` | 采样 + 统一格式 (local_screen 输入) |
| `screen_keep.jsonl` | 通过视觉筛选的记录 |
| `screen_reject.jsonl` | 被拒绝的记录 |
| `screen_results.jsonl` | 全部结果 (含 `_screen` 字段) |

## 文件说明

| 文件 | 用途 |
|------|------|
| `text_filter.py` | Step 1: 文本/元数据筛选 (时长、去重、域均衡) |
| `sample_per_source.py` | Step 2: 格式转换 + 每源采样 |
| `local_screen.py` | Step 3: 本地 VLM 视觉预筛选 (vLLM batch inference) |
| `run_pipeline.sh` | 一键运行三步流程 |
| `explore_data.py` | 数据探索工具 (查看分布、统计) |
