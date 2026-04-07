# Grounding + Segmentation (Segment-as-CoT)

层次分割的变体：将**分割作为推理过程（CoT）**而非最终目标。

模型接收到抽象的 reasoning query，必须先 **定位 (ground)** 视频中的相关内容，再 **分割 (segment)** 出精细的时间边界。

## 核心思路

```
传统层次分割:  视频 → 固定 prompt("分割为 phases/events") → 时间戳
Grounding+Seg: 视频 + 抽象 query("定位核心活动并按 X 标准分割") → 时间戳
```

**关键差异：**
- Query 是视频特定的、由 VLM 自动生成的，不泄露答案
- 模型输出格式不变 (`<events>[[s,e],...]</events>`)
- 可选 `<think>` CoT 推理过程
- 支持"一鱼多吃"：单个视频可生成多条不同 query 的训练数据

## 数据流

```
data_curation 结果 (screen_keep.jsonl / 全量)
  │
  ├── [已有] extract_frames.py → frames/{clip_key}/*.jpg
  │
  ├── [新] annotate_gseg.py    → gseg_annotations/{clip_key}.json
  │         VLM 一步生成: abstract query + GT segments
  │
  └── [新] build_gseg_data.py  → gseg_train_data/train.jsonl, val.jsonl
            拆分 query→学生 prompt, segments→answer
```

## 快速 Demo (推荐)

> 所有命令在 `train/` 目录下运行。

一键运行完整管线 (采样 → 提帧 → 标注 → 构建训练数据):

```bash
# 默认: 每个 source 50 条, Gemini Flash
bash proxy_data/youcook2_seg/grounding_seg/run_demo.sh

# 自定义配置
PER_SOURCE=20 WORKERS=8 LIMIT=100 \
    bash proxy_data/youcook2_seg/grounding_seg/run_demo.sh

# 跳过已完成的步骤 (增量运行)
SKIP_SAMPLE=true SKIP_FRAMES=true \
    bash proxy_data/youcook2_seg/grounding_seg/run_demo.sh
```

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PER_SOURCE` | 50 | 每个 source domain 采样数 |
| `WORKERS` | 4 | 并行 workers |
| `MODEL` | pa/gmn-2.5-fl | VLM 模型 |
| `API_BASE` | https://api.novita.ai/v3/openai | VLM API |
| `LIMIT` | 0 (全部) | 标注时处理的 clip 上限 |
| `SKIP_SAMPLE` | false | 跳过采样步骤 |
| `SKIP_FRAMES` | false | 跳过提帧步骤 |
| `SKIP_ANNOTATE` | false | 跳过标注步骤 |
| `INPUT_JSONL` | candidates.jsonl | 输入数据源 |

## 分步执行

### Step 0: 设置变量

```bash
SCRIPT_DIR="proxy_data/youcook2_seg/grounding_seg"
HIER_SCRIPT_DIR="proxy_data/youcook2_seg/hier_seg_annotation"

DATA_ROOT="/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/grounding_seg"
FRAMES_DIR="${DATA_ROOT}/frames"
ANN_DIR="${DATA_ROOT}/annotations"
TRAIN_DIR="${DATA_ROOT}/train_data"
VIDEO_DIR="/m2v_intern/xuboshen/zgw/data/ET-Instruct-164K/videos"

JSONL="${JSONL:-proxy_data/data_curation/results/merged/candidates.jsonl}"
MODEL="${MODEL:-pa/gmn-2.5-fl}"
API_BASE="${API_BASE:-https://api.novita.ai/v3/openai}"
WORKERS="${WORKERS:-4}"
```

### Step 0.5: 均衡采样 (可选)

```bash
python "$SCRIPT_DIR/sample_for_demo.py" \
    --input "$JSONL" \
    --output "${DATA_ROOT}/sampled_demo.jsonl" \
    --per-source 50

# 后续步骤使用采样后的 JSONL
JSONL="${DATA_ROOT}/sampled_demo.jsonl"
```

### Step 1: 提取帧 (复用现有 extract_frames.py)

```bash
python "$HIER_SCRIPT_DIR/extract_frames.py" \
    --jsonl "$JSONL" \
    --output-dir "$FRAMES_DIR" \
    --fps 1 \
    --workers "$WORKERS"
```

### Step 2: VLM 标注 — 生成 query + GT

```bash
python "$SCRIPT_DIR/annotate_gseg.py" \
    --jsonl "$JSONL" \
    --frames-dir "$FRAMES_DIR" \
    --output-dir "$ANN_DIR" \
    --api-base "$API_BASE" \
    --model "$MODEL" \
    --max-frames-per-call 64 \
    --workers "$WORKERS"
```

输出示例 (`annotations/{clip_key}.json`):
```json
{
  "clip_key": "video_001",
  "clip_duration_sec": 180,
  "n_tasks": 2,
  "tasks": [
    {
      "query_style": "A",
      "query": "Locate the continuous exercise routine and segment each distinct movement...",
      "grounding": {"start_time": 22, "end_time": 155},
      "segments": [
        {"id": 1, "start_time": 22, "end_time": 38, "label": "Person performs jumping jacks..."},
        {"id": 2, "start_time": 39, "end_time": 52, "label": "Person transitions to push-ups..."}
      ]
    }
  ]
}
```

### Step 3: 构建训练数据

```bash
python "$SCRIPT_DIR/build_gseg_data.py" \
    --annotation-dir "$ANN_DIR" \
    --output-dir "$TRAIN_DIR" \
    --video-dir "$VIDEO_DIR" \
    --min-segments 2 \
    --max-segments 15
```

## 训练数据格式

每条 JSONL 记录:

```json
{
  "messages": [{"role": "user", "content": "Watch the following video...\n<video>\n\n{query}\n\n..."}],
  "prompt": "...",
  "answer": "<events>[[22, 38], [39, 52], ...]</events>",
  "videos": ["/path/to/video.mp4"],
  "data_type": "video",
  "problem_type": "grounding_seg",
  "metadata": {
    "clip_key": "...",
    "query_style": "A",
    "output_count": 5,
    "grounding_start": 22,
    "grounding_end": 155,
    "noise_ratio": 0.25,
    "source_video_path": "..."
  }
}
```

## Query Style 说明

VLM 根据视频内容自动选择:

| Style | 含义 | 典型场景 |
|-------|------|---------|
| **A** | Grounding + Segmentation | 视频有大段无关内容，核心活动需要先定位 |
| **B** | Full Segmentation | 教程/操作类，整个视频都是有结构的 |
| **C** | Cyclic Identification | 周期性重复动作 (健身、组装) |
| **D** | Causal Chain | 有明确起始→终态的过程 |
| **E** | Thread Extraction | 噪音多的视频中提取活动主线 |

## 文件结构

```
grounding_seg/
├── README.md            # 本文件
├── prompts_gseg.py      # VLM 标注 prompt + 学生训练 prompt
├── annotate_gseg.py     # Step 2: VLM 标注管线
├── build_gseg_data.py   # Step 3: 标注 → 训练 JSONL
├── sample_for_demo.py   # Step 0: 按 source 均衡采样
└── run_demo.sh          # 一键 demo 脚本
```

## 服务器目录结构

```
/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/
├── grounding_seg/              ← 本管线数据根目录 (DATA_ROOT)
│   ├── frames/                 ← Step 1 输出: 1fps 帧
│   │   ├── {clip_key}/
│   │   │   ├── 0001.jpg
│   │   │   ├── 0002.jpg
│   │   │   └── meta.json
│   │   └── ...
│   ├── annotations/            ← Step 2 输出: query + GT JSON
│   │   ├── {clip_key}.json
│   │   └── ...
│   └── train_data/             ← Step 3 输出: 训练 JSONL
│       ├── train.jsonl
│       └── val.jsonl
├── hier_seg_annotation/        ← 现有层次分割数据
├── youcook2_seg/
├── youcook2_aot/
└── ...
```

## 与现有管线的关系

- 复用 `hier_seg_annotation/extract_frames.py` 提取帧
- 复用 `hier_seg_annotation/annotate.py` 的 VLM 调用基础设施
- 输出 JSONL 格式与 `build_hier_data.py` 一致，可直接接入训练
- `problem_type` 为 `grounding_seg` (或 `grounding_seg_cot`)，需在 reward 函数中注册
