# Data Curation — 数据筛选与候选构建

从多数据源中筛选适合进入 **层次分割标注 (Hierarchical Segmentation Annotation)** 的视频样本。

---

## 设计目标

1. **文本先行**：利用已有文本标注快速过滤，降低 GPU 使用
2. **本地 VLM 筛选**：用 Qwen3-VL-4B (vLLM) 直接看视频评估层次结构潜力
3. **可选二阶段**：Stage 2 针对 repetitive-loop 假阳性做补充过滤
4. **断点续跑**：`--resume` 自动跳过已有结果
5. **可扩展**：每个数据源独立目录，新增数据源只需添加一个子目录 + 配置
6. **可追溯**：每步过滤均保留 passed / rejected 列表及原因

---

## 目录结构

```
data_curation/
├── README.md                          ← 本文件
├── PIPELINE_REPORT.md                 ← Pipeline 设计汇报 (legacy)
│
├── configs/                           ← 筛选配置（每数据源一份 YAML）
│   ├── et_instruct_164k.yaml
│   └── timelens_100k.yaml
│
├── shared/                            ← 共享代码模块
│   ├── local_screen.py               ← 核心: 本地 VLM 视觉预筛选 (vLLM batch inference)
│   └── visualize_distribution.py     ← 数据分布可视化（source/duration）
│
├── et_instruct_164k/                  ← ET-Instruct-164K 数据源脚本
│   ├── text_filter.py                ← Step 1: 元数据过滤
│   ├── sample_per_source.py          ← Step 2: 格式转换 + 可选采样
│   ├── explore_data.py               ← 数据探索
│   └── run_pipeline.sh               ← 一键运行脚本
│
├── timelens_100k/                     ← TimeLens-100K 数据源脚本
│   ├── text_filter.py                ← Step 1: 元数据过滤
│   ├── sample_per_source.py          ← Step 2: 格式转换 + 可选采样
│   └── run_pipeline.sh               ← 一键运行脚本
│
└── results/                           ← 所有数据产出（代码与数据分离）
    ├── et_instruct_164k/
    └── timelens_100k/
```

---

## 三步筛选流水线

```
[数据源 JSON/JSONL]
  │
  ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Step 1: 元数据过滤 — text_filter.py                                  │
│  • 60s ≤ duration ≤ 240s                                             │
│  • (可选) events ≥ N                                                 │
│  • dedup by video                                                    │
│  • domain cap                                                        │
│  → passed.jsonl                                                      │
└──────────────────────────────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Step 2: 格式转换 + 采样 — sample_per_source.py                       │
│  • 原始格式 → 统一 JSONL (videos[], metadata{}, source, duration)    │
│  • 可选每 source 抽 N 条                                             │
│  → sample_dev.jsonl                                                  │
└──────────────────────────────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Step 3: 本地 VLM 预筛选 — shared/local_screen.py (Qwen3-VL-4B)      │
│                                                                      │
│  Stage 1 (必选): 四维评估                                            │
│    • L1_SCORE (1-5): phase structure — 视频有无多阶段结构             │
│    • L2_SCORE (1-5): event structure — 阶段内有无子事件结构           │
│    • DOMAIN: domain_l1 / domain_l2 分类                              │
│    • QUALITY: good / bad 视觉质量                                    │
│    决策: L1≥3 AND L2≥3 AND quality==good → keep                     │
│                                                                      │
│  Stage 2 (可选, --secondary_screen): 时序进展质量                    │
│    • PROG_TYPE: procedural / narrative / repetitive_loop             │
│    • VISUAL_DIVERSITY: high / medium / low                           │
│    • ORDER_DEPENDENCY: strict / loose / none                         │
│    决策: repetitive_loop → reject; low+none → reject; else keep     │
│                                                                      │
│  → screen_keep.jsonl (最终候选 → 标注流水线)                         │
│  → screen_reject.jsonl (淘汰)                                        │
│  → screen_results.jsonl (全部结果含 _screen / _screen_2)             │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 数据流 & 输出 Schema

### 统一记录格式 (sample_dev.jsonl → screen_keep.jsonl)

```jsonc
{
  // ─── 基础字段 (downstream: extract_frames.py / annotate.py) ───
  "videos": ["/abs/path/to/video.mp4"],      // 绝对视频路径
  "metadata": {
    "clip_key": "video_stem",                 // 唯一标识 (Path stem)
    "video_id": "video_stem",
    "clip_start": 0,
    "clip_end": 120.5,                        // = duration
    "clip_duration": 120.5,
    "original_duration": 120.5,
    "is_full_video": true,
    "source": "coin"
  },
  "source": "coin",
  "dataset": "ET-Instruct-164K",             // or "TimeLens-100K"
  "duration": 120.5,

  // ─── Stage 1 筛选字段 ───
  "_screen": {
    "l1_score": 4,                            // 1-5
    "est_phases": 3,
    "l2_score": 4,                            // 1-5
    "est_events": 8,
    "domain_l1": "procedural",
    "domain_l2": "cooking",
    "quality": "good",
    "reason": "Video shows distinct phases...",
    "decision": "keep"                        // or "reject"
  },

  // ─── Stage 2 筛选字段 (仅 --secondary_screen) ───
  "_screen_2": {
    "prog_type": "procedural",               // procedural / narrative / repetitive_loop
    "visual_diversity": "high",              // high / medium / low
    "order_dependency": "strict",            // strict / loose / none
    "reason": "Objects change permanently...",
    "decision": "keep"                       // or "reject"
  },

  // ─── 溯源字段 (来自 sample_per_source.py) ───
  "_et_raw": { ... },                        // ET-Instruct 原始字段
  "_origin": { ... }                         // 筛选来源信息
}
```

### 字段用途对照

| 字段 | 阶段 | 用途 |
|------|------|------|
| `videos`, `metadata.*`, `duration` | 基础 | extract_frames.py / annotate.py 必需 |
| `source`, `dataset` | 基础 | 域均衡采样 |
| `_screen.l1_score/l2_score` | 筛选 | 层次结构潜力判断 |
| `_screen.domain_l1/l2` | 筛选 | 域均衡参考 (annotate.py 会独立分类) |
| `_screen.quality` | 筛选 | 视觉质量门槛 |
| `_screen_2.prog_type` | 筛选 | repetitive-loop 过滤 |
| `_screen_2.visual_diversity` | 筛选 | 视觉多样性门槛 |
| `_screen_2.order_dependency` | 筛选 | 时序依赖性门槛 |
| `_et_raw`, `_tl_raw`, `_origin` | 溯源 | 可追溯原始数据 |

---

## 运行指令

### 一键 Pipeline (推荐)

```bash
# ET-Instruct-164K
cd proxy_data/data_curation/et_instruct_164k/
bash run_pipeline.sh                              # 默认 2 GPU, resume=on

# TimeLens-100K
cd proxy_data/data_curation/timelens_100k/
bash run_pipeline.sh

# 常用环境变量
NUM_GPUS=8 bash run_pipeline.sh                   # 8 卡数据并行
PER_SOURCE=5 bash run_pipeline.sh                 # 每 source 抽 5 条试跑
SECONDARY=1 bash run_pipeline.sh                  # 开启二阶段筛选
RESUME=0 bash run_pipeline.sh                     # 关闭断点续跑 (全量重跑)
```

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LOCAL_MODEL` | `/home/xuboshen/models/Qwen3-VL-4B-Instruct` | 本地 VLM 路径 |
| `NUM_GPUS` | `2` | 数据并行 GPU 数 |
| `PER_SOURCE` | `0` | 每 source 采样条数 (0 = 全量) |
| `SECONDARY` | `0` | 二阶段筛选 (1 = 开启) |
| `RESUME` | `1` | 断点续跑 (1 = 跳过已有结果) |
| `OUTPUT_ROOT` | `../results/{source}/` | 输出目录 |

### local_screen.py 单独运行

```bash
# 单 GPU，自动续跑
python shared/local_screen.py \
    --input_jsonl results/sample_dev.jsonl \
    --output_jsonl results/screen_results.jsonl \
    --keep_jsonl results/screen_keep.jsonl \
    --reject_jsonl results/screen_reject.jsonl \
    --model_path /path/to/Qwen3-VL-4B-Instruct \
    --resume

# 单 GPU + 二阶段
python shared/local_screen.py \
    --input_jsonl results/sample_dev.jsonl \
    --output_jsonl results/screen_results.jsonl \
    --keep_jsonl results/screen_keep.jsonl \
    --reject_jsonl results/screen_reject.jsonl \
    --model_path /path/to/Qwen3-VL-4B-Instruct \
    --resume --secondary_screen

# 多 GPU 数据并行 (8 卡)
for i in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=$i python shared/local_screen.py \
        --input_jsonl results/sample_dev.jsonl \
        --output_jsonl results/screen_shard${i}.jsonl \
        --keep_jsonl results/keep_shard${i}.jsonl \
        --reject_jsonl results/reject_shard${i}.jsonl \
        --model_path /path/to/Qwen3-VL-4B-Instruct \
        --shard_id $i --num_shards 8 --resume &
done
wait
cat results/keep_shard*.jsonl > results/screen_keep.jsonl
```

---

## 产出文件

| 文件 | 说明 |
|------|------|
| `passed.jsonl` | Step 1 文本过滤后的候选 |
| `sample_dev.jsonl` | Step 2 统一格式 (local_screen 输入) |
| `screen_results.jsonl` | 全部结果 (含 `_screen` / `_screen_2` 字段) |
| `screen_keep.jsonl` | **最终候选** → 下游标注 (extract_frames.py) |
| `screen_reject.jsonl` | 被拒绝的记录 |

---

## 与下游标注的衔接

```
screen_keep.jsonl
    │
    ▼  extract_frames.py (reads: videos, metadata.clip_key/clip_start/clip_end)
    │  → frames/{clip_key}/ (1fps JPEGs + meta.json)
    │
    ▼  annotate.py --level merged (reads: frame directories)
    │  → annotations/{clip_key}.json (L1+L2 hierarchical annotation)
    │
    ▼  annotate.py --level 3 (reads: per-event frames)
    │  → annotations/{clip_key}.json (adds L3 grounding)
    │
    ▼  build_hier_data.py (reads: annotations/*.json)
    │  → train.jsonl / val.jsonl (for RL training)
    │
    ▼  prepare_clips.py (reads: train/val JSONL)
       → L1/L2/L3 video clips for training
```

---

## 新增数据源流程

1. 创建数据源目录：`{source_name}/`
2. 实现 `text_filter.py`（元数据过滤）
3. 实现 `sample_per_source.py`（格式转换为统一 schema）
4. 在 `configs/` 下新建 `{source_name}.yaml` 配置文件
5. 创建 `run_pipeline.sh`（可复制现有模板）
6. 运行完整 pipeline，结果写入 `results/{source_name}/`
