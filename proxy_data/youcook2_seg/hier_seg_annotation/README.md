# Hierarchical Temporal Segmentation — Annotation Pipeline

3-level hierarchical annotation pipeline for video temporal segmentation.
Domain-agnostic: supports any procedural/activity video (cooking, sports, crafting, repair, etc.).

> **本目录是三条下游 Proxy 流水线的共享标注数据源**：
> - `local_scripts/hier_seg_ablations/build_hier_data.py` — Hierarchical Seg 训练数据
> - `../temporal_aot/build_aot_from_seg.py` — Temporal AoT 训练数据
> - `../event_logic/build_l2_event_logic.py` — Event Logic 训练数据
>
> 通过 `../../shared/seg_source.py` 统一加载和路径命名。扩充标注后，三条流水线自动受益。

---

## 端到端流程总览

```
┌───────────────────────────────────────────────┐
│  上游：Data Curation Pipeline                  │
│  proxy_data/data_curation/                     │
│  text_filter → Stage A (LLM) → VLM Vision     │
│  产出: vision_results_keep.jsonl               │
└───────────────────┬───────────────────────────┘
                    │  过滤后的候选视频列表
                    ▼
┌───────────────────────────────────────────────┐
│  Step 1: extract_frames.py (1fps)              │
│  从原始视频抽取 1fps JPEG 帧（L1+L2 标注用）    │
│  输出: frames/{clip_key}/0001.jpg ...          │
└───────────────────┬───────────────────────────┘
                    ▼
┌───────────────────────────────────────────────┐
│  Step 2: annotate.py --level merged            │
│  L1+L2 合并标注（单次 VLM 调用）               │
│  全视频 → domain(L1/L2) + 宏观阶段 + 活动事件  │
│  输出: annotations/{clip_key}.json             │
└───────────────────┬───────────────────────────┘
                    ▼
┌───────────────────────────────────────────────┐
│  Step 3: extract_frames.py --annotation-dir    │
│  基于 L2 event 边界抽取 2fps 帧（L3 标注用）   │
│  输出: frames_l3/{clip_key}_ev{N}/ ...        │
└───────────────────┬───────────────────────────┘
                    ▼
┌───────────────────────────────────────────────┐
│  Step 4: annotate.py --level 3                 │
│  L3 Atomic Action Grounding（动作级）           │
│  逐 L2 event → 原子状态变化定位（2fps 输入）    │
│  输出: annotations/{clip_key}.json (追加)      │
└───────────────────┬───────────────────────────┘
                    ▼ (可选)
┌───────────────────────────────────────────────┐
│  Step 4b: annotate.py --level 2c / 3c          │
│  Quality Check（粒度审核）                      │
│  L2 粒度 + L3 粒度级联审核与补充               │
└───────────────────┬───────────────────────────┘
                    ▼
┌───────────────────────────────────────────────┐
│  Step 5: prepare_clips.py                      │
│  物理视频截取 (ffmpeg)                          │
│  输出: clips/L1/*.mp4, L2/*.mp4, L3/*.mp4     │
└───────────────────┬───────────────────────────┘
                    ▼
┌───────────────────────────────────────────────┐
│  Step 6: build_hier_data.py                    │
│  构建训练 JSONL (L1/L2/L3/L3_seg)             │
│  输出: *_train.jsonl, *_val.jsonl              │
└───────────────────────────────────────────────┘
```

---

## 文件状态速查

| 文件 | 状态 | 说明 |
|------|------|------|
| `prompts.py` | **ACTIVE** | 核心 prompt 模板库（domain-agnostic，两层领域分类） |
| `extract_frames.py` | **ACTIVE** | 帧抽取：1fps 全视频（L1+L2）/ 2fps per-event（L3） |
| `annotate.py` | **ACTIVE** | VLM 分层标注（merged L1+L2 → L3 → Check） |
| `annotate_check.py` | **ACTIVE** | 独立质量审核脚本 |
| `prepare_clips.py` | **ACTIVE** | 物理视频截取 (ffmpeg)，用于训练数据生成 |
| `build_dataset.py` | **DEPRECATED** | 已被 `build_hier_data.py` 替代 |
| `sample_mixed_dataset.py` | **DEPRECATED** | 已被 `build_hier_data.py` 内置采样替代 |
| `run_build.sh` | **DEPRECATED** | 旧一键脚本 |

---

## 从 Data Curation 过滤结果到标注的完整步骤

### 前置条件

Data Curation Pipeline 产出的 `vision_results_keep.jsonl` 或 `stage_a_results_keep.jsonl`，每条包含一个候选视频的元信息（视频路径、时长、source 等）。

### Step 1: 抽取 1fps 帧（L1+L2 标注用）

```bash
python proxy_data/youcook2_seg/hier_seg_annotation/extract_frames.py \
  --original-video-root /path/to/videos \
  --output-dir /path/to/frames \
  --fps 1.0 \
  --workers 8
```

**输出**: `frames/{clip_key}/0001.jpg, 0002.jpg, ... + meta.json`

> 若使用 JSONL 指定视频列表，添加 `--jsonl path/to/keep.jsonl`

### Step 2: L1+L2 合并标注（单次 VLM 调用）

```bash
export FRAMES_DIR=/path/to/frames
export ANN_DIR=/path/to/annotations

python proxy_data/youcook2_seg/hier_seg_annotation/annotate.py \
  --frames-dir $FRAMES_DIR \
  --output-dir $ANN_DIR \
  --level merged \
  --api-base https://api.novita.ai/v3/openai \
  --model qwen/qwen2.5-vl-72b-instruct \
  --workers 4 \
  --max-frames-per-call 24
```

单次调用同时输出：
- `domain_l1` / `domain_l2`（两层领域分类）
- `level1.macro_phases`（L1：2-6 个宏观阶段）
- `level2.events`（L2：每个 phase 内的活动事件）

### Step 3: 抽取 2fps per-event 帧（L3 标注用）

此步骤读取 merged 标注文件，为每个 L2 event 独立抽帧，精度更高。

```bash
python proxy_data/youcook2_seg/hier_seg_annotation/extract_frames.py \
  --annotation-dir $ANN_DIR \
  --original-video-root /path/to/videos \
  --output-dir /path/to/frames_l3 \
  --fps 2 \
  --workers 8
```

**输出**: `frames_l3/{clip_key}_ev{event_id}/0001.jpg ... + meta.json`
（meta.json 记录 `event_start_sec`，供 annotate.py 恢复绝对时间戳）

### Step 4: L3 Atomic Action Grounding（依赖 Step 2+3）

```bash
python proxy_data/youcook2_seg/hier_seg_annotation/annotate.py \
  --frames-dir $FRAMES_DIR \
  --l3-frames-dir /path/to/frames_l3 \
  --output-dir $ANN_DIR \
  --level 3 \
  --model qwen/qwen2.5-vl-72b-instruct \
  --workers 4
```

- 优先使用 `frames_l3/{clip_key}_ev{N}/`（2fps per-event 帧）
- 未找到时自动回退到 `frames/{clip_key}/`（1fps 全视频帧过滤）

### Step 4b: 质量审核（推荐）

```bash
# 级联审核：L2 check → 孤儿清理 → L3 check
python proxy_data/youcook2_seg/hier_seg_annotation/annotate.py \
  --frames-dir $FRAMES_DIR \
  --l3-frames-dir /path/to/frames_l3 \
  --output-dir $ANN_DIR \
  --level 2c \
  --model qwen/qwen2.5-vl-72b-instruct \
  --workers 4

python proxy_data/youcook2_seg/hier_seg_annotation/annotate.py \
  --frames-dir $FRAMES_DIR \
  --l3-frames-dir /path/to/frames_l3 \
  --output-dir $ANN_DIR \
  --level 3c \
  --model qwen/qwen2.5-vl-72b-instruct \
  --workers 4
```

### Step 5: 截取训练用视频 Clips

```bash
# L2 clips
python proxy_data/youcook2_seg/hier_seg_annotation/prepare_clips.py \
  --input /path/to/L2_dataset.jsonl \
  --output /path/to/L2_clipped.jsonl \
  --clip-dir /path/to/clips/L2 \
  --workers 8

# L3 clips
python proxy_data/youcook2_seg/hier_seg_annotation/prepare_clips.py \
  --input /path/to/L3_dataset.jsonl \
  --output /path/to/L3_clipped.jsonl \
  --clip-dir /path/to/clips/L3 \
  --workers 8
```

### Step 6: 构建训练数据

```bash
python local_scripts/hier_seg_ablations/build_hier_data.py \
  --annotation-dir $ANN_DIR \
  --clip-dir-l2 /path/to/clips/L2 \
  --clip-dir-l3 /path/to/clips/L3 \
  --output-dir /path/to/output \
  --levels L1 L2 L3 L3_seg \
  --total-val 200 \
  --complete-only
```

---

## Annotation JSON 格式

每个标注文件 `annotations/{clip_key}.json` 包含三层层级结构：

```json
{
  "clip_key": "video_001",
  "video_path": "/path/to/video_001.mp4",
  "clip_duration_sec": 180.0,
  "domain_l1": "procedural",
  "domain_l2": "cooking",
  "summary": "Demonstrating how to make pasta from scratch.",
  "n_frames": 180,
  "frame_dir": "frames/video_001",
  "level1": {
    "macro_phases": [
      {
        "phase_id": 1,
        "start_time": 0, "end_time": 55,
        "phase_name": "Material Preparation",
        "narrative_summary": "Gather and organize required materials."
      }
    ]
  },
  "level2": {
    "events": [
      {
        "event_id": 1,
        "start_time": 5, "end_time": 30,
        "parent_phase_id": 1,
        "instruction": "Measure and cut materials to required dimensions",
        "visual_keywords": ["ruler", "cutting tool", "material piece"]
      }
    ]
  },
  "level3": {
    "grounding_results": [
      {
        "action_id": 1,
        "start_time": 5, "end_time": 10,
        "parent_event_id": 1,
        "sub_action": "Mark measurement lines on the material",
        "pre_state": "Unmarked flat material on work surface",
        "post_state": "Visible measurement marks drawn on material"
      }
    ]
  }
}
```

---

## 两层领域分类（Domain Taxonomy）

领域分类在 merged 标注阶段由 VLM 自动判定，写入 `domain_l1` 和 `domain_l2`。

| L1 大类 | L2 细类 |
|---------|---------|
| `procedural` | `cooking`, `construction_building`, `crafting_diy`, `repair_maintenance` |
| `physical` | `sports`, `fitness_exercise`, `music_performance` |
| `lifestyle` | `beauty_grooming`, `cleaning_housework`, `gardening_outdoor`, `vehicle_operation` |
| `educational` | `science_experiment` |
| `other` | `other` |

定义位于 `prompts.py` 中的 `DOMAIN_TAXONOMY`（唯一数据源）。

---

## 三层标注层级定义

| Level | 名称 | 粒度 | 典型时长 | 定义 |
|-------|------|------|----------|------|
| **L1** | Macro Phase | 阶段级 | 30-120s | 宏观活动阶段，如 preparation → execution → finishing |
| **L2** | Activity Event | 活动级 | 10-60s | 目标导向的工作流，改变材料/对象状态或完成过程子目标 |
| **L3** | Atomic Action | 动作级 | 2-6s | 单一不可逆物理状态变化，一个对象一次变化 |

**粒度光谱判据**（用于审核）：
- L2 不能粗到等于 L1（复述整个阶段），也不能细到等于 L3（单个瞬时动作）
- L3 不能粗到等于 L2（多步骤工作流），也不能细到无物理变化（纯手部运动）

---

## FPS 设计说明

| 层级 | 帧率 | 来源 | 原因 |
|------|------|------|------|
| L1+L2 标注 | 1fps | `extract_frames.py` 全视频 | 全局时间感知，1fps 足够捕捉阶段/事件边界 |
| L3 标注 | 2fps | `extract_frames.py --annotation-dir` per-event | 原子动作 2-6s，需要更高密度帧 |
| 训练用 L1 clip | 1fps | `prepare_clips.py _process_l1` | 保留时序感，低传输成本 |
| 训练用 L2 clip | stream copy | `prepare_clips.py _process_l2` | 保留原视频质量 |
| 训练用 L3 clip | stream copy | `prepare_clips.py _process_l3` | 保留原视频质量 |

---

## Training Prompt 函数对照

| 函数 | 用途 | 调用者 |
|------|------|--------|
| `get_merged_l1l2_prompt(n_frames, duration)` | L1+L2 合并标注（标注用） | `annotate.py` |
| `get_level1_train_prompt_temporal(duration)` | L1 训练（时间戳模式，推荐） | `build_hier_data.py` |
| `get_level1_train_prompt(n_frames)` | L1 训练（warped 帧号，旧版） | — |
| `get_level2_train_prompt(duration)` | L2 训练 | `build_hier_data.py` |
| `get_level3_query_prompt(queries, duration)` | L3 grounding 训练 | `build_hier_data.py` |
| `get_level3_seg_prompt(duration)` | L3 segmentation 训练 | `build_hier_data.py` |
| `get_chain_seg_prompt(events, duration)` | Chain-of-Segment（L2+L3 联合） | chain_seg ablation |
