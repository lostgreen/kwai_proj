# Hierarchical Temporal Segmentation — Annotation Pipeline

3-level hierarchical annotation pipeline for video temporal segmentation.
Domain-agnostic: supports cooking, sports, crafting, repair, and any procedural/activity video.

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
│  Step 1: extract_frames.py                     │
│  从视频中抽取 1fps JPEG 帧                      │
│  输出: frames/{clip_key}/0001.jpg ...          │
└───────────────────┬───────────────────────────┘
                    ▼
┌───────────────────────────────────────────────┐
│  Step 2: annotate.py --level 1                 │
│  L1 Macro Phase Segmentation (阶段级)          │
│  全视频 → 3-5 个宏观阶段                       │
└───────────────────┬───────────────────────────┘
                    ▼
┌───────────────────────────────────────────────┐
│  Step 3: annotate.py --level 2                 │
│  L2 Event Detection (活动级)                   │
│  逐 L1 phase → 检测多秒级活动事件              │
└───────────────────┬───────────────────────────┘
                    ▼
┌───────────────────────────────────────────────┐
│  Step 4: annotate.py --level 3                 │
│  L3 Atomic Action Grounding (动作级)           │
│  逐 L2 event → 原子状态变化定位                 │
└───────────────────┬───────────────────────────┘
                    ▼ (可选)
┌───────────────────────────────────────────────┐
│  Step 4b: annotate_check.py --levels 2c,3c    │
│  Quality Check (粒度审核)                      │
│  L2 粒度 + L3 粒度级联审核                     │
└───────────────────┬───────────────────────────┘
                    ▼
┌───────────────────────────────────────────────┐
│  Step 5: prepare_clips.py                      │
│  物理视频截取 (ffmpeg)                         │
│  输出: clips/L2/*.mp4, clips/L3/*.mp4         │
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
| `prompts.py` | **ACTIVE** | 核心 prompt 模板库 (已泛化为领域无关) |
| `extract_frames.py` | **ACTIVE** | 1fps 抽帧 |
| `annotate.py` | **ACTIVE** | VLM 分层标注 (L1→L2→L3 + Check) |
| `annotate_check.py` | **ACTIVE** | 独立质量审核脚本 |
| `prepare_clips.py` | **ACTIVE** | 物理视频截取 (ffmpeg) |
| `build_dataset.py` | **DEPRECATED** | 已被 `build_hier_data.py` 替代 |
| `sample_mixed_dataset.py` | **DEPRECATED** | 已被 `build_hier_data.py` 内置采样替代 |
| `run_build.sh` | **DEPRECATED** | 旧一键脚本 |

---

## 从 Data Curation 过滤结果到标注的完整步骤

### 前置条件

Data Curation Pipeline 产出的 `vision_results_keep.jsonl` 或 `stage_a_results_keep.jsonl`，每条包含一个候选视频的元信息（视频路径、时长、source 等）。

### Step 1: 抽取帧

```bash
# 从原始视频抽取 1fps JPEG 帧
python proxy_data/youcook2_seg/hier_seg_annotation/extract_frames.py \
  --original-video-root /path/to/videos \
  --output-dir /path/to/hier_seg_annotation/frames \
  --fps 1.0 \
  --workers 8
```

**输入**: 视频文件目录
**输出**: `frames/{clip_key}/0001.jpg, 0002.jpg, ... + meta.json`

> 若使用 JSONL 指定视频列表，添加 `--jsonl path/to/keep.jsonl`

### Step 2: L1 Macro Phase 标注

```bash
export FRAMES_DIR=/path/to/hier_seg_annotation/frames
export ANN_DIR=/path/to/hier_seg_annotation/annotations

python proxy_data/youcook2_seg/hier_seg_annotation/annotate.py \
  --frames-dir $FRAMES_DIR \
  --output-dir $ANN_DIR \
  --level 1 \
  --api-base https://api.novita.ai/v3/openai \
  --model qwen/qwen2.5-vl-72b-instruct \
  --workers 4 \
  --max-frames-per-call 1024
```

L1 将整段视频分割为 3-5 个宏观阶段（如 preparation → execution → finishing）。

### Step 3: L2 Event Detection（依赖 L1）

```bash
python proxy_data/youcook2_seg/hier_seg_annotation/annotate.py \
  --frames-dir $FRAMES_DIR \
  --output-dir $ANN_DIR \
  --level 2 \
  --model qwen/qwen2.5-vl-72b-instruct \
  --workers 4
```

L2 对每个 L1 phase 检测多秒级活动事件（10-60s workflow）。

### Step 4: L3 Atomic Action（依赖 L2）

```bash
python proxy_data/youcook2_seg/hier_seg_annotation/annotate.py \
  --frames-dir $FRAMES_DIR \
  --output-dir $ANN_DIR \
  --level 3 \
  --model qwen/qwen2.5-vl-72b-instruct \
  --workers 4
```

L3 对每个 L2 event 定位原子状态变化（2-6s 物理交互）。

### Step 4b: 质量审核（推荐）

```bash
# 级联审核：L2 check → 孤儿清理 → L3 check
python proxy_data/youcook2_seg/hier_seg_annotation/annotate_check.py \
  --frames-dir $FRAMES_DIR \
  --annotation-dir $ANN_DIR \
  --output-dir /path/to/annotations_checked \
  --levels 2c,3c \
  --model qwen/qwen2.5-vl-72b-instruct \
  --workers 4

# Dry run: 只扫描统计，不调 API
python proxy_data/youcook2_seg/hier_seg_annotation/annotate_check.py ... --dry-run
```

### Step 5: 截取视频 Clips

```bash
# L2 clips (128s sliding window)
python proxy_data/youcook2_seg/hier_seg_annotation/prepare_clips.py \
  --input /path/to/L2_dataset.jsonl \
  --output /path/to/L2_clipped.jsonl \
  --clip-dir /path/to/clips/L2 \
  --workers 8

# L3 clips (event + 5s padding)
python proxy_data/youcook2_seg/hier_seg_annotation/prepare_clips.py \
  --input /path/to/L3_dataset.jsonl \
  --output /path/to/L3_clipped.jsonl \
  --clip-dir /path/to/clips/L3 \
  --workers 8
```

### Step 6: 构建训练数据

```bash
python local_scripts/hier_seg_ablations/build_hier_data.py \
  --annotation-dir /path/to/annotations \
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
  "n_frames": 180,
  "frame_dir": "frames/video_001",
  "level1": {
    "macro_phases": [
      {
        "phase_id": 1,
        "start_time": 0, "end_time": 55,
        "start_frame": 1, "end_frame": 22,
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

## Training Prompt 函数对照

| 函数 | 用途 | 调用者 |
|------|------|--------|
| `get_level1_train_prompt_temporal(duration)` | L1 训练 (时间戳模式, **推荐**) | `build_hier_data.py` |
| `get_level1_train_prompt(n_frames)` | L1 训练 (warped 帧号, 旧版) | — |
| `get_level2_train_prompt(duration)` | L2 训练 | `build_hier_data.py` |
| `get_level3_query_prompt(queries, duration)` | L3 grounding 训练 | `build_hier_data.py` |
| `get_level3_seg_prompt(duration)` | L3 segmentation 训练 | `build_hier_data.py` |
| `get_chain_seg_prompt(events, duration)` | Chain-of-Segment (L2+L3 联合) | chain_seg ablation |
