# YouCook2 Hierarchical DVC Annotation Pipeline

3-level hierarchical Dense Video Captioning (DVC) annotation tool for YouCook2 windowed clips.

## 核心设计理念

**标注阶段**和**训练数据构造**对 Level 2 采用不同的策略：

- **标注时**：L2 依赖 L1 的 macro phase 结果，逐 phase 检测 events（不做滑窗）。这保证了每个 event 的标注有清晰的 phase 上下文，标注质量更高。
- **训练时**：`build_dataset.py` 无视 L1 的 phase 划分，在全视频时间线上构造 128s 滑窗，筛选出窗口内包含 ≥N 个 event 的窗口作为训练样本。这让模型学习在任意时间窗口内检测多个 events。

```
标注流水线:
  L1 (全视频) ──► macro phases
       │
       ▼
  L2 (逐 phase) ──► events（依赖 L1 phase 范围）
       │
       ▼
  L3 (逐 event) ──► atomic grounding（依赖 L2 event 范围）

训练数据构造:
  L1: 每个 clip 一条训练样本
  L2: 128s 滑窗 × 筛选多 event 窗口（无视 L1 phase）
  L3: 每个 L2 event 一条训练样本
```

## Directory layout

```
youcook2_seg_annotation/
  prompts.py          — Annotation prompts (Level 0-3)
  extract_frames.py   — ffmpeg 1fps frame extraction
  annotate.py         — LLM annotation pipeline (OpenAI-compatible VLM API)
  build_dataset.py    — Convert annotations → EasyR1 training JSONL
  README.md
  frames/             — extracted frames (git-ignored)
  annotations/        — per-clip annotation JSONs (git-ignored)
```

## Annotation levels

| Level | Name | 标注策略 | 训练数据策略 | Prompt |
|---|---|---|---|---|
| 0 | System Prompt | — | — | `prompts.SYSTEM_PROMPT` |
| 1 | Macro Phase (阶段级) | 全视频 uniform sampling, warped timeline | 每 clip 一条 | `prompts.get_level1_prompt()` |
| 2 | Activity-level (活动级) | **逐 L1 phase** 检测 events | **128s 滑窗**，筛选 ≥2 events 的窗口 | `prompts.get_level2_prompt()` |
| 3 | Atomic Step (动作级) | 逐 L2 event 做 temporal grounding | 每 event 一条 | `prompts.get_level3_prompt()` |

## Quick start

### Step 1: Extract 1fps frames

```bash
python proxy_data/youcook2_seg_annotation/extract_frames.py \
  --jsonl proxy_data/youcook2_train_easyr1.jsonl \
  --video-dir /path/to/Youcook2_windowed \
  --output-dir proxy_data/youcook2_seg_annotation/frames \
  --fps 1.0 \
  --workers 8
```

### Step 2: Run Level 1 annotation

L1 对全视频做 warped-time macro phase 分割：

```bash
python /home/xuboshen/zgw/EasyR1/proxy_data/youcook2_seg_annotation/annotate.py \
  --frames-dir /m2v_intern/xuboshen/zgw/data/youcook2_seg_annotation/frames \
  --output-dir /m2v_intern/xuboshen/zgw/data/youcook2_seg_annotation/annotations \
  --level 1 \
  --api-base https://api.novita.ai/v3/openai \
  --model pa/gmn-2.5-pr \
  --workers 4 \
  --limit 50 \
  --max-frames-per-call 1024
```

### Step 3: Run Level 2 annotation（依赖 L1）

L2 读取 L1 的 phase 结果，逐 phase 检测 cooking events。**必须先完成 L1**。

```bash
python /home/xuboshen/zgw/EasyR1/proxy_data/youcook2_seg_annotation/annotate.py \
  --frames-dir /m2v_intern/xuboshen/zgw/data/youcook2_seg_annotation/frames \
  --output-dir /m2v_intern/xuboshen/zgw/data/youcook2_seg_annotation/annotations \
  --level 2 \
  --api-base https://api.novita.ai/v3/openai \
  --model pa/gmn-2.5-pr \
  --workers 4 \
  --limit 10 \
  --max-frames-per-call 1024
```

### Step 4: Run Level 3 annotation（依赖 L2）

L3 读取 L2 的 event 结果，逐 event 做 atomic temporal grounding。**必须先完成 L2**。

```bash
python /home/xuboshen/zgw/EasyR1/proxy_data/youcook2_seg_annotation/annotate.py \
  --frames-dir /m2v_intern/xuboshen/zgw/data/youcook2_seg_annotation/frames \
  --output-dir /m2v_intern/xuboshen/zgw/data/youcook2_seg_annotation/annotations \
  --level 3 \
  --api-base https://api.novita.ai/v3/openai \
  --model pa/gmn-2.5-pr \
  --workers 4 \
  --limit 10 \
  --max-frames-per-call 1024
```

### Step 5: Build training dataset

```bash
# Level 1: 每个 clip 一条训练样本
python build_dataset.py \
  --annotation-dir annotations \
  --output youcook2_hier_L1_train.jsonl \
  --level 1

# Level 2: 128s 滑窗，只保留包含 ≥2 个 event 的窗口
python build_dataset.py \
  --annotation-dir annotations \
  --output youcook2_hier_L2_train.jsonl \
  --level 2 \
  --l2-window-size 128 \
  --l2-stride 64 \
  --l2-min-events 2

# Level 3: 每个 L2 event 一条训练样本
python build_dataset.py \
  --annotation-dir annotations \
  --output youcook2_hier_L3_train.jsonl \
  --level 3
```

## Annotation JSON format

```json
{
  "clip_key": "GLd3aX16zBg_90_174",
  "video_path": "/m2v_intern/.../GLd3aX16zBg_90_174.mp4",
  "clip_duration_sec": 84.0,
  "n_frames": 84,
  "frame_dir": "frames/GLd3aX16zBg_90_174",
  "level1": {
    "macro_phases": [
      {"phase_id": 1, "start_time": 0, "end_time": 18,
       "start_frame": 1, "end_frame": 7,
       "phase_name": "Ingredient Preparation",
       "narrative_summary": "Spreads margarine on bread slices."}
    ]
  },
  "level2": {
    "events": [
      {"event_id": 1, "start_time": 2, "end_time": 16,
       "parent_phase_id": 1,
       "instruction": "Spread margarine evenly on sliced bread",
       "visual_keywords": ["bread", "margarine", "spreading knife"]}
    ]
  },
  "level3": {
    "grounding_results": [
      {"action_id": 1, "start_time": 3, "end_time": 8,
       "parent_event_id": 1,
       "sub_action": "Scoop margarine and spread on first slice",
       "pre_state": "Dry bread slice on cutting board",
       "post_state": "Thin layer of margarine covering the bread surface"}
    ]
  },
  "annotated_at": "2025-..."
}
```
