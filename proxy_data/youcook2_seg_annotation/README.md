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
       │
       ▼ (可选)
  L2c + L3c (审核) ──► 粒度光谱审核 → keep/revise/remove/supplement

训练数据构造:
  L1: 每个 clip 一条训练样本
  L2: 128s 滑窗 × 筛选多 event 窗口（无视 L1 phase）
  L3: 每个 L2 event 一条训练样本，两种模式可选：
      --level 3  (query grounding): 给定 action 文本列表，按序定位
      --level 3s (segmentation):    无 query，自行检测所有原子动作
```

## Directory layout

```
youcook2_seg_annotation/
  prompts.py          — Annotation + check prompts (Level 0-3, L2c/L3c)
  extract_frames.py   — ffmpeg 1fps frame extraction
  annotate.py         — LLM annotation pipeline (OpenAI-compatible VLM API)
  annotate_check.py   — Standalone quality audit script (L2+L3 check)
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
| 2c | L2 Check (审核) | 逐 L1 phase 审核 L2 events | — | `prompts.get_level2_check_prompt()` |
| 3 | Atomic Step (动作级) | 逐 L2 event 做 temporal grounding | 每 event 一条（query / seg 两种模式） | `prompts.get_level3_prompt()` |
| 3c | L3 Check (审核) | 逐 L2 event 审核 L3 actions | — | `prompts.get_level3_check_prompt()` |

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
  --limit 500 \
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
  --limit 500 \
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
  --limit 100 \
  --max-frames-per-call 1024
```

### Step 4b: Quality Check — L2 / L3 审核（可选但推荐）

使用更强的模型对 L2/L3 标注进行粒度审核。审核基于**粒度光谱（Granularity Spectrum）**判据：

- **L2 check**：逐 L1 phase 审核 L2 events，判断每个 event 粒度是否正确（不过粗如 L1，不过细如 L3）
- **L3 check**：逐 L2 event 审核 L3 atomic actions，判断每个 action 是否为真正的原子状态变化

两种运行方式：

**方式 1：通过 `annotate.py` 内嵌运行**

```bash
# L2 审核（需要 L1+L2 已完成）
python annotate.py \
  --frames-dir frames/ --output-dir annotations/ \
  --level 2c --model gpt-4o --workers 4

# L3 审核（需要 L2+L3 已完成）
python annotate.py \
  --frames-dir frames/ --output-dir annotations/ \
  --level 3c --model gpt-4o --workers 4
```

**方式 2：通过独立审核脚本 `annotate_check.py`（推荐）**

独立脚本支持 L2+L3 级联审核，输出到单独目录不覆盖原始标注：

```bash
# 级联审核：L2 check → 孤儿清理 → L3 check
python annotate_check.py \
  --frames-dir frames/ \
  --annotation-dir annotations/ \
  --output-dir annotations_checked/ \
  --levels 2c,3c \
  --model gpt-4o \
  --workers 4

# Dry run：只扫描统计，不调 API
python annotate_check.py ... --dry-run

# 只做 L2 审核
python annotate_check.py ... --levels 2c
```

审核后的 JSON 新增字段：
- `level2._check_stats` / `level3._check_stats`：kept/revised/removed/supplemented 统计
- 每个被修改的 event/action 带 `_checked` 标记（`"revised"` / `"supplemented"`）
- `_audit_meta`：审核模型、时间、原始标注路径

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

# Level 3 Query Grounding (--level 3): 给定 action caption 列表，按序定位
#   --l3-order sequential: 原始时序顺序
#   --l3-order shuffled:   打乱顺序（测试模型视觉理解而非序列先验）
#   --l3-order both:       两种都生成，每个 event 产出 2 条
python build_dataset.py \
  --annotation-dir annotations \
  --output youcook2_hier_L3_train.jsonl \
  --level 3 \
  --l3-min-actions 3 \
  --l3-order both

# Level 3 Segmentation (--level 3s): 无 query，模型自行检测所有原子动作
#   使用 F1-IoU reward（与 L1/L2 相同），段数不定
python build_dataset.py \
  --annotation-dir annotations \
  --output youcook2_hier_L3_seg_train.jsonl \
  --level 3s \
  --l3-min-actions 3

DATA_PATH=/m2v_intern/xuboshen/zgw/data/youcook2_seg_annotation/datasets/youcook2_hier_L3_train.jsonl \
  PORT=8890 \
  MAX_SAMPLES=10 \
  bash data_visualization/segmentation_visualize/run.sh
```

说明：

- L3 有两种训练模式：
  - **Query Grounding** (`--level 3`)：给定 action caption 列表，模型按序输出每个 action 的 `[start, end]`。Reward 用 position-aligned mean tIoU。
  - **Segmentation** (`--level 3s`)：不给 query，模型自行检测所有原子动作。Reward 用 F1-IoU（与 L1/L2 相同）。
- L3 训练样本会过滤，只保留在对应 L2 event 内至少有 `3` 个 atomic actions 的样本（`--l3-min-actions`）。
- 如果使用 `annotate_check.py` 审核后的标注（`annotations_checked/`），将 `--annotation-dir` 指向审核后的目录即可。

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
