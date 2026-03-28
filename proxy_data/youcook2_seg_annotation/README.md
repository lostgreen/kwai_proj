# YouCook2 Hierarchical DVC Annotation Pipeline

3-level hierarchical Dense Video Captioning (DVC) annotation tool for YouCook2 windowed clips.

## 文件状态速查

| 文件 | 状态 | 说明 |
|------|------|------|
| `prompts.py` | **ACTIVE** | 核心 prompt 模板库，被 `build_hier_data.py` 和标注工具链 import |
| `extract_frames.py` | Annotation-only | 标注阶段 1fps 抽帧，标注完成后无需运行 |
| `annotate.py` | Annotation-only | VLM 分层标注 (L1→L2→L3)，标注完成后无需运行 |
| `annotate_check.py` | Annotation-only | L2/L3 质量审核，审核完成后无需运行 |
| `build_dataset.py` | **DEPRECATED** | 已被 `local_scripts/hier_seg_ablations/build_hier_data.py` 替代 |
| `prepare_clips.py` | **ACTIVE** | 物理视频截取 (ffmpeg)，生成 `clips/L2/` 和 `clips/L3/` |
| `sample_mixed_dataset.py` | **DEPRECATED** | 已被 `build_hier_data.py` 内置采样替代 |
| `run_build.sh` | **DEPRECATED** | 旧流水线一键脚本 (因 build_dataset.py 损坏已失效) |

## 数据构造流程

### 新流程 (推荐) — 一步构建

```bash
# 从 annotation JSON 直接构建训练数据 (支持 L1/L2/L3/L3_seg)
python local_scripts/hier_seg_ablations/build_hier_data.py \
  --annotation-dir /path/to/annotations \
  --clip-dir-l2 /path/to/clips/L2 \
  --clip-dir-l3 /path/to/clips/L3 \
  --output-dir /path/to/output \
  --levels L1 L2 L3_seg \
  --complete-only
```

新流程的改进：
- L1 改为真实时间戳 (秒数)，不再使用 warped 帧号
- L2 直接在构建时归零到窗口相对坐标
- 一步生成 train.jsonl / val.jsonl，无需中间 `_clipped.jsonl`

### 旧流程 (5 步，已废弃)

```
extract_frames.py → annotate.py → build_dataset.py → prepare_clips.py → sample_mixed_dataset.py
```

Note: `build_dataset.py` 的 `main()` 已损坏 (unreachable code bug)。旧流水线不再可运行。

## Annotation levels

| Level | Name | 标注策略 | Prompt |
|---|---|---|---|
| 0 | System Prompt | — | `prompts.SYSTEM_PROMPT` |
| 1 | Macro Phase (阶段级) | 全视频 uniform sampling, warped timeline | `prompts.get_level1_prompt()` |
| 2 | Activity-level (活动级) | **逐 L1 phase** 检测 events | `prompts.get_level2_prompt()` |
| 2c | L2 Check (审核) | 逐 L1 phase 审核 L2 events | `prompts.get_level2_check_prompt()` |
| 3 | Atomic Step (动作级) | 逐 L2 event 做 temporal grounding | `prompts.get_level3_prompt()` |
| 3c | L3 Check (审核) | 逐 L2 event 审核 L3 actions | `prompts.get_level3_check_prompt()` |

## Training prompts

| 函数 | 用途 |
|------|------|
| `get_level1_train_prompt_temporal(duration)` | L1 训练 (真实时间戳模式, **推荐**) |
| `get_level1_train_prompt(n_frames)` | L1 训练 (warped 帧号, 旧版) |
| `get_level2_train_prompt(duration)` | L2 训练 |
| `get_level3_query_prompt(queries, duration)` | L3 grounding 训练 |
| `get_level3_seg_prompt(duration)` | L3 segmentation 训练 |
| `get_chain_seg_prompt(events, duration)` | Chain-of-Segment (L2+L3 联合) |

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

> **已废弃**: 旧的 `build_dataset.py` 已被替代。请使用新的统一构建脚本。

```bash
# 新方式 (推荐): 一步构建
python local_scripts/hier_seg_ablations/build_hier_data.py \
  --annotation-dir /path/to/annotations \
  --clip-dir-l2 /path/to/clips/L2 \
  --clip-dir-l3 /path/to/clips/L3 \
  --output-dir /path/to/output \
  --levels L1 L2 L3_seg \
  --total-val 200 \
  --complete-only
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
