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
│  产出: dev_100.jsonl / sampled_1k.jsonl        │
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
│  L1+L2+Topology+Criterion 合并标注             │
│  输出: annotations/{clip_key}.json             │
└───────────────────┬───────────────────────────┘
                    ▼
┌───────────────────────────────────────────────┐
│  Step 3: extract_frames.py --annotation-dir    │
│  Leaf-node 路由: L3 帧提取                     │
│  有 events → _ev{id}/  无 events → _ph{id}/   │
│  输出: frames_l3/{clip_key}_{ev|ph}{N}/ ...   │
└───────────────────┬───────────────────────────┘
                    ▼
┌───────────────────────────────────────────────┐
│  Step 4: annotate.py --level 3                 │
│  L3 Micro Grounding (leaf-node 路由)           │
│  输出: annotations/{clip_key}.json (追加 L3)   │
└───────────────────┬───────────────────────────┘
                    ▼ (可选)
┌───────────────────────────────────────────────┐
│  Step 5: rewrite_criteria_hints.py             │
│  Criterion → 通用 Training Hint 改写           │
│  输出: 同一 JSON 中追加 *_hint 字段            │
└───────────────────┬───────────────────────────┘
                    ▼
┌───────────────────────────────────────────────┐
│  Step 6: prepare_clips.py + build_hier_data.py │
│  物理视频截取 → 构建训练 JSONL                  │
└───────────────────────────────────────────────┘
```

---

## 文件状态速查

| 文件 | 状态 | 说明 |
|------|------|------|
| `prompts.py` | **ACTIVE** | 核心 prompt 模板库（domain-agnostic，topology-adaptive，含 criterion 指令） |
| `extract_frames.py` | **ACTIVE** | 帧抽取：1fps 全视频（L1+L2）/ 2fps leaf-node（L3） |
| `annotate.py` | **ACTIVE** | VLM 分层标注（merged L1+L2+Topology → L3 leaf-node → Check） |
| `annotate_check.py` | **ACTIVE** | 独立质量审核脚本 |
| `prepare_clips.py` | **ACTIVE** | 物理视频截取 (ffmpeg)，用于训练数据生成 |
| `rewrite_criteria_hints.py` | **ACTIVE** | LLM 后处理：将 criterion 改写为通用 training hint |
| `build_dataset.py` | **DEPRECATED** | 已被 `build_hier_data.py` 替代 |
| `sample_mixed_dataset.py` | **DEPRECATED** | 已被 `build_hier_data.py` 内置采样替代 |
| `run_build.sh` | **DEPRECATED** | 旧一键脚本 |

---

## 从 Data Curation 过滤结果到标注的完整步骤

### 路径约定

```bash
SCRIPT_DIR=proxy_data/youcook2_seg/hier_seg_annotation
DATA_ROOT=/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/hier_seg_annotation

JSONL_DEV=/home/xuboshen/zgw/EasyR1/proxy_data/data_curation/results/merged/sampled/dev_100.jsonl
JSONL_1K=/home/xuboshen/zgw/EasyR1/proxy_data/data_curation/results/merged/sampled/sampled_1k.jsonl

MODEL=pa/gemini-3.1-pro-preview
```

### Step 1: 抽取 1fps 帧（L1+L2 标注用，仅首次）

```bash
python $SCRIPT_DIR/extract_frames.py \
    --jsonl $JSONL_DEV \
    --output-dir $DATA_ROOT/frames \
    --fps 1 --workers 4
```

**输出**: `frames/{clip_key}/0001.jpg, 0002.jpg, ... + meta.json`

### Step 2: Merged 标注（L1+L2+Topology+Criterion）

```bash
python $SCRIPT_DIR/annotate.py \
    --jsonl $JSONL_DEV \
    --frames-dir $DATA_ROOT/frames \
    --output-dir $DATA_ROOT/annotations \
    --level merged \
    --model $MODEL --workers 4
```

单次调用同时输出：
- `topology_type` / `topology_confidence`（时间拓扑分类）
- `global_phase_criterion`（L1 切分依据）
- `level1.macro_phases`（L1 + 每个 phase 的 `event_split_criterion`）
- `level2.events`（L2，procedural 单一动作 phase 允许 `events: []`）

### Step 3: L3 帧提取（leaf-node 路由）

基于 merged 标注，自动按 leaf-node 逻辑抽帧：
- 有 events 的 phase → events 各自抽帧 (`_ev{id}/`)
- 无 events 的 phase → phase 整体抽帧 (`_ph{id}/`)

```bash
python $SCRIPT_DIR/extract_frames.py \
    --annotation-dir $DATA_ROOT/annotations \
    --output-dir $DATA_ROOT/frames_l3 \
    --fps 2 --workers 4
```

### Step 4: L3 标注（leaf-node 路由，自动跳过 sequence/flat）

```bash
python $SCRIPT_DIR/annotate.py \
    --jsonl $JSONL_DEV \
    --frames-dir $DATA_ROOT/frames \
    --l3-frames-dir $DATA_ROOT/frames_l3 \
    --output-dir $DATA_ROOT/annotations \
    --level 3 \
    --model $MODEL --workers 8
```

- 输出 `micro_split_criterion`（L3 切分依据）
- 优先使用 `frames_l3/` 的 2fps 帧，未找到时回退到 `frames/` 的 1fps 帧

### Step 5 (可选): Criterion → 通用 Training Hint 改写

```bash
python $SCRIPT_DIR/rewrite_criteria_hints.py \
    --annotation-dir $DATA_ROOT/annotations \
    --api-base $API_BASE \
    --model gpt-4o-mini --workers 4
```

### Step 6 (可选): 质量审核

```bash
python $SCRIPT_DIR/annotate.py \
    --jsonl $JSONL_DEV \
    --frames-dir $DATA_ROOT/frames \
    --l3-frames-dir $DATA_ROOT/frames_l3 \
    --output-dir $DATA_ROOT/annotations \
    --level 2c --model $MODEL --workers 4

python $SCRIPT_DIR/annotate.py \
    --jsonl $JSONL_DEV \
    --frames-dir $DATA_ROOT/frames \
    --l3-frames-dir $DATA_ROOT/frames_l3 \
    --output-dir $DATA_ROOT/annotations \
    --level 3c --model $MODEL --workers 4
```

### 续接 sampled_1k 数据集

```bash
# 将 JSONL 替换为 $JSONL_1K，其余路径不变。已标注的 clip 会被自动跳过。
# 按顺序执行 Step 2 → Step 3 → Step 4 即可。
```

### Step 7: 截取训练用视频 Clips + 构建训练数据

```bash
# 详见 DESIGN.md 的 Phase 2 章节（待实现）
python local_scripts/hier_seg_ablations/build_hier_data.py \
    --annotation-dir $DATA_ROOT/annotations \
    --output-dir /path/to/output \
    --levels L1 L2 L3 L3_seg \
    --complete-only
```

---

## Annotation JSON 格式

每个标注文件 `annotations/{clip_key}.json` 包含 topology-adaptive 层级结构：

```json
{
  "clip_key": "video_001",
  "video_path": "/path/to/video_001.mp4",
  "clip_duration_sec": 180.0,
  "domain_l1": "procedural",
  "domain_l2": "cooking",
  "topology_type": "procedural",
  "topology_confidence": 0.92,
  "topology_reason": "Step-by-step process with distinct sub-goals",
  "summary": "Demonstrating how to make pasta from scratch.",
  "global_phase_criterion": "Split by preparation vs. cooking vs. plating stages.",
  "level1": {
    "macro_phases": [
      {
        "phase_id": 1,
        "start_time": 0, "end_time": 55,
        "phase_name": "Material Preparation",
        "narrative_summary": "Gather and organize required materials.",
        "event_split_criterion": "Contains distinct sequential sub-tasks: measuring, mixing, kneading."
      },
      {
        "phase_id": 2,
        "start_time": 56, "end_time": 120,
        "phase_name": "Boiling Pasta",
        "narrative_summary": "Boil the pasta in salted water.",
        "event_split_criterion": "Single continuous operation with no sub-steps."
      }
    ]
  },
  "level2": {
    "events": [
      {
        "event_id": 1,
        "start_time": 5, "end_time": 30,
        "parent_phase_id": 1,
        "instruction": "Measure and mix dry ingredients",
        "visual_keywords": ["flour", "bowl", "measuring cup"]
      }
    ]
  },
  "level3": {
    "micro_type": "state_change",
    "micro_split_criterion": "Individual visible physical transformations of ingredients.",
    "grounding_results": [
      {
        "action_id": 1,
        "start_time": 5, "end_time": 10,
        "parent_event_id": 1,
        "sub_action": "Pour flour into mixing bowl",
        "pre_state": "Empty bowl on counter",
        "post_state": "Bowl containing flour"
      }
    ]
  }
}
```

> Phase 2 无 events (`events: []`) → L3 直接从该 phase 下钻（leaf-node 路由）。
> 详细设计参见 `DESIGN.md` Phase 1.5 章节。

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
| **L1** | Macro Phase | 阶段级 | 30-120s | 宏观活动阶段，按意图/活动类型切分 |
| **L2** | Activity Event | 活动级 | 10-60s | **可选**。仅当 phase 包含多个截然不同的子步骤时存在 |
| **L3** | Micro Action | 动作级 | 2-6s | 挂载在叶子节点上（event 或无 events 的 phase） |

**变长层级（Phase 1.5）**：L2 按需生成，单一连续动作的 phase → `events: []`，L3 直接从 phase 下钻。

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
| `get_level1_train_prompt_temporal(duration)` | L1 训练（粒度引导版） | `build_hier_data.py` |
| `get_level2_train_prompt(duration)` | L2 训练（三层粒度频谱版） | `build_hier_data.py`, `prepare_clips.py` |
| `get_level3_query_prompt(queries, duration)` | L3 grounding 训练 | `build_hier_data.py` |
| `get_level3_seg_prompt(duration)` | L3 segmentation 训练（三层粒度频谱版） | `build_hier_data.py` |
| `get_level1_train_prompt_with_hint(duration, hint)` | L1 训练 + hint | `build_hier_data.py` |
| `get_level2_train_prompt_with_hint(duration, hint)` | L2 训练 + hint | `build_hier_data.py` |
| `get_level3_seg_prompt_with_hint(duration, hint)` | L3 训练 + hint | `build_hier_data.py` |
| `get_level2_check_prompt(...)` | L2 质量审核（标注用） | `annotate.py` |
| `get_level3_check_prompt(...)` | L3 质量审核（标注用） | `annotate.py` |
