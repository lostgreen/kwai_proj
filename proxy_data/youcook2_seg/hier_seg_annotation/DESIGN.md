# Topology-Adaptive 分层标注 Pipeline

> 实现状态：**Phase 2 已完成** (2026-03-31) — Quality Check Pipeline

---

## 1. 问题与动机

原始 pipeline 假设所有视频都适合固定三层 `L1 → L2 → L3`，但这只对 procedural 类视频（做饭/装配/维修）天然成立。主要问题：

| # | 问题 | 症状 |
|---|------|------|
| 1 | 固定三层假设过强 | periodic/sequence 视频被迫硬造层级 |
| 2 | L3 定义过于 procedural | periodic 域 L3 空标注或 hallucinate |
| 3 | L1/L2 容易塌缩 | 单一 routine 视频 L1 被强切 2-6 个 phase |
| 4 | L3 抽帧绑定 L2 | periodic 无稳定 L2，经由 event 下钻很别扭 |

---

## 2. 解决方案：Topology-Adaptive

核心思想：在 merged 标注阶段先判断视频的**时间拓扑类型**，再动态决定哪些层存在、以及每层的语义。

### 四种拓扑类型

| 拓扑 | 典型视频 | 层级结构 | l2_mode | l3_mode |
|------|---------|---------|---------|---------|
| **procedural** | 做饭、维修、手工 | L1 → L2 → L3 | workflow | state_change |
| **periodic** | 举重、拉伸、挥拍 | L1 → L3 | optional | repetition_unit |
| **sequence** | 跑酷、训狗、滑雪 | L1 → L2 | episode | skip |
| **flat** | 讲话、vlog、混杂 | L1 only | skip | skip |

### 双阈值保守规则

- `topology_confidence < 0.5` → 降级为 flat（l2_mode=skip, l3_mode=skip）
- `topology_confidence < 0.6` → l3_mode 强制 skip

---

## 3. 已实现内容 (Phase 1)

### 3.1 已修改文件

| 文件 | 改动 |
|------|------|
| `prompts.py` | 新增拓扑常量 + 新版 merged/L3 prompt |
| `annotate.py` | 解析拓扑字段 + topology-aware L3 路由 |
| `extract_frames.py` | L3 抽帧按拓扑分流 |
| `../../shared/seg_source.py` | complete_only 过滤 topology-aware |

### 3.2 Prompt — Merged (L1+L2+Topology)

4 个 PART 一次调用完成：

**PART 1 — DOMAIN CLASSIFICATION**
- 输出 `domain_l1`, `domain_l2`（沿用现有二级分类）

**PART 2 — TOPOLOGY CLASSIFICATION** (新增)
- 输出 `topology_type`: procedural | periodic | sequence | flat
- 输出 `topology_confidence`: 0.0–1.0
- 输出 `topology_reason`: 一句话解释
- 关键规则：
  - 拓扑看时间结构，不只看领域
  - 重复 ≠ procedural
  - 镜头切换 ≠ 拓扑边界
  - 结构不清 → flat

**PART 3 — VIDEO SUMMARY & MACRO PHASES (L1)**
- 1–6 个 macro phases（原来是 2–6，现在允许单 phase）
- 跳过片头片尾、静态段、纯讲话段
- 单一 continuous routine 合法只输出 1 个 phase

**PART 4 — EVENT DETECTION (L2)**
- 按 topology_type 条件定义事件语义：
  - procedural → multi-second workflow
  - sequence → complete episode/trial
  - periodic → events 可选（允许 `[]`）
  - flat → 强制 `events: []`

输出 JSON 示例：

```json
{
  "domain_l1": "cooking",
  "domain_l2": "baking",
  "topology_type": "procedural",
  "topology_confidence": 0.92,
  "topology_reason": "Step-by-step cake preparation with distinct phases",
  "l2_mode": "workflow",
  "summary": "A person prepares and bakes a chocolate cake.",
  "macro_phases": [
    {
      "phase_id": 1,
      "start_time": 5,
      "end_time": 120,
      "phase_name": "Ingredient preparation",
      "narrative_summary": "...",
      "events": [
        {
          "event_id": 1,
          "start_time": 8,
          "end_time": 45,
          "instruction": "Mix dry ingredients in a bowl",
          "visual_keywords": ["flour", "bowl", "whisk"]
        }
      ]
    }
  ]
}
```

### 3.3 Prompt — L3 (Topology-Aware Micro Grounding)

双定义 prompt，按 `topology_type` 切换语义：

**procedural → state_change**
- 对象发生可见物理变化的原子动作
- pre_state / post_state 描述变化
- 忽略伸手、停顿、纯手部调整

**periodic → repetition_unit**
- 单次完整重复/循环/击打/拉伸
- start = 循环启动，end = 循环完成（回到起始姿态）
- post_state 可以 ≈ pre_state

输出 JSON 示例：

```json
{
  "micro_type": "state_change",
  "grounding_results": [
    {
      "action_id": 1,
      "start_time": 42,
      "end_time": 47,
      "sub_action": "Pour batter into pan",
      "pre_state": "Batter in mixing bowl, empty pan",
      "post_state": "Batter filling the pan"
    }
  ]
}
```

### 3.4 代码路由逻辑

**`annotate.py` — annotate_clip() level "3" 分支：**

```python
topology_type = existing.get("topology_type", "procedural")
topology_confidence = existing.get("topology_confidence", 1.0)
l3_mode = existing.get("l3_mode") or TOPOLOGY_TO_L3_MODE.get(topology_type)

if topology_confidence < 0.6:
    l3_mode = "skip"

if l3_mode == "skip":
    # → 写空 level3, 不调 VLM
elif topology_type == "periodic":
    # → L3 from L1 phases, frame dirs: {clip_key}_ph{phase_id}/
else:
    # → L3 from L2 events, frame dirs: {clip_key}_ev{event_id}/
```

**`extract_frames.py` — 按拓扑抽帧：**

| 拓扑 | L3 帧来源 | 目录前缀 |
|------|----------|---------|
| procedural | L2 events | `_ev{id}` |
| periodic | L1 phases | `_ph{id}` |
| sequence | 跳过 | — |
| flat | 跳过 | — |

**`seg_source.py` — complete_only 过滤：**
- `l3_mode == "skip"` → 不要求 level3 存在
- 无 topology 字段（旧标注）→ 仍要求 L1+L2+L3 都存在

### 3.5 Annotation JSON Schema (Phase 1)

```json
{
  "clip_key": "video_0_120",
  "video_path": "...",
  "source_video_path": "...",
  "clip_duration_sec": 120,

  "domain_l1": "cooking",
  "domain_l2": "baking",

  "topology_type": "procedural",
  "topology_confidence": 0.92,
  "topology_reason": "...",
  "l2_mode": "workflow",
  "l3_mode": "state_change",

  "summary": "...",

  "level1": {
    "macro_phases": [
      {
        "phase_id": 1,
        "start_time": 5,
        "end_time": 120,
        "phase_name": "...",
        "narrative_summary": "..."
      }
    ],
    "_sampling": { "n_sampled_frames": 30, "resize_max_width": 0, "jpeg_quality": 60 }
  },

  "level2": {
    "events": [
      {
        "event_id": 1,
        "start_time": 8,
        "end_time": 45,
        "instruction": "...",
        "visual_keywords": ["..."],
        "parent_phase_id": 1
      }
    ]
  },

  "level3": {
    "micro_type": "state_change",
    "grounding_results": [
      {
        "action_id": 1,
        "start_time": 42,
        "end_time": 47,
        "sub_action": "...",
        "pre_state": "...",
        "post_state": "...",
        "parent_event_id": 1
      }
    ],
    "_segment_calls": [...]
  }
}
```

**periodic 视频的 level3：**
```json
{
  "micro_type": "repetition_unit",
  "grounding_results": [
    {
      "action_id": 1,
      "start_time": 10,
      "end_time": 14,
      "sub_action": "One push-up rep",
      "pre_state": "Arms extended, body in plank",
      "post_state": "Arms extended, body in plank",
      "parent_phase_id": 1
    }
  ]
}
```

**sequence/flat 视频的 level3：**
```json
{
  "micro_type": "skip",
  "grounding_results": [],
  "_skip_reason": "l3_mode=skip (topology=sequence, conf=0.88)"
}
```

---

## 4. 向后兼容

| 场景 | 处理 |
|------|------|
| 旧标注无 topology_type | 所有代码默认 "procedural" — 行为不变 |
| 新字段 topology_type/l2_mode/l3_mode/micro_type | 下游 builder 不读这些字段，安全 |
| level2.events = [] (periodic/flat) | 下游 builder 已处理空 events |
| level3 = {micro_type: "skip", ...} | 下游 builder 跳过空 L3 |

---

## 5. 使用方式

### 路径约定

```bash
# 脚本目录
SCRIPT_DIR=proxy_data/youcook2_seg/hier_seg_annotation

# 数据根目录
DATA_ROOT=/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/hier_seg_annotation

# 输入 JSONL (dev 100 条 / 正式 1k 条)
JSONL_DEV=/home/xuboshen/zgw/EasyR1/proxy_data/data_curation/results/merged/sampled/dev_100.jsonl
JSONL_1K=/home/xuboshen/zgw/EasyR1/proxy_data/data_curation/results/merged/sampled/sampled_1k.jsonl

# VLM 配置
MODEL=pa/gemini-3.1-pro-preview
```

### 完整 4 步流程

```bash
# Step 1: 全视频 1fps 抽帧 (仅首次需要)
python $SCRIPT_DIR/extract_frames.py \
    --jsonl $JSONL_DEV \
    --output-dir $DATA_ROOT/frames \
    --fps 1 --workers 4

# Step 2: Merged 标注 (L1+L2+Topology+Criterion)
python $SCRIPT_DIR/annotate.py \
    --jsonl $JSONL_DEV \
    --frames-dir $DATA_ROOT/frames \
    --output-dir $DATA_ROOT/annotations \
    --level merged \
    --model $MODEL --workers 4

# Step 3: L3 帧提取 (leaf-node 路由: events→_ev{id}/, eventless phases→_ph{id}/)
python $SCRIPT_DIR/extract_frames.py \
    --annotation-dir $DATA_ROOT/annotations \
    --output-dir $DATA_ROOT/frames_l3 \
    --fps 2 --workers 4

# Step 4: L3 标注 (自动跳过 sequence/flat)
python $SCRIPT_DIR/annotate.py \
    --jsonl $JSONL_DEV \
    --frames-dir $DATA_ROOT/frames \
    --l3-frames-dir $DATA_ROOT/frames_l3 \
    --output-dir $DATA_ROOT/annotations \
    --level 3 \
    --model $MODEL --workers 8

# Step 5 (可选): Criterion → 通用 Training Hint 改写
python $SCRIPT_DIR/rewrite_criteria_hints.py \
    --annotation-dir $DATA_ROOT/annotations \
    --api-base $API_BASE \
    --model gpt-4o-mini \
    --workers 4
```

### 续接 1k 数据集

```bash
# 将 JSONL 切换为 sampled_1k.jsonl，其余路径不变
# 已标注的 clip 会被自动跳过 (按 clip_key 去重)

# Merged 标注 (续接)
python $SCRIPT_DIR/annotate.py \
    --jsonl $JSONL_1K \
    --frames-dir $DATA_ROOT/frames \
    --output-dir $DATA_ROOT/annotations \
    --level merged \
    --model $MODEL --workers 4

# L3 帧提取 (续接)
python $SCRIPT_DIR/extract_frames.py \
    --annotation-dir $DATA_ROOT/annotations \
    --output-dir $DATA_ROOT/frames_l3 \
    --fps 2 --workers 4

# L3 标注 (续接)
python $SCRIPT_DIR/annotate.py \
    --jsonl $JSONL_1K \
    --frames-dir $DATA_ROOT/frames \
    --l3-frames-dir $DATA_ROOT/frames_l3 \
    --output-dir $DATA_ROOT/annotations \
    --level 3 \
    --model $MODEL --workers 8
```
```

---

## 6. Phase 1.5: Leaf-Node L3 路由 (已实现)

### 6.1 问题
`procedural` 视频中，某些 phase 只包含单一连续/循环动作（如"一直钉钉子 60 秒"），VLM 被迫造出与 phase 几乎一样的 event → L1 ≈ L2 冗余。

### 6.2 核心设计：变长层级

放弃死板的 L1→L2→L3 固定三层，改为 **"父节点 → 子节点"** 的相对概念：

| 角色 | 定义 | 对应原层级 |
|------|------|-----------|
| **Macro (宏观阶段)** | 视频的必然组成部分 | L1 |
| **Meso (中观事件)** | 完全可选。仅当 Macro 包含多个截然不同的子动作时才存在 | L2 |
| **Micro (微观动作)** | 永远只挂载在**叶子节点**上 | L3 |

**核心规则**: L3 标注目标 = 树的叶子节点（Leaf Nodes）

```
情况 A: Phase 有 events → events 是叶子 → L3 从 events 下钻
         Phase ─┬─ Event 1 (leaf) ─── L3 micro
                └─ Event 2 (leaf) ─── L3 micro

情况 B: Phase 无 events → phase 自身是叶子 → L3 从 phase 直接下钻
         Phase (leaf) ─── L3 micro

情况 C: 混合 → 同一视频中两种情况并存
         Phase 1 ─┬─ Event 1 (leaf) ─── L3 micro
                  └─ Event 2 (leaf) ─── L3 micro
         Phase 2 (leaf, 单一连续动作) ─── L3 micro
```

### 6.3 完整 Prompt：Merged (L1+L2+Topology)

> 对应 `prompts.py` 中 `_MERGED_L1L2_BASE` 模板。`{duration}`, `{n_frames}`, `{domain_taxonomy_str}` 为运行时填充。

```
You are given a {duration}s video clip (timestamps 0 to {duration}) with {n_frames} frames.
Your task has four parts.

## PART 1 — DOMAIN CLASSIFICATION
Classify the video using a two-level taxonomy.
Choose ONE broad category (domain_l1) and ONE fine-grained subcategory (domain_l2)
from the list below:
{domain_taxonomy_str}

## PART 2 — TOPOLOGY CLASSIFICATION (CRITICAL)
Analyze the TEMPORAL STRUCTURE of the visible activity and assign exactly ONE topology_type.

Topology types:
- procedural:
  A step-by-step process with meaningful sub-goals that progress toward an outcome.
  Typical examples: cooking, assembling, repairing, crafting.

- periodic:
  A repeated cycle of the same motion or operation.
  Typical examples: stretching repetitions, weightlifting reps, repetitive factory motions.

- sequence:
  A continuous traversal, trial, run, or episode with one coherent trajectory or attempt.
  Typical examples: dog agility runs, parkour runs, skiing descents, obstacle traversals.

- flat:
  A single continuous activity with no stable internal hierarchy, or mixed/unclear
  structure that should not be over-segmented.
  Typical examples: idle talking, continuous walking vlog, loosely mixed footage.

Important topology rules:
1. Topology is about temporal structure, NOT about domain label alone.
2. Repetition alone does NOT imply procedural structure.
3. Camera cuts do NOT define topology.
4. If the structure is weak or unclear, choose flat rather than inventing hierarchy.

Also output:
- topology_confidence: a float from 0.0 to 1.0
- topology_reason: one brief sentence explaining the decision

## PART 3 — VIDEO SUMMARY & MACRO PHASES (L1)
Write ONE sentence summarizing the video.

Then segment the video into 1–6 macro phases.
A macro phase is a broad stage of activity organized by overall intent.
- Skip intros, outros, static non-activity spans, and talking-only spans.
- Phases do NOT need to cover the entire video.
- Do NOT split by camera cuts.
- It is valid to output only 1 macro phase if the entire video is one continuous routine.

## PART 4 — EVENT DETECTION (L2)
Detect events nested inside each macro phase.
Apply the event definition STRICTLY based on topology_type:

If topology_type = procedural:
- An event is a multi-second workflow (typically 10–60s) that completes a process sub-goal.
- Group related manipulations together.
- Do NOT fragment into atomic tool motions.
- **If a macro phase consists of a single continuous operation or a simple repetitive action
  (e.g., "hammering nails for 60 seconds", "stirring continuously"), leave its "events": [].**
- **ONLY create events when the phase contains distinct sequential sub-steps.**

If topology_type = sequence:
- An event is a complete episode, trial, or continuous traversal with one coherent
  trajectory or objective.
- Do NOT split by local body motions, individual obstacles, or camera cuts.
- A whole run/trial should usually be one event.

If topology_type = periodic:
- Events are optional.
- You may leave "events": [] for a phase, or output ONE event only if it exactly
  matches the whole phase as a container for later micro annotation.
- Do NOT create one event per repetition.

If topology_type = flat:
- Output "events": [].
- Do NOT invent L2 structure.

General L2 rules:
- Events must not overlap.
- Use absolute integer seconds.
- **It is valid for a phase to contain zero events.**
- **Do not force extra events to make the hierarchy deeper.**

Output JSON:
{
  "domain_l1": "<one broad category>",
  "domain_l2": "<one fine-grained subcategory>",
  "topology_type": "procedural | periodic | sequence | flat",
  "topology_confidence": 0.95,
  "topology_reason": "<one sentence>",
  "l2_mode": "workflow | episode | optional | skip",
  "summary": "<one sentence>",
  "macro_phases": [
    {
      "phase_id": 1,
      "start_time": 5,
      "end_time": 60,
      "phase_name": "Material Preparation",
      "narrative_summary": "Gather and organize all required materials.",
      "events": [
        {
          "event_id": 1,
          "start_time": 8,
          "end_time": 25,
          "instruction": "Sort and measure the raw materials",
          "visual_keywords": ["hands", "materials", "measuring tool"]
        }
      ]
    }
  ]
}
```

**Phase 1.5 关键约束（加粗部分）**:
- **procedural events 按需留空**: 单一连续动作/简单循环动作的 phase → `"events": []`
- **不强行加深层级**: 只有包含截然不同子步骤的 phase 才创建 events
- **零 events 合法**: 任何 topology 下 phase 都允许 `events: []`

### 6.4 完整 Prompt：L3 (Topology-Aware Micro Grounding)

> 对应 `prompts.py` 中 `_LEVEL3_BASE` 模板。`{clip_start}`, `{clip_end}`, `{action_query}`, `{topology_type}` 为运行时填充。

```
You are a temporal grounding model. You are viewing frames from a clip
({clip_start}s to {clip_end}s).
The input query is: "{action_query}"
The topology_type of the source video is: "{topology_type}".

Your task is to pinpoint every atomic micro-action in this clip.

IMPORTANT:
- If topology_type is "sequence" or "flat", this prompt should not be used.
- Use absolute integer seconds from the full video timeline.

LEVEL 3 DEFINITIONS

If topology_type = procedural:
- micro_type = "state_change"
- **Find brief atomic actions where an object undergoes a clear visible physical change.**
- Valid examples: cutting, pouring into a container, attaching a part, spreading
  material, separating pieces.
- start_time = onset of the actual physical interaction or transformation
- end_time = the moment the new visible state is established
- **Ignore reaching, idle pauses, pure hand repositioning, and narration**

If topology_type = periodic:
- micro_type = "repetition_unit"
- **Find each individual completed repetition, cycle, strike, or stretch.**
- start_time = initiation of one repetition cycle
- end_time = completion of that same repetition cycle, usually when the body/equipment
  returns to its resting or starting position
- **Ignore idle pauses between repetitions**
- IMPORTANT: post_state may be similar or identical to pre_state if the repetition
  returns to the starting posture

General rules:
1. Typical duration is 2–6 seconds, but use shorter or longer spans if the unit is
   clearly visible.
2. Allow gaps between micro-actions.
3. Merge uninterrupted motion belonging to the same single repetition or same single
   state change.
4. Do not force full coverage.

For each micro-action, provide:
- action_id: Sequential integer starting from 1.
- start_time / end_time: Timestamps in integer seconds (absolute within the full video).
- sub_action: Brief description of the specific physical interaction or repetition.
- pre_state: The EXPLICIT visual state BEFORE the interaction.
- post_state: The EXPLICIT visual state AFTER the interaction.

Output JSON:
{
  "micro_type": "state_change | repetition_unit",
  "grounding_results": [
    {
      "action_id": 1,
      "start_time": 42,
      "end_time": 47,
      "sub_action": "Transfer material A into container B",
      "pre_state": "Empty container with prepared surface",
      "post_state": "Material A distributed across the container surface"
    }
  ]
}
```

**L3 Prompt 关键约束**:
- **state_change**: 仅找可见物理变化的原子动作，忽略纯手部调整
- **repetition_unit**: 找每个完整重复周期，post_state ≈ pre_state 是合法的
- **不强制全覆盖**: 允许 micro-action 之间有间隔

### 6.5 JSON 输出示例（含可选 L2）

**屋顶维修视频**：Phase 1 有递进子步骤 → 产生 events；Phase 2 是单一循环动作 → events 为空

```json
{
  "topology_type": "procedural",
  "topology_confidence": 0.88,
  "macro_phases": [
    {
      "phase_id": 1,
      "start_time": 0,
      "end_time": 51,
      "phase_name": "Removing the Damaged Shingle",
      "events": [
        {"event_id": 1, "start_time": 0, "end_time": 30,
         "instruction": "Pry up surrounding shingles with flat bar"},
        {"event_id": 2, "start_time": 31, "end_time": 51,
         "instruction": "Pull out the damaged shingle"}
      ]
    },
    {
      "phase_id": 2,
      "start_time": 52,
      "end_time": 157,
      "phase_name": "Securing the New Shingle",
      "events": []
    }
  ]
}
```

→ Phase 1 的叶子: Event 1, Event 2（标注细粒度 state_change）
→ Phase 2 的叶子: Phase 2 自身（直接标注 52s-157s 内的重复钉钉动作）

### 6.6 运行流程图

```
                        ┌─────────────────────────────┐
                        │  Step 1: extract_frames.py   │
                        │  全视频 1fps 抽帧             │
                        └──────────────┬──────────────┘
                                       │
                        ┌──────────────▼──────────────┐
                        │  Step 2: annotate.py         │
                        │  --level merged              │
                        │  L1 + L2(可选) + Topology    │
                        └──────────────┬──────────────┘
                                       │
                    ┌──────────────────▼──────────────────┐
                    │     _split_merged_response()         │
                    │  拆分为 level1 + level2 (可能为空)    │
                    └──────────────────┬──────────────────┘
                                       │
                        ┌──────────────▼──────────────┐
                        │  Step 3: extract_frames.py   │
                        │  L3 帧提取 (leaf-node 路由)  │
                        └──────────────┬──────────────┘
                                       │
                    ┌─────────────────▼─────────────────┐
                    │          Leaf-Node 收集              │
                    │                                      │
                    │  for phase in macro_phases:           │
                    │    if phase has events:               │
                    │      → events 各自抽帧 (_ev{id}/)    │
                    │    else:                              │
                    │      → phase 整体抽帧 (_ph{id}/)     │
                    └─────────────────┬─────────────────┘
                                       │
                        ┌──────────────▼──────────────┐
                        │  Step 4: annotate.py         │
                        │  --level 3                   │
                        │  L3 标注 (leaf-node 路由)    │
                        └──────────────┬──────────────┘
                                       │
                    ┌─────────────────▼─────────────────┐
                    │    _annotate_level3() 内部路由       │
                    │                                      │
                    │  periodic?                           │
                    │    → 所有 phases 作为 sources         │
                    │  else (procedural/sequence):          │
                    │    → leaf-node 收集:                  │
                    │      有 events 的 phase → events      │
                    │      无 events 的 phase → phase 自身  │
                    │      无 L1 数据 → events fallback     │
                    │                                      │
                    │  每个 source 独立调用 VLM:            │
                    │    event source → parent_event_id     │
                    │    phase source → parent_phase_id     │
                    └──────────────────────────────────┘
```

### 6.7 代码路由逻辑（更新后）

**`_annotate_level3()` — leaf-node 收集**:

```python
# 1. periodic: 所有 phases 作为 sources (不变)
if topology_type == "periodic" and l1_result is not None:
    sources = [phase → source_type="phase" for phase in l1_result]

# 2. 其他: leaf-node 收集
else:
    phase_events = {phase_id: [events...]} mapping
    for phase in l1_phases:
        if phase has events:
            sources += [event → source_type="event" for event in phase_events[pid]]
        else:
            sources += [phase → source_type="phase"]

    # 3. Fallback: 无 L1 数据时用 events (向后兼容旧标注)
    if not l1_phases:
        sources = [event → source_type="event" for event in events]
```

**`annotate_clip()` — level "3" 分支变更**:

```python
# 新增: procedural 分支也传入 l1_result
else:
    l1_result=existing.get("level1")  # ← 之前不传，现在传
```

### 6.8 改动文件
| 文件 | 改动 |
|------|------|
| `prompts.py` | PART 4 procedural 规则：单一动作 phase → events=[] |
| `annotate.py` | `_annotate_level3()` source 构建改为 leaf-node 收集 |
| `annotate.py` | `annotate_clip()` procedural 分支传入 `l1_result` |
| `extract_frames.py` | `run_l3_extraction()` 非 periodic 分支改为 leaf-node 路由 |

### 6.9 向后兼容
- 旧标注（所有 phase 都有 events）走原路径，行为不变
- 无 events 的 phase 生成 `_ph{id}` 帧目录和 `parent_phase_id` 标记
- **训练 builder 暂不改**：`build_l3_records` 仍按 `parent_event_id` 过滤，phase-based L3 数据被静默跳过（Phase 2 再支持）

### 6.10 Split Criterion 字段（三层切分依据）

标注阶段 VLM 在输出分割结果的同时，输出"切分依据"——解释为什么这样切。这些 criterion 未来作为训练数据中的 reasoning hint。

| 字段 | 层级 | 生成阶段 | 存储位置 |
|------|------|---------|---------|
| `global_phase_criterion` | L1 | merged 标注 | 顶层 (与 summary 同级) |
| `event_split_criterion` | L2 (per phase) | merged 标注 | `level1.macro_phases[].event_split_criterion` |
| `micro_split_criterion` | L3 (per leaf) | L3 标注 | `level3.micro_split_criterion` + `_segment_calls[].micro_split_criterion` |

**设计原则**：
- criterion 描述**分割逻辑/粒度标准**，不描述视频内容
- VLM 自主生成，我们只在 prompt 中指导输出格式
- 向后兼容：旧标注无 criterion → `.get("field", "")` 返回空字符串

**JSON 示例**：
```json
{
  "summary": "A person first goes snow tubing, then builds a snowman.",
  "global_phase_criterion": "Split by fundamental shift of activity type: unstructured recreation vs. goal-oriented procedural assembly.",
  "topology_type": "procedural",

  "level1": {
    "macro_phases": [
      {
        "phase_id": 1,
        "phase_name": "Snow Tubing",
        "event_split_criterion": "Repetitive recreational activity with no sequential progression; no event segmentation needed.",
        "events": []
      },
      {
        "phase_id": 2,
        "phase_name": "Building a Snowman",
        "event_split_criterion": "Procedural task segmented by logical assembly progression: forming base, stacking body, decorating.",
        "events": [{"event_id": 1, "instruction": "Roll the snowball to form the base"}, ...]
      }
    ]
  },

  "level3": {
    "micro_type": "state_change",
    "micro_split_criterion": "Broke down by individual state-changing operations where material visibly transforms.",
    "grounding_results": [...],
    "_segment_calls": [
      {"parent_event_id": 1, "micro_split_criterion": "Broke down by individual state-changing operations..."}
    ]
  }
}
```

### 6.11 Criterion → Training Hint 改写

VLM 输出的 criterion 可能包含具体视频内容（对象名、动作细节），直接用于训练会让模型退化为 grounding。`rewrite_criteria_hints.py` 脚本调用 LLM 将 criterion 改写为内容无关的通用分割 hint。

**改写规则**：去除所有具体视频内容引用，仅保留结构性分割逻辑。

| 原始 criterion | 改写后 hint |
|---|---|
| "Segmented by removing wires, disconnecting hoses, and unbolting cover." | "Segmented by distinct sequential sub-tasks each completing a specific sub-goal." |
| "Repetitive recreational activity; no event segmentation needed." | "Single repetitive activity with no sequential progression; no sub-event segmentation needed." |

**新增字段**（写回同一 annotation JSON）：

| 字段 | 来源 |
|------|------|
| `global_phase_hint` | 改写自 `global_phase_criterion` |
| phases[].`event_split_hint` | 改写自 phases[].`event_split_criterion` |
| level3.`micro_split_hint` | 改写自 level3.`micro_split_criterion` |

**使用方式**：
```bash
python rewrite_criteria_hints.py \
    --annotation-dir annotations/ \
    --api-base $API_BASE \
    --model gpt-4o-mini \
    --workers 4

# Dry run (不调 LLM，只显示哪些字段会被改写):
python rewrite_criteria_hints.py --annotation-dir annotations/ --dry-run
```

---

## 7. Phase 2: Quality Check Pipeline (已实现)

### 7.1 概述

标注完成后，通过 **独立的 check 流程** 对 L1+L2+L3 进行模型级质量复审。check 通过 `annotate_check.py` 独立运行，读取已有标注 JSON，调用（可能更强的）VLM 对结果进行 keep/revise/remove 三分类审核，并补充遗漏标注（supplement）。

**核心设计原则**:
1. **幂等安全**: 每层 check 完成后写入 `_check_stats` 标记字段，re-run 时自动跳过
2. **级联一致**: merged_c 先审核 L1+L2，3c 基于 checked 结果审核 L3
3. **保守策略**: VLM 未 review 到的条目默认 keep（safety net）
4. **划分依据传递**: check prompt 包含原始标注的 criterion/split_criterion，让审核模型了解原始分割意图

### 7.2 Check 数据流

```
annotations/                      annotations_checked/
├─ {clip}.json                    ├─ {clip}.json
│  level1                         │  level1 + _check_stats
│  level2          ──(Step 5)──→  │  level2 + _check_stats
│  level3 (原始)                  │  level3 (孤儿已清理)
│  global_phase_criterion         │  _audit_meta
│  summary, topology_type         │
│                                 │
│                  ──(Step 6)──→  │  level3 + _check_stats
│                                 │  _audit_meta (更新)
```

**run_pipeline.sh 中的执行**:

| Step | 命令 | 入参 | 输出 |
|------|------|------|------|
| 5 | `annotate_check.py --levels merged_c` | `annotations/` → `annotations_checked/` | L1+L2 checked |
| 6 | `annotate_check.py --levels 3c` | `annotations_checked/` → `annotations_checked/` (in-place) | L3 checked |

### 7.3 Skip (幂等) 机制

```python
# annotate_check.py check_clip() 中的跳过判断:

#  merged_c: 需要 L1 和 L2 的 _check_stats 都已存在
if l1.get("_check_stats") is not None and l2.get("_check_stats") is not None:
    skip

# 3c: 需要 L3 的 _check_stats 已存在
if l3.get("_check_stats") is not None:
    skip

# 可用 --overwrite 强制重新 check
```

**重要**: merged_c 和 3c 的 skip 判断**独立**——即使 L1+L2 已 check，L3 仍然会被 check（如果还没做过）。

### 7.4 merged_c — L1+L2 联合 Check

**入口**: `annotate_check.py:check_clip()` → `annotate.py:_check_merged_l1l2()`

**输入给 VLM 的信息**:
| 信息 | 来源 |
|------|------|
| 全视频 1fps 帧 (采样至 max_frames) | `frames/{clip_key}/` |
| 现有 L1 phases (嵌套 L2 events) | `ann["level1"]`, `ann["level2"]` |
| 视频摘要 | `ann["summary"]` |
| 拓扑类型 + 置信度 | `ann["topology_type"]`, `ann["topology_confidence"]` |
| L1 划分依据 | `ann["global_phase_criterion"]` |
| L2 划分依据 (per phase) | `phase["event_split_criterion"]` |

**Prompt 结构** (`prompts.py:_MERGED_CHECK_BASE`):

```
Context:
  - {duration}s video, {n_frames} frames
  - summary, topology_type, topology_confidence
  - global_phase_criterion (原始 L1 划分依据)

Existing annotations (nested JSON):
  - L1 phases (含 event_split_criterion)
    └─ L2 events per phase

L1 PHASE REVIEW (granularity spectrum + 6 criteria):
  1. Phase Boundaries
  2. Not Too Broad
  3. Not Too Fine
  4. Phase Naming
  5. Camera Cut Independence
  6. Completeness

L2 EVENT REVIEW (granularity spectrum + 7 criteria):
  1. Not Too Coarse
  2. Not Too Fine
  3. Temporal Accuracy
  4. Description Quality
  5. Activity Relevance
  6. Temporal Overlap
  7. Completeness

Output:
  { phase_reviews, phase_supplements, event_reviews, event_supplements }
```

**Verdict 处理逻辑** (`_apply_l1_check_results` + `_check_merged_l1l2`):

```
for review in phase_reviews:
  keep    → 原样保留
  revise  → 合并 revised 字段 → 校验 start<end → 成功标记 _checked=revised
          → 失败保留原始，标记 revise_failed_kept_original
  remove  → 删除

未被 review 到的 phase → 默认 keep (safety net)
phase_supplements → 追加，标记 _checked=supplemented

排序 + 重编号 phase_id
同样逻辑处理 event_reviews / event_supplements
孤儿 events (parent_phase_id 不存在) → 删除
排序 + 重编号 event_id
```

**级联 L3 孤儿清理** (`annotate_check.py:197-210`):
如果 L3 已存在:
1. `_remove_orphaned_l3_results()`: 删除 parent event/phase 已被删除的 L3
2. `_remove_out_of_bounds_l3()`: 删除 parent event 边界变了导致完全越界的 L3

### 7.5 3c — L3 Check

**入口**: `annotate_check.py:check_clip()` → `annotate.py:_check_level3()`

**输入给 VLM 的信息**:
| 信息 | 来源 |
|------|------|
| 帧 (优先 2fps per-event, fallback 1fps) | `frames_l3/{clip_key}_ev{id}/` 或 `frames/{clip_key}/` |
| 现有 L3 grounding_results (per event) | `ann["level3"]["grounding_results"]` |
| L2 event context (instruction, 时间范围) | `ann["level2"]["events"]` |
| 微动作类型 | `l3_result["micro_type"]` (state_change / repetition_unit) |
| L3 划分依据 | `l3_result["micro_split_criterion"]` |

**Prompt 结构** (`prompts.py:_LEVEL3_CHECK_BASE`):

```
Context:
  - Event clip ({event_start}s to {event_end}s)
  - action_query = L2 event instruction
  - micro_type (state_change / repetition_unit)
  - micro_split_criterion (原始 L3 划分依据)

Existing L3 annotations (JSON array):
  - action_id, start_time, end_time, sub_action, pre_state, post_state

GRANULARITY SPECTRUM (三层定义):
  L2 Event (ABOVE — too coarse)
  L3 Atomic Action (THIS LEVEL — correct)
  Sub-atomic motion (BELOW — too fine)

7 Review Criteria:
  1. Granularity — Not Too Coarse
  2. Granularity — Not Too Fine
  3. Temporal Accuracy
  4. State Description Quality
  5. Activity Relevance
  6. Boundary Compliance (start >= event_start, end <= event_end)
  7. Completeness

Output:
  { reviews, supplements }
```

**处理逻辑** (`_apply_l3_check_results`):
- 与 L1/L2 完全一致的 keep/revise/remove + supplement 模式
- revise 字段: start_time, end_time, sub_action, pre_state, post_state
- supplement 自动标记 parent_event_id

### 7.6 Check 输出 JSON Schema

check 完成后，annotation JSON 中新增/修改的字段:

```json
{
  "level1": {
    "macro_phases": [
      {
        "phase_id": 1,
        "_checked": "revised"
      }
    ],
    "_check_stats": {
      "kept": 3,
      "revised": 1,
      "removed": 0,
      "supplemented": 1
    }
  },
  "level2": {
    "events": [
      {
        "event_id": 1,
        "_checked": "supplemented"
      }
    ],
    "_check_stats": {
      "kept": 5,
      "revised": 2,
      "removed": 1,
      "supplemented": 0
    }
  },
  "level3": {
    "grounding_results": [...],
    "_check_calls": [
      {
        "event_id": 1,
        "n_before": 4,
        "n_after": 5,
        "n_supplements": 1
      }
    ],
    "_check_stats": {
      "kept": 8,
      "revised": 3,
      "removed": 1,
      "supplemented": 2
    }
  },
  "_audit_meta": {
    "audit_model": "pa/gmn-2.5-pr",
    "audit_levels": ["merged_c", "3c"],
    "audited_at": "2026-03-31T...",
    "original_annotation": "annotations/clip_key.json"
  }
}
```

### 7.7 Check Prompt 信息传递对照表

| Check 层级 | 传入的划分依据 | 传入的标注结构 | 传入的帧 |
|-----------|--------------|-------------|---------|
| merged_c | `global_phase_criterion` + per-phase `event_split_criterion` | L1 phases (嵌套 L2 events) | 全视频 1fps 采样 |
| 3c | `micro_type` + `micro_split_criterion` | per-event L3 grounding_results | per-event 2fps (fallback 1fps) |

### 7.8 `--frames-dir` 在 L3 check 中的必要性

即使只运行 3c，`--frames-dir`（1fps 帧目录）仍然必须提供:

| 用途 | 代码位置 |
|------|---------|
| 加载 `clip_duration` (from `frame_meta.json`) | `annotate_check.py:154-159` |
| 获取全局 fps (from `frame_meta.json`) | `annotate.py:_check_level3():1217-1218` |
| Fallback 帧源 (per-event 帧不存在时) | `annotate.py:_check_level3():1272-1273` |

---

## 8. 后续阶段（未实现）

### Phase 3: Schema + 训练样本改造
- training builder 支持 `parent_phase_id`，构建 phase-based L3 训练数据
- `seg_source.py` 的 `get_l3_clip_path()` 和 `prepare_clips.py` 支持 `_L3_ph{id}_` 命名
- 可选：统一训练 prompt 为 "parent → child" 通用模板
- 加入控制 token: `<granularity=macro/meso/micro>`, `<micro_type=...>`

### Phase 4: 筛选规则 + QC 联动
- Route D 从固定 `events >= 3` 扩展到 topology-aware
- 2c / 3c 检查 prompt 按 topology 改写
- periodic/sequence 新增专用 QC 规则
