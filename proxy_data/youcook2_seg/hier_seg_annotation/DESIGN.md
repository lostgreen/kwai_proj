# Topology-Adaptive 分层标注 Pipeline

> 实现状态：**Phase 1 已完成** (2026-03-30)

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

```bash
# Step 1: 全视频 1fps 抽帧
python extract_frames.py \
    --original-video-root /path/to/videos \
    --output-dir frames/ --fps 1

# Step 2: Merged 标注（L1+L2+Topology）
python annotate.py \
    --frames-dir frames/ --output-dir annotations/ \
    --level merged --api-base $API_BASE --model $MODEL

# Step 3: L3 抽帧（自动按 topology 路由）
python extract_frames.py \
    --annotation-dir annotations/ \
    --original-video-root /path/to/videos \
    --output-dir frames_l3/ --fps 2

# Step 4: L3 标注（自动跳过 sequence/flat）
python annotate.py \
    --frames-dir frames/ --l3-frames-dir frames_l3/ \
    --output-dir annotations/ --level 3 \
    --api-base $API_BASE --model $MODEL
```

---

## 6. 后续阶段（未实现）

### Phase 2: Schema + 训练样本改造
- training builder 按 topology 生成不同粒度样本
- 加入控制 token: `<granularity=macro/meso/micro>`, `<micro_type=...>`

### Phase 3: 筛选规则 + QC 联动
- Route D 从固定 `events >= 3` 扩展到 topology-aware
- 2c / 3c 检查 prompt 按 topology 改写
- periodic/sequence 新增专用 QC 规则
