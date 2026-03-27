# 层次分割 Prompt 设计文档

> 状态：V2 设计草稿（通用化 + L3 改分割）
> 涉及文件：`prompt_variants_v2.py`

---

## 一、设计动机

### 当前 V1（P1-P4）的问题

| 问题 | 具体表现 |
|------|----------|
| **过度领域绑定** | 所有 prompt 硬编码 "cooking"、"ingredients"、"recipe" 等词汇，训练后很难迁移到其他程序性视频 |
| **L3 是 grounding，不是分割** | 给定 action query 列表 → 定位每条，测试时依赖文字查询，泛化能力受限 |
| **CoT 无结构约束** | `<think>` 内容完全自由，模型可能跳过关键推理步骤 |

### V2 设计目标

1. **通用化**（Domain-agnostic）：去除所有领域专用词汇，使 prompt 在 cooking/surgery/manufacturing/sports 等场景均可复用
2. **L3 改为自由分割**：不再给 query 列表，模型自主检测原子操作（与 L1/L2 任务结构对齐）
3. **保留 2×2 消融轴**：粒度描述 × 推理链，各自控制一个变量，方便对比

---

## 二、通用术语映射

| 旧（领域特定） | 新（通用） |
|---|---|
| cooking video | procedural activity video |
| macro cooking phases | high-level activity phases |
| cooking events | goal-directed activity segments |
| atomic cooking actions | atomic physical operations |
| transforms ingredients / recipe subgoal | transforms state / advances a sub-goal |
| idle waiting, narration, beauty shots | non-active spans (idle, narration, setup) |
| ingredient preparation, plating | preparation, execution, completion |

---

## 三、新框架：3 Level × 4 Variant（2×2）

```
                   No-CoT              Structured CoT
                ┌─────────────────┬──────────────────────────┐
Minimal         │  V1  New Baseline│  V3  CoT-only            │
Granularity-    │  V2  Gran-only   │  V4  Gran + CoT (Full)   │
Enhanced        └─────────────────┴──────────────────────────┘
```

每个 variant 为 L1 / L2 / L3 各自实现一套模板，共 12 个模板。

---

## 四、L1 — 高层阶段分割（帧编号）

**任务**：给定均匀采样帧序列，分割为大粒度活动阶段

### V1（Minimal baseline，通用）
```
You are given {n_frames} frames uniformly sampled from a video, numbered 1 to {n_frames}.
Segment the frame sequence into high-level activity phases.
Skip non-active spans such as narration, idle waiting, or irrelevant content.

Output the start and end frame number for each phase in order:
<events>[[start_frame, end_frame], ...]</events>

Example: <events>[[3, 80], [95, 150], [160, 220]]</events>
```

### V2（Granularity-Enhanced）
在 V1 基础上新增：
- **PHASE DEFINITION**：broad structural stage, typically 3-5, spans many frames
- **DO NOT**：more than 6 phases / a single phase covering almost all frames / non-active content
- **Prior**：`Expect 3-6 phases for a {n_frames}-frame sequence`

### V3（Structured CoT）
在 V1 基础上，将 `<think>` 结构化为三步：
```
<think>
Observations: [describe the main activities visible in the frames chronologically]
Stage boundaries: [identify where the activity's primary intent clearly shifts]
Non-active spans: [list any narration, idle, or irrelevant spans to skip]
</think>
<events>[[start_frame, end_frame], ...]</events>
```

### V4（Gran + Structured CoT）
V2 的粒度约束 + V3 的结构化推理格式

---

## 五、L2 — 中层目标事件检测（秒级时间戳）

**任务**：给定视频片段（0 ~ duration 秒），检测所有目标导向的活动片段

### V1（Minimal baseline，通用）
```
You are given a {duration}s video clip (timestamps 0 to {duration}).
Detect all goal-directed activity segments in this clip.
Each segment is a multi-second, purposeful sequence of actions that advances toward a specific sub-goal.
Skip non-active spans such as idle waiting, narration, or setup without progress.

Output the start and end time (integer seconds, 0-based) for each segment in order:
<events>[[start_time, end_time], ...]</events>

Example: <events>[[5, 42], [55, 90]]</events>
```

### V2（Granularity-Enhanced）
新增：
- **SEGMENT DEFINITION**：typically 10-60s，MORE specific than a whole activity stage，LESS granular than a single physical motion
- **DO NOT**：segments shorter than 5s / segments that restate the entire clip
- **Prior**：`Expect roughly 2-8 segments for a {duration}s clip`

### V3（Structured CoT）
```
<think>
Observations: [describe what happens in the video chronologically]
Segment grouping: [identify which actions share a unified sub-goal and should be grouped]
Non-active spans: [identify idle/narration periods to exclude]
</think>
<events>[[start_time, end_time], ...]</events>
```

### V4（Gran + Structured CoT）
V2 的粒度约束 + V3 的结构化推理格式

---

## 六、L3 — 原子操作分割（改为自由分割）

**任务变更**：不再给 action query 列表，模型自主检测所有原子物理操作

**旧任务（grounding）**：`给定 [action1, action2, ...]，找每条的时间段`
**新任务（segmentation）**：`给定视频片段，自主检测所有原子操作的时间段`

这使 L3 的任务结构与 L1/L2 对齐，且无需文字查询就能推断。

### V1（Minimal baseline，通用）
```
You are given a {duration}s video clip.
Detect all atomic physical operations in this clip.
Each operation is a brief, single-step physical state change (e.g., a pour, cut, stir, or transfer).
Skip idle pauses, repositioning, or narration.

Output the start and end time (integer seconds, 0-based) for each operation in chronological order:
<events>[[start_time, end_time], ...]</events>

Example: <events>[[2, 6], [9, 13], [15, 20]]</events>
```

### V2（Granularity-Enhanced）
新增：
- **OPERATION DEFINITION**：typically 2-8s，a single discrete physical interaction，ONE object undergoes ONE state change
- **DO NOT**：operations shorter than 1s / pure body motion without object state change / multi-step sequences
- **Prior**：`Expect 3-8 operations for a {duration}s clip`

### V3（Structured CoT）
```
<think>
Observations: [describe the physical interactions visible in chronological order]
Operation boundaries: [identify each discrete state-change moment and its duration]
Skip list: [note any repositioning, idle pause, or narration to exclude]
</think>
<events>[[start_time, end_time], ...]</events>
```

### V4（Gran + Structured CoT）
V2 的粒度约束 + V3 的结构化推理格式

---

## 七、各 Variant 的实现差异对比

| | V1 | V2 | V3 | V4 |
|---|---|---|---|---|
| 粒度先验（典型时长/数量） | ✗ | ✅ | ✗ | ✅ |
| DO NOT 规则 | ✗ | ✅ | ✗ | ✅ |
| 推理步骤 | ✗ | ✗ | 3-step think | 3-step think |
| Prompt token 量（估算）| ~50 | ~150 | ~80 | ~200 |
| MAX_RESPONSE_LEN | 512 | 512 | 1024 | 1024 |
| 领域词汇 | 无 | 无 | 无 | 无 |

---

## 八、与旧版（P1-P4）的对比关系

| 旧版 | 新版 | 变化 |
|------|------|------|
| P1 Minimal No-CoT | **V1** | 去掉 "cooking" 词汇，**新 Baseline** |
| P2 Gran No-CoT | V2 | 通用化描述 |
| P3 CoT | V3 | 自由 `<think>` → **结构化 3-step** |
| P4 Gran+CoT | V4 | V2 粒度 + V3 结构化 think |

---

## 九、消融实验矩阵

```
跨 Level 固定 Variant：  V1_L1 + V1_L2 + V1_L3  vs.  V2_L1 + V2_L2 + V2_L3  ...
                         ↑ 评估粒度描述的影响

跨 Level 固定 Gran：     V1_all  vs.  V3_all
                         ↑ 评估结构化推理链的影响

L3 任务变更对比：        旧 L3 grounding  vs.  新 L3 V1 segmentation
                         ↑ 测量 "有无 query" 对 L3 性能的影响
```

---

## 十、待讨论问题

1. **结构化 CoT 的三步顺序**（Observations → Grouping → Skip）是否适合所有层级？L1 的 "Stage boundaries" 和 L3 的 "Operation boundaries" 描述是否需要差异化？
2. **L3 改为分割后，reward 函数**是否需要从 `position-aligned mean tIoU` 改为 `F1-IoU`（和 L1/L2 对齐）？
3. **V1 是否还需要约束段数/时长**，还是完全 "minimal"？（当前设计：V1 不约束任何数量，让 V2 专门控制这个变量）
4. **实验启动方式**：是扩展 `exp_prompt_ablation.sh` 支持 V1-V4，还是另建脚本？

---

*文件路径：`local_scripts/hier_seg_ablations/prompt_design.md`*
