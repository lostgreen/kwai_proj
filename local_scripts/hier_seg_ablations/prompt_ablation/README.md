# Prompt Ablation — 层次分割 Prompt 消融实验

> 核心问题：**Prompt 的表述方式如何影响 RL 训练的时序分割质量？**
>
> 对比：**语义描述式 prompt**（PA1, cooking 领域词） vs **边界判据式 prompt**（PA2, domain-agnostic + sparse-aware）。
>
> 两组实验均使用 F1-IoU reward，仅改变 prompt。

---

## 消融目的

Prompt 是 VLM 理解任务的唯一输入，其表述方式直接影响模型的分割策略。两个关键假设：

1. **「边界在哪」vs「是什么」**：告诉模型"如何判断边界"比"检测什么事件"更有效——前者给出明确的切分判据，后者依赖模型自行理解语义
2. **稀疏采样感知**：显式告知"1-2 fps 采样，不要依赖单帧变化"可以减少过碎分割——模型不知道输入是稀疏采样时，可能把每个帧间差异都当作边界

消融实验量化这两个假设的实际增益。

---

## 实验设计

### 两组对比

| 实验 | Prompt 来源 | 核心特点 | CoT | MAX_RESP |
|------|------------|---------|-----|----------|
| **PA1** | `prompts.py`（原始标注） | 语义描述、cooking 领域词、无稀疏约束 | 无 | 512 |
| **PA2** | `prompt_variants_v3.py` V2 | 边界判据、领域无关、稀疏感知、硬规则 | 无 | 512 |

### 控制变量

| 维度 | 两组共用 |
|------|---------|
| 模型 | Qwen3-VL-8B-Instruct |
| 数据层 | L2（事件检测）+ L3（自由分割）— 跳过 L1 |
| Reward | F1-IoU（Hungarian 匹配） |
| 算法 | EMA-GRPO, LR=5e-7, cosine decay |
| 训练 | 60 steps, rollout_bs=16, 8 GPU |

---

## 完整 Prompt 文本

### PA1: L2 原始标注 Prompt（事件检测）

```
You are given a {duration}s cooking video clip (timestamps 0 to {duration}).
Detect all complete cooking events in this clip.
Each event is a multi-second, goal-directed workflow that transforms ingredients
or completes a recipe subgoal.
Skip idle waiting, narration, tool pickup, or beauty shots.

Output the start and end time (integer seconds, 0-based) for each event in order:
<events>[[start_time, end_time], ...]</events>

Example: <events>[[5, 42], [55, 90]]</events>
```

### PA1: L3 原始标注 Prompt（自由分割）

```
You are given a {duration}s cooking video clip.
Detect all atomic cooking actions (state-changing physical operations like
cutting, stirring, pouring) in this clip.
Skip idle waiting, narration, or tool pickup.

Output the start and end time (integer seconds, 0-based) for each action
in chronological order:
<events>[[start_time, end_time], ...]</events>

Example: <events>[[3, 7], [10, 14], [16, 22]]</events>
```

### PA2: L2 V3 边界判据 Prompt（事件检测）

```
You are given a {duration}s video clip (timestamps 0 to {duration}), sampled at 1-2 fps.

Detect all LOCAL TASK UNITS in this clip.

BOUNDARY CRITERION — cut when:
- A self-contained local task is completed (the sub-goal is achieved or abandoned).
- The person starts working toward a clearly different sub-goal.
DO NOT cut when:
- The person switches tools/materials but continues the same task.
- Brief pauses, adjustments, or repositioning occur within the same task.

IMPORTANT — SPARSE SAMPLING:
This clip is sampled at 1-2 fps (not continuous video).
Do NOT rely on single-frame micro-motions, instantaneous contact changes,
or camera cuts to place boundaries.
Create a boundary ONLY when the change is sustained across multiple sampled frames
or when the task/state clearly shifts.

HARD RULES:
- Minimum segment duration: 5 seconds. Merge shorter segments with neighbors.
- Expected count: 2-8 task units for a {duration}s clip.
- Gaps between segments are expected — not every second needs to be covered.
- If two adjacent segments pursue the same sub-goal, merge them.

Output the start and end time (integer seconds, 0-based) for each task unit
in chronological order:
<events>[[start_time, end_time], ...]</events>

Example: <events>[[5, 42], [55, 90]]</events>
```

### PA2: L3 V3 边界判据 Prompt（自由分割）

```
You are given a {duration}s video clip, sampled at 1-2 fps.

Detect all VISIBLE STATE-CHANGE segments in this clip.

BOUNDARY CRITERION — cut when:
- A new visible object/material change begins (something starts to deform,
  separate, merge, transfer, or change state).
- An ongoing state change completes (the object reaches its new state
  and motion stops).
DO NOT cut when:
- Hands or body parts reposition without changing any object's state.
- Camera angle changes or brief occlusions occur.
- You see a single-frame flicker that is not sustained across ≥2 sampled frames.

IMPORTANT — SPARSE SAMPLING:
This clip is sampled at 1-2 fps (not continuous video).
Do NOT rely on single-frame micro-motions, instantaneous contact changes,
or camera cuts to place boundaries.
Create a boundary ONLY when the change is sustained across multiple sampled frames
or when the task/state clearly shifts.

HARD RULES:
- Minimum segment duration: 2 seconds. If a change appears shorter, extend
  boundaries to the nearest sustained-change frames.
- Maximum segment duration: 15 seconds — if longer, it likely contains
  multiple changes; split them.
- Expected count: 3-8 segments for a {duration}s clip.
- Gaps between segments are expected — not every second is active.
- If two adjacent segments involve the same object undergoing the same
  continuous change, merge them into one.

Output the start and end time (integer seconds, 0-based) for each segment
in chronological order:
<events>[[start_time, end_time], ...]</events>

Example: <events>[[2, 6], [9, 13], [15, 20]]</events>
```

---

## 逐维度设计分析

### 1. 任务定义方式：语义描述 vs 边界判据

| | PA1（语义描述） | PA2（边界判据） |
|---|---|---|
| L2 | "complete cooking events" — 定义**是什么** | "LOCAL TASK UNIT" + "cut when sub-goal completed" — 定义**何时切** |
| L3 | "atomic cooking actions" — 用动作名列举 | "VISIBLE STATE-CHANGE" + "cut when object state changes" — 用可观察标准 |

**设计理由**: 语义描述要求模型先为每段命名（"这是切菜"），再确定边界；边界判据直接告诉模型判断规则（"看到目标变化就切"），跳过命名步骤，更适合 RL 训练中的 trial-and-error 学习。

### 2. 领域词汇：cooking-specific vs domain-agnostic

| | PA1 | PA2 |
|---|---|---|
| 领域词 | "cooking", "recipe subgoal", "ingredients", "cutting, stirring, pouring" | "task unit", "sub-goal", "object/material change" |

**设计理由**: PA1 的 cooking 词汇在 YoucCook2 上可能略优（精准语义），但 PA2 的通用表述训练后的模型可能更容易迁移到其他程序性视频（手术、装配、运动）。消融实验量化领域词汇在域内训练的增益/损失。

### 3. 稀疏采样感知

| | PA1 | PA2 |
|---|---|---|
| 帧率说明 | 无 | "sampled at 1-2 fps (not continuous video)" |
| 行为约束 | 无 | "Do NOT rely on single-frame micro-motions" |
| 边界标准 | 无 | "boundary ONLY when change is sustained across multiple frames" |

**设计理由**: VLM 预训练通常接触连续视频，可能不知道输入已稀疏采样。不告知时，模型可能：
- 把帧间差异当作真实边界（实际是采样间隔造成的跳变）
- 对"瞬间动作"过于敏感，产生大量 <5s 的碎片段
- PA2 显式声明稀疏采样，预期减少碎片化分割

### 4. 硬规则 (Hard Rules)

| 规则 | PA1 | PA2-L2 | PA2-L3 |
|------|-----|--------|--------|
| 最小时长 | 无 | 5s | 2s |
| 最大时长 | 无 | — | 15s |
| 预期段数 | 无 | 2-8 | 3-8 |
| 合并规则 | 无 | 同 sub-goal 合并 | 同对象同变化合并 |
| 切分规则 | 无 | — | 状态变化暂停 >3s 则切分 |

**设计理由**: 硬规则在 RL 自由探索初期（reward 信号弱时）提供先验约束，帮助模型更快收敛到合理输出范围。相当于"不用从零试错，先知道大致该输出几段、每段多长"。

---

## 预期行为对比

| 维度 | PA1（语义描述） | PA2（边界判据） |
|------|---------------|----------------|
| 输出段数 | 可能偏多/偏少（无先验） | 趋向 2-8 (L2) / 3-8 (L3) |
| 段长分布 | 可能有 <5s 碎片 | 被硬规则约束 |
| 碎片化风险 | 高（不知道是稀疏采样） | 低（显式约束） |
| 领域绑定 | 强（cooking 词汇） | 弱（通用表述） |
| 收敛速度 | 可能较慢（纯探索） | 可能较快（硬规则 = 先验） |

---

## 目录结构

```
prompt_ablation/
  exp_pa1_original.sh           PA1: 原始标注 prompt 实验
  exp_pa2_v3boundary.sh         PA2: V3 边界判据 prompt 实验
  run_prompt_ablation.sh        批量运行 PA1/PA2
  prompt_variants_v3.py         V3 prompt 模板定义（L2/L3 × V1-V4）
  prepare_prompt_data.py        数据准备（替换 prompt + 生成 train/val）
  prompt_design.md              设计文档（术语映射、消融逻辑）
  README.md                     本文档
```
