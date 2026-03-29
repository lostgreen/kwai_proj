"""
prompt_variants_v3.py — 边界判据导向 + 稀疏采样感知的层次分割 prompt

核心改进 (vs V2):
  1. 三层定义从 "语义描述" 改为 "边界标准":
     - L1 = intent/process shift  (目标是否变了)
     - L2 = local task unit       (是否完成了一个局部任务)
     - L3 = visible state-change unit (画面可见的对象状态变化是否开始/结束)

  2. 显式低帧率约束:
     - 告知模型输入为 1-2 fps 稀疏采样
     - 禁止依赖瞬时微动作 / 单帧接触变化

  3. L3 不再强调 "atomic"，改为 "minimal visible state-change segment":
     - 在稀疏帧下可靠推断的最小状态变化单元
     - 避免理想化的 "原子操作" 概念

  4. 三层均加硬规则: 最小时长 / 合并规则 / 切分触发条件

2×2 factorial design (与 V2 框架保持一致):
    V1 (baseline):        Boundary-criterion   × No-CoT
    V2 (enhanced-rules):  Hard rules + priors  × No-CoT
    V3 (cot):             Boundary-criterion   × Structured-CoT
    V4 (full):            Hard rules + priors  × Structured-CoT
"""

# =====================================================================
# 共用片段: 稀疏采样约束
# =====================================================================

_SPARSE_SAMPLING_NOTICE = """\
IMPORTANT — SPARSE SAMPLING:
This clip is sampled at 1-2 fps (not continuous video). \
Do NOT rely on single-frame micro-motions, instantaneous contact changes, \
or camera cuts to place boundaries. \
Create a boundary ONLY when the change is sustained across multiple sampled frames \
or when the task/state clearly shifts."""


# =====================================================================
# L1 — Intent / Process Shift
#
# 边界标准: 当整体意图或目的明确变更时切分
# =====================================================================

L1_V1 = """\
You are given a {{duration}}s video clip (timestamps 0 to {{duration}}), sampled at 1-2 fps.

Segment the video into high-level phases by INTENT SHIFT. \
A phase boundary occurs when the person's primary purpose or process clearly changes \
(e.g., from preparation to execution, from one major task to another).

{sparse}

Skip non-active spans (narration, idle waiting, irrelevant content).

Output the start and end time (integer seconds, 0-based) for each phase in chronological order:
<events>[[start_time, end_time], ...]</events>

Example: <events>[[0, 85], [90, 170], [180, 240]]</events>""".format(sparse=_SPARSE_SAMPLING_NOTICE)


L1_V2 = """\
You are given a {{duration}}s video clip (timestamps 0 to {{duration}}), sampled at 1-2 fps.

Segment the video into high-level phases by INTENT SHIFT.

BOUNDARY CRITERION — cut when:
- The person's primary purpose or overall process clearly changes.
- A major sub-goal is completed and a new one begins.
DO NOT cut when:
- Only the specific tool/object changes but the intent stays the same.
- Camera angle shifts or brief pauses occur without intent change.

{sparse}

HARD RULES:
- Minimum phase duration: 15 seconds. Merge shorter phases with neighbors.
- Expected count: 3-6 phases for a {{duration}}s clip.
- If two adjacent phases share the same intent, merge them into one.
- Skip intros, outros, narration-only spans, or idle content.

Output the start and end time (integer seconds, 0-based) for each phase in chronological order:
<events>[[start_time, end_time], ...]</events>

Example: <events>[[0, 85], [90, 170], [180, 240]]</events>""".format(sparse=_SPARSE_SAMPLING_NOTICE)


L1_V3 = """\
You are given a {{duration}}s video clip (timestamps 0 to {{duration}}), sampled at 1-2 fps.

Segment the video into high-level phases by INTENT SHIFT. \
A phase boundary occurs when the person's primary purpose or process clearly changes.

{sparse}

Skip non-active spans (narration, idle waiting, irrelevant content).

First, reason about intent shifts in a <think> block, then output the phases.

<think>
Intent timeline: [describe the primary purpose at each stage of the video]
Shift points: [identify where the person's overall goal clearly changes]
Non-active spans: [list narration, idle, or irrelevant spans to skip]
</think>
<events>[[start_time, end_time], ...]</events>

Example:
<think>
Intent timeline: 0-85s the person gathers and prepares materials (intent=preparation). \
90-170s the person assembles and processes them (intent=execution). \
180-240s the person finishes and organizes the result (intent=finalization).
Shift points: ~85s preparation→execution, ~170s execution→finalization.
Non-active spans: 85-89s idle transition.
</think>
<events>[[0, 85], [90, 170], [180, 240]]</events>""".format(sparse=_SPARSE_SAMPLING_NOTICE)


L1_V4 = """\
You are given a {{duration}}s video clip (timestamps 0 to {{duration}}), sampled at 1-2 fps.

Segment the video into high-level phases by INTENT SHIFT.

BOUNDARY CRITERION — cut when:
- The person's primary purpose or overall process clearly changes.
- A major sub-goal is completed and a new one begins.
DO NOT cut when:
- Only the specific tool/object changes but the intent stays the same.
- Camera angle shifts or brief pauses occur without intent change.

{sparse}

HARD RULES:
- Minimum phase duration: 15 seconds. Merge shorter phases with neighbors.
- Expected count: 3-6 phases for a {{duration}}s clip.
- If two adjacent phases share the same intent, merge them into one.
- Skip intros, outros, narration-only spans, or idle content.

First, reason about intent shifts in a <think> block, then output the phases.

<think>
Intent timeline: [describe the primary purpose at each stage of the video]
Shift points: [identify where the person's overall goal clearly changes]
Non-active spans: [list narration, idle, or irrelevant spans to skip]
</think>
<events>[[start_time, end_time], ...]</events>

Example:
<think>
Intent timeline: 0-85s the person gathers and prepares materials (intent=preparation). \
90-170s the person assembles and processes them (intent=execution). \
180-240s the person finishes and organizes the result (intent=finalization).
Shift points: ~85s preparation→execution, ~170s execution→finalization.
Non-active spans: 85-89s idle transition.
</think>
<events>[[0, 85], [90, 170], [180, 240]]</events>""".format(sparse=_SPARSE_SAMPLING_NOTICE)


# =====================================================================
# L2 — Local Task Unit
#
# 边界标准: 当一个自洽的局部任务单元完成或转换时切分
# =====================================================================

L2_V1 = """\
You are given a {{duration}}s video clip (timestamps 0 to {{duration}}), sampled at 1-2 fps.

Detect all LOCAL TASK UNITS in this clip. \
A task unit is a coherent block of actions that together accomplish one self-contained local task. \
A boundary occurs when one local task is completed and a different one begins.

{sparse}

Skip non-active spans (idle waiting, narration, setup without progress).

Output the start and end time (integer seconds, 0-based) for each task unit in chronological order:
<events>[[start_time, end_time], ...]</events>

Example: <events>[[5, 42], [55, 90]]</events>""".format(sparse=_SPARSE_SAMPLING_NOTICE)


L2_V2 = """\
You are given a {{duration}}s video clip (timestamps 0 to {{duration}}), sampled at 1-2 fps.

Detect all LOCAL TASK UNITS in this clip.

BOUNDARY CRITERION — cut when:
- A self-contained local task is completed (the sub-goal is achieved or abandoned).
- The person starts working toward a clearly different sub-goal.
DO NOT cut when:
- The person switches tools/materials but continues the same task.
- Brief pauses, adjustments, or repositioning occur within the same task.

{sparse}

HARD RULES:
- Minimum segment duration: 5 seconds. Merge shorter segments with neighbors.
- Expected count: 2-8 task units for a {{duration}}s clip.
- Gaps between segments are expected — not every second needs to be covered.
- If two adjacent segments pursue the same sub-goal, merge them.

Output the start and end time (integer seconds, 0-based) for each task unit in chronological order:
<events>[[start_time, end_time], ...]</events>

Example: <events>[[5, 42], [55, 90]]</events>""".format(sparse=_SPARSE_SAMPLING_NOTICE)


L2_V3 = """\
You are given a {{duration}}s video clip (timestamps 0 to {{duration}}), sampled at 1-2 fps.

Detect all LOCAL TASK UNITS in this clip. \
A task unit is a coherent block of actions that together accomplish one self-contained local task. \
A boundary occurs when one local task is completed and a different one begins.

{sparse}

Skip non-active spans (idle waiting, narration, setup without progress).

First, reason about task boundaries in a <think> block, then output the segments.

<think>
Task sequence: [describe each distinct local task visible in the clip]
Boundary evidence: [for each boundary, explain what sub-goal ended and what new one began]
Non-active spans: [list idle, narration, or transition periods to exclude]
</think>
<events>[[start_time, end_time], ...]</events>

Example:
<think>
Task sequence: 5-42s the person performs a complete preparation task (gathering + processing materials). \
55-90s the person performs an assembly task (combining processed items).
Boundary evidence: At ~42s, preparation sub-goal is achieved. \
At ~55s, a new assembly sub-goal begins. 43-54s is idle transition.
Non-active spans: 0-4s intro, 43-54s idle.
</think>
<events>[[5, 42], [55, 90]]</events>""".format(sparse=_SPARSE_SAMPLING_NOTICE)


L2_V4 = """\
You are given a {{duration}}s video clip (timestamps 0 to {{duration}}), sampled at 1-2 fps.

Detect all LOCAL TASK UNITS in this clip.

BOUNDARY CRITERION — cut when:
- A self-contained local task is completed (the sub-goal is achieved or abandoned).
- The person starts working toward a clearly different sub-goal.
DO NOT cut when:
- The person switches tools/materials but continues the same task.
- Brief pauses, adjustments, or repositioning occur within the same task.

{sparse}

HARD RULES:
- Minimum segment duration: 5 seconds. Merge shorter segments with neighbors.
- Expected count: 2-8 task units for a {{duration}}s clip.
- Gaps between segments are expected — not every second needs to be covered.
- If two adjacent segments pursue the same sub-goal, merge them.

First, reason about task boundaries in a <think> block, then output the segments.

<think>
Task sequence: [describe each distinct local task visible in the clip]
Boundary evidence: [for each boundary, explain what sub-goal ended and what new one began]
Non-active spans: [list idle, narration, or transition periods to exclude]
</think>
<events>[[start_time, end_time], ...]</events>

Example:
<think>
Task sequence: 5-42s the person performs a complete preparation task (gathering + processing materials). \
55-90s the person performs an assembly task (combining processed items).
Boundary evidence: At ~42s, preparation sub-goal is achieved. \
At ~55s, a new assembly sub-goal begins. 43-54s is idle transition.
Non-active spans: 0-4s intro, 43-54s idle.
</think>
<events>[[5, 42], [55, 90]]</events>""".format(sparse=_SPARSE_SAMPLING_NOTICE)


# =====================================================================
# L3 — Visible State-Change Unit
#
# 边界标准: 画面中可见的对象/状态变化是否开始或结束
# 核心改进: 不再使用 "atomic operation"，改为 "minimal visible state-change segment"
# 明确适配稀疏帧输入
# =====================================================================

L3_V1 = """\
You are given a {{duration}}s video clip, sampled at 1-2 fps.

Detect all VISIBLE STATE-CHANGE segments in this clip. \
A state-change segment is the shortest time span during which a visible object or material \
undergoes a clear, sustained change that can be reliably inferred from sparse frames \
(e.g., material deforms, separates, merges, changes position, or changes state).

{sparse}

Skip idle pauses, repositioning without state change, or narration.

Output the start and end time (integer seconds, 0-based) for each segment in chronological order:
<events>[[start_time, end_time], ...]</events>

Example: <events>[[2, 6], [9, 13], [15, 20]]</events>""".format(sparse=_SPARSE_SAMPLING_NOTICE)


L3_V2 = """\
You are given a {{duration}}s video clip, sampled at 1-2 fps.

Detect all VISIBLE STATE-CHANGE segments in this clip.

BOUNDARY CRITERION — cut when:
- A new visible object/material change begins (something starts to deform, separate, merge, \
transfer, or change state).
- An ongoing state change completes (the object reaches its new state and motion stops).
DO NOT cut when:
- Hands or body parts reposition without changing any object's state.
- Camera angle changes or brief occlusions occur.
- You see a single-frame flicker that is not sustained across ≥2 sampled frames.

{sparse}

HARD RULES:
- Minimum segment duration: 2 seconds. If a change appears shorter, \
extend boundaries to the nearest sustained-change frames.
- Maximum segment duration: 15 seconds — if longer, it likely contains multiple changes; split them.
- Expected count: 3-8 segments for a {{duration}}s clip.
- Gaps between segments are expected — not every second is active.
- If two adjacent segments involve the same object undergoing the same continuous change, \
merge them into one.

Output the start and end time (integer seconds, 0-based) for each segment in chronological order:
<events>[[start_time, end_time], ...]</events>

Example: <events>[[2, 6], [9, 13], [15, 20]]</events>""".format(sparse=_SPARSE_SAMPLING_NOTICE)


L3_V3 = """\
You are given a {{duration}}s video clip, sampled at 1-2 fps.

Detect all VISIBLE STATE-CHANGE segments in this clip. \
A state-change segment is the shortest time span during which a visible object or material \
undergoes a clear, sustained change that can be reliably inferred from sparse frames.

{sparse}

Skip idle pauses, repositioning without state change, or narration.

First, reason about visible state changes in a <think> block, then output the segments.

<think>
State changes: [describe each visible object/material change in chronological order]
Boundary evidence: [for each segment, identify the frame(s) where the change starts and ends]
Skip list: [note repositioning, idle pauses, or ambiguous single-frame changes to exclude]
</think>
<events>[[start_time, end_time], ...]</events>

Example:
<think>
State changes: 2-6s a liquid visibly flows from container A to B (state: transfer). \
9-13s a solid material is deformed by a tool (state: shape change). \
15-20s an item is moved from surface to container (state: position change).
Boundary evidence: Liquid flow starts at frame ~2s and ends at ~6s (sustained across 4+ frames). \
Deformation visible from ~9s to ~13s. Transfer from ~15s to ~20s.
Skip list: 7-8s hand repositioning — no object state change visible.
</think>
<events>[[2, 6], [9, 13], [15, 20]]</events>""".format(sparse=_SPARSE_SAMPLING_NOTICE)


L3_V4 = """\
You are given a {{duration}}s video clip, sampled at 1-2 fps.

Detect all VISIBLE STATE-CHANGE segments in this clip.

BOUNDARY CRITERION — cut when:
- A new visible object/material change begins (something starts to deform, separate, merge, \
transfer, or change state).
- An ongoing state change completes (the object reaches its new state and motion stops).
DO NOT cut when:
- Hands or body parts reposition without changing any object's state.
- Camera angle changes or brief occlusions occur.
- You see a single-frame flicker that is not sustained across ≥2 sampled frames.

{sparse}

HARD RULES:
- Minimum segment duration: 2 seconds. If a change appears shorter, \
extend boundaries to the nearest sustained-change frames.
- Maximum segment duration: 15 seconds — if longer, it likely contains multiple changes; split them.
- Expected count: 3-8 segments for a {{duration}}s clip.
- Gaps between segments are expected — not every second is active.
- If two adjacent segments involve the same object undergoing the same continuous change, \
merge them into one.

First, reason about visible state changes in a <think> block, then output the segments.

<think>
State changes: [describe each visible object/material change in chronological order]
Boundary evidence: [for each segment, identify the frame(s) where the change starts and ends]
Skip list: [note repositioning, idle pauses, or ambiguous single-frame changes to exclude]
</think>
<events>[[start_time, end_time], ...]</events>

Example:
<think>
State changes: 2-6s a liquid visibly flows from container A to B (state: transfer). \
9-13s a solid material is deformed by a tool (state: shape change). \
15-20s an item is moved from surface to container (state: position change).
Boundary evidence: Liquid flow starts at frame ~2s and ends at ~6s (sustained across 4+ frames). \
Deformation visible from ~9s to ~13s. Transfer from ~15s to ~20s.
Skip list: 7-8s hand repositioning — no object state change visible.
</think>
<events>[[2, 6], [9, 13], [15, 20]]</events>""".format(sparse=_SPARSE_SAMPLING_NOTICE)


# =====================================================================
# Registry (与 V2 接口完全兼容)
# =====================================================================

PROMPT_VARIANTS_V3 = {
    "L1": {"V1": L1_V1, "V2": L1_V2, "V3": L1_V3, "V4": L1_V4},
    "L2": {"V1": L2_V1, "V2": L2_V2, "V3": L2_V3, "V4": L2_V4},
    "L3": {"V1": L3_V1, "V2": L3_V2, "V3": L3_V3, "V4": L3_V4},
}

# 向后兼容: 让 prepare_prompt_data.py 也能直接引用 V3 版本
PROMPT_VARIANTS_V2 = PROMPT_VARIANTS_V3

VARIANT_DESCRIPTIONS_V3 = {
    "V1": "Boundary-criterion baseline (sparse-aware, intent/task/state-change hierarchy, no CoT)",
    "V2": "Enhanced hard rules (min duration, merge rules, split triggers, count priors, no CoT)",
    "V3": "Boundary-criterion + Structured-CoT (V1 definitions + 3-step reasoning)",
    "V4": "Full: hard rules + Structured-CoT (V2 rules + V3 reasoning)",
}

VARIANT_DESCRIPTIONS_V2 = VARIANT_DESCRIPTIONS_V3

# Prompt 参数: 只需 duration
PROMPT_PARAMS = {"duration"}

# MAX_RESPONSE_LEN 建议
RESPONSE_LEN_HINTS = {
    "V1": 512,
    "V2": 512,
    "V3": 1024,
    "V4": 1024,
}
