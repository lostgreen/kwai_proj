"""
prompt_variants_v2.py — 通用层次分割 prompt 消融实验（V2 版本）

设计原则：
  1. 领域无关（Domain-agnostic）：不含 "cooking"、"recipe"、"ingredients" 等领域词汇
  2. L3 全部改为自由分割（no query），与 L1/L2 任务结构对齐
  3. 保留 2×2 消融设计（粒度描述 × 结构化 CoT）

2×2 factorial design:
    V1 (baseline):        Minimal           × No-CoT
    V2 (granularity):     Granularity-Enhanced × No-CoT
    V3 (cot):             Minimal           × Structured-CoT
    V4 (gran+cot):        Granularity-Enhanced × Structured-CoT

每个 variant 提供 L1（帧编号）、L2（秒级时间戳）、L3（秒级时间戳，自由分割）三套模板。
"""

# ==========================================================================
# L1 — 高层阶段分割（时间戳模式, duration 参数）
# ==========================================================================

L1_V1 = """\
You are given a {duration}s video clip (timestamps 0 to {duration}). \
Segment the video into high-level activity phases. \
Skip non-active spans such as narration, idle waiting, or irrelevant content.

Output the start and end time (integer seconds, 0-based) for each phase in order:
<events>[[start_time, end_time], ...]</events>

Example: <events>[[0, 85], [90, 170], [180, 240]]</events>"""


L1_V2 = """\
You are given a {duration}s video clip (timestamps 0 to {duration}). \
Segment the video into high-level activity phases.

PHASE DEFINITION:
- A high-level phase is a broad structural stage (typically 3-5 per video) \
that represents a distinct process block or sub-goal within the overall activity.
- Each phase spans many seconds and may contain multiple fine-grained actions.
- Phases do NOT need to cover the entire {duration}s clip. \
Skip intros, outros, narration-only spans, or idle content not advancing the activity.

DO NOT output:
- More than 6 phases (too fragmented — you are splitting too finely)
- A single phase covering almost the entire clip (too coarse — look for natural stage boundaries)
- Phases for non-active content (narration, reactions, idle setup)

SEGMENT RULES:
- Group by intent, not by camera cut or single-motion change.
- Expect 3-6 phases for a {duration}s clip.

Output the start and end time (integer seconds, 0-based) for each phase in order:
<events>[[start_time, end_time], ...]</events>

Example: <events>[[0, 85], [90, 170], [180, 240]]</events>"""


L1_V3 = """\
You are given a {duration}s video clip (timestamps 0 to {duration}). \
Segment the video into high-level activity phases. \
Skip non-active spans such as narration, idle waiting, or irrelevant content.

First, reason about the activity structure in a <think> block. Then output the phases.

<think>
Observations: [describe the main activities visible in the video chronologically]
Stage boundaries: [identify where the activity's primary intent clearly shifts]
Non-active spans: [list any narration, idle, or irrelevant spans to skip]
</think>
<events>[[start_time, end_time], ...]</events>

Example:
<think>
Observations: The video shows three distinct stages: setup and preparation in the early seconds, \
a main execution phase in the middle, and assembly or finalization near the end.
Stage boundaries: Activity intent shifts at ~90s (setup → execution) and ~170s (execution → finalization).
Non-active spans: 85-89s show narration with no hands-on activity.
</think>
<events>[[0, 85], [90, 170], [180, 240]]</events>"""


L1_V4 = """\
You are given a {duration}s video clip (timestamps 0 to {duration}). \
Segment the video into high-level activity phases.

PHASE DEFINITION:
- A high-level phase is a broad structural stage (typically 3-5 per video) \
that represents a distinct process block or sub-goal within the overall activity.
- Each phase spans many seconds and may contain multiple fine-grained actions.
- Phases do NOT need to cover the entire {duration}s clip. \
Skip intros, outros, narration-only spans, or idle content not advancing the activity.

DO NOT output:
- More than 6 phases (too fragmented — you are splitting too finely)
- A single phase covering almost the entire clip (too coarse — look for natural stage boundaries)
- Phases for non-active content (narration, reactions, idle setup)

SEGMENT RULES:
- Group by intent, not by camera cut or single-motion change.
- Expect 3-6 phases for a {duration}s clip.

First, reason about the activity structure in a <think> block. Then output the phases.

<think>
Observations: [describe the main activities visible in the video chronologically]
Stage boundaries: [identify where the activity's primary intent clearly shifts]
Non-active spans: [list any narration, idle, or irrelevant spans to skip]
</think>
<events>[[start_time, end_time], ...]</events>

Example:
<think>
Observations: The video shows three distinct stages: setup and preparation in the early seconds, \
a main execution phase in the middle, and assembly or finalization near the end.
Stage boundaries: Activity intent shifts at ~90s (setup → execution) and ~170s (execution → finalization).
Non-active spans: 85-89s show narration with no hands-on activity.
</think>
<events>[[0, 85], [90, 170], [180, 240]]</events>"""


# ==========================================================================
# L2 — 中层目标事件检测（秒级时间戳, duration 参数）
# ==========================================================================

L2_V1 = """\
You are given a {duration}s video clip (timestamps 0 to {duration}). \
Detect all goal-directed activity segments in this clip. \
Each segment is a multi-second, purposeful sequence of actions that advances toward a specific sub-goal. \
Skip non-active spans such as idle waiting, narration, or setup without progress.

Output the start and end time (integer seconds, 0-based) for each segment in order:
<events>[[start_time, end_time], ...]</events>

Example: <events>[[5, 42], [55, 90]]</events>"""


L2_V2 = """\
You are given a {duration}s video clip (timestamps 0 to {duration}). \
Detect all goal-directed activity segments in this clip.

SEGMENT DEFINITION:
- A goal-directed activity segment is a multi-second, purposeful sequence (typically 10-60s) \
that transforms state or completes a meaningful sub-goal.
- It involves multiple physical interactions unified by a single intent.
- Each segment must be MORE specific than a whole activity stage, \
but LESS granular than a single physical motion.

DO NOT output:
- Segments shorter than 5 seconds (too atomic — single actions, not segments)
- Segments that merely restate the entire clip's content (too coarse)
- Non-active spans (idle periods, narration, setup without state change)

SEGMENT RULES:
- Segments do NOT need to cover the entire clip. Gaps are expected.
- Expect roughly 2-8 segments for a {duration}s clip.

Output the start and end time (integer seconds, 0-based) for each segment in order:
<events>[[start_time, end_time], ...]</events>

Example: <events>[[5, 42], [55, 90]]</events>"""


L2_V3 = """\
You are given a {duration}s video clip (timestamps 0 to {duration}). \
Detect all goal-directed activity segments in this clip. \
Each segment is a multi-second, purposeful sequence of actions that advances toward a specific sub-goal. \
Skip non-active spans such as idle waiting, narration, or setup without progress.

First, reason about the activity structure in a <think> block. Then output the segments.

<think>
Observations: [describe what happens in the video chronologically]
Segment grouping: [identify which actions share a unified sub-goal and should be grouped together]
Non-active spans: [identify idle, narration, or setup periods to exclude]
</think>
<events>[[start_time, end_time], ...]</events>

Example:
<think>
Observations: The clip shows two main activity blocks separated by an idle period. \
The first involves a multi-step preparation sequence (5-42s), \
then idle waiting and narration (43-54s), \
then a second goal-directed sequence completing assembly (55-90s).
Segment grouping: Group 5-42s as one segment (unified sub-goal: preparation). \
Group 55-90s as another (sub-goal: assembly).
Non-active spans: 43-54s is idle waiting — skip.
</think>
<events>[[5, 42], [55, 90]]</events>"""


L2_V4 = """\
You are given a {duration}s video clip (timestamps 0 to {duration}). \
Detect all goal-directed activity segments in this clip.

SEGMENT DEFINITION:
- A goal-directed activity segment is a multi-second, purposeful sequence (typically 10-60s) \
that transforms state or completes a meaningful sub-goal.
- It involves multiple physical interactions unified by a single intent.
- Each segment must be MORE specific than a whole activity stage, \
but LESS granular than a single physical motion.

DO NOT output:
- Segments shorter than 5 seconds (too atomic — single actions, not segments)
- Segments that merely restate the entire clip's content (too coarse)
- Non-active spans (idle periods, narration, setup without state change)

SEGMENT RULES:
- Segments do NOT need to cover the entire clip. Gaps are expected.
- Expect roughly 2-8 segments for a {duration}s clip.

First, reason about the activity structure in a <think> block. Then output the segments.

<think>
Observations: [describe what happens in the video chronologically]
Segment grouping: [identify which actions share a unified sub-goal and should be grouped together]
Non-active spans: [identify idle, narration, or setup periods to exclude]
</think>
<events>[[start_time, end_time], ...]</events>

Example:
<think>
Observations: The clip shows two main activity blocks separated by an idle period. \
The first involves a multi-step preparation sequence (5-42s), \
then idle waiting and narration (43-54s), \
then a second goal-directed sequence completing assembly (55-90s).
Segment grouping: Group 5-42s as one segment (unified sub-goal: preparation). \
Group 55-90s as another (sub-goal: assembly).
Non-active spans: 43-54s is idle waiting — skip.
</think>
<events>[[5, 42], [55, 90]]</events>"""


# ==========================================================================
# L3 — 原子操作分割（秒级时间戳, duration 参数）
#
# [变更] 不再是 query-conditioned grounding，改为自由分割：
#   - 不给 action 描述列表
#   - 模型自主检测所有原子物理操作
#   - 与 L1/L2 任务结构对齐
# ==========================================================================

L3_V1 = """\
You are given a {duration}s video clip. \
Detect all atomic physical operations in this clip. \
Each operation is a brief, single-step physical state change (e.g., a pour, cut, stir, or transfer). \
Skip idle pauses, repositioning, or narration.

Output the start and end time (integer seconds, 0-based) for each operation in chronological order:
<events>[[start_time, end_time], ...]</events>

Example: <events>[[2, 6], [9, 13], [15, 20]]</events>"""


L3_V2 = """\
You are given a {duration}s video clip. \
Detect all atomic physical operations in this clip.

OPERATION DEFINITION:
- An atomic physical operation is a brief (typically 2-8s), single-step interaction \
where exactly one object undergoes one discrete state change \
(e.g., deformation, separation, merging, transfer, material state change).
- It must be MORE specific than a multi-step sequence, \
but LESS granular than a sub-motion (reaching, gripping, repositioning).

DO NOT output:
- Operations shorter than 1 second (too instantaneous to be a full state change)
- Pure body or limb movements with no object state change (reaching, adjusting grip)
- Multi-step sequences that could be split into multiple atomic operations

SEGMENT RULES:
- Operations do NOT need to cover the entire clip. Gaps between operations are expected.
- Expect roughly 3-8 operations for a {duration}s clip.

Output the start and end time (integer seconds, 0-based) for each operation in chronological order:
<events>[[start_time, end_time], ...]</events>

Example: <events>[[2, 6], [9, 13], [15, 20]]</events>"""


L3_V3 = """\
You are given a {duration}s video clip. \
Detect all atomic physical operations in this clip. \
Each operation is a brief, single-step physical state change (e.g., a pour, cut, stir, or transfer). \
Skip idle pauses, repositioning, or narration.

First, reason about the atomic operations in a <think> block. Then output the timestamps.

<think>
Observations: [describe the physical interactions visible in chronological order]
Operation boundaries: [identify each discrete state-change moment and its approximate duration]
Skip list: [note any repositioning, idle pause, or narration to exclude]
</think>
<events>[[start_time, end_time], ...]</events>

Example:
<think>
Observations: The clip shows several distinct physical interactions: \
a liquid pour (2-6s), then brief repositioning (7-8s, skip), \
a mixing motion with a utensil (9-13s), and a transfer of material (15-20s).
Operation boundaries: Pour starts at contact (2s) ends when flow stops (6s); \
mix starts with utensil contact (9s) ends when motion stops (13s); \
transfer starts at pickup (15s) ends when object placed (20s).
Skip list: 7-8s repositioning — no object state change.
</think>
<events>[[2, 6], [9, 13], [15, 20]]</events>"""


L3_V4 = """\
You are given a {duration}s video clip. \
Detect all atomic physical operations in this clip.

OPERATION DEFINITION:
- An atomic physical operation is a brief (typically 2-8s), single-step interaction \
where exactly one object undergoes one discrete state change \
(e.g., deformation, separation, merging, transfer, material state change).
- It must be MORE specific than a multi-step sequence, \
but LESS granular than a sub-motion (reaching, gripping, repositioning).

DO NOT output:
- Operations shorter than 1 second (too instantaneous to be a full state change)
- Pure body or limb movements with no object state change (reaching, adjusting grip)
- Multi-step sequences that could be split into multiple atomic operations

SEGMENT RULES:
- Operations do NOT need to cover the entire clip. Gaps between operations are expected.
- Expect roughly 3-8 operations for a {duration}s clip.

First, reason about the atomic operations in a <think> block. Then output the timestamps.

<think>
Observations: [describe the physical interactions visible in chronological order]
Operation boundaries: [identify each discrete state-change moment and its approximate duration]
Skip list: [note any repositioning, idle pause, or narration to exclude]
</think>
<events>[[start_time, end_time], ...]</events>

Example:
<think>
Observations: The clip shows several distinct physical interactions: \
a liquid pour (2-6s), then brief repositioning (7-8s, skip), \
a mixing motion with a utensil (9-13s), and a transfer of material (15-20s).
Operation boundaries: Pour starts at contact (2s) ends when flow stops (6s); \
mix starts with utensil contact (9s) ends when motion stops (13s); \
transfer starts at pickup (15s) ends when object placed (20s).
Skip list: 7-8s repositioning — no object state change.
</think>
<events>[[2, 6], [9, 13], [15, 20]]</events>"""


# ==========================================================================
# Registry
# ==========================================================================

PROMPT_VARIANTS_V2 = {
    "L1": {"V1": L1_V1, "V2": L1_V2, "V3": L1_V3, "V4": L1_V4},
    "L2": {"V1": L2_V1, "V2": L2_V2, "V3": L2_V3, "V4": L2_V4},
    "L3": {"V1": L3_V1, "V2": L3_V2, "V3": L3_V3, "V4": L3_V4},
}

VARIANT_DESCRIPTIONS_V2 = {
    "V1": "Baseline (minimal, domain-agnostic, no CoT)",
    "V2": "Granularity-Enhanced (explicit duration/count priors + DO-NOT rules, no CoT)",
    "V3": "Structured-CoT (minimal + 3-step think: Observations / Grouping / Skip)",
    "V4": "Granularity + Structured-CoT (V2 definition + V3 reasoning)",
}

# 所有层级的 prompt params 都只需要 duration 参数
# L1: duration (时间戳模式) / L2: duration / L3: duration (无 query 列表)
PROMPT_PARAMS = {"duration"}

# MAX_RESPONSE_LEN 建议
RESPONSE_LEN_HINTS = {
    "V1": 512,
    "V2": 512,
    "V3": 1024,   # <think> 内容占用更多 token
    "V4": 1024,
}
