"""
prompt_variants_v4.py — Shot-First 两步式训练 prompt

核心思路 (vs V3 boundary-criterion):
  V3: 直接描述边界判据 (intent shift / task unit / state change) + 硬性数量/时长规则
  V4: 先识别 shot boundaries → 再按层级规则做 MERGE/SPLIT (镜像 scene-first 标注流程)
      不限定具体数量和时长，让模型根据视觉内容自行判断粒度

  三层统一框架:
    STEP 1 — 识别 visual shot transitions (所有层共用)
    STEP 2 — 按层级粒度执行:
      L1: 聚合 shots → phases (intent shift)
      L2: KEEP / MERGE / SPLIT shots → events (task unit)
      L3: shot 内按 state change → sub-actions (最细粒度)

  设计选择:
    - 仅保留 no-CoT 版本
    - 所有 prompt 领域无关 (无 cooking/sports 等词汇)
    - 不含硬性时长/数量规则
    - 模板参数: {duration} (秒), 与 prepare_prompt_data.py 兼容
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
# L1 — Shot → Phase Aggregation
#
# 输入: 全视频 (1fps resampled)
# 输出: 高层 phase 分割
# 策略: 识别 shots → 按 intent/process shift 聚合为 phases
# =====================================================================

L1_V1 = """\
You are given a {{duration}}s video clip (timestamps 0 to {{duration}}), sampled at 1-2 fps.

Segment the video into high-level phases using a SHOT-FIRST approach:

STEP 1 — IDENTIFY SHOTS:
Scan the video for visual shot transitions — moments where the scene, \
camera angle, setting, or subject clearly changes. \
These transitions are your temporal anchors.

STEP 2 — GROUP INTO PHASES:
Group consecutive shots that share the same overall intent or process into phases. \
Place a phase boundary when the grouped intent clearly shifts \
(e.g., from preparation to execution, from one major task to another).

MERGE consecutive shots into one phase when:
- They share the same overall goal or process stage.
- Only the camera angle or framing changes, not the underlying intent.

Keep shots SEPARATE (new phase) when:
- The person's primary purpose or overall process clearly changes.
- A major sub-goal is completed and a distinctly different one begins.
- There is a location/time jump or a completely different activity starts.

{sparse}

Include ALL shots — every shot must belong to a phase. Do not skip intros, outros, or narration spans if they contain visual shot content.

Output the start and end time (integer seconds, 0-based) for each phase in chronological order:
<events>[[start_time, end_time], ...]</events>

Example: <events>[[0, 85], [90, 170], [180, 240]]</events>""".format(sparse=_SPARSE_SAMPLING_NOTICE)


# =====================================================================
# L2 — Shot → Event via KEEP / MERGE / SPLIT
#
# 输入: Phase clip
# 输出: 中层 event 检测
# 策略: 识别 shots → 用 KEEP/MERGE/SPLIT 三种操作重组为 events
# =====================================================================

L2_V1 = """\
You are given a {{duration}}s video clip (timestamps 0 to {{duration}}), sampled at 1-2 fps.

Detect all events in this clip using a SHOT-FIRST approach:

STEP 1 — IDENTIFY SHOTS:
Scan the clip for visual shot transitions — moments where the scene, \
camera angle, setting, or subject clearly changes. \
These transitions are your temporal anchors.

STEP 2 — RESTRUCTURE INTO EVENTS using three operations:
- KEEP: A single shot that contains one coherent task becomes one event.
- MERGE: Consecutive shots that belong to the same local task \
(e.g., different angles of the same activity, shot/reverse-shot within one action) \
become one event.
- SPLIT: A long single-shot segment that contains multiple distinct tasks \
should be split into separate events at the task transition points.

When to MERGE: Combine consecutive shots into ONE event when they show \
the same ongoing activity from different angles, or continuous action across camera cuts.

When to keep SEPARATE: A new self-contained local task begins; \
a different location, subject, or sub-goal appears.

When to SPLIT: Within an uncut shot, the person clearly transitions from \
one self-contained task to a different one.

{sparse}

Output the start and end time (integer seconds, 0-based) for each event in chronological order:
<events>[[start_time, end_time], ...]</events>

Example: <events>[[5, 42], [55, 90]]</events>""".format(sparse=_SPARSE_SAMPLING_NOTICE)


# =====================================================================
# L3 — Within-Shot State-Change Segmentation
#
# 输入: Event clip (带 padding)
# 输出: 细粒度 sub-action 分割
# 策略: 识别 discontinuities (多为单镜头) → 在连续镜头内按 state change 拆分
# =====================================================================

L3_V1 = """\
You are given a {{duration}}s video clip, sampled at 1-2 fps.

Detect all fine-grained sub-actions in this clip using a SHOT-FIRST approach:

STEP 1 — IDENTIFY DISCONTINUITIES:
Scan for any visual discontinuities — shot transitions, camera cuts, \
or abrupt scene changes. At this granularity, most clips are single-shot, \
so discontinuities may be absent.

STEP 2 — SEGMENT BY STATE CHANGES:
Within each continuous shot (or the entire clip if no discontinuities exist), \
identify sub-actions based on visible state changes — moments where an object \
or material undergoes a clear, sustained transformation \
(deforms, separates, merges, changes position, or changes state).

Place a boundary when:
- A new visible object/material change begins.
- An ongoing state change completes and a different one starts.

Do NOT place a boundary when:
- Hands or body parts reposition without changing any object's state.
- Camera angle changes or brief occlusions occur.
- A single-frame flicker is not sustained across 2 or more sampled frames.

{sparse}

Output the start and end time (integer seconds, 0-based) for each segment in chronological order:
<events>[[start_time, end_time], ...]</events>

Example: <events>[[2, 6], [9, 13], [15, 20]]</events>""".format(sparse=_SPARSE_SAMPLING_NOTICE)


# =====================================================================
# Registry (与 prepare_prompt_data.py 接口兼容)
# =====================================================================

PROMPT_VARIANTS_V4 = {
    "L1": {"V1": L1_V1},
    "L2": {"V1": L2_V1},
    "L3": {"V1": L3_V1},
}

VARIANT_DESCRIPTIONS_V4 = {
    "V1": "Shot-first two-step: identify shots → merge/split by level (no CoT)",
}

# 模板参数: 只需 duration
PROMPT_PARAMS = {"duration"}

# MAX_RESPONSE_LEN 建议
RESPONSE_LEN_HINTS_V4 = {
    "V1": 512,
}
