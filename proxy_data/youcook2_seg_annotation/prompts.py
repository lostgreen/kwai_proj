"""
3-level hierarchical annotation prompts for YouCook2 DVC.

Level 0: System Prompt (global context — injected as system role)
Level 1: Macro Phase  (3-5 high-level narrative phases per video)
Level 2: Activity     (RESERVED — TODO: fill in when ready)
Level 3: Atomic Step  (RESERVED — TODO: fill in when ready)

Usage:
    from prompts import SYSTEM_PROMPT, get_level1_prompt, get_level2_prompt

The Level 1 call takes the video frames as visual input.
Level 2 takes both the video AND the Level 1 output as context.
Level 3 (if used) takes video + Level 1 + Level 2 context.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Level 0: System Prompt
# Injected as the system role message in every API call.
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert in structured video analysis, specializing in instructional content. \
Your task is to accurately parse narrative logic, temporal actions, and visual state transitions.

Strictly adhere to the following timestamp annotation rules:
1. [Sparsity over Continuity]: ONLY annotate segments with clear semantic meaning or \
visual actions. Skip irrelevant segments such as host monologues, meaningless static shots, \
B-roll, or transition effects. DO NOT force adjacent segments to be perfectly contiguous; \
gaps between timestamps are expected and encouraged.
2. [Task-Relevance First]: ONLY keep segments that materially advance preparation, cooking, \
assembly, or finalization of the target dish. Exclude pure narration, face-to-camera talking, \
beauty shots, eating/tasting reactions, unrelated setup/cleanup, idle waiting with no visible \
food change, and tool-only motions that do not affect ingredients or the dish.
3. [Precise Boundaries]: The 'start_time' must capture the exact moment an action or \
intention begins, and the 'end_time' must mark the exact moment the action concludes or \
the visual state solidifies.
4. [Formatting]: Output strictly in valid JSON format."""


# ─────────────────────────────────────────────────────────────────────────────
# Level 1: Macro Phase  (阶段级)
# Input:  1fps frame strip of the full windowed clip
# Output: JSON with macro_phases list
# ─────────────────────────────────────────────────────────────────────────────
_LEVEL1_BASE = """\
Analyze the instructional video and segment it into high-level macro phases \
(typically 3 to 5 phases). These phases should represent the major cooking-oriented \
subgoals required to complete the dish, not every visible scene change.

LEVEL 1 DEFINITION:
- A Level 1 phase is a broad task stage such as ingredient preparation, sauce making, \
cooking/heating, assembly, or plating/finalization.
- A Level 1 phase may be relatively long and may contain multiple fine-grained actions inside it.
- Level 1 phases do NOT need to cover the whole clip. Skip intros, outros, talking-only spans, \
beauty shots, unrelated serving/eating moments, or any content not advancing the dish.
- Prefer grouping by recipe intent, not by camera cut or tiny motion change.

For each phase, provide:
- phase_id: Sequential ID (starting from 1).
- start_time / end_time: The boundary timestamps (Format: MM:SS).
- phase_name: A concise noun phrase representing the phase (e.g., "Ingredient Preparation").
- narrative_summary: A single-sentence summary of the phase's core objective.

Output JSON format example:
{{
  "macro_phases": [
    {{
      "phase_id": 1,
      "start_time": "00:15",
      "end_time": "02:30",
      "phase_name": "Ingredient Preparation",
      "narrative_summary": "Prepare and wash all necessary vegetables and proteins."
    }}
  ]
}}"""


def get_level1_prompt(clip_duration_sec: float) -> str:
    """
    Build the Level 1 (Macro Phase) user-turn prompt.

    Args:
        clip_duration_sec: Duration of the windowed clip in seconds.
    Returns:
        User-turn prompt string (append frames as image/video input separately).
    """
    mm_ss = f"{int(clip_duration_sec)//60:02d}:{int(clip_duration_sec)%60:02d}"
    return (
        f"The video clip is {clip_duration_sec:.0f} seconds long (max timestamp {mm_ss}).\n\n"
        + _LEVEL1_BASE
    )


# ─────────────────────────────────────────────────────────────────────────────
# Level 2: Activity-level  (活动级)  ← RESERVED / TODO
# Input:  1fps frames + Level 1 macro_phases JSON context
# Output: activities within each macro phase
# ─────────────────────────────────────────────────────────────────────────────
# TODO: Fill in Level 2 prompt when ready.
# Expected output schema:
# {
#   "activities": [
#     {
#       "activity_id": int,
#       "parent_phase_id": int,
#       "start_time": "MM:SS",
#       "end_time": "MM:SS",
#       "activity_name": str,
#       "action_description": str
#     }
#   ]
# }
_LEVEL2_BASE = """Identify the core cooking events (Level 2 meso steps) that drive task progression within this macro phase.

LEVEL 2 DEFINITION:
- A Level 2 step is a cooking event: a multi-second, goal-directed workflow inside the current macro phase.
- It must directly transform ingredients, cookware contents, or the dish state, OR complete a meaningful recipe subgoal.
- It is smaller than a macro phase but larger than a single atomic motion.

CRITICAL GRANULARITY RULES FOR LEVEL 2:
1. [Aggregation Rule]: A Level 2 step MUST be a complete "Logical Event" or "Workflow", NOT an isolated atomic action. 
2. [Anti-Fragmentation]: DO NOT extract single, momentary physical actions (e.g., "pick up a spoon", "place filling", "fold a corner"). These are Level 3 details.
3. [Good vs. Bad Example]: 
   - BAD (Too granular): "Place a spoonful of filling onto the wrapper" (1 second).
   - GOOD (Correct Level 2): "Assemble the wonton by filling and folding the wrapper" (15-20 seconds).
4. [Cooking-Only Filter]: ONLY keep events that are truly part of making the dish. Exclude:
   - talking or explaining without corresponding food manipulation
   - showing ingredients/tools without using them
   - idle hand motion, repositioning, or tool pickup with no food-state change
   - waiting/resting/heating spans with no visible progress
   - beauty shots, serving, tasting, or reaction shots unless the instructional goal is specifically plating/final garnish
5. [Empty Is Valid]: If this macro phase contains no cooking-relevant event after filtering, return an empty list: {"meso_steps": []}.

For each grouped step, provide:
- step_id: Sequential ID.
- parent_phase_id: The phase_id of the macro phase this step belongs to.
- start_time / end_time: Precise boundary timestamps covering the ENTIRE logical event.
- instruction: A high-level description of the completed cooking event (e.g., "Assemble the dumplings").
- visual_keywords: 3 to 5 key visual elements present across this workflow.

Output JSON format example:
{
  "meso_steps": [
    {
      "step_id": 1,
      "parent_phase_id": 1,
      "start_time": "01:42",
      "end_time": "02:03",
      "instruction": "Assemble one wonton by filling and folding it",
      "visual_keywords": ["wonton wrapper", "meat filling", "hands folding"]
    }
  ]
}
"""


def get_level2_prompt(
    clip_duration_sec: float,
    level1_result: dict,
) -> str:
    """
    Build the Level 2 (Activity) user-turn prompt.

    Args:
        clip_duration_sec: Duration of the windowed clip in seconds.
        level1_result: Parsed JSON dict from Level 1 annotation.
    Returns:
        User-turn prompt string.
    Raises:
        NotImplementedError: Until the Level 2 prompt is filled in.
    """
    import json as _json
    if _LEVEL2_BASE.startswith("TODO"):
        raise NotImplementedError(
            "Level 2 prompt is not yet defined. "
            "Edit proxy_data/youcook2_seg_annotation/prompts.py and replace _LEVEL2_BASE."
        )
    l1_json = _json.dumps(level1_result, ensure_ascii=False, indent=2)
    return (
        f"The video clip is {clip_duration_sec:.0f} seconds long.\n\n"
        f"Level 1 macro phase annotation (use as context):\n{l1_json}\n\n"
        + _LEVEL2_BASE
    )


# ─────────────────────────────────────────────────────────────────────────────
# Level 3: Atomic Step  (动作级)  ← RESERVED / TODO
# Input:  1fps frames + Level 1 + Level 2 context
# Output: atomic steps within each activity
# ─────────────────────────────────────────────────────────────────────────────
# TODO: Fill in Level 3 prompt when ready.
# Expected output schema:
# {
#   "steps": [
#     {
#       "step_id": int,
#       "parent_activity_id": int,
#       "start_time": "MM:SS",
#       "end_time": "MM:SS",
#       "step_name": str,
#       "object_state_change": str   # optional
#     }
#   ]
# }
_LEVEL3_BASE = """Now, deep dive into the given Level 2 core step and break it down into Atomic State Transitions (Level 3).

CRITICAL GRANULARITY RULES FOR LEVEL 3:
1. [Physics over Recipe]: You are NO LONGER writing recipe instructions. Your focus MUST shift entirely to the physical, visual changes of the objects.
2. [State Transition Focus]: Only extract moments where a target object undergoes a VISUAL, IRREVERSIBLE change (e.g., deformation, separation, merging, transfer, material state change).
3. [Semantic Chunk, Not Instant]: A Level 3 chunk should be the SMALLEST COMPLETE visual event, not the narrowest contact instant. Include the short lead-in where the state change clearly begins and the short tail where the new state becomes stable.
4. [Boundary Rule]: The start_time should be when the object starts entering the state change, not just the exact collision/contact frame. The end_time should be when the transfer/transformation is completed and the new visual state is clearly established.
5. [Typical Duration]: Most Level 3 chunks should last about 2 to 6 seconds when visible at this sampling rate. Avoid 1-frame or 1-second spans unless the change is truly instantaneous and still visually complete.
6. [Continuity Rule]: If several consecutive frames depict one uninterrupted micro-process with the same intent and target state change, merge them into one chunk rather than splitting too narrowly.
7. [Ignore Empty Motions]: Ignore purely human limb movements (like reaching for a tool or moving a hand) if the object's state doesn't change.
8. [Food-Relevance Filter]: Ignore narration, waiting, presentation, tasting, and any motion unrelated to a visible state change in ingredients, cookware contents, or the dish.
9. [Example]: If minced garlic is poured into a pan across several frames, annotate the whole transfer-and-settle process (e.g., "add garlic to pan"), not only the single frame where garlic first touches the pan.

For each atomic state transition chunk, provide:
- chunk_id: Sequential ID.
- parent_step_id: The step_id of the core step this chunk belongs to.
- start_time / end_time: The specific timestamps covering the full state-change chunk (typically about 2-6 seconds, but can be shorter if the change is truly instantaneous and visually complete).
- sub_action: A brief description of the specific interaction.
- pre_state: The EXPLICIT visual state of the target object BEFORE the interaction (e.g., "A flat, empty wonton wrapper").
- post_state: The EXPLICIT visual state of the target object AFTER the interaction (e.g., "A dollop of meat filling is now resting in the center of the wrapper").

Output JSON format example:
{
  "key_state_chunks": [
    {
      "chunk_id": 1,
      "parent_step_id": 1,
      "start_time": "01:42",
      "end_time": "01:43",
      "sub_action": "Deposit filling",
      "pre_state": "A flat, empty wonton wrapper",
      "post_state": "A dollop of meat filling is resting on the wrapper"
    }
  ]
}
"""


def get_level3_prompt(
    clip_duration_sec: float,
    level1_result: dict,
    level2_result: dict,
) -> str:
    """
    Build the Level 3 (Atomic Step) user-turn prompt.

    Raises:
        NotImplementedError: Until the Level 3 prompt is filled in.
    """
    import json as _json
    if _LEVEL3_BASE.startswith("TODO"):
        raise NotImplementedError(
            "Level 3 prompt is not yet defined. "
            "Edit proxy_data/youcook2_seg_annotation/prompts.py and replace _LEVEL3_BASE."
        )
    l1_json = _json.dumps(level1_result, ensure_ascii=False, indent=2)
    l2_json = _json.dumps(level2_result, ensure_ascii=False, indent=2)
    return (
        f"The video clip is {clip_duration_sec:.0f} seconds long.\n\n"
        f"Level 1 macro phases:\n{l1_json}\n\n"
        f"Level 2 activities:\n{l2_json}\n\n"
        + _LEVEL3_BASE
    )
