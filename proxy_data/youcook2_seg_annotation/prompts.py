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
2. [Precise Boundaries]: The 'start_time' must capture the exact moment an action or \
intention begins, and the 'end_time' must mark the exact moment the action concludes or \
the visual state solidifies.
3. [Formatting]: Output strictly in valid JSON format."""


# ─────────────────────────────────────────────────────────────────────────────
# Level 1: Macro Phase  (阶段级)
# Input:  1fps frame strip of the full windowed clip
# Output: JSON with macro_phases list
# ─────────────────────────────────────────────────────────────────────────────
_LEVEL1_BASE = """\
Analyze the instructional video and segment it into high-level macro phases \
(typically 3 to 5 phases). These phases should represent the global narrative steps \
required to complete the ultimate task.

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
_LEVEL2_BASE = """Identify the core logical events (Meso Steps) that drive the task progression within this macro phase.

CRITICAL GRANULARITY RULES FOR LEVEL 2:
1. [Aggregation Rule]: A Level 2 step MUST be a complete "Logical Event" or "Workflow", NOT an isolated atomic action. 
2. [Anti-Fragmentation]: DO NOT extract single, momentary physical actions (e.g., "pick up a spoon", "place filling", "fold a corner"). These are Level 3 details.
3. [Good vs. Bad Example]: 
   - BAD (Too granular): "Place a spoonful of filling onto the wrapper" (1 second).
   - GOOD (Correct Level 2): "Assemble the wonton by filling and folding the wrapper" (15-20 seconds).

For each grouped step, provide:
- step_id: Sequential ID.
- parent_phase_id: The phase_id of the macro phase this step belongs to.
- start_time / end_time: Precise boundary timestamps covering the ENTIRE logical event.
- instruction: A high-level description of the completed workflow (e.g., "Assemble the dumplings").
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
2. [State Transition Focus]: Only extract moments where a target object undergoes a VISUAL, IRREVERSIBLE change (e.g., deformation, separation, merging, material state change).
3. [Ignore Empty Motions]: Ignore purely human limb movements (like reaching for a tool or moving a hand) if the object's state doesn't change.

For each atomic state transition chunk, provide:
- chunk_id: Sequential ID.
- parent_step_id: The step_id of the core step this chunk belongs to.
- start_time / end_time: The specific timestamps where the state transition occurs (can be just 1-3 seconds).
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
