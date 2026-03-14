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
_LEVEL2_BASE = """Identify the core, instructional steps that drive the task progression in the video. You need to extract specific, continuous action segments with clear start and end boundaries.

For each step, provide:
- step_id: Sequential ID.
- parent_phase_id: The phase_id of the macro phase this step belongs to.
- start_time / end_time: Precise boundary timestamps.
- instruction: A description of the step using an imperative or verb-object structure (e.g., "Slice the onions into thin strips").
- visual_keywords: 3 to 5 key visual elements or action cues present in the segment (output as an array of strings, e.g., ["knife", "onion", "cutting board", "slicing"]).

Remember, temporal gaps between consecutive steps are allowed.

Output JSON format example:
{
  "meso_steps": [
    {
      "step_id": 1,
      "parent_phase_id": 1,
      "start_time": "00:20",
      "end_time": "00:45",
      "instruction": "Slice the onion and set aside",
      "visual_keywords": ["knife", "onion", "chopping"]
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
_LEVEL3_BASE = """Now, deep dive into the core steps (Level 2) and break them down into semantically complete sub-steps or key state chunks (typically lasting 5-15 seconds).
Ignore micro body movements. Focus strictly on continuous operation units that result in a substantial, visually contrastive change in the state of the core target object.

For each key state chunk, provide:
- chunk_id: Sequential ID.
- parent_step_id: The step_id of the core step this chunk belongs to.
- start_time / end_time: Timestamps for this operation unit.
- sub_action: A brief description of the sub-step (e.g., "Continuously chop the onion and push it into the bowl").
- pre_state: The explicit visual state of the target object BEFORE the operation begins (e.g., "Halved onions resting on the cutting board").
- post_state: The explicit visual state of the target object AFTER the operation concludes (e.g., "Onions are fully minced and transferred into a bowl").

Output JSON format example:
{
  "key_state_chunks": [
    {
      "chunk_id": 1,
      "parent_step_id": 1,
      "start_time": "00:22",
      "end_time": "00:35",
      "sub_action": "Mince onion and transfer",
      "pre_state": "Halved onions resting on the cutting board",
      "post_state": "Onions are fully minced and transferred into a bowl"
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
