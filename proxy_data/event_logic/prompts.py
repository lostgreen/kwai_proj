#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt templates for Event Logic proxy tasks (add / replace / sort).

Design philosophy:
  - All prompts include CoT (<think>/<answer>) instructions baked in,
    consistent with the format used by temporal_aot and youcook2_seg_annotation.
  - add:     given N context video clips, select the correct next-step description.
  - replace: given a sequence with one [MISSING] step, select the correct filler description.
  - sort:    given N shuffled video clips, output the correct chronological order.

Usage:
    from prompts import get_add_prompt, get_replace_prompt, get_sort_prompt
"""

_LETTERS = [chr(ord("A") + i) for i in range(26)]


def _option_labels(n: int) -> list[str]:
    if n < 2 or n > len(_LETTERS):
        raise ValueError(f"Option count must be between 2 and {len(_LETTERS)}, got {n}")
    return _LETTERS[:n]


# ─────────────────────────────────────────────────────────────────────────────
# Add — Predict the next cooking step
# Input:  N consecutive event clips (context), 4 text options
# Output: single letter A/B/C/D
# ─────────────────────────────────────────────────────────────────────────────

def get_add_prompt(num_ctx: int, options: list[str]) -> str:
    """
    Build the Add task prompt.

    Args:
        num_ctx: Number of context video clips.
        options: List of candidate text descriptions (including the correct one).

    Returns:
        User-turn prompt string with <video> placeholders and CoT instructions.
    """
    labels = _option_labels(len(options))
    lines = ["Context Video Sequence:"]
    for i in range(num_ctx):
        lines.append(f"{i + 1}. <video>")

    lines += [
        "",
        "Based on the continuous actions shown in the Context Video Sequence above, "
        "which of the following textual options shows the most logical continuous next cooking step?",
        "Options:",
    ]
    for label, opt in zip(labels, options):
        lines.append(f"{label}. {opt}")

    lines += [
        "",
        "First, carefully observe the actions and visual content in each Context Video "
        "to understand the cooking progression. Then, reason about which text option best continues the sequence.",
        "",
        f"Think step by step inside <think> </think> tags, then provide your final answer "
        f"(a single letter from {', '.join(labels)}) inside <answer> </answer> tags.",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Replace — Fill in the missing step
# Input:  N event clips with one position replaced by [MISSING], 4 text options
# Output: single letter A/B/C/D
# ─────────────────────────────────────────────────────────────────────────────

def get_replace_prompt(total_steps: int, missing_pos: int, options: list[str]) -> str:
    """
    Build the Replace task prompt.

    Args:
        total_steps: Total number of steps in the sequence (including the missing one).
        missing_pos: Zero-based index of the missing step.
        options: List of candidate text descriptions (including the correct one).

    Returns:
        User-turn prompt string with <video> placeholders and CoT instructions.
    """
    labels = _option_labels(len(options))
    lines = [
        "Watch the following cooking process carefully. The sequence has a [MISSING] step.",
        "Context Sequence:",
    ]
    for i in range(total_steps):
        if i == missing_pos:
            lines.append(f"Step {i + 1}: [MISSING]")
        else:
            lines.append(f"Step {i + 1}: <video>")

    lines += [
        "",
        "Based on the chronological visual content of the sequence, "
        "pick the correct textual option to fill in the [MISSING] step.",
        "Options:",
    ]
    for label, opt in zip(labels, options):
        lines.append(f"{label}. {opt}")

    lines += [
        "",
        "First, carefully observe the Context Sequence to understand the cooking flow "
        "before and after the [MISSING] step. Then, reason about which text option best fills the gap.",
        "",
        f"Think step by step inside <think> </think> tags, then provide your final answer "
        f"(a single letter from {', '.join(labels)}) inside <answer> </answer> tags.",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Sort — Reorder shuffled cooking clips
# Input:  N event clips presented in shuffled order
# Output: digit sequence like "31245" (correct temporal order of clip indices)
# ─────────────────────────────────────────────────────────────────────────────

def get_sort_prompt(num_clips: int) -> str:
    """
    Build the Sort task prompt.

    The model is shown N clips in a shuffled order and must output a digit
    sequence representing the correct chronological ordering. For example,
    "312" means the correct sequence is: Clip3 → Clip1 → Clip2.

    Answer encoding: answer[i] = 1-based clip number that belongs at position i
    in the chronologically sorted sequence.

    Args:
        num_clips: Number of video clips in the shuffled sequence.

    Returns:
        User-turn prompt string with <video> placeholders and CoT instructions.
    """
    lines = [
        f"The following {num_clips} video clips show steps from a cooking process, "
        "but they are presented in a shuffled order.",
        "Video Clips (shuffled order):",
    ]
    for i in range(num_clips):
        lines.append(f"Clip {i + 1}: <video>")

    lines += [
        "",
        "Determine the correct chronological order of these clips to reconstruct the original cooking sequence.",
        "",
        "First, carefully observe the cooking actions and food states in each clip. "
        "Reason about which preparation step comes first, which transformation follows, and which final state is last.",
        "",
        "Think step by step inside <think> </think> tags, then provide your final answer "
        "as a sequence of clip numbers with no spaces or separators "
        f"(e.g., {''.join(str(i) for i in range(num_clips, 0, -1))}) inside <answer> </answer> tags.",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# AI Causality Filter — Verify uniqueness and sufficiency of causal context
# Used in build_l2_event_logic.py
# Input: keyframes of context events + shuffled text options
# Output: {"causal_valid": bool, "reason": str, "confidence": float}
# ─────────────────────────────────────────────────────────────────────────────

CAUSALITY_SYSTEM_PROMPT = (
    "You are an expert cooking video analysis assistant specializing in causal event chains. "
    "Your task is to verify if a video-based Event Logic question has a clear, unambiguous, "
    "and uniquely correct answer. Respond only in JSON."
)

# Template fields: {task_type} and {options_str}
CAUSALITY_USER_PROMPT = """\
You are given keyframes sampled from cooking video events (shown as images). \
The task is to {task_type}.

The following text options are provided:
{options_str}

Evaluate:
1. Do the context event frames provide ENOUGH visual information to distinguish between the options?
2. Is there EXACTLY ONE option that logically and visually follows/fills from the context?
3. Could a different option also be plausible given only the context shown?

Pay attention to:
- Whether the dish/ingredient state is clearly shown in context.
- Whether the cooking progression is visually unambiguous.
- Whether the target action is the ONLY logical continuation or filling.

Respond strictly in JSON format:
{{"causal_valid": <true if unique correct answer exists with sufficient context, false otherwise>, "reason": "<brief explanation in 1-2 sentences>", "confidence": <float between 0.0 and 1.0>}}"""
