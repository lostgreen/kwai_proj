#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt templates for Event Logic proxy tasks (add / replace / sort / t2v variants).

Design philosophy:
  - All task prompts include CoT (<think>/<answer>) instructions baked in.
  - V→T tasks: video context clips → text options (add, replace, sort).
  - T→V tasks: text context descriptions → video option clips (add_t2v, replace_t2v).
  - Step caption prompts: used by annotate_l2_step_captions.py to generate
    recipe-instruction-style descriptions for each L2 event clip.

Usage:
    from prompts import get_add_prompt, get_replace_prompt, get_sort_prompt
    from prompts import get_add_t2v_prompt, get_replace_t2v_prompt
    from prompts import STEP_CAPTION_SYSTEM_PROMPT, get_step_caption_prompt
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

def get_add_prompt_generic(num_ctx: int, options: list[str], cot: bool = False) -> str:
    """
    Domain-generic Predict-Next prompt (no cooking references).

    Args:
        num_ctx: Number of context video clips.
        options: List of candidate text descriptions (including the correct one).
        cot:     If True, include <think>...</think> CoT instruction.
                 If False (default), ask for direct <answer> only.

    Returns:
        User-turn prompt string with <video> placeholders.
    """
    labels = _option_labels(len(options))
    lines = ["Context Video Sequence:"]
    for i in range(num_ctx):
        lines.append(f"{i + 1}. <video>")

    lines += [
        "",
        "Based on the continuous actions shown in the Context Video Sequence above, "
        "which of the following textual options shows the most logical next step?",
        "Options:",
    ]
    for label, opt in zip(labels, options):
        lines.append(f"{label}. {opt}")

    lines.append("")
    if cot:
        lines += [
            "First, carefully observe the actions and visual content in each Context Video "
            "to understand the progression. Then, reason about which text option best continues the sequence.",
            "",
            f"Think step by step inside <think> </think> tags, then provide your final answer "
            f"(a single letter from {', '.join(labels)}) inside <answer> </answer> tags.",
        ]
    else:
        lines.append(
            f"Provide your answer (a single letter from {', '.join(labels)}) inside <answer> </answer> tags."
        )
    return "\n".join(lines)


def get_replace_prompt_generic(total_steps: int, missing_pos: int, options: list[str], cot: bool = False) -> str:
    """
    Domain-generic Fill-in-the-Blank prompt (no cooking references).

    Args:
        total_steps: Total number of steps in the sequence (including the missing one).
        missing_pos: Zero-based index of the missing step.
        options: List of candidate text descriptions (including the correct one).
        cot:     If True, include <think>...</think> CoT instruction.
                 If False (default), ask for direct <answer> only.

    Returns:
        User-turn prompt string with <video> placeholders.
    """
    labels = _option_labels(len(options))
    lines = [
        "Watch the following process carefully. The sequence has a [MISSING] step.",
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

    lines.append("")
    if cot:
        lines += [
            "First, carefully observe the Context Sequence to understand the flow "
            "before and after the [MISSING] step. Then, reason about which text option best fills the gap.",
            "",
            f"Think step by step inside <think> </think> tags, then provide your final answer "
            f"(a single letter from {', '.join(labels)}) inside <answer> </answer> tags.",
        ]
    else:
        lines.append(
            f"Provide your answer (a single letter from {', '.join(labels)}) inside <answer> </answer> tags."
        )
    return "\n".join(lines)


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
# Sort (Generic) — Domain-agnostic variant for cross-domain hier seg data
# Same structure as get_sort_prompt but without cooking-specific language.
# ─────────────────────────────────────────────────────────────────────────────

def get_sort_prompt_generic(num_clips: int, cot: bool = False) -> str:
    """
    Build a domain-generic Sort task prompt (no cooking references).

    Args:
        num_clips: Number of video clips in the shuffled sequence.
        cot:       If True, include <think>...</think> CoT instruction.
                   If False (default), ask for direct <answer> only.

    Returns:
        User-turn prompt string with <video> placeholders.
    """
    lines = [
        f"The following {num_clips} video clips show steps from a continuous process, "
        "but they are presented in a shuffled order.",
        "Video Clips (shuffled order):",
    ]
    for i in range(num_clips):
        lines.append(f"Clip {i + 1}: <video>")

    lines += [
        "",
        "Determine the correct chronological order of these clips to reconstruct the original sequence.",
        "",
    ]
    if cot:
        lines += [
            "First, carefully observe the actions, object states, and scene changes in each clip. "
            "Reason about which step comes first, which transformation follows, and which final state is last.",
            "",
            "Think step by step inside <think> </think> tags, then provide your final answer "
            "as a sequence of clip numbers with no spaces or separators "
            f"(e.g., {''.join(str(i) for i in range(num_clips, 0, -1))}) inside <answer> </answer> tags.",
        ]
    else:
        lines.append(
            "Provide your answer as a sequence of clip numbers with no spaces or separators "
            f"(e.g., {''.join(str(i) for i in range(num_clips, 0, -1))}) inside <answer> </answer> tags."
        )
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


# ─────────────────────────────────────────────────────────────────────────────
# T→V Add — Predict next step video from text context
# Input:  N consecutive step descriptions (text), 4 video clip options (<video>)
# Output: single letter A/B/C/D
# ─────────────────────────────────────────────────────────────────────────────

def get_add_t2v_prompt(context_steps: list[str], num_options: int) -> str:
    """
    Build the T→V Add task prompt.

    Args:
        context_steps: List of recipe step descriptions (text context).
        num_options: Number of video option clips (typically 4).

    Returns:
        User-turn prompt string with <video> placeholders for options and CoT instructions.
    """
    labels = _option_labels(num_options)
    lines = ["Context Sequence (recipe steps):"]
    for i, step in enumerate(context_steps):
        lines.append(f"Step {i + 1}: {step}")

    lines += [
        "",
        "Based on the cooking progression described above, "
        "which of the following video clips shows the most logical next cooking step?",
        "Options:",
    ]
    for label in labels:
        lines.append(f"{label}. <video>")

    lines += [
        "",
        "First, carefully read the Context Sequence to understand what has been done so far. "
        "Then, reason about which video clip best shows the logical next step in the cooking process.",
        "",
        f"Think step by step inside <think> </think> tags, then provide your final answer "
        f"(a single letter from {', '.join(labels)}) inside <answer> </answer> tags.",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# T→V Replace — Fill in missing step with a video clip
# Input:  N step descriptions with one [MISSING], 4 video clip options (<video>)
# Output: single letter A/B/C/D
# ─────────────────────────────────────────────────────────────────────────────

def get_replace_t2v_prompt(all_steps: list[str | None], missing_pos: int, num_options: int) -> str:
    """
    Build the T→V Replace task prompt.

    Args:
        all_steps: List of step descriptions; all_steps[missing_pos] should be None
                   (it will be rendered as [MISSING]).
        missing_pos: Zero-based index of the missing step.
        num_options: Number of video option clips (typically 4).

    Returns:
        User-turn prompt string with <video> placeholders for options and CoT instructions.
    """
    labels = _option_labels(num_options)
    lines = [
        "The following cooking sequence has a [MISSING] step.",
        "Context Sequence (recipe steps):",
    ]
    for i, step in enumerate(all_steps):
        if i == missing_pos or step is None:
            lines.append(f"Step {i + 1}: [MISSING]")
        else:
            lines.append(f"Step {i + 1}: {step}")

    lines += [
        "",
        "Based on the recipe steps before and after the [MISSING] step, "
        "which video clip correctly shows the missing cooking action?",
        "Options:",
    ]
    for label in labels:
        lines.append(f"{label}. <video>")

    lines += [
        "",
        "First, carefully read the surrounding steps to understand what state the dish is in "
        "before and after the [MISSING] step. Then, identify which video clip fills the gap logically.",
        "",
        f"Think step by step inside <think> </think> tags, then provide your final answer "
        f"(a single letter from {', '.join(labels)}) inside <answer> </answer> tags.",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# T→V Causality Filter — Verify uniqueness and sufficiency for T→V tasks
# Input: text context steps + keyframes of 4 option video clips
# Output: {"causal_valid": bool, "reason": str, "confidence": float}
# ─────────────────────────────────────────────────────────────────────────────

CAUSALITY_T2V_SYSTEM_PROMPT = (
    "You are an expert cooking video analysis assistant specializing in causal event chains. "
    "Your task is to verify if a text-context video-option Event Logic question has a clear, "
    "unambiguous, and uniquely correct answer. Respond only in JSON."
)

# Template fields: {task_type}, {text_context_str}, {correct_option}
CAUSALITY_T2V_USER_PROMPT = """\
You are given a cooking process described in text, followed by keyframes from 4 candidate video clips.
The task is to {task_type}.

Text Context (in order):
{text_context_str}

Candidate video clips are shown as keyframe images (Options A, B, C, D follow).
Correct answer: Option {correct_option}

Evaluate:
1. Does the text context clearly establish a causal cooking progression with sufficient detail?
2. Is there EXACTLY ONE video option that logically continues or fills the described sequence?
3. Could the correct answer be identified from the text context alone without ambiguity?

Pay attention to:
- Whether the described steps have clear cause-and-effect relationships.
- Whether the correct video option is visually distinct from the distractors.
- Whether the text context narrows down the next step sufficiently.

Respond strictly in JSON format:
{{"causal_valid": <true if unique correct video answer with clear text context, false otherwise>, "reason": "<brief explanation in 1-2 sentences>", "confidence": <float between 0.0 and 1.0>}}"""


# ─────────────────────────────────────────────────────────────────────────────
# Step Caption — Recipe-instruction-style description for L2 event clips
# Used by annotate_l2_step_captions.py to generate T→V text context data
# Input: keyframes of one event clip
# Output: {"caption": str, "confidence": float}
# ─────────────────────────────────────────────────────────────────────────────

STEP_CAPTION_SYSTEM_PROMPT = (
    "You are a culinary expert who writes precise, concise recipe instructions. "
    "Given video frames from a single cooking step, write a clear recipe instruction "
    "that describes this step. Respond only in JSON."
)


def get_step_caption_prompt(instruction_hint: str = "") -> str:
    """
    Build the step caption annotation prompt.

    Args:
        instruction_hint: Optional original L2 annotation text to guide the VLM.
                          Used as a hint only; the VLM should primarily rely on the video.

    Returns:
        User-turn prompt string with instructions for generating a step caption.
    """
    hint_line = (
        f"\nFor reference, the original annotation for this step is: \"{instruction_hint}\"\n"
        "You may use this as a guide, but your description should be based on what you see in the video."
        if instruction_hint else ""
    )
    return f"""\
Watch the following video frames from a single cooking step carefully.{hint_line}

Write a concise recipe instruction (1-2 sentences) that describes this cooking step. Your description must:
1. State the main cooking action being performed (e.g., "Sauté", "Whisk", "Fold", "Simmer")
2. Name the key ingredients or tools involved
3. Describe what happens to the food during this step (texture, color, or state changes if visible)
4. Be written in imperative style, like a recipe instruction

Good examples:
- "Add the diced onions to the heated pan and sauté over medium heat until they become translucent."
- "Whisk the eggs and milk together in a bowl until a smooth, uniform mixture forms."
- "Fold the dough over twice and press firmly to seal the edges before cutting into portions."

Respond strictly in JSON format:
{{"caption": "<your 1-2 sentence recipe instruction>", "confidence": <float 0.0-1.0 reflecting how clearly this step is visible in the frames>}}"""
