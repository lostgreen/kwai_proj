#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
English prompt templates for the LLM "Task Architect" that designs
event-logic MCQ / sort questions from an annotation's action script.

Three task types:
  1. Predict Next  — given consecutive context, predict the next step
  2. Fill-in-the-Blank — given before/after context, identify the missing step
  3. Sequence Sort — select steps with irreversible ordering for reorder task

The LLM receives only text (the action script); no video frames are sent.
"""

# ─────────────────────────────────────────────────────────────────────────────
# System prompt (shared across all three tasks)
# ─────────────────────────────────────────────────────────────────────────────

TASK_ARCHITECT_SYSTEM_PROMPT = """\
You are a professional Task Architect for multimodal video understanding \
training data.

Given an "Action Script" that describes a video's hierarchical event \
structure (Phases → Events → Actions), you design high-quality \
multiple-choice or sequence-ordering questions that test temporal \
reasoning, causal logic, and procedural understanding.

Rules you MUST follow:
1. **Granularity Adaptation** — Choose EITHER Event level (IDs like \
"Event 3") OR Action level (IDs like "Action 2.1") for each question. \
Pick whichever produces a more meaningful, challenging question. \
NEVER mix Event and Action IDs within the same question.
2. **Distractor Quality** — Every distractor must be plausible in the \
general context of the video but logically incorrect due to violations \
of physical laws, temporal dependencies, or procedural common sense.
3. **ID Format** — Reference items by their EXACT IDs from the script \
(e.g., "Event 2", "Action 3.1"). Do not invent IDs that are not present.
4. **Output** — Respond ONLY in valid JSON. No markdown fences, no \
explanation outside the JSON object.\
"""


# ─────────────────────────────────────────────────────────────────────────────
# Task 1: Predict Next
# ─────────────────────────────────────────────────────────────────────────────

def get_predict_next_user_prompt(script_text: str) -> str:
    """Build user prompt for the Predict-Next task architect call."""
    return f"""\
{script_text}

─── Task: Predict Next ───

Design a "predict the next step" single-choice question from the action \
script above.

Instructions:
1. **Granularity**: Decide whether Event-level or Action-level produces \
a more logically challenging question. Stick to ONE level throughout.
2. Select 2-4 CONSECUTIVE items as the known context.
3. The item that IMMEDIATELY follows the context must be the unique, \
logically necessary next step (the correct answer).
4. Write exactly 3 DISTRACTORS — plausible step descriptions that fit \
the video's general topic but contain a fatal flaw in physical logic, \
temporal ordering, or procedural common sense.
5. Provide the correct answer as a text description (copy or closely \
paraphrase the script entry).

If the script lacks a clear sequential chain (too few items or purely \
repetitive / interchangeable steps), set "suitable" to false.

Respond in this exact JSON format:
{{
  "suitable": true,
  "granularity": "Event",
  "reasoning": "Brief explanation of the causal chain and why distractors fail",
  "context_ids": ["Event 1", "Event 2"],
  "correct_next_id": "Event 3",
  "correct_next_text": "Description of the correct next step",
  "distractors": [
    "Distractor 1: seems plausible but violates the physical state established earlier",
    "Distractor 2: a common action in this domain but premature at this point",
    "Distractor 3: involves an abrupt, illogical change of subject or object"
  ]
}}"""


# ─────────────────────────────────────────────────────────────────────────────
# Task 2: Fill-in-the-Blank
# ─────────────────────────────────────────────────────────────────────────────

def get_fill_blank_user_prompt(script_text: str) -> str:
    """Build user prompt for the Fill-in-the-Blank task architect call."""
    return f"""\
{script_text}

─── Task: Fill in the Blank ───

Design a "fill in the missing step" single-choice question from the \
action script above.

Instructions:
1. **Granularity**: Decide whether Event-level or Action-level yields a \
more meaningful "before → core change → after" pattern. Stick to ONE level.
2. Find a chain of 3+ consecutive items where the MIDDLE item represents \
a critical state change or transition (the "bridge" between cause and effect).
3. The BEFORE item(s) and AFTER item(s) will be shown as video context; \
the middle item is REMOVED — the model must identify it.
4. Write exactly 3 DISTRACTORS — plausible step descriptions that CANNOT \
correctly bridge the before and after states. Each should fail for a \
distinct reason:
   - One that connects to the BEFORE but cannot lead to the AFTER.
   - One that connects to the AFTER but contradicts the BEFORE state.
   - One that is topically related but physically / procedurally impossible.
5. Provide the correct answer as a text description.

If no clear before→change→after chain exists, set "suitable" to false.

Respond in this exact JSON format:
{{
  "suitable": true,
  "granularity": "Action",
  "reasoning": "Why the removed step is the critical bridge and why distractors fail",
  "before_ids": ["Action 3.1"],
  "missing_id": "Action 3.2",
  "after_ids": ["Action 3.3", "Action 3.4"],
  "correct_text": "Description of the missing step",
  "distractors": [
    "Distractor 1: bridges to 'before' but cannot cause the 'after' state",
    "Distractor 2: could cause 'after' but contradicts the 'before' state",
    "Distractor 3: topically plausible but physically impossible here"
  ]
}}"""


# ─────────────────────────────────────────────────────────────────────────────
# Task 3: Sequence Sorting
# ─────────────────────────────────────────────────────────────────────────────

def get_sequence_sort_user_prompt(script_text: str) -> str:
    """Build user prompt for the Sequence Sorting task architect call."""
    return f"""\
{script_text}

─── Task: Sequence Sorting ───

Design a "reorder shuffled steps" question from the action script above.

Instructions:
1. **Granularity**: Decide whether a macro Event sequence (e.g., \
wash → cut → stir-fry) or a micro Action sequence (e.g., pick up knife \
→ cut first slice → set knife down) has a more ABSOLUTELY IRREVERSIBLE \
ordering. Stick to ONE level.
2. Select 3 to 5 items whose chronological order is dictated by strict \
causal or physical prerequisites (e.g., "you must open the lid before \
pouring the contents").
3. **Exclusion criteria**: If any pair of selected items could be \
swapped without logical contradiction (e.g., interchangeable repetitive \
motions, decorating left vs. right), do NOT include them.
4. Return the selected item IDs in their TRUE chronological order.

If the script contains no sequence of 3+ items with absolute, \
irreversible ordering, set "suitable" to false.

Respond in this exact JSON format:
{{
  "suitable": true,
  "granularity": "Event",
  "reasoning": "Why this ordering is strictly irreversible (physical laws, causal chain)",
  "ordered_ids": ["Event 1", "Event 2", "Event 3", "Event 4"]
}}"""
