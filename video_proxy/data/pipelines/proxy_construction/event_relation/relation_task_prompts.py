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

2. **Skip Non-Actions** — Do NOT select items that describe non-actions \
such as "talks to the camera", "gestures", "holds X and explains", or \
any talking / pausing shot. The correct answer and all context items \
MUST describe purposeful physical actions that change the state of \
objects or the environment.

3. **Anti-Shortcut Distractor Design** — This is the MOST CRITICAL rule. \
All distractors and the correct answer MUST share the same core \
object(s) or entity(ies) that appear in the context. Distractors must \
ONLY differ from the correct answer in:
   - **Action intent** (e.g., "slice tomato" vs "crush tomato")
   - **Physical result** (e.g., "boil until soft" vs "boil until dry")
   - **Temporal direction** (e.g., "add salt before frying" vs "add salt after frying")
   - **Tool / manner** (e.g., "stir with spatula" vs "stir with whisk")
   NEVER create a distractor that introduces a completely different \
object or topic (e.g., correct="chop tomato" → distractor="open oven"). \
If you cannot produce 3 same-object distractors, set "suitable" to false.

4. **Temporal Lock for Action Granularity** — When you choose Action \
level (IDs like "Action X.Y"), distractors MUST represent micro-actions \
that could theoretically happen in the EXACT SAME short timeframe, \
scene, and physical setup as the correct answer. \
DO NOT copy macro-events from a completely different phase or much \
later in the script (e.g., "bake in oven" or "plate and garnish" when \
the current scene is mixing ingredients in a bowl). \
Instead, alter the TOOL used, the DIRECTION of the movement, the \
INTENSITY of the action, or the specific INTERACTION with the current \
ingredients to create confusion. \
Example — if the correct Action is "fold the batter gently with a \
spatula":
  - GOOD distractor: "vigorously whisk the batter until foamy" \
(same object+tool, wrong technique)
  - GOOD distractor: "scrape the batter out of the bowl onto the \
counter" (same object+tool, wrong destination)
  - BAD distractor: "decorate the cake with strawberries" \
(different phase entirely — violates Temporal Lock)

5. **Uniqueness Self-Check** — Before finalizing, verify:
   (a) Given ONLY the before/after context, is the correct answer the \
SOLE logically necessary step? If multiple options could be valid, \
re-design the question or set "suitable" to false.
   (b) Can a distractor be eliminated purely by keyword matching (e.g., \
different object name) or by recognizing it belongs to a completely \
different stage of the process? If yes, rewrite it.

6. **ID Format** — Reference items by their EXACT IDs from the script \
(e.g., "Event 2", "Action 3.1"). Do not invent IDs that are not present.

7. **Causal Dependency (CRITICAL)** — Every consecutive pair of items \
in any selected sequence MUST have a genuine cause-effect or \
state-transformation relationship: the earlier step PRODUCES a physical \
state that the later step CONSUMES or DEPENDS ON (e.g., "crack egg into \
bowl" → "whisk egg" — the egg must be in the bowl before it can be \
whisked). \
The following do NOT qualify as causal chains and MUST be rejected:
   - **Multi-angle / replay shots**: the same action filmed from a \
different camera angle, slow-motion replay, or close-up insert. These \
show the SAME state, not a state change.
   - **Parallel / interchangeable actions**: actions that could happen \
in any order with identical outcome (e.g., "sprinkle salt" and \
"sprinkle pepper" — swapping them changes nothing).
   - **Decorative / cosmetic variations**: steps that differ only in \
position or appearance but share no physical prerequisite (e.g., \
"place cherry on left" vs. "place cherry on right").
If the script contains no sequence of 3+ items with strict causal \
ordering, set "suitable" to false rather than force-fitting.

8. **Anti-Flashback / Intro Rule** — REJECT sequences that represent \
a video "intro", "teaser", or "flashback" where a finished product or \
end-state is shown BEFORE the steps to produce it. The selected sequence \
MUST represent forward-moving chronological physical time. If the script \
starts with a preview of the final result (e.g., a completed dish, a \
finished craft) followed by the actual process, do NOT include the \
preview item in any context or answer — begin the sequence only from \
the first real construction step.

9. **Anti-Slideshow Rule** — REJECT sequences that merely cut between \
static images, logos, title cards, text overlays, or separate independent \
demonstrations (e.g., a slideshow of different optical illusions, a \
montage of unrelated product shots). There MUST be a continuous physical \
entity undergoing progressive change across the selected items. If the \
script describes a compilation of disconnected static scenes with no \
shared evolving object, set "suitable" to false.

10. **Strict Visual Object Lock for Distractors** — When writing \
distractors, do NOT invent distinct colors, materials, or objects that \
are absent from the correct answer and context (e.g., do NOT write \
"blue liquid" if the context only mentions "red liquid", do NOT write \
"white powder" if no powder appears). Distractors must be visually \
confusable with the correct answer — they should describe plausible \
alternative actions on the EXACT SAME visible objects with the SAME \
visual attributes. If you cannot create 3 visually confusable \
distractors, set "suitable" to false.

11. **Output** — Respond ONLY in valid JSON. No markdown fences, no \
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

Design "predict the next step" single-choice questions from the action \
script above. Return ALL high-quality questions the script supports — \
do not artificially limit yourself to one.

Instructions:
1. **Granularity**: For each question, decide whether Event-level or \
Action-level produces a more logically challenging question. Different \
questions may use different granularities. Stick to ONE level within \
each question.
2. Select 2-4 STRICTLY CONSECUTIVE items as the known context (NO GAPS \
— e.g., Action 2.1 → 2.2 → 2.3, not 2.1 → 2.3). All context items \
must be physical actions — if a non-action (talking shot) sits between \
two good items, choose a different window that has no gaps.
   **Causal Chain Requirement**: Every consecutive pair in the context \
MUST have a genuine cause-effect link — the earlier step produces a \
physical state that the later step consumes or depends on. \
REJECT sequences where consecutive items are merely different camera \
angles of the same action, slow-motion replays, or parallel / \
interchangeable actions (e.g., "sprinkle salt" → "sprinkle pepper" — \
swapping changes nothing). If no causal chain of 2+ items exists, set \
"suitable" to false for that question.
3. The item that IMMEDIATELY follows the context must be the unique, \
logically necessary next step (the correct answer). It must also be a \
purposeful physical action, NOT a talking or gesturing shot. The \
correct answer must DEPEND on the state produced by the last context \
step — it should be impossible (or physically nonsensical) to perform \
the correct answer without the context having happened first.
4. Write exactly 3 DISTRACTORS following BOTH the Anti-Shortcut rule \
AND the Temporal Lock rule:
   - All 3 distractors MUST involve the SAME core object(s) as the \
correct answer (e.g., if correct="fold the batter with a spatula", \
all distractors must also involve "batter" and/or "spatula").
   - **If Action granularity**: distractors MUST be micro-actions that \
could happen in the EXACT SAME short scene and physical setup. \
DO NOT grab macro-events from later phases (e.g., "bake in oven", \
"plate and garnish") when the current scene is a mixing bowl. \
Instead, alter the TOOL, DIRECTION, INTENSITY, or INTERACTION:
     (a) Same object + tool, but wrong TECHNIQUE (e.g., "vigorously \
whisk" instead of "gently fold").
     (b) Same object + tool, but wrong DIRECTION or DESTINATION \
(e.g., "scrape batter onto counter" instead of "fold batter in bowl").
     (c) Same object, but an action that ALREADY HAPPENED in the \
context (temporal duplicate — e.g., "add flour again" when flour \
was added two steps ago).
   - **If Event granularity**: distractors should be events from the \
same domain but with a wrong causal position:
     (a) Premature — skips a required intermediate event.
     (b) Wrong direction — reverses a physical state change.
     (c) Already completed — repeats an event from the context.
5. **Causal Predictability Rule (CRITICAL)**: The test model can ONLY \
see the context clips — it CANNOT see the future. Therefore the correct \
next step MUST be logically or physically predictable from the momentum, \
setup, or procedural intent visible in the LAST context clip. \
DO NOT ask the model to predict random or highly contingent outcomes \
(e.g., exact scores, precise distances, foul vs. valid, success vs. \
failure of an attempt) UNLESS the very end of the context provides \
unambiguous visual foreshadowing (e.g., an athlete is already slipping). \
The correct answer should describe the immediate next MECHANICAL ACTION \
(e.g., "the athlete begins to spin and release the shot put"), NOT the \
ultimate unpredictable outcome (e.g., "throws 55 feet 3 inches"). \
If the next item in the script is an unpredictable result rather than \
a deterministic action, move the context window forward so the result \
becomes part of the context and the next action is the correct answer, \
or set "suitable" to false.
6. **Uniqueness check**: Re-read the context. Is the correct answer the \
ONLY step that is both (i) physically possible given the current state \
and (ii) not yet done? If another distractor could also be valid, \
redesign or set "suitable" to false.
7. Provide the correct answer as a text description (copy or closely \
paraphrase the script entry).
8. **Multiple questions**: Scan the ENTIRE script for different viable \
context windows. Each question MUST use a DIFFERENT set of context IDs \
(no overlap). Only include questions with genuine causal chains and \
strong distractors — prefer quality over quantity.

If the script lacks any clear sequential chain, return an empty \
questions array.

Respond in this exact JSON format:
{{
  "questions": [
    {{
      "suitable": true,
      "granularity": "Event",
      "reasoning": "Brief explanation of: (1) the causal chain, (2) why each distractor shares the same object/scene but fails logically, (3) uniqueness verification",
      "context_ids": ["Event 1", "Event 2"],
      "correct_next_id": "Event 3",
      "correct_next_text": "Description of the correct next step",
      "distractors": [
        "Distractor 1 (same object/scene, wrong technique or intensity): ...",
        "Distractor 2 (same object/scene, wrong direction or destination): ...",
        "Distractor 3 (same object/scene, temporal duplicate or already done): ..."
      ]
    }}
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

Design "fill in the missing step" single-choice questions from the \
action script above. Return ALL high-quality questions the script \
supports — do not artificially limit yourself to one.

Instructions:
1. **Granularity**: For each question, decide whether Event-level or \
Action-level yields a more meaningful "before → core change → after" \
pattern. Different questions may use different granularities. Stick to \
ONE level within each question.
2. Find chains of 3+ STRICTLY CONSECUTIVE items (NO GAPS — e.g., \
Action 3.1 → 3.2 → 3.3, not 3.1 → 3.2 → 3.4) where the MIDDLE item \
represents a critical state change or transition (the "bridge" between \
cause and effect). All items in the chain must be purposeful physical \
actions — skip any talking / explaining / pausing shots. If a \
non-action item (talking shot) sits between two good action items, you \
MUST NOT skip over it — instead, choose a different chain that has no \
gaps.
   **Causal Chain Requirement**: Every consecutive pair in the full \
chain (before → missing → after) MUST have a genuine cause-effect \
link. The "before" step(s) must PRODUCE a state that the missing step \
CONSUMES, and the missing step must PRODUCE a state that the "after" \
step(s) CONSUME. \
REJECT chains where items are merely different camera angles of the \
same action, slow-motion replays, or parallel / interchangeable \
actions that could be reordered without consequence. If no such causal \
chain exists, set "suitable" to false for that question.
3. The BEFORE item(s) and AFTER item(s) will be shown as video context; \
the middle item is REMOVED — the model must identify it.
4. Write exactly 3 DISTRACTORS following BOTH the Anti-Shortcut rule \
AND the Temporal Lock rule:
   - All 3 distractors MUST involve the SAME core object(s) as the \
correct answer (e.g., if correct="stir the sauce with a wooden spoon", \
all distractors must also involve "sauce" and/or "spoon").
   - **If Action granularity**: distractors MUST be micro-actions that \
could happen in the EXACT SAME short scene and physical setup. \
DO NOT use macro-events from a different phase (e.g., "serve on plate" \
when the scene is still at the stovetop). \
Instead, alter the TOOL, DIRECTION, INTENSITY, or INTERACTION:
     (a) Same object + tool, but produces a WRONG STATE that cannot \
lead to the "after" context (e.g., "pour out the sauce" when the \
after shows sauce still in the pan being stirred).
     (b) Same object, but CONTRADICTS what the "before" already shows \
(e.g., "add raw tomatoes to the sauce" when before already shows a \
smooth cooked sauce).
     (c) Same object + tool, but wrong INTENSITY or MANNER that would \
produce a physically different outcome (e.g., "rapidly boil the sauce \
until reduced to paste" when the after shows a liquid sauce).
   - **If Event granularity**: distractors should be events from the \
same domain but with wrong causal bridging:
     (a) Connects to before but cannot produce the after state.
     (b) Could produce the after state but contradicts the before state.
     (c) Same-domain but physically impossible as a bridge.
5. **Causal Predictability Rule**: The missing step MUST be a \
deterministic physical action that is the ONLY way to bridge the \
"before" state to the "after" state — NOT a random or contingent \
outcome (e.g., exact measurements, scores, success/failure). \
The test model sees the before and after clips; the correct answer \
should describe a concrete mechanical action whose necessity is \
visually obvious from the state change between before and after. \
If the removed step is an unpredictable outcome rather than a \
deterministic action, choose a different item to remove, or set \
"suitable" to false.
6. **Uniqueness check**: Given ONLY the before and after clips, verify \
that the correct answer is the SOLE physically necessary bridge step. \
If a distractor could also validly connect before→after, redesign or \
set "suitable" to false.
7. Provide the correct answer as a text description.
8. **Multiple questions**: Scan the ENTIRE script for different viable \
chains. Each question MUST use a DIFFERENT missing item (no overlap in \
missing_id). Only include questions with genuine causal bridges and \
strong distractors — prefer quality over quantity.

If no clear before→change→after chain exists anywhere in the script, \
return an empty questions array.

Respond in this exact JSON format:
{{
  "questions": [
    {{
      "suitable": true,
      "granularity": "Action",
      "reasoning": "Brief explanation of: (1) why the removed step is the critical bridge, (2) why each distractor shares the same object/scene but fails as a bridge, (3) uniqueness verification",
      "before_ids": ["Action 3.1"],
      "missing_id": "Action 3.2",
      "after_ids": ["Action 3.3", "Action 3.4"],
      "correct_text": "Description of the missing step",
      "distractors": [
        "Distractor 1 (same object/scene, wrong state for after): ...",
        "Distractor 2 (same object/scene, contradicts before): ...",
        "Distractor 3 (same object/scene, wrong intensity/manner): ..."
      ]
    }}
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

Design "reorder shuffled steps" questions from the action script above. \
Return ALL high-quality questions the script supports — do not \
artificially limit yourself to one.

Instructions:
1. **Granularity**: For each question, decide whether a macro Event \
sequence (e.g., wash → cut → stir-fry) or a micro Action sequence \
(e.g., pick up knife → cut first slice → set knife down) has a more \
ABSOLUTELY IRREVERSIBLE ordering. Different questions may use different \
granularities. Stick to ONE level within each question.
2. Select 3 to 5 items whose chronological order is dictated by strict \
causal or physical prerequisites (e.g., "you must open the lid before \
pouring the contents"). Every consecutive pair MUST have a cause-effect \
link — the earlier step produces a physical state that the later step \
depends on.
3. **Exclusion criteria**: Do NOT include items where any pair could be \
swapped without logical contradiction. Specifically reject:
   - Interchangeable repetitive motions (e.g., stir stroke #1 vs. #2)
   - Different camera angles / replays of the same action
   - Parallel actions with no dependency (e.g., "sprinkle salt" and \
"sprinkle pepper")
   - Decorative variations (e.g., place item left vs. right)
4. Return the selected item IDs in their TRUE chronological order.
5. **Multiple questions**: Scan the ENTIRE script for different viable \
sequences. Each question MUST use a DIFFERENT set of ordered IDs (no \
overlap). Only include questions with strict irreversible ordering — \
prefer quality over quantity.

If the script contains no sequence of 3+ items with absolute, \
irreversible ordering, return an empty questions array.

Respond in this exact JSON format:
{{
  "questions": [
    {{
      "suitable": true,
      "granularity": "Event",
      "reasoning": "Why this ordering is strictly irreversible (physical laws, causal chain)",
      "ordered_ids": ["Event 1", "Event 2", "Event 3", "Event 4"]
    }}
  ]
}}"""
