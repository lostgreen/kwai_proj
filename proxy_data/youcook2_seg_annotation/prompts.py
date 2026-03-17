"""
3-level hierarchical annotation prompts for YouCook2 DVC.

Design philosophy:
  Level 1 — Warped-Time Segmentation: model sees uniformly sampled frames
             numbered 1..N (no real timestamps), predicts phase boundaries on
             the warped frame axis.  Engineering maps back to real time.
  Level 2 — Phase-Based Event Detection: for each L1 macro phase, model
             detects cooking events within that phase scope.  Depends on L1.
             (128s sliding windows are only used at training-data construction
             time in build_dataset.py, NOT during annotation.)
  Level 3 — Local Temporal Grounding: given an L2 event clip + text query,
             model pinpoints start/end of atomic state-change moments.

Usage:
    from prompts import SYSTEM_PROMPT, get_level1_prompt, get_level2_prompt, get_level3_prompt
"""

# ─────────────────────────────────────────────────────────────────────────────
# Level 0: System Prompt
# Injected as the system role message in every API call.
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert in structured video analysis, specializing in instructional cooking content. \
Your task is to accurately parse temporal actions and visual state transitions.

Core annotation principles:
1. [Sparsity over Continuity]: ONLY annotate segments with clear semantic meaning or \
visual actions. Gaps between annotated segments are expected and encouraged. \
Do NOT force adjacent segments to be contiguous.
2. [Precise Boundaries]: Boundaries must reflect the exact moment an action begins \
or the moment a visual state solidifies.
3. [Formatting]: Output strictly in valid JSON format."""


# ─────────────────────────────────────────────────────────────────────────────
# Level 1: Macro Phase — Warped-Time Segmentation (阶段级)
# Input:  N uniformly sampled frames labelled [Frame 1] .. [Frame N]
# Output: JSON with macro_phases, boundaries in warped frame indices
# ─────────────────────────────────────────────────────────────────────────────
_LEVEL1_BASE = """\
You are given {n_frames} frames uniformly sampled from a cooking video. \
The frames are numbered 1 to {n_frames}. These numbers are ordinal positions, \
NOT real-world timestamps. Your task is to segment the frame sequence into \
high-level macro phases (typically 3 to 5).

LEVEL 1 DEFINITION:
- A macro phase is a broad cooking stage such as ingredient preparation, \
sauce making, cooking/heating, assembly, or plating/finalization.
- A macro phase may span many frames and contain multiple fine-grained actions.
- Macro phases do NOT need to cover all {n_frames} frames. Skip intros, outros, \
talking-only spans, beauty shots, or any content not advancing the dish.
- Group by recipe intent, not by camera cut or tiny motion change.

COOKING RELEVANCE FILTER — exclude frames/spans that show:
- Pure narration or face-to-camera talking with no food manipulation
- Beauty shots, eating/tasting reactions, idle waiting
- Unrelated setup/cleanup, tool-only motions not affecting food

For each phase, provide:
- phase_id: Sequential integer starting from 1.
- start_frame / end_frame: Boundary frame numbers (1-indexed, within 1..{n_frames}).
- phase_name: A concise noun phrase (e.g., "Ingredient Preparation").
- narrative_summary: One sentence describing the phase's core objective.

Output JSON:
{{
  "macro_phases": [
    {{
      "phase_id": 1,
      "start_frame": 3,
      "end_frame": 12,
      "phase_name": "Ingredient Preparation",
      "narrative_summary": "Wash and dice all vegetables and proteins."
    }}
  ]
}}"""


def get_level1_prompt(n_frames: int) -> str:
    """
    Build the Level 1 (Warped-Time Macro Phase) user-turn prompt.

    Args:
        n_frames: Number of uniformly sampled frames the model will see.
    """
    return _LEVEL1_BASE.format(n_frames=n_frames)


# ─────────────────────────────────────────────────────────────────────────────
# Level 2: Cooking Event — Phase-Based Event Detection (活动级)
# Input:  frames within an L1 macro phase (real timestamps) + phase context
# Output: events detected in this phase
# ─────────────────────────────────────────────────────────────────────────────
_LEVEL2_BASE = """\
You are an event detector. You are viewing frames from a cooking phase \
({phase_start}s to {phase_end}s, duration {phase_duration}s). \
{phase_context}\
Identify all complete cooking events that occur within this phase.

LEVEL 2 DEFINITION:
- A cooking event is a multi-second, goal-directed workflow that transforms \
ingredients, cookware contents, or the dish state, OR completes a meaningful \
recipe subgoal.
- It is larger than a single atomic motion but smaller than a full recipe stage.

CRITICAL GRANULARITY RULES:
1. [Aggregation Constraint]: Each event MUST be a complete logical sub-process, \
NOT an isolated atomic action.
2. [Anti-Fragmentation]: DO NOT split into single momentary actions (e.g., \
"pick up spoon", "place filling", "fold corner"). Group them into one coherent event.
3. [Good vs. Bad]:
   - BAD (too granular): "Place a spoonful of filling onto the wrapper" (~1s)
   - GOOD (correct): "Assemble the wonton by filling and folding the wrapper" (~15-20s)
4. [Cooking-Only Filter]: Exclude:
   - talking/explaining without food manipulation
   - showing ingredients/tools without using them
   - idle motions, repositioning, tool pickup with no food-state change
   - waiting/resting with no visible progress
   - beauty shots, serving, tasting reactions
5. [Boundary Events]: If an event extends slightly beyond phase boundaries, \
still annotate it with accurate timestamps.
6. [Empty Is Valid]: If no cooking event occurs in this phase, return: {{"events": []}}

For each event, provide:
- event_id: Sequential integer starting from 1.
- start_time / end_time: Timestamps in integer seconds (absolute, not phase-relative).
- instruction: High-level description of the complete cooking event.
- visual_keywords: 3-5 key visual elements observed during this event.

Output JSON:
{{
  "events": [
    {{
      "event_id": 1,
      "start_time": 32,
      "end_time": 55,
      "instruction": "Assemble the wontons by filling and folding wrappers",
      "visual_keywords": ["wonton wrapper", "meat filling", "hands folding"]
    }}
  ]
}}"""


def get_level2_prompt(
    phase_start_sec: int,
    phase_end_sec: int,
    phase_name: str = "",
    narrative_summary: str = "",
) -> str:
    """
    Build the Level 2 (Phase-Based Event Detection) user-turn prompt.

    Args:
        phase_start_sec: Start of the L1 macro phase (seconds).
        phase_end_sec: End of the L1 macro phase (seconds).
        phase_name: Name of the macro phase from L1 (optional).
        narrative_summary: L1 narrative summary for context (optional).
    """
    phase_duration = phase_end_sec - phase_start_sec
    context_parts = []
    if phase_name:
        context_parts.append(f'Phase: "{phase_name}".')
    if narrative_summary:
        context_parts.append(f'Phase summary: "{narrative_summary}".')
    phase_context = " ".join(context_parts) + "\n" if context_parts else ""
    return _LEVEL2_BASE.format(
        phase_duration=phase_duration,
        phase_start=phase_start_sec,
        phase_end=phase_end_sec,
        phase_context=phase_context,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Training Prompts — simplified <events> output format
# Used by build_dataset.py (NOT annotate.py).  The model outputs segments as:
#   <events>[[start, end], [start, end], ...]</events>
# ─────────────────────────────────────────────────────────────────────────────

_LEVEL1_TRAIN_BASE = """\
You are given {n_frames} frames uniformly sampled from a cooking video, numbered 1 to {n_frames}. \
Segment the frame sequence into 3–5 high-level macro cooking phases \
(e.g., ingredient preparation, cooking/heating, assembly, plating). \
Skip non-cooking spans such as narration, beauty shots, or idle waiting.

Output the start and end frame number for each phase in order:
<events>[[start_frame, end_frame], ...]</events>

Example: <events>[[3, 80], [95, 150], [160, 220]]</events>"""


def get_level1_train_prompt(n_frames: int) -> str:
    """Training prompt for Level 1 (warped-frame macro phase segmentation)."""
    return _LEVEL1_TRAIN_BASE.format(n_frames=n_frames)


_LEVEL2_TRAIN_BASE = """\
You are given a {duration}s cooking video clip (timestamps 0 to {duration}). \
Detect all complete cooking events in this clip. \
Each event is a multi-second, goal-directed workflow that transforms ingredients or completes a recipe subgoal. \
Skip idle waiting, narration, tool pickup, or beauty shots.

Output the start and end time (integer seconds, 0-based) for each event in order:
<events>[[start_time, end_time], ...]</events>

Example: <events>[[5, 42], [55, 90]]</events>"""


def get_level2_train_prompt(duration: int) -> str:
    """Training prompt for Level 2 (event detection, 0-based seconds)."""
    return _LEVEL2_TRAIN_BASE.format(duration=duration)


# ─────────────────────────────────────────────────────────────────────────────
# Level 3: Atomic Interaction — Local Temporal Grounding (动作级)
# Input:  frames within an L2 event clip + action query text
# Output: grounding results with pre/post state descriptions
# ─────────────────────────────────────────────────────────────────────────────
_LEVEL3_BASE = """\
You are a temporal grounding model. You are viewing frames from a cooking event \
clip ({event_start}s to {event_end}s). The cooking event is: "{action_query}"

Your task: pinpoint every atomic state-change moment within this clip.

LEVEL 3 DEFINITION — KINEMATIC BOUNDARIES:
1. [Physics over Recipe]: You are NOT writing recipe steps. Focus ENTIRELY on \
physical, visual changes of objects (deformation, separation, merging, transfer, \
material state change).
2. [State Transition Focus]: Only annotate moments where a target object undergoes \
a VISUAL, IRREVERSIBLE change.
3. [Boundary Precision]:
   - start_time = the moment physical contact begins OR the object starts entering \
the state change (NOT the reaching/approaching motion).
   - end_time = the moment the transfer/transformation completes and the new visual \
state is clearly established.
4. [Typical Duration]: Most atomic moments span 2-6 seconds. Avoid 1-second spans \
unless the change is truly instantaneous yet visually complete.
5. [Continuity Rule]: If consecutive frames show one uninterrupted micro-process \
with the same intent, merge into one annotation rather than splitting.
6. [Allow Gaps]: Do NOT force timestamps to cover every second. Skip hand \
repositioning, tool pickup, idle pauses, narration, and any motion that does NOT \
change the object's state.
7. [Ignore Empty Motions]: Purely human limb movements (reaching, adjusting posture) \
without any object state change must be excluded.

For each atomic state-change moment, provide:
- action_id: Sequential integer starting from 1.
- start_time / end_time: Timestamps in integer seconds (absolute within the full video).
- sub_action: Brief description of the specific physical interaction.
- pre_state: The EXPLICIT visual state of the target object BEFORE the interaction.
- post_state: The EXPLICIT visual state of the target object AFTER the interaction.

Output JSON:
{{
  "grounding_results": [
    {{
      "action_id": 1,
      "start_time": 42,
      "end_time": 47,
      "sub_action": "Pour minced garlic into the heated pan",
      "pre_state": "Dry pan with a thin layer of oil",
      "post_state": "Minced garlic scattered across the pan surface, beginning to sizzle"
    }}
  ]
}}"""


def get_level3_prompt(
    event_start_sec: int,
    event_end_sec: int,
    action_query: str,
) -> str:
    """
    Build the Level 3 (Local Temporal Grounding) user-turn prompt.

    Args:
        event_start_sec: Start of the L2 event clip (seconds).
        event_end_sec: End of the L2 event clip (seconds).
        action_query: The L2 event instruction to ground into atomic moments.
    """
    return _LEVEL3_BASE.format(
        event_start=event_start_sec,
        event_end=event_end_sec,
        action_query=action_query,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Level 3 Query: Query-conditioned Atomic Grounding (查询式动作级 grounding)
# Input:  an event video clip (possibly with padding) + ordered list of action captions
# Output: start/end times for each action, in the given query order
# ─────────────────────────────────────────────────────────────────────────────
_LEVEL3_QUERY_BASE = """\
You are given a {duration}s cooking video clip and a numbered list of actions to locate. \
Find the time segment for each action, answering in the given order.

Actions to locate:
{action_list}

Rules:
- answer in the same order as the list above (1, 2, 3, ...)
- start_time / end_time are integer seconds from the start of the clip (0-based)
- each segment must satisfy: 0 ≤ start_time < end_time ≤ {duration}

Output one [start_time, end_time] pair per action in order:
<events>[[start_time, end_time], ...]</events>

Example: <events>[[3, 7], [0, 2], [10, 14]]</events>"""


def get_level3_query_prompt(queries: list[str], duration: int) -> str:
    """
    Build the Level 3 Query (query-conditioned grounding) user-turn prompt.

    Args:
        queries: Ordered list of action caption strings to locate.
                 May be in original annotation order or shuffled.
        duration: Duration of the video clip in seconds (0-based timestamps).
    """
    action_list = "\n".join(f'{i + 1}. "{q}"' for i, q in enumerate(queries))
    return _LEVEL3_QUERY_BASE.format(duration=duration, action_list=action_list)


# Input:  frames within an L2 event clip + existing L3 grounding_results
# Output: reviewed results with verdicts, corrections, and supplemented actions
# ─────────────────────────────────────────────────────────────────────────────
_LEVEL3_CHECK_BASE = """\
You are a quality reviewer for temporal grounding annotations. You are viewing \
frames from a cooking event clip ({event_start}s to {event_end}s). \
The cooking event is: "{action_query}"

Below are the EXISTING L3 atomic action annotations for this event:
{existing_annotations}

Your task: REVIEW each existing annotation AND identify any MISSING atomic actions.

REVIEW CRITERIA:
1. [Temporal Accuracy]: Does start_time/end_time match what is actually visible in \
the frames? Is the boundary precise (start = physical contact begins, end = new \
state established)?
2. [State Description Quality]: Are pre_state and post_state specific, concrete, \
and visually verifiable? Vague descriptions like "food on table" or "ingredients \
ready" are INSUFFICIENT — they should describe exact visual appearance.
3. [Relevance]: Does the sub_action describe a real physical state change of an \
object? Exclude pure hand movements, tool pickups, or posture adjustments without \
any object state change.
4. [Granularity]: Is the action truly atomic (2-6s typical)? If a single annotation \
covers multiple distinct state changes, it should be split.
5. [Boundary Compliance]: start_time must be >= {event_start} and end_time must \
be <= {event_end}.
6. [Completeness]: Are there any visible atomic state-change moments in the frames \
that are NOT covered by the existing annotations?

For each existing annotation, output a verdict:
- "keep": Annotation is correct as-is.
- "revise": Annotation has issues — provide the corrected version.
- "remove": Annotation is invalid (no real state change, or duplicate).

Then, list any MISSING actions that should be added.

Output JSON:
{{
  "reviews": [
    {{
      "action_id": 1,
      "verdict": "keep"
    }},
    {{
      "action_id": 2,
      "verdict": "revise",
      "revised": {{
        "start_time": 45,
        "end_time": 49,
        "sub_action": "Corrected description of the physical interaction",
        "pre_state": "More specific pre-state description",
        "post_state": "More specific post-state description"
      }}
    }},
    {{
      "action_id": 3,
      "verdict": "remove",
      "reason": "No visible state change; just hand repositioning"
    }}
  ],
  "supplements": [
    {{
      "start_time": 55,
      "end_time": 59,
      "sub_action": "Description of a missed atomic action",
      "pre_state": "Pre-state of the missed action",
      "post_state": "Post-state of the missed action"
    }}
  ]
}}

IMPORTANT:
- You MUST review every existing annotation by action_id.
- "supplements" can be an empty list if nothing is missing.
- Do NOT invent actions that are not visible in the provided frames.
- Be strict: vague or non-physical annotations should be revised or removed."""


def get_level3_check_prompt(
    event_start_sec: int,
    event_end_sec: int,
    action_query: str,
    existing_results: list[dict],
) -> str:
    """
    Build the Level 3 Check (Quality Judge & Supplement) user-turn prompt.

    Args:
        event_start_sec: Start of the L2 event clip (seconds).
        event_end_sec: End of the L2 event clip (seconds).
        action_query: The L2 event instruction.
        existing_results: List of existing grounding_results dicts for this event.
    """
    import json as _json
    # Format existing annotations compactly for the prompt
    display_results = []
    for r in existing_results:
        display_results.append({
            "action_id": r.get("action_id"),
            "start_time": r.get("start_time"),
            "end_time": r.get("end_time"),
            "sub_action": r.get("sub_action"),
            "pre_state": r.get("pre_state"),
            "post_state": r.get("post_state"),
        })
    annotations_str = _json.dumps(display_results, ensure_ascii=False, indent=2)

    return _LEVEL3_CHECK_BASE.format(
        event_start=event_start_sec,
        event_end=event_end_sec,
        action_query=action_query,
        existing_annotations=annotations_str,
    )
