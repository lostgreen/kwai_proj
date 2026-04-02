"""
Topology-adaptive hierarchical annotation prompts (domain-agnostic).

Design philosophy:
  Merged  — Single-call L1+L2+Topology: model sees full video (real timestamps),
             classifies temporal topology, then outputs macro phases, nested
             activity events (conditional on topology), domain, and summary.
  Level 3 — Topology-aware micro grounding: definition adapts per topology_type.
             procedural → state_change (object state transitions)
             periodic   → repetition_unit (single rep/cycle/strike)
             sequence/flat → skipped (no L3)

Usage:
    from prompts import SYSTEM_PROMPT, get_merged_l1l2_prompt, get_level3_prompt
    from prompts import DOMAIN_TAXONOMY, TOPOLOGY_TYPES
    from prompts import TOPOLOGY_TO_L2_MODE, TOPOLOGY_TO_L3_MODE
"""

# ─────────────────────────────────────────────────────────────────────────────
# Level 0: System Prompt
# Injected as the system role message in every API call.
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert in structured video analysis, specializing in temporal \
structure analysis of procedural, periodic, sequential, and general activity content. \
Your task is to accurately parse temporal actions and visual state transitions.

Core annotation principles:
1. [Sparsity over Continuity]: ONLY annotate segments with clear semantic meaning or \
visual actions. Gaps between annotated segments are expected and encouraged. \
Do NOT force adjacent segments to be contiguous.
2. [Precise Boundaries]: Boundaries must reflect the exact moment an action begins \
or the moment a visual state solidifies.
3. [Formatting]: Output strictly in valid JSON format."""


# ─────────────────────────────────────────────────────────────────────────────
# Domain Taxonomy (two-level hierarchy)
# L1 = broad activity category; L2 = fine-grained subcategory.
# Used by merged L1+L2 annotation; VLM picks one L1 and one L2 label.
# ─────────────────────────────────────────────────────────────────────────────
DOMAIN_TAXONOMY: dict[str, list[str]] = {
    "procedural": [
        "cooking",               # Food preparation, baking, meal assembly
        "construction_building", # Woodworking, metalwork, structural assembly
        "crafting_diy",          # Arts, crafts, handmade projects
        "repair_maintenance",    # Fixing, servicing mechanical/electronic items
    ],
    "physical": [
        "sports",                # Athletic activities, games, competitions
        "fitness_exercise",      # Workout routines, yoga, training
        "music_performance",     # Playing instruments, dance, stage performance
    ],
    "lifestyle": [
        "beauty_grooming",       # Hair, makeup, skincare, personal care
        "cleaning_housework",    # Household chores, organization, laundry
        "gardening_outdoor",     # Planting, landscaping, outdoor work
        "vehicle_operation",     # Driving, cycling, vehicle maintenance
    ],
    "educational": [
        "science_experiment",    # Lab work, chemistry, physics demonstrations
    ],
    "other": [
        "other",                 # Catch-all for unmatched domains
    ],
}

# Flat set of all valid L2 subcategory values (for validation)
DOMAIN_L2_ALL: set[str] = {sub for subs in DOMAIN_TAXONOMY.values() for sub in subs}


# ─────────────────────────────────────────────────────────────────────────────
# Topology Classification (four temporal structure types)
# Determined during merged annotation; drives L2/L3 routing downstream.
# ─────────────────────────────────────────────────────────────────────────────
TOPOLOGY_TYPES: set[str] = {"procedural", "periodic", "sequence", "flat"}
L2_MODES: set[str] = {"workflow", "episode", "optional", "skip"}
L3_MODES: set[str] = {"state_change", "repetition_unit", "skip"}

TOPOLOGY_TO_L2_MODE: dict[str, str] = {
    "procedural": "workflow",
    "sequence": "episode",
    "periodic": "optional",
    "flat": "skip",
}
TOPOLOGY_TO_L3_MODE: dict[str, str] = {
    "procedural": "state_change",
    "periodic": "repetition_unit",
    "sequence": "skip",
    "flat": "skip",
}


def _format_taxonomy_for_prompt() -> str:
    """Format DOMAIN_TAXONOMY as indented bullet list for prompt injection."""
    lines = []
    for l1, l2_list in DOMAIN_TAXONOMY.items():
        lines.append(f"  {l1}:")
        for l2 in l2_list:
            lines.append(f"    - {l2}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Merged L1+L2+Topology: Single-Call Phase + Event + Domain + Topology
# Input:  full video frames with real timestamps
# Output: domain, topology, summary, macro_phases (with nested events)
# ─────────────────────────────────────────────────────────────────────────────
_MERGED_L1L2_BASE = """\
You are given a {duration}s video clip (timestamps 0 to {duration}) with {n_frames} frames.
Your task has four parts.

## PART 1 — DOMAIN CLASSIFICATION
Classify the video using a two-level taxonomy.
Choose ONE broad category (domain_l1) and ONE fine-grained subcategory (domain_l2) \
from the list below:
{domain_taxonomy_str}

## PART 2 — TOPOLOGY CLASSIFICATION (CRITICAL)
Analyze the TEMPORAL STRUCTURE of the visible activity and assign exactly ONE topology_type.

Topology types:
- procedural:
  A step-by-step process with meaningful sub-goals that progress toward an outcome.
  Typical examples: cooking, assembling, repairing, crafting.

- periodic:
  A repeated cycle of the same motion or operation.
  Typical examples: stretching repetitions, weightlifting reps, repetitive factory motions.

- sequence:
  A continuous traversal, trial, run, or episode with one coherent trajectory or attempt.
  Typical examples: dog agility runs, parkour runs, skiing descents, obstacle traversals.

- flat:
  A single continuous activity with no stable internal hierarchy, or mixed/unclear \
structure that should not be over-segmented.
  Typical examples: idle talking, continuous walking vlog, loosely mixed footage.

Important topology rules:
1. Topology is about temporal structure, NOT about domain label alone.
2. Repetition alone does NOT imply procedural structure.
3. Camera cuts do NOT define topology.
4. If the structure is weak or unclear, choose flat rather than inventing hierarchy.

Also output:
- topology_confidence: a float from 0.0 to 1.0
- topology_reason: one brief sentence explaining the decision

## PART 3 — VIDEO SUMMARY & MACRO PHASES (L1)
Write ONE sentence summarizing the video.

Then segment the video into 1–6 macro phases.
A macro phase is a broad stage of activity organized by overall intent.
- Skip intros, outros, static non-activity spans, and talking-only spans.
- Phases do NOT need to cover the entire video.
- Do NOT split by camera cuts.
- It is valid to output only 1 macro phase if the entire video is one continuous routine.

CAPTION QUALITY (CRITICAL):
- phase_name MUST be a descriptive phrase of 5–15 words that conveys the specific \
goal, key objects, and outcome of the stage. One-or-two-word labels like \
"Preparation" or "Assembly" are NOT acceptable.
  Good: "Preparing and measuring dry ingredients for the base mixture"
  Bad:  "Material Preparation"
- narrative_summary MUST be 2–3 sentences describing the key actions performed, \
the objects/materials involved, and the visible state changes during this phase.
  Good: "The person measures flour and sugar into a large mixing bowl. Both dry \
ingredients are whisked together until the mixture appears uniform."
  Bad:  "Gather and organize all required materials."

Then explain your phase segmentation logic in one sentence (global_phase_criterion): \
what criterion distinguishes one phase from the next. Focus on the structural or \
intentional boundary (e.g., shift of goal, change of activity type, transition between \
setup and execution), NOT a description of video content.

## PART 4 — EVENT DETECTION (L2)
Detect events nested inside each macro phase.
Apply the event definition STRICTLY based on topology_type:

If topology_type = procedural:
- An event is a multi-second workflow (typically 10–60s) that completes a process sub-goal.
- Group related manipulations together.
- Do NOT fragment into atomic tool motions.
- If a macro phase consists of a single continuous operation or a simple repetitive action \
(e.g., "hammering nails for 60 seconds", "stirring continuously"), leave its "events": [].
- ONLY create events when the phase contains distinct sequential sub-steps.

If topology_type = sequence:
- An event is a complete episode, trial, or continuous traversal with one coherent \
trajectory or objective.
- Do NOT split by local body motions, individual obstacles, or camera cuts.
- A whole run/trial should usually be one event.

If topology_type = periodic:
- Events are optional.
- You may leave "events": [] for a phase, or output ONE event only if it exactly \
matches the whole phase as a container for later micro annotation.
- Do NOT create one event per repetition.

If topology_type = flat:
- Output "events": [].
- Do NOT invent L2 structure.

General L2 rules:
- Events must not overlap.
- Use absolute integer seconds.
- It is valid for a phase to contain zero events.
- Do not force extra events to make the hierarchy deeper.

EVENT CAPTION QUALITY (CRITICAL):
- instruction MUST be a descriptive sentence of 8–20 words that clearly states \
WHAT is being done, with WHICH objects/materials, and toward WHAT outcome.
  Good: "Whisk dry flour and sugar together in a large bowl until evenly blended"
  Bad:  "Mix ingredients"
  Bad:  "Sort and measure the raw materials"
- visual_keywords MUST include specific visible objects, tools, and materials \
(not abstract concepts).
- For each macro phase, provide an event_split_criterion: a one-sentence explanation \
of WHY this phase does or does not contain sub-events. \
If events exist, explain the boundary logic (e.g., "segmented by logical progression \
of sub-goals"). If events are empty, explain why (e.g., "single continuous action with \
no sequential progression"). Focus on segmentation logic, not content description.

Output JSON:
{{
  "domain_l1": "<one broad category>",
  "domain_l2": "<one fine-grained subcategory>",
  "topology_type": "procedural | periodic | sequence | flat",
  "topology_confidence": 0.95,
  "topology_reason": "<one sentence>",
  "l2_mode": "workflow | episode | optional | skip",
  "summary": "<one sentence>",
  "global_phase_criterion": "<one sentence explaining WHY the video is split into these phases>",
  "macro_phases": [
    {{
      "phase_id": 1,
      "start_time": 5,
      "end_time": 60,
      "phase_name": "Preparing and measuring dry ingredients for the base mixture",
      "narrative_summary": "The person retrieves flour and sugar from storage containers. Both dry ingredients are measured using a scale and poured into a large mixing bowl.",
      "event_split_criterion": "<one sentence explaining WHY this phase does/does not have events>",
      "events": [
        {{
          "event_id": 1,
          "start_time": 8,
          "end_time": 25,
          "instruction": "Measure flour and sugar on a kitchen scale and transfer into a mixing bowl",
          "visual_keywords": ["flour bag", "sugar container", "digital scale", "mixing bowl"]
        }}
      ]
    }}
  ]
}}"""


def get_merged_l1l2_prompt(n_frames: int, duration_sec: int) -> str:
    """
    Build the merged L1+L2 (phase segmentation + event detection + domain) prompt.

    Single VLM call outputs domain_l1, domain_l2, summary, macro_phases with nested events.
    Uses real timestamps (not warped frames).

    Args:
        n_frames: Number of frames the model will see.
        duration_sec: Total video duration in seconds.
    """
    return _MERGED_L1L2_BASE.format(
        n_frames=n_frames,
        duration=duration_sec,
        domain_taxonomy_str=_format_taxonomy_for_prompt(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Training Prompts — simplified <events> output format
# Used by build_hier_data.py (NOT annotate.py). The model outputs segments as:
#   <events>[[start, end], [start, end], ...]</events>
# ─────────────────────────────────────────────────────────────────────────────

_LEVEL1_TRAIN_TEMPORAL_BASE = """\
You are given a {duration}s video clip (timestamps 0 to {duration}). \
Segment the video into its high-level macro phases (broad activity stages).

Granularity guide:
- A macro phase is a broad stage of activity organized by overall intent \
(e.g., preparation → execution → finishing).
- Output 1–6 phases. A single-phase video is valid if it is one continuous routine.
- Skip intros, outros, static non-activity spans, narration-only, and idle waiting.
- Phases do NOT need to cover the entire video — gaps are expected.
- Do NOT split by camera cuts or brief pauses.

Output the start and end time (integer seconds, 0-based) for each phase in order:
<events>[[start_time, end_time], ...]</events>

Example: <events>[[0, 85], [90, 170], [180, 240]]</events>"""


def get_level1_train_prompt_temporal(duration: int) -> str:
    """Training prompt for Level 1 (temporal macro phase segmentation, 0-based seconds)."""
    return _LEVEL1_TRAIN_TEMPORAL_BASE.format(duration=duration)


_LEVEL2_TRAIN_BASE = """\
You are given a {duration}s video clip (timestamps 0 to {duration}). \
Detect all activity events in this clip.

Granularity guide — an event sits between two levels:
- TOO COARSE (phase-level): a broad stage summarizing the whole clip. \
Do NOT output an event that restates the overall activity.
- CORRECT (event-level): a multi-second workflow (typically 10–60s) that \
completes a meaningful sub-goal. It involves a sequence of physical \
interactions unified by a single intent.
- TOO FINE (atomic action): a single brief motion (2–6s) like one pour, \
one cut stroke, or picking up a tool. Do NOT fragment into atomic actions.

Rules:
- Events must not overlap.
- Group related manipulations into one event rather than splitting by each tool motion.
- Skip idle waiting, narration, tool-only movements, or non-activity content.
- It is valid to output fewer events if the clip has few distinct sub-goals.
- Gaps between events are expected and encouraged.

Output the start and end time (integer seconds, 0-based) for each event in order:
<events>[[start_time, end_time], ...]</events>

Example: <events>[[5, 42], [55, 90]]</events>"""


def get_level2_train_prompt(duration: int) -> str:
    """Training prompt for Level 2 (event detection, 0-based seconds)."""
    return _LEVEL2_TRAIN_BASE.format(duration=duration)


# ─────────────────────────────────────────────────────────────────────────────
# Level 3: Topology-Aware Micro Grounding (动作级 / 重复单元级)
# Input:  frames within a clip + action/phase query + topology_type
# Output: grounding results with pre/post state descriptions
# ─────────────────────────────────────────────────────────────────────────────
_LEVEL3_BASE = """\
You are a temporal grounding model. You are viewing frames from a clip \
({clip_start}s to {clip_end}s).
The input query is: "{action_query}"
The topology_type of the source video is: "{topology_type}".

Your task is to pinpoint every atomic micro-action in this clip.

IMPORTANT:
- If topology_type is "sequence" or "flat", this prompt should not be used.
- Use absolute integer seconds from the full video timeline.

LEVEL 3 DEFINITIONS

If topology_type = procedural:
- micro_type = "state_change"
- Find brief atomic actions where an object undergoes a clear visible physical change.
- Valid examples: cutting, pouring into a container, attaching a part, spreading \
material, separating pieces.
- start_time = onset of the actual physical interaction or transformation
- end_time = the moment the new visible state is established
- Ignore reaching, idle pauses, pure hand repositioning, and narration

If topology_type = periodic:
- micro_type = "repetition_unit"
- Find each individual completed repetition, cycle, strike, or stretch.
- start_time = initiation of one repetition cycle
- end_time = completion of that same repetition cycle, usually when the body/equipment \
returns to its resting or starting position
- Ignore idle pauses between repetitions
- IMPORTANT: post_state may be similar or identical to pre_state if the repetition \
returns to the starting posture

General rules:
1. Typical duration is 2–6 seconds, but use shorter or longer spans if the unit is \
clearly visible.
2. Allow gaps between micro-actions.
3. Merge uninterrupted motion belonging to the same single repetition or same single \
state change.
4. Do not force full coverage.

Also explain your micro-action splitting criterion in one sentence (micro_split_criterion): \
what level of granularity you chose and why. Focus on the splitting logic \
(e.g., "broke down by individual state-changing operations where material visibly \
transforms" or "each full repetition cycle from start to return position"), \
NOT a summary of the actions found.

For each micro-action, provide:
- action_id: Sequential integer starting from 1.
- start_time / end_time: Timestamps in integer seconds (absolute within the full video).
- sub_action: A complete action phrase (5–15 words) describing the specific \
physical interaction, the object(s) involved, and the resulting change. \
One-or-two-word labels like "pouring" or "cutting" are NOT acceptable. \
  Good: "Pour measured flour from the bag into the steel mixing bowl" \
  Bad: "Pour flour"
- pre_state: An EXPLICIT, visually specific description of the scene BEFORE \
the interaction — mention the exact objects, their positions, and appearance. \
  Good: "A sealed bag of flour sits on the counter next to an empty mixing bowl" \
  Bad: "Empty container with prepared surface"
- post_state: An EXPLICIT, visually specific description of the scene AFTER \
the interaction — describe the new positions, appearance, and state of objects. \
  Good: "White flour fills the bottom of the mixing bowl; the bag sits open beside it" \
  Bad: "Material A distributed across the container surface"

Output JSON:
{{
  "micro_type": "state_change | repetition_unit",
  "micro_split_criterion": "<one sentence explaining the granularity logic>",
  "grounding_results": [
    {{
      "action_id": 1,
      "start_time": 42,
      "end_time": 47,
      "sub_action": "Pour measured flour from the bag into the steel mixing bowl",
      "pre_state": "A sealed bag of flour sits on the counter next to an empty mixing bowl",
      "post_state": "White flour fills the bottom of the mixing bowl; the bag sits open beside it"
    }}
  ]
}}"""


def get_level3_prompt(
    event_start_sec: int,
    event_end_sec: int,
    action_query: str,
    topology_type: str = "procedural",
) -> str:
    """
    Build the Level 3 (Topology-Aware Micro Grounding) user-turn prompt.

    Args:
        event_start_sec: Start of the clip (seconds).
        event_end_sec: End of the clip (seconds).
        action_query: The event/phase description to ground into micro-actions.
        topology_type: Temporal topology ("procedural" or "periodic").
    """
    return _LEVEL3_BASE.format(
        clip_start=event_start_sec,
        clip_end=event_end_sec,
        action_query=action_query,
        topology_type=topology_type,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Level 3 Query: Query-conditioned Atomic Grounding (查询式动作级 grounding)
# Input:  an event video clip (possibly with padding) + ordered list of action captions
# Output: start/end times for each action, in the given query order
# ─────────────────────────────────────────────────────────────────────────────
_LEVEL3_QUERY_BASE = """\
You are given a {duration}s video clip and a numbered list of actions to locate. \
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


# ─────────────────────────────────────────────────────────────────────────────
# Level 3 Seg: Atomic Action Segmentation (无查询分割版)
# Input:  an event video clip (possibly with padding)
# Output: time segments for all atomic actions (no text queries given)
# ─────────────────────────────────────────────────────────────────────────────
_LEVEL3_SEG_BASE = """\
You are given a {duration}s video clip. \
Detect all atomic actions in this clip.

Granularity guide — an atomic action is the finest annotation level:
- TOO COARSE (event-level): a multi-step workflow spanning 10–60s. \
If an action covers multiple distinct state changes, split it.
- CORRECT (atomic action): a single, discrete physical interaction (typically \
2–6s) where one object undergoes one visible state change. The start is \
the onset of the physical interaction; the end is when the new state is \
visually established. \
Examples: cutting material, pouring into a container, attaching a part, \
flipping a piece, applying adhesive.
- TOO FINE (sub-atomic): a partial body movement that does not produce a \
complete state change (reaching, gripping, adjusting posture). \
Do NOT annotate these.

Rules:
- Allow gaps between actions — do not force full coverage.
- Skip idle waiting, narration, tool pickup without progress, or reactions.
- Merge uninterrupted motion belonging to one single state change.

Output the start and end time (integer seconds, 0-based) for each action in chronological order:
<events>[[start_time, end_time], ...]</events>

Example: <events>[[3, 7], [10, 14], [16, 22]]</events>"""


def get_level3_seg_prompt(duration: int) -> str:
    """Training prompt for Level 3 segmentation (no queries, detect all atomic actions)."""
    return _LEVEL3_SEG_BASE.format(duration=duration)


# ─────────────────────────────────────────────────────────────────────────────
# Hint-Aware Training Prompts
# Append criterion/hint text to each level's training prompt.
# Hint fields are generated by rewrite_criteria_hints.py from VLM criteria.
# ─────────────────────────────────────────────────────────────────────────────

def get_level1_train_prompt_with_hint(duration: int, hint: str) -> str:
    """L1 training prompt with global_phase_hint appended."""
    base = get_level1_train_prompt_temporal(duration)
    return base + f"\n\nHint: {hint}"


def get_level2_train_prompt_with_hint(duration: int, hint: str) -> str:
    """L2 training prompt with event_split_hint appended."""
    base = get_level2_train_prompt(duration)
    return base + f"\n\nHint: {hint}"


def get_level3_seg_prompt_with_hint(duration: int, hint: str) -> str:
    """L3 seg training prompt with micro_split_hint appended."""
    base = get_level3_seg_prompt(duration)
    return base + f"\n\nHint: {hint}"
# ─────────────────────────────────────────────────────────────────────────────
# Level 2 Check: Event Granularity Review (活动级审核)
# Input:  frames within an L1 phase + existing L2 event annotations
# Output: reviewed events with verdicts, corrections, and supplements
# ─────────────────────────────────────────────────────────────────────────────
_LEVEL2_CHECK_BASE = """\
You are a quality reviewer for activity event annotations. You are viewing \
frames from a macro phase ({phase_start}s to {phase_end}s). \
Phase: "{phase_name}". Summary: "{narrative_summary}"

Below are the EXISTING L2 event annotations for this phase:
{existing_annotations}

Your task: REVIEW each existing event AND identify any MISSING events.

GRANULARITY SPECTRUM — use this to judge every annotation:

This is a LEVEL 2 annotation review. Level 2 events sit in the middle of a \
three-level hierarchy:

  L1 Phase (ABOVE — too coarse for L2):
    A broad activity stage spanning the entire phase you are reviewing. \
If an "event" essentially restates or summarizes the whole phase, it is \
NOT a valid L2 event — it belongs at L1.
    Example of TOO COARSE: "Prepare all materials" when the phase IS \
"Material Preparation".

  L2 Event (THIS LEVEL — correct granularity):
    A multi-second, goal-directed workflow that transforms materials/objects \
or completes a meaningful process sub-goal. It typically involves a sequence \
of physical interactions unified by a single intent.
    Duration guide: typically 10-60 seconds.
    Examples: "Assemble components by fitting and securing parts", \
"Process materials until target state reached", "Shape raw material into desired form".

  L3 Atomic Action (BELOW — too fine for L2):
    A single momentary physical interaction (2-6s) that changes one object's \
state once. If an annotation describes a single hand motion, a single pour, \
or a single cut stroke, it is TOO FINE for L2.
    Examples of TOO FINE: "Place part onto surface", "Pick up tool", \
"Pour liquid into container", "Flip one piece".

REVIEW CRITERIA:

1. [Granularity — Not Too Coarse]: Does the event describe something more \
specific than the phase itself? An event that merely paraphrases the phase \
name/summary should be REMOVED or REVISED to be more specific.

2. [Granularity — Not Too Fine]: Does the event encompass a multi-step workflow \
rather than a single atomic motion? If it describes only one brief physical \
interaction (< 5 seconds, single object state change), it should be MERGED \
with adjacent actions into a proper event, or REMOVED.

3. [Temporal Accuracy]: Do start_time/end_time match the visible activity? \
The event should start when the first contributing action begins and end \
when the goal is achieved.

4. [Description Quality]: Does the instruction clearly convey the goal \
being accomplished? It should describe WHAT is being achieved, not just \
the physical motion.

5. [Activity Relevance]: Does the event involve actual physical object/material \
transformation? Exclude narration, idle waiting, tool-only movements, \
non-activity content, or reactions without physical manipulation.

6. [Temporal Overlap]: Do multiple events cover the same time span with the \
same intent? If so, the duplicate should be removed.

7. [Completeness]: Are there any visible activity workflows in the frames that \
are NOT covered by existing annotations?

For each existing event, output a verdict:
- "keep": Event is correct as-is.
- "revise": Event has issues — provide corrected fields.
- "remove": Event is invalid (wrong granularity, not relevant, or duplicate).

Then list any MISSING events.

Output JSON:
{{
  "reviews": [
    {{
      "event_id": 1,
      "verdict": "keep"
    }},
    {{
      "event_id": 2,
      "verdict": "revise",
      "issue": "too_fine|too_coarse|bad_boundary|bad_description|overlap",
      "revised": {{
        "start_time": 35,
        "end_time": 58,
        "instruction": "Corrected event description",
        "visual_keywords": ["keyword1", "keyword2", "keyword3"]
      }}
    }},
    {{
      "event_id": 3,
      "verdict": "remove",
      "issue": "too_fine|too_coarse|not_relevant|duplicate",
      "reason": "Brief explanation of why this event is removed"
    }}
  ],
  "supplements": [
    {{
      "start_time": 70,
      "end_time": 90,
      "instruction": "Description of missed activity event",
      "visual_keywords": ["keyword1", "keyword2"]
    }}
  ]
}}

IMPORTANT:
- You MUST review every existing event by event_id.
- "supplements" can be an empty list if nothing is missing.
- Do NOT invent events not visible in the provided frames.
- Always include the "issue" field for revise/remove verdicts.
- Be strict on granularity: single atomic motions do NOT belong at L2."""


def get_level2_check_prompt(
    phase_start_sec: int,
    phase_end_sec: int,
    phase_name: str,
    narrative_summary: str,
    existing_events: list[dict],
) -> str:
    """
    Build the Level 2 Check (Quality Judge & Supplement) user-turn prompt.

    Args:
        phase_start_sec: Start of the L1 macro phase (seconds).
        phase_end_sec: End of the L1 macro phase (seconds).
        phase_name: Name of the macro phase from L1.
        narrative_summary: L1 narrative summary for context.
        existing_events: List of existing event dicts for this phase.
    """
    import json as _json
    display_events = []
    for ev in existing_events:
        display_events.append({
            "event_id": ev.get("event_id"),
            "start_time": ev.get("start_time"),
            "end_time": ev.get("end_time"),
            "instruction": ev.get("instruction"),
            "visual_keywords": ev.get("visual_keywords"),
        })
    annotations_str = _json.dumps(display_events, ensure_ascii=False, indent=2)

    return _LEVEL2_CHECK_BASE.format(
        phase_start=phase_start_sec,
        phase_end=phase_end_sec,
        phase_name=phase_name,
        narrative_summary=narrative_summary,
        existing_annotations=annotations_str,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Level 3 Check: Atomic Action Granularity Review (动作级审核)
# Input:  frames within an L2 event clip + existing L3 annotations
# Output: reviewed results with verdicts, corrections, and supplemented actions
# ─────────────────────────────────────────────────────────────────────────────
_LEVEL3_CHECK_BASE = """\
You are a quality reviewer for atomic action annotations. You are viewing \
frames from an event clip ({event_start}s to {event_end}s). \
The event is: "{action_query}"
Micro-action type: {micro_type}
Original splitting criterion: "{micro_split_criterion}"

Below are the EXISTING L3 atomic action annotations for this event:
{existing_annotations}

Your task: REVIEW each existing annotation AND identify any MISSING atomic actions.

GRANULARITY SPECTRUM — use this to judge every annotation:

This is a LEVEL 3 annotation review. Level 3 atomic actions are the finest \
level in a three-level hierarchy:

  L2 Event (ABOVE — too coarse for L3):
    A goal-directed workflow spanning 10-60 seconds with multiple \
physical steps. If an annotation's sub_action essentially restates the \
parent event instruction, it is NOT a valid L3 action — it belongs at L2.
    Example of TOO COARSE: "Assemble the components" when the parent event IS \
"Assemble components by fitting and securing parts".

  L3 Atomic Action (THIS LEVEL — correct granularity):
    A single, discrete physical interaction (2-6 seconds) where exactly ONE \
object undergoes ONE irreversible visual state change. The start is the \
moment of physical contact or the onset of the transformation; the end is \
when the new state is visually established.
    Examples: "Transfer material into the container", \
"Flip the piece with a tool", "Apply adhesive along the edge".

  Sub-atomic motion (BELOW — too fine for L3):
    A partial body movement that does not by itself produce a complete object \
state change. Reaching, gripping, lifting a tool, repositioning hands, or \
adjusting posture are NOT valid L3 annotations.
    Examples of TOO FINE: "Reach for the tool", "Lift hand from the surface", \
"Adjust grip on handle".

REVIEW CRITERIA:

1. [Granularity — Not Too Coarse]: Does the sub_action describe a SINGLE \
physical state change, not a multi-step workflow? If it covers multiple \
distinct state changes, it should be SPLIT (revise into one, supplement \
the others).

2. [Granularity — Not Too Fine]: Does the sub_action produce a complete, \
visible object state change? If it is merely a hand/body motion without \
any object transformation, it should be REMOVED.

3. [Temporal Accuracy]: Does start_time/end_time match what is visible? \
start = physical contact or transformation onset. end = new state established. \
Typical duration: 2-6 seconds.

4. [State Description Quality]: Are pre_state and post_state specific, concrete, \
and visually verifiable? Vague descriptions ("task done", "materials on \
table") are insufficient — they must describe exact visual appearance.

5. [Activity Relevance]: Does the sub_action involve real physical state change \
of objects, materials, or workspace contents? Exclude pure hand movements, \
tool pickups, posture adjustments, narration, or idle frames.

6. [Boundary Compliance]: start_time >= {event_start} and end_time <= {event_end}.

7. [Completeness]: Are there visible atomic state changes in the frames not \
covered by existing annotations?

For each existing annotation, output a verdict:
- "keep": Annotation is correct as-is.
- "revise": Annotation has issues — provide corrected fields.
- "remove": Annotation is invalid (no real state change, wrong granularity, or duplicate).

Then list any MISSING actions.

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
      "issue": "too_fine|too_coarse|bad_boundary|bad_state_desc|overlap",
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
      "issue": "too_fine|too_coarse|not_relevant|duplicate",
      "reason": "Brief explanation"
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
- Do NOT invent actions not visible in the provided frames.
- Always include the "issue" field for revise/remove verdicts.
- Be strict: vague, non-physical, or multi-step annotations should be revised or removed."""


def get_level3_check_prompt(
    event_start_sec: int,
    event_end_sec: int,
    action_query: str,
    existing_results: list[dict],
    micro_type: str = "state_change",
    micro_split_criterion: str = "",
) -> str:
    """
    Build the Level 3 Check (Quality Judge & Supplement) user-turn prompt.

    Args:
        event_start_sec: Start of the L2 event clip (seconds).
        event_end_sec: End of the L2 event clip (seconds).
        action_query: The L2 event instruction.
        existing_results: List of existing grounding_results dicts for this event.
        micro_type: The micro-action type (state_change or repetition_unit).
        micro_split_criterion: The original splitting criterion from annotation.
    """
    import json as _json
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
        micro_type=micro_type,
        micro_split_criterion=micro_split_criterion,
        existing_annotations=annotations_str,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Leaf-Node Check: Per-leaf L3 review + parent boundary shrinkage
# Input:  frames within a leaf node (event or eventless phase) + existing L3
# Output: L3 verdicts + shrunk parent boundaries
# ─────────────────────────────────────────────────────────────────────────────
_LEAF_CHECK_BASE = """\
You are a quality reviewer for hierarchical video annotations. You are viewing \
frames from a {parent_type} spanning [{parent_start}s – {parent_end}s].

Parent {parent_type}: "{parent_name}"
Micro-action type: {micro_type}
Original splitting criterion: "{micro_split_criterion}"

Below are the EXISTING L3 atomic action annotations within this {parent_type}:
{existing_annotations}

You have THREE tasks:

## TASK 1 — L3 MICRO-ACTION REVIEW (Granularity & Completeness)

Review each existing L3 annotation AND identify any MISSING atomic actions.

GRANULARITY SPECTRUM — use this to judge every annotation:

  Parent {parent_type} (ABOVE — too coarse for L3):
    If an annotation's sub_action essentially restates the parent description, \
it is NOT a valid L3 action — it belongs at the parent level.

  L3 Atomic Action (THIS LEVEL — correct granularity):
    A single, discrete physical interaction (2–6 seconds) where exactly ONE \
object undergoes ONE visible state change. The start is the moment of physical \
contact or the onset of the transformation; the end is when the new state is \
visually established.
    Examples: "Transfer material into the container", \
"Flip the piece with a tool", "Apply adhesive along the edge".

  Sub-atomic motion (BELOW — too fine for L3):
    A partial body movement that does not by itself produce a complete object \
state change. Reaching, gripping, lifting a tool, repositioning hands, or \
adjusting posture are NOT valid L3 annotations.
    Examples of TOO FINE: "Reach for the tool", "Lift hand from the surface", \
"Adjust grip on handle".

L3 REVIEW CRITERIA:

1. [Granularity — Not Too Coarse]: Does the sub_action describe a SINGLE \
physical state change, not a multi-step workflow? If it covers multiple \
distinct state changes, it should be SPLIT (revise into one, supplement \
the others).

2. [Granularity — Not Too Fine]: Does the sub_action produce a complete, \
visible object state change? If it is merely a hand/body motion without \
any object transformation, it should be REMOVED.

3. [Temporal Accuracy]: Does start_time/end_time match what is visible? \
start = physical contact or transformation onset. end = new state established. \
Typical duration: 2-6 seconds.

4. [State Description Quality]: Are pre_state and post_state specific, concrete, \
and visually verifiable? Vague descriptions ("task done", "materials on table") \
are insufficient — they must describe exact visual appearance.

5. [Activity Relevance]: Does the sub_action involve real physical state change \
of objects, materials, or workspace contents? Exclude pure hand movements, \
tool pickups, posture adjustments, narration, or idle frames.

6. [Completeness]: Are there visible atomic state changes in the frames not \
covered by existing annotations?

For each existing annotation, output a verdict:
- "keep": Annotation is correct as-is.
- "revise": Annotation has issues — provide corrected fields.
- "remove": Annotation is invalid (no real state change, wrong granularity, or duplicate).

Then list any MISSING actions.

## TASK 2 — PARENT BOUNDARY SHRINKAGE (Critical)

Examine the frames near the START and END of the parent's time range.

Determine: does the parent's [{parent_start}s – {parent_end}s] contain "dead zones" \
— stretches of time at the beginning or end where NO meaningful physical action occurs?

Rules:
- If actual physical activity starts AFTER {parent_start}s (e.g., idle, talking, \
static frames at the beginning), output a tighter shrunk_start.
- If actual physical activity ends BEFORE {parent_end}s (e.g., idle tail, \
static frames at the end), output a tighter shrunk_end.
- The shrunk boundaries should tightly wrap the FIRST and LAST visible physical \
actions (aligned with L3 annotations when possible).
- If the boundaries are already tight, output shrunk_start = {parent_start} \
and shrunk_end = {parent_end} (no change).
- Shrinkage must preserve ALL kept/revised/supplemented L3 actions within bounds.
- Use integer seconds.

## TASK 3 — TEMPORAL ORDER DISTINGUISHABILITY JUDGMENT

Consider the sequence of L3 micro-actions (after applying your TASK 1 verdicts) in their \
temporal order within this {parent_type}.

Question: If the temporal order of these micro-actions were REVERSED (last action played \
first, first action played last), could a viewer reliably distinguish the forward \
video from the reversed video based purely on visual cues?

Consider these factors:
- Causal dependency: Does each action rely on the visible output of the previous \
action (e.g., material is cut in action 1, then the cut pieces are assembled in action 2)?
- Progressive state change: Do objects visibly evolve in a direction that cannot \
be mistaken for the reverse (e.g., whole → pieces, empty → full, raw → processed)?
- Tool/material availability: Does a tool or material appear only after a prior \
action produces or reveals it?
- Irreversibility: Are the physical transformations clearly one-way (breaking, \
mixing, applying adhesive) vs. reversible (placing/removing, opening/closing)?
- Symmetry: Are the actions largely interchangeable without visual inconsistency?

Output:
- order_distinguishable: true if a viewer CAN reliably tell forward from reversed; \
false if the actions are largely symmetric or interchangeable.
- order_cue: ONE sentence explaining the primary visual cue (or lack thereof).
- order_confidence: Float 0.0–1.0 reflecting how certain you are.

## OUTPUT FORMAT

{{{{
  "l3_reviews": [
    {{{{
      "action_id": 1,
      "verdict": "keep"
    }}}},
    {{{{
      "action_id": 2,
      "verdict": "revise",
      "issue": "too_fine|too_coarse|bad_boundary|bad_state_desc|overlap",
      "revised": {{{{
        "start_time": 45,
        "end_time": 49,
        "sub_action": "Corrected description",
        "pre_state": "Specific pre-state",
        "post_state": "Specific post-state"
      }}}}
    }}}},
    {{{{
      "action_id": 3,
      "verdict": "remove",
      "issue": "too_fine|too_coarse|not_relevant|duplicate",
      "reason": "Brief explanation"
    }}}}
  ],
  "l3_supplements": [
    {{{{
      "start_time": 55,
      "end_time": 59,
      "sub_action": "Description of missed action",
      "pre_state": "Pre-state",
      "post_state": "Post-state"
    }}}}
  ],
  "shrunk_start": {parent_start},
  "shrunk_end": {parent_end},
  "order_distinguishable": true,
  "order_cue": "Action 1 cuts raw material into pieces that Action 3 then assembles — the assembled state cannot precede the cutting.",
  "order_confidence": 0.9
}}}}

IMPORTANT:
- You MUST review every existing annotation by action_id.
- "l3_supplements" can be an empty list if nothing is missing.
- Do NOT invent actions not visible in the provided frames.
- Always include the "issue" field for revise/remove verdicts.
- shrunk_start/shrunk_end MUST satisfy: shrunk_start <= min(L3 start_times) \
and shrunk_end >= max(L3 end_times) for all kept/revised/supplemented L3 actions.
- Be strict: vague, non-physical, or multi-step annotations should be revised or removed.
- For order distinguishability, judge based on the FINAL set of L3 actions after your review, \
not the original input."""


def get_leaf_check_prompt(
    parent_type: str,
    parent_name: str,
    parent_start: int,
    parent_end: int,
    existing_results: list[dict],
    micro_type: str = "state_change",
    micro_split_criterion: str = "",
) -> str:
    """
    Build the Leaf-Node Check prompt for combined L3 review + parent shrinkage.

    Args:
        parent_type: "event" or "phase".
        parent_name: The event instruction or phase name.
        parent_start: Parent start time in seconds.
        parent_end: Parent end time in seconds.
        existing_results: List of existing L3 grounding_results for this leaf.
        micro_type: The micro-action type (state_change or repetition_unit).
        micro_split_criterion: The original splitting criterion from annotation.
    """
    import json as _json
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

    return _LEAF_CHECK_BASE.format(
        parent_type=parent_type,
        parent_name=parent_name,
        parent_start=parent_start,
        parent_end=parent_end,
        micro_type=micro_type,
        micro_split_criterion=micro_split_criterion,
        existing_annotations=annotations_str,
    )


# ─────────────────────────────────────────────────────────────────────────────
# L2 Shrink Check: Per-phase L2 review + L1 boundary shrinkage + order judge
# Input:  frames within an L1 phase + existing L2 event annotations
# Output: L2 verdicts, L1 shrunk boundaries, order distinguishability
# ─────────────────────────────────────────────────────────────────────────────
_L2_SHRINK_CHECK_BASE = """\
You are a quality reviewer for hierarchical video annotations. You are viewing \
frames from a macro phase spanning [{phase_start}s – {phase_end}s].

Phase: "{phase_name}"
Summary: "{narrative_summary}"

Below are the EXISTING L2 event annotations within this phase:
{existing_annotations}

You have THREE tasks:

## TASK 1 — L2 EVENT REVIEW (Granularity & Completeness)

Review each existing L2 event AND identify any MISSING events.

GRANULARITY SPECTRUM — use this to judge every annotation:

  L1 Phase (ABOVE — too coarse for L2):
    A broad activity stage spanning the entire phase you are reviewing. \
If an "event" essentially restates or summarizes the whole phase, it is \
NOT a valid L2 event — it belongs at L1.
    Example of TOO COARSE: "Prepare all materials" when the phase IS \
"Material Preparation".

  L2 Event (THIS LEVEL — correct granularity):
    A multi-second, goal-directed workflow that transforms materials/objects \
or completes a meaningful process sub-goal. It typically involves a sequence \
of physical interactions unified by a single intent.
    Duration guide: typically 10-60 seconds.
    Examples: "Assemble components by fitting and securing parts", \
"Process materials until target state reached", "Shape raw material into desired form".

  L3 Atomic Action (BELOW — too fine for L2):
    A single momentary physical interaction (2-6s) that changes one object's \
state once. If an annotation describes a single hand motion, a single pour, \
or a single cut stroke, it is TOO FINE for L2.
    Examples of TOO FINE: "Place part onto surface", "Pick up tool", \
"Pour liquid into container", "Flip one piece".

L2 REVIEW CRITERIA:

1. [Granularity — Not Too Coarse]: Does the event describe something more \
specific than the phase itself? An event that merely paraphrases the phase \
name/summary should be REMOVED or REVISED to be more specific.

2. [Granularity — Not Too Fine]: Does the event encompass a multi-step workflow \
rather than a single atomic motion? If it describes only one brief physical \
interaction (< 5 seconds, single object state change), it should be MERGED \
with adjacent actions into a proper event, or REMOVED.

3. [Temporal Accuracy]: Do start_time/end_time match the visible activity? \
The event should start when the first contributing action begins and end \
when the goal is achieved.

4. [Description Quality]: Does the instruction clearly convey the goal \
being accomplished in 8–20 words? It should describe WHAT is being achieved, \
with WHICH objects/materials, toward WHAT outcome. Vague or overly brief \
instructions (< 5 words) must be REVISED.

5. [Activity Relevance]: Does the event involve actual physical object/material \
transformation? Exclude narration, idle waiting, tool-only movements, \
non-activity content, or reactions without physical manipulation.

6. [Temporal Overlap]: Do multiple events cover the same time span with the \
same intent? If so, the duplicate should be removed.

7. [Completeness]: Are there any visible activity workflows in the frames that \
are NOT covered by existing annotations?

For each existing event, output a verdict:
- "keep": Event is correct as-is.
- "revise": Event has issues — provide corrected fields.
- "remove": Event is invalid (wrong granularity, not relevant, or duplicate).

Then list any MISSING events.

## TASK 2 — L1 PHASE BOUNDARY SHRINKAGE (Critical)

Examine the frames near the START and END of the phase's time range.

Determine: does the phase's [{phase_start}s – {phase_end}s] contain "dead zones" \
— stretches of time at the beginning or end where NO meaningful physical action occurs?

Rules:
- If actual physical activity starts AFTER {phase_start}s (e.g., idle, talking, \
static frames at the beginning), output a tighter shrunk_start.
- If actual physical activity ends BEFORE {phase_end}s (e.g., idle tail, \
static frames at the end), output a tighter shrunk_end.
- The shrunk boundaries should tightly wrap the FIRST and LAST visible physical \
actions (aligned with L2 event boundaries when possible).
- If the boundaries are already tight, output shrunk_start = {phase_start} \
and shrunk_end = {phase_end} (no change).
- Shrinkage must preserve ALL kept/revised/supplemented L2 events within bounds.
- Use integer seconds.

## TASK 3 — TEMPORAL ORDER DISTINGUISHABILITY JUDGMENT

Consider the sequence of L2 events (after applying your TASK 1 verdicts) in their \
temporal order within this phase.

Question: If the temporal order of these events were REVERSED (last event played \
first, first event played last), could a viewer reliably distinguish the forward \
video from the reversed video based purely on visual cues?

Consider these factors:
- Causal dependency: Does each event rely on the visible output of the previous \
event (e.g., material is prepared in event 1, then assembled in event 2)?
- Progressive state change: Do objects visibly evolve in a direction that cannot \
be mistaken for the reverse (e.g., raw → cooked, scattered → organized)?
- Tool/material availability: Does a tool or material appear only after a prior \
event produces or reveals it?
- Symmetry: Are the events largely interchangeable without visual inconsistency?

Output:
- order_distinguishable: true if a viewer CAN reliably tell forward from reversed; \
false if the events are largely symmetric or interchangeable.
- order_cue: ONE sentence explaining the primary visual cue (or lack thereof).
- order_confidence: Float 0.0–1.0 reflecting how certain you are.

## OUTPUT FORMAT

{{{{
  "event_reviews": [
    {{{{
      "event_id": 1,
      "verdict": "keep"
    }}}},
    {{{{
      "event_id": 2,
      "verdict": "revise",
      "issue": "too_fine|too_coarse|bad_boundary|bad_description|overlap",
      "revised": {{{{
        "start_time": 35,
        "end_time": 58,
        "instruction": "Corrected event description with specific objects and outcome",
        "visual_keywords": ["keyword1", "keyword2"]
      }}}}
    }}}},
    {{{{
      "event_id": 3,
      "verdict": "remove",
      "issue": "too_fine|too_coarse|not_relevant|duplicate",
      "reason": "Brief explanation"
    }}}}
  ],
  "event_supplements": [
    {{{{
      "start_time": 70,
      "end_time": 90,
      "instruction": "Description of missed activity event",
      "visual_keywords": ["keyword1", "keyword2"]
    }}}}
  ],
  "shrunk_start": {phase_start},
  "shrunk_end": {phase_end},
  "order_distinguishable": true,
  "order_cue": "Event 1 produces raw pieces that Event 2 then assembles — the assembled state cannot precede the raw state.",
  "order_confidence": 0.9
}}}}

IMPORTANT:
- You MUST review every existing event by event_id.
- "event_supplements" can be an empty list if nothing is missing.
- Do NOT invent events not visible in the provided frames.
- Always include the "issue" field for revise/remove verdicts.
- shrunk_start/shrunk_end MUST satisfy: shrunk_start <= min(L2 start_times) \
and shrunk_end >= max(L2 end_times) for all kept/revised/supplemented L2 events.
- Be strict: vague, non-physical, or wrong-granularity annotations should be revised or removed.
- For order distinguishability, judge based on the FINAL set of events after your review, \
not the original input."""


def get_l2_shrink_check_prompt(
    phase_name: str,
    phase_start: int,
    phase_end: int,
    narrative_summary: str,
    existing_events: list[dict],
) -> str:
    """
    Build the L2 Shrink Check prompt: L2 event review + L1 boundary shrinkage
    + temporal order distinguishability.

    Args:
        phase_name: The L1 phase name.
        phase_start: Phase start time in seconds.
        phase_end: Phase end time in seconds.
        narrative_summary: The L1 phase narrative summary.
        existing_events: List of L2 events within this phase.
    """
    import json as _json
    display_events = []
    for ev in existing_events:
        display_events.append({
            "event_id": ev.get("event_id"),
            "start_time": ev.get("start_time"),
            "end_time": ev.get("end_time"),
            "instruction": ev.get("instruction"),
            "visual_keywords": ev.get("visual_keywords"),
        })
    annotations_str = _json.dumps(display_events, ensure_ascii=False, indent=2)

    return _L2_SHRINK_CHECK_BASE.format(
        phase_name=phase_name,
        phase_start=phase_start,
        phase_end=phase_end,
        narrative_summary=narrative_summary,
        existing_annotations=annotations_str,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Merged L1+L2 Check: Simultaneous Phase + Event Quality Review
# Input:  full video frames (1fps) + existing L1 phases + L2 events
# Output: reviewed phases with verdicts, reviewed events with verdicts
# ─────────────────────────────────────────────────────────────────────────────
_MERGED_CHECK_BASE = """\
You are a quality reviewer for hierarchical video annotations. You are viewing \
a {duration}s video clip (timestamps 0 to {duration}) with {n_frames} frames.

Existing annotation summary: "{summary}"
Topology: {topology_type} (confidence: {topology_confidence})
Original phase-splitting criterion: "{global_phase_criterion}"

Below are the EXISTING annotations — L1 macro phases with their nested L2 events:
{existing_annotations}

Your task: REVIEW both L1 phases and L2 events simultaneously, then identify \
any MISSING phases or events.

## L1 PHASE REVIEW

GRANULARITY SPECTRUM for L1 phases:

  L1 Phase (THIS LEVEL — correct granularity):
    A broad stage of activity organized by overall intent. Each phase should \
represent a distinct functional stage in the video, typically 30-120 seconds. \
Phases do NOT need to cover the entire video — gaps are expected.
    Examples: "Material Preparation", "Assembly", "Finishing".

  L2 Event (BELOW — too fine for L1):
    If a phase describes a single specific action rather than a broad stage, \
it is TOO FINE for L1.

L1 REVIEW CRITERIA:
1. [Phase Boundaries]: Do start_time/end_time match the visible stage transitions?
2. [Phase Granularity — Not Too Broad]: Does a phase contain clearly distinct \
stages that should be separate phases?
3. [Phase Granularity — Not Too Fine]: Is a phase too narrow — describing a \
single action rather than a broad stage? Should it be merged with adjacent phases?
4. [Phase Naming]: Does phase_name accurately describe the intent of this stage?
5. [Camera Cut Independence]: Phases should NOT be split by camera cuts — only \
by semantic intent changes.
6. [Completeness]: Are there visible activity stages NOT covered by any phase?

## L2 EVENT REVIEW (per phase)

GRANULARITY SPECTRUM for L2 events:

  L1 Phase (ABOVE — too coarse for L2):
    If an event essentially restates the whole phase, it belongs at L1, not L2.

  L2 Event (THIS LEVEL — correct granularity):
    A multi-second, goal-directed workflow (typically 10-60s) that completes a \
meaningful process sub-goal. It involves a sequence of physical interactions \
unified by a single intent.

  L3 Atomic Action (BELOW — too fine for L2):
    A single momentary physical interaction (2-6s). Single hand motions, single \
pours, or single cut strokes are TOO FINE for L2.

L2 REVIEW CRITERIA:
1. [Granularity — Not Too Coarse]: Does the event describe something more \
specific than the phase itself?
2. [Granularity — Not Too Fine]: Does the event encompass a multi-step workflow?
3. [Temporal Accuracy]: Do start_time/end_time match the visible activity?
4. [Description Quality]: Does the instruction clearly convey the goal?
5. [Activity Relevance]: Does the event involve actual physical object/material \
transformation?
6. [Temporal Overlap]: Do multiple events cover the same time span?
7. [Completeness]: Are there visible workflows NOT covered by existing events?

## OUTPUT FORMAT

Output JSON with BOTH phase-level and event-level reviews:
{{
  "phase_reviews": [
    {{
      "phase_id": 1,
      "verdict": "keep"
    }},
    {{
      "phase_id": 2,
      "verdict": "revise",
      "issue": "bad_boundary|too_broad|too_fine|bad_name",
      "revised": {{
        "start_time": 35,
        "end_time": 90,
        "phase_name": "Corrected phase name",
        "narrative_summary": "Corrected summary"
      }}
    }},
    {{
      "phase_id": 3,
      "verdict": "remove",
      "issue": "too_fine|not_relevant|duplicate",
      "reason": "Brief explanation"
    }}
  ],
  "phase_supplements": [
    {{
      "start_time": 120,
      "end_time": 170,
      "phase_name": "Missed stage name",
      "narrative_summary": "Description of the missed stage"
    }}
  ],
  "event_reviews": [
    {{
      "event_id": 1,
      "verdict": "keep"
    }},
    {{
      "event_id": 2,
      "verdict": "revise",
      "issue": "too_fine|too_coarse|bad_boundary|bad_description|overlap",
      "revised": {{
        "start_time": 40,
        "end_time": 58,
        "instruction": "Corrected event description",
        "visual_keywords": ["keyword1", "keyword2"]
      }}
    }},
    {{
      "event_id": 3,
      "verdict": "remove",
      "issue": "too_fine|too_coarse|not_relevant|duplicate",
      "reason": "Brief explanation"
    }}
  ],
  "event_supplements": [
    {{
      "start_time": 70,
      "end_time": 90,
      "instruction": "Description of missed activity event",
      "visual_keywords": ["keyword1", "keyword2"],
      "parent_phase_id": 1
    }}
  ]
}}

IMPORTANT:
- You MUST review every existing phase by phase_id.
- You MUST review every existing event by event_id.
- "phase_supplements" and "event_supplements" can be empty lists.
- Do NOT invent phases or events not visible in the provided frames.
- Always include the "issue" field for revise/remove verdicts.
- Each supplemented event MUST include "parent_phase_id" linking to an existing \
or supplemented phase.
- Be strict on granularity at both levels."""


def get_merged_check_prompt(
    n_frames: int,
    duration_sec: int,
    summary: str,
    topology_type: str,
    topology_confidence: float,
    l1_phases: list[dict],
    l2_events: list[dict],
    global_phase_criterion: str = "",
) -> str:
    """
    Build the merged L1+L2 check (quality review) prompt.

    Presents the full video context with existing L1 phases and nested L2 events,
    mirroring the merged annotation structure.
    """
    import json as _json

    # Build nested display: phases with their events
    display_phases = []
    for phase in l1_phases:
        phase_id = phase.get("phase_id")
        phase_events = [
            {
                "event_id": e.get("event_id"),
                "start_time": e.get("start_time"),
                "end_time": e.get("end_time"),
                "instruction": e.get("instruction"),
                "visual_keywords": e.get("visual_keywords"),
            }
            for e in l2_events
            if e.get("parent_phase_id") == phase_id
        ]
        display_phases.append({
            "phase_id": phase_id,
            "start_time": phase.get("start_time"),
            "end_time": phase.get("end_time"),
            "phase_name": phase.get("phase_name"),
            "narrative_summary": phase.get("narrative_summary"),
            "event_split_criterion": phase.get("event_split_criterion", ""),
            "events": phase_events,
        })
    annotations_str = _json.dumps(display_phases, ensure_ascii=False, indent=2)

    return _MERGED_CHECK_BASE.format(
        n_frames=n_frames,
        duration=duration_sec,
        summary=summary,
        topology_type=topology_type,
        topology_confidence=topology_confidence,
        global_phase_criterion=global_phase_criterion,
        existing_annotations=annotations_str,
    )
