"""
prompts_gseg.py — Prompts for the Grounding+Segmentation (Segment-as-CoT) pipeline.

Two prompt families:
  1. ANNOTATION prompts  — sent to the VLM teacher (Gemini / GPT-4o) to generate
                           abstract queries + ground-truth segmentations.
  2. TRAINING prompts    — used by build_gseg_data.py to format student-model inputs.
"""

# ─────────────────────────────────────────────────────────────────────────────
# System prompt (shared by all annotation calls)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert video analyst specialising in temporal reasoning.
Your job is to design challenging reasoning tasks where a student model
must LOCATE relevant content in a video and SEGMENT it into meaningful
temporal units — purely by watching the video and following an abstract query.

Core principles:
  1. The query must force grounding + segmentation as a reasoning process.
  2. The query must NOT leak timestamps, segment counts, or specific action names.
  3. Gaps between segments are expected — not every second must be covered.
  4. Output strict JSON without additional commentary.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Annotation prompt — VLM generates (query + ground-truth) in one call
# ─────────────────────────────────────────────────────────────────────────────

_ANNOTATION_BASE = """\
You are given {n_frames} frames uniformly sampled from a {duration}s video.
Each frame is labelled with its absolute timestamp in seconds (e.g. [42s]).

## Your Task

1. **Watch carefully** — understand the content, temporal structure, and any
   irrelevant / noisy portions (intro, outro, idle chatter, B-roll, etc.).
2. **Choose a Query Style** (see below) that best fits this video.
3. **Write an abstract reasoning query** that a student model must answer by
   producing precise temporal segments.
4. **Provide the ground-truth answer** — the segments the student should output.

────────────────────────────────────────────────────
## Query Styles — pick the MOST appropriate one
────────────────────────────────────────────────────

### A  GROUNDING + SEGMENTATION
For videos that mix relevant activity with unrelated content.
The student must first *locate* the relevant portion, then segment within it.

Example query pattern:
  "Locate the continuous [abstract activity] in this video and segment
   each distinct [abstract unit] by [abstract criterion]."

### B  FULL SEGMENTATION
For tutorial / instructional / how-to videos where essentially ALL content
is relevant and procedurally structured.

Example query pattern:
  "Segment this procedural demonstration into all its distinct operational
   steps, ordered chronologically."

### C  CYCLIC IDENTIFICATION
For periodic / repetitive content (exercise reps, assembly-line cycles, …).

Example query pattern:
  "Identify all complete repetitions of the core movement pattern and
   segment each repetition into its constituent phases."

### D  CAUSAL CHAIN
For process / transformation videos where a visible end-state results from
a sequence of preparatory + transformative actions.

Example query pattern:
  "Determine the chronological sequence of actions that produced the
   [abstract end-state], and segment each distinct causal step."

### E  THREAD EXTRACTION
For noisy videos (vlogs, livestreams) with one coherent activity thread
buried in chatter, transitions, and idle moments.

Example query pattern:
  "Extract only the hands-on [abstract activity thread] from this video,
   filtering out dialogue and transitions, and segment by [criterion]."

────────────────────────────────────────
## Query Design Rules (CRITICAL)
────────────────────────────────────────

✅  Describe WHAT to look for using high-level, abstract language.
✅  Specify the CRITERION for splitting (e.g., "by obstacle type",
    "by procedural step", "by ingredient") — but do NOT enumerate the
    concrete items.
✅  Make the query specific enough that a single correct segmentation
    exists (or at most minor boundary variation).

❌  Do NOT mention any specific timestamps or time ranges.
❌  Do NOT state the exact number of segments.
❌  Do NOT name specific actions, objects, or motions that the student
    could simply text-match against frames — use abstract descriptions
    instead (e.g., "the main crafting activity" instead of "sewing the zipper").
❌  Do NOT make the query so vague that multiple fundamentally different
    segmentations would be equally valid.

Note: Some domain awareness is acceptable — e.g., "the stretching routine"
is fine; "the hamstring stretch then the quad stretch then the calf stretch"
is leaking the answer.

────────────────────────────────────────
## Multi-Task: One Video, Multiple Tasks
────────────────────────────────────────

If the video is rich and contains **multiple distinct, unrelated activity
threads** (e.g., a vlog showing cooking, then a band playing, then swimming),
you SHOULD output multiple tasks — one per thread, each using a different
Query Style where appropriate.

If the video has a single coherent theme, output exactly ONE task.

────────────────────────────────────────
## Output JSON Schema
────────────────────────────────────────

Output a JSON **array** of task objects.  Each task follows this schema:

```json
[
  {{
    "video_summary": "<one-sentence factual summary of the entire video>",
    "domain": "<domain label, e.g. fitness, cooking, vlog, pet_training, crafting>",
    "noise_description": "<what irrelevant content exists — null if none>",

    "query_style": "<A|B|C|D|E>",
    "query": "<the abstract reasoning query for the student model>",

    "grounding": {{
      "start_time": <int seconds — start of the relevant portion>,
      "end_time":   <int seconds — end of the relevant portion>,
      "rationale":  "<why this is the relevant portion>"
    }},

    "segments": [
      {{
        "id": 1,
        "start_time": <int seconds>,
        "end_time":   <int seconds>,
        "label":      "<brief factual description of this segment>"
      }}
    ],

    "reasoning_trace": "<step-by-step explanation of how a student should reason to arrive at this segmentation — 3–5 sentences>"
  }}
]
```

Output a JSON array `[{{task1}}, {{task2}}, ...]` even for a single task.
Different tasks targetting the same video MUST have non-overlapping
grounding ranges.

Rules for timestamps:
  - All times are integer seconds, 0-based, within [0, {duration}].
  - Segments must be within the grounding boundaries.
  - Segments must not overlap and should be in chronological order.
  - Gaps between segments are allowed (idle, transitional, or irrelevant frames).
  - For Style B (full seg), grounding.start_time may be 0 and
    grounding.end_time may equal the video duration.
  - Aim for 3–12 segments per task. If the video genuinely has fewer distinct
    units, that is fine.
"""


def get_annotation_prompt(n_frames: int, duration: int) -> str:
    """Return the VLM annotation prompt with video metadata filled in."""
    return _ANNOTATION_BASE.format(n_frames=n_frames, duration=duration)


# ─────────────────────────────────────────────────────────────────────────────
# Training prompts — student model input
# ─────────────────────────────────────────────────────────────────────────────

_TRAINING_BASE = """\
Watch the following video clip carefully:
<video>

{query}

Output the temporal segments as integer-second [start, end] pairs.
Format: <events>[[start, end], ...]</events>
"""

_TRAINING_THINK_BASE = """\
Watch the following video clip carefully:
<video>

{query}

First, think step-by-step about what you observe in the video and how it
relates to the query. Then output the temporal segments.

<think>
(your reasoning here)
</think>
<events>[[start, end], ...]</events>
"""


def get_training_prompt(query: str) -> str:
    """Construct the student-model prompt (no CoT)."""
    return _TRAINING_BASE.format(query=query)


def get_training_prompt_with_think(query: str) -> str:
    """Construct the student-model prompt with <think> CoT."""
    return _TRAINING_THINK_BASE.format(query=query)
