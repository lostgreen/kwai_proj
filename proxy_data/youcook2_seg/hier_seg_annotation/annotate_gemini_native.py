#!/usr/bin/env python3
"""
annotate_gemini_native.py — Full 3-level hierarchical video annotation using
Gemini native API with full video input (no frame extraction needed).

One VLM call → classify + L1 phases + L2 events + L3 sub-actions (all three levels).
Uses the v5 unified prompt (per-paradigm cutting rules) from archetypes.py,
extended with inline L3 micro-action annotation.

Usage:
    export GEMINI_API_KEY="your-api-key"

    python annotate_gemini_native.py \
        --data-path videos.jsonl \
        --save-path output.jsonl \
        --model gemini-2.5-pro \
        --fps 2 \
        --limit 3

Input JSONL format:
    {"video_path": "/path/to/video.mp4", ...}

Requires:
    pip install google-genai
"""

import argparse
import json
import os
import re
import sys
import time
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from google import genai
from google.genai import types
from google.genai.types import (
    HarmBlockThreshold,
    HarmCategory,
    SafetySetting,
)

# Import prompt helpers from archetypes
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from archetypes import (
    SYSTEM_PROMPT,
    PARADIGMS,
    PARADIGM_IDS,
    DOMAIN_L2_TO_L1,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SAFETY_SETTINGS = [
    SafetySetting(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=HarmBlockThreshold.OFF),
    SafetySetting(category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=HarmBlockThreshold.OFF),
    SafetySetting(category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=HarmBlockThreshold.OFF),
    SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=HarmBlockThreshold.OFF),
]

# Augment SYSTEM_PROMPT: ignore audio
SYSTEM_PROMPT_NO_AUDIO = SYSTEM_PROMPT + """

CRITICAL — Ignore Audio:
You MUST completely ignore any audio track in the video. Do NOT use speech, narration, \
music, sound effects, or any audio cues to inform your segmentation, classification, \
or descriptions. Base ALL analysis purely on VISUAL information — what you can SEE \
in the video frames. If you detect speech or narration, do NOT transcribe or reference it. \
Treat the video as if it were muted."""


# ─────────────────────────────────────────────────────────────────────────────
# Build the full 3-level prompt (reuse archetypes.py helpers)
# ─────────────────────────────────────────────────────────────────────────────

def _format_paradigm_table() -> str:
    """Paradigm classification table for Part 1."""
    lines = []
    for pid, cfg in PARADIGMS.items():
        lines.append(
            f"- **{pid}** ({cfg.display_name_en}): {cfg.description}\n"
            f"  Signals: {cfg.classification_signals}\n"
            f"  Typical: {', '.join(cfg.typical_videos[:3])}"
        )
    return "\n".join(lines)


def _format_domain_l2_list() -> str:
    """Domain L2 list for classification."""
    by_l1: dict[str, list[str]] = {}
    for l2, l1 in DOMAIN_L2_TO_L1.items():
        by_l1.setdefault(l1, []).append(l2)
    lines = []
    for l1, l2s in sorted(by_l1.items()):
        lines.append(f"- **{l1}**: {', '.join(sorted(l2s))}")
    return "\n".join(lines)


def _format_paradigm_annotation_table() -> str:
    """Per-paradigm L1/L2 cutting rules for Part 2."""
    topology_tips = {
        "procedural": "Skip intros/outros/idle spans. Focus on task progress milestones.",
        "periodic": "Organize by rhythm/intensity cycles. Rest intervals can be gaps.",
        "sequence": "Follow narrative/topic/scene flow. Intros and transitions ARE valid phases.",
        "observation": "Track subject behavior state changes. Long static spans are valid gaps.",
    }
    blocks = []
    for pid, cfg in PARADIGMS.items():
        l1_ex = "; ".join(f'"{e}"' for e in cfg.l1.examples[:2])
        l2_ex = "; ".join(f'"{e}"' for e in cfg.l2.examples[:2])
        tip = topology_tips.get(cfg.topology, "")
        block = (
            f"### {pid} ({cfg.display_name_en})\n"
            f"- **L1 — {cfg.l1.name}**: {cfg.l1.definition}\n"
            f"  Boundary signals: {cfg.l1.boundary_signals}\n"
            f"  Good: {l1_ex}\n"
            f"  NOT L1: {'; '.join(cfg.l1.anti_examples[:1])}\n"
            f"- **L2 — {cfg.l2.name}**: {cfg.l2.definition}\n"
            f"  Boundary signals: {cfg.l2.boundary_signals}\n"
            f"  Good: {l2_ex}\n"
            f"  NOT L2: {'; '.join(cfg.l2.anti_examples[:1])}\n"
            f"- **Annotation tip**: {tip}"
        )
        blocks.append(block)
    return "\n\n".join(blocks)


def _format_paradigm_l3_table() -> str:
    """Per-paradigm L3 definitions for Part 3."""
    blocks = []
    for pid, cfg in PARADIGMS.items():
        if not cfg.l3.enabled:
            blocks.append(f"- **{pid}**: L3 disabled — do NOT output sub_actions for this paradigm.")
            continue
        l3_ex = "; ".join(f'"{e}"' for e in cfg.l3.examples[:2])
        l3_anti = "; ".join(f'"{e}"' for e in cfg.l3.anti_examples[:1])
        block = (
            f"- **{pid}** — L3 = **{cfg.l3.name}**: {cfg.l3.definition}\n"
            f"  Boundary signals: {cfg.l3.boundary_signals}\n"
            f"  Good: {l3_ex}\n"
            f"  NOT L3: {l3_anti}"
        )
        blocks.append(block)
    return "\n".join(blocks)


def _format_l3_feasibility_ref() -> str:
    """L3 feasibility reference."""
    lines = []
    for pid, cfg in PARADIGMS.items():
        if cfg.l3.enabled:
            lines.append(f"- **{pid}**: L3 = {cfg.l3.name} — {cfg.l3.definition[:80]}...")
        else:
            lines.append(f"- **{pid}**: L3 disabled")
    return "\n".join(lines)


def build_3level_prompt(n_frames: int, duration_sec: int) -> str:
    """Build the full 3-level annotation prompt (L1 + L2 + L3 in one call)."""
    paradigm_ids = ", ".join(sorted(PARADIGM_IDS))

    return f"""\
You are given a {duration_sec}s video clip (timestamps 0 to {duration_sec}) sampled at ~{n_frames} frames.

Your task has THREE parts — all in a SINGLE response:
1. **CLASSIFY** the video (paradigm, domain, feasibility, metadata, caption)
2. **ANNOTATE** L1 phases + L2 events (hierarchical temporal structure)
3. **ANNOTATE** L3 sub-actions (micro-action grounding within each L2 event)

════════════════════════════════════════════════
## PART 1 — CLASSIFICATION
════════════════════════════════════════════════

### 1A. PARADIGM (temporal structure)

Classify into exactly ONE paradigm based on the video's dominant temporal structure:

{_format_paradigm_table()}

If the video does NOT fit any paradigm above:
- **Talk-dominant** (people in conversation, minimal physical action): set feasibility.skip=true, skip_reason="talk_dominant"
- **Ambient/static** (no identifiable subject, no sequential progression): set feasibility.skip=true, skip_reason="ambient_static"

### 1B. DOMAIN (content topic)

Domain is WHAT the video is about — orthogonal to paradigm (HOW it is structured).
Choose domain_l2 from:
{_format_domain_l2_list()}
If none fits, use domain_l2="other".

### 1C. FEASIBILITY

Assess whether this video supports hierarchical temporal annotation.
- **Skip** if: people only talking with no visual action changes; ambient/static footage; \
screen recordings of static content.
- **Annotate** if: there are visually distinct activities, location/scene changes, \
or progressive actions — even if the video also contains talking.

### 1D. VIDEO CAPTION

Write a detailed description of the entire video (3-5 sentences).
Cover: setting/environment, main subjects, key objects, overall progression, and outcome.
Every statement must be grounded in what is visible in the frames.

════════════════════════════════════════════════
## PART 2 — L1 + L2 HIERARCHICAL ANNOTATION
════════════════════════════════════════════════

CRITICAL: In Part 1, you chose ONE paradigm. In this section, you MUST ONLY read and apply \
the L1/L2 definitions for YOUR CHOSEN paradigm. IGNORE the rules for the other paradigms.

**If feasibility.skip=true**, output `"macro_phases": []` and skip Parts 2 & 3.

{_format_paradigm_annotation_table()}

### UNIVERSAL TEMPORAL & FORMATTING RULES (Apply to ALL paradigms)

**Visual Priority — Intra-Scene vs Inter-Scene Cuts**: \
Not all cuts are equal. You MUST distinguish two types:
- **Intra-Scene Cut** (do NOT split): Camera angle, zoom, or focal length change on the SAME \
subject in the SAME location during the SAME ongoing activity.
- **Inter-Scene Cut** (MUST split): Cut to a DIFFERENT location, subject, person, or visual modality. \
Check the **background** — if it changes, it is inter-scene.

**No Audio Memory**: You are a VISUAL model — do NOT infer segment boundaries from assumed \
narration, dialogue, or audio cues. If a speaker's voice continues over a visual cut, \
treat the cut as the boundary, not the voice.

**Describe-before-timestamp**: For each L1 phase and L2 event, first write the descriptive fields \
(phase_name, narrative_summary, instruction, dense_caption), then assign start_time/end_time.

**Temporal Logic:**
- **Sparsity & Gaps**: Phases/Events do NOT need to cover the entire video. Gaps are expected.
- **No Overlaps**: L2 events must NOT overlap each other in time.
- **Strict Nesting**: Every L2 event MUST be within its parent L1 phase timeframe.
- **Anti-Fragmentation**: L2 event MUST be >= 5 seconds. But inter-scene cuts MUST create new events.
- **Empty L2**: `"events": []` is valid if an L1 phase has no meaningful sub-structure.

**Text Generation Constraints:**
- `phase_name`: 5–15 words. `narrative_summary`: 2–3 sentences.
- `instruction`: 8–20 words, objective (no "Watch...", "Observe...").
- `dense_caption`: 2–4 sentences, visual detail only.
- Do NOT use people's names. Use "a person", "the host", etc.
- FORBIDDEN words: "explains", "discusses", "describes", "talks about". Replace with visual actions.

════════════════════════════════════════════════
## PART 3 — L3 SUB-ACTION ANNOTATION
════════════════════════════════════════════════

For EACH L2 event where `l3_feasible=true`, annotate L3 micro-actions (atomic sub-actions, \
typically 2-6 seconds each) as `sub_actions` within that event.

### Per-paradigm L3 definitions (use the one matching YOUR paradigm):

{_format_paradigm_l3_table()}

### L3 RULES
- Each sub-action: 2-6 seconds, describing ONE atomic visible action.
- Use absolute integer seconds from the FULL VIDEO timeline.
- Allow gaps between sub-actions (no forced full coverage).
- Sub-actions must be within their parent event's [start_time, end_time].
- For events where `l3_feasible=false`, output `"sub_actions": []`.

### L3 FEASIBILITY (per-phase AND per-event)

{_format_l3_feasibility_ref()}

Set `l3_feasible=false` if:
- Dominated by talking/interviews/static scenes with no physical actions.
- Too short (< 10 seconds) or too abstract.
- Visual detail insufficient (distant shot, blurry, occluded).
Set `l3_feasible=true` if: clear, observable physical actions or state changes at 2fps.

### VISUAL SIGNAL REFERENCE
- Scene/Space: Background/layout/location change, character entry/exit.
- Subject Behavior: Pose transition, gaze shift, speed change, interaction start/end.
- Object State: Appearance/texture/color/position/quantity change.
- Camera/Editing: Rhythm change, montage sequence, focus shift, cut to close-up.

════════════════════════════════════════════════
## OUTPUT JSON (ALL THREE LEVELS IN ONE RESPONSE)
════════════════════════════════════════════════

{{
  "paradigm": "<one of: {paradigm_ids}>",
  "paradigm_confidence": 0.85,
  "paradigm_reason": "<one sentence>",
  "domain_l2": "<domain_l2 category or 'other'>",
  "video_caption": "<3-5 sentences>",
  "feasibility": {{
    "score": 0.85,
    "skip": false,
    "skip_reason": null,
    "estimated_n_phases": 3,
    "estimated_n_events": 8,
    "visual_dynamics": "high"
  }},
  "video_metadata": {{
    "has_text_overlay": false,
    "has_narration": true,
    "camera_style": "<static_tripod | handheld | multi_angle | first_person>",
    "editing_style": "<continuous | jump_cut | montage | mixed>"
  }},
  "summary": "<one sentence>",
  "global_phase_criterion": "<one sentence>",
  "macro_phases": [
    {{
      "phase_id": 1,
      "start_time": 5,
      "end_time": 60,
      "phase_name": "<5-15 words>",
      "narrative_summary": "<2-3 sentences>",
      "event_split_criterion": "<one sentence>",
      "l3_feasible": true,
      "l3_reason": "<1 sentence>",
      "events": [
        {{
          "event_id": 1,
          "start_time": 5,
          "end_time": 28,
          "instruction": "<8-20 words>",
          "dense_caption": "<2-4 sentences>",
          "visual_keywords": ["kw1", "kw2"],
          "l3_feasible": true,
          "l3_reason": "<1 sentence>",
          "sub_actions": [
            {{
              "action_id": 1,
              "start_time": 5,
              "end_time": 10,
              "sub_action": "<5-15 word action phrase>",
              "caption": "<1-2 sentences: detailed visual description>",
              "pre_state": "<visual state before, or null>",
              "post_state": "<visual state after, or null>"
            }}
          ]
        }},
        {{
          "event_id": 2,
          "start_time": 30,
          "end_time": 45,
          "instruction": "<8-20 words>",
          "dense_caption": "<2-4 sentences>",
          "visual_keywords": ["kw1", "kw2"],
          "l3_feasible": false,
          "l3_reason": "Talking to camera, no physical actions.",
          "sub_actions": []
        }}
      ]
    }}
  ]
}}

## RULES
1. Output strictly valid JSON. No markdown code blocks.
2. All timestamps: absolute integer seconds within [0, {duration_sec}].
3. Strict nesting: L2 events within L1 phases; L3 sub_actions within L2 events.
4. Anti-fragmentation: L2 events >= 5 seconds; L3 sub_actions 2-6 seconds each.
5. No names: never use people's names.
6. Feasibility enums: skip_reason is null or "talk_dominant" | "ambient_static"; \
visual_dynamics is "high" | "medium" | "low".
7. If feasibility.skip=true, output "macro_phases": [].
8. For events with l3_feasible=false, output "sub_actions": [].
9. IGNORE ALL AUDIO. Base every decision on visual information only."""


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    import subprocess
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])


def _extract_json(text: str) -> dict | None:
    """Extract the first valid JSON object from VLM output text."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    depth = 0
    start = None
    for i, c in enumerate(text):
        if c == '{':
            if depth == 0:
                start = i
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    start = None
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Gemini API call
# ─────────────────────────────────────────────────────────────────────────────

def call_gemini(
    client: genai.Client,
    model: str,
    video_path: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_retries: int = 3,
    fps: float = 2.0,
) -> tuple[dict | None, str | None, dict | None]:
    """
    Call Gemini with full video inline + structured prompt.
    Returns (parsed_json, thinking_text, usage_info) or (None, None, None) on failure.
    """
    with open(video_path, "rb") as f:
        video_bytes = f.read()

    ext = os.path.splitext(video_path)[1].lower()
    mime_map = {
        ".mp4": "video/mp4", ".webm": "video/webm", ".avi": "video/x-msvideo",
        ".mov": "video/quicktime", ".mkv": "video/x-matroska",
    }
    mime_type = mime_map.get(ext, "video/mp4")

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part(text=user_prompt),
                types.Part(
                    inline_data=types.Blob(data=video_bytes, mime_type=mime_type),
                    video_metadata=types.VideoMetadata(fps=fps),
                ),
            ],
        )
    ]

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=temperature,
        top_p=1,
        thinking_config=types.ThinkingConfig(include_thoughts=True),
        safety_settings=SAFETY_SETTINGS,
        response_mime_type="application/json",
    )

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model, contents=contents, config=config,
            )
            caption = ""
            thinking = ""
            for part in response.candidates[0].content.parts:
                if not part.text:
                    continue
                if part.thought:
                    thinking += part.text + "\n"
                else:
                    caption += part.text + "\n"

            parsed = _extract_json(caption)
            if parsed is not None:
                # Extract usage metadata
                usage = {}
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    um = response.usage_metadata
                    usage = {
                        "prompt_tokens": getattr(um, "prompt_token_count", None),
                        "candidates_tokens": getattr(um, "candidates_token_count", None),
                        "total_tokens": getattr(um, "total_token_count", None),
                        "thinking_tokens": getattr(um, "thinking_token_count", None),
                    }
                return parsed, thinking, usage
            print(f"  [attempt {attempt + 1}] JSON parse failed, retrying...", flush=True)
            print(f"  Raw output: {caption[:300]}...", flush=True)
        except Exception:
            traceback.print_exc()
        time.sleep(2 ** attempt)

    return None, None, None


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Gemini native 3-level video annotation")
    p.add_argument("--data-path", type=str, required=True,
                   help="Input JSONL with video_path field (one video per line)")
    p.add_argument("--output-dir", type=str, required=True,
                   help="Output directory — one {clip_key}.json per video")
    p.add_argument("--frames-dir", type=str, default=None,
                   help="Frames directory (for data_visualization). "
                        "If set, writes frame_dir into each output JSON.")
    p.add_argument("--model", default="gemini-2.5-pro", help="Gemini model name")
    p.add_argument("--fps", type=float, default=2.0, help="FPS hint for Gemini video processing")
    p.add_argument("--api-key", default=None, help="Gemini API key (or set GEMINI_API_KEY env)")
    p.add_argument("--credential-json", default=None,
                   help="Path to Vertex AI service account JSON (or set GOOGLE_APPLICATION_CREDENTIALS)")
    p.add_argument("--project-id", default=None, help="GCP project ID (required for Vertex AI mode)")
    p.add_argument("--location", default="global", help="Vertex AI location (default: global)")
    p.add_argument("--overwrite", action="store_true", help="Re-annotate even if output JSON exists")
    p.add_argument("--workers", type=int, default=1, help="Parallel workers (concurrent API calls)")
    p.add_argument("--total-card", type=int, default=1, help="Total parallel shards")
    p.add_argument("--cur-card", type=int, default=0, help="Current shard index")
    p.add_argument("--limit", type=int, default=0, help="Max videos to process (0=all)")
    return p.parse_args()


def _create_client(args) -> genai.Client:
    """Create Gemini client — auto-detect Vertex AI vs native API key mode.

    Priority:
    1. --credential-json / GOOGLE_APPLICATION_CREDENTIALS → Vertex AI mode
    2. --api-key / GEMINI_API_KEY → native API key mode
    """
    credential_json = args.credential_json or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")

    if credential_json:
        # ── Vertex AI mode (service account JSON) ──
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credential_json
        with open(credential_json) as f:
            cred_info = json.load(f)
        project_id = args.project_id or cred_info.get("project_id", "")
        if not project_id:
            print("ERROR: --project-id required for Vertex AI mode (or include in JSON)", file=sys.stderr)
            sys.exit(1)
        print(f"Auth: Vertex AI (project={project_id}, location={args.location})")
        return genai.Client(vertexai=True, project=project_id, location=args.location)

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY", "")
    if api_key:
        # ── Native API key mode ──
        print("Auth: Gemini API key")
        return genai.Client(api_key=api_key)

    print("ERROR: Set --api-key / GEMINI_API_KEY or --credential-json / GOOGLE_APPLICATION_CREDENTIALS",
          file=sys.stderr)
    sys.exit(1)


def _video_to_clip_key(video_path: str) -> str:
    """Derive clip_key from video filename (stem without extension)."""
    return Path(video_path).stem


def _split_vlm_result(result: dict, duration: float) -> dict:
    """Split raw VLM 3-level result into annotate.py-compatible level1/level2/level3 structure.

    Input:  { "macro_phases": [ { "events": [ { "sub_actions": [...] } ] } ] }
    Output: { "level1": {"macro_phases": [...]},
              "level2": {"events": [...]},
              "level3": {"grounding_results": [...]} }
    """
    phases = result.get("macro_phases", [])
    flat_events = []
    flat_l3 = []

    for phase in phases:
        events = phase.pop("events", [])
        for ev in events:
            ev["parent_phase_id"] = phase["phase_id"]
            sub_actions = ev.pop("sub_actions", [])
            flat_events.append(ev)

            for sa in sub_actions:
                sa["parent_event_id"] = ev["event_id"]
                sa["parent_phase_id"] = phase["phase_id"]
                flat_l3.append(sa)

    # Re-number events globally by start_time
    flat_events.sort(key=lambda e: e.get("start_time", 0))
    for i, ev in enumerate(flat_events, 1):
        ev["event_id"] = i

    return {
        "level1": {"macro_phases": phases},
        "level2": {"events": flat_events},
        "level3": {"grounding_results": flat_l3},
    }


def _process_one(
    rec: dict, idx: int, total: int,
    client: genai.Client, model: str, fps: float,
    output_dir: Path, frames_dir: Path | None, overwrite: bool,
    counter: dict, lock: threading.Lock,
) -> None:
    """Process a single video — thread-safe."""
    # Resolve video path
    if "video_path" in rec:
        video_path = rec["video_path"]
    elif "videos" in rec and rec["videos"]:
        video_path = rec["videos"][0]
    else:
        print(f"[{idx}/{total}] SKIP — no video_path or videos field", flush=True)
        return

    meta = rec.get("metadata", {})
    clip_key = meta.get("clip_key") or rec.get("clip_key") or _video_to_clip_key(video_path)
    out_file = output_dir / f"{clip_key}.json"

    if out_file.exists() and not overwrite:
        print(f"[{idx}/{total}] {clip_key} — SKIP (exists)", flush=True)
        return

    print(f"[{idx}/{total}] {clip_key}", flush=True)

    try:
        t0 = time.time()
        duration = _get_video_duration(video_path)
        n_frames = int(duration * fps)

        prompt = build_3level_prompt(n_frames=n_frames, duration_sec=int(duration))

        result, thinking, usage = call_gemini(
            client, model, video_path,
            SYSTEM_PROMPT_NO_AUDIO, prompt,
            fps=fps,
        )

        elapsed = time.time() - t0

        # Accumulate token usage
        if usage:
            with lock:
                for k, v in usage.items():
                    if v is not None:
                        counter[k] = counter.get(k, 0) + v
                counter["n_calls"] = counter.get("n_calls", 0) + 1

        if result is None:
            print(f"  [{clip_key}] FAILED: no result after retries ({elapsed:.1f}s)", flush=True)
            return

        levels = _split_vlm_result(deepcopy(result), duration)

        ann = {
            "clip_key": clip_key,
            "video_path": video_path,
            "source_video_path": video_path,
            "clip_duration_sec": duration,
            "annotation_fps": fps,
            "frame_dir": str(frames_dir / clip_key) if frames_dir else None,
            "archetype": result.get("paradigm"),
            "archetype_confidence": result.get("paradigm_confidence"),
            "archetype_reason": result.get("paradigm_reason"),
            "domain_l2": result.get("domain_l2"),
            "domain_l1": DOMAIN_L2_TO_L1.get(result.get("domain_l2", ""), "other"),
            "video_caption": result.get("video_caption"),
            "feasibility": result.get("feasibility"),
            "video_metadata": result.get("video_metadata"),
            "summary": result.get("summary"),
            "global_phase_criterion": result.get("global_phase_criterion"),
            "level1": levels["level1"],
            "level2": levels["level2"],
            "level3": levels["level3"],
            "annotation_model": model,
            "annotation_thinking": thinking,
            "token_usage": usage,
            "annotated_at": datetime.now(timezone.utc).isoformat(),
        }

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(ann, f, ensure_ascii=False, indent=2)

        n_ph = len(levels["level1"]["macro_phases"])
        n_ev = len(levels["level2"]["events"])
        n_l3 = len(levels["level3"]["grounding_results"])
        paradigm = result.get("paradigm", "?")
        domain = result.get("domain_l2", "?")
        skip = result.get("feasibility", {}).get("skip", False)
        tok_str = ""
        if usage and usage.get("total_tokens"):
            tok_str = f" | tokens={usage['total_tokens']}"
            if usage.get("prompt_tokens"):
                tok_str += f" (prompt={usage['prompt_tokens']}, out={usage.get('candidates_tokens', '?')})"
        print(f"  [{clip_key}] {elapsed:.1f}s | paradigm={paradigm}, domain={domain}, "
              f"{n_ph}ph/{n_ev}ev/{n_l3}l3{tok_str}"
              f"{' [SKIP]' if skip else ''}", flush=True)

    except Exception as e:
        print(f"  [{clip_key}] ERROR: {e}", flush=True)
        traceback.print_exc()


def main():
    args = parse_args()
    client = _create_client(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load input
    video_list = []
    with open(args.data_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line.strip())
            video_list.append(rec)

    if args.limit > 0:
        video_list = video_list[:args.limit]

    # Shard
    stride = len(video_list) // args.total_card + 1
    video_list = video_list[stride * args.cur_card : stride * (args.cur_card + 1)]
    print(f"Shard {args.cur_card}/{args.total_card}: {len(video_list)} videos, "
          f"model={args.model}, fps={args.fps}, workers={args.workers}")

    # Global token counter
    token_counter: dict = {}
    counter_lock = threading.Lock()
    frames_dir = Path(args.frames_dir) if args.frames_dir else None

    if args.workers <= 1:
        # Sequential
        for idx, rec in enumerate(video_list):
            _process_one(
                rec, idx + 1, len(video_list),
                client, args.model, args.fps,
                output_dir, frames_dir, args.overwrite,
                token_counter, counter_lock,
            )
    else:
        # Parallel
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {}
            for idx, rec in enumerate(video_list):
                fut = pool.submit(
                    _process_one,
                    rec, idx + 1, len(video_list),
                    client, args.model, args.fps,
                    output_dir, frames_dir, args.overwrite,
                    token_counter, counter_lock,
                )
                futures[fut] = idx
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as e:
                    print(f"  Worker error: {e}", flush=True)

    # Print token usage summary
    print("\n========== TOKEN USAGE SUMMARY ==========")
    if token_counter:
        n_calls = token_counter.get("n_calls", 0)
        total_tok = token_counter.get("total_tokens", 0)
        prompt_tok = token_counter.get("prompt_tokens", 0)
        out_tok = token_counter.get("candidates_tokens", 0)
        think_tok = token_counter.get("thinking_tokens", 0)
        print(f"API calls:       {n_calls}")
        print(f"Total tokens:    {total_tok:,}")
        print(f"  Prompt tokens: {prompt_tok:,}")
        print(f"  Output tokens: {out_tok:,}")
        if think_tok:
            print(f"  Think tokens:  {think_tok:,}")
        if n_calls > 0:
            print(f"  Avg per call:  {total_tok // n_calls:,}")
        # Estimate: 1 image token ≈ 258 tokens (Gemini low-res image)
        # Video at 2fps for Ns = 2N frames → ~2N*258 = 516N image tokens
        if prompt_tok and n_calls:
            avg_prompt = prompt_tok // n_calls
            print(f"\n  Avg prompt tokens/video: {avg_prompt:,}")
            print(f"  (Gemini video: ~258 tokens per frame at low-res)")
    else:
        print("No successful API calls.")
    print("Done.")


if __name__ == "__main__":
    main()
