# Hierarchical Video Annotation Pipeline v4 — Design Document

> Target: NeurIPS 2026
> Status: Design Draft (2026-04-09)
> Replaces: v3 archetype-driven pipeline (8 archetypes)

---

## Quick Start

```bash
# ── One-shot pipeline (Stage 1 classify + Stage 2 annotate) ─────
bash proxy_data/youcook2_seg/hier_seg_annotation/run_pipeline.sh

# ── Test mode: process only 5 clips ─────────────────────────────
LIMIT=5 bash proxy_data/youcook2_seg/hier_seg_annotation/run_pipeline.sh

# ── Custom config ────────────────────────────────────────────────
JSONL=/path/to/screen_keep.jsonl \
MODEL=pa/gemini-3.1-pro-preview \
WORKERS=8 \
LIMIT=50 \
    bash proxy_data/youcook2_seg/hier_seg_annotation/run_pipeline.sh
```

**Pipeline steps** (`run_pipeline.sh`):
| Step | Script | Description |
|------|--------|-------------|
| S1.1 | `extract_frames.py` | Extract 64 frames/video for classification |
| S1.2 | `stage1_classify.py` | VLM paradigm + domain + feasibility filtering |
| S2.1 | `extract_frames.py` | Extract 1fps frames for annotation (keep-only) |
| S2.2 | `annotate.py --level merged` | Classify archetype + merged L1+L2 annotation |
| S2.3 | `extract_frames.py` | Extract 2fps L3 frames per L2 event |
| S2.4 | `annotate.py --level 3` | L3 micro-action annotation |

Environment variables: `JSONL`, `MODEL`, `WORKERS`, `LIMIT` (0=all).

---

## 1. Design Goals

1. **Stage 1 — Metadata + Domain + Feasibility**: Classify every video's **paradigm** (temporal structure) and **domain** (content category), then filter out low-value samples before expensive annotation.
2. **Stage 2 — Paradigm-Driven Hierarchical Annotation**: 7 independent paradigms, each with archetype-specific L1/L2/L3 definitions and dense captions at every level.
3. **Downstream Data Derivation**: A single annotation JSON simultaneously produces:
   - Hierarchical temporal segmentation training data (L1/L2/L3)
   - Event logic training data (next-event, middle-segment, temporal ordering, causal reasoning)
   - Dense video caption training data (per-segment multi-granularity descriptions)

---

## 2. Two-Level Domain Taxonomy

**Content-topic classification**, fully orthogonal to paradigm (temporal structure).
Paradigm describes HOW a video is structured; domain describes WHAT the video is about.

Example: A cooking video → domain = `food_cooking` (L1 = `task_howto`), but paradigm could be:
- `tutorial` (step-by-step recipe)
- `sports_match` (cooking competition)
- `vlog` (restaurant review)
- `educational` (food science lecture)

Reference datasets: COIN (12 domains), HowTo100M (12 categories), ActivityNet (200 activities), Kinetics-700.

### 2.1 Domain L1 (Coarse — 6 content topics)

| ID | Domain L1 | Description |
|----|-----------|-------------|
| `knowledge_education` | Knowledge & Education | Lectures, science experiments, tech, humanities, news |
| `film_entertainment` | Film & Entertainment | Movies, TV dramas, animation, variety shows |
| `sports_esports` | Sports & Esports | Ball sports, athletics, extreme sports, video games |
| `lifestyle_vlog` | Lifestyle & Vlog | Daily life, travel, food tasting, pets |
| `arts_performance` | Arts & Performance | Music, dance, theater, magic, visual arts |
| `task_howto` | Task & How-to | Cooking, crafts/DIY, repair, beauty tutorials |

### 2.2 Domain L2 (Fine — 22 content categories)

| Domain L1 | Domain L2 | Typical Content |
|-----------|-----------|-----------------|
| **knowledge_education** | `science_tech` | Science experiments, tech reviews, programming |
| | `humanities_history` | History, language, culture, social documentaries |
| | `lecture_speech` | Classroom lectures, speeches, whiteboard explanations |
| | `news_report` | News broadcasts, current affairs commentary |
| **film_entertainment** | `movie_drama` | Movie clips, TV dramas, short films |
| | `animation_cg` | 2D/3D animation, game CG, motion graphics |
| | `variety_show` | Variety shows, reality TV, talk shows |
| **sports_esports** | `ball_sport` | Basketball, football, tennis, volleyball |
| | `athletics_fitness` | Track & field, gymnastics, gym workouts |
| | `outdoor_extreme` | Surfing, skiing, racing, rock climbing |
| | `video_game` | Game streams, esport matches |
| **lifestyle_vlog** | `daily_vlog` | Personal daily life, shopping, social gatherings |
| | `travel_scenery` | Travel vlogs, nature scenery, city walks |
| | `food_tasting` | Mukbang, food reviews, restaurant visits |
| | `pet_animal` | Pets, wildlife observation |
| **arts_performance** | `music_audio` | Instrument playing, concerts, bands |
| | `dance_choreography` | Street dance, ballet, traditional dance |
| | `theater_magic` | Stage plays, magic shows, circus acts |
| **task_howto** | `food_cooking` | Kitchen cooking, baking, food preparation |
| | `crafts_diy` | Woodwork, sewing, origami, handmade crafts |
| | `repair_assembly` | Electronics repair, furniture assembly, car repair |
| | `beauty_grooming` | Makeup tutorials, hairstyling, personal care |

### 2.3 Domain Classification Rules

```
IMPORTANT — Domain is orthogonal to paradigm:
  Domain = WHAT the video is about (content topic)
  Paradigm = HOW the video is temporally structured

Rules:
1. Domain L2 strictly determines Domain L1 (parent-child mapping, lookup table)
2. If video spans multiple domains, classify by MAJORITY visible content
3. "other" is allowed for L2 when no fine category fits → L1 = closest match
4. Do NOT let paradigm influence domain:
   - A dance tutorial → domain=arts_performance, paradigm=tutorial
   - A dance competition → domain=arts_performance, paradigm=sports_match
   - A dance documentary → domain=arts_performance, paradigm=cinematic
```

---

## 3. Stage 1 — Metadata + Classification + Feasibility Filter

### 3.1 Purpose

Before expensive multi-step annotation, quickly classify and filter videos using a single VLM call with 64 uniformly sampled frames.

### 3.2 Output Schema

```json
{
  "paradigm": "tutorial | educational | cinematic | vlog | sports_match | cyclical | continuous",
  "paradigm_confidence": 0.85,
  "paradigm_reason": "Step-by-step physical task with progressive stages...",

  "domain_l1": "task_howto",
  "domain_l2": "food_cooking",

  "feasibility": {
    "score": 0.92,
    "skip": false,
    "skip_reason": null,
    "estimated_n_phases": 4,
    "estimated_n_events": 12,
    "visual_dynamics": "high"
  },

  "video_metadata": {
    "has_text_overlay": true,
    "has_narration": true,
    "camera_style": "static_tripod | handheld | multi_angle | first_person",
    "editing_style": "continuous | jump_cut | montage | mixed"
  }
}
```

### 3.3 Feasibility Filter Rules

| Condition | Action | Reason |
|-----------|--------|--------|
| `paradigm == "talk"` equivalent (detected as conversation-dominant) | `skip=true, skip_reason="talk_dominant"` | Low visual dynamics, language-driven |
| `paradigm == "ambient"` equivalent (no subject/structure) | `skip=true, skip_reason="ambient_static"` | No temporal structure worth annotating |
| `visual_dynamics == "low"` AND `estimated_n_events < 2` | `skip=true, skip_reason="low_visual_dynamics"` | Insufficient content for hierarchical annotation |
| `feasibility.score < 0.4` | `skip=true, skip_reason="low_feasibility"` | VLM self-assessed low annotability |
| `clip_duration < 15s` | `skip=true, skip_reason="too_short"` | Cannot form meaningful hierarchy |

### 3.4 Changes from v3

| Aspect | v3 (Current) | v4 (Proposed) |
|--------|-------------|---------------|
| Archetype count | 8 (incl. talk, ambient) | 7 paradigms (talk/ambient filtered in Stage 1) |
| Domain | Single `domain_l2` flat list (22 values) | Two-level: `domain_l1` (6) + `domain_l2` (22) |
| Feasibility | None (all videos proceed) | Score + filter before annotation |
| Camera/editing metadata | None | Captured for downstream analysis |

---

## 4. Stage 2 — 7 Independent Paradigms

### 4.1 Paradigm Overview

| Paradigm | Topology | L1 Name | L2 Name | L3 Name | L3 Micro Type | L3 Parent |
|----------|----------|---------|---------|---------|---------------|-----------|
| **tutorial** | procedural | Process Stage | Sub-goal Workflow | Object State Change | state_change | event |
| **educational** | procedural | Knowledge Module | Explanation/Demo Unit | Key Info Beat | state_change | event |
| **cinematic** | sequence | Narrative Act | Scene Unit | Interaction Beat | interaction_unit | event |
| **vlog** | sequence | Topic/Location Segment | Activity Clip | Key Moment | interaction_unit | event |
| **sports_match** | sequence | Match Phase | Rally/Play | Key Action | state_change | event |
| **cyclical** | periodic | Macro-Unit/Routine | Sub-Unit (optional→enabled) | Beat/Step | repetition_unit | phase or event |
| **continuous** | observation | Focus State | Behavior State | Movement Detail | state_change | event |

### 4.2 Key Changes from v3

| Change | Detail |
|--------|--------|
| `talk` removed | Filtered in Stage 1 |
| `ambient` removed | Filtered in Stage 1 |
| `performance` → `cyclical` | Renamed for generality; L2 now **enabled** (was disabled) |
| `continuous` added | New paradigm for long-take observation videos (wildlife, surveillance, extreme sports) |
| Dense caption fields | Added at ALL levels (L1/L2/L3) |

### 4.3 Paradigm Definitions (Detailed)

#### 4.3.1 tutorial — Step-by-Step Task

> Videos where a person performs sequential physical operations toward a tangible outcome.

- **L1 (Process Stage)**: Broad stage organized by intent/goal shift. Pattern: Preparation → Processing → Assembly → Finishing. Duration: 30-120s. Max: 6.
- **L2 (Sub-goal Workflow)**: Multi-step workflow (10-60s) completing a verifiable sub-goal. Max: 5 per phase.
- **L3 (Object State Change)**: Single discrete physical interaction (2-6s) where ONE object undergoes ONE visible state change.
- **Typical domains**: `culinary.*`, `instructional.*`, `professional.manufacturing`
- **Classification signals**: Visible materials/tools being manipulated; clear progression from raw to finished state; person performing sequential operations.

#### 4.3.2 educational — Knowledge Delivery

> Videos where structured knowledge is the primary deliverable.

- **L1 (Knowledge Module)**: Distinct teaching topic or knowledge unit covering one concept/theorem/experiment. Duration: 30-300s. Max: 5.
- **L2 (Explanation/Demo Unit)**: Focused explanation or demonstration of one sub-concept. Duration: 15-90s. Max: 4 per module.
- **L3 (Key Info Beat)**: Moment where new information appears or key visual state change occurs. Duration: 3-15s.
- **Typical domains**: `education_science.*`, `instructional.digital_creation`
- **Classification signals**: PPT/whiteboard visible; instructor speaking to camera; structured knowledge delivery; text/diagrams on screen.

#### 4.3.3 cinematic — Scripted Narrative

> Professionally edited narrative content with scene structure and story arc.

- **L1 (Narrative Act)**: Distinct emotional/narrative stage. Pattern: Setup → Confrontation → Climax → Resolution. Duration: 30-300s. Max: 5.
- **L2 (Scene Unit)**: Continuous narrative within same space-time and character set. Duration: 10-90s. Max: 5 per act.
- **L3 (Interaction Beat)**: Complete social/physical interaction or distinct emotional shift. Duration: 2-10s.
- **Typical domains**: `performing_arts.theater_film`, `performing_arts.gaming`
- **Classification signals**: Professional cinematography; scene transitions; multiple characters; emotional arc; deliberate camera work.

#### 4.3.4 vlog — Topic/Location Driven

> Personal-perspective content driven by location changes, topic shifts, or daily activity flow.

- **L1 (Topic/Location Segment)**: Organized by location change or topic shift. Duration: 30-180s. Max: 6.
- **L2 (Activity Clip)**: Specific activity within a location/topic. Duration: 10-60s. Max: 4 per segment.
- **L3 (Key Moment)**: Information increment — first impression, reaction, discovery. Duration: 2-8s.
- **Typical domains**: `lifestyle.travel_tourism`, `lifestyle.social_event`, `culinary.food_presentation`
- **Classification signals**: First-person/creator-facing camera; location transitions; casual narration; topic-driven segments.

#### 4.3.5 sports_match — Rule-Based Competition

> Competitive events with structured rounds, periods, or plays defined by sport rules.

- **L1 (Match Phase)**: Structural period defined by sport rules (halves, rounds, sets). Duration: 60-600s. Max: 4.
- **L2 (Rally/Play)**: Complete rally, play, or scoring sequence. Duration: 5-30s. Max: 10 per phase.
- **L3 (Key Action)**: Decisive physical action — shot, score, foul, save. Duration: 1-5s.
- **Typical domains**: `sports_fitness.team_sport`, `sports_fitness.individual_sport`, `sports_fitness.combat_sport`
- **Classification signals**: Scoreboard; referee; structured back-and-forth play; crowd; uniforms.

#### 4.3.6 cyclical — Repeating Motion Patterns

> Activities with rhythmic, repeating motion cycles in a stable environment.

- **L1 (Macro-Unit/Routine)**: Complete repetition logic or routine section. Pattern: Warm-up → High-intensity → Cool-down. Duration: 30-180s. Max: 4.
- **L2 (Sub-Unit)**: Fixed action combination within a cycle (e.g., left step 4-beat, right step 4-beat). Duration: 8-30s. Max: 6 per macro-unit. **[NEW: was disabled in v3]**
- **L3 (Beat/Step)**: Single repetition, cycle, or step. Duration: 2-8s. Max: 20.
- **L3 Parent**: `event` when L2 is populated, `phase` as fallback when L2 is empty.
- **Typical domains**: `sports_fitness.fitness_exercise`, `sports_fitness.dance`, `performing_arts.music_instrument`
- **Classification signals**: Same motion pattern repeating; body returns to starting position; intensity changes.

#### 4.3.7 continuous — Unstructured Observation

> Long-take or minimally edited continuous recording focused on a subject's ongoing activity. **[NEW paradigm]**

- **L1 (Focus State)**: Core attention object or major state. Duration: 30-300s. Max: 5.
- **L2 (Behavior State)**: Observable state change of the focus subject. Duration: 10-60s. Max: 5 per focus.
- **L3 (Movement Detail)**: Fine-grained physical adjustment or motion. Duration: 2-8s.
- **Typical domains**: `nature_environment.wildlife`, `sports_fitness.extreme_sport`, `professional.vehicle_operation`
- **Classification signals**: Single continuous shot or minimal cuts; no human-imposed structure; focus on natural/ongoing activity; no narrator-driven segmentation.

---

## 5. Dense Caption Design

### 5.1 Per-Level Caption Fields

Dense captions are generated **within the same VLM call** as segmentation (no extra API cost).

#### L1 — Phase-Level Caption

```json
{
  "phase_id": 1,
  "start_time": 5,
  "end_time": 60,
  "phase_name": "Gathering ingredients and preparing dry mixture",
  "narrative_summary": "The baker collects all necessary ingredients from the pantry and measures out dry components including flour, sugar, and cocoa powder. These are combined in a large stainless steel mixing bowl on the granite countertop.",
  "scene_description": "Bright kitchen with overhead lighting. Wooden pantry to the left, granite countertop center frame. Ingredients arranged in a row: flour bag, sugar container, cocoa tin. Large steel bowl prominently placed.",
  "event_split_criterion": "..."
}
```

| Field | Type | Description | New? |
|-------|------|-------------|------|
| `phase_name` | str, 5-15 words | Descriptive phase title | Existing |
| `narrative_summary` | str, 2-3 sentences | What happens and why, temporal progression | Existing (enhanced) |
| `scene_description` | str, 1-2 sentences | Visual environment: objects, layout, lighting | **NEW** |

#### L2 — Event-Level Caption

```json
{
  "event_id": 1,
  "start_time": 5,
  "end_time": 15,
  "instruction": "Measure flour and sugar from bags into the mixing bowl",
  "dense_caption": "The baker reaches for the flour bag on the left side of the counter, scoops flour with a metal measuring cup, and pours it into the steel bowl. A small cloud of flour rises. Then picks up the sugar container, measures one cup, and adds it to the bowl alongside the flour.",
  "visual_keywords": ["measuring cup", "flour", "sugar", "steel bowl"],
  "parent_phase_id": 1
}
```

| Field | Type | Description | New? |
|-------|------|-------------|------|
| `instruction` | str, 8-20 words | Concise action description (WHAT + WITH WHAT + OUTCOME) | Existing |
| `dense_caption` | str, 2-4 sentences | Detailed process: action sequence, object interactions, spatial relations | **NEW** |
| `visual_keywords` | list[str] | Key visible objects | Existing |

#### L3 — Action-Level Caption

```json
{
  "action_id": 1,
  "start_time": 5,
  "end_time": 8,
  "sub_action": "Pour measured flour from the bag into steel mixing bowl",
  "pre_state": "Flour bag held above the empty steel bowl, measuring cup with flour on counter",
  "post_state": "White flour visible in bowl bottom, measuring cup lowered to counter",
  "action_detail": "Right hand tilts the flour bag at approximately 45 degrees over the bowl opening. Flour streams in a steady flow for 2 seconds. Left hand steadies the bowl rim.",
  "parent_event_id": 1,
  "parent_phase_id": 1
}
```

| Field | Type | Description | New? |
|-------|------|-------------|------|
| `sub_action` | str, 5-15 words | Atomic action phrase | Existing |
| `pre_state` | str, 1 sentence | Visual state BEFORE action | Existing |
| `post_state` | str, 1 sentence | Visual state AFTER action | Existing |
| `action_detail` | str, 1-2 sentences | Body movement trajectory, object dynamics, spatial specifics | **NEW** |

### 5.2 Caption Quality Requirements

1. **Grounded in visual evidence**: Every statement must be verifiable from frames. No inference about off-screen events.
2. **Spatial specificity**: Use directional language (left/right, above/below, foreground/background).
3. **Object identity consistency**: Same object must use the same referring expression across all levels.
4. **Temporal progression**: L2 `dense_caption` must describe events in chronological order.
5. **No redundancy across levels**: L1 summarizes the "what and why", L2 details the "how", L3 captures the "exact motion".

---

## 6. Complete Annotation JSON Schema (v4)

```json
{
  // ─── Video Metadata ───
  "clip_key": "video_abc_001",
  "video_path": "path/to/video.mp4",
  "source_video_path": "path/to/original.mp4",
  "source_mode": "full_video | windowed_clip | full_video_prefix",
  "clip_duration_sec": 120.0,
  "n_frames": 120,
  "frame_dir": "/path/to/frames/video_abc_001",

  // ─── Stage 1: Classification (NEW in v4) ───
  "paradigm": "tutorial",
  "paradigm_confidence": 0.92,
  "paradigm_reason": "Clear step-by-step physical task with progressive stages",
  "domain_l1": "task_howto",
  "domain_l2": "food_cooking",
  "feasibility": {
    "score": 0.92,
    "skip": false,
    "skip_reason": null,
    "estimated_n_phases": 4,
    "estimated_n_events": 12,
    "visual_dynamics": "high"
  },
  "video_metadata": {
    "has_text_overlay": true,
    "has_narration": true,
    "camera_style": "static_tripod",
    "editing_style": "continuous"
  },

  // ─── Stage 2: Hierarchical Annotation ───
  "summary": "A person prepares and bakes a chocolate cake from scratch",
  "global_phase_criterion": "Organized by major goal shifts from preparation through finishing",

  "l3_feasibility": {
    "suitable": true,
    "reason": "Clear object manipulations with visible state changes in each event",
    "estimated_l3_actions": 12
  },

  "level1": {
    "macro_phases": [
      {
        "phase_id": 1,
        "start_time": 5,
        "end_time": 60,
        "phase_name": "Gathering ingredients and preparing dry mixture",
        "narrative_summary": "The baker collects ingredients and measures dry components...",
        "scene_description": "Bright kitchen with overhead lighting, granite countertop...",
        "event_split_criterion": "Events represent distinct measurement and mixing operations"
      }
    ],
    "_sampling": { "n_sampled_frames": 24, "jpeg_quality": 60 }
  },

  "level2": {
    "events": [
      {
        "event_id": 1,
        "start_time": 5,
        "end_time": 15,
        "instruction": "Measure flour and sugar from bags into the mixing bowl",
        "dense_caption": "The baker reaches for the flour bag, scoops with measuring cup...",
        "visual_keywords": ["measuring cup", "flour", "sugar", "bowl"],
        "parent_phase_id": 1
      }
    ]
  },

  "level3": {
    "micro_type": "state_change",
    "micro_split_criterion": "Each action = one object's one visible state change",
    "grounding_results": [
      {
        "action_id": 1,
        "start_time": 5,
        "end_time": 8,
        "sub_action": "Pour measured flour from bag into steel mixing bowl",
        "pre_state": "Flour bag held above empty steel bowl",
        "post_state": "White flour visible in bowl bottom",
        "action_detail": "Right hand tilts flour bag at 45 degrees, flour streams steadily for 2s",
        "parent_event_id": 1,
        "parent_phase_id": 1
      }
    ],
    "_segment_calls": [...]
  },

  "annotated_at": "2026-04-09T14:32:10.123456+00:00"
}
```

---

## 7. Event Logic Data Derivation

From a single hierarchical annotation, automatically generate 4 types of event logic training data (NO extra VLM calls needed).

### 7.1 Task Definitions

| Task | Input | Output | Form | Reward |
|------|-------|--------|------|--------|
| **Next-Event Prediction** | Video + first k events | Select next event from 4 candidates | MCQ (4-way) | Binary correctness |
| **Middle-Segment Selection** | Video + event_before + event_after | Select missing middle event from 4 candidates | MCQ (4-way) | Binary correctness |
| **Temporal Ordering** | Video + shuffled event list | Restore correct chronological order | Free-form permutation | Kendall's tau / exact match |
| **Causal Reasoning** | Video + two adjacent events' dense_captions | Explain WHY event B follows event A | Open-ended QA | Semantic similarity to GT |

### 7.2 Distractor Generation Strategy

For MCQ tasks (Next-Event, Middle-Segment), distractors come from:

1. **Same video, different phase**: Events from a different L1 phase (temporal confusion).
2. **Same domain, different video**: Events from another video with matching `domain_l2` (semantic confusion).
3. **Random**: Events from unrelated videos (easy negative).

Ratio per question: 1 correct + 1 same-video + 1 same-domain + 1 random = 4 options.

### 7.3 Data Construction (in `build_hier_data.py`)

```python
def build_event_logic_records(ann: dict, all_anns: list[dict]) -> list[dict]:
    """
    From one annotation, generate event logic training records.
    
    Requires: level2.events with dense_caption field.
    Uses all_anns for cross-video distractor mining.
    """
    events = ann["level2"]["events"]  # must have dense_caption
    
    records = []
    
    # 1. Next-Event Prediction
    for i in range(1, len(events)):
        context_events = events[:i]
        target = events[i]
        distractors = mine_distractors(target, ann, all_anns, n=3)
        records.append(build_next_event_mcq(context_events, target, distractors))
    
    # 2. Middle-Segment Selection
    for i in range(1, len(events) - 1):
        before, target, after = events[i-1], events[i], events[i+1]
        distractors = mine_distractors(target, ann, all_anns, n=3)
        records.append(build_middle_segment_mcq(before, after, target, distractors))
    
    # 3. Temporal Ordering (per-phase groups)
    for phase_events in group_events_by_phase(events):
        if len(phase_events) >= 3:
            records.append(build_ordering_task(phase_events))
    
    # 4. Causal Reasoning (using dense_caption)
    for i in range(1, len(events)):
        if events[i-1].get("dense_caption") and events[i].get("dense_caption"):
            records.append(build_causal_qa(events[i-1], events[i]))
    
    return records
```

### 7.4 Problem Types

```python
PROBLEM_TYPES_EVENT_LOGIC = {
    "next_event": "event_logic_next_event",
    "middle_segment": "event_logic_middle_segment",
    "temporal_ordering": "event_logic_temporal_ordering",
    "causal_reasoning": "event_logic_causal_reasoning",
}
```

---

## 8. Dense Caption Training Data

### 8.1 Task Definitions

| Task | Input | Output | Granularity |
|------|-------|--------|-------------|
| **Phase Caption** | Video (L1 clip) + phase timestamps | `narrative_summary` + `scene_description` | L1 |
| **Event Caption** | Video (L2 clip) + event timestamps | `dense_caption` | L2 |
| **Action Caption** | Video (L3 clip) + action timestamps | `action_detail` + `pre_state` + `post_state` | L3 |
| **Hierarchical Caption** | Full video | All-level structured output | L1+L2+L3 |

### 8.2 Problem Types

```python
PROBLEM_TYPES_CAPTION = {
    "phase_caption": "dense_caption_L1",
    "event_caption": "dense_caption_L2",
    "action_caption": "dense_caption_L3",
    "hier_caption": "dense_caption_hierarchical",
}
```

---

## 9. Pipeline Data Flow

```
                            ┌─────────────────────────────────────┐
                            │       Raw Video Collection          │
                            └──────────────┬──────────────────────┘
                                           │
                          ┌────────────────▼─────────────────┐
                          │   STAGE 1: Classification+Filter  │
                          │   (64 frames, 1 VLM call/video)   │
                          │                                    │
                          │   Output:                          │
                          │     - paradigm (7 types)           │
                          │     - domain_l1 (6) + domain_l2 (22) │
                          │     - feasibility score            │
                          │     - video_metadata               │
                          └──────┬───────────────┬─────────────┘
                                 │               │
                         feasibility.skip     feasibility.skip
                           == false              == true
                                 │               │
                                 ▼               ▼
                          ┌──────────┐    ┌────────────┐
                          │ Continue │    │  Discard   │
                          │ to Stage2│    │ (talk,     │
                          └────┬─────┘    │  ambient,  │
                               │          │  static)   │
                               │          └────────────┘
                               │
              ┌────────────────▼─────────────────────┐
              │     STAGE 2: Hierarchical Annotation   │
              │     (paradigm-driven prompts)           │
              │                                        │
              │  Step 1: Merged L1+L2 (all frames)     │
              │    → phases + events + dense captions   │
              │                                        │
              │  Step 2: L3 frame extraction            │
              │    → per-event/phase high-fps frames    │
              │                                        │
              │  Step 3: L3 grounding (per source)     │
              │    → micro-actions + state descriptions │
              └──────────────────┬─────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  Annotation JSON (v4)   │
                    │  per-video .json file    │
                    └────────────┬────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
    ┌─────────▼──────┐  ┌───────▼────────┐  ┌─────▼──────────┐
    │  Hier-Seg Data │  │ Event Logic    │  │ Dense Caption  │
    │  (L1/L2/L3)    │  │ Data           │  │ Data           │
    │                │  │                │  │                │
    │ problem_types: │  │ problem_types: │  │ problem_types: │
    │ temporal_seg_* │  │ event_logic_*  │  │ dense_caption_*│
    └────────────────┘  └────────────────┘  └────────────────┘
```

---

## 10. Training Data Statistics Schema

For paper reporting, collect statistics grouped by paradigm and domain:

```
Statistics axes:
  1. By paradigm:    tutorial | educational | cinematic | vlog | sports_match | cyclical | continuous
  2. By domain_l1:   knowledge_education | film_entertainment | sports_esports | lifestyle_vlog | arts_performance | task_howto
  3. By domain_l2:   food_cooking | ball_sport | ... (22 categories)
  4. By task type:   hier_seg | event_logic | dense_caption
  5. By level:       L1 | L2 | L3

Key metrics per group:
  - n_videos: number of source videos
  - n_samples: number of training samples
  - avg_duration: average clip duration (seconds)
  - avg_n_segments: average number of segments per sample
  - avg_segment_duration: average segment duration (seconds)
```

---

## 11. Code Change Summary

| File | Change Type | Details |
|------|-------------|---------|
| `archetypes.py` | **Major refactor** | Remove `talk`/`ambient`; rename `performance`→`cyclical` (enable L2); add `continuous`; add dense caption fields to `LevelConfig`; redesign classification prompt with feasibility; implement 2-level domain taxonomy (V2: 6 L1 / 22 L2); add L3 feasibility assessment to merged prompt |
| `annotate.py` | **Moderate** | Stage 1: add feasibility scoring + skip logic; merged L1+L2: add `scene_description` (L1), `dense_caption` (L2), `l3_feasibility` to prompt output schema; L3: add `action_detail` to prompt output schema; skip L3 when `l3_feasibility.suitable=false` |
| `build_hier_data.py` | **Major addition** | Add `build_event_logic_records()`, `build_dense_caption_records()`, distractor mining; new problem types; update balanced sampling to use `domain_l1` |
| `prepare_clips.py` | **Minor** | Support `cyclical` L2 clips (was skipped for `performance`) |
| `run_pipeline.sh` | **Minor** | Add Stage 1 filter step before annotation |
| `shared/seg_source.py` | **Minor** | Update `ARCHETYPE_IDS`, domain constants |

---

## 12. Migration from v3

### 12.1 Backward Compatibility

```python
# Existing v3 annotations with old archetype names still work:
PARADIGM_MIGRATION = {
    "performance": "cyclical",   # renamed
    "talk": None,                # filtered out (skip)
    "ambient": None,             # filtered out (skip)
    # All others: name unchanged
}

# Old topology → paradigm fallback (for pre-v3 annotations):
TOPOLOGY_TO_DEFAULT_PARADIGM = {
    "procedural": "tutorial",
    "periodic": "cyclical",
    "sequence": "cinematic",
    "flat": None,  # skip
}
```

### 12.2 Re-annotation Strategy

1. **Existing v3 annotations**: Keep as-is for hier-seg training (paradigm names auto-mapped).
2. **Dense caption fields**: Require re-annotation (or supplementary VLM call) since v3 lacks `dense_caption`, `scene_description`, `action_detail`.
3. **Domain reclassification**: Run Stage 1 on all existing videos to populate two-level domain taxonomy.

---

## Appendix A: Paradigm Classification Prompt (Draft)

```
You are given a {duration}s video clip with {n_frames} frames.

TASK 1 — PARADIGM CLASSIFICATION
Classify the video's dominant temporal structure into exactly one paradigm:

- **tutorial**: Step-by-step physical task with sub-goals → tangible outcome.
  Signals: tools/materials being manipulated, raw→finished progression.
- **educational**: Knowledge delivery via lecture, demo, or experiment.
  Signals: PPT/whiteboard, instructor, structured explanation.
- **cinematic**: Scripted narrative with scene structure and story arc.
  Signals: professional editing, scene transitions, character dialogue.
- **vlog**: Personal perspective, location/topic shifts, creator narration.
  Signals: first-person camera, jump cuts, casual narration.
- **sports_match**: Rule-based competition with structured periods/rounds.
  Signals: scoreboard, referee, back-and-forth play, uniforms.
- **cyclical**: Rhythmic repeating motion cycles in stable environment.
  Signals: same motion repeating, body returns to start, intensity changes.
- **continuous**: Long-take observation of ongoing activity, minimal editing.
  Signals: single continuous shot, no imposed structure, natural activity.

TASK 2 — DOMAIN CLASSIFICATION
Classify the content domain at two levels:

domain_l1: {domain_l1_list}
domain_l2: {domain_l2_list}

TASK 3 — FEASIBILITY ASSESSMENT
Assess whether this video is worth annotating with hierarchical temporal structure.

Output JSON:
{
  "paradigm": "...",
  "paradigm_confidence": 0.85,
  "paradigm_reason": "...",
  "domain_l1": "...",
  "domain_l2": "...",
  "feasibility": {
    "score": 0.0-1.0,
    "skip": true/false,
    "skip_reason": null | "talk_dominant" | "ambient_static" | "too_short" | "low_visual_dynamics",
    "estimated_n_phases": 3,
    "estimated_n_events": 8,
    "visual_dynamics": "high | medium | low"
  },
  "video_metadata": {
    "has_text_overlay": true/false,
    "has_narration": true/false,
    "camera_style": "static_tripod | handheld | multi_angle | first_person",
    "editing_style": "continuous | jump_cut | montage | mixed"
  }
}
```

---

## Appendix B: Dense Caption Prompt Addition (Merged L1+L2)

The following fields are **added** to the existing merged prompt output schema:

```
For each L1 phase, ALSO output:
  "scene_description": "<1-2 sentences: visual environment — objects present, spatial layout, lighting, camera angle>"

For each L2 event, ALSO output:
  "dense_caption": "<2-4 sentences: detailed chronological description of what happens — action sequence, object interactions, spatial movement, state changes>"
```

Quality rules for dense captions:
- Every noun must be visually verifiable in the frames.
- Use spatial language: left/right, foreground/background, above/below.
- Describe temporal order: "first... then... finally..."
- No inference about intent or off-screen events.
