"""
archetypes.py — Video paradigm definitions for universal hierarchical segmentation.

v4: Paradigm-driven pipeline with 2-level domain taxonomy.

Single source of truth for:
  - System prompt (VLM role message)
  - 2-level domain taxonomy (content classification, orthogonal to paradigm)
  - 7 paradigm definitions with L1/L2/L3 configs
  - Classification prompt (Stage 1: paradigm + domain + feasibility)
  - Paradigm-driven annotation prompts (merged L1+L2, L3)
  - Training prompt generators (L1/L2/L3 + hint variants)
  - Backward compat: topology constants, archetype aliases

Usage:
    from archetypes import SYSTEM_PROMPT, PARADIGMS, get_paradigm
    from archetypes import get_classification_prompt, get_paradigm_merged_prompt
    from archetypes import get_paradigm_l1_train_prompt, DOMAIN_L1_ALL, DOMAIN_L2_ALL
    # Backward compat aliases:
    from archetypes import ARCHETYPES, get_archetype  # same as PARADIGMS / get_paradigm
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ─────────────────────────────────────────────────────────────────────────────
# System Prompt (injected as system role in every VLM call)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert in structured video analysis with cross-domain cognitive ability, \
specializing in temporal structure analysis of procedural, periodic, narrative, \
and general activity content. \
Your task is to accurately parse temporal actions, visual state transitions, \
and semantic/narrative structure.

Core annotation principles:
1. [Sparsity over Continuity]: ONLY annotate segments with clear semantic meaning or \
visual actions. Gaps between annotated segments are expected and encouraged. \
Do NOT force adjacent segments to be contiguous.
2. [Precise Boundaries]: Boundaries must reflect the exact moment an action begins \
or the moment a visual state solidifies.
3. [Information Delta]: Focus on moments where new information is produced or \
irreversible state changes occur — physical, visual, or semantic.
4. [Formatting]: Output strictly in valid JSON format."""


# ─────────────────────────────────────────────────────────────────────────────
# Domain Taxonomy (V2) — 6 L1 Domains, ~3-4 L2 Categories per L1
#
# Domain = WHAT the video is about (content topic)
# Paradigm = HOW the video is temporally structured
#
# References:
#   Video-MME (2024): Knowledge, Film&TV, Sports, Art, Life Record, Multi-lang
#   COIN (2019): 12 domains (Dishes, Vehicles, Gadgets, Housework, ...)
#   ActivityNet: Sports, Household, Personal Care, Eating, Socializing, ...
#   LongVideoBench (2024): Entertainment, Knowledge, Lifestyle
# ─────────────────────────────────────────────────────────────────────────────

DOMAIN_L1_ALL: set[str] = {
    "knowledge_education",  # 知识科普、新闻、讲座、纪录片
    "film_entertainment",   # 电影、电视剧、动画、综艺节目
    "sports_esports",       # 竞技体育、健身、极限运动、电子竞技
    "lifestyle_vlog",       # 日常生活、旅行风景、吃播探店、宠物
    "arts_performance",     # 音乐、舞蹈、舞台剧、魔术视觉艺术
    "task_howto",           # 做菜、手工DIY、维修、美妆穿搭教程
}

# L2 → L1 parent mapping (strict, every L2 has exactly one L1 parent)
DOMAIN_L2_TO_L1: dict[str, str] = {
    # knowledge_education (4)
    "science_tech": "knowledge_education",
    "humanities_history": "knowledge_education",
    "lecture_speech": "knowledge_education",
    "news_report": "knowledge_education",
    # film_entertainment (3)
    "movie_drama": "film_entertainment",
    "animation_cg": "film_entertainment",
    "variety_show": "film_entertainment",
    # sports_esports (4)
    "ball_sport": "sports_esports",
    "athletics_fitness": "sports_esports",
    "outdoor_extreme": "sports_esports",
    "video_game": "sports_esports",
    # lifestyle_vlog (4)
    "daily_vlog": "lifestyle_vlog",
    "travel_scenery": "lifestyle_vlog",
    "food_tasting": "lifestyle_vlog",
    "pet_animal": "lifestyle_vlog",
    # arts_performance (3)
    "music_audio": "arts_performance",
    "dance_choreography": "arts_performance",
    "theater_magic": "arts_performance",
    # task_howto (4)
    "food_cooking": "task_howto",
    "crafts_diy": "task_howto",
    "repair_assembly": "task_howto",
    "beauty_grooming": "task_howto",
}

DOMAIN_L2_ALL: set[str] = set(DOMAIN_L2_TO_L1.keys()) | {"other"}

# Reverse: L1 → list of L2
DOMAIN_L1_TO_L2: dict[str, list[str]] = {}
for _l2, _l1 in DOMAIN_L2_TO_L1.items():
    DOMAIN_L1_TO_L2.setdefault(_l1, []).append(_l2)


def resolve_domain_l1(domain_l2: str) -> str:
    """Given a domain_l2, return its parent domain_l1. Returns 'other' for unknowns."""
    return DOMAIN_L2_TO_L1.get(domain_l2, "other")


def _format_domain_l2_for_prompt() -> str:
    """Format domain_l2 values grouped by L1 for the classification prompt."""
    lines = []
    for l1 in sorted(DOMAIN_L1_ALL):
        l2_list = sorted(DOMAIN_L1_TO_L2.get(l1, []))
        lines.append(f"  {l1}: {', '.join(l2_list)}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Backward Compatibility: Topology constants (used during transition)
# ─────────────────────────────────────────────────────────────────────────────

TOPOLOGY_TYPES: set[str] = {"procedural", "periodic", "sequence", "flat"}

TOPOLOGY_TO_L2_MODE: dict[str, str] = {
    "procedural": "workflow",
    "sequence": "episode",
    "periodic": "optional",
    "flat": "skip",
}
TOPOLOGY_TO_L3_MODE: dict[str, str] = {
    "procedural": "state_change",
    "periodic": "repetition_unit",
    "sequence": "interaction_unit",
    "flat": "skip",
}

# ─────────────────────────────────────────────────────────────────────────────
# Archetype Configuration
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Paradigm Configuration (v4: replaces "Archetype" terminology)
# ─────────────────────────────────────────────────────────────────────────────

ParadigmID = Literal[
    "tutorial",
    "cyclical",       # was "performance"
    "cinematic",
    "vlog",
    "educational",
    "sports_match",
    "continuous",      # new in v4
]

# Backward compat alias
ArchetypeID = ParadigmID

TopologyType = Literal["procedural", "periodic", "sequence", "flat", "observation"]

PARADIGM_IDS: set[str] = {
    "tutorial", "cyclical", "cinematic", "vlog",
    "educational", "sports_match", "continuous",
}
# Backward compat alias
ARCHETYPE_IDS = PARADIGM_IDS | {"performance", "talk", "ambient"}  # accept old names

PARADIGM_TO_TOPOLOGY: dict[str, TopologyType] = {
    "tutorial": "procedural",
    "cyclical": "periodic",
    "cinematic": "sequence",
    "vlog": "sequence",
    "educational": "procedural",
    "sports_match": "sequence",
    "continuous": "observation",
}
# Backward compat alias
ARCHETYPE_TO_TOPOLOGY: dict[str, TopologyType] = {
    **PARADIGM_TO_TOPOLOGY,
    "performance": "periodic",  # old name
    "talk": "flat",
    "ambient": "flat",
}

# Reverse mapping: topology → default paradigm (for backward compat)
TOPOLOGY_TO_DEFAULT_PARADIGM: dict[str, str] = {
    "procedural": "tutorial",
    "periodic": "cyclical",
    "sequence": "cinematic",
    "observation": "continuous",
    "flat": "continuous",  # flat now maps to continuous (talk/ambient filtered out)
}
TOPOLOGY_TO_DEFAULT_ARCHETYPE = TOPOLOGY_TO_DEFAULT_PARADIGM  # alias

# Migration map: old archetype → new paradigm (None = filtered in Stage 1)
ARCHETYPE_TO_PARADIGM: dict[str, str | None] = {
    "tutorial": "tutorial",
    "performance": "cyclical",
    "cinematic": "cinematic",
    "vlog": "vlog",
    "educational": "educational",
    "sports_match": "sports_match",
    "talk": None,      # filtered out
    "ambient": None,   # filtered out
    "continuous": "continuous",
    "cyclical": "cyclical",
}


@dataclass(frozen=True)
class LevelConfig:
    """Configuration for a single annotation level (L1/L2/L3)."""
    enabled: bool
    name: str  # e.g., "Process Stage"
    name_zh: str  # e.g., "过程阶段"
    definition: str  # English definition for prompt injection
    boundary_signals: str  # Visual signals for boundary detection
    examples: tuple[str, ...] = ()  # Positive examples
    anti_examples: tuple[str, ...] = ()  # What NOT to annotate
    micro_type: str = ""  # L3 only: state_change, repetition_unit, etc.
    parent: str = ""  # L3 only: "event" or "phase"


@dataclass(frozen=True)
class ArchetypeConfig:
    """Complete archetype configuration with L1/L2/L3 definitions."""
    archetype_id: ArchetypeID
    display_name: str  # Chinese display name
    display_name_en: str  # English display name
    topology: TopologyType
    description: str  # Brief description for classification prompt
    typical_videos: tuple[str, ...]
    classification_signals: str  # What visual signals identify this archetype

    l1: LevelConfig
    l2: LevelConfig
    l3: LevelConfig


# ─────────────────────────────────────────────────────────────────────────────
# Archetype Definitions
# ─────────────────────────────────────────────────────────────────────────────

_TUTORIAL = ArchetypeConfig(
    archetype_id="tutorial",
    display_name="教程/任务型",
    display_name_en="Tutorial / Task",
    topology="procedural",
    description=(
        "Step-by-step task-oriented process with meaningful physical sub-goals "
        "progressing toward a tangible outcome."
    ),
    typical_videos=(
        "cooking recipes", "furniture assembly", "car repair",
        "DIY crafts", "electronics soldering", "home renovation",
    ),
    classification_signals=(
        "Visible materials/tools being manipulated; "
        "clear progression from raw to finished state; "
        "person performing sequential physical operations."
    ),

    l1=LevelConfig(
        enabled=True,
        name="Process Stage",
        name_zh="过程阶段",
        definition=(
            "A broad stage organized by overall intent or goal shift in the task. "
            "Each phase represents a distinct functional milestone. "
            "Typical pattern: Preparation → Processing → Assembly → Finishing."
        ),
        boundary_signals=(
            "Goal/intent shift (from preparing to executing); "
            "primary tool or material change; "
            "completion of a major sub-result visible in workspace."
        ),
        examples=(
            "Preparing and measuring dry ingredients for the base mixture",
            "Assembling the frame by fitting and securing wooden planks",
            "Finishing and decorating the cake with frosting and toppings",
        ),
        anti_examples=(
            "Too vague: 'Preparation'",
            "Too fine: 'Pouring flour into bowl' (this is L2/L3)",
        ),
    ),

    l2=LevelConfig(
        enabled=True,
        name="Sub-goal Workflow",
        name_zh="子目标工作流",
        definition=(
            "A multi-step workflow that completes a verifiable sub-goal "
            "(e.g., 'Kneading the dough until smooth'). "
            "This workflow is ONE continuous event even if shown through multiple "
            "camera angles (e.g., wide shot → close-up of hands). "
            "Intra-scene angle changes do NOT break the event."
        ),
        boundary_signals=(
            "**Valid Boundaries (Inter-Scene)**: Sub-goal completion "
            "(dough finished kneading, one component attached); "
            "switch to a completely different task (kneading → chopping vegetables). "
            "**NOT Boundaries (Intra-Scene)**: Camera angle or zoom changes "
            "within the same ongoing task."
        ),
        examples=(
            "Whisk dry ingredients together until evenly blended",
            "Attach side panel to the base using screws and drill",
            "Knead dough from start to finish (shown via wide + close-up shots)",
        ),
        anti_examples=(
            "Splitting 'Kneading the dough' into 'Wide shot of kneading' and "
            "'Close-up of kneading' — these are ONE event with angle changes",
            "A single atomic motion like 'Pick up the whisk' (this is L3)",
        ),
    ),

    l3=LevelConfig(
        enabled=True,
        name="Object State Change",
        name_zh="物体状态变化",
        definition=(
            "A single, discrete physical interaction (2-6s) where exactly ONE "
            "object undergoes ONE visible physical state change. "
            "Start = contact/onset, End = new state established."
        ),
        boundary_signals=(
            "Physical contact between tool and material; "
            "visible object state transition (texture, color, shape, position); "
            "release after transformation."
        ),
        examples=(
            "Pour measured flour from the bag into the steel mixing bowl",
            "Cut the carrot into thin slices on the cutting board",
            "Tighten the screw into the wooden plank with the drill",
        ),
        anti_examples=(
            "Reaching for a tool", "Idle pauses", "Pure hand repositioning",
        ),
        micro_type="state_change",
        parent="event",
    ),
)


_CYCLICAL = ArchetypeConfig(
    archetype_id="cyclical",
    display_name="周期/节律型",
    display_name_en="Cyclical / Rhythmic Activity",
    topology="periodic",
    description=(
        "Rhythm-oriented activity with repeated cycles of the same motion or "
        "operation. Core action pattern repeats with intensity/speed variation."
    ),
    typical_videos=(
        "fitness exercise", "yoga routine", "dance choreography",
        "weightlifting sets", "jump rope", "martial arts drills",
        "line dance", "assembly line",
    ),
    classification_signals=(
        "Same motion pattern repeating; body returns to starting position; "
        "intensity or speed changes across the video; "
        "typical gym/studio/outdoor training environment."
    ),

    l1=LevelConfig(
        enabled=True,
        name="Macro-Unit / Routine",
        name_zh="大循环单元",
        definition=(
            "A complete repetition logic or routine section organized by "
            "intensity, rhythm, or movement type. "
            "Typical pattern: Warm-up → High-intensity Sets → Cool-down."
        ),
        boundary_signals=(
            "Movement type switch (stretching → lifting); "
            "intensity change (slow → fast → slow); "
            "rest/transition intervals."
        ),
        examples=(
            "Warm-up dynamic stretching sequence targeting legs and hips",
            "High-intensity interval of weighted squats with progressive load",
            "Cool-down static stretches and deep breathing recovery",
        ),
        anti_examples=(
            "One single repetition (too fine, belongs at L3)",
        ),
    ),

    l2=LevelConfig(
        enabled=True,  # v4: ENABLED (was disabled in v3 performance)
        name="Sub-Unit / Set",
        name_zh="小节单元",
        definition=(
            "A fixed action combination or exercise set within a routine section. "
            "Groups 4-20 repetitions of the same movement into a coherent unit. "
            "E.g., 'left step 4-beat → right step 4-beat' or '10 push-ups set'."
        ),
        boundary_signals=(
            "Movement direction change (left→right); "
            "brief pause between sets; "
            "transition to different variation of the same exercise."
        ),
        examples=(
            "Set of 10 bicep curls with the left arm",
            "8-count step sequence to the right in line dance",
            "One sun salutation flow in yoga",
        ),
        anti_examples=(
            "A single repetition (too fine, belongs at L3)",
            "The entire warm-up section (too coarse, belongs at L1)",
        ),
    ),

    l3=LevelConfig(
        enabled=True,
        name="Beat / Step",
        name_zh="单拍动作",
        definition=(
            "A single completed repetition, cycle, strike, or step. "
            "Start = initiation of the motion. End = return to starting position. "
            "Note: post_state may be similar or identical to pre_state."
        ),
        boundary_signals=(
            "Start of movement initiation; "
            "reaching peak position; "
            "return to starting posture."
        ),
        examples=(
            "One complete push-up: chest lowers to floor then pushes back up",
            "One jump rope cycle: feet leave ground, rope passes under, feet land",
            "One bicep curl: arm extends then curls weight to shoulder",
        ),
        anti_examples=(
            "Half a repetition", "Rest pause between sets",
        ),
        micro_type="repetition_unit",
        parent="event",  # v4: parent=event (L2 now enabled); falls back to phase if L2 empty
    ),
)


_CINEMATIC = ArchetypeConfig(
    archetype_id="cinematic",
    display_name="影视/剧情型",
    display_name_en="Cinematic / Scripted Narrative",
    topology="sequence",
    description=(
        "Scripted or edited narrative content with scene structure, "
        "story arc progression, and deliberate cinematography."
    ),
    typical_videos=(
        "movie clips", "short films", "TV drama scenes",
        "music videos with story", "animated sequences", "commercials with narrative",
    ),
    classification_signals=(
        "Professional cinematography/editing; scene transitions; "
        "multiple characters with dialogue; emotional arc (tension/release); "
        "deliberate camera angles, lighting changes, score/music."
    ),

    l1=LevelConfig(
        enabled=True,
        name="Narrative Act / Scene Group",
        name_zh="叙事幕/场景组",
        definition=(
            "A distinct narrative act, emotional stage, or scene group. "
            "Organized by story arc progression. "
            "Typical pattern: Setup → Confrontation → Climax → Resolution."
        ),
        boundary_signals=(
            "Major narrative turning point; emotional tone shift (tense→relaxed); "
            "significant spatiotemporal jump; character constellation change; "
            "marked editing rhythm change (montage → slow)."
        ),
        examples=(
            "Opening act: protagonist arrives at the abandoned warehouse alone",
            "Confrontation: heated argument escalates in the office",
            "Resolution: characters reconcile in the park at sunset",
        ),
        anti_examples=(
            "Single camera cut (cuts ≠ scene boundaries)",
            "One line of dialogue",
        ),
    ),

    l2=LevelConfig(
        enabled=True,
        name="Scene Unit",
        name_zh="场景单元",
        definition=(
            "A continuous narrative unit within the SAME space-time and character set. "
            "All shots within the same scene (shot/reverse-shot, close-ups, wide shots) "
            "are grouped into ONE event — these are intra-scene cuts. "
            "Intercut or parallel-edited sequences that jump to a DIFFERENT location "
            "or character group are inter-scene cuts and MUST be separate events."
        ),
        boundary_signals=(
            "**Inter-Scene (split)**: Cut to a different location or time "
            "(office → flashback to childhood room, cross-cutting between "
            "two parallel storylines). "
            "**Intra-Scene (do NOT split)**: Shot/reverse-shot during dialogue, "
            "zoom to close-up within the same scene."
        ),
        examples=(
            "Two friends discuss plans over coffee (shown through multiple cuts and angles)",
            "A tense chase sequence through a warehouse hallway",
            "Flashback sequence showing the character as a child in a garden",
        ),
        anti_examples=(
            "Splitting one continuous dialogue scene into events per speaker turn "
            "— shot/reverse-shot is intra-scene, keep as one event",
            "A reaction shot of the same character in the same location (too fine)",
        ),
    ),

    l3=LevelConfig(
        enabled=True,
        name="Interaction Beat / Expression Change",
        name_zh="交互节拍/表情变化",
        definition=(
            "A complete social/physical interaction beat OR a distinct shift in "
            "facial emotion or focus. "
            "interaction_unit: handshake, object handover, dialogue turn with visual cue. "
            "expression_change: smile→serious, surprise reaction, look away→eye contact."
        ),
        boundary_signals=(
            "Social cue: hand extension, eye contact established/broken, head nod; "
            "emotional cue: eyebrow raise, mouth corner lift, tension release; "
            "camera/editing: cut to reaction shot, zoom to face."
        ),
        examples=(
            "Handshake between two characters meeting for the first time",
            "Character's expression shifts from confident smile to visible worry",
            "Object handover — protagonist passes the envelope across the table",
        ),
        anti_examples=(
            "Ambient background movement",
            "Camera wobble or minor reframing",
        ),
        micro_type="interaction_unit",
        parent="event",
    ),
)


_VLOG = ArchetypeConfig(
    archetype_id="vlog",
    display_name="Vlog/日常型",
    display_name_en="Vlog / Daily Life",
    topology="sequence",
    description=(
        "Personal-perspective content driven by location changes, topic shifts, "
        "or daily activity flow. Creator is typically on-camera or narrating."
    ),
    typical_videos=(
        "travel vlog", "daily routine", "food/restaurant review",
        "moving-in vlog", "shopping haul", "day-in-my-life",
    ),
    classification_signals=(
        "First-person or creator-facing camera; location/environment transitions; "
        "casual narration to camera; topic-driven segments; "
        "transition effects or jump cuts between activities."
    ),

    l1=LevelConfig(
        enabled=True,
        name="Topic / Location Segment",
        name_zh="话题/地点段落",
        definition=(
            "A segment organized by location change or topic shift. "
            "Each phase covers a distinct place, activity context, or conversation topic. "
            "Typical pattern: Hotel departure → Sightseeing A → Lunch → Sightseeing B."
        ),
        boundary_signals=(
            "Location change (indoor→outdoor, restaurant→street); "
            "explicit topic transition ('now let's talk about...'); "
            "transition effects (fade, jump cut to new setting)."
        ),
        examples=(
            "Morning routine and getting ready at the hotel room",
            "Exploring the local market and trying street food stalls",
            "Evening reflection and sunset viewing from the rooftop",
        ),
        anti_examples=(
            "Single camera angle change within the same activity",
        ),
    ),

    l2=LevelConfig(
        enabled=True,
        name="Activity / Interaction Clip",
        name_zh="活动/交互片段",
        definition=(
            "A continuous clip focused on a single activity, subject, and location. "
            "Inter-scene cuts (host → B-roll footage, Person A → Person B interview) "
            "create SEPARATE events. "
            "Intra-scene cuts (zoom on the host's face, angle change on the same subject) "
            "do NOT — keep as one event."
        ),
        boundary_signals=(
            "**Inter-Scene (split)**: Cut to a different location, subject, or visual "
            "modality (host → B-roll, Person A → Person B). "
            "**Intra-Scene (do NOT split)**: Camera angle change on the same subject "
            "in the same location."
        ),
        examples=(
            "Interview segment with participant Anthony",
            "Kayaking action sequence shown as B-roll",
            "Tasting a local dish and giving a reaction to the camera",
        ),
        anti_examples=(
            "Merging an interview segment with the separate B-roll footage it describes "
            "— these are inter-scene cuts and MUST be two events",
            "Merging interviews from two different people into one event",
        ),
    ),

    l3=LevelConfig(
        enabled=True,
        name="Key Moment",
        name_zh="关键时刻",
        definition=(
            "A moment with clear information increment — first impression, "
            "key reaction, discovery, or decision point. "
            "Focus on moments where NEW information is produced."
        ),
        boundary_signals=(
            "First-time reaction (first bite, first view); "
            "emotional response (surprise, delight, disappointment); "
            "decision moment (choosing an item, taking a turn)."
        ),
        examples=(
            "First taste of the local dish with visible reaction",
            "Turning the corner and seeing the landmark for the first time",
            "Receiving unexpected good news visible in facial expression",
        ),
        anti_examples=(
            "Walking without events", "Looking at camera with no information change",
        ),
        micro_type="interaction_unit",
        parent="event",
    ),
)


_EDUCATIONAL = ArchetypeConfig(
    archetype_id="educational",
    display_name="教育/知识型",
    display_name_en="Educational / Knowledge",
    topology="procedural",
    description=(
        "Knowledge delivery as the primary goal. May involve lectures, "
        "demonstrations, experiments, or whiteboard explanations."
    ),
    typical_videos=(
        "classroom lecture", "online course", "science experiment",
        "whiteboard tutorial", "coding tutorial", "documentary with narration",
    ),
    classification_signals=(
        "PPT/whiteboard/blackboard visible; instructor speaking to camera/audience; "
        "structured knowledge delivery; demonstration of concepts; "
        "text/diagrams appearing on screen."
    ),

    l1=LevelConfig(
        enabled=True,
        name="Knowledge Module",
        name_zh="知识模块",
        definition=(
            "A distinct teaching topic or knowledge unit. "
            "Each module covers one concept, theorem, or experimental setup. "
            "Typical pattern: Introduction → Core Explanation → Demonstration → Summary."
        ),
        boundary_signals=(
            "PPT slide change to new chapter; blackboard cleared and rewritten; "
            "topic transition ('now let us move on to...'); "
            "experimental setup replacement."
        ),
        examples=(
            "Introduction of the concept of photosynthesis with diagrams",
            "Step-by-step derivation of the quadratic formula on whiteboard",
            "Live demonstration of chemical reaction between acid and base",
        ),
        anti_examples=(
            "A single sentence or definition (too fine)",
        ),
    ),

    l2=LevelConfig(
        enabled=True,
        name="Explanation / Demo Unit",
        name_zh="讲解/演示单元",
        definition=(
            "A focused explanation or demonstration around one specific concept "
            "within the module, in a consistent visual context. "
            "Inter-scene cuts (instructor on camera → screen recording / animation / "
            "experiment close-up) create SEPARATE events. "
            "Intra-scene cuts (zoom on whiteboard, angle change on the same instructor) "
            "do NOT — keep as one event."
        ),
        boundary_signals=(
            "**Inter-Scene (split)**: Cut between instructor-on-camera and "
            "screen recording / animation / diagram close-up / experiment footage. "
            "**Intra-Scene (do NOT split)**: Camera angle or zoom change on the "
            "same instructor at the same whiteboard/desk. "
            "Other signals: focus shift to new sub-concept; transition from theory to example."
        ),
        examples=(
            "Instructor explaining a concept while pointing at the whiteboard",
            "Screen recording showing code execution in an IDE",
            "Close-up of a chemical reaction in a beaker on the lab bench",
        ),
        anti_examples=(
            "Merging an instructor's on-camera lecture with the separate "
            "screen recording they are narrating — inter-scene cut, MUST be two events",
            "Writing a single word (too fine)",
        ),
    ),

    l3=LevelConfig(
        enabled=True,
        name="Key Info Beat",
        name_zh="关键信息点",
        definition=(
            "A moment where new information appears or a key state change occurs: "
            "new term first written, formula step completed, experiment visibly reacts."
        ),
        boundary_signals=(
            "New text/symbol appearing on screen; "
            "experimental substance visibly changing; "
            "instructor pointing at key element; "
            "graph/chart transition."
        ),
        examples=(
            "Instructor writes 'E = mc²' on the whiteboard for the first time",
            "Chemical solution changes from clear to purple upon adding reagent",
            "New slide appears showing the comparison chart of results",
        ),
        anti_examples=(
            "Instructor pacing without new content",
            "Repeating previously shown information",
        ),
        micro_type="state_change",
        parent="event",
    ),
)


_SPORTS_MATCH = ArchetypeConfig(
    archetype_id="sports_match",
    display_name="比赛/竞技型",
    display_name_en="Sports Match / Competition",
    topology="sequence",
    description=(
        "Competitive event with structured rounds, periods, or plays. "
        "Outcome-driven with scoring and rule-based phases."
    ),
    typical_videos=(
        "football match", "basketball game", "tennis match",
        "boxing round", "esports match", "swimming race",
    ),
    classification_signals=(
        "Scoreboard visible; referee/officials present; "
        "structured back-and-forth play; crowd/audience; "
        "uniforms/team colors; starting whistle or signals."
    ),

    l1=LevelConfig(
        enabled=True,
        name="Match Phase",
        name_zh="赛事阶段",
        definition=(
            "A structural period defined by the sport's rules. "
            "Typical: First Half → Halftime → Second Half, or Round 1 → Round 2."
        ),
        boundary_signals=(
            "Whistle/signal; scoreboard change showing period transition; "
            "players leaving/entering the field; timeout/break visuals."
        ),
        examples=(
            "First half of the football match from kickoff to halftime whistle",
            "Third set of the tennis match starting from deuce court",
            "Round 2 of the boxing match from bell to bell",
        ),
        anti_examples=(
            "A single play/rally (too fine, belongs at L2)",
        ),
    ),

    l2=LevelConfig(
        enabled=True,
        name="Rally / Play",
        name_zh="回合/攻防",
        definition=(
            "A complete rally, play, or scoring sequence. "
            "Starts with possession/serve, ends with score/turnover/dead ball."
        ),
        boundary_signals=(
            "Ball possession change; serve initiation; "
            "goal/point scored; out-of-bounds; dead ball/time stop."
        ),
        examples=(
            "Counter-attack sequence ending with a shot on goal",
            "Tennis rally from serve to point won by net volley",
            "Fast break leading to a slam dunk and celebration",
        ),
        anti_examples=(
            "Walking back to position (not an action sequence)",
        ),
    ),

    l3=LevelConfig(
        enabled=True,
        name="Key Action",
        name_zh="关键动作",
        definition=(
            "A decisive physical action within a play — the shot, score, foul, "
            "save, or turning-point moment."
        ),
        boundary_signals=(
            "Ball striking foot/bat/racket; ball entering goal/net; "
            "body contact (foul/tackle); celebration start."
        ),
        examples=(
            "Striker shoots the ball into the top-left corner of the goal",
            "Goalkeeper dives right and deflects the penalty kick",
            "Player executes a spinning backhand that lands on the baseline",
        ),
        anti_examples=(
            "Players jogging without the ball",
            "Camera panning across the crowd",
        ),
        micro_type="state_change",
        parent="event",
    ),
)


_CONTINUOUS = ArchetypeConfig(
    archetype_id="continuous",
    display_name="连续观察型",
    display_name_en="Continuous Observation",
    topology="observation",
    description=(
        "Long-take or minimally edited continuous recording focused on a "
        "subject's ongoing activity. No human-imposed temporal structure."
    ),
    typical_videos=(
        "wildlife documentary", "tightrope walking", "surveillance footage",
        "nature timelapse", "extreme sports POV", "surgery recording",
        "factory monitoring", "animal behavior study",
    ),
    classification_signals=(
        "Single continuous shot or minimal cuts; "
        "no human-imposed chapter structure; "
        "focus on natural/ongoing activity without narration-driven segmentation; "
        "slow state changes over extended periods."
    ),

    l1=LevelConfig(
        enabled=True,
        name="Focus State",
        name_zh="焦点状态",
        definition=(
            "A major attention focus or state of the observed subject. "
            "Shifts when the core object/activity fundamentally changes. "
            "E.g., 'eagle perched' → 'eagle hunting' → 'eagle feeding'."
        ),
        boundary_signals=(
            "Subject behavior mode change (resting → active → resting); "
            "environment/lighting significant shift; "
            "new subject entering the primary focus area."
        ),
        examples=(
            "Lion pride resting in the shade during midday",
            "Tightrope walker attempting the first crossing over the canyon",
            "Storm cell approaching from the west horizon",
        ),
        anti_examples=(
            "Minor head turn or tail flick (too fine)",
            "Entire 30-minute observation (too coarse if multiple focus states)",
        ),
    ),

    l2=LevelConfig(
        enabled=True,
        name="Behavior State",
        name_zh="行为状态",
        definition=(
            "An observable behavioral state or sub-activity within a focus. "
            "Marks when the subject's visible behavior noticeably transitions."
        ),
        boundary_signals=(
            "Posture change; movement speed change; "
            "interaction with new object/individual; "
            "direction reversal; visible emotional state shift."
        ),
        examples=(
            "Tightrope walker pauses to regain balance midway",
            "Eagle dives from perch toward the water surface",
            "Cat stalks prey behind the bush, crouching low",
        ),
        anti_examples=(
            "Breathing rhythm change (too subtle)",
        ),
    ),

    l3=LevelConfig(
        enabled=True,
        name="Movement Detail",
        name_zh="动作细节",
        definition=(
            "A fine-grained physical adjustment or motion within a behavior state. "
            "Captures the smallest visible intentional movement."
        ),
        boundary_signals=(
            "Limb position change; weight shift; "
            "head/gaze direction change; "
            "tool/object manipulation onset/completion."
        ),
        examples=(
            "Walker extends arms to the side for balance after a wobble",
            "Eagle tucks wings and enters steep dive angle",
            "Cat's hindquarters lower as it prepares to pounce",
        ),
        anti_examples=(
            "Wind blowing fur/feathers (not intentional movement)",
            "Camera shake or zoom (not subject movement)",
        ),
        micro_type="state_change",
        parent="event",
    ),
)


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

PARADIGMS: dict[str, ArchetypeConfig] = {
    "tutorial": _TUTORIAL,
    "cyclical": _CYCLICAL,
    "cinematic": _CINEMATIC,
    "vlog": _VLOG,
    "educational": _EDUCATIONAL,
    "sports_match": _SPORTS_MATCH,
    "continuous": _CONTINUOUS,
}

# Backward compat: old archetype names → new paradigm configs
ARCHETYPES: dict[str, ArchetypeConfig] = {
    **PARADIGMS,
    "performance": _CYCLICAL,  # old name → new config
    # talk/ambient: not in registry → KeyError if accessed (filtered in Stage 1)
}


def get_paradigm(paradigm_id: str) -> ArchetypeConfig:
    """Get paradigm config by ID. Supports old archetype names via migration."""
    # Migrate old name if needed
    if paradigm_id in ARCHETYPE_TO_PARADIGM:
        migrated = ARCHETYPE_TO_PARADIGM[paradigm_id]
        if migrated is None:
            raise KeyError(
                f"Archetype '{paradigm_id}' was removed in v4 "
                f"(filtered in Stage 1). Valid paradigms: {sorted(PARADIGMS.keys())}"
            )
        paradigm_id = migrated

    if paradigm_id not in PARADIGMS:
        raise KeyError(
            f"Unknown paradigm '{paradigm_id}'. "
            f"Valid: {sorted(PARADIGMS.keys())}"
        )
    return PARADIGMS[paradigm_id]


# Backward compat alias
get_archetype = get_paradigm


def archetype_to_topology(archetype_id: str) -> TopologyType:
    """Map archetype to its topology type."""
    return ARCHETYPE_TO_TOPOLOGY.get(archetype_id, "procedural")


def topology_to_default_archetype(topology: str) -> str:
    """Map topology back to a default paradigm (for backward compatibility)."""
    return TOPOLOGY_TO_DEFAULT_PARADIGM.get(topology, "continuous")


def get_active_levels(paradigm_id: str) -> list[str]:
    """Return list of active level names for a paradigm: ['l1'], ['l1','l2'], etc."""
    cfg = get_paradigm(paradigm_id)
    levels = ["l1"]  # L1 always active
    if cfg.l2.enabled:
        levels.append("l2")
    if cfg.l3.enabled:
        levels.append("l3")
    return levels


def get_l3_parent_type(paradigm_id: str) -> str | None:
    """Return L3 parent type ('event' or 'phase') or None if L3 disabled."""
    cfg = get_paradigm(paradigm_id)
    if not cfg.l3.enabled:
        return None
    return cfg.l3.parent


def migrate_archetype_to_paradigm(archetype_id: str) -> str | None:
    """Migrate old archetype name to new paradigm name. Returns None if filtered."""
    return ARCHETYPE_TO_PARADIGM.get(archetype_id, archetype_id)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Classification Prompt (paradigm + domain + feasibility)
# ─────────────────────────────────────────────────────────────────────────────

def _format_paradigm_table_for_prompt() -> str:
    """Format paradigm descriptions as a list for the classification prompt."""
    lines = []
    for pid, cfg in PARADIGMS.items():
        lines.append(
            f"- **{pid}** ({cfg.display_name_en}): {cfg.description}\n"
            f"  Typical: {', '.join(cfg.typical_videos[:3])}.\n"
            f"  Signals: {cfg.classification_signals}"
        )
    return "\n".join(lines)


_CLASSIFICATION_PROMPT = """\
You are given a {duration}s video clip (timestamps 0 to {duration}) with {n_frames} frames.

Your task has FOUR parts:
1. Classify the video's temporal structure (paradigm)
2. Classify the video's content topic (domain)
3. Assess annotation feasibility
4. Write a global video caption

────────────────────────────────────────────────
## PART 1 — PARADIGM (temporal structure)

Classify into exactly ONE paradigm based on the video's dominant temporal structure:

{paradigm_table}

### If the video does NOT fit any paradigm above:
- **Talk-dominant** (people in conversation, minimal physical action): set feasibility.skip=true, skip_reason="talk_dominant"
- **Ambient/static** (no identifiable subject, no sequential progression): set feasibility.skip=true, skip_reason="ambient_static"

────────────────────────────────────────────────
## PART 2 — DOMAIN (content topic)

Domain is WHAT the video is about — orthogonal to paradigm (HOW it is structured).
A cooking tutorial and a cooking competition share the SAME domain but DIFFERENT paradigms.

Choose domain_l2 (fine-grained) from this hierarchy — domain_l1 is determined automatically:
{domain_l2_list}

If none fits, use domain_l2="other".

────────────────────────────────────────────────
## PART 3 — FEASIBILITY

Assess whether this video is worth annotating with hierarchical temporal structure.
Consider: visual dynamics, number of distinct segments, clarity of boundaries.

────────────────────────────────────────────────
## PART 4 — VIDEO CAPTION

Write a detailed description of the entire video (3-5 sentences).
Cover: setting/environment, main subjects, key objects, overall progression, and outcome.
Every statement must be grounded in what is visible in the frames.

────────────────────────────────────────────────
## OUTPUT JSON

{{
  "paradigm": "<one of: {paradigm_ids}>",
  "paradigm_confidence": 0.85,
  "paradigm_reason": "<one sentence explaining the paradigm decision>",
  "domain_l2": "<one of the domain_l2 categories above, or 'other'>",
  "video_caption": "<3-5 sentences: detailed description of the entire video>",
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
  }}
}}

Rules:
- paradigm_confidence: 0.0 to 1.0, your confidence in the paradigm choice.
- feasibility.score: 0.0 to 1.0, how suitable this video is for hierarchical annotation.
- feasibility.skip: true if the video should NOT be annotated (talk, ambient, low dynamics).
- feasibility.skip_reason: null if skip=false, else one of: "talk_dominant", "ambient_static", "low_visual_dynamics", "too_short".
- visual_dynamics: "high" (frequent visible state changes), "medium" (moderate), "low" (mostly static).
- estimated_n_phases / estimated_n_events: rough count, not exact."""


def get_classification_prompt(n_frames: int, duration_sec: int) -> str:
    """Build the Stage 1 paradigm + domain + feasibility classification prompt."""
    return _CLASSIFICATION_PROMPT.format(
        n_frames=n_frames,
        duration=duration_sec,
        paradigm_table=_format_paradigm_table_for_prompt(),
        paradigm_ids=", ".join(sorted(PARADIGM_IDS)),
        domain_l2_list=_format_domain_l2_for_prompt(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Paradigm-Driven Level Prompts
# ─────────────────────────────────────────────────────────────────────────────

_ARCHETYPE_MERGED_BASE = """\
You are given a {duration}s video clip (timestamps 0 to {duration}) with {n_frames} frames.
This video has been classified as archetype: **{archetype}** ({archetype_display}).

Your task: Annotate this video with a hierarchical L1{l2_header}{l3_header} structure \
using the archetype-specific definitions below.

## L1 — {l1_name}

**Definition**: {l1_definition}

**Boundary Signals**: {l1_boundary_signals}

**Good examples**: {l1_examples}
**NOT valid at L1**: {l1_anti_examples}

**Rules**:
- Skip intros, outros, static non-activity spans.
- Phases do NOT need to cover the entire video — gaps are expected.
- Do NOT split by camera cuts alone.
- A single-phase video is valid if it is one continuous {archetype} segment.
- phase_name MUST be a descriptive phrase of 5–15 words.
- narrative_summary MUST be 2–3 sentences.
{l2_section}{l3_section}
## L3 FEASIBILITY ASSESSMENT

After annotating L1 and L2, assess whether this video supports fine-grained L3 annotation.

{l3_feasibility_criteria}

Consider:
- Are there enough L2 events (>=3) with sufficient duration (>=5s each) for micro-action grounding?
- Are fine-grained visual state changes or physical interactions clearly visible in the frames?
- Is the framing/resolution adequate to observe micro-actions at 2fps?

## VISUAL SIGNAL REFERENCE
- Scene/Space: Background/layout/location change, character entry/exit.
- Subject Behavior: Pose transition, gaze direction, speed change, interaction start/end.
- Object State: Appearance/texture/color/position/quantity change.
- Narrative/Emotion: Shift in emotional tone, topic change, conflict resolution.
- Camera/Editing: Rhythm change, montage sequence, focus shift, cut to close-up.

## OUTPUT JSON
{{
  "summary": "<one sentence summarizing the video>",
  "global_phase_criterion": "<one sentence: why split into these phases>",
  "l3_feasibility": {{
    "suitable": true,
    "reason": "<1 sentence: why L3 micro-action annotation is/isn't feasible for this video>",
    "estimated_l3_actions": 8
  }},
  "macro_phases": [
    {{
      "phase_id": 1,
      "start_time": 5,
      "end_time": 60,
      "phase_name": "<5-15 word descriptive phrase>",
      "narrative_summary": "<2-3 sentences: what happens in this phase, objects involved, outcome>",
      "event_split_criterion": "<one sentence: why this phase has/lacks events>"{events_field}
    }}
  ]
}}

## QUALITY CHECKLIST
- [ ] Each L1 phase represents a distinct semantic stage?
- [ ] Boundary triggers are specific and reproducible?
- [ ] Criterion fields describe splitting LOGIC, not content?
{l2_checklist}{l3_note}"""


def _build_l2_section(cfg: ArchetypeConfig) -> str:
    if not cfg.l2.enabled:
        return (
            '\n## L2 — Events\n'
            'L2 events are **NOT applicable** for this archetype. '
            'Output `"events": []` for every phase.\n'
        )

    examples_str = "; ".join(f'"{e}"' for e in cfg.l2.examples[:3])
    anti_str = "; ".join(f'"{e}"' for e in cfg.l2.anti_examples[:2])

    return f"""
## L2 — {cfg.l2.name}

**Definition**: {cfg.l2.definition}

**Boundary Signals**: {cfg.l2.boundary_signals}

**Good examples**: {examples_str}
**NOT valid at L2**: {anti_str}

**Rules**:
- Detect events nested inside each L1 phase.
- Events must not overlap. Use absolute integer seconds.
- `"events": []` is valid if the phase has no sub-structure.
- instruction MUST be 8–20 words: WHAT + WITH WHICH objects + WHAT outcome.
"""


def _build_l3_note(cfg: ArchetypeConfig) -> str:
    if not cfg.l3.enabled:
        return ""
    return (
        f"\n**Note**: After this annotation, L3 ({cfg.l3.name}) will be annotated "
        f"in a separate pass with finer granularity."
    )


def _build_l3_feasibility_criteria(cfg: ArchetypeConfig) -> str:
    """Build paradigm-specific L3 feasibility criteria for the merged prompt."""
    if not cfg.l3.enabled:
        return (
            "L3 annotation is **disabled** for this paradigm. "
            'Set `"l3_feasibility": {"suitable": false, "reason": "L3 disabled for this paradigm", "estimated_l3_actions": 0}`.'
        )
    return (
        f"For **{cfg.archetype_id}** videos, L3 = **{cfg.l3.name}**: {cfg.l3.definition}\n\n"
        f"L3 boundary signals: {cfg.l3.boundary_signals}\n\n"
        f"Set `suitable=false` if: the video lacks clear visual detail for these micro-actions, "
        f"or L2 events are too short/abstract to decompose further."
    )


def get_archetype_merged_prompt(
    archetype_id: str,
    n_frames: int,
    duration_sec: int,
) -> str:
    """Build the paradigm-driven merged L1(+L2) annotation prompt.

    Accepts both old archetype names and new paradigm names.
    v4: adds dense caption fields (scene_description for L1, dense_caption for L2).
    """
    cfg = get_paradigm(archetype_id)
    # Resolve to canonical paradigm ID
    paradigm_id = cfg.archetype_id

    l1_examples = "; ".join(f'"{e}"' for e in cfg.l1.examples[:3])
    l1_anti = "; ".join(f'"{e}"' for e in cfg.l1.anti_examples[:2])

    l2_header = " / L2" if cfg.l2.enabled else ""
    l3_header = ""  # L3 always a separate pass
    l2_section = _build_l2_section(cfg)
    events_field = (
        ',\n      "events": [\n'
        '        {\n'
        '          "event_id": 1,\n'
        '          "start_time": 5,\n'
        '          "end_time": 28,\n'
        '          "instruction": "<8-20 word description>",\n'
        '          "dense_caption": "<2-4 sentences: detailed process description — actions, objects, spatial relations, state changes>",\n'
        '          "visual_keywords": ["kw1", "kw2"],\n'
        '          "l3_feasible": true,\n'
        '          "l3_reason": "<1 sentence>"\n'
        '        }\n'
        '      ]'
    ) if cfg.l2.enabled else ',\n      "events": []'

    l2_checklist = (
        "- [ ] Each L2 event completes a verifiable unit?\n"
        "- [ ] Events are not too coarse (≈ phase) or too fine (single motion)?\n"
    ) if cfg.l2.enabled else ""

    return _ARCHETYPE_MERGED_BASE.format(
        duration=duration_sec,
        n_frames=n_frames,
        archetype=paradigm_id,
        archetype_display=cfg.display_name_en,
        l1_name=cfg.l1.name,
        l1_definition=cfg.l1.definition,
        l1_boundary_signals=cfg.l1.boundary_signals,
        l1_examples=l1_examples,
        l1_anti_examples=l1_anti,
        l2_header=l2_header,
        l3_header=l3_header,
        l2_section=l2_section,
        l3_section="",
        l3_feasibility_criteria=_build_l3_feasibility_criteria(cfg),
        events_field=events_field,
        l2_checklist=l2_checklist,
        l3_note=_build_l3_note(cfg),
    )


# Backward compat alias
get_paradigm_merged_prompt = get_archetype_merged_prompt


# ─────────────────────────────────────────────────────────────────────────────
# Unified Merged Prompt (v5: fuse classification + annotation into one call)
# ─────────────────────────────────────────────────────────────────────────────

def _format_paradigm_annotation_table() -> str:
    """Build a compact per-paradigm reference table of L1/L2 definitions."""

    # Per-topology annotation tips
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


def _format_l3_feasibility_all_paradigms() -> str:
    """Build L3 feasibility reference for all paradigms."""
    lines = []
    for pid, cfg in PARADIGMS.items():
        if cfg.l3.enabled:
            lines.append(f"- **{pid}**: L3 = {cfg.l3.name} — {cfg.l3.definition[:80]}...")
        else:
            lines.append(f"- **{pid}**: L3 disabled")
    return "\n".join(lines)


_UNIFIED_MERGED_PROMPT = """\
You are given a {duration}s video clip (timestamps 0 to {duration}) with {n_frames} frames.

Your task has TWO parts:
1. **CLASSIFY** the video (paradigm, domain, feasibility, metadata, caption)
2. **ANNOTATE** the video with hierarchical L1 + L2 structure based on the paradigm you identified

════════════════════════════════════════════════
## PART 1 — CLASSIFICATION
════════════════════════════════════════════════

### 1A. PARADIGM (temporal structure)

Classify into exactly ONE paradigm based on the video's dominant temporal structure:

{paradigm_table}

If the video does NOT fit any paradigm above:
- **Talk-dominant** (people in conversation, minimal physical action): set feasibility.skip=true, skip_reason="talk_dominant"
- **Ambient/static** (no identifiable subject, no sequential progression): set feasibility.skip=true, skip_reason="ambient_static"

### 1B. DOMAIN (content topic)

Domain is WHAT the video is about — orthogonal to paradigm (HOW it is structured).
Choose domain_l2 from:
{domain_l2_list}
If none fits, use domain_l2="other".

### 1C. FEASIBILITY

Assess whether this video is worth annotating with hierarchical temporal structure.
Consider: visual dynamics, number of distinct segments, clarity of boundaries.

### 1D. VIDEO CAPTION

Write a detailed description of the entire video (3-5 sentences).
Cover: setting/environment, main subjects, key objects, overall progression, and outcome.
Every statement must be grounded in what is visible in the frames.

════════════════════════════════════════════════
## PART 2 — HIERARCHICAL ANNOTATION
════════════════════════════════════════════════

CRITICAL: In Part 1, you chose ONE paradigm. In this section, you MUST ONLY read and apply \
the L1/L2 definitions for YOUR CHOSEN paradigm. IGNORE the rules for the other 6 paradigms completely.

**If feasibility.skip=true**, output `"macro_phases": []` and skip annotation.

{paradigm_annotation_table}

### UNIVERSAL TEMPORAL & FORMATTING RULES (Apply to ALL paradigms)

**Visual Priority — Intra-Scene vs Inter-Scene Cuts**: \
Not all cuts are equal. You MUST distinguish two types:
- **Intra-Scene Cut** (do NOT split): Camera angle, zoom, or focal length change on the SAME \
subject in the SAME location during the SAME ongoing activity. Examples: wide shot → close-up \
of hands kneading dough; shot/reverse-shot during a dialogue in one room.
- **Inter-Scene Cut** (MUST split): Cut to a DIFFERENT location, subject, person, or visual modality. \
Examples: host on camera → B-roll footage; instructor → screen recording; Person A interview → Person B interview; \
office scene → flashback.
An inter-scene cut MUST trigger a new L2 event boundary, regardless of audio or narrative continuity.

**No Audio Memory**: You are a VISUAL model — base ALL segmentation decisions on what you SEE in the frames. \
Do NOT infer segment boundaries from assumed narration, dialogue, or audio cues that are not visible. \
If a speaker's voice might continue over a visual cut, treat the cut as the boundary, not the voice.

**Describe-before-timestamp**: For each L1 phase and L2 event, first write the descriptive fields \
(phase_name, narrative_summary, instruction, dense_caption), then assign start_time/end_time. \
This grounds your timestamps in actual visual evidence rather than guessing.

**Temporal Logic (Crucial):**
- **Sparsity & Gaps**: Phases/Events do NOT need to cover the entire video. Gaps between annotated segments are expected. Skip intros, outros, and idle spans.
- **No Overlaps**: L2 events must NOT overlap each other in time.
- **Strict Nesting**: Every L2 event MUST be strictly nested within its parent L1 phase timeframe.
- **Anti-Fragmentation**: Intra-scene cuts (angle/zoom changes within the same scene) MUST NOT create new events. An L2 event MUST be >= 5 seconds. However, inter-scene cuts (different location/subject/modality) MUST create a new event even if the resulting segment is short.
- **Empty L2**: `"events": []` is perfectly valid if an L1 phase has no meaningful sub-structure. A single-phase video is also valid.

**Text Generation Constraints:**
- `phase_name`: MUST be a concise descriptive phrase (5–15 words).
- `narrative_summary`: MUST be exactly 2–3 sentences detailing what happens and the outcome.
- `instruction`: MUST be an objective description (8–20 words) stating WHAT the action is and WITH WHICH objects. Do NOT use imperative verbs (e.g., avoid "Watch...", "Observe...").
- `dense_caption`: MUST be detailed visual descriptions (2–4 sentences) covering actions, objects, spatial relations, and state changes.

### L3 FEASIBILITY (per-phase AND per-event)

For EACH L1 phase AND each L2 event, assess whether it supports fine-grained L3 micro-action annotation.
Different phases/events in the same video may have very different L3 suitability \
(e.g., an active cooking event is feasible, but a conversation event is not).

Per-paradigm L3 reference:
{l3_feasibility_ref}

Set `l3_feasible=false` for a phase/event if:
- It is dominated by talking, interviews, or static scenes with no physical actions.
- It is too short or abstract to decompose into micro-actions.
- Visual detail is insufficient (distant shot, blurry, occluded).
Set `l3_feasible=true` if it has clear, observable physical actions or state changes at 2fps.

### VISUAL SIGNAL REFERENCE
- Scene/Space: Background/layout/location change, character entry/exit.
- Subject Behavior: Pose transition, gaze direction, speed change, interaction start/end.
- Object State: Appearance/texture/color/position/quantity change.
- Narrative/Emotion: Shift in emotional tone, topic change, conflict resolution.
- Camera/Editing: Rhythm change, montage sequence, focus shift, cut to close-up.

════════════════════════════════════════════════
## OUTPUT JSON
════════════════════════════════════════════════

{{
  "paradigm": "<one of: {paradigm_ids}>",
  "paradigm_confidence": 0.85,
  "paradigm_reason": "<one sentence explaining the paradigm decision>",
  "domain_l2": "<one of the domain_l2 categories above, or 'other'>",
  "video_caption": "<3-5 sentences: detailed description of the entire video>",
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
  "summary": "<one sentence summarizing the video>",
  "global_phase_criterion": "<one sentence: why split into these phases>",
  "macro_phases": [
    {{
      "phase_id": 1,
      "start_time": 5,
      "end_time": 60,
      "phase_name": "<5-15 word descriptive phrase>",
      "narrative_summary": "<2-3 sentences>",
      "event_split_criterion": "<one sentence: why this phase has/lacks events>",
      "l3_feasible": true,
      "l3_reason": "<1 sentence: why this phase does/doesn't support L3 micro-action annotation>",
      "events": [
        {{
          "event_id": 1,
          "start_time": 5,
          "end_time": 28,
          "instruction": "<8-20 word description>",
          "dense_caption": "<2-4 sentences: detailed process description>",
          "visual_keywords": ["kw1", "kw2"],
          "l3_feasible": true,
          "l3_reason": "<1 sentence: why this event does/doesn't support L3 micro-actions>"
        }}
      ]
    }}
  ]
}}

## RULES & CONSTRAINTS
1. **Formatting**: Output strictly valid JSON. No markdown code blocks around it.
2. **Timestamps**: All `start_time` and `end_time` MUST be absolute integer seconds within [0, {duration}].
3. **Strict Nesting**: Every L2 event's `[start_time, end_time]` MUST be entirely within its parent L1 `macro_phase` boundaries. Do not hallucinate timestamps outside the phase.
4. **Granularity (Anti-L3)**: L2 events are NOT momentary actions. Any event shorter than 5 seconds likely belongs to L3. Group related actions into a larger continuous event.
5. **Enums & Values**:
   - `paradigm_confidence` / `feasibility.score`: Float between 0.0 and 1.0.
   - `feasibility.skip_reason`: null if skip=false, else one of: "talk_dominant", "ambient_static".
   - `visual_dynamics`: "high" | "medium" | "low".
   - `camera_style`: "static_tripod" | "handheld" | "multi_angle" | "first_person".
   - `editing_style`: "continuous" | "jump_cut" | "montage" | "mixed".
6. **Skip Logic**: If `feasibility.skip` is true, output `"macro_phases": []` (empty array).

## QUALITY CHECKLIST
- [ ] Each L1 phase represents a distinct semantic stage per the chosen paradigm's definition?
- [ ] Boundary triggers match the paradigm-specific signals, not borrowed from other paradigms?
- [ ] L2 event timestamps are strictly within parent L1 phase boundaries?
- [ ] No L2 event is shorter than 5 seconds?
- [ ] Descriptive fields (phase_name, instruction, dense_caption) are grounded in visible frames?"""


def get_unified_merged_prompt(n_frames: int, duration_sec: int) -> str:
    """Build the unified classification + annotation prompt (v5).

    Replaces the separate get_classification_prompt() + get_archetype_merged_prompt() calls.
    The model self-classifies paradigm and applies the matching annotation rules in one pass.
    """
    return _UNIFIED_MERGED_PROMPT.format(
        n_frames=n_frames,
        duration=duration_sec,
        paradigm_table=_format_paradigm_table_for_prompt(),
        paradigm_ids=", ".join(sorted(PARADIGM_IDS)),
        domain_l2_list=_format_domain_l2_for_prompt(),
        paradigm_annotation_table=_format_paradigm_annotation_table(),
        l3_feasibility_ref=_format_l3_feasibility_all_paradigms(),
    )


_L1_AGGREGATION_PROMPT = """\
You are given key frames from {n_events} annotated events in a {duration}s video clip.

Each frame is labeled with its parent event ID and timestamp.
Below is the full event annotation from a previous analysis step:

```json
{events_json}
```

**Video summary**: {summary}
**Phase grouping criterion**: {global_phase_criterion}

Your task: Group these events into L1 macro-phases (thematic segments).

════════════════════════════════════════════════
## L1 — Thematic Segment
════════════════════════════════════════════════

**Definition**: A broad segment of the video unified by a single overarching theme, goal, \
location, or activity mode. A new L1 phase starts when there is a **major shift** in what \
the video is about.

**Boundary Signals** (any ONE is sufficient):
- **Goal/intent shift**: The purpose changes (preparing → cooking → plating).
- **Location/setting change**: The scene moves to a visibly different place.
- **Subject/topic switch**: The main focus shifts to a different person, object, or topic.
- **Activity mode change**: The nature of activity changes (explaining → demonstrating, \
warm-up → high-intensity, interview → B-roll montage).
- **Explicit transition**: Fade/dissolve, title card, or clear editing break.

════════════════════════════════════════════════
## RULES
════════════════════════════════════════════════

1. Assign EVERY event to exactly one phase using `member_event_ids`.
2. Output 1-6 phases. A single-phase grouping is valid for uniform-theme videos.
3. Events within a phase must be temporally contiguous — no interleaving. \
(Phase A's events must all come before Phase B's events in time.)
4. `phase_name`: 5-15 word descriptive phrase for the thematic segment.
5. `narrative_summary`: 2-3 sentences covering what happens in the phase.
6. `event_split_criterion`: one sentence explaining why events within this phase are distinct.
7. `l3_feasible`: true if ANY member event has l3_feasible=true in the event list above.
8. Do NOT output start_time or end_time — these are computed automatically from member events.

════════════════════════════════════════════════
## OUTPUT JSON
════════════════════════════════════════════════

{{
  "macro_phases": [
    {{
      "phase_id": 1,
      "phase_name": "<5-15 word descriptive phrase>",
      "narrative_summary": "<2-3 sentences>",
      "event_split_criterion": "<one sentence>",
      "l3_feasible": true,
      "l3_reason": "<1 sentence>",
      "member_event_ids": [1, 2, 3]
    }}
  ]
}}

## RULES
1. Output strictly valid JSON. No markdown code blocks.
2. Every event_id from the input MUST appear in exactly one phase's member_event_ids.
3. member_event_ids within each phase must be in ascending order.
4. Phases must be ordered by their earliest member event."""


def get_l1_aggregation_prompt(
    events_json: str,
    summary: str,
    global_phase_criterion: str,
    n_events: int,
    duration_sec: int,
) -> str:
    """Build the L1 aggregation prompt (v7 Stage 2).

    Groups previously annotated L2 events into L1 phases using key frames
    and the full event list as context.
    """
    return _L1_AGGREGATION_PROMPT.format(
        events_json=events_json,
        summary=summary,
        global_phase_criterion=global_phase_criterion,
        n_events=n_events,
        duration=duration_sec,
    )


# ─────────────────────────────────────────────────────────────────────────────
# L2+L3 First Prompt (v8: bottom-up L2 events + inline L3 sub-actions)
# ─────────────────────────────────────────────────────────────────────────────

def _format_paradigm_l2l3_table() -> str:
    """Per-paradigm L2 + L3 definitions for the combined L2L3-first prompt."""
    blocks = []
    for pid, cfg in PARADIGMS.items():
        l2_ex = "; ".join(f'"{e}"' for e in cfg.l2.examples[:2])
        l2_anti = "; ".join(f'"{e}"' for e in cfg.l2.anti_examples[:1])
        block = (
            f"- **{pid}** — L2 = **{cfg.l2.name}**: {cfg.l2.definition}\n"
            f"  Boundary signals: {cfg.l2.boundary_signals}\n"
            f"  Good examples: {l2_ex}\n"
            f"  NOT valid at L2: {l2_anti}"
        )
        if cfg.l3.enabled:
            l3_ex = "; ".join(f'"{e}"' for e in cfg.l3.examples[:2])
            l3_anti = "; ".join(f'"{e}"' for e in cfg.l3.anti_examples[:1])
            block += (
                f"\n  L3 = **{cfg.l3.name}**: {cfg.l3.definition}\n"
                f"  L3 boundary signals: {cfg.l3.boundary_signals}\n"
                f"  L3 good: {l3_ex}\n"
                f"  NOT L3: {l3_anti}"
            )
        else:
            block += "\n  L3: disabled — do NOT output sub_actions for this paradigm."
        blocks.append(block)
    return "\n".join(blocks)


_L2L3_FIRST_PROMPT = """\
You are given a {duration}s video clip (timestamps 0 to {duration}) with {n_frames} frames \
sampled at 2fps.

Your task has FOUR parts:
1. **CLASSIFY** the video (paradigm, domain, feasibility, caption, metadata)
2. **ANNOTATE** L2 events (dense video captioning — the primary task)
3. **ANNOTATE** L3 sub-actions within each L2 event (micro-action grounding)
4. **PROVIDE** aggregation hints for downstream L1 phase grouping

════════════════════════════════════════════════
## PART 1 — CLASSIFICATION
════════════════════════════════════════════════

### 1A. PARADIGM (temporal structure — informational only, does NOT affect annotation)

Classify into exactly ONE paradigm based on the video's dominant temporal structure:

{paradigm_table}

If the video does NOT fit any paradigm above:
- **Talk-dominant** (people in conversation, minimal physical action): set feasibility.skip=true, \
skip_reason="talk_dominant"
- **Ambient/static** (no identifiable subject, no sequential progression): set feasibility.skip=true, \
skip_reason="ambient_static"

### 1B. DOMAIN (content topic)

Domain is WHAT the video is about — orthogonal to paradigm (HOW it is structured).
Choose domain_l2 from:
{domain_l2_list}
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
## PART 2 — L2 EVENT ANNOTATION + L3 SUB-ACTIONS
════════════════════════════════════════════════

**If feasibility.skip=true**, output `"events": []` and skip annotation.

### L2 — Visual Event (Dense Caption)

**Universal definition**: A continuous visual segment focused on **one coherent activity, \
one primary subject, and one consistent scene**.

**IMPORTANT**: After choosing your paradigm in Part 1, use its L2 definition below to \
determine event granularity and boundaries. Only apply the L2 rules for YOUR paradigm:

{paradigm_l2l3_table}

**ONE EVENT = ONE SCENE**: Every event must describe a SINGLE continuous scene. \
If your description contains "then", "followed by", "next", "after that", or describes \
two different activities/locations — you MUST split it into separate events. \
A correct event description should read as ONE ongoing visual activity, not a sequence.

**Critical Rule — Scene Boundaries and Inter-Scene Cuts**:
Some frames are preceded by a `[SCENE BREAK]` marker, which indicates an algorithmically \
detected shot boundary at that point. These markers are strong signals — you SHOULD start \
a new event at each `[SCENE BREAK]` unless the visual content clearly shows the same \
activity continuing in the same location (rare false positive).

Additionally, scan through ALL frames in order. When consecutive frames show a DIFFERENT \
background, location, person, or visual modality — even without a `[SCENE BREAK]` marker — \
that is an **inter-scene cut** and you MUST start a new event. \
The `[SCENE BREAK]` markers catch most cuts, but you may find additional boundaries \
that the algorithm missed.

- **Intra-Scene Cut** (do NOT split): Camera angle, zoom, or focal length change on the \
SAME subject in the SAME location during the SAME ongoing activity. \
Example: wide shot of person kneading dough → close-up of hands kneading = ONE event.
- **Inter-Scene Cut** (MUST split into separate events): Cut to a DIFFERENT location, \
subject, person, or visual modality. Check the **background** — if it changes, it is \
inter-scene. \
Examples: gymnasium → rooftop; host on camera → B-roll footage; Person A interview → \
Person B interview; kitchen → dining room; instructor on camera → screen recording.

**Rules**:
- Events of any duration are allowed, including short events (< 5 seconds). \
If a brief but visually distinct segment exists (e.g., a quick cut-away, transition shot), \
output it as its own event — do NOT merge it into an adjacent event.
- Events must not overlap. Gaps between events are expected.
- `"events": []` is valid if the video has no meaningful sub-structure.

**SELF-CHECK** (mandatory before outputting JSON):
1. Re-read each event's `instruction` and `dense_caption`. If any text contains \
"then", "followed by", "next", "after that", "subsequently", or describes TWO \
different activities/locations — SPLIT that event into multiple events NOW.
2. Check each caption for non-visual language: "explains", "describes", "discusses", \
"talks about". Replace with observable visual descriptions.
3. Verify each event covers only ONE continuous scene with ONE consistent background.

### L3 — Sub-Action Annotation (inline per event)

For EACH L2 event, annotate L3 micro-actions as `sub_actions` \
within that event. Each sub-action is an atomic, visible action of 2-6 seconds.

**L3 ANNOTATION** (per-event):
- Always attempt to decompose the event into sub-actions. Set `l3_feasible=true` for \
all events where you can identify at least one visible action or state change.
- Only set `l3_feasible=false` if the event is extremely short (< 3 seconds) or contains \
absolutely no observable action (e.g., a completely static frame with zero motion).
- For events with `l3_feasible=false`, output `"sub_actions": []`.

**L3 Rules**:
- Each sub-action: 2-6 seconds, describing ONE atomic visible action.
- Use absolute integer seconds from the full video timeline.
- Allow gaps between sub-actions (no forced full coverage).
- Sub-actions must be within their parent event's [start_time, end_time].

### KEY FRAME SELECTION

For EACH event, select 1-2 frame indices (from the input frames, numbered 1 to {n_frames}) \
that best represent the event's core visual content. Choose frames that:
- Show the main action or subject in the clearest view
- Are near the temporal midpoint of the event (avoid the very first/last frame)
- If 2 frames: pick one from the first half and one from the second half of the event

### ANNOTATION WORKFLOW

For best results, follow this order:
1. **Watch all frames** and identify inter-scene cuts → event boundaries.
2. **Write descriptions first** (instruction, dense_caption), then assign timestamps. \
This grounds your timestamps in visual evidence.
3. **Select key frames** for each event after boundaries are finalized.
4. **Annotate L3 sub-actions** for all events (attempt decomposition for every event).

### TEXT GENERATION RULES

**CRITICAL — Visual-Only Descriptions**:
Pretend you have NEVER seen this video before and know NOTHING about it. \
You are a camera that can only describe what it records — shapes, colors, movements, \
positions, objects, and spatial relations. You CANNOT hear audio or read context.

- **DO**: Describe actions, objects, body movements, spatial layout, colors, textures, \
scene changes, camera movement, on-screen text/graphics as they appear in the frames.
- **DO NOT**:
  - Use people's names (say "a person", "the host", "a man in a blue shirt")
  - Infer dialogue content, narration topics, or what someone is "explaining"
  - Describe product features, brand names, or purposes that require external knowledge
  - Use words like "explains", "discusses", "describes" (these imply audio understanding)
  - Assume narrative context beyond what is visually shown
- If someone is talking to the camera: describe their body language, gestures, and setting — \
NOT what they might be saying. Write "A person gestures toward the floor" NOT "A host explains carpet features".
- `instruction`: 8-20 words, objective visual description of WHAT happens WITH WHICH objects.
- `dense_caption`: 2-4 sentences. Academic dense video captioning style: describe the \
visual content in detail — actions, objects, spatial relations, and visible state changes.
- L3 `sub_action`: 5-15 word action phrase.
- L3 `caption`: 1-2 sentences, detailed visual description of the atomic action.
- **FORBIDDEN words in event descriptions**: "explains", "discusses", "describes", "talks about", \
"demonstrates how to", "shows how". Replace with what is VISUALLY happening.

### VISUAL SIGNAL REFERENCE
- **Scene/Space**: Background change, location switch, character entry/exit.
- **Subject**: Pose transition, gaze shift, speed change, new interaction.
- **Object**: Appearance/position/quantity change.
- **Camera/Editing**: Scene cut, focus shift, montage sequence, zoom change.

════════════════════════════════════════════════
## PART 3 — AGGREGATION HINTS
════════════════════════════════════════════════

Provide a one-sentence summary and a one-sentence criterion for how the events could be \
grouped into higher-level thematic phases. These help a downstream stage aggregate events.

════════════════════════════════════════════════
## OUTPUT JSON
════════════════════════════════════════════════

{{
  "paradigm": "<one of: {paradigm_ids}>",
  "paradigm_confidence": 0.85,
  "paradigm_reason": "<one sentence explaining the paradigm decision>",
  "domain_l2": "<one of the domain_l2 categories above, or 'other'>",
  "video_caption": "<3-5 sentences describing the entire video>",
  "feasibility": {{
    "score": 0.85,
    "skip": false,
    "skip_reason": null,
    "estimated_n_events": 8,
    "visual_dynamics": "high"
  }},
  "video_metadata": {{
    "has_text_overlay": false,
    "has_narration": true,
    "camera_style": "<static_tripod | handheld | multi_angle | first_person>",
    "editing_style": "<continuous | jump_cut | montage | mixed>"
  }},
  "summary": "<one sentence summarizing the entire video>",
  "global_phase_criterion": "<one sentence: principle for grouping events into thematic phases>",
  "events": [
    {{
      "event_id": 1,
      "instruction": "<8-20 words: WHAT happens WITH WHICH objects>",
      "dense_caption": "<2-4 sentences: detailed visual description>",
      "start_time": 5,
      "end_time": 28,
      "visual_keywords": ["kw1", "kw2"],
      "key_frame_indices": [7, 15],
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
      "instruction": "<8-20 words>",
      "dense_caption": "<2-4 sentences>",
      "start_time": 30,
      "end_time": 45,
      "visual_keywords": ["kw1", "kw2"],
      "key_frame_indices": [40],
      "l3_feasible": true,
      "l3_reason": "Person gestures and moves position.",
      "sub_actions": [
        {{
          "action_id": 1,
          "start_time": 32,
          "end_time": 37,
          "sub_action": "<5-15 word action phrase>",
          "caption": "<1-2 sentences>",
          "pre_state": null,
          "post_state": null
        }}
      ]
    }}
  ]
}}

## RULES
1. Output strictly valid JSON. No markdown code blocks.
2. All timestamps: absolute integer seconds within [0, {duration}].
3. Short events are fine — do NOT merge short segments into adjacent events. \
Each visually distinct segment should be its own event regardless of duration.
4. No names: never use people's names. Use descriptive labels ("a person", "the host").
5. Feasibility enums: skip_reason is null or "talk_dominant" | "ambient_static"; \
visual_dynamics is "high" | "medium" | "low".
6. If feasibility.skip=true, output "events": [].
7. key_frame_indices: integers in [1, {n_frames}], 1-2 per event.
8. L3 sub_actions must be within parent event [start_time, end_time]; 2-6 seconds each.
9. For events with l3_feasible=false, output "sub_actions": []. Prefer l3_feasible=true for most events.
10. L3 timestamps: absolute integer seconds from the full video timeline."""


def get_l2l3_first_prompt(n_frames: int, duration_sec: int) -> str:
    """Build the L2+L3 first prompt (v8 Stage 1: bottom-up with inline L3).

    Produces flat L2 events with nested L3 sub-actions, key frame indices,
    and classification metadata. L1 phases come from a separate aggregation step.
    Uses 2fps frame input for finer temporal resolution.
    """
    return _L2L3_FIRST_PROMPT.format(
        n_frames=n_frames,
        duration=duration_sec,
        paradigm_table=_format_paradigm_table_for_prompt(),
        paradigm_ids=", ".join(sorted(PARADIGM_IDS)),
        domain_l2_list=_format_domain_l2_for_prompt(),
        paradigm_l2l3_table=_format_paradigm_l2l3_table(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Scene-First Prompt (v9: scene boundaries as hard anchors → merge → caption)
# ─────────────────────────────────────────────────────────────────────────────

_SCENE_FIRST_PROMPT = """\
You are given a {duration}s video clip with {n_frames} frames sampled at 1fps.

An automatic shot detector has pre-segmented this clip into {n_scenes} scene(s). \
These scenes are useful anchors — but not infallible. \
Frames labeled [SCENE BREAK] mark where each detected scene starts.

Pre-detected input scenes (start/end in whole seconds):
```json
{scenes_json}
```

Your task has THREE parts:
1. **CLASSIFY** the video (domain + caption)
2. **RESTRUCTURE + CAPTION**: Merge adjacent scenes and/or split long scenes into \
well-formed events, then caption each event
3. **AGGREGATION HINTS**: one-sentence summary + phase grouping criterion

════════════════════════════════════════════════
## PART 1 — CLASSIFY
════════════════════════════════════════════════

### 1A. DOMAIN (content topic)

Choose domain_l2 from:
{domain_l2_list}
If none fits, use domain_l2="other" and fill `domain_l2_note` with a short free-text \
description of what domain this video belongs to (e.g., "medical surgery", "industrial assembly").

### 1B. VIDEO CAPTION

Write a detailed description of the entire video (3-5 sentences). \
Cover: setting/environment, main subjects, key objects, overall progression, and outcome. \
Every statement must be grounded in what is visible in the frames.

════════════════════════════════════════════════
## PART 2 — RESTRUCTURE + CAPTION
════════════════════════════════════════════════

### 2A. THREE OPERATIONS ON SCENES

Use these three operations to turn the {n_scenes} input scenes into well-formed events:

────────────────────────────────────────────
**OPERATION 1: KEEP** (default — one scene = one event)
- `scene_ids: [k]`, `merge_reason: null`, `split_reason: null`
- start_time / end_time = scene k's known boundaries (from the scenes list above)
────────────────────────────────────────────
**OPERATION 2: MERGE** (combine consecutive scenes into one event)
- `scene_ids: [k, k+1, ...]`, provide `merge_reason`
- start_time = first scene's start_time; end_time = last scene's end_time

**⚠ CRITICAL — TIMESTAMP RULE FOR KEEP AND MERGE**: \
For KEEP and MERGE operations, you MUST blindly copy the exact `start_time` of the \
first scene and `end_time` of the last scene from the input JSON. \
Do NOT alter, round, or recompute these timestamps.

**When to MERGE**: Combine consecutive scenes into ONE event ONLY if they represent \
the SAME unbroken temporal event in the SAME continuous space. Valid merge scenarios:
  1. **Angle/Framing Change**: Wide shot → close-up of the SAME person doing the SAME \
ongoing activity (e.g., cooking, exercising).
  2. **Shot/Reverse-Shot (Dialogue)**: Alternating camera angles between two people talking \
in the SAME room/location. Even though the subject and background strictly change per shot, \
they share the same continuous space and time → MERGE.
  3. **Continuous Pan/Tracking**: The camera continuously follows an action or pans across \
the same location.

**When to keep SEPARATE (Do NOT Merge)**:
  ✗ **Location/Time jump**: Cut to a visibly different place, or a clearly different time \
(e.g., day → night).
  ✗ **Subject jump**: A completely different person/object becomes the focus in a different context.
  ✗ **Activity shift**: A new step, sub-task, or phase begins \
(e.g., prep ends → cooking begins).
  ✗ **B-Roll / Cut-away**: An establishing shot, title card, or illustrative B-roll \
inserted between main actions.
  ✗ **Title cards / Intro / Outro / Slideshows**: Scenes that contain static text overlays, \
scrolling credits, logo bumpers, or slideshow-style content MUST be kept as standalone events — \
NEVER merge them with adjacent content scenes.

**merge_reason**: 1-2 sentences of concrete visual evidence explaining WHY the scenes \
belong to the same unbroken event:
  - GOOD: "Both scenes show the same person kneading dough on the same floured board \
from different angles — same activity, only framing changed."
  - GOOD: "Shot/reverse-shot between two people talking across a table in the same café — \
continuous dialogue in the same space."
  - BAD: "Same person appears in both." (no evidence of continuous space/time)
  - BAD: "Same sport / topic." (topic match is not spatial/temporal continuity)
────────────────────────────────────────────
**OPERATION 3: SPLIT** (break a long scene into sub-events)
- `scene_ids: [k]`, provide `split_reason`, explicit `start_time` and `end_time`
- Multiple events can share `scene_ids: [k]` — they are all sub-segments of scene k
- Timestamps must be within scene k's time range (from the scenes list above)

**When to SPLIT**: A single detected scene spans multiple distinct activities that the \
detector missed (e.g., a 45s scene where a person first chops vegetables then moves to the stove).
- Minimum sub-event duration: 5 seconds
- split_reason: 1 sentence explaining what distinct activities were found in the scene
────────────────────────────────────────────

**CRITICAL — CONSECUTIVE SCENES ONLY**:

scene_ids represents a CONTIGUOUS BLOCK of the video timeline. \
Scenes are ordered in TIME. scene_ids [2, 3, 4] means the event covers the time \
from scene 2's start to scene 4's end — a contiguous time segment. \
You CANNOT skip scenes or group non-adjacent scenes.

  ✓ CORRECT: scene_ids [3, 4, 5]  — consecutive, covers a contiguous time block
  ✗ WRONG:   scene_ids [2, 4, 6]  — NON-CONSECUTIVE, these are spread across the timeline
  ✗ WRONG:   scene_ids [1, 3, 5, 7] — alternating scenes from different times

The scenes list is ordered chronologically. Think of them as LEGO bricks on a timeline — \
you can only merge adjacent bricks, not grab every other one.

**PARTITION RULE**: Together, all events must cover every scene exactly once. \
The events define a non-overlapping, gap-free partition of the full timeline.

  Example with 5 scenes:
  ✓ CORRECT: events with scene_ids [1], [2,3], [4], [5]   — full coverage, no gaps, no skips
  ✗ WRONG:   events with scene_ids [1,3,5], [2,4]         — non-consecutive, interleaved

**COVERAGE RULE**: The union of scene_ids across all events must cover every scene ID \
from 1 to {n_scenes}. No scene may be left uncovered.

### 2B. EVENT CAPTION

For each event, write:
- `instruction`: 8-20 words. WHAT happens WITH WHICH objects. Observable visual action, not intent.
- `dense_caption`: 2-4 sentences. Dense video captioning style: describe visible actions, \
objects, spatial relations, and state changes in detail.
- `visual_keywords`: 3-6 keyword tags reflecting the main visual elements.
- `key_frame_indices`: 1-2 frame indices (integers in [1, {n_frames}]) best representing \
the event's core visual content. Choose frames near the temporal midpoint.
- `l3_worthy`: boolean. Set to `true` if this event contains rich internal dynamics that \
would benefit from finer-grained L3 sub-segmentation. Indicators: multiple distinct \
sub-actions within the event, visible action transitions, object state changes, or \
camera movement revealing different activities. Set to `false` for static / uniform / \
very short events where internal splitting adds no value.

════════════════════════════════════════════════
## PART 3 — AGGREGATION HINTS
════════════════════════════════════════════════

Provide a one-sentence video summary and a one-sentence principle for how events could be \
grouped into higher-level thematic phases (for downstream L1 aggregation).

════════════════════════════════════════════════
## OUTPUT JSON
════════════════════════════════════════════════

{{
  "domain_l2": "<one of the domain_l2 categories above, or 'other'>",
  "domain_l2_note": "<if domain_l2 is 'other', describe the domain here; otherwise null>",
  "video_caption": "<3-5 sentences describing the entire video>",
  "summary": "<one sentence: overall video summary>",
  "global_phase_criterion": "<one sentence: principle for grouping events into thematic phases>",
  "events": [
    {{
      "event_id": 1,
      "scene_ids": [1],
      "merge_reason": null,
      "split_reason": null,
      "start_time": 0,
      "end_time": 12,
      "instruction": "<8-20 words: WHAT happens WITH WHICH objects>",
      "dense_caption": "<2-4 sentences: detailed visual description>",
      "visual_keywords": ["kw1", "kw2", "kw3"],
      "key_frame_indices": [5, 10],
      "l3_worthy": false
    }},
    {{
      "event_id": 2,
      "scene_ids": [2, 3],
      "merge_reason": "Wide shot and close-up both show the same person chopping vegetables on the same board — only framing changed, the chopping is continuous.",
      "split_reason": null,
      "start_time": 12,
      "end_time": 35,
      "instruction": "<8-20 words>",
      "dense_caption": "<2-4 sentences>",
      "visual_keywords": ["kw1"],
      "key_frame_indices": [20],
      "l3_worthy": true
    }},
    {{
      "event_id": 3,
      "scene_ids": [4],
      "merge_reason": null,
      "split_reason": "Scene 4 spans two distinct activities: chopping ends at 50s and stirring begins.",
      "start_time": 35,
      "end_time": 50,
      "instruction": "<first activity in scene 4>",
      "dense_caption": "<2-4 sentences>",
      "visual_keywords": [],
      "key_frame_indices": [40],
      "l3_worthy": true
    }},
    {{
      "event_id": 4,
      "scene_ids": [4],
      "merge_reason": null,
      "split_reason": "Scene 4 spans two distinct activities: chopping ends at 50s and stirring begins.",
      "start_time": 50,
      "end_time": 65,
      "instruction": "<second activity in scene 4>",
      "dense_caption": "<2-4 sentences>",
      "visual_keywords": [],
      "key_frame_indices": [55],
      "l3_worthy": false
    }}
  ]
}}

════════════════════════════════════════════════
## VALIDATION RULES
════════════════════════════════════════════════

1. Output strictly valid JSON. No markdown code blocks.
2. scene_ids coverage: the union of scene_ids across all events must include every ID in [1..{n_scenes}], forming a complete partition of the timeline.
3. MERGE: scene_ids MUST be consecutive integers [k, k+1, ..., k+m] with no gaps. NON-CONSECUTIVE merges (e.g., [2,4,6]) are INVALID — they interleave scenes from different time blocks.
4. SPLIT: split_reason required; start_time/end_time must be within source scene's boundaries.
5. All timestamps: absolute integer seconds in [0, {duration}]. Output as plain integers (e.g., 33, not "00:33" or "33s").
6. key_frame_indices: integers in [1, {n_frames}], 1-2 per event."""

def get_scene_first_prompt(
    n_frames: int,
    duration_sec: int,
    n_scenes: int,
    scenes_json_str: str,
) -> str:
    """Build the scene-first prompt (v9 Stage 1: scene-anchored merge → caption).

    The model receives pre-detected scenes as hard anchors and decides which adjacent
    scenes to merge into events, then captions each event. L3 is done in a separate pass.
    Event timestamps are NOT output by the model — they are derived from scene_ids.
    """
    return _SCENE_FIRST_PROMPT.format(
        n_frames=n_frames,
        duration=duration_sec,
        n_scenes=n_scenes,
        scenes_json=scenes_json_str,
        domain_l2_list=_format_domain_l2_for_prompt(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Scene-first L3 prompt (per-event sub-split, separate pass)
# ─────────────────────────────────────────────────────────────────────────────

_SCENE_FIRST_L3_PROMPT = """\
You are given {n_frames} frames from a video event spanning {start_time}s to {end_time}s \
(duration: {duration}s). Frame labels show absolute timestamps [t=Xs]. \
Frames labeled [SCENE BREAK] mark where a detected shot/scene boundary occurs within this event.

**Event context** (from a previous annotation pass):
- **instruction**: {instruction}
- **dense_caption**: {dense_caption}
- **scene_ids**: {scene_ids} ({n_scenes} scene(s) in this event)

Your task: Split this event into visually distinct micro-segments (L3 sub-actions).

════════════════════════════════════════════════
## L3 SUB-SPLIT RULES
════════════════════════════════════════════════

Each L3 entry represents a visually distinct micro-segment within this event. \
Think of it as finding the natural internal boundaries.

**When to create sub-actions**:
- The event contains multiple visually distinct phases → split at boundaries
- Multi-scene merged event ({n_scenes} scenes) → at least {n_scenes} sub_action(s), \
one per original scene minimum, or finer-grained if the scenes are themselves splittable

**When to output `"sub_actions": []`**:
- The event is a single continuous visual unit with no distinct internal phases
- Static content (title card, frozen frame, text overlay, talkshow interview with zero motion)

**What counts as an L3 boundary** (any visually distinct window):
- Physical action change: different motion/task begins
- Camera shot / framing change: cut to different angle
- Subject state change: person pauses, changes pose
- Object appearance/disappearance: new object enters or exits frame
- Environmental shift: lighting change, background change

**Rules**:
- Timestamps: absolute integer seconds (matching the [t=Xs] frame labels)
- Must fall within [{start_time}, {end_time}]
- Gaps between entries are OK — do NOT force full coverage
- No duration constraint on individual entries

**`sub_action`** (5-15 words): concise visual label. \
Examples: "close-up of hands folding dough", "person looks off-frame to the right".

**`caption`** (1-2 sentences): detailed visual description. \
Purely observable facts — no inference, no audio, no domain knowledge. \
Write as if you can only see, not hear.

**PROHIBITIONS** (these corrupt training data):
- Words implying hearing: "explains", "says", "narrates"
- Words implying intent: "demonstrates", "shows how to", "prepares to"
- Domain-specific names unless readable as on-screen text
- Cross-event references: "after the previous step", "before this step"

════════════════════════════════════════════════
## OUTPUT JSON
════════════════════════════════════════════════

{{
  "sub_actions": [
    {{
      "action_id": 1,
      "start_time": {start_time},
      "end_time": <integer>,
      "sub_action": "<5-15 word visual label>",
      "caption": "<1-2 sentences: detailed visual description>"
    }}
  ]
}}

Output strictly valid JSON. No markdown code blocks. \
If no sub-actions, output `{{"sub_actions": []}}`."""


def get_scene_first_l3_prompt(
    n_frames: int,
    start_time: int,
    end_time: int,
    instruction: str,
    dense_caption: str,
    scene_ids: str,
    n_scenes: int,
) -> str:
    """Build the per-event L3 sub-split prompt for scene_first pass 2."""
    return _SCENE_FIRST_L3_PROMPT.format(
        n_frames=n_frames,
        start_time=start_time,
        end_time=end_time,
        duration=end_time - start_time,
        instruction=instruction,
        dense_caption=dense_caption,
        scene_ids=scene_ids,
        n_scenes=n_scenes,
    )
