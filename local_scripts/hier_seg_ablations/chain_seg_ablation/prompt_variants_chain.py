"""
Chain-of-Segment 消融实验 prompt 模板。

两个变体 (不带 CoT):
  V1 (dual-seg):    128s 视频 → 自由分割 L2 事件 + L3 原子动作（无 caption）
  V2 (ground-seg):  128s 视频 + 1 条事件描述 → grounding L2 + 分割 L3
"""

# ==================================================================
# V1: Dual Free Segmentation (no caption)
# ==================================================================
DUAL_SEG_TEMPLATE = (
    "Watch the following video clip carefully:\n<video>\n\n"
    "You are given a {duration}s video clip of a procedural activity. "
    "Your task has two steps:\n"
    "1. **Segment** the video into major events (coarse-level temporal segments).\n"
    "2. **Decompose** each event into its atomic actions "
    "(fine-grained sub-segments within each event).\n\n"
    "Rules:\n"
    "- L2 (events): output each event as [start_time, end_time]; events should be "
    "non-overlapping and cover the main activity\n"
    "- L3 (atomic actions): for each event, output the atomic actions as "
    "[[start, end], ...] within that event's time range\n"
    "- all timestamps are integer seconds (0-based, 0 ≤ start < end ≤ {duration})\n"
    "- atomic actions are brief (2-6s) physical state changes\n"
    "- skip idle or narration; gaps between atomic actions are fine\n\n"
    "Output format:\n"
    "<l2_events>[[start, end], ...]</l2_events>\n"
    "<l3_events>[[[start, end], ...], [[start, end], ...], ...]</l3_events>\n\n"
    "Example (3 events, with 2, 3, and 1 atomic actions respectively):\n"
    "<l2_events>[[0, 25], [28, 55], [60, 75]]</l2_events>\n"
    "<l3_events>[[[2, 8], [10, 22]], [[28, 35], [37, 45], [48, 55]], [[62, 73]]]</l3_events>"
)


# ==================================================================
# V2: Single-Caption Grounding + Segmentation
# ==================================================================
GROUND_SEG_TEMPLATE = (
    "Watch the following video clip carefully:\n<video>\n\n"
    "You are given a {duration}s video clip of a procedural activity and "
    "a description of one event that occurs in it.\n\n"
    "Event description: \"{event_description}\"\n\n"
    "Your task has two steps:\n"
    "1. **Locate** the described event's time segment in the clip.\n"
    "2. **Decompose** the event into its atomic actions "
    "(fine-grained sub-segments within the event).\n\n"
    "Rules:\n"
    "- L2: output exactly one [start_time, end_time] for the described event\n"
    "- L3: output the atomic actions as [[start, end], ...] within the event's time range\n"
    "- all timestamps are integer seconds (0-based, 0 ≤ start < end ≤ {duration})\n"
    "- atomic actions are brief (2-6s) physical state changes\n"
    "- skip idle or narration; gaps between atomic actions are fine\n\n"
    "Output format:\n"
    "<l2_events>[[start, end]]</l2_events>\n"
    "<l3_events>[[[start, end], ...]]</l3_events>\n\n"
    "Example (event located at 10-45s with 3 atomic actions):\n"
    "<l2_events>[[10, 45]]</l2_events>\n"
    "<l3_events>[[[12, 18], [20, 30], [35, 43]]]</l3_events>"
)


# ==================================================================
# Registry
# ==================================================================
CHAIN_PROMPT_VARIANTS = {
    "V1": DUAL_SEG_TEMPLATE,
    "V2": GROUND_SEG_TEMPLATE,
}

VARIANT_DESCRIPTIONS = {
    "V1": "Dual free segmentation (no caption, multi-event)",
    "V2": "Single-caption grounding + segmentation",
}

RESPONSE_LEN_HINTS = {
    "V1": 1024,   # multi-event → longer output
    "V2": 512,    # single event → shorter
}

# Template parameters per variant
VARIANT_PARAMS = {
    "V1": {"duration"},
    "V2": {"duration", "event_description"},
}
