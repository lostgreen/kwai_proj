#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt templates for Temporal AoT V2T/T2V data annotation and dataset building.

These prompts are intentionally short and structured so they can be reused by:
- caption annotation scripts
- V2T/T2V dataset builders
"""

SYSTEM_PROMPT = (
    "You are a careful video understanding assistant. "
    "Your job is to describe the visible temporal order of actions, not just the objects or the overall activity. "
    "Pay close attention to what happens first, what changes in the middle, and what state appears at the end. "
    "If the motion appears reversed or temporally unnatural, describe the observed reversed order instead of normalizing it into a plausible forward action. "
    "Do not speculate beyond the visible actions."
)


def get_forward_reverse_caption_prompt() -> str:
    """
    Prompt for captioning a single clip.
    Intended to be used independently on forward and reversed versions.
    """
    return (
        "Watch the video carefully.\n"
        "<video>\n\n"
        "Write one concise sentence describing the observed action sequence in time order.\n"
        "Requirements:\n"
        "1. Describe the sequence using explicit temporal order, such as 'first', 'then', 'finally', or equivalent phrasing.\n"
        "2. Mention the visible start state and the visible end state when possible.\n"
        "3. Focus on what is actually seen changing over time, not just the overall task category.\n"
        "4. If the clip appears temporally reversed or visually unnatural, describe that observed reversed order instead of rewriting it as a normal forward action.\n"
        "5. Avoid generic descriptions like 'someone is cooking'.\n"
        "6. Keep it to one sentence.\n"
        "7. Also judge whether this clip has a visually clear temporal direction: set direction_clear to true only if "
        "the action involves an observable state change (e.g. empty→filled, uncut→sliced, raw→cooked) such that "
        "reversing the video would look noticeably different. Set it to false for cyclic or oscillatory actions "
        "where the reversed video looks nearly identical (e.g. stirring, mixing, kneading, shaking, whisking).\n"
        "8. Output valid JSON with keys: caption, confidence, direction_clear.\n"
        "Example:\n"
        "{\"caption\": \"First a person pours milk into an empty glass, then fills it to the brim.\", "
        "\"confidence\": 0.88, \"direction_clear\": true}"
    )


def get_v2t_prompt(option_a: str, option_b: str) -> str:
    return (
        "Watch the video carefully.\n"
        "<video>\n\n"
        "Which caption best matches the temporal direction of this video?\n"
        f"Options:\nA. {option_a}\nB. {option_b}\n\n"
        "First, carefully observe the visual content of the video from beginning to end. "
        "Pay attention to what happens first, what changes in the middle, and what state appears at the end. "
        "Then compare both captions against the visible temporal order and reason about which caption matches the video better.\n\n"
        "Think step by step inside <think> </think> tags, then provide your final answer "
        "(a single letter A or B) inside <answer> </answer> tags."
    )


def get_t2v_prompt(
    caption: str,
    option_a_text: str = "The first segment",
    option_b_text: str = "The second segment",
) -> str:
    return (
        "The input video contains two segments separated by a black screen.\n"
        "<video>\n\n"
        f'Which segment best matches the caption "{caption}"?\n'
        f"Options:\nA. {option_a_text}\nB. {option_b_text}\n\n"
        "First, carefully observe both segments and use the black screen as the boundary between them. "
        "Reason about the visible action order in the first segment and in the second segment, "
        "then compare them with the caption to decide which segment matches better.\n\n"
        "Think step by step inside <think> </think> tags, then provide your final answer "
        "(a single letter A or B) inside <answer> </answer> tags."
    )


def get_shuffle_caption_prompt(n_segments: int = 0, segment_sec: float = 2.0) -> str:
    """
    Prompt for captioning a temporally-shuffled clip.
    The video has been cut into fixed-length segments and randomly reordered,
    so the VLM should describe the *observed* (incoherent) action sequence as-is,
    not try to reconstruct the plausible original order.

    Parameters
    ----------
    n_segments : int
        Number of segments the video was cut into (0 = unknown).
    segment_sec : float
        Duration of each segment in seconds.
    """
    if n_segments > 0:
        seg_info = (
            f"This video was created by cutting the original clip into {n_segments} segments "
            f"of approximately {segment_sec:.0f} seconds each and randomly reordering them, "
            "so the events may appear in an incoherent or jumbled sequence.\n\n"
        )
    else:
        seg_info = (
            f"This video was created by cutting the original clip into segments of approximately "
            f"{segment_sec:.0f} seconds each and randomly reordering them, "
            "so the events may appear in an incoherent or jumbled sequence.\n\n"
        )
    return (
        "Watch the video carefully.\n"
        "<video>\n\n"
        + seg_info
        + "Describe the observed action sequence exactly as it appears, in the order you see it.\n"
        "Requirements:\n"
        "1. Write ONE concise sentence using temporal markers ('first', 'then', 'next', 'finally') "
        "to describe the key action transitions you observe. Cover the main phases but keep it brief.\n"
        "2. If the sequence appears logically inconsistent (e.g. the food is plated before it is "
        "cooked), describe it that way—do not reorder to make it plausible.\n"
        "3. Mention the visible start and end states.\n"
        "4. Focus on concrete state changes (shape, color, position, quantity) rather than "
        "generic activity labels.\n"
        "5. Keep it to one sentence — similar in length and style to a forward/reverse caption.\n"
        "6. Set direction_clear to false, since the temporal order has been deliberately disrupted.\n"
        "7. Output valid JSON with keys: caption, confidence, direction_clear.\n"
        "Example:\n"
        "{\"caption\": \"First the pan appears empty, then food is tossed in oil, "
        "next diced onions sit on the board, and finally raw ingredients are placed down.\", "
        "\"confidence\": 0.75, \"direction_clear\": false}"
    )


def get_3way_v2t_prompt(
    option_a: str,
    option_b: str,
    option_c: str,
) -> str:
    """3-option V2T prompt: forward / reverse / shuffled caption."""
    return (
        "Watch the video carefully.\n"
        "<video>\n\n"
        "Which caption best matches the temporal order observed in this video?\n"
        f"Options:\nA. {option_a}\nB. {option_b}\nC. {option_c}\n\n"
        "Carefully observe the full video from beginning to end. "
        "Pay attention to what happens first, what changes in the middle, and what state appears at the end. "
        "Compare all three captions against the visible temporal order and reason about which one matches best.\n\n"
        "Think step by step inside <think> </think> tags, then provide your final answer "
        "(a single letter A, B, or C) inside <answer> </answer> tags."
    )


def get_3way_t2v_prompt(caption: str) -> str:
    """3-option T2V prompt: given a caption, pick the matching video from
    {forward, reverse, shuffle video} (A/B/C).
    Each clip is shown as a separate <video> token so the model sees all 3."""
    return (
        "Three video clips are shown below.\n"
        "Clip A:\n<video>\n"
        "Clip B:\n<video>\n"
        "Clip C:\n<video>\n\n"
        f'Which clip best matches the caption "{caption}"?\n\n'
        "Carefully observe all three clips from beginning to end. "
        "Pay attention to the temporal order of actions in each clip: what happens first, "
        "what changes in the middle, and what state appears at the end. "
        "Compare each clip against the caption and reason about which one matches the described temporal order best.\n\n"
        "Think step by step inside <think> </think> tags, then provide your final answer "
        "(a single letter A, B, or C) inside <answer> </answer> tags."
    )


def get_check_and_refine_prompt(
    forward_caption: str,
    reverse_caption: str,
    shuffle_caption: str = "",
) -> str:
    """Prompt for checking and refining forward/reverse/shuffle captions.

    The caller is expected to attach video frames in three labeled sections
    (forward, reverse, shuffle) before this text prompt.
    """
    shuffle_block = ""
    shuffle_rule = ""
    if shuffle_caption:
        shuffle_block = (
            f'\n  Shuffle caption: "{shuffle_caption}"'
        )
        shuffle_rule = (
            "\n- If the shuffle caption is also provided, refine it to describe the "
            "observed jumbled sequence. It should NOT look like a simple rewrite of "
            "the forward or reverse caption."
        )

    return (
        "You are reviewing temporal captions for a cooking video clip.\n\n"
        "Three versions of the same clip are shown above:\n"
        "- **Forward video**: the original clip in natural chronological order.\n"
        "- **Reverse video**: the same clip played backwards frame-by-frame.\n"
        + ("- **Shuffle video**: the clip cut into segments and randomly reordered.\n" if shuffle_caption else "")
        + "\nExisting captions:\n"
        f'  Forward caption: "{forward_caption}"\n'
        f'  Reverse caption: "{reverse_caption}"'
        + shuffle_block
        + "\n\n"
        "## Task\n\n"
        "1. **Check**: Can a reader distinguish the forward caption from the reverse "
        "caption based on text alone — without seeing the video? A good pair should "
        "describe opposite state transitions (e.g., 'empty pan → full of food' vs. "
        "'food disappears from pan → pan is empty'). A bad pair uses vague or "
        "symmetric descriptions that could fit either direction.\n\n"
        "2. **Refine** (only if the pair is NOT clearly distinguishable): "
        "Rewrite the captions so they are clearly distinguishable.\n\n"
        "## Refinement rules\n"
        "- Each caption must be ONE concise sentence.\n"
        "- Forward and reverse captions must have similar word count "
        "(within 30% of each other).\n"
        "- Use explicit start→end state transitions: mention the visible starting "
        "state and the visible ending state.\n"
        "- Focus on concrete, directional changes: filling/emptying, cutting/assembling, "
        "heating/cooling, raw→cooked, whole→sliced, etc.\n"
        "- Use temporal markers (first, then, finally) to anchor the sequence.\n"
        "- Do NOT use generic descriptions like 'someone is cooking' or "
        "'a person prepares food'.\n"
        "- The reverse caption must describe the actual observed reversed sequence, "
        "not just negate the forward caption."
        + shuffle_rule
        + "\n\n"
        "## Output\n"
        "Return a **raw JSON string** — do NOT wrap it in Markdown code fences "
        "(no ```json ... ```).\n"
        "The JSON must have these keys:\n"
        "- `distinguishable` (bool): true if existing captions are already clearly "
        "distinguishable, false otherwise.\n"
        "- `reason` (string): brief explanation of your judgment.\n"
        "- `forward_caption` (string): the (possibly refined) forward caption.\n"
        "- `reverse_caption` (string): the (possibly refined) reverse caption.\n"
        + ('- `shuffle_caption` (string): the (possibly refined) shuffle caption.\n' if shuffle_caption else "")
        + "\nIf the captions are already distinguishable, return them unchanged.\n"
        "Ensure every string value is on a single line (no embedded newlines).\n\n"
        "Example output:\n"
        '{"distinguishable": false, "reason": "Both captions just say ingredients are '
        'being mixed without describing direction of change.", '
        '"forward_caption": "First, raw diced onions are placed into an empty pan, '
        'then oil is added, and finally the onions turn translucent as they cook.", '
        '"reverse_caption": "First, translucent cooked onions sit in an oily pan, '
        'then they gradually appear raw and uncooked, and finally the empty pan is '
        'shown with no ingredients."'
        + (', "shuffle_caption": "First, onions appear partially cooked in oil, '
           'then raw diced onions sit on an empty pan, and finally the onions are '
           'fully translucent."' if shuffle_caption else "")
        + "}"
    )
