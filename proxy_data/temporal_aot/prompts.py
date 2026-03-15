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
        "7. Output valid JSON with keys: caption, confidence.\n"
        "Example:\n"
        "{\"caption\": \"First a person sprinkles seasoning into a bowl, then they stir the mixture, and finally they taste it with a finger.\", "
        "\"confidence\": 0.82}"
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


def get_t2v_prompt(caption: str) -> str:
    return (
        "The input video contains two segments separated by a black screen.\n"
        "<video>\n\n"
        f'Which segment best matches the caption "{caption}"?\n'
        "Options:\nA. The first segment\nB. The second segment\n\n"
        "First, carefully observe both segments and use the black screen as the boundary between them. "
        "Reason about the visible action order in the first segment and in the second segment, "
        "then compare them with the caption to decide which segment matches better.\n\n"
        "Think step by step inside <think> </think> tags, then provide your final answer "
        "(a single letter A or B) inside <answer> </answer> tags."
    )
