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
    "Focus on temporal direction and describe what is visually happening. "
    "Do not speculate beyond the visible actions."
)


def get_forward_reverse_caption_prompt() -> str:
    """
    Prompt for captioning a single clip.
    Intended to be used independently on forward and reversed versions.
    """
    return (
        "Watch the video carefully and write one concise caption describing the action sequence.\n"
        "Requirements:\n"
        "1. Mention the visible action order when it is clear.\n"
        "2. Avoid generic descriptions like 'someone is cooking'.\n"
        "3. Keep it to one sentence.\n"
        "4. Output valid JSON with keys: caption, confidence.\n"
        "Example:\n"
        "{\"caption\": \"A person places cheese on bread and then puts the sandwich into a pan.\", "
        "\"confidence\": 0.82}"
    )


def get_v2t_prompt(option_a: str, option_b: str) -> str:
    return (
        "Watch the video carefully. Which caption best matches the temporal direction of this video?\n"
        f"Options:\nA. {option_a}\nB. {option_b}\n\n"
        "Think inside <think> </think> and output only the final option letter inside <answer> </answer>."
    )


def get_t2v_prompt(caption: str) -> str:
    return (
        "The input video contains two segments separated by a black screen.\n"
        f'Which segment best matches the caption "{caption}"?\n'
        "Options:\nA. The first segment\nB. The second segment\n\n"
        "Think inside <think> </think> and output only the final option letter inside <answer> </answer>."
    )
