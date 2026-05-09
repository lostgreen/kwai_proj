#!/usr/bin/env python3
"""
Simple MCQ reward for LLaVA-Video-178K rollout filtering.

For a base model (pre-RL) that doesn't output <answer> tags, we simply
extract the first capital letter from the response and compare to ground truth.

For RL-trained models that output <answer>X</answer>, we extract from the tag first.

compute_score interface matches offline_rollout_filter.py expectations:
    reward_inputs: list[dict] with keys: response, ground_truth, problem_type, ...
    returns: list[dict] with key: overall (float 0.0 or 1.0)
"""

from __future__ import annotations

import re
from typing import Any

# Match <answer>X</answer> pattern (RL model format)
_ANSWER_TAG = re.compile(r"<answer>\s*([A-Za-z])\s*</answer>", re.IGNORECASE)
# Match standalone letter at start: "E", "E.", "E. training"
_LEADING_LETTER = re.compile(r"^\s*([A-Za-z])[\s.\)\:]")
# Match "The answer is X" pattern
_ANSWER_IS = re.compile(r"(?:answer|option)\s+(?:is|:)\s*([A-Za-z])", re.IGNORECASE)


def _extract_letter(response: str) -> str | None:
    """Extract answer letter from model response, trying multiple patterns."""
    # 1. Try <answer> tag first (RL model)
    m = _ANSWER_TAG.search(response)
    if m:
        return m.group(1).upper()

    # 2. Try "the answer is X" pattern
    m = _ANSWER_IS.search(response)
    if m:
        return m.group(1).upper()

    # 3. Try leading letter (base model often starts with "E. ...")
    m = _LEADING_LETTER.match(response)
    if m:
        return m.group(1).upper()

    # 4. Fallback: find any single capital letter that's an option (A-E)
    letters = re.findall(r"\b([A-E])\b", response)
    if len(letters) == 1:
        return letters[0]

    return None


def compute_score(
    reward_inputs: list[dict[str, Any]],
    **kwargs,
) -> list[dict[str, float]]:
    """Batch MCQ reward: exact match on extracted letter vs ground truth."""
    results = []
    for item in reward_inputs:
        response = item.get("response", "") or ""
        ground_truth = (item.get("ground_truth", "") or "").strip().upper()

        pred = _extract_letter(response)
        if pred is not None and pred == ground_truth:
            results.append({"overall": 1.0, "accuracy": 1.0})
        else:
            results.append({"overall": 0.0, "accuracy": 0.0})

    return results
