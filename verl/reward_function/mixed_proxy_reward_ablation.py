# Copyright 2024 Bytedance Ltd. and/or its affiliates
# -*- coding: utf-8 -*-
"""Multi-task reward wrapper with switchable hier-seg reward dispatch.

This keeps the existing mixed-task behavior for MCQ / TG / sort tasks while
allowing hier-seg reward ablations to swap only the hier-seg reward family.

Environment:
    HIER_REWARD_MODE = f1_iou | seg_match | dp_f1
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from verl.reward_function.dp_f1_reward import compute_score as compute_dp_f1_score
from verl.reward_function.mixed_proxy_reward import compute_score as compute_base_score
from verl.reward_function.seg_match_reward import compute_score as compute_seg_match_score

_HIER_TASKS = {
    "temporal_seg_hier_L1",
    "temporal_seg_hier_L2",
    "temporal_seg_hier_L3_seg",
}

_HIER_REWARD_DISPATCH = {
    "f1_iou": compute_base_score,
    "seg_match": compute_seg_match_score,
    "dp_f1": compute_dp_f1_score,
}


def _get_hier_reward_fn():
    mode = os.environ.get("HIER_REWARD_MODE", "f1_iou").strip().lower()
    try:
        return mode, _HIER_REWARD_DISPATCH[mode]
    except KeyError as exc:
        supported = ", ".join(sorted(_HIER_REWARD_DISPATCH))
        raise ValueError(f"Unsupported HIER_REWARD_MODE={mode!r}. Supported: {supported}") from exc


def compute_score(
    reward_inputs: List[Dict[str, Any]],
    **kwargs,
) -> List[Dict[str, float]]:
    """Batch reward entry with switchable hier reward."""
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for this reward function.")

    _mode, hier_reward_fn = _get_hier_reward_fn()

    results: List[Dict[str, float] | None] = [None] * len(reward_inputs)
    base_inputs: list[dict[str, Any]] = []
    base_indices: list[int] = []
    hier_inputs: list[dict[str, Any]] = []
    hier_indices: list[int] = []

    for idx, item in enumerate(reward_inputs):
        problem_type = (item.get("problem_type", "") or "").strip()
        if problem_type in _HIER_TASKS:
            hier_inputs.append(item)
            hier_indices.append(idx)
        else:
            base_inputs.append(item)
            base_indices.append(idx)

    if base_inputs:
        base_results = compute_base_score(base_inputs, **kwargs)
        for idx, score in zip(base_indices, base_results):
            results[idx] = score

    if hier_inputs:
        hier_results = hier_reward_fn(hier_inputs, **kwargs)
        for idx, score in zip(hier_indices, hier_results):
            results[idx] = score

    return [score if score is not None else {"overall": 0.0, "format": 0.0, "accuracy": 0.0} for score in results]
