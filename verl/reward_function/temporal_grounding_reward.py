# Copyright 2024 Bytedance Ltd. and/or its affiliates
# -*- coding: utf-8 -*-
"""
Temporal Grounding Reward — IoU × distance_penalty (Time-R1 style)。

适配 <answer>start to end</answer> 格式（兼容旧版 <events>[[s, e]]</events>）。

Reward 计算:
  1. 从 <answer> 或 <events> 标签中解析预测的 [start, end]
  2. 计算与 GT 的 temporal IoU
  3. 乘以归一化距离惩罚: (1 - |Δs/dur|) × (1 - |Δe/dur|)
     惩罚端点偏移，抑制超长片段的"懒惰最优解"
  4. 格式不合法时返回 0.0

无 format 奖励 — overall = accuracy ∈ [0, 1]。

输出格式（兼容 EasyR1 batch reward 接口）:
    {"overall": float, "format": float, "accuracy": float}
"""

import re
from typing import Any, Dict, List, Optional, Tuple


# ===========================
# 解析模式
# ===========================
# <answer>12.54 to 17.83</answer>  (Time-R1 style)
_ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_TIME_RANGE_PATTERN = re.compile(
    r"(\d+\.?\d*)\s+(?:to|and)\s+(\d+\.?\d*)", re.IGNORECASE
)

# <events>[[12.5, 17.8]]</events>  (legacy)
_EVENTS_PATTERN = re.compile(r"<events>(.*?)</events>", re.DOTALL)
_SEGMENT_PATTERN = re.compile(
    r"\[\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\]"
)

_ZERO = {"overall": 0.0, "format": 0.0, "accuracy": 0.0}


def _parse_single_segment(text: str) -> Optional[Tuple[float, float]]:
    """
    解析单个时间段。优先从 <answer> 标签解析，其次从 <events> 标签。
    """
    # 尝试 <answer>start to end</answer>
    answer_match = _ANSWER_PATTERN.search(text)
    if answer_match is not None:
        content = answer_match.group(1)
        m = _TIME_RANGE_PATTERN.search(content)
        if m is not None:
            try:
                start = float(m.group(1))
                end = float(m.group(2))
                if start >= 0 and end >= 0 and start < end:
                    return (start, end)
            except (ValueError, TypeError):
                pass

    # 回退: <events>[[s, e]]</events>
    events_match = _EVENTS_PATTERN.search(text)
    if events_match is not None:
        events_block = events_match.group(1)
        m = _SEGMENT_PATTERN.search(events_block)
        if m is not None:
            try:
                start = float(m.group(1))
                end = float(m.group(2))
                if start >= 0 and end >= 0 and start < end:
                    return (start, end)
            except (ValueError, TypeError):
                pass

    return None


def temporal_grounding_reward(
    response: str,
    ground_truth: str,
    metadata: Optional[Dict] = None,
) -> Dict[str, float]:
    """
    Temporal Grounding reward: IoU × distance_penalty ∈ [0, 1]。

    无 format 奖励，overall = accuracy。
    """
    # 反黑客
    if response.count("<answer>") > 1 or response.count("</answer>") > 1:
        return dict(_ZERO)
    if response.count("<events>") > 1 or response.count("</events>") > 1:
        return dict(_ZERO)
    if re.search(r"\[\d+-\d+\]", response):
        return dict(_ZERO)

    # 解析 GT
    gt_pair = _parse_single_segment(ground_truth)
    if gt_pair is None:
        return dict(_ZERO)
    gt_s, gt_e = gt_pair

    # 解析预测
    pred_pair = _parse_single_segment(response)
    if pred_pair is None:
        return dict(_ZERO)
    pred_s, pred_e = pred_pair

    # IoU
    intersection = max(0.0, min(pred_e, gt_e) - max(pred_s, gt_s))
    union = max(pred_e, gt_e) - min(pred_s, gt_s)
    iou = intersection / union if union > 0 else 0.0

    # distance_penalty: 惩罚端点偏移，抑制超长片段
    if metadata is None:
        metadata = {}
    duration = metadata.get("duration", 0.0)
    if duration > 0:
        dist_penalty = (
            (1.0 - abs(gt_s - pred_s) / duration)
            * (1.0 - abs(gt_e - pred_e) / duration)
        )
        dist_penalty = max(0.0, dist_penalty)
        accuracy = iou * dist_penalty
    else:
        accuracy = iou

    return {
        "overall": float(accuracy),
        "format": 0.0,
        "accuracy": float(accuracy),
    }


# ===========================
# Batch reward 接口
# ===========================

def compute_score(
    reward_inputs: List[Dict[str, Any]],
    **kwargs,
) -> List[Dict[str, float]]:
    """
    Batch reward 接口（与 EasyR1 BatchFunctionRewardManager 兼容）。

    overall = tIoU × distance_penalty ∈ [0, 1]。
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for this reward function.")

    results: List[Dict[str, float]] = []

    for item in reward_inputs:
        try:
            response = str(item.get("response", ""))
            ground_truth = str(item.get("ground_truth", ""))
            metadata = item.get("metadata") or {}
            score = temporal_grounding_reward(response, ground_truth, metadata)
        except Exception as e:
            print(f"[temporal_grounding_reward] Error: {e}")
            score = dict(_ZERO)
        results.append(score)

    return results
