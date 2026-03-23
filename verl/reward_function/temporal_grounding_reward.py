# -*- coding: utf-8 -*-
"""
Temporal Grounding Reward 函数 — 纯 Temporal IoU。

适配 <events>[[start, end]]</events> 格式。

Reward 计算:
  1. 从 <events> 标签中解析预测的 [start, end]
  2. 计算与 GT 的 temporal IoU
  3. 格式不合法时返回 0.0

输出格式（兼容 EasyR1 batch reward 接口）:
    {"overall": float, "format": float, "accuracy": float}
"""

import re
from typing import Dict, List, Optional, Tuple

from verl.reward_function.youcook2_temporal_seg_reward import (
    EVENTS_PATTERN,
    SEGMENT_PATTERN,
    has_events_tag,
)


def _parse_single_segment(text: str) -> Optional[Tuple[float, float]]:
    """
    从 <events>...</events> 标签中提取第一个 [start, end] 对。
    temporal grounding 任务只需要一个 segment。
    """
    events_match = EVENTS_PATTERN.search(text)
    if events_match is None:
        return None

    events_block = events_match.group(1)
    m = SEGMENT_PATTERN.search(events_block)
    if m is None:
        return None

    try:
        start = float(m.group(1))
        end = float(m.group(2))
    except (ValueError, TypeError):
        return None

    if start < 0 or end < 0 or start >= end:
        return None

    return (start, end)


def temporal_grounding_reward(
    response: str,
    ground_truth: str,
    metadata: Optional[Dict] = None,
) -> Dict[str, float]:
    """
    Temporal Grounding 纯 IoU reward。

    Args:
        response: 模型回复（包含 <events>[[s, e]]</events>）
        ground_truth: 标准答案（同格式）
        metadata: 保留参数，当前不使用

    Returns:
        {"overall": float, "format": float, "accuracy": float}
    """
    # 反黑客
    if response.count("<events>") > 1 or response.count("</events>") > 1:
        return {"overall": 0.0, "format": 0.0, "accuracy": 0.0}
    if re.search(r"\[\d+-\d+\]", response):
        return {"overall": 0.0, "format": 0.0, "accuracy": 0.0}

    # 格式检查
    if not has_events_tag(response):
        return {"overall": 0.0, "format": 0.0, "accuracy": 0.0}

    # 解析 GT
    gt_pair = _parse_single_segment(ground_truth)
    if gt_pair is None:
        return {"overall": 0.0, "format": 0.0, "accuracy": 0.0}
    gt_s, gt_e = gt_pair

    # 解析预测
    pred_pair = _parse_single_segment(response)
    if pred_pair is None:
        # 格式标签存在但内容无法解析
        return {"overall": 0.0, "format": 1.0, "accuracy": 0.0}
    pred_s, pred_e = pred_pair

    # IoU
    intersection = max(0.0, min(pred_e, gt_e) - max(pred_s, gt_s))
    union = max(pred_e, gt_e) - min(pred_s, gt_s)
    iou = intersection / union if union > 0 else 0.0

    return {
        "overall": float(iou),
        "format": 1.0,
        "accuracy": float(iou),
    }
