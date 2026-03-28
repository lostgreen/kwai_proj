# Copyright 2024 Bytedance Ltd. and/or its affiliates
# -*- coding: utf-8 -*-
"""
YouCook2 链式分割 Reward 函数 (Chain-of-Segment) — 仅 V2 (ground-seg)。

problem_type: temporal_seg_chain_ground_seg
  单 caption grounding (L2) + 单事件内原子动作分割 (L3)

输出格式:
    <l2_events>[[start, end]]</l2_events>
    <l3_events>[[[s1, e1], [s2, e2], ...]]</l3_events>

Reward:
    R_L2 = temporal_iou(pred_l2, gt_l2)
    R_L3 = F1-IoU(clipped_l3, gt_l3)  — 硬裁剪到 pred L2 边界
           如果 L3 有任何段越界 pred L2 → R_L3 × 0.5
    overall = 0.4 × R_L2 + 0.6 × R_L3

兼容 EasyR1 BatchFunctionRewardManager 接口。
"""

import random
import re
from typing import Any, Dict, List

from verl.reward_function.youcook2_temporal_seg_reward import (
    compute_f1_iou,
    temporal_iou,
)


# ===========================
# 常量
# ===========================
W_L2 = 0.4
W_L3 = 0.6
OOB_PENALTY = 0.5  # L3 越界时 reward 砍半


# ===========================
# 正则解析
# ===========================
L2_EVENTS_PATTERN = re.compile(
    r"<l2_events>(.*?)</l2_events>",
    re.DOTALL,
)
L3_EVENTS_PATTERN = re.compile(
    r"<l3_events>(.*?)</l3_events>",
    re.DOTALL,
)
SEGMENT_PATTERN = re.compile(
    r"\[\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\]"
)


def _parse_flat_segments(text: str) -> List[List[float]]:
    """从文本中提取 [start, end] 列表。"""
    segments: List[List[float]] = []
    for m in SEGMENT_PATTERN.finditer(text):
        try:
            start = float(m.group(1))
            end = float(m.group(2))
        except (ValueError, TypeError):
            continue
        if start >= end or start < 0:
            continue
        segments.append([start, end])
    return segments


def parse_l2_events(text: str) -> List[List[float]]:
    """解析 <l2_events>...</l2_events> 中的 L2 段列表。"""
    match = L2_EVENTS_PATTERN.search(str(text))
    if match is None:
        return []
    return _parse_flat_segments(match.group(1))


def parse_l3_events(text: str) -> List[List[List[float]]]:
    """解析 <l3_events>...</l3_events> 中的嵌套 L3 段列表。"""
    match = L3_EVENTS_PATTERN.search(str(text))
    if match is None:
        return []

    raw = match.group(1).strip()
    if not raw:
        return []

    result: List[List[List[float]]] = []

    # 去掉最外层 []
    raw = raw.strip()
    if raw.startswith("["):
        raw = raw[1:]
    if raw.endswith("]"):
        raw = raw[:-1]

    # 按顶层 [] 边界拆分
    depth = 0
    current_start = -1
    for i, ch in enumerate(raw):
        if ch == "[":
            if depth == 0:
                current_start = i
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0 and current_start >= 0:
                sub_block = raw[current_start: i + 1]
                segs = _parse_flat_segments(sub_block)
                result.append(segs)
                current_start = -1

    return result


def has_chain_tags(text: str) -> bool:
    """检查文本是否包含 <l2_events> 和 <l3_events> 标签。"""
    text = str(text)
    return (L2_EVENTS_PATTERN.search(text) is not None
            and L3_EVENTS_PATTERN.search(text) is not None)


# ===========================
# Anti-hacking 检查
# ===========================
def _anti_hack_check(response: str) -> bool:
    """返回 True 表示通过，False 表示触发反作弊应得 0 分。"""
    if re.search(r"\[\d+-\d+\]", response):
        return False
    if response.count("<l2_events>") > 1 or response.count("</l2_events>") > 1:
        return False
    if response.count("<l3_events>") > 1 or response.count("</l3_events>") > 1:
        return False
    return True


_ZERO = {"overall": 0.0, "format": 0.0, "accuracy": 0.0,
         "l2_reward": 0.0, "l3_reward": 0.0}


# ===========================
# L3 边界裁剪
# ===========================
def clip_l3_to_l2_bounds(
    l2_segs: List[List[float]],
    l3_nested: List[List[List[float]]],
) -> List[List[List[float]]]:
    """将每组 L3 段裁剪到对应 L2 段边界内。

    越界部分被裁剪；完全在 L2 段外的 L3 段被丢弃。
    """
    clipped: List[List[List[float]]] = []
    for i, l2_seg in enumerate(l2_segs):
        if i >= len(l3_nested):
            break
        l2_start, l2_end = l2_seg
        event_segs: List[List[float]] = []
        for seg in l3_nested[i]:
            s = max(seg[0], l2_start)
            e = min(seg[1], l2_end)
            if s < e:
                event_segs.append([s, e])
        clipped.append(event_segs)
    return clipped


def _has_oob(pred_l3_segs: List[List[float]], l2_seg: List[float]) -> bool:
    """检查是否有任何 L3 段越界 pred L2 边界。"""
    l2_start, l2_end = l2_seg
    for s, e in pred_l3_segs:
        if s < l2_start or e > l2_end:
            return True
    return False


# ===========================
# Ground-Seg Reward (单 caption, 单事件)
# ===========================
def ground_seg_reward(
    response: str,
    ground_truth: str,
) -> Dict[str, float]:
    """
    单 caption grounding + L3 segmentation reward.

    R_L2 = temporal_iou(pred_l2[0], gt_l2[0])
    R_L3 = F1-IoU(clipped_l3, gt_l3)
           如果 L3 有越界 pred L2 → R_L3 × 0.5
    overall = 0.4 × R_L2 + 0.6 × R_L3
    """
    if not _anti_hack_check(response):
        return dict(_ZERO)
    if not has_chain_tags(response):
        return dict(_ZERO)

    gt_l2 = parse_l2_events(ground_truth)
    gt_l3 = parse_l3_events(ground_truth)
    if not gt_l2:
        return dict(_ZERO)

    pred_l2 = parse_l2_events(response)
    pred_l3 = parse_l3_events(response)
    if not pred_l2:
        return dict(_ZERO)

    # L2: 单段 temporal_iou
    r_l2 = temporal_iou(pred_l2[0], gt_l2[0])

    # L3: 硬裁剪到 pred L2 bounds → F1-IoU，越界则砍半
    r_l3 = 0.0
    if gt_l3 and pred_l3 and gt_l3[0] and pred_l3[0]:
        l2_start, l2_end = pred_l2[0]
        oob = _has_oob(pred_l3[0], pred_l2[0])
        clipped = [
            [max(s, l2_start), min(e, l2_end)]
            for s, e in pred_l3[0]
            if max(s, l2_start) < min(e, l2_end)
        ]
        f1 = compute_f1_iou(clipped, gt_l3[0]) if clipped else 0.0
        r_l3 = f1 * OOB_PENALTY if oob else f1

    overall = W_L2 * r_l2 + W_L3 * r_l3

    return {
        "overall": float(overall),
        "format": 0.0,
        "accuracy": float(overall),
        "l2_reward": float(r_l2),
        "l3_reward": float(r_l3),
    }


# ===========================
# 统一 dispatch（EasyR1 batch reward 接口）
# ===========================
_DISPATCH = {
    "temporal_seg_chain_ground_seg": ground_seg_reward,
}


def compute_score(
    reward_inputs: List[Dict[str, Any]],
    **kwargs,
) -> List[Dict[str, float]]:
    """Batch reward 接口（与 EasyR1 BatchFunctionRewardManager 兼容）。"""
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for this reward function.")

    results: List[Dict[str, float]] = []

    for item in reward_inputs:
        try:
            response = item.get("response", "") or ""
            ground_truth = item.get("ground_truth", "") or ""
            problem_type = item.get("problem_type", "") or ""

            reward_fn = _DISPATCH.get(problem_type, ground_seg_reward)
            results.append(reward_fn(response, ground_truth))

        except Exception:
            results.append(dict(_ZERO))

    if random.random() < 0.05:
        for idx, item in enumerate(reward_inputs):
            print(f"[Chain-Seg] GT: {item.get('ground_truth', '')[:200]}")
            print(f"[Chain-Seg] Resp: {item.get('response', '')[:200]}")
            print(f"[Chain-Seg] Scores: {results[idx]}")

    return results
