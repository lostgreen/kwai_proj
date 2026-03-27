# Copyright 2024 Bytedance Ltd. and/or its affiliates
# -*- coding: utf-8 -*-
"""
YouCook2 链式层次分割 Reward 函数 (Chain-of-Segment)。

支持三种 problem_type:
  - temporal_seg_chain_L2L3:       原始链式 (多 caption grounding + L3 seg)
  - temporal_seg_chain_dual_seg:   V1 双层自由分割 (无 caption)
  - temporal_seg_chain_ground_seg: V2 单 caption grounding + L3 seg

输出格式:
    <l2_events>[[s1, e1], [s2, e2], ...]</l2_events>
    <l3_events>[[[a1, b1], [a2, b2]], [[c1, d1]], ...]</l3_events>

Reward 结构:
    R = w_l2 × R_L2 + w_l3 × R_L3 × φ(R_L2)
    φ(x) = max(x, CASCADE_FLOOR)

L3 边界约束: L3 段在评估前被裁剪到对应 L2 段边界内。

兼容 EasyR1 BatchFunctionRewardManager 接口。
"""

import random
import re
from typing import Any, Dict, List, Optional

from verl.reward_function.youcook2_temporal_seg_reward import (
    _hungarian_assignment,
    compute_f1_iou,
    temporal_iou,
)


# ===========================
# 常量
# ===========================
W_L2 = 0.4             # L2 权重 (chain_reward / ground_seg_reward)
W_L3 = 0.6             # L3 权重 (chain_reward / ground_seg_reward)
CASCADE_FLOOR = 0.3     # 级联因子下限

# V1 (dual-seg) 专用: L2/L3 任务对称, 等权
_W_L2_DUAL = 0.5
_W_L3_DUAL = 0.5

# L3 越界惩罚强度 ∈ (0, 1]:
#   1.0 → score × compliance        (强惩罚, 完全越界得 0)
#   0.3 → score × (1 - 0.3*(1-c))  (弱惩罚, 完全越界仍保留 70%)
BOUNDARY_PENALTY_ALPHA = 0.3


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
    """
    解析 <l3_events>...</l3_events> 中的嵌套 L3 段列表。

    格式: [[[a1, b1], [a2, b2]], [[c1, d1]], ...]
    返回: 二维列表，外层对应 L2 事件，内层为该事件的 L3 段
    """
    match = L3_EVENTS_PATTERN.search(str(text))
    if match is None:
        return []

    raw = match.group(1).strip()
    if not raw:
        return []

    # 策略: 按顶层 [] 拆分 — 找到最外层的 [...] 对
    # 嵌套结构: [[[a,b],[c,d]], [[e,f]]]
    # 需要解析出 [[a,b],[c,d]] 和 [[e,f]] 两个子列表
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
    # 畸形 [数字-数字] 格式
    if re.search(r"\[\d+-\d+\]", response):
        return False
    # 多重标签复读
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


def _l3_boundary_compliance(
    pred_l3_segs: List[List[float]],
    l2_seg: List[float],
) -> float:
    """计算 L3 预测段在 L2 父段边界内的时长比例。

    返回值 ∈ [0, 1]:
        1.0 = 全部在 L2 范围内 (无惩罚)
        0.0 = 全部在 L2 范围外 (最大惩罚)
    """
    if not pred_l3_segs:
        return 1.0
    l2_start, l2_end = l2_seg
    original_total = sum(max(0.0, e - s) for s, e in pred_l3_segs)
    if original_total <= 0:
        return 1.0
    in_bounds_total = sum(
        max(0.0, min(e, l2_end) - max(s, l2_start))
        for s, e in pred_l3_segs
    )
    return in_bounds_total / original_total


def _compute_l3_with_boundary(
    pred_l3: List[List[List[float]]],
    gt_l3: List[List[List[float]]],
    pred_l2: List[List[float]],
) -> float:
    """逐 L2 事件计算 L3 F1-IoU，附带软边界合规惩罚。

    每事件得分 = F1-IoU(clipped L3, gt_l3) × penalty_factor
    penalty_factor = 1 - BOUNDARY_PENALTY_ALPHA * (1 - compliance)
    compliance    = in_bounds_duration / total_pred_duration

    最终取所有 GT 事件的均值（缺失事件贡献 0）。
    """
    n_gt = len(gt_l3)
    if n_gt == 0:
        return 0.0

    total_score = 0.0
    for i in range(n_gt):
        if i >= len(pred_l3) or not pred_l3[i] or not gt_l3[i]:
            continue

        if i >= len(pred_l2):
            # 无对应 L2 段 → 不裁剪也不惩罚，直接 F1-IoU
            total_score += compute_f1_iou(pred_l3[i], gt_l3[i])
            continue

        l2_seg = pred_l2[i]
        compliance = _l3_boundary_compliance(pred_l3[i], l2_seg)
        penalty_factor = 1.0 - BOUNDARY_PENALTY_ALPHA * (1.0 - compliance)

        # 裁剪后计算 F1-IoU
        l2_start, l2_end = l2_seg
        clipped = [
            [max(s, l2_start), min(e, l2_end)]
            for s, e in pred_l3[i]
            if max(s, l2_start) < min(e, l2_end)
        ]
        f1 = compute_f1_iou(clipped, gt_l3[i]) if clipped else 0.0
        total_score += f1 * penalty_factor

    return total_score / n_gt


# ===========================
# L2 Grounding Reward (position-aligned mean tIoU)
# ===========================
def compute_aligned_iou(
    pred_segs: List[List[float]],
    gt_segs: List[List[float]],
) -> float:
    """
    位置对齐的 mean tIoU — L2 grounding 评估。

    pred[i] 与 gt[i] 直接配对（caption 给出了对齐关系）。
    分母取 max(n_pred, n_gt)，自动惩罚缺失/多余段。
    """
    n_gt = len(gt_segs)
    n_pred = len(pred_segs)
    if n_gt == 0 or n_pred == 0:
        return 0.0

    n_eval = min(n_pred, n_gt)
    sum_iou = sum(temporal_iou(pred_segs[i], gt_segs[i]) for i in range(n_eval))
    return sum_iou / max(n_pred, n_gt)


# ===========================
# L3 Per-Event Segmentation Reward
# ===========================
def compute_l3_reward(
    pred_l3: List[List[List[float]]],
    gt_l3: List[List[List[float]]],
) -> float:
    """
    逐 L2 事件的 L3 F1-IoU 均值。

    pred_l3[i] 和 gt_l3[i] 分别是第 i 个 L2 事件内的 L3 预测/GT 段列表。
    - 如果 pred 的 L2 事件数少于 GT，缺失事件贡献 0
    - 如果 pred 的 L2 事件数多于 GT，多余事件忽略
    """
    n_gt = len(gt_l3)
    if n_gt == 0:
        return 0.0

    n_pred = len(pred_l3)
    total_score = 0.0

    for i in range(n_gt):
        if i < n_pred and pred_l3[i] and gt_l3[i]:
            total_score += compute_f1_iou(pred_l3[i], gt_l3[i])
        # else: contribute 0 for missing events

    return total_score / n_gt


# ===========================
# 链式级联 Reward
# ===========================
def chain_reward(
    response: str,
    ground_truth: str,
) -> Dict[str, float]:
    """
    Chain-of-Segment reward: L2 grounding + L3 per-event segmentation + 级联因子。

    R = W_L2 × R_L2 + W_L3 × R_L3 × φ(R_L2)
    φ(x) = max(x, CASCADE_FLOOR)
    """
    # --- 反作弊 ---
    if not _anti_hack_check(response):
        return dict(_ZERO)

    # --- 格式检查 ---
    if not has_chain_tags(response):
        return dict(_ZERO)

    # --- 解析 GT ---
    gt_l2 = parse_l2_events(ground_truth)
    gt_l3 = parse_l3_events(ground_truth)
    if not gt_l2:
        return dict(_ZERO)

    # --- 解析 Pred ---
    pred_l2 = parse_l2_events(response)
    pred_l3 = parse_l3_events(response)
    if not pred_l2:
        return dict(_ZERO)

    # --- L2 Reward ---
    r_l2 = compute_aligned_iou(pred_l2, gt_l2)

    # --- L3 Reward (boundary clip + compliance penalty) ---
    if gt_l3 and pred_l3:
        r_l3 = _compute_l3_with_boundary(pred_l3, gt_l3, pred_l2)
    else:
        r_l3 = 0.0

    # --- 级联组合 ---
    cascade = max(r_l2, CASCADE_FLOOR)
    overall = W_L2 * r_l2 + W_L3 * r_l3 * cascade

    return {
        "overall": float(overall),
        "format": 0.0,
        "accuracy": float(overall),
        "l2_reward": float(r_l2),
        "l3_reward": float(r_l3),
    }


# ===========================
# V1: Dual-Seg Reward (无 caption, Hungarian 匹配 L2)
# ===========================
def dual_seg_reward(
    response: str,
    ground_truth: str,
) -> Dict[str, float]:
    """
    双层自由分割 reward: L2 用 F1-IoU (Hungarian), L3 按匹配配对后 F1-IoU.

    L2/L3 等权 (_W_L2_DUAL = _W_L3_DUAL = 0.5)
    R = 0.5 × R_L2 + 0.5 × R_L3 × φ(R_L2)
    L3 越界软惩罚: score × (1 - BOUNDARY_PENALTY_ALPHA × (1 - compliance))
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

    # L2: F1-IoU (unordered, Hungarian matching)
    r_l2 = compute_f1_iou(pred_l2, gt_l2)

    # L3: Hungarian 匹配 L2 后, 按配对评估 L3
    r_l3 = 0.0
    if gt_l3 and pred_l3:
        # 构建 IoU cost matrix for L2 Hungarian matching
        n_pred = len(pred_l2)
        n_gt = len(gt_l2)
        cost_matrix = []
        for i in range(n_pred):
            row = [1.0 - temporal_iou(pred_l2[i], gt_l2[j]) for j in range(n_gt)]
            cost_matrix.append(row)
        matches = _hungarian_assignment(cost_matrix)

        # 按 matched L2 配对, clip L3 到 L2 bounds + compliance 惩罚
        l3_scores = []
        matched_gt_indices = set()
        for pred_idx, gt_idx in matches:
            matched_gt_indices.add(gt_idx)
            if pred_idx >= len(pred_l3) or gt_idx >= len(gt_l3):
                l3_scores.append(0.0)
                continue
            pred_l3_segs = pred_l3[pred_idx]
            gt_l3_segs = gt_l3[gt_idx]
            if not pred_l3_segs or not gt_l3_segs:
                l3_scores.append(0.0)
                continue
            # 软 compliance 惩罚
            compliance = _l3_boundary_compliance(pred_l3_segs, pred_l2[pred_idx])
            penalty_factor = 1.0 - BOUNDARY_PENALTY_ALPHA * (1.0 - compliance)
            # clip L3 to matched L2 bounds
            l2_start, l2_end = pred_l2[pred_idx]
            clipped = [
                [max(s, l2_start), min(e, l2_end)]
                for s, e in pred_l3_segs
                if max(s, l2_start) < min(e, l2_end)
            ]
            f1 = compute_f1_iou(clipped, gt_l3_segs) if clipped else 0.0
            l3_scores.append(f1 * penalty_factor)

        # 未匹配的 GT 事件贡献 0
        for gt_idx in range(n_gt):
            if gt_idx not in matched_gt_indices:
                l3_scores.append(0.0)

        r_l3 = sum(l3_scores) / len(gt_l3) if gt_l3 else 0.0

    cascade = max(r_l2, CASCADE_FLOOR)
    overall = _W_L2_DUAL * r_l2 + _W_L3_DUAL * r_l3 * cascade

    return {
        "overall": float(overall),
        "format": 0.0,
        "accuracy": float(overall),
        "l2_reward": float(r_l2),
        "l3_reward": float(r_l3),
    }


# ===========================
# V2: Ground-Seg Reward (单 caption, 单事件)
# ===========================
def ground_seg_reward(
    response: str,
    ground_truth: str,
) -> Dict[str, float]:
    """
    单 caption grounding + L3 segmentation reward.

    L2: temporal_iou (单段配对)
    L3: clip 到 pred L2 bounds → F1-IoU
    R = W_L2 × R_L2 + W_L3 × R_L3 × φ(R_L2)
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

    # L3: clip 到 pred L2 bounds + 软 compliance 惩罚 → F1-IoU
    r_l3 = 0.0
    if gt_l3 and pred_l3:
        gt_l3_first = gt_l3[0] if gt_l3 else []
        if gt_l3_first and pred_l3[0]:
            compliance = _l3_boundary_compliance(pred_l3[0], pred_l2[0])
            penalty_factor = 1.0 - BOUNDARY_PENALTY_ALPHA * (1.0 - compliance)
            l2_start, l2_end = pred_l2[0]
            clipped = [
                [max(s, l2_start), min(e, l2_end)]
                for s, e in pred_l3[0]
                if max(s, l2_start) < min(e, l2_end)
            ]
            f1 = compute_f1_iou(clipped, gt_l3_first) if clipped else 0.0
            r_l3 = f1 * penalty_factor

    cascade = max(r_l2, CASCADE_FLOOR)
    overall = W_L2 * r_l2 + W_L3 * r_l3 * cascade

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
    "temporal_seg_chain_L2L3":       chain_reward,
    "temporal_seg_chain_dual_seg":   dual_seg_reward,
    "temporal_seg_chain_ground_seg": ground_seg_reward,
}


def compute_score(
    reward_inputs: List[Dict[str, Any]],
    **kwargs,
) -> List[Dict[str, float]]:
    """
    Batch reward 接口（与 EasyR1 BatchFunctionRewardManager 兼容）。
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for this reward function.")

    results: List[Dict[str, float]] = []

    for item in reward_inputs:
        try:
            response = item.get("response", "") or ""
            ground_truth = item.get("ground_truth", "") or ""
            problem_type = item.get("problem_type", "") or ""

            reward_fn = _DISPATCH.get(problem_type, chain_reward)
            results.append(reward_fn(response, ground_truth))

        except Exception:
            results.append(dict(_ZERO))

    # 采样日志（5% 概率）
    if random.random() < 0.05:
        for idx, item in enumerate(reward_inputs):
            print(f"[Chain-Seg] GT: {item.get('ground_truth', '')[:200]}")
            print(f"[Chain-Seg] Resp: {item.get('response', '')[:200]}")
            print(f"[Chain-Seg] Scores: {results[idx]}")

    return results
