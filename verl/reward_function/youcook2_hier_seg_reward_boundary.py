# Copyright 2024 Bytedance Ltd. and/or its affiliates
# -*- coding: utf-8 -*-
"""
YouCook2 分层时序分割 Reward 函数 — Boundary-Aware 变体。

与 youcook2_hier_seg_reward.py（F1-IoU）形成 Reward Ablation 对照:

  - F1-IoU (baseline):  段级最优匹配 + F1 调和平均
  - Boundary-Aware (本文件): 边界命中 F1 + 段数准确性 + 时序覆盖率

三大组件:
  1. Boundary Hit F1 (权重 0.5):
     - 提取 pred/gt 所有边界点 (start + end)
     - pred 边界在 τ 秒内命中 GT 边界 → 计为 hit
     - 贪心最近匹配，F1 = 2·P·R / (P+R)

  2. Count Accuracy (权重 0.2):
     - Gaussian 惩罚: exp(-|n_pred - n_gt|² / (2·σ²))
     - 鼓励模型预测正确的段数

  3. Coverage IoU (权重 0.3):
     - 将 pred/gt 各自合并为时间线 mask
     - IoU = mask 交集 / mask 并集
     - 确保整体时间覆盖正确

所有 level (L1/L2/L3) 统一使用相同 reward，与 baseline 消融实验对齐。
"""

import math
import random
import re
from typing import Any, Dict, List, Tuple

from verl.reward_function.youcook2_temporal_seg_reward import (
    has_events_tag,
    parse_segments,
)


# ===========================
# 超参数
# ===========================
BOUNDARY_TAU = 3.0        # 边界命中阈值 (秒)
COUNT_SIGMA = 2.0         # 段数高斯惩罚 σ
W_BOUNDARY = 0.5          # 边界 F1 权重
W_COVERAGE = 0.3          # 覆盖率 IoU 权重
W_COUNT = 0.2             # 段数准确性权重


# ===========================
# Anti-hacking 检查（与 baseline 对齐）
# ===========================

def _anti_hack_check(response: str) -> bool:
    """返回 True 表示通过，False 表示触发反作弊应得 0 分。"""
    if re.search(r"\[\d+-\d+\]", response):
        return False
    if response.count("<events>") > 1 or response.count("</events>") > 1:
        return False
    return True


_ZERO = {"overall": 0.0, "format": 0.0, "accuracy": 0.0}


# ===========================
# 组件 1: Boundary Hit F1
# ===========================

def _extract_boundaries(segments: List[List[float]]) -> List[float]:
    """从 segment 列表中提取所有边界点 (start, end 交替)"""
    boundaries = []
    for seg in segments:
        boundaries.append(seg[0])
        boundaries.append(seg[1])
    return sorted(boundaries)


def _boundary_hit_f1(
    pred_segs: List[List[float]],
    gt_segs: List[List[float]],
    tau: float = BOUNDARY_TAU,
) -> float:
    """
    边界命中 F1 (贪心最近匹配)。

    对每个 pred 边界，找最近 GT 边界，距离 ≤ τ 则命中。
    同一 GT 边界只能被命中一次（贪心匹配）。
    """
    pred_bounds = _extract_boundaries(pred_segs)
    gt_bounds = _extract_boundaries(gt_segs)

    if not pred_bounds or not gt_bounds:
        return 0.0

    n_pred = len(pred_bounds)
    n_gt = len(gt_bounds)

    # 构建距离矩阵 -> 贪心最近匹配
    pairs: List[Tuple[float, int, int]] = []
    for i, pb in enumerate(pred_bounds):
        for j, gb in enumerate(gt_bounds):
            dist = abs(pb - gb)
            if dist <= tau:
                pairs.append((dist, i, j))

    # 按距离排序，贪心匹配
    pairs.sort(key=lambda x: x[0])
    matched_pred = set()
    matched_gt = set()
    hits = 0

    for dist, pi, gj in pairs:
        if pi not in matched_pred and gj not in matched_gt:
            matched_pred.add(pi)
            matched_gt.add(gj)
            hits += 1

    recall = hits / n_gt if n_gt > 0 else 0.0
    precision = hits / n_pred if n_pred > 0 else 0.0
    denom = recall + precision
    return float(2.0 * recall * precision / denom) if denom > 0 else 0.0


# ===========================
# 组件 2: Count Accuracy
# ===========================

def _count_accuracy(n_pred: int, n_gt: int, sigma: float = COUNT_SIGMA) -> float:
    """段数准确性: Gaussian(|n_pred - n_gt|, σ)"""
    diff = n_pred - n_gt
    return math.exp(-(diff * diff) / (2.0 * sigma * sigma))


# ===========================
# 组件 3: Coverage IoU
# ===========================

def _merge_intervals(segments: List[List[float]]) -> List[List[float]]:
    """合并重叠区间"""
    if not segments:
        return []
    sorted_segs = sorted(segments, key=lambda s: s[0])
    merged = [sorted_segs[0][:]]
    for seg in sorted_segs[1:]:
        if seg[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], seg[1])
        else:
            merged.append(seg[:])
    return merged


def _interval_length(intervals: List[List[float]]) -> float:
    """计算区间列表总长度"""
    return sum(seg[1] - seg[0] for seg in intervals)


def _interval_intersection(
    a_intervals: List[List[float]],
    b_intervals: List[List[float]],
) -> float:
    """计算两组区间交集总长度 (双指针法)"""
    if not a_intervals or not b_intervals:
        return 0.0

    total = 0.0
    i, j = 0, 0
    while i < len(a_intervals) and j < len(b_intervals):
        start = max(a_intervals[i][0], b_intervals[j][0])
        end = min(a_intervals[i][1], b_intervals[j][1])
        if start < end:
            total += end - start
        if a_intervals[i][1] < b_intervals[j][1]:
            i += 1
        else:
            j += 1
    return total


def _coverage_iou(
    pred_segs: List[List[float]],
    gt_segs: List[List[float]],
) -> float:
    """
    时间线级别覆盖率 IoU。

    将 pred/gt 各自合并为不重叠区间，计算 IoU = 交集长度 / 并集长度。
    """
    pred_merged = _merge_intervals(pred_segs)
    gt_merged = _merge_intervals(gt_segs)

    inter = _interval_intersection(pred_merged, gt_merged)
    len_pred = _interval_length(pred_merged)
    len_gt = _interval_length(gt_merged)
    union = len_pred + len_gt - inter

    return float(inter / union) if union > 0 else 0.0


# ===========================
# Boundary-Aware Reward
# ===========================

def _boundary_reward(response: str, ground_truth: str) -> Dict[str, float]:
    """
    Boundary-Aware 三组件 reward。

    overall = W_BOUNDARY × boundary_f1 + W_COUNT × count_acc + W_COVERAGE × coverage_iou
    """
    gt_segs = parse_segments(ground_truth)
    if not gt_segs:
        return dict(_ZERO)

    if not _anti_hack_check(response):
        return dict(_ZERO)

    if not has_events_tag(response):
        return dict(_ZERO)

    pred_segs = parse_segments(response)
    if not pred_segs:
        return dict(_ZERO)

    # 三组件计算
    bf1 = _boundary_hit_f1(pred_segs, gt_segs)
    cnt = _count_accuracy(len(pred_segs), len(gt_segs))
    cov = _coverage_iou(pred_segs, gt_segs)

    overall = W_BOUNDARY * bf1 + W_COUNT * cnt + W_COVERAGE * cov

    return {
        "overall": float(overall),
        "format": 0.0,
        "accuracy": float(overall),
        "boundary_f1": float(bf1),
        "count_accuracy": float(cnt),
        "coverage_iou": float(cov),
    }


# ===========================
# Dispatch（与 baseline 完全对齐）
# ===========================

_DISPATCH = {
    "temporal_seg_hier_L1":     _boundary_reward,
    "temporal_seg_hier_L2":     _boundary_reward,
    "temporal_seg_hier_L3":     _boundary_reward,
    "temporal_seg_hier_L3_seg": _boundary_reward,
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

            reward_fn = _DISPATCH.get(problem_type)
            if reward_fn is None:
                if has_events_tag(ground_truth):
                    reward_fn = _boundary_reward
                else:
                    results.append(dict(_ZERO))
                    continue

            results.append(reward_fn(response, ground_truth))

        except Exception:
            results.append(dict(_ZERO))

    # 采样日志 (5%)
    if random.random() < 0.05:
        for idx, item in enumerate(reward_inputs[:3]):
            pt = item.get("problem_type", "?")
            gt = (item.get("ground_truth", "") or "")[:120]
            resp = (item.get("response", "") or "")[:200]
            print(
                f"[BoundaryReward] type={pt} | gt={gt!r} | "
                f"resp={resp!r} | scores={results[idx]}"
            )

    return results
