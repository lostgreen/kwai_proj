# Copyright 2024 Bytedance Ltd. and/or its affiliates
# -*- coding: utf-8 -*-
"""
YouCook2 分层时序分割 Reward 函数。

支持三种 problem_type：
  - temporal_seg_hier_L1: 宏观阶段分割 → F1-IoU（NMS + 匈牙利匹配）
  - temporal_seg_hier_L2: 滑窗事件检测 → F1-IoU（同 L1）
  - temporal_seg_hier_L3: 查询驱动定位 → position-aligned mean tIoU

L1/L2: 预测段数不定，需 NMS 去重 + 匈牙利匹配 → F1-IoU
L3:    预测段数 = query 数（固定），位置对齐 → 逐位 tIoU 均值

所有 level 统一 <events>[[s,e],...]</events> 格式。
"""

import random
import re
from typing import Any, Dict, List

from verl.reward_function.youcook2_temporal_seg_reward import (
    compute_f1_iou,
    has_events_tag,
    parse_segments,
    temporal_iou,
)


# ===========================
# Anti-hacking 检查（共用）
# ===========================

def _anti_hack_check(response: str) -> bool:
    """返回 True 表示通过，False 表示触发反作弊应得 0 分。"""
    # 畸形 [数字-数字] 格式
    if re.search(r"\[\d+-\d+\]", response):
        return False
    # 多重 <events> / </events> 复读
    if response.count("<events>") > 1 or response.count("</events>") > 1:
        return False
    return True


_ZERO = {"overall": 0.0, "format": 0.0, "accuracy": 0.0}


# ===========================
# L1 / L2: F1-IoU
# ===========================

def _l1_l2_reward(response: str, ground_truth: str) -> Dict[str, float]:
    """
    F1-IoU reward（复用 youcook2_temporal_seg_reward 核心逻辑）。

    适用 L1（宏观阶段）和 L2（滑窗事件），两者均为变长 segment 列表。
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

    f1 = compute_f1_iou(pred_segs, gt_segs)
    return {"overall": float(f1), "format": 0.0, "accuracy": float(f1)}


# ===========================
# L3: position-aligned mean tIoU
# ===========================

def compute_aligned_iou(
    pred_segs: List[List[float]],
    gt_segs: List[List[float]],
) -> float:
    """
    位置对齐的 mean tIoU —— L3 专用。

    pred[i] 与 gt[i] 直接配对，无需匈牙利匹配。

    分母取 max(n_pred, n_gt)：
      - 少输出：缺失段贡献 0，分母 = n_gt  → 自动惩罚
      - 多输出：多余段不计分，分母 = n_pred → 精度被稀释
      - 数目正确：退化为 standard mean tIoU
    """
    n_gt = len(gt_segs)
    n_pred = len(pred_segs)
    if n_gt == 0 or n_pred == 0:
        return 0.0

    n_eval = min(n_pred, n_gt)
    sum_iou = sum(temporal_iou(pred_segs[i], gt_segs[i]) for i in range(n_eval))
    return sum_iou / max(n_pred, n_gt)


def _l3_reward(response: str, ground_truth: str) -> Dict[str, float]:
    """
    L3 query-conditioned grounding reward。

    输出段数应等于 query 数（= GT 段数），位置一一对齐。
    不使用 NMS（每段对应不同 query，语义不同不应合并）。
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

    acc = compute_aligned_iou(pred_segs, gt_segs)
    return {"overall": float(acc), "format": 0.0, "accuracy": float(acc)}


# ===========================
# 统一 dispatch（EasyR1 batch reward 接口）
# ===========================

_DISPATCH = {
    "temporal_seg_hier_L1": _l1_l2_reward,
    "temporal_seg_hier_L2": _l1_l2_reward,
    "temporal_seg_hier_L3": _l3_reward,
    "temporal_seg_hier_L3_seg": _l1_l2_reward,  # L3 segmentation: F1-IoU (like L1/L2)
}


def compute_score(
    reward_inputs: List[Dict[str, Any]],
    **kwargs,
) -> List[Dict[str, float]]:
    """
    Batch reward 接口（与 EasyR1 BatchFunctionRewardManager 兼容）。

    根据 problem_type 分发到 L1/L2（F1-IoU）或 L3（aligned-IoU）。
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
                # fallback: 如果 GT 里有 <events> 就用 F1-IoU
                if has_events_tag(ground_truth):
                    reward_fn = _l1_l2_reward
                else:
                    results.append(dict(_ZERO))
                    continue

            results.append(reward_fn(response, ground_truth))

        except Exception:
            results.append(dict(_ZERO))

    # 采样日志（5% 概率）
    if random.random() < 0.05:
        for idx, item in enumerate(reward_inputs[:3]):
            pt = item.get("problem_type", "?")
            gt = (item.get("ground_truth", "") or "")[:120]
            resp = (item.get("response", "") or "")[:200]
            print(
                f"[HierSeg] type={pt} | gt={gt!r} | "
                f"resp={resp!r} | scores={results[idx]}"
            )

    return results
