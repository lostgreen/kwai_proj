# Copyright 2024 Bytedance Ltd. and/or its affiliates
# -*- coding: utf-8 -*-
"""
YouCook2 分层时序分割 Reward 函数 — F1-IoU 分层奖励。

支持五种 problem_type：
  - temporal_seg_hier_L1:     宏观阶段分割    → F1-IoU
  - temporal_seg_hier_L2:     滑窗事件检测    → F1-IoU
  - temporal_seg_hier_L3:     原子操作分割    → F1-IoU
  - temporal_seg_hier_L3_seg: 同上（别名）    → F1-IoU
  - sort:                     事件排序        → Jigsaw Displacement

所有层统一 Hungarian 匹配 + NMS 去重 + F1-IoU。
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
# L1 / L2 (legacy): F1-IoU
# ===========================

def _l1_l2_reward(response: str, ground_truth: str) -> Dict[str, float]:
    """F1-IoU reward（旧版，保留供对比）。"""
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
# V2 → V3: 统一 F1-IoU reward
# ===========================


def _parse_and_check(response: str, ground_truth: str):
    """共用的解析+反作弊检查，返回 (pred_segs, gt_segs) 或 None。"""
    gt_segs = parse_segments(ground_truth)
    if not gt_segs:
        return None
    if not _anti_hack_check(response):
        return None
    if not has_events_tag(response):
        return None
    pred_segs = parse_segments(response)
    if not pred_segs:
        return None
    return pred_segs, gt_segs


def _l1_reward(response: str, ground_truth: str) -> Dict[str, float]:
    """L1: F1-IoU（宏观阶段分割）。"""
    result = _parse_and_check(response, ground_truth)
    if result is None:
        return dict(_ZERO)
    pred_segs, gt_segs = result
    f1 = compute_f1_iou(pred_segs, gt_segs)
    return {"overall": float(f1), "format": 0.0, "accuracy": float(f1)}


def _l2_reward(response: str, ground_truth: str) -> Dict[str, float]:
    """L2: F1-IoU（滑窗事件检测）。"""
    result = _parse_and_check(response, ground_truth)
    if result is None:
        return dict(_ZERO)
    pred_segs, gt_segs = result
    f1 = compute_f1_iou(pred_segs, gt_segs)
    return {"overall": float(f1), "format": 0.0, "accuracy": float(f1)}


def _l3_reward_v2(response: str, ground_truth: str) -> Dict[str, float]:
    """L3: F1-IoU（原子操作分割）。"""
    result = _parse_and_check(response, ground_truth)
    if result is None:
        return dict(_ZERO)
    pred_segs, gt_segs = result
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
# Sort: Jigsaw Displacement reward
# ===========================

_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_DIGIT_RE = re.compile(r"\d")


def _parse_sort_digits(text: str) -> List[int] | None:
    """解析排序序列 "13245" 或 "1 3 2 4 5" → [1, 3, 2, 4, 5]。"""
    if not text:
        return None
    digits = _DIGIT_RE.findall(text)
    if not digits:
        return None
    try:
        seq = [int(d) for d in digits]
        return seq if len(seq) >= 2 else None
    except (ValueError, TypeError):
        return None


def _compute_jigsaw_displacement(pred_seq: List[int], gt_seq: List[int]) -> float:
    """Jigsaw Displacement: R = 1 - E_jigsaw / E_max。"""
    n = len(gt_seq)
    if n <= 1:
        return 1.0 if pred_seq == gt_seq else 0.0
    gt_pos = {elem: i for i, elem in enumerate(gt_seq)}
    e_jigsaw = 0.0
    for i, elem in enumerate(pred_seq):
        gt_p = gt_pos.get(elem)
        if gt_p is None:
            e_jigsaw += n - 1
        else:
            e_jigsaw += abs(i - gt_p)
    e_max = sum(abs(i - (n - 1 - i)) for i in range(n))
    if e_max == 0:
        return 1.0
    return max(0.0, 1.0 - e_jigsaw / e_max)


def _sort_reward(response: str, ground_truth: str) -> Dict[str, float]:
    """排序题 Jigsaw Displacement reward。

    - 必须包含 <answer> 标签
    - 从 <answer> 内解析数字序列
    - 计算 jigsaw displacement reward
    """
    gt_seq = _parse_sort_digits(ground_truth)
    if gt_seq is None:
        return dict(_ZERO)

    matches = _ANSWER_TAG_RE.findall(response)
    if not matches:
        return {"overall": 0.0, "format": 0.0, "accuracy": 0.0}

    tag_content = matches[-1].strip()
    pred_seq = _parse_sort_digits(tag_content)
    if pred_seq is None:
        return {"overall": 0.0, "format": 1.0, "accuracy": 0.0}

    # 长度不匹配的处理
    if len(pred_seq) != len(gt_seq):
        if len(pred_seq) > len(gt_seq):
            pred_seq = pred_seq[:len(gt_seq)]
        else:
            missing = [e for e in gt_seq if e not in pred_seq]
            pred_seq = pred_seq + missing[:len(gt_seq) - len(pred_seq)]

    jigsaw_r = _compute_jigsaw_displacement(pred_seq, gt_seq)
    accuracy = 1.0 if pred_seq == gt_seq else jigsaw_r
    return {"overall": float(accuracy), "format": 1.0, "accuracy": float(accuracy)}


# ===========================
# 统一 dispatch（EasyR1 batch reward 接口）
# ===========================

_DISPATCH = {
    "temporal_seg_hier_L1":     _l1_reward,      # F1-IoU
    "temporal_seg_hier_L2":     _l2_reward,      # F1-IoU
    "temporal_seg_hier_L3":     _l3_reward_v2,   # F1-IoU
    "temporal_seg_hier_L3_seg": _l3_reward_v2,   # 同上
    "sort":                     _sort_reward,     # Jigsaw Displacement
}


def compute_score(
    reward_inputs: List[Dict[str, Any]],
    **kwargs,
) -> List[Dict[str, float]]:
    """
    Batch reward 接口（与 EasyR1 BatchFunctionRewardManager 兼容）。

    根据 problem_type 分发到对应 reward 函数（L1/L2/L3 均使用 F1-IoU）。
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
                # fallback: 如果 GT 里有 <events> 就用 F1-IoU；纯数字序列就用 sort
                if has_events_tag(ground_truth):
                    reward_fn = _l1_l2_reward
                elif re.match(r"^\d+$", ground_truth.strip()):
                    reward_fn = _sort_reward
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
