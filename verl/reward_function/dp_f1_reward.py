# -*- coding: utf-8 -*-
"""
DP-F1 + Instance Count Reward — TAL-style temporal segmentation reward.

Two complementary components:
  R_num:   Penalizes mismatch between predicted and ground-truth instance counts
           via exponential decay: exp(-|N_pred - N_gt| / (min(N_gt, 3) * sigma)).
  R_match: DP-based sequential matching → F1-IoU.
           Unlike Hungarian (unordered), DP preserves temporal ordering:
           earlier predictions match earlier ground truths.

  R_loc = R_num + R_match   (range [0, 2])

Reference: Algorithm 1 — DP matching of predicted intervals to ground truths.
"""

import math
import random
import re
from typing import Any, Dict, List, Tuple

from verl.reward_function.reward_utils import (
    has_events_tag,
    nms_1d,
    parse_segments,
    temporal_iou,
)

# ===========================
# Constants
# ===========================
SIGMA = 1.0  # R_num penalty sharpness


# ===========================
# Anti-hacking (shared with hier_seg_reward)
# ===========================

def _anti_hack_check(response: str) -> bool:
    if re.search(r"\[\d+-\d+\]", response):
        return False
    if response.count("<events>") > 1 or response.count("</events>") > 1:
        return False
    return True


_ZERO = {"overall": 0.0, "format": 0.0, "accuracy": 0.0, "r_num": 0.0, "r_match": 0.0}


# ===========================
# R_num: Instance Count Reward
# ===========================

def compute_instance_count_reward(
    n_pred: int,
    n_gt: int,
    sigma: float = SIGMA,
) -> float:
    """R_num = exp(-|N_pred - N_gt| / (min(N_gt, 3) * sigma)).

    Returns a value in [0, 1].  Perfect count → 1.0.
    """
    if n_gt == 0:
        return 0.0
    denom = min(n_gt, 3) * sigma
    if denom <= 0:
        return 0.0
    return math.exp(-abs(n_pred - n_gt) / denom)


# ===========================
# DP Matching (Algorithm 1)
# ===========================

def _dp_matching(
    pred_segs: List[List[float]],
    gt_segs: List[List[float]],
) -> Tuple[List[Tuple[int, int]], float]:
    """DP-based sequential matching of predicted intervals to ground truths.

    Both pred_segs and gt_segs must be sorted by start time.
    Preserves temporal ordering: earlier predictions match earlier GTs.

    Returns:
        matched_pairs: list of (pred_idx, gt_idx)
        total_iou:     sum of IoU for all matched pairs
    """
    m = len(pred_segs)
    n = len(gt_segs)

    # IoU matrix
    iou_mat = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            iou_mat[i][j] = temporal_iou(pred_segs[i], gt_segs[j])

    # DP table: D[i][j] = max total IoU using pred[0..i-1] and gt[0..j-1]
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    # Choice tracking: 0=skip_pred(from up), 1=skip_gt(from left), 2=match(from diag)
    choice = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            a = dp[i - 1][j]      # skip pred[i-1]
            b = dp[i][j - 1]      # skip gt[j-1]
            c = dp[i - 1][j - 1] + iou_mat[i - 1][j - 1]  # match

            if c >= a and c >= b:
                dp[i][j] = c
                choice[i][j] = 2
            elif a >= b:
                dp[i][j] = a
                choice[i][j] = 0
            else:
                dp[i][j] = b
                choice[i][j] = 1

    # Backtrack to recover matched pairs
    matched = []
    i, j = m, n
    while i > 0 and j > 0:
        if choice[i][j] == 2:
            matched.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif choice[i][j] == 0:
            i -= 1
        else:
            j -= 1

    matched.reverse()
    total_iou = dp[m][n]
    return matched, total_iou


# ===========================
# R_match: F1-IoU from DP matching
# ===========================

def compute_dp_f1_iou(
    pred_segs: List[List[float]],
    gt_segs: List[List[float]],
) -> float:
    """F1-IoU using DP sequential matching (no NMS).

    P = s_IoU / N_pred,  R = s_IoU / N_gt,  F1 = 2*P*R / (P+R).
    """
    n_pred = len(pred_segs)
    n_gt = len(gt_segs)
    if n_pred == 0 or n_gt == 0:
        return 0.0

    # Sort by start time (DP assumes ordering)
    pred_sorted = sorted(pred_segs, key=lambda s: s[0])
    gt_sorted = sorted(gt_segs, key=lambda s: s[0])

    _, s_iou = _dp_matching(pred_sorted, gt_sorted)

    precision = s_iou / n_pred
    recall = s_iou / n_gt
    denom = precision + recall
    return float(2.0 * precision * recall / denom) if denom > 0 else 0.0


# ===========================
# Combined: R_loc = R_num + R_match
# ===========================

def _dp_f1_reward(
    response: str,
    ground_truth: str,
    sigma: float = SIGMA,
) -> Dict[str, float]:
    """DP-F1 + Instance Count reward.  overall = R_num + R_match ∈ [0, 2]."""
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

    # NMS dedup (same as Hungarian version for fair comparison)
    pred_segs = nms_1d(pred_segs)

    r_num = compute_instance_count_reward(len(pred_segs), len(gt_segs), sigma)
    r_match = compute_dp_f1_iou(pred_segs, gt_segs)
    r_loc = r_num + r_match

    return {
        "overall": float(r_loc),
        "format": 0.0,
        "accuracy": float(r_match),
        "r_num": float(r_num),
        "r_match": float(r_match),
    }


# ===========================
# Dispatch
# ===========================

_DISPATCH = {
    "temporal_seg_hier_L1":     _dp_f1_reward,
    "temporal_seg_hier_L2":     _dp_f1_reward,
    "temporal_seg_hier_L3_seg": _dp_f1_reward,
}


def compute_score(
    reward_inputs: List[Dict[str, Any]],
    **kwargs,
) -> List[Dict[str, float]]:
    """Batch reward interface (EasyR1 compatible).

    R_loc = R_num + R_match ∈ [0, 2].
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
                    reward_fn = _dp_f1_reward
                else:
                    results.append(dict(_ZERO))
                    continue

            results.append(reward_fn(response, ground_truth))

        except Exception:
            results.append(dict(_ZERO))

    if random.random() < 0.05:
        for idx, item in enumerate(reward_inputs[:3]):
            pt = item.get("problem_type", "?")
            gt = (item.get("ground_truth", "") or "")[:120]
            resp = (item.get("response", "") or "")[:200]
            r = results[idx]
            print(
                f"[DP-F1] type={pt} | r_num={r.get('r_num', 0):.3f} "
                f"r_match={r.get('r_match', 0):.3f} | overall={r['overall']:.3f}"
            )

    return results
