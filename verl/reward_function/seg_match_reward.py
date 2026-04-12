# -*- coding: utf-8 -*-
"""
Segment Matching Reward — Global + Local matching.

Two complementary components:
  r_G (Global):  Coverage overlap ratio of the union of all GT segments
                 with the union of all predicted segments.
  r_L (Local):   Sort-based positional matching → mean NGIoU.
                 N = max(|GT|, |pred|); unmatched segments pair with φ → NGIoU = 0.

  r_M = (r_G + r_L) / 2   (range [0, 1])

Key properties:
  - NGIoU provides gradient signal even when pred/gt don't overlap.
  - Count mismatch is penalized by averaging over N = max(|GT|, |pred|).
  - r_G captures video-level coverage; r_L captures per-segment precision.
"""

import random
import re
from typing import Any, Dict, List

from verl.reward_function.reward_utils import (
    has_events_tag,
    ngiou,
    nms_1d,
    parse_segments,
)


# ===========================
# Anti-hacking (shared)
# ===========================

def _anti_hack_check(response: str) -> bool:
    if re.search(r"\[\d+-\d+\]", response):
        return False
    if response.count("<events>") > 1 or response.count("</events>") > 1:
        return False
    return True


_ZERO = {"overall": 0.0, "format": 0.0, "accuracy": 0.0, "r_global": 0.0, "r_local": 0.0}


# ===========================
# r_G: Global Coverage Overlap
# ===========================

def _merge_intervals(segs: List[List[float]]) -> List[List[float]]:
    """Merge overlapping intervals into non-overlapping set."""
    if not segs:
        return []
    sorted_segs = sorted(segs, key=lambda s: s[0])
    merged = [sorted_segs[0][:]]
    for s in sorted_segs[1:]:
        if s[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], s[1])
        else:
            merged.append(s[:])
    return merged


def _total_length(segs: List[List[float]]) -> float:
    """Total duration of non-overlapping intervals."""
    return sum(s[1] - s[0] for s in segs)


def _intersection_length(merged_a: List[List[float]], merged_b: List[List[float]]) -> float:
    """Compute total intersection of two sorted, non-overlapping interval sets."""
    total = 0.0
    i, j = 0, 0
    while i < len(merged_a) and j < len(merged_b):
        start = max(merged_a[i][0], merged_b[j][0])
        end = min(merged_a[i][1], merged_b[j][1])
        if start < end:
            total += end - start
        if merged_a[i][1] < merged_b[j][1]:
            i += 1
        else:
            j += 1
    return total


def compute_global_overlap(
    pred_segs: List[List[float]],
    gt_segs: List[List[float]],
) -> float:
    """r_G = Σ_{i,j} |G_i ∩ P_j| / |(∪G_i) ∪ (∪P_j)|.

    Equivalent to: intersection(∪G, ∪P) / union(∪G, ∪P)  (IoU of merged unions).
    """
    merged_gt = _merge_intervals(gt_segs)
    merged_pred = _merge_intervals(pred_segs)

    inter = _intersection_length(merged_gt, merged_pred)
    len_gt = _total_length(merged_gt)
    len_pred = _total_length(merged_pred)
    union = len_gt + len_pred - inter

    return inter / union if union > 0 else 0.0


# ===========================
# r_L: Local Positional Matching (mean NGIoU)
# ===========================

def compute_local_ngiou(
    pred_segs: List[List[float]],
    gt_segs: List[List[float]],
) -> float:
    """r_L = Σ NGIoU(G_n, P_n) / N, where N = max(|GT|, |pred|).

    Both lists sorted by start time. Unmatched segments pair with φ → NGIoU = 0.
    """
    pred_sorted = sorted(pred_segs, key=lambda s: s[0])
    gt_sorted = sorted(gt_segs, key=lambda s: s[0])

    n_pred = len(pred_sorted)
    n_gt = len(gt_sorted)
    n_matched = min(n_pred, n_gt)
    N = max(n_pred, n_gt)

    if N == 0:
        return 0.0

    total_ngiou = sum(ngiou(pred_sorted[k], gt_sorted[k]) for k in range(n_matched))
    # Unmatched pairs → NGIoU(G, φ) = NGIoU(φ, P) = 0, already covered by N denominator
    return total_ngiou / N


# ===========================
# Combined: r_M = (r_G + r_L) / 2
# ===========================

def _seg_match_reward(
    response: str,
    ground_truth: str,
) -> Dict[str, float]:
    """Segment Matching Reward.  overall = r_M ∈ [0, 1]."""
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

    pred_segs = nms_1d(pred_segs)

    r_g = compute_global_overlap(pred_segs, gt_segs)
    r_l = compute_local_ngiou(pred_segs, gt_segs)
    r_m = (r_g + r_l) / 2.0

    return {
        "overall": float(r_m),
        "format": 0.0,
        "accuracy": float(r_m),
        "r_global": float(r_g),
        "r_local": float(r_l),
    }


# ===========================
# Dispatch
# ===========================

_DISPATCH = {
    "temporal_seg_hier_L1":     _seg_match_reward,
    "temporal_seg_hier_L2":     _seg_match_reward,
    "temporal_seg_hier_L3_seg": _seg_match_reward,
}


def compute_score(
    reward_inputs: List[Dict[str, Any]],
    **kwargs,
) -> List[Dict[str, float]]:
    """Batch reward interface (EasyR1 compatible).

    r_M = (r_G + r_L) / 2 ∈ [0, 1].
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
                    reward_fn = _seg_match_reward
                else:
                    results.append(dict(_ZERO))
                    continue

            results.append(reward_fn(response, ground_truth))

        except Exception:
            results.append(dict(_ZERO))

    if random.random() < 0.05:
        for idx, item in enumerate(reward_inputs[:3]):
            pt = item.get("problem_type", "?")
            r = results[idx]
            print(
                f"[SegMatch] type={pt} | r_G={r.get('r_global', 0):.3f} "
                f"r_L={r.get('r_local', 0):.3f} | overall={r['overall']:.3f}"
            )

    return results
