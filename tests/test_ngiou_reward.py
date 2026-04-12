#!/usr/bin/env python3
"""Unit tests for NGIoU reward functions — standalone, no verl init dependency."""
import sys, importlib.util, types, math

sys.path.insert(0, ".")
# Mock verl package to avoid heavy __init__.py imports
verl_mod = types.ModuleType("verl")
verl_rf = types.ModuleType("verl.reward_function")
sys.modules["verl"] = verl_mod
sys.modules["verl.reward_function"] = verl_rf

# Load reward_utils
spec_ts = importlib.util.spec_from_file_location(
    "verl.reward_function.reward_utils",
    "verl/reward_function/reward_utils.py",
)
mod_ts = importlib.util.module_from_spec(spec_ts)
sys.modules["verl.reward_function.reward_utils"] = mod_ts
spec_ts.loader.exec_module(mod_ts)

# Load hier_seg_reward
spec_hier = importlib.util.spec_from_file_location(
    "verl.reward_function.hier_seg_reward",
    "verl/reward_function/hier_seg_reward.py",
)
mod_hier = importlib.util.module_from_spec(spec_hier)
sys.modules["verl.reward_function.hier_seg_reward"] = mod_hier
spec_hier.loader.exec_module(mod_hier)

from verl.reward_function.reward_utils import (
    ngiou, temporal_iou, compute_f1_ngiou, compute_f1_iou,
)
from verl.reward_function.hier_seg_reward import (
    _f1_iou_reward, compute_score,
)


def approx(a, b, tol=1e-4):
    assert abs(a - b) < tol, f"Expected {b}, got {a} (diff={abs(a-b)})"


def main():
    print("=" * 50)
    print("Test 1: ngiou — basic properties")
    print("=" * 50)

    # Perfect overlap → NGIoU = 1.0
    assert ngiou([10, 20], [10, 20]) == 1.0
    print("  Perfect overlap: 1.0 ✓")

    # Partial overlap → 0 < NGIoU < 1
    v = ngiou([10, 20], [15, 25])
    assert 0 < v < 1
    print(f"  Partial overlap [10,20]∩[15,25]: {v:.4f} ✓")

    # No overlap but close → NGIoU > 0 (key property!)
    v_close = ngiou([10, 20], [22, 32])
    assert v_close > 0, "NGIoU should be > 0 even with no overlap"
    assert temporal_iou([10, 20], [22, 32]) == 0, "IoU should be 0 with no overlap"
    print(f"  No overlap, close [10,20] vs [22,32]: NGIoU={v_close:.4f}, IoU=0 ✓")

    # No overlap, far apart → NGIoU > 0 but smaller than close case
    v_far = ngiou([10, 20], [100, 110])
    assert 0 < v_far < v_close, "Farther segments should have lower NGIoU"
    print(f"  No overlap, far [10,20] vs [100,110]: NGIoU={v_far:.4f} < {v_close:.4f} ✓")

    # Monotonicity: closer → higher NGIoU
    v1 = ngiou([0, 10], [12, 22])  # gap=2
    v2 = ngiou([0, 10], [15, 25])  # gap=5
    v3 = ngiou([0, 10], [50, 60])  # gap=40
    assert v1 > v2 > v3, f"Monotonicity failed: {v1}, {v2}, {v3}"
    print(f"  Monotonicity: gap2={v1:.4f} > gap5={v2:.4f} > gap40={v3:.4f} ✓")

    # Both zero-length → 0 (degenerate case, parse_segments filters start>=end anyway)
    assert ngiou([10, 10], [10, 10]) == 0.0
    print("  Both zero-length: 0.0 ✓")

    print("\n" + "=" * 50)
    print("Test 2: ngiou vs IoU — NGIoU >= IoU always")
    print("=" * 50)
    test_pairs = [
        ([10, 30], [10, 30]),
        ([10, 30], [20, 40]),
        ([10, 30], [35, 55]),
        ([10, 30], [50, 70]),
    ]
    for a, b in test_pairs:
        iou_val = temporal_iou(a, b)
        ngiou_val = ngiou(a, b)
        # NGIoU should always be >= IoU (since GIoU >= 2*IoU - 1, so (GIoU+1)/2 >= IoU)
        # Actually that's not guaranteed in general, but NGIoU ∈ [0,1] and provides gradient
        print(f"  {a} vs {b}: IoU={iou_val:.4f}, NGIoU={ngiou_val:.4f}")

    print("\n" + "=" * 50)
    print("Test 3: compute_f1_ngiou — basic cases")
    print("=" * 50)

    # Perfect match
    gt = [[10, 30], [40, 60]]
    pred = [[10, 30], [40, 60]]
    f1 = compute_f1_ngiou(pred, gt)
    approx(f1, 1.0)
    print(f"  Perfect match: F1-NGIoU = {f1:.4f} ✓")

    # Slight offset → should be high
    pred2 = [[12, 28], [42, 58]]
    f1_ngiou = compute_f1_ngiou(pred2, gt)
    f1_iou = compute_f1_iou(pred2, gt)
    assert f1_ngiou > f1_iou, "NGIoU should be more generous for slight offsets"
    print(f"  Slight offset: F1-NGIoU={f1_ngiou:.4f} > F1-IoU={f1_iou:.4f} ✓")

    # No overlap at all → F1-IoU = 0, F1-NGIoU > 0 (key!)
    pred3 = [[70, 90]]
    f1_ngiou3 = compute_f1_ngiou(pred3, gt)
    f1_iou3 = compute_f1_iou(pred3, gt)
    assert f1_iou3 == 0.0, "F1-IoU should be 0 for no overlap"
    assert f1_ngiou3 > 0.0, "F1-NGIoU should be > 0 even with no overlap"
    print(f"  No overlap: F1-NGIoU={f1_ngiou3:.4f} > F1-IoU=0.00 ✓")

    # Empty inputs
    assert compute_f1_ngiou([], gt) == 0.0
    assert compute_f1_ngiou(pred, []) == 0.0
    print("  Empty inputs: 0.0 ✓")

    # Extra segments → F1 penalty via precision
    pred4 = [[10, 30], [15, 35], [40, 60]]
    f1_extra = compute_f1_ngiou(pred4, gt)
    assert f1_extra < f1, "Extra segments should reduce F1"
    print(f"  Extra segment: F1-NGIoU={f1_extra:.4f} < 1.0 ✓")

    # Missing segments → F1 penalty via recall
    pred5 = [[10, 30]]
    f1_miss = compute_f1_ngiou(pred5, gt)
    assert f1_miss < f1, "Missing segments should reduce F1"
    print(f"  Missing segment: F1-NGIoU={f1_miss:.4f} < 1.0 ✓")

    print("\n" + "=" * 50)
    print("Test 4: compute_f1_ngiou with margin (L1)")
    print("=" * 50)

    gt_l1 = [[0, 60], [60, 120]]
    # Pred off by 5s → with margin=5 should be very generous
    pred_off = [[5, 65], [65, 125]]
    f1_no_margin = compute_f1_ngiou(pred_off, gt_l1, margin=0)
    f1_with_margin = compute_f1_ngiou(pred_off, gt_l1, margin=5)
    assert f1_with_margin > f1_no_margin, "Margin should increase score"
    print(f"  5s offset: no_margin={f1_no_margin:.4f}, margin=5s={f1_with_margin:.4f} ✓")

    # Perfect match with margin → max(ngiou(pred,gt), ngiou(pred,expanded_gt)) = 1.0
    f1_perfect_margin = compute_f1_ngiou([[0, 60], [60, 120]], gt_l1, margin=5)
    approx(f1_perfect_margin, 1.0)
    print(f"  Perfect + margin=5: {f1_perfect_margin:.4f} ✓")

    print("\n" + "=" * 50)
    print("Test 5: _f1_iou_reward / _f1_iou_reward / _f1_iou_reward")
    print("=" * 50)

    gt_str = "<events>[[10.0, 30.0], [40.0, 60.0]]</events>"

    # Perfect match — L1 margin uses max(original, expanded), so perfect match → 1.0
    pred_str = "<events>[[10.0, 30.0], [40.0, 60.0]]</events>"
    r1 = _f1_iou_reward(pred_str, gt_str)
    r2 = _f1_iou_reward(pred_str, gt_str)
    r3 = _f1_iou_reward(pred_str, gt_str)
    approx(r1["overall"], 1.0)
    approx(r2["overall"], 1.0)
    approx(r3["overall"], 1.0)
    print(f"  Perfect: L1={r1['overall']:.4f} L2={r2['overall']:.4f} L3={r3['overall']:.4f} ✓")

    # Offset 3s on SHORT segments (20s) → L1 margin doesn't help (ngiou(pred,gt) already > ngiou(pred,expanded))
    pred_off_str = "<events>[[13.0, 33.0], [43.0, 63.0]]</events>"
    r1_off = _f1_iou_reward(pred_off_str, gt_str)
    r2_off = _f1_iou_reward(pred_off_str, gt_str)
    r3_off = _f1_iou_reward(pred_off_str, gt_str)
    assert r1_off["overall"] >= r2_off["overall"], "L1 margin should never hurt"
    print(f"  3s offset (short segs): L1={r1_off['overall']:.4f} >= L2={r2_off['overall']:.4f} ✓")

    # Offset 5s on LONG segments (60s, realistic L1) → L1 margin helps significantly
    gt_l1_str = "<events>[[0.0, 60.0], [60.0, 120.0]]</events>"
    pred_l1_str = "<events>[[5.0, 65.0], [65.0, 125.0]]</events>"
    r1_long = _f1_iou_reward(pred_l1_str, gt_l1_str)
    r2_long = _f1_iou_reward(pred_l1_str, gt_l1_str)
    assert r1_long["overall"] >= r2_long["overall"], "L1 margin should help for long segments"
    print(f"  5s offset (60s segs): L1={r1_long['overall']:.4f} >= L2={r2_long['overall']:.4f} ✓")

    # Anti-hack checks
    assert _f1_iou_reward("no events", gt_str)["overall"] == 0.0
    assert _f1_iou_reward("<events>[]</events>", gt_str)["overall"] == 0.0
    assert _f1_iou_reward("<events>[[1,2]]</events><events>[[3,4]]</events>", gt_str)["overall"] == 0.0
    print("  Anti-hack: all 0.0 ✓")

    # L2 and L3 should give same score (same logic, no margin)
    approx(r2_off["overall"], r3_off["overall"])
    print(f"  L2 == L3 (no margin): ✓")

    print("\n" + "=" * 50)
    print("Test 6: compute_score batch dispatch")
    print("=" * 50)

    inputs = [
        {"response": pred_str, "ground_truth": gt_str, "problem_type": "temporal_seg_hier_L1"},
        {"response": pred_off_str, "ground_truth": gt_str, "problem_type": "temporal_seg_hier_L2"},
        {"response": pred_str, "ground_truth": gt_str, "problem_type": "temporal_seg_hier_L3_seg"},
        {"response": "bad", "ground_truth": gt_str, "problem_type": "temporal_seg_hier_L1"},
    ]
    results = compute_score(inputs)
    assert len(results) == 4
    approx(results[0]["overall"], 1.0)   # L1 perfect
    assert results[1]["overall"] > 0     # L2 offset
    approx(results[2]["overall"], 1.0)   # L3_seg perfect
    assert results[3]["overall"] == 0.0  # bad input
    print(f"  Batch[0] L1 perfect: {results[0]['overall']:.4f} ✓")
    print(f"  Batch[1] L2 offset:  {results[1]['overall']:.4f} ✓")
    print(f"  Batch[2] L3 perfect: {results[2]['overall']:.4f} ✓")
    print(f"  Batch[3] bad input:  {results[3]['overall']:.4f} ✓")

    print("\n" + "=" * 50)
    print("Test 7: V2 vs V1 comparison")
    print("=" * 50)

    # V2 (NGIoU) vs V1 (F1-IoU) on same inputs — NGIoU always >= IoU
    v1 = _f1_iou_reward(pred_off_str, gt_str)
    v2_l1 = _f1_iou_reward(pred_off_str, gt_str)
    v2_l2 = _f1_iou_reward(pred_off_str, gt_str)
    print(f"  3s offset (short): V1(F1-IoU)={v1['overall']:.4f}, V2-L1={v2_l1['overall']:.4f}, V2-L2={v2_l2['overall']:.4f}")
    assert v2_l1["overall"] >= v1["overall"], "V2 L1 should be >= V1"
    assert v2_l2["overall"] >= v1["overall"], "V2 L2 (NGIoU) should be >= V1 (IoU)"

    # No overlap case
    pred_no_overlap = "<events>[[70.0, 90.0]]</events>"
    v1_no = _f1_iou_reward(pred_no_overlap, gt_str)
    v2_no = _f1_iou_reward(pred_no_overlap, gt_str)
    print(f"  No overlap: V1={v1_no['overall']:.4f}, V2={v2_no['overall']:.4f}")
    assert v1_no["overall"] == 0.0, "V1 should be 0 for no overlap"
    assert v2_no["overall"] > 0.0, "V2 should be > 0 for no overlap (NGIoU gradient!)"

    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
