#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Boundary-Aware Reward 单元测试。

验证新 reward 函数在各种场景下的行为符合预期，
并与 F1-IoU baseline 做对比分析。
"""
import sys
sys.path.insert(0, ".")

from verl.reward_function.youcook2_hier_seg_reward_boundary import (
    _boundary_hit_f1,
    _boundary_reward,
    _count_accuracy,
    _coverage_iou,
    compute_score,
)
from verl.reward_function.youcook2_hier_seg_reward import (
    _l1_l2_reward as f1_iou_reward,
)


def test_perfect_match():
    """完美匹配: 两种 reward 都应接近 1.0"""
    gt = "<events>[[10.0, 20.0], [30.0, 45.0]]</events>"
    pred = "<events>[[10.0, 20.0], [30.0, 45.0]]</events>"

    r_boundary = _boundary_reward(pred, gt)
    r_f1iou = f1_iou_reward(pred, gt)

    print(f"[Perfect] Boundary: {r_boundary['overall']:.4f}  F1-IoU: {r_f1iou['overall']:.4f}")
    assert r_boundary["overall"] > 0.95, f"Boundary reward too low: {r_boundary}"
    assert r_f1iou["overall"] > 0.95, f"F1-IoU reward too low: {r_f1iou}"


def test_slight_offset():
    """边界微偏: Boundary reward 有 τ 容忍, F1-IoU 则按 IoU 下降"""
    gt = "<events>[[10.0, 20.0], [30.0, 45.0]]</events>"
    pred = "<events>[[11.0, 21.0], [31.0, 46.0]]</events>"  # 偏移 1s

    r_boundary = _boundary_reward(pred, gt)
    r_f1iou = f1_iou_reward(pred, gt)

    print(f"[Offset 1s] Boundary: {r_boundary['overall']:.4f}  F1-IoU: {r_f1iou['overall']:.4f}")
    # 边界偏移 1s 在 τ=3s 内，boundary F1 应为 1.0
    assert r_boundary["boundary_f1"] > 0.99, f"Boundary F1 should be 1.0 within τ: {r_boundary}"
    assert r_boundary["overall"] > r_f1iou["overall"], (
        f"Boundary reward should be more forgiving for small offsets"
    )


def test_wrong_count():
    """段数错误: Count accuracy 应显著惩罚"""
    gt = "<events>[[10.0, 20.0], [30.0, 45.0]]</events>"
    pred = "<events>[[10.0, 20.0], [30.0, 45.0], [50.0, 60.0], [65.0, 70.0], [75.0, 80.0]]</events>"

    r_boundary = _boundary_reward(pred, gt)
    print(f"[Wrong count] count_accuracy: {r_boundary['count_accuracy']:.4f}  overall: {r_boundary['overall']:.4f}")
    assert r_boundary["count_accuracy"] < 0.5, f"Count should penalize 5 vs 2: {r_boundary}"


def test_no_overlap():
    """零重叠: 两种 reward 都应接近 0"""
    gt = "<events>[[10.0, 20.0], [30.0, 40.0]]</events>"
    pred = "<events>[[50.0, 60.0], [70.0, 80.0]]</events>"

    r_boundary = _boundary_reward(pred, gt)
    r_f1iou = f1_iou_reward(pred, gt)

    print(f"[No overlap] Boundary: {r_boundary['overall']:.4f}  F1-IoU: {r_f1iou['overall']:.4f}")
    assert r_boundary["coverage_iou"] < 0.01
    assert r_f1iou["overall"] < 0.01


def test_empty_and_hack():
    """空输出和反作弊: 都应返回 0"""
    gt = "<events>[[10.0, 20.0]]</events>"

    # 无 events tag
    assert _boundary_reward("no events", gt)["overall"] == 0.0
    # 空 events
    assert _boundary_reward("<events>[]</events>", gt)["overall"] == 0.0
    # 多重 tag (hack)
    assert _boundary_reward("<events>[[1,2]]</events><events>[[3,4]]</events>", gt)["overall"] == 0.0
    # 畸形格式
    assert _boundary_reward("<events>[10-20]</events>", gt)["overall"] == 0.0


def test_batch_interface():
    """Batch 接口兼容性测试"""
    inputs = [
        {
            "response": "<events>[[10.0, 20.0], [30.0, 45.0]]</events>",
            "ground_truth": "<events>[[10.0, 20.0], [30.0, 45.0]]</events>",
            "problem_type": "temporal_seg_hier_L2",
        },
        {
            "response": "<events>[[5.0, 15.0]]</events>",
            "ground_truth": "<events>[[10.0, 20.0]]</events>",
            "problem_type": "temporal_seg_hier_L1",
        },
    ]
    results = compute_score(inputs)
    assert len(results) == 2
    assert all("overall" in r for r in results)
    print(f"[Batch] R1: {results[0]['overall']:.4f}  R2: {results[1]['overall']:.4f}")


def test_component_functions():
    """独立测试三个组件"""
    pred = [[10.0, 20.0], [30.0, 45.0]]
    gt = [[10.0, 20.0], [30.0, 45.0]]

    assert _boundary_hit_f1(pred, gt) > 0.99
    assert _count_accuracy(2, 2) > 0.99
    assert _coverage_iou(pred, gt) > 0.99

    # 偏移测试
    pred_offset = [[12.0, 22.0], [32.0, 47.0]]
    assert _boundary_hit_f1(pred_offset, gt, tau=3.0) > 0.99  # 偏移 2s < τ=3
    assert _boundary_hit_f1(pred_offset, gt, tau=1.0) < 0.01  # 偏移 2s > τ=1

    # 覆盖率
    pred_partial = [[10.0, 15.0]]  # 只覆盖一半
    assert 0.1 < _coverage_iou(pred_partial, gt) < 0.5


if __name__ == "__main__":
    test_perfect_match()
    test_slight_offset()
    test_wrong_count()
    test_no_overlap()
    test_empty_and_hack()
    test_batch_interface()
    test_component_functions()
    print("\n✅ All tests passed!")
