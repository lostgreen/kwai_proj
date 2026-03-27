#!/usr/bin/env python3
"""
Chain-of-Segment Reward 单元测试。

测试:
    1. 解析器 (L2/L3 tags)
    2. aligned-IoU 计算
    3. L3 per-event F1-IoU
    4. 链式级联 reward
    5. 反作弊检查
    6. batch compute_score 接口
    7. 与真实生成数据的端到端验证
"""

import os
import sys
import importlib.util

# 直接加载模块，跳过 verl/__init__.py 的大量依赖
train_dir = os.path.join(os.path.dirname(__file__), "..")

# 先加载 youcook2_temporal_seg_reward (chain_seg 依赖它)
spec_tseg = importlib.util.spec_from_file_location(
    "verl.reward_function.youcook2_temporal_seg_reward",
    os.path.join(train_dir, "verl", "reward_function", "youcook2_temporal_seg_reward.py"),
)
mod_tseg = importlib.util.module_from_spec(spec_tseg)
sys.modules["verl.reward_function.youcook2_temporal_seg_reward"] = mod_tseg
spec_tseg.loader.exec_module(mod_tseg)

# 再加载 chain_seg_reward
spec_chain = importlib.util.spec_from_file_location(
    "verl.reward_function.youcook2_chain_seg_reward",
    os.path.join(train_dir, "verl", "reward_function", "youcook2_chain_seg_reward.py"),
)
mod_chain = importlib.util.module_from_spec(spec_chain)
sys.modules["verl.reward_function.youcook2_chain_seg_reward"] = mod_chain
spec_chain.loader.exec_module(mod_chain)

# 从加载的模块导入需要的符号
parse_l2_events = mod_chain.parse_l2_events
parse_l3_events = mod_chain.parse_l3_events
has_chain_tags = mod_chain.has_chain_tags
compute_aligned_iou = mod_chain.compute_aligned_iou
compute_l3_reward = mod_chain.compute_l3_reward
chain_reward = mod_chain.chain_reward
dual_seg_reward = mod_chain.dual_seg_reward
ground_seg_reward = mod_chain.ground_seg_reward
clip_l3_to_l2_bounds = mod_chain.clip_l3_to_l2_bounds
_l3_boundary_compliance = mod_chain._l3_boundary_compliance
_compute_l3_with_boundary = mod_chain._compute_l3_with_boundary
compute_score = mod_chain.compute_score
_anti_hack_check = mod_chain._anti_hack_check
W_L2 = mod_chain.W_L2
W_L3 = mod_chain.W_L3
_W_L2_DUAL = mod_chain._W_L2_DUAL
_W_L3_DUAL = mod_chain._W_L3_DUAL
BOUNDARY_PENALTY_ALPHA = mod_chain.BOUNDARY_PENALTY_ALPHA
CASCADE_FLOOR = mod_chain.CASCADE_FLOOR


def test_parse_l2_events():
    """L2 段解析"""
    text = "<l2_events>[[5, 30], [35, 55]]</l2_events>"
    segs = parse_l2_events(text)
    assert segs == [[5.0, 30.0], [35.0, 55.0]], f"Got {segs}"

    # 带空格
    text = "<l2_events>  [[ 5 , 30 ], [ 35, 55 ]]  </l2_events>"
    segs = parse_l2_events(text)
    assert segs == [[5.0, 30.0], [35.0, 55.0]]

    # 无标签
    assert parse_l2_events("no tags here") == []

    # 空标签
    assert parse_l2_events("<l2_events></l2_events>") == []

    # 非法段 (start >= end)
    text = "<l2_events>[[30, 5], [35, 55]]</l2_events>"
    segs = parse_l2_events(text)
    assert segs == [[35.0, 55.0]]

    print("  ✓ test_parse_l2_events")


def test_parse_l3_events():
    """L3 嵌套段解析"""
    text = "<l3_events>[[[5, 10], [12, 20], [22, 30]], [[35, 42], [45, 55]]]</l3_events>"
    segs = parse_l3_events(text)
    assert len(segs) == 2
    assert segs[0] == [[5.0, 10.0], [12.0, 20.0], [22.0, 30.0]]
    assert segs[1] == [[35.0, 42.0], [45.0, 55.0]]

    # 单事件
    text = "<l3_events>[[[1, 5], [7, 10]]]</l3_events>"
    segs = parse_l3_events(text)
    assert len(segs) == 1
    assert segs[0] == [[1.0, 5.0], [7.0, 10.0]]

    # 空列表
    assert parse_l3_events("<l3_events>[]</l3_events>") == []
    assert parse_l3_events("no tags") == []

    # 嵌套为空的子列表
    text = "<l3_events>[[], [[1, 5]]]</l3_events>"
    segs = parse_l3_events(text)
    assert len(segs) == 2
    assert segs[0] == []
    assert segs[1] == [[1.0, 5.0]]

    print("  ✓ test_parse_l3_events")


def test_has_chain_tags():
    """标签检测"""
    assert has_chain_tags("<l2_events>[[1,2]]</l2_events><l3_events>[[[1,2]]]</l3_events>")
    assert not has_chain_tags("<l2_events>[[1,2]]</l2_events>")  # 缺 l3
    assert not has_chain_tags("<l3_events>[[[1,2]]]</l3_events>")  # 缺 l2
    assert not has_chain_tags("no tags")
    print("  ✓ test_has_chain_tags")


def test_anti_hack():
    """反作弊检查"""
    assert _anti_hack_check("normal text <l2_events>[[1,2]]</l2_events>")
    assert not _anti_hack_check("text [10-20] bad")
    assert not _anti_hack_check("<l2_events>x</l2_events><l2_events>y</l2_events>")
    assert not _anti_hack_check("<l3_events>x</l3_events><l3_events>y</l3_events>")
    print("  ✓ test_anti_hack")


def test_aligned_iou_perfect():
    """完美匹配 aligned IoU"""
    pred = [[5, 30], [35, 55]]
    gt = [[5, 30], [35, 55]]
    iou = compute_aligned_iou(pred, gt)
    assert abs(iou - 1.0) < 1e-6, f"Expected 1.0, got {iou}"
    print("  ✓ test_aligned_iou_perfect")


def test_aligned_iou_partial():
    """部分匹配 aligned IoU"""
    pred = [[5, 25], [40, 60]]
    gt = [[5, 30], [35, 55]]
    iou = compute_aligned_iou(pred, gt)
    # pred[0] vs gt[0]: IoU(5-25, 5-30) = 20/25 = 0.8
    # pred[1] vs gt[1]: IoU(40-60, 35-55) = 15/25 = 0.6
    # mean = (0.8 + 0.6) / 2 = 0.7
    assert abs(iou - 0.7) < 1e-6, f"Expected 0.7, got {iou}"
    print("  ✓ test_aligned_iou_partial")


def test_aligned_iou_count_mismatch():
    """段数不匹配时的惩罚"""
    pred = [[5, 30]]  # 只预测了 1 段
    gt = [[5, 30], [35, 55]]  # GT 有 2 段
    iou = compute_aligned_iou(pred, gt)
    # IoU(5-30, 5-30) = 1.0, 缺失段贡献 0
    # mean = 1.0 / max(1, 2) = 0.5
    assert abs(iou - 0.5) < 1e-6, f"Expected 0.5, got {iou}"

    # 多预测
    pred = [[5, 30], [35, 55], [60, 80]]  # 预测了 3 段
    gt = [[5, 30], [35, 55]]  # GT 2 段
    iou = compute_aligned_iou(pred, gt)
    # IoU sum = 1.0 + 1.0 = 2.0
    # mean = 2.0 / max(3, 2) = 2/3
    assert abs(iou - 2.0 / 3) < 1e-6, f"Expected {2/3}, got {iou}"

    print("  ✓ test_aligned_iou_count_mismatch")


def test_l3_reward_perfect():
    """L3 per-event perfect match"""
    pred_l3 = [[[5, 10], [12, 20]], [[35, 42]]]
    gt_l3 = [[[5, 10], [12, 20]], [[35, 42]]]
    score = compute_l3_reward(pred_l3, gt_l3)
    assert abs(score - 1.0) < 1e-6, f"Expected 1.0, got {score}"
    print("  ✓ test_l3_reward_perfect")


def test_l3_reward_partial():
    """L3 per-event partial: one event perfect, one event absent"""
    pred_l3 = [[[5, 10], [12, 20]]]  # 只预测了第一个事件
    gt_l3 = [[[5, 10], [12, 20]], [[35, 42]]]
    score = compute_l3_reward(pred_l3, gt_l3)
    # event 0: F1-IoU = 1.0, event 1: missing = 0.0
    # mean = (1.0 + 0.0) / 2 = 0.5
    assert abs(score - 0.5) < 1e-6, f"Expected 0.5, got {score}"
    print("  ✓ test_l3_reward_partial")


def test_chain_reward_perfect():
    """完美预测的链式 reward"""
    gt = "<l2_events>[[5, 30], [35, 55]]</l2_events>\n<l3_events>[[[5, 10], [12, 20], [22, 30]], [[35, 42], [45, 55]]]</l3_events>"
    resp = "<l2_events>[[5, 30], [35, 55]]</l2_events>\n<l3_events>[[[5, 10], [12, 20], [22, 30]], [[35, 42], [45, 55]]]</l3_events>"
    result = chain_reward(resp, gt)
    expected = W_L2 * 1.0 + W_L3 * 1.0 * max(1.0, CASCADE_FLOOR)
    assert abs(result["overall"] - expected) < 1e-6, f"Expected {expected}, got {result['overall']}"
    assert abs(result["l2_reward"] - 1.0) < 1e-6
    assert abs(result["l3_reward"] - 1.0) < 1e-6
    print("  ✓ test_chain_reward_perfect")


def test_chain_reward_zero_format():
    """格式错误得 0 分"""
    gt = "<l2_events>[[5, 30]]</l2_events>\n<l3_events>[[[5, 10]]]</l3_events>"
    # 缺少 l3_events 标签
    resp = "<l2_events>[[5, 30]]</l2_events>"
    result = chain_reward(resp, gt)
    assert result["overall"] == 0.0
    print("  ✓ test_chain_reward_zero_format")


def test_chain_reward_cascade():
    """级联因子: L2 差 → L3 被降权 + 边界裁剪"""
    gt = "<l2_events>[[5, 30], [35, 55]]</l2_events>\n<l3_events>[[[5, 10], [12, 20]], [[35, 42], [45, 55]]]</l3_events>"
    # L2 完全错误, L3 完美（但 L3 在 pred L2 [0,1][2,3] 外 → 被裁剪为空）
    resp = "<l2_events>[[0, 1], [2, 3]]</l2_events>\n<l3_events>[[[5, 10], [12, 20]], [[35, 42], [45, 55]]]</l3_events>"
    result = chain_reward(resp, gt)
    # L2 aligned IoU ≈ 0 (no overlap)
    # L3 被 clip 到 pred L2 [0,1]/[2,3] → 全部丢弃 → r_l3 = 0
    # overall ≈ 0
    assert result["l2_reward"] < 0.01
    assert result["l3_reward"] == 0.0  # 边界裁剪后全部落在 L2 外
    assert result["overall"] < 0.01
    print("  ✓ test_chain_reward_cascade")


def test_compute_score_batch():
    """batch 接口测试"""
    gt = "<l2_events>[[5, 30], [35, 55]]</l2_events>\n<l3_events>[[[5, 10], [12, 20]], [[35, 42]]]</l3_events>"
    inputs = [
        {
            "response": gt,
            "ground_truth": gt,
            "problem_type": "temporal_seg_chain_L2L3",
        },
        {
            "response": "garbage",
            "ground_truth": gt,
            "problem_type": "temporal_seg_chain_L2L3",
        },
    ]
    results = compute_score(inputs)
    assert len(results) == 2
    assert results[0]["overall"] > 0.9
    assert results[1]["overall"] == 0.0
    print("  ✓ test_compute_score_batch")


def test_with_real_data():
    """用真实生成的数据验证端到端"""
    # 模拟 prepare_chain_seg_data.py 的输出格式
    gt = "<l2_events>[[26, 41], [79, 111]]</l2_events>\n<l3_events>[[[28, 29], [31, 34], [35, 36], [38, 39], [40, 41]], [[79, 85], [86, 89], [91, 93], [95, 110]]]</l3_events>"

    # 模型输出: L2 稍有偏移, L3 部分正确
    resp = "<l2_events>[[25, 42], [78, 112]]</l2_events>\n<l3_events>[[[27, 30], [31, 35], [36, 38], [39, 41]], [[80, 86], [87, 90], [92, 94], [96, 109]]]</l3_events>"

    result = chain_reward(resp, gt)
    assert result["overall"] > 0.0
    assert result["l2_reward"] > 0.5  # L2 基本对
    assert result["l3_reward"] > 0.3  # L3 部分对
    print(f"  ✓ test_with_real_data: overall={result['overall']:.3f}, l2={result['l2_reward']:.3f}, l3={result['l3_reward']:.3f}")


# ===========================
# clip_l3_to_l2_bounds 测试
# ===========================
def test_clip_l3_to_l2_bounds():
    """L3 边界裁剪"""
    # 完全内部 — 无变化
    l2 = [[10, 50]]
    l3 = [[[15, 20], [25, 30], [35, 45]]]
    clipped = clip_l3_to_l2_bounds(l2, l3)
    assert clipped == [[[15, 20], [25, 30], [35, 45]]]

    # 部分越界 — 裁剪
    l2 = [[10, 40]]
    l3 = [[[5, 20], [30, 50]]]  # [5,20] 左越界, [30,50] 右越界
    clipped = clip_l3_to_l2_bounds(l2, l3)
    assert clipped == [[[10, 20], [30, 40]]]

    # 完全外部 — 丢弃
    l2 = [[20, 30]]
    l3 = [[[5, 10], [40, 50]]]  # 全在 L2 外面
    clipped = clip_l3_to_l2_bounds(l2, l3)
    assert clipped == [[]]

    # 多事件
    l2 = [[10, 30], [40, 60]]
    l3 = [[[8, 25], [28, 35]], [[38, 55], [58, 65]]]
    clipped = clip_l3_to_l2_bounds(l2, l3)
    assert clipped == [[[10, 25], [28, 30]], [[40, 55], [58, 60]]]

    # L3 比 L2 多 — 只处理有对应 L2 的部分
    l2 = [[10, 30]]
    l3 = [[[15, 20]], [[40, 50]]]  # 第二组没有对应 L2
    clipped = clip_l3_to_l2_bounds(l2, l3)
    assert len(clipped) == 1
    assert clipped == [[[15, 20]]]

    print("  ✓ test_clip_l3_to_l2_bounds")


# ===========================
# _l3_boundary_compliance 测试
# ===========================
def test_boundary_compliance_full():
    """完全在 L2 范围内 → compliance = 1.0"""
    segs = [[15.0, 20.0], [25.0, 30.0]]
    l2 = [10.0, 40.0]
    c = _l3_boundary_compliance(segs, l2)
    assert abs(c - 1.0) < 1e-6, f"Expected 1.0, got {c}"
    print("  ✓ test_boundary_compliance_full")


def test_boundary_compliance_partial():
    """一半在 L2 范围外 → compliance ≈ 0.5"""
    # L2=[10,20], L3=[[5,15], [20,30]]
    # [5,15]: 原始10s, 在界内5s (10-15)
    # [20,30]: 原始10s, 在界内0s (20是右边界)
    # total_orig=20, in_bounds=5 → 5/20=0.25
    segs = [[5.0, 15.0], [20.0, 30.0]]
    l2 = [10.0, 20.0]
    c = _l3_boundary_compliance(segs, l2)
    assert abs(c - 0.25) < 1e-6, f"Expected 0.25, got {c}"
    print("  ✓ test_boundary_compliance_partial")


def test_boundary_compliance_outside():
    """完全在 L2 范围外 → compliance = 0.0"""
    segs = [[50.0, 60.0], [70.0, 80.0]]
    l2 = [10.0, 40.0]
    c = _l3_boundary_compliance(segs, l2)
    assert abs(c - 0.0) < 1e-6, f"Expected 0.0, got {c}"
    print("  ✓ test_boundary_compliance_outside")


def test_boundary_compliance_empty():
    """空 L3 列表 → compliance = 1.0 (无惩罚)"""
    c = _l3_boundary_compliance([], [10.0, 40.0])
    assert abs(c - 1.0) < 1e-6
    print("  ✓ test_boundary_compliance_empty")


def test_compute_l3_with_boundary_perfect():
    """完全在 L2 内的完美 L3 → compliance=1.0, score=1.0"""
    pred_l3 = [[[5, 10], [12, 20]], [[35, 42], [45, 55]]]
    gt_l3 = [[[5, 10], [12, 20]], [[35, 42], [45, 55]]]
    pred_l2 = [[0, 30], [30, 60]]   # 完全包含 L3
    score = _compute_l3_with_boundary(pred_l3, gt_l3, pred_l2)
    assert abs(score - 1.0) < 1e-6, f"Expected 1.0, got {score}"
    print("  ✓ test_compute_l3_with_boundary_perfect")


def test_compute_l3_with_boundary_penalty():
    """L3 部分越界 → 软惩罚 penalty_factor = 1 - ALPHA*(1-compliance)"""
    gt_l3 = [[[10, 20], [25, 35]]]
    # pred: [5,15] 左越界, [20,40] 右越界; L2=[10,30]
    # compliance = 15/30 = 0.5, penalty_factor = 1 - 0.3*0.5 = 0.85
    pred_l3 = [[[5, 15], [20, 40]]]
    pred_l2 = [[10, 30]]

    score = _compute_l3_with_boundary(pred_l3, gt_l3, pred_l2)
    assert score > 0.0
    # penalty_factor = 0.85, so score < 0.85
    assert score < 0.85
    print(f"  ✓ test_compute_l3_with_boundary_penalty: score={score:.3f} (penalty_factor=0.85)")


# ===========================
# V1 dual_seg_reward 测试
# ===========================
def test_dual_seg_reward_perfect():
    """V1: 完美匹配 (1:1 等权)"""
    gt = "<l2_events>[[5, 30], [35, 55]]</l2_events>\n<l3_events>[[[5, 10], [12, 20]], [[35, 42], [45, 55]]]</l3_events>"
    resp = gt  # 完美预测
    result = dual_seg_reward(resp, gt)
    # V1 用 _W_L2_DUAL=0.5, _W_L3_DUAL=0.5; compliance=1.0 → penalty_factor=1.0
    expected = _W_L2_DUAL * 1.0 + _W_L3_DUAL * 1.0 * max(1.0, CASCADE_FLOOR)
    assert abs(result["overall"] - expected) < 1e-6, f"Expected {expected}, got {result['overall']}"
    assert abs(result["l2_reward"] - 1.0) < 1e-6
    assert abs(result["l3_reward"] - 1.0) < 1e-6
    print("  ✓ test_dual_seg_reward_perfect")


def test_dual_seg_reward_reorder():
    """V1: L2 乱序但 Hungarian 能匹配"""
    gt = "<l2_events>[[5, 30], [35, 55]]</l2_events>\n<l3_events>[[[5, 10], [12, 20]], [[35, 42], [45, 55]]]</l3_events>"
    # 预测顺序反了
    resp = "<l2_events>[[35, 55], [5, 30]]</l2_events>\n<l3_events>[[[35, 42], [45, 55]], [[5, 10], [12, 20]]]</l3_events>"
    result = dual_seg_reward(resp, gt)
    # Hungarian 应该能正确匹配: pred[0]↔gt[1], pred[1]↔gt[0]
    assert result["l2_reward"] > 0.9, f"L2 should be high, got {result['l2_reward']}"
    assert result["l3_reward"] > 0.9, f"L3 should be high, got {result['l3_reward']}"
    print(f"  ✓ test_dual_seg_reward_reorder: overall={result['overall']:.3f}")


def test_dual_seg_reward_zero_format():
    """V1: 格式错误得 0"""
    gt = "<l2_events>[[5, 30]]</l2_events>\n<l3_events>[[[5, 10]]]</l3_events>"
    resp = "no tags at all"
    result = dual_seg_reward(resp, gt)
    assert result["overall"] == 0.0
    print("  ✓ test_dual_seg_reward_zero_format")


# ===========================
# V2 ground_seg_reward 测试
# ===========================
def test_ground_seg_reward_perfect():
    """V2: 完美单事件匹配"""
    gt = "<l2_events>[[10, 45]]</l2_events>\n<l3_events>[[[12, 18], [20, 30], [35, 43]]]</l3_events>"
    resp = gt
    result = ground_seg_reward(resp, gt)
    expected = W_L2 * 1.0 + W_L3 * 1.0 * max(1.0, CASCADE_FLOOR)
    assert abs(result["overall"] - expected) < 1e-6
    assert abs(result["l2_reward"] - 1.0) < 1e-6
    assert abs(result["l3_reward"] - 1.0) < 1e-6
    print("  ✓ test_ground_seg_reward_perfect")


def test_ground_seg_reward_l3_boundary():
    """V2: L3 越界被裁剪后评估"""
    gt = "<l2_events>[[10, 40]]</l2_events>\n<l3_events>[[[12, 20], [25, 35]]]</l3_events>"
    # L3 预测越界: [8,22] 左越界, [30,50] 右越界
    resp = "<l2_events>[[10, 40]]</l2_events>\n<l3_events>[[[8, 22], [30, 50]]]</l3_events>"
    result = ground_seg_reward(resp, gt)
    # L2 完美匹配, L3 裁剪后变为 [[10,22],[30,40]]
    assert abs(result["l2_reward"] - 1.0) < 1e-6
    # L3 reward > 0 (裁剪后有部分匹配)
    assert result["l3_reward"] > 0.0
    assert result["l3_reward"] < 1.0  # 不是完美匹配
    print(f"  ✓ test_ground_seg_reward_l3_boundary: l3={result['l3_reward']:.3f}")


def test_ground_seg_reward_cascade():
    """V2: L2 差时级联降权"""
    gt = "<l2_events>[[10, 40]]</l2_events>\n<l3_events>[[[12, 20], [25, 35]]]</l3_events>"
    # L2 完全错误
    resp = "<l2_events>[[80, 100]]</l2_events>\n<l3_events>[[[12, 20], [25, 35]]]</l3_events>"
    result = ground_seg_reward(resp, gt)
    assert result["l2_reward"] < 0.01
    # L3 被 clip 到 pred L2 [80,100] 范围 → [12,20] 和 [25,35] 全部在范围外 → 被丢弃
    assert result["l3_reward"] == 0.0
    assert result["overall"] < 0.01
    print("  ✓ test_ground_seg_reward_cascade")


def test_compute_score_dispatch():
    """batch 接口 dispatch 到 V1/V2"""
    gt = "<l2_events>[[5, 30]]</l2_events>\n<l3_events>[[[5, 10], [15, 25]]]</l3_events>"
    inputs = [
        {"response": gt, "ground_truth": gt, "problem_type": "temporal_seg_chain_dual_seg"},
        {"response": gt, "ground_truth": gt, "problem_type": "temporal_seg_chain_ground_seg"},
        {"response": "garbage", "ground_truth": gt, "problem_type": "temporal_seg_chain_dual_seg"},
    ]
    results = compute_score(inputs)
    assert len(results) == 3
    assert results[0]["overall"] > 0.9  # V1 perfect
    assert results[1]["overall"] > 0.9  # V2 perfect
    assert results[2]["overall"] == 0.0  # garbage
    print("  ✓ test_compute_score_dispatch")


def main():
    print("Running Chain-of-Segment Reward tests...\n")

    # 原有测试
    test_parse_l2_events()
    test_parse_l3_events()
    test_has_chain_tags()
    test_anti_hack()
    test_aligned_iou_perfect()
    test_aligned_iou_partial()
    test_aligned_iou_count_mismatch()
    test_l3_reward_perfect()
    test_l3_reward_partial()
    test_chain_reward_perfect()
    test_chain_reward_zero_format()
    test_chain_reward_cascade()
    test_compute_score_batch()
    test_with_real_data()

    # 边界裁剪
    test_clip_l3_to_l2_bounds()

    # 边界合规惩罚
    test_boundary_compliance_full()
    test_boundary_compliance_partial()
    test_boundary_compliance_outside()
    test_boundary_compliance_empty()
    test_compute_l3_with_boundary_perfect()
    test_compute_l3_with_boundary_penalty()

    # V1 dual-seg reward
    test_dual_seg_reward_perfect()
    test_dual_seg_reward_reorder()
    test_dual_seg_reward_zero_format()

    # V2 ground-seg reward
    test_ground_seg_reward_perfect()
    test_ground_seg_reward_l3_boundary()
    test_ground_seg_reward_cascade()

    # dispatch
    test_compute_score_dispatch()

    print("\n✅ All 30 tests passed!")


if __name__ == "__main__":
    main()
