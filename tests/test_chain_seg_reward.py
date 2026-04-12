#!/usr/bin/env python3
"""
Chain-of-Segment Reward 单元测试 — 仅 V2 (ground-seg)。

测试:
    1. 解析器 (L2/L3 tags)
    2. 反作弊检查
    3. L3 边界裁剪
    4. L3 越界砍半惩罚
    5. ground_seg_reward 端到端
    6. batch compute_score 接口
"""

import os
import sys
import importlib.util

# 直接加载模块，跳过 verl/__init__.py 的大量依赖
train_dir = os.path.join(os.path.dirname(__file__), "..")

# 先加载 reward_utils (chain_seg 依赖它)
spec_tseg = importlib.util.spec_from_file_location(
    "verl.reward_function.reward_utils",
    os.path.join(train_dir, "verl", "reward_function", "reward_utils.py"),
)
mod_tseg = importlib.util.module_from_spec(spec_tseg)
sys.modules["verl.reward_function.reward_utils"] = mod_tseg
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
ground_seg_reward = mod_chain.ground_seg_reward
clip_l3_to_l2_bounds = mod_chain.clip_l3_to_l2_bounds
_has_oob = mod_chain._has_oob
compute_score = mod_chain.compute_score
_anti_hack_check = mod_chain._anti_hack_check
W_L2 = mod_chain.W_L2
W_L3 = mod_chain.W_L3
OOB_PENALTY = mod_chain.OOB_PENALTY


# ===========================
# 解析器测试
# ===========================
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


# ===========================
# L3 边界裁剪测试
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

    print("  ✓ test_clip_l3_to_l2_bounds")


# ===========================
# _has_oob 测试
# ===========================
def test_has_oob():
    """越界检测"""
    # 全在界内
    assert not _has_oob([[15, 20], [25, 30]], [10, 40])

    # 左越界
    assert _has_oob([[5, 20], [25, 30]], [10, 40])

    # 右越界
    assert _has_oob([[15, 20], [25, 50]], [10, 40])

    # 空列表 — 不越界
    assert not _has_oob([], [10, 40])

    # 恰好贴边 — 不越界
    assert not _has_oob([[10, 20], [30, 40]], [10, 40])

    print("  ✓ test_has_oob")


# ===========================
# ground_seg_reward 测试
# ===========================
def test_ground_seg_reward_perfect():
    """V2: 完美单事件匹配，无越界"""
    gt = "<l2_events>[[10, 45]]</l2_events>\n<l3_events>[[[12, 18], [20, 30], [35, 43]]]</l3_events>"
    resp = gt
    result = ground_seg_reward(resp, gt)
    expected = W_L2 * 1.0 + W_L3 * 1.0  # 无级联, 无越界
    assert abs(result["overall"] - expected) < 1e-6, f"Expected {expected}, got {result['overall']}"
    assert abs(result["l2_reward"] - 1.0) < 1e-6
    assert abs(result["l3_reward"] - 1.0) < 1e-6
    print("  ✓ test_ground_seg_reward_perfect")


def test_ground_seg_reward_l3_oob_penalty():
    """V2: L3 越界 → reward 砍半"""
    gt = "<l2_events>[[10, 40]]</l2_events>\n<l3_events>[[[12, 20], [25, 35]]]</l3_events>"
    # L3 预测越界: [8,22] 左越界, [30,50] 右越界
    resp = "<l2_events>[[10, 40]]</l2_events>\n<l3_events>[[[8, 22], [30, 50]]]</l3_events>"
    result = ground_seg_reward(resp, gt)
    # L2 完美匹配
    assert abs(result["l2_reward"] - 1.0) < 1e-6
    # L3 有越界 → 砍半；裁剪后 [[10,22],[30,40]] vs GT [[12,20],[25,35]]
    assert result["l3_reward"] > 0.0
    assert result["l3_reward"] < 0.5  # 砍半后一定 < 0.5 (裁剪后 F1-IoU < 1.0)
    print(f"  ✓ test_ground_seg_reward_l3_oob_penalty: l3={result['l3_reward']:.3f}")


def test_ground_seg_reward_l3_no_oob():
    """V2: L3 在界内 → 不砍半"""
    gt = "<l2_events>[[10, 40]]</l2_events>\n<l3_events>[[[12, 20], [25, 35]]]</l3_events>"
    # L3 预测完全在 L2 [10,40] 内
    resp = "<l2_events>[[10, 40]]</l2_events>\n<l3_events>[[[13, 19], [26, 34]]]</l3_events>"
    result = ground_seg_reward(resp, gt)
    assert abs(result["l2_reward"] - 1.0) < 1e-6
    # 不砍半, 只是 F1-IoU < 1.0 (略有偏移)
    assert result["l3_reward"] > 0.5
    print(f"  ✓ test_ground_seg_reward_l3_no_oob: l3={result['l3_reward']:.3f}")


def test_ground_seg_reward_l2_wrong():
    """V2: L2 完全错误，无级联 floor → overall 很低"""
    gt = "<l2_events>[[10, 40]]</l2_events>\n<l3_events>[[[12, 20], [25, 35]]]</l3_events>"
    # L2 完全错误
    resp = "<l2_events>[[80, 100]]</l2_events>\n<l3_events>[[[12, 20], [25, 35]]]</l3_events>"
    result = ground_seg_reward(resp, gt)
    assert result["l2_reward"] < 0.01
    # L3 被 clip 到 pred L2 [80,100] → [12,20] 和 [25,35] 全在范围外 → 被丢弃
    assert result["l3_reward"] == 0.0
    # 无级联 floor → overall ≈ 0
    assert result["overall"] < 0.01
    print("  ✓ test_ground_seg_reward_l2_wrong")


def test_ground_seg_reward_zero_format():
    """V2: 格式错误得 0 分"""
    gt = "<l2_events>[[10, 40]]</l2_events>\n<l3_events>[[[12, 20]]]</l3_events>"
    resp = "no tags at all"
    result = ground_seg_reward(resp, gt)
    assert result["overall"] == 0.0
    print("  ✓ test_ground_seg_reward_zero_format")


def test_ground_seg_reward_with_real_data():
    """用模拟真实数据验证端到端"""
    gt = "<l2_events>[[26, 41]]</l2_events>\n<l3_events>[[[28, 29], [31, 34], [35, 36], [38, 39], [40, 41]]]</l3_events>"
    # 模型输出: L2 稍有偏移, L3 部分正确
    resp = "<l2_events>[[25, 42]]</l2_events>\n<l3_events>[[[27, 30], [31, 35], [36, 38], [39, 41]]]</l3_events>"
    result = ground_seg_reward(resp, gt)
    assert result["overall"] > 0.0
    assert result["l2_reward"] > 0.8  # L2 基本对
    assert result["l3_reward"] > 0.2  # L3 部分对
    print(f"  ✓ test_ground_seg_reward_with_real_data: overall={result['overall']:.3f}, l2={result['l2_reward']:.3f}, l3={result['l3_reward']:.3f}")


# ===========================
# batch compute_score 测试
# ===========================
def test_compute_score_dispatch():
    """batch 接口 dispatch"""
    gt = "<l2_events>[[5, 30]]</l2_events>\n<l3_events>[[[5, 10], [15, 25]]]</l3_events>"
    inputs = [
        {"response": gt, "ground_truth": gt, "problem_type": "temporal_seg_chain_ground_seg"},
        {"response": "garbage", "ground_truth": gt, "problem_type": "temporal_seg_chain_ground_seg"},
    ]
    results = compute_score(inputs)
    assert len(results) == 2
    assert results[0]["overall"] > 0.9  # V2 perfect
    assert results[1]["overall"] == 0.0  # garbage
    print("  ✓ test_compute_score_dispatch")


def test_compute_score_default_dispatch():
    """未知 problem_type 默认走 ground_seg_reward"""
    gt = "<l2_events>[[5, 30]]</l2_events>\n<l3_events>[[[5, 10], [15, 25]]]</l3_events>"
    inputs = [
        {"response": gt, "ground_truth": gt, "problem_type": "unknown_type"},
    ]
    results = compute_score(inputs)
    assert results[0]["overall"] > 0.9
    print("  ✓ test_compute_score_default_dispatch")


def main():
    print("Running Chain-of-Segment Reward tests (V2 only)...\n")

    # 解析器
    test_parse_l2_events()
    test_parse_l3_events()
    test_has_chain_tags()
    test_anti_hack()

    # 边界裁剪
    test_clip_l3_to_l2_bounds()

    # 越界检测
    test_has_oob()

    # ground-seg reward
    test_ground_seg_reward_perfect()
    test_ground_seg_reward_l3_oob_penalty()
    test_ground_seg_reward_l3_no_oob()
    test_ground_seg_reward_l2_wrong()
    test_ground_seg_reward_zero_format()
    test_ground_seg_reward_with_real_data()

    # batch dispatch
    test_compute_score_dispatch()
    test_compute_score_default_dispatch()

    print("\n✅ All 14 tests passed!")


if __name__ == "__main__":
    main()
