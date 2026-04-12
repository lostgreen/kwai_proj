#!/usr/bin/env python3
"""Quick test for boundary reward - standalone, no verl init dependency."""
import sys, importlib.util, types

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

# Load boundary reward
spec_bnd = importlib.util.spec_from_file_location(
    "verl.reward_function.youcook2_hier_seg_reward_boundary",
    "verl/reward_function/youcook2_hier_seg_reward_boundary.py",
)
mod_bnd = importlib.util.module_from_spec(spec_bnd)
sys.modules["verl.reward_function.youcook2_hier_seg_reward_boundary"] = mod_bnd
spec_bnd.loader.exec_module(mod_bnd)

# Load baseline reward
spec_bl = importlib.util.spec_from_file_location(
    "verl.reward_function.hier_seg_reward",
    "verl/reward_function/hier_seg_reward.py",
)
mod_bl = importlib.util.module_from_spec(spec_bl)
sys.modules["verl.reward_function.hier_seg_reward"] = mod_bl
spec_bl.loader.exec_module(mod_bl)

# Import functions
from verl.reward_function.youcook2_hier_seg_reward_boundary import (
    _boundary_hit_f1, _boundary_reward, _count_accuracy, _coverage_iou, compute_score,
)
from verl.reward_function.hier_seg_reward import _f1_iou_reward as f1_iou_reward

gt = "<events>[[10.0, 20.0], [30.0, 45.0]]</events>"

# 1. Perfect match
pred = "<events>[[10.0, 20.0], [30.0, 45.0]]</events>"
r_b = _boundary_reward(pred, gt)
r_f = f1_iou_reward(pred, gt)
print(f"[Perfect] Boundary: {r_b['overall']:.4f}  F1-IoU: {r_f['overall']:.4f}")
assert r_b["overall"] > 0.95
assert r_f["overall"] > 0.95

# 2. Slight offset
pred2 = "<events>[[11.0, 21.0], [31.0, 46.0]]</events>"
r_b2 = _boundary_reward(pred2, gt)
r_f2 = f1_iou_reward(pred2, gt)
print(f"[Offset 1s] Boundary: {r_b2['overall']:.4f} (bf1={r_b2['boundary_f1']:.4f})  F1-IoU: {r_f2['overall']:.4f}")

# 3. Wrong count
pred3 = "<events>[[10.0, 20.0], [30.0, 45.0], [50.0, 60.0], [65.0, 70.0], [75.0, 80.0]]</events>"
r_b3 = _boundary_reward(pred3, gt)
print(f"[Wrong count] count_acc={r_b3['count_accuracy']:.4f}  overall={r_b3['overall']:.4f}")
assert r_b3["count_accuracy"] < 0.5

# 4. No overlap
pred4 = "<events>[[50.0, 60.0], [70.0, 80.0]]</events>"
r_b4 = _boundary_reward(pred4, gt)
r_f4 = f1_iou_reward(pred4, gt)
print(f"[No overlap] Boundary: {r_b4['overall']:.4f}  F1-IoU: {r_f4['overall']:.4f}")

# 5. Anti-hack
assert _boundary_reward("no events", gt)["overall"] == 0.0
assert _boundary_reward("<events>[]</events>", gt)["overall"] == 0.0
assert _boundary_reward("<events>[[1,2]]</events><events>[[3,4]]</events>", gt)["overall"] == 0.0

# 6. Batch interface
inputs = [
    {"response": pred, "ground_truth": gt, "problem_type": "temporal_seg_hier_L2"},
    {"response": pred2, "ground_truth": gt, "problem_type": "temporal_seg_hier_L1"},
]
results = compute_score(inputs)
assert len(results) == 2
print(f"[Batch] R1: {results[0]['overall']:.4f}  R2: {results[1]['overall']:.4f}")

# 7. Component functions
assert _boundary_hit_f1([[10, 20], [30, 45]], [[10, 20], [30, 45]]) > 0.99
assert _count_accuracy(2, 2) > 0.99
assert _coverage_iou([[10, 20], [30, 45]], [[10, 20], [30, 45]]) > 0.99

print("\nAll tests passed!")
