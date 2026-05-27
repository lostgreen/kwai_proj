from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parents[1] / "verl" / "reward_function" / "reward_utils.py"
_SPEC = importlib.util.spec_from_file_location("reward_utils_under_test", _MODULE_PATH)
reward_utils = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = reward_utils
_SPEC.loader.exec_module(reward_utils)


def _load_reward_module(module_name: str):
    package = sys.modules.setdefault("verl.reward_function", type(sys)("verl.reward_function"))
    setattr(package, "reward_utils", reward_utils)
    sys.modules["verl.reward_function.reward_utils"] = reward_utils
    module_path = _MODULE_PATH.parent / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(f"{module_name}_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_segments_accepts_answer_wrapped_timestamps():
    assert reward_utils.parse_segments("<answer>[[0, 10], [10, 20]]</answer>") == [
        [0.0, 10.0],
        [10.0, 20.0],
    ]


def test_parse_segments_keeps_legacy_events_compatibility():
    assert reward_utils.parse_segments("<events>[[0, 10], [10, 20]]</events>") == [
        [0.0, 10.0],
        [10.0, 20.0],
    ]


def test_has_segment_answer_tag_accepts_new_and_legacy_containers():
    assert reward_utils.has_segment_answer_tag("<answer>[[0, 10]]</answer>")
    assert reward_utils.has_segment_answer_tag("<events>[[0, 10]]</events>")
    assert not reward_utils.has_segment_answer_tag("[[0, 10]]")


def test_seg_rewards_accept_answer_wrapped_timestamps():
    gt = "<events>[[0, 10], [10, 20]]</events>"
    pred = "<answer>[[0, 10], [10, 20]]</answer>"

    hier_seg_reward = _load_reward_module("hier_seg_reward")
    seg_match_reward = _load_reward_module("seg_match_reward")
    dp_f1_reward = _load_reward_module("dp_f1_reward")

    assert hier_seg_reward._f1_iou_reward(pred, gt)["overall"] > 0.99
    assert seg_match_reward._seg_match_reward(pred, gt)["overall"] > 0.99
    assert dp_f1_reward._dp_f1_reward(pred, gt)["overall"] > 0.99
