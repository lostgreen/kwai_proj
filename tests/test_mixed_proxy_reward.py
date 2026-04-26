from __future__ import annotations

import sys
import types
import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

verl_mod = types.ModuleType("verl")
reward_pkg = types.ModuleType("verl.reward_function")
tg_mod = types.ModuleType("verl.reward_function.temporal_grounding_reward")
utils_mod = types.ModuleType("verl.reward_function.reward_utils")
tg_mod.temporal_grounding_reward = lambda *_args, **_kwargs: {"overall": 0.0, "format": 0.0, "accuracy": 0.0}
utils_mod.compute_f1_iou = lambda *_args, **_kwargs: 0.0
utils_mod.has_events_tag = lambda *_args, **_kwargs: False
utils_mod.parse_segments = lambda *_args, **_kwargs: []
sys.modules.setdefault("verl", verl_mod)
sys.modules.setdefault("verl.reward_function", reward_pkg)
sys.modules.setdefault("verl.reward_function.temporal_grounding_reward", tg_mod)
sys.modules.setdefault("verl.reward_function.reward_utils", utils_mod)

SPEC = importlib.util.spec_from_file_location(
    "mixed_proxy_reward_test_module",
    REPO_ROOT / "verl" / "reward_function" / "mixed_proxy_reward.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
compute_score = MODULE.compute_score


def test_choice_reward_accepts_bare_six_way_event_logic_answer():
    scores = compute_score([
        {
            "response": "F",
            "ground_truth": "F",
            "problem_type": "event_logic_predict_next",
            "data_type": "video",
        }
    ])

    assert scores == [{"overall": 1.0, "format": 0.0, "accuracy": 1.0}]


def test_choice_reward_accepts_tagged_six_way_event_logic_answer():
    scores = compute_score([
        {
            "response": "<answer>E</answer>",
            "ground_truth": "E",
            "problem_type": "event_logic_fill_blank",
            "data_type": "video",
        }
    ])

    assert scores == [{"overall": 1.0, "format": 1.0, "accuracy": 1.0}]
