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


def test_cot_format_reward_penalizes_repaired_correct_answer_without_changing_accuracy():
    scores = compute_score(
        [
            {
                "response": "<answer>A</answer>",
                "ground_truth": "A",
                "problem_type": "event_logic_predict_next",
                "data_type": "video",
                "cot_budget_debug": {
                    "cot_budget_enabled": True,
                    "cot_repaired": True,
                },
            }
        ],
        cot_format_reward_enabled=True,
        cot_format_truncated=0.5,
        cot_format_ok=1.0,
    )

    assert scores == [
        {
            "overall": 0.5,
            "format": 1.0,
            "accuracy": 1.0,
            "overall_base": 1.0,
            "cot_format": 0.5,
        }
    ]


def test_cot_format_reward_penalizes_missing_cot_start_on_correct_answer():
    scores = compute_score(
        [
            {
                "response": "<answer>A</answer>",
                "ground_truth": "A",
                "problem_type": "event_logic_predict_next",
                "data_type": "video",
                "cot_budget_debug": {
                    "cot_budget_enabled": True,
                    "cot_start_detected": False,
                    "cot_end_detected": False,
                    "cot_repaired": False,
                },
            }
        ],
        cot_format_reward_enabled=True,
        cot_format_missing=0.0,
        cot_format_truncated=0.5,
        cot_format_ok=1.0,
    )

    assert scores == [
        {
            "overall": 0.0,
            "format": 1.0,
            "accuracy": 1.0,
            "overall_base": 1.0,
            "cot_format": 0.0,
        }
    ]


def test_cot_format_reward_keeps_unrepaired_correct_answer_at_full_reward():
    scores = compute_score(
        [
            {
                "response": "<answer>B</answer>",
                "ground_truth": "B",
                "problem_type": "event_logic_fill_blank",
                "data_type": "video",
                "cot_budget_debug": {
                    "cot_budget_enabled": True,
                    "cot_repaired": False,
                },
            }
        ],
        cot_format_reward_enabled=True,
        cot_format_truncated=0.5,
        cot_format_ok=1.0,
    )

    assert scores[0]["overall"] == 1.0
    assert scores[0]["accuracy"] == 1.0
    assert scores[0]["overall_base"] == 1.0
    assert scores[0]["cot_format"] == 1.0


def test_cot_format_reward_does_not_give_wrong_answers_positive_reward():
    scores = compute_score(
        [
            {
                "response": "<answer>C</answer>",
                "ground_truth": "D",
                "problem_type": "event_logic_fill_blank",
                "data_type": "video",
                "cot_budget_debug": {
                    "cot_budget_enabled": True,
                    "cot_repaired": False,
                },
            }
        ],
        cot_format_reward_enabled=True,
        cot_format_truncated=0.5,
        cot_format_ok=1.0,
    )

    assert scores[0]["overall"] == 0.0
    assert scores[0]["accuracy"] == 0.0
    assert scores[0]["overall_base"] == 0.0
    assert scores[0]["cot_format"] == 1.0
