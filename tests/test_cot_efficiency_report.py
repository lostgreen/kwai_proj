from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATH = REPO_ROOT / "video_proxy" / "training" / "tools" / "cot_efficiency_report.py"
_SPEC = importlib.util.spec_from_file_location("cot_efficiency_report", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_summarize_experiment_reports_learning_and_cot_status(tmp_path: Path):
    exp_dir = tmp_path / "exp"
    _write_jsonl(
        exp_dir / "experiment_log.jsonl",
        [
            {"step": 0, "val": {"reward_score": 0.1}},
            {"step": 1, "reward": {"overall": 0.2}},
            {"step": 50, "reward": {"overall": 0.5}, "val": {"reward_score": 0.3}},
            {
                "step": 100,
                "reward": {"overall": 0.8},
                "val": {
                    "reward_score": 0.6,
                    "temporal_grounding": {"overall_reward": 0.4, "count": 100},
                    "llava_mcq": {"overall_reward": 0.8, "count": 100},
                    "seg_aot_action_t2v_binary": {"overall_reward": 0.7, "count": 50},
                },
                "cot_budget": {
                    "end_detected_ratio": 0.75,
                    "repaired_ratio": 0.25,
                    "final_token_len_mean": 128,
                },
            },
        ],
    )
    _write_jsonl(
        exp_dir / "rollouts" / "step_000100.jsonl",
        [
            {
                "uid": "a",
                "problem_type": "seg_aot_action_t2v_binary",
                "reward": 1.0,
                "response": "<thought>ok</thought><answer>A</answer>",
                "cot_budget_debug": {
                    "cot_budget_enabled": True,
                    "cot_start_detected": True,
                    "cot_end_detected": True,
                    "cot_repaired": False,
                    "remaining_tokens": 12,
                    "final_token_len": 90,
                },
            },
            {
                "uid": "b",
                "problem_type": "seg_aot_action_t2v_binary",
                "reward": 0.0,
                "response": "<thought>long</thought>",
                "cot_budget_debug": {
                    "cot_budget_enabled": True,
                    "cot_start_detected": True,
                    "cot_end_detected": False,
                    "cot_repaired": True,
                    "remaining_tokens": 0,
                    "final_token_len": 256,
                },
            },
        ],
    )

    summary = _MODULE.summarize_experiment(exp_dir, label="aot_cot", max_step=100)

    assert summary["label"] == "aot_cot"
    assert summary["train_final"] == 0.8
    assert summary["train_delta"] == pytest.approx(0.6)
    assert summary["val_final"] == 0.6
    assert summary["val_delta"] == pytest.approx(0.5)
    assert summary["step_to_train_0.5"] == 50
    assert summary["log_cot_end_ratio"] == 0.75
    assert summary["log_cot_repaired_ratio"] == 0.25
    assert summary["log_cot_final_token_len_mean"] == 128
    assert summary["base_val_final"] == pytest.approx(0.6)
    assert summary["proxy_val_final"] == pytest.approx(0.7)
    assert summary["val_task_finals"]["seg_aot_action_t2v_binary"] == pytest.approx(0.7)
    assert summary["rollout_step"] == 100
    assert summary["cot_start_ratio"] == 1.0
    assert summary["cot_end_ratio"] == 0.5
    assert summary["cot_repaired_ratio"] == 0.5


def test_compare_cot_delta_uses_matching_family(tmp_path: Path):
    nocot = {
        "family": "aot",
        "mode": "nocot",
        "val_final": 0.4,
        "train_final": 0.5,
        "proxy_val_final": 0.3,
    }
    cot = {
        "family": "aot",
        "mode": "cot",
        "val_final": 0.6,
        "train_final": 0.7,
        "proxy_val_final": 0.8,
    }
    other = {"family": "seg", "mode": "cot", "val_final": 0.9, "train_final": 0.9}

    rows = _MODULE.add_pairwise_deltas([nocot, cot, other])

    assert rows[1]["delta_vs_nocot_val_final"] == pytest.approx(0.2)
    assert rows[1]["delta_vs_nocot_train_final"] == pytest.approx(0.2)
    assert rows[1]["delta_vs_nocot_proxy_val_final"] == pytest.approx(0.5)
    assert "delta_vs_nocot_val_final" not in rows[2]


def test_family_proxy_task_matching():
    assert _MODULE.is_proxy_task("aot", "seg_aot_action_t2v_binary")
    assert _MODULE.is_proxy_task("seg", "temporal_seg_hier_L2")
    assert _MODULE.is_proxy_task("logic", "event_logic_sort")
    assert not _MODULE.is_proxy_task("seg", "llava_mcq")
