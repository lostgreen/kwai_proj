from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATH = REPO_ROOT / "video_proxy" / "training" / "tools" / "val_cot_change_cases.py"
_SPEC = importlib.util.spec_from_file_location("val_cot_change_cases", _MODULE_PATH)
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


def test_collect_val_cot_change_cases_groups_same_uid_across_val_steps(tmp_path: Path):
    exp_dir = tmp_path / "exp"
    _write_jsonl(
        exp_dir / "rollouts" / "val_step_000000.jsonl",
        [
            {
                "uid": "same-question",
                "problem_type": "event_logic_sort",
                "prompt": "Sort events A/B/C.",
                "ground_truth": "ABC",
                "reward": 0.0,
                "response": "<thought>Guess the order from the listed events.</thought><answer>BAC</answer>",
                "cot_budget_debug": {"cot_repaired": False, "final_token_len": 18},
            },
            {
                "uid": "stable-question",
                "problem_type": "event_logic_sort",
                "prompt": "Stable",
                "ground_truth": "A",
                "reward": 1.0,
                "response": "<thought>Same reasoning.</thought><answer>A</answer>",
            },
        ],
    )
    _write_jsonl(
        exp_dir / "rollouts" / "val_step_000025.jsonl",
        [
            {
                "uid": "same-question",
                "problem_type": "event_logic_sort",
                "prompt": "Sort events A/B/C.",
                "ground_truth": "ABC",
                "reward": 1.0,
                "response": "<thought>Use temporal cues: A happens before B, then C follows.</thought><answer>ABC</answer>",
                "cot_budget_debug": {"cot_repaired": False, "final_token_len": 24},
            },
            {
                "uid": "stable-question",
                "problem_type": "event_logic_sort",
                "prompt": "Stable",
                "ground_truth": "A",
                "reward": 1.0,
                "response": "<thought>Same reasoning.</thought><answer>A</answer>",
            },
        ],
    )

    cases = _MODULE.collect_val_cot_change_cases(exp_dir, label="logic")

    assert [case.uid for case in cases] == ["same-question"]
    case = cases[0]
    assert case.label == "logic"
    assert case.problem_type == "event_logic_sort"
    assert case.reward_min == 0.0
    assert case.reward_max == 1.0
    assert case.answer_changed is True
    assert [attempt.step for attempt in case.attempts] == [0, 25]
    assert case.attempts[0].thought == "Guess the order from the listed events."
    assert case.attempts[1].answer == "ABC"


def test_collect_val_cot_change_cases_uses_content_fingerprint_without_uid(tmp_path: Path):
    exp_dir = tmp_path / "exp"
    base_row = {
        "problem_type": "temporal_seg_hier_L2",
        "prompt": "Split this clip into events.",
        "ground_truth": "<events>[[0, 4], [4, 8]]</events>",
        "metadata": {"clip_key": "clip-7", "parent_event_id": 2},
    }
    _write_jsonl(
        exp_dir / "rollouts" / "val_step_000000.jsonl",
        [
            {
                **base_row,
                "reward": 0.2,
                "response": "<thought>One broad action.</thought><events>[[0, 8]]</events>",
            }
        ],
    )
    _write_jsonl(
        exp_dir / "rollouts" / "val_step_000050.jsonl",
        [
            {
                **base_row,
                "reward": 1.0,
                "response": "<thought>The camera shifts after the first action, so split at 4.</thought><events>[[0, 4], [4, 8]]</events>",
            }
        ],
    )

    cases = _MODULE.collect_val_cot_change_cases(exp_dir, label="seg")

    assert len(cases) == 1
    assert cases[0].uid.startswith("fingerprint:")
    assert cases[0].display_id == "clip-7"
    assert cases[0].cot_changed is True


def test_collect_val_cot_change_cases_ignores_ephemeral_val_uid(tmp_path: Path):
    exp_dir = tmp_path / "exp"
    base_row = {
        "problem_type": "event_logic_fill_blank",
        "prompt": "Fill the missing event.",
        "ground_truth": "C",
        "metadata": {"clip_key": "clip-9", "missing_id": "Action 2.1"},
    }
    _write_jsonl(
        exp_dir / "rollouts" / "val_step_000000.jsonl",
        [
            {
                **base_row,
                "uid": "val-random-a",
                "reward": 0.0,
                "response": "<thought>Pick the visible static shot.</thought><answer>A</answer>",
            }
        ],
    )
    _write_jsonl(
        exp_dir / "rollouts" / "val_step_000025.jsonl",
        [
            {
                **base_row,
                "uid": "val-random-b",
                "reward": 1.0,
                "response": "<thought>The missing action must bridge the tool approaching and the engine spinning.</thought><answer>C</answer>",
            }
        ],
    )

    cases = _MODULE.collect_val_cot_change_cases(exp_dir, label="logic")

    assert len(cases) == 1
    assert cases[0].uid.startswith("fingerprint:")
    assert [attempt.step for attempt in cases[0].attempts] == [0, 25]


def test_collect_val_cot_change_cases_can_filter_problem_type(tmp_path: Path):
    exp_dir = tmp_path / "exp"
    rows_0 = [
        {
            "uid": "logic-1",
            "problem_type": "event_logic_sort",
            "prompt": "logic",
            "ground_truth": "A",
            "reward": 0.0,
            "response": "<thought>old logic</thought><answer>B</answer>",
        },
        {
            "uid": "mcq-1",
            "problem_type": "llava_mcq",
            "prompt": "mcq",
            "ground_truth": "A",
            "reward": 0.0,
            "response": "<thought>old mcq</thought><answer>B</answer>",
        },
    ]
    rows_25 = [
        {
            **rows_0[0],
            "reward": 1.0,
            "response": "<thought>new logic</thought><answer>A</answer>",
        },
        {
            **rows_0[1],
            "reward": 1.0,
            "response": "<thought>new mcq</thought><answer>A</answer>",
        },
    ]
    _write_jsonl(exp_dir / "rollouts" / "val_step_000000.jsonl", rows_0)
    _write_jsonl(exp_dir / "rollouts" / "val_step_000025.jsonl", rows_25)

    cases = _MODULE.collect_val_cot_change_cases(exp_dir, task_filter="logic")

    assert [case.problem_type for case in cases] == ["event_logic_sort"]


def test_extract_thought_and_answer_handles_repaired_truncated_cot():
    response = "<thought>Compare A and B but the thought is cut<answer>B</answer>"

    assert _MODULE.extract_thought(response) == "Compare A and B but the thought is cut"
    assert _MODULE.extract_answer(response) == "B"
