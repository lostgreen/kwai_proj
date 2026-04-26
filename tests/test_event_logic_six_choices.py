from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from proxy_data.youcook2_seg.event_logic import build_event_logic_vlm as el


def _annotation() -> dict:
    events = [
        {"event_id": 1, "instruction": "wash the tomatoes", "start_time": 0, "end_time": 5, "parent_phase_id": 1},
        {"event_id": 2, "instruction": "slice the tomatoes", "start_time": 5, "end_time": 10, "parent_phase_id": 1},
        {"event_id": 3, "instruction": "mix the tomatoes with oil", "start_time": 10, "end_time": 15, "parent_phase_id": 1},
        {"event_id": 4, "instruction": "season the tomato mixture", "start_time": 15, "end_time": 20, "parent_phase_id": 1},
        {"event_id": 5, "instruction": "plate the tomato salad", "start_time": 20, "end_time": 25, "parent_phase_id": 1},
        {"event_id": 6, "instruction": "garnish the tomato salad", "start_time": 25, "end_time": 30, "parent_phase_id": 1},
    ]
    return {
        "clip_key": "videoA",
        "domain_l1": "cooking",
        "domain_l2": "salad",
        "level2": {"events": events},
        "level3": {"grounding_results": []},
    }


def _option_lines(prompt: str) -> list[str]:
    return [line for line in prompt.splitlines() if len(line) > 3 and line[1:3] == ". "]


def test_predict_next_uses_six_options_with_two_same_video_instruction_distractors(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(el, "_concat_video_files", lambda _paths, _output: True)
    parsed = {
        "suitable": True,
        "granularity": "Event",
        "context_ids": ["Event 1", "Event 2"],
        "correct_next_id": "Event 3",
        "correct_next_text": "mix the tomatoes with oil",
        "distractors": [
            "mix the tomatoes without oil",
            "slice the tomatoes again",
            "pour the tomatoes onto the counter",
        ],
    }

    records = el._assemble_predict_next(parsed, _annotation(), str(tmp_path), False, el.random.Random(7))

    assert len(records) == 1
    record = records[0]
    assert record["answer"] in list("ABCDEF")
    assert [line[:2] for line in _option_lines(record["prompt"])] == ["A.", "B.", "C.", "D.", "E.", "F."]
    assert len(record["metadata"]["distractors"]) == 5
    assert record["metadata"]["same_video_instruction_distractors"]


def test_fill_blank_uses_six_options_with_two_same_video_instruction_distractors(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(el, "_concat_clips_with_black", lambda _before, _after, _output: True)
    parsed = {
        "suitable": True,
        "granularity": "Event",
        "before_ids": ["Event 1", "Event 2"],
        "missing_id": "Event 3",
        "after_ids": ["Event 4"],
        "correct_text": "mix the tomatoes with oil",
        "distractors": [
            "mix the tomatoes without oil",
            "slice the tomatoes again",
            "pour the tomatoes onto the counter",
        ],
    }

    records = el._assemble_fill_blank(parsed, _annotation(), str(tmp_path), False, el.random.Random(11))

    assert len(records) == 1
    record = records[0]
    assert record["answer"] in list("ABCDEF")
    assert [line[:2] for line in _option_lines(record["prompt"])] == ["A.", "B.", "C.", "D.", "E.", "F."]
    assert len(record["metadata"]["distractors"]) == 5
    assert record["metadata"]["same_video_instruction_distractors"]


def test_event_logic_rollout_script_defaults_to_qwen3_vl_8b_with_eight_rollouts():
    script = (REPO_ROOT / "proxy_data/youcook2_seg/event_logic/run_event_logic_rollout.sh").read_text(
        encoding="utf-8"
    )

    assert 'Qwen3-VL-8B-Instruct' in script
    assert 'NUM_ROLLOUTS="${NUM_ROLLOUTS:-8}"' in script
