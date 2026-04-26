from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from proxy_data.youcook2_seg.event_logic import build_event_logic_vlm as el
from proxy_data.shared.frame_cache import build_source_cache_dir


def _annotation(source_video_path: str = "/videos/videoA.mp4") -> dict:
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
        "source_video_path": source_video_path,
        "clip_duration_sec": 30.0,
        "level2": {"events": events},
        "level3": {"grounding_results": []},
    }


def _option_lines(prompt: str) -> list[str]:
    return [line for line in prompt.splitlines() if len(line) > 3 and line[1:3] == ". "]


def _make_fake_source_cache(tmp_path: Path, source_video_path: str, n_frames: int = 60) -> Path:
    frames_root = tmp_path / "source_2fps"
    cache_dir = build_source_cache_dir(frames_root, "videoA", source_video_path, 2.0)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(1, n_frames + 1):
        (cache_dir / f"{idx:06d}.jpg").write_bytes(b"jpg")
    return frames_root


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


def test_predict_next_can_emit_shared_source_frame_list_from_cache(tmp_path: Path):
    source_video = tmp_path / "videoA.mp4"
    source_video.write_bytes(b"")
    frames_root = _make_fake_source_cache(tmp_path, str(source_video))
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

    records = el._assemble_predict_next(
        parsed,
        _annotation(str(source_video)),
        str(tmp_path / "clips"),
        False,
        el.random.Random(7),
        frame_cache_root=str(frames_root),
        cache_fps=2.0,
        frame_view_fps=2.0,
    )

    assert len(records) == 1
    record = records[0]
    assert len(record["videos"]) == 1
    assert isinstance(record["videos"][0], list)
    assert len(record["videos"][0]) == 20
    assert record["videos"][0][0].endswith("000001.jpg")
    assert record["videos"][0][-1].endswith("000020.jpg")
    assert record["metadata"]["shared_source_frames"]["cache_fps"] == 2.0
    assert record["metadata"]["video_fps_override"] == 2.0


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


def test_fill_blank_can_emit_step_frame_lists_from_shared_source_cache(tmp_path: Path):
    source_video = tmp_path / "videoA.mp4"
    source_video.write_bytes(b"")
    frames_root = _make_fake_source_cache(tmp_path, str(source_video))
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

    records = el._assemble_fill_blank(
        parsed,
        _annotation(str(source_video)),
        str(tmp_path / "clips"),
        False,
        el.random.Random(11),
        frame_cache_root=str(frames_root),
        cache_fps=2.0,
        frame_view_fps=2.0,
    )

    assert len(records) == 1
    record = records[0]
    assert len(record["videos"]) == 3
    assert all(isinstance(video, list) for video in record["videos"])
    assert [len(video) for video in record["videos"]] == [10, 10, 10]
    assert "BLACK SCREEN" not in record["prompt"]
    assert "Step 3: [MISSING]" in record["prompt"]
    assert record["metadata"]["shared_source_frames"]["cache_fps"] == 2.0


def test_event_logic_rollout_script_defaults_to_qwen3_vl_8b_with_eight_rollouts():
    script = (REPO_ROOT / "proxy_data/youcook2_seg/event_logic/run_event_logic_rollout.sh").read_text(
        encoding="utf-8"
    )

    assert 'Qwen3-VL-8B-Instruct' in script
    assert 'NUM_ROLLOUTS="${NUM_ROLLOUTS:-8}"' in script


def test_event_logic_vlm_script_defaults_to_shared_source_frame_cache():
    script = (REPO_ROOT / "proxy_data/youcook2_seg/event_logic/run_event_logic_vlm.sh").read_text(
        encoding="utf-8"
    )

    assert "frame_cache/source_2fps" in script
    assert "--frame-cache-root" in script
    assert "USE_SHARED_FRAMES" in script
    assert 'CACHE_DIR="${CACHE_DIR:-${OUTPUT_DIR}/cache}"' in script
    assert '--cache-dir "$CACHE_DIR"' in script
