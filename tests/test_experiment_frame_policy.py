from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from local_scripts.data.frame_policy import apply_frame_policy, parse_frame_policy


def _frame_dir(tmp_path: Path, name: str, n_frames: int) -> Path:
    frame_dir = tmp_path / name
    frame_dir.mkdir(parents=True)
    for idx in range(n_frames):
        (frame_dir / f"frame_{idx:06d}.jpg").write_text("", encoding="utf-8")
    return frame_dir


def _record(frame_dir: Path, duration_sec: float) -> dict:
    return {
        "prompt": "p",
        "answer": "A",
        "problem_type": "llava_mcq",
        "data_type": "video",
        "videos": [["stale_frame_list_should_not_be_used.jpg"]],
        "metadata": {
            "video_fps_override": 2.0,
            "offline_frame_extraction": {
                "effective_fps": 2.0,
                "uncapped_extraction": True,
                "duration_sec": duration_sec,
                "videos": [{"frame_dir": str(frame_dir)}],
            },
        },
    }


def test_frame_policy_reads_cache_dir_and_downsamples_by_duration(tmp_path: Path):
    short_dir = _frame_dir(tmp_path, "short", 120)
    long_dir = _frame_dir(tmp_path, "long", 240)

    short, long = apply_frame_policy(
        [_record(short_dir, 60.0), _record(long_dir, 120.0)],
        policy="0:60:2.0,60:inf:1.0",
        max_frames=256,
    )

    assert len(short["videos"][0]) == 120
    assert len(long["videos"][0]) == 120
    assert str(long_dir / "frame_000000.jpg") == long["videos"][0][0]
    assert long["metadata"]["experiment_frame_sampling"]["policy"] == "0:60:2.0,60:inf:1.0"
    assert long["metadata"]["experiment_frame_sampling"]["max_frames"] == 256
    assert long["metadata"]["experiment_frame_sampling"]["videos"][0]["target_fps"] == 1.0


def test_frame_policy_uniform_caps_after_fps_downsampling(tmp_path: Path):
    frame_dir = _frame_dir(tmp_path, "very_long", 800)

    sampled = apply_frame_policy(
        [_record(frame_dir, 400.0)],
        policy="0:60:2.0,60:inf:1.0",
        max_frames=256,
    )[0]

    assert len(sampled["videos"][0]) == 256
    assert sampled["videos"][0][0] == str(frame_dir / "frame_000000.jpg")
    assert sampled["videos"][0][-1] == str(frame_dir / "frame_000799.jpg")
    assert sampled["metadata"]["experiment_frame_sampling"]["videos"][0]["after_fps_frames"] == 400


def test_frame_policy_skips_non_uncapped_or_non_2fps_records(tmp_path: Path):
    frame_dir = _frame_dir(tmp_path, "old_cache", 240)
    record = _record(frame_dir, 120.0)
    record["metadata"]["offline_frame_extraction"]["uncapped_extraction"] = False
    record["videos"] = [["existing.jpg"]]

    sampled = apply_frame_policy(
        [record],
        policy="0:60:2.0,60:inf:1.0",
        max_frames=256,
    )[0]

    assert sampled["videos"] == [["existing.jpg"]]
    assert "experiment_frame_sampling" not in sampled["metadata"]


def test_parse_frame_policy_supports_uniform_rule():
    rules = parse_frame_policy("0:128:2.0,128:inf:uniform")

    assert rules[0].fps == 2.0
    assert rules[1].fps is None
