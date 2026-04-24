from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from local_scripts.data.frame_policy import apply_frame_policy, parse_frame_policy


def _frame_dir(tmp_path: Path, name: str, n_frames: int) -> Path:
    frame_dir = tmp_path / name
    frame_dir.mkdir(parents=True)
    for idx in range(1, n_frames + 1):
        (frame_dir / f"{idx:06d}.jpg").write_text("", encoding="utf-8")
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
        cache_roots=[tmp_path],
    )

    assert len(short["videos"][0]) == 120
    assert len(long["videos"][0]) == 120
    assert str(long_dir / "000001.jpg") == long["videos"][0][0]
    assert long["metadata"]["experiment_frame_sampling"]["policy"] == "0:60:2.0,60:inf:1.0"
    assert long["metadata"]["experiment_frame_sampling"]["max_frames"] == 256
    assert long["metadata"]["experiment_frame_sampling"]["rules"][1]["max_sec"] is None
    assert long["metadata"]["experiment_frame_sampling"]["videos"][0]["target_fps"] == 1.0


def test_frame_policy_metadata_is_pyarrow_safe(tmp_path: Path):
    pa = pytest.importorskip("pyarrow")
    frame_dir = _frame_dir(tmp_path, "arrow_safe", 120)

    sampled = apply_frame_policy(
        [_record(frame_dir, 60.0)],
        policy="0:60:2.0,60:inf:1.0",
        max_frames=256,
        cache_roots=[tmp_path],
    )[0]

    table = pa.Table.from_pylist([sampled])

    assert table.num_rows == 1
    assert sampled["metadata"]["experiment_frame_sampling"]["rules"][1]["max_sec"] is None


def test_frame_policy_uniform_caps_after_fps_downsampling(tmp_path: Path):
    frame_dir = _frame_dir(tmp_path, "very_long", 800)

    sampled = apply_frame_policy(
        [_record(frame_dir, 400.0)],
        policy="0:60:2.0,60:inf:1.0",
        max_frames=256,
        cache_roots=[tmp_path],
    )[0]

    assert len(sampled["videos"][0]) == 256
    assert sampled["videos"][0][0] == str(frame_dir / "000001.jpg")
    assert sampled["videos"][0][-1] == str(frame_dir / "000800.jpg")
    assert sampled["metadata"]["experiment_frame_sampling"]["videos"][0]["after_fps_frames"] == 400


def test_frame_policy_infers_cache_dir_from_existing_frame_lists(tmp_path: Path):
    frame_dir = _frame_dir(tmp_path, "base_cache_2fps", 240)
    record = _record(frame_dir, 120.0)
    record["metadata"]["offline_frame_extraction"]["uncapped_extraction"] = False
    record["videos"] = [[str(frame_dir / f"{idx:06d}.jpg") for idx in range(1, 241, 2)]]

    sampled = apply_frame_policy(
        [record],
        policy="0:60:2.0,60:inf:1.0",
        max_frames=256,
        cache_roots=[tmp_path],
    )[0]

    assert len(sampled["videos"][0]) == 120
    assert sampled["videos"][0][0] == str(frame_dir / "000001.jpg")
    assert sampled["metadata"]["experiment_frame_sampling"]["videos"][0]["source"] == "inferred_2fps_cache"


def test_frame_policy_rejects_existing_frame_lists_outside_trusted_cache_roots(tmp_path: Path):
    old_dir = _frame_dir(tmp_path, "old_1fps_cache", 120)
    trusted_dir = tmp_path / "base_cache_2fps"
    trusted_dir.mkdir()
    record = {
        "prompt": "p",
        "answer": "A",
        "problem_type": "temporal_grounding",
        "data_type": "video",
        "videos": [[str(old_dir / "000001.jpg"), str(old_dir / "000002.jpg")]],
        "metadata": {"video_fps_override": 1.0, "duration": 120.0},
    }

    sampled = apply_frame_policy(
        [record],
        policy="0:60:2.0,60:inf:1.0",
        max_frames=256,
        cache_roots=[trusted_dir],
    )[0]

    assert sampled["videos"] == record["videos"]
    assert "experiment_frame_sampling" not in sampled["metadata"]


def test_frame_policy_reads_hier_shared_source_cache(tmp_path: Path):
    cache_dir = _frame_dir(tmp_path, "hier_cache", 300)
    record = {
        "prompt": "p",
        "answer": "a",
        "problem_type": "temporal_seg_hier_L1",
        "data_type": "video",
        "videos": [[str(cache_dir / "000001.jpg")]],
        "metadata": {
            "video_fps_override": 1.0,
            "shared_source_frames": {
                "cache_dir": str(cache_dir),
                "cache_fps": 2.0,
                "segment_start_sec": 0.0,
                "segment_end_sec": 120.0,
                "target_view_fps": 1.0,
                "n_frames": 120,
            },
        },
    }

    sampled = apply_frame_policy(
        [record],
        policy="0:60:2.0,60:inf:1.0",
        max_frames=256,
        cache_roots=[tmp_path],
    )[0]

    assert len(sampled["videos"][0]) == 120
    assert sampled["videos"][0][0] == str(cache_dir / "000001.jpg")
    assert sampled["metadata"]["experiment_frame_sampling"]["videos"][0]["source"] == "shared_source_cache"
    assert sampled["metadata"]["experiment_frame_sampling"]["implementation_version"] == "trusted_2fps_cache_v2"


def test_parse_frame_policy_supports_uniform_rule():
    rules = parse_frame_policy("0:128:2.0,128:inf:uniform")

    assert rules[0].fps == 2.0
    assert rules[1].fps is None
