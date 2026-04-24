from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "video_fps_under_test",
    REPO_ROOT / "verl" / "utils" / "video_fps.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

resolve_video_fps = _MODULE.resolve_video_fps
resolve_video_fps_list = _MODULE.resolve_video_fps_list


def test_frame_policy_fps_uses_effective_output_frames_over_base_override():
    metadata = {
        "video_fps_override": 2.0,
        "experiment_frame_sampling": {
            "videos": [
                {
                    "duration_sec": 48.38,
                    "base_fps": 2.0,
                    "target_fps": 1.0,
                    "input_frames": 97,
                    "after_fps_frames": 49,
                    "output_frames": 49,
                }
            ]
        },
    }

    fps = resolve_video_fps(metadata, default_fps=2.0)

    assert abs(fps - 49 / 48.38) < 1e-6


def test_frame_policy_uniform_cap_uses_average_effective_fps():
    metadata = {
        "video_fps_override": 2.0,
        "experiment_frame_sampling": {
            "videos": [
                {
                    "duration_sec": 200.0,
                    "base_fps": 2.0,
                    "target_fps": None,
                    "input_frames": 400,
                    "after_fps_frames": 400,
                    "output_frames": 64,
                }
            ]
        },
    }

    fps = resolve_video_fps(metadata, default_fps=2.0)

    assert fps == 0.32


def test_legacy_video_fps_override_still_works_without_frame_policy():
    assert resolve_video_fps({"video_fps_override": 1.0}, default_fps=2.0) == 1.0
    assert resolve_video_fps({"level": 1, "l1_fps": 1}, default_fps=2.0) == 1.0


def test_per_video_fps_list_uses_each_sampling_video():
    metadata = {
        "experiment_frame_sampling": {
            "videos": [
                {"duration_sec": 10.0, "output_frames": 20},
                {"duration_sec": 20.0, "output_frames": 10},
            ]
        },
    }

    assert resolve_video_fps_list(metadata, default_fps=2.0, n_videos=2) == [2.0, 0.5]
