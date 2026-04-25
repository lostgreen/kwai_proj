from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "video_fps_under_test",
    REPO_ROOT / "verl" / "utils" / "video_fps.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

build_video_vision_info = MODULE.build_video_vision_info
resolve_video_fps = MODULE.resolve_video_fps
resolve_video_fps_list = MODULE.resolve_video_fps_list


def test_frame_policy_metadata_overrides_legacy_video_fps_override():
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

    assert abs(resolve_video_fps(metadata, default_fps=2.0) - 49 / 48.38) < 1e-6


def test_uniform_cap_uses_average_effective_fps_not_target_fps():
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

    assert resolve_video_fps(metadata, default_fps=2.0) == 0.32


def test_frame_list_vision_info_sets_qwen_sample_and_raw_fps():
    info = build_video_vision_info(
        ["000001.jpg", "000003.jpg", "000005.jpg"],
        min_pixels=3136,
        max_pixels=65536,
        max_frames=64,
        video_fps=1.013,
    )

    assert info["fps"] == 1.013
    assert info["sample_fps"] == 1.013
    assert info["raw_fps"] == 1.013


def test_mp4_vision_info_does_not_invent_frame_list_metadata():
    info = build_video_vision_info(
        "/tmp/video.mp4",
        min_pixels=3136,
        max_pixels=65536,
        max_frames=64,
        video_fps=2.0,
    )

    assert info["fps"] == 2.0
    assert "sample_fps" not in info
    assert "raw_fps" not in info


def test_per_video_resolver_keeps_multi_video_fps_separate():
    metadata = {
        "experiment_frame_sampling": {
            "videos": [
                {"duration_sec": 10.0, "output_frames": 20},
                {"duration_sec": 20.0, "output_frames": 10},
            ]
        }
    }

    assert resolve_video_fps_list(metadata, default_fps=2.0, n_videos=2) == [2.0, 0.5]


def test_runtime_paths_do_not_read_video_fps_override_directly():
    checked_files = [
        REPO_ROOT / "verl" / "utils" / "dataset.py",
        REPO_ROOT / "verl" / "workers" / "rollout" / "vllm_rollout_spmd.py",
        REPO_ROOT / "verl" / "workers" / "fsdp_workers.py",
    ]

    for path in checked_files:
        text = path.read_text(encoding="utf-8")
        assert 'meta.get("video_fps_override")' not in text
        assert "meta.get('video_fps_override')" not in text


def test_worker_paths_use_sample_level_video_fps():
    vllm_text = (REPO_ROOT / "verl" / "workers" / "rollout" / "vllm_rollout_spmd.py").read_text(
        encoding="utf-8"
    )
    fsdp_text = (REPO_ROOT / "verl" / "workers" / "fsdp_workers.py").read_text(encoding="utf-8")

    assert 'sample_fps = multi_modal_data.get("video_fps", video_fps)' in vllm_text
    assert 'sample_fps = multi_modal_data.get("video_fps", data.meta_info.get("video_fps", 2.0))' in fsdp_text
    assert 'kwargs["video_fps"] = sample_fps' in vllm_text
    assert 'kwargs["video_fps"] = sample_fps' in fsdp_text


def test_offline_rollout_filter_uses_shared_fps_resolver():
    text = (REPO_ROOT / "local_scripts" / "offline_rollout_filter.py").read_text(encoding="utf-8")

    assert "resolve_video_fps_list" in text
    assert 'effective_fps = meta.get("video_fps_override")' not in text
    assert "video_fps_overrides = resolve_video_fps_list" in text
