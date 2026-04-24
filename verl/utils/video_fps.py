from __future__ import annotations

from typing import Any


def _float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _sampling_video_fps(video_meta: dict[str, Any]) -> float | None:
    duration = _float_or_none(video_meta.get("duration_sec"))
    output_frames = _float_or_none(video_meta.get("output_frames"))
    if duration is not None and output_frames is not None:
        return output_frames / duration

    target_fps = _float_or_none(video_meta.get("target_fps"))
    if target_fps is not None:
        return target_fps

    return _float_or_none(video_meta.get("base_fps"))


def resolve_video_fps_list(metadata: dict[str, Any] | None, default_fps: float, n_videos: int) -> list[float]:
    """Resolve per-video fps for already-sampled frame-list inputs.

    Experiment JSONL can pre-sample frame lists at task-specific fps/max-frame
    policies. In that case the training processor must use the effective fps of
    the sampled list, not the base cache fps stored in older metadata fields.
    """
    n_videos = max(1, int(n_videos))
    metadata = metadata or {}

    sampling = metadata.get("experiment_frame_sampling")
    if isinstance(sampling, dict):
        sampling_videos = sampling.get("videos")
        if isinstance(sampling_videos, list) and sampling_videos:
            fps_values: list[float] = []
            for idx in range(n_videos):
                video_meta = sampling_videos[min(idx, len(sampling_videos) - 1)]
                if isinstance(video_meta, dict):
                    fps = _sampling_video_fps(video_meta)
                    if fps is not None:
                        fps_values.append(fps)
                        continue
                fps_values.append(float(default_fps))
            return fps_values

    override_fps = _float_or_none(metadata.get("video_fps_override"))
    if override_fps is None and metadata.get("level") == 1:
        override_fps = _float_or_none(metadata.get("l1_fps"))
    fps = override_fps if override_fps is not None else float(default_fps)
    return [fps] * n_videos


def resolve_video_fps(metadata: dict[str, Any] | None, default_fps: float, n_videos: int = 1) -> float:
    return resolve_video_fps_list(metadata, default_fps, n_videos)[0]
