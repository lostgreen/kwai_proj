"""Frame-list downsampling for experiment JSONL derivation.

Base data can keep dense 2fps frame lists while each experiment writes a
smaller train/val JSONL according to its own duration/fps/max-frame policy.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FramePolicyRule:
    min_sec: float
    max_sec: float
    fps: float | None


def _parse_bound(value: str) -> float:
    value = value.strip().lower()
    if value in {"inf", "+inf", "infty", "infinite"}:
        return math.inf
    return float(value)


def parse_frame_policy(policy: str) -> list[FramePolicyRule]:
    """Parse "0:60:2.0,60:inf:1.0" into duration/fps rules.

    Use fps value "uniform" to skip fps downsampling and only apply the final
    max-frame uniform cap.
    """
    rules: list[FramePolicyRule] = []
    policy = (policy or "").strip()
    if not policy:
        return rules

    for raw_part in policy.split(","):
        part = raw_part.strip()
        if not part:
            continue
        pieces = [piece.strip() for piece in part.split(":")]
        if len(pieces) != 3:
            raise ValueError(f"Invalid frame policy rule {part!r}; expected min:max:fps")
        min_sec = _parse_bound(pieces[0])
        max_sec = _parse_bound(pieces[1])
        if max_sec <= min_sec:
            raise ValueError(f"Invalid frame policy rule {part!r}; max must be > min")
        fps_value = pieces[2].lower()
        fps = None if fps_value == "uniform" else float(fps_value)
        if fps is not None and fps <= 0:
            raise ValueError(f"Invalid frame policy rule {part!r}; fps must be positive")
        rules.append(FramePolicyRule(min_sec=min_sec, max_sec=max_sec, fps=fps))
    return rules


def _duration_from_record(record: dict[str, Any], frames: list[str], base_fps: float) -> float:
    meta = record.get("metadata") or {}
    extraction = meta.get("offline_frame_extraction") or {}
    for value in (
        extraction.get("duration_sec"),
        meta.get("duration"),
        meta.get("clip_duration"),
        record.get("duration"),
    ):
        try:
            duration = float(value)
        except (TypeError, ValueError):
            continue
        if duration > 0:
            return duration
    return len(frames) / max(base_fps, 1e-6)


def _base_fps_from_record(record: dict[str, Any]) -> float:
    meta = record.get("metadata") or {}
    extraction = meta.get("offline_frame_extraction") or {}
    for value in (
        extraction.get("effective_fps"),
        meta.get("video_fps_override"),
    ):
        try:
            fps = float(value)
        except (TypeError, ValueError):
            continue
        if fps > 0:
            return fps
    return 2.0


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _cache_frames_from_record(record: dict[str, Any]) -> tuple[list[list[str]] | None, str]:
    meta = record.get("metadata") or {}
    extraction = meta.get("offline_frame_extraction") or {}
    base_fps = _base_fps_from_record(record)

    if not _is_truthy(extraction.get("uncapped_extraction")):
        return None, "offline_frame_extraction.uncapped_extraction is not true"
    if abs(base_fps - 2.0) > 1e-6:
        return None, f"base fps is {base_fps}, expected 2.0"

    videos_meta = extraction.get("videos")
    if not isinstance(videos_meta, list) or not videos_meta:
        return None, "offline_frame_extraction.videos missing"

    cache_videos: list[list[str]] = []
    for idx, video_meta in enumerate(videos_meta):
        if not isinstance(video_meta, dict):
            return None, f"offline_frame_extraction.videos[{idx}] is not an object"
        frame_dir = video_meta.get("frame_dir")
        if not frame_dir:
            return None, f"offline_frame_extraction.videos[{idx}].frame_dir missing"
        frame_dir_path = Path(str(frame_dir))
        if not frame_dir_path.is_absolute():
            return None, f"offline_frame_extraction.videos[{idx}].frame_dir is not absolute"
        if not frame_dir_path.is_dir():
            return None, f"frame_dir not found: {frame_dir_path}"
        frames = sorted(str(path) for path in frame_dir_path.glob("*.jpg"))
        if not frames:
            return None, f"frame_dir has no jpg frames: {frame_dir_path}"
        cache_videos.append(frames)
    return cache_videos, ""


def _rule_for_duration(rules: list[FramePolicyRule], duration_sec: float) -> FramePolicyRule | None:
    for rule in rules:
        if duration_sec >= rule.min_sec and duration_sec <= rule.max_sec:
            return rule
    return None


def _uniform_indices(n_items: int, target_count: int) -> list[int]:
    if target_count <= 0 or target_count >= n_items:
        return list(range(n_items))
    if target_count == 1:
        return [0]
    return sorted({round(i * (n_items - 1) / (target_count - 1)) for i in range(target_count)})


def _downsample_by_fps(frames: list[str], base_fps: float, target_fps: float | None) -> list[str]:
    if target_fps is None or target_fps >= base_fps:
        return list(frames)
    target_count = max(1, math.ceil(len(frames) * target_fps / base_fps))
    return [frames[idx] for idx in _uniform_indices(len(frames), target_count)]


def _cap_uniform(frames: list[str], max_frames: int) -> list[str]:
    if max_frames <= 0 or len(frames) <= max_frames:
        return list(frames)
    return [frames[idx] for idx in _uniform_indices(len(frames), max_frames)]


def apply_frame_policy_to_record(
    record: dict[str, Any],
    rules: list[FramePolicyRule],
    max_frames: int,
    policy: str = "",
) -> dict[str, Any]:
    cache_videos, _skip_reason = _cache_frames_from_record(record)
    if cache_videos is None:
        return record

    base_fps = _base_fps_from_record(record)
    n_videos = len(cache_videos)
    max_frames_per_video = max(1, max_frames // n_videos) if max_frames > 0 and n_videos > 1 else max_frames
    rewritten = copy.deepcopy(record)
    rewritten_videos: list[list[str]] = []
    policy_meta: list[dict[str, Any]] = []

    for frames in cache_videos:
        duration_sec = _duration_from_record(record, frames, base_fps)
        rule = _rule_for_duration(rules, duration_sec)
        target_fps = rule.fps if rule is not None else None
        after_fps = _downsample_by_fps(list(frames), base_fps, target_fps)
        final_frames = _cap_uniform(after_fps, max_frames_per_video)
        rewritten_videos.append(final_frames)
        policy_meta.append(
            {
                "duration_sec": duration_sec,
                "base_fps": base_fps,
                "target_fps": target_fps,
                "max_frames": max_frames_per_video,
                "input_frames": len(frames),
                "after_fps_frames": len(after_fps),
                "output_frames": len(final_frames),
            }
        )

    meta = dict(rewritten.get("metadata") or {})
    meta["experiment_frame_sampling"] = {
        "policy": policy,
        "max_frames": max_frames,
        "rules": [
            {
                "min_sec": rule.min_sec,
                "max_sec": "inf" if math.isinf(rule.max_sec) else rule.max_sec,
                "fps": "uniform" if rule.fps is None else rule.fps,
            }
            for rule in rules
        ],
        "videos": policy_meta,
    }
    rewritten["metadata"] = meta
    rewritten["videos"] = rewritten_videos
    return rewritten


def apply_frame_policy(
    records: list[dict[str, Any]],
    policy: str,
    max_frames: int,
) -> list[dict[str, Any]]:
    rules = parse_frame_policy(policy)
    if not rules and max_frames <= 0:
        return records
    return [apply_frame_policy_to_record(record, rules, max_frames, policy) for record in records]


def summarize_frame_policy_application(records: list[dict[str, Any]]) -> dict[str, int]:
    applied = 0
    skipped = 0
    for record in records:
        meta = record.get("metadata") or {}
        if "experiment_frame_sampling" in meta:
            applied += 1
        else:
            cache_videos, _reason = _cache_frames_from_record(record)
            if cache_videos is None:
                skipped += 1
    return {"applied": applied, "skipped": skipped}
