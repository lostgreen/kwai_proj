"""Frame-list downsampling for experiment JSONL derivation.

Base data can keep dense 2fps frame lists while each experiment writes a
smaller train/val JSONL according to its own duration/fps/max-frame policy.
"""

from __future__ import annotations

import copy
import json
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FRAME_POLICY_IMPLEMENTATION_VERSION = "trusted_2fps_cache_v2"


@dataclass(frozen=True)
class FramePolicyRule:
    min_sec: float
    max_sec: float
    fps: float | None


@dataclass(frozen=True)
class FrameVideoSource:
    frame_dir: "FrameDirInfo"
    source_indices: list[int]
    base_fps: float
    duration_sec: float
    source: str


@dataclass(frozen=True)
class FrameDirInfo:
    path: Path
    n_frames: int
    prefix: str
    first_index: int
    suffix: str = ".jpg"
    width: int = 6

    def path_for_index(self, zero_based_index: int) -> str:
        frame_index = self.first_index + zero_based_index
        return str(self.path / f"{self.prefix}{frame_index:0{self.width}d}{self.suffix}")


@dataclass
class FramePolicyContext:
    cache_roots: tuple[Path, ...]
    frame_dir_cache: dict[Path, FrameDirInfo]


@dataclass
class FramePolicyProgress:
    label: str
    total: int
    interval: int = 1000
    started_at: float = 0.0

    def __post_init__(self) -> None:
        self.started_at = time.time()

    def maybe_update(self, done: int, cache_dirs: int) -> None:
        if not self.label or self.total <= 0:
            return
        interval = max(1, self.interval)
        if done not in {1, self.total} and done % interval != 0:
            return

        elapsed = max(time.time() - self.started_at, 1e-6)
        rate = done / elapsed
        pct = 100.0 * done / self.total
        bar_width = 24
        filled = min(bar_width, max(0, int(bar_width * done / self.total)))
        bar = "#" * filled + "-" * (bar_width - filled)
        message = (
            f"\r  [{self.label}] |{bar}| {done}/{self.total} "
            f"{pct:5.1f}% elapsed={elapsed:6.1f}s rate={rate:6.1f}/s "
            f"cache_dirs={cache_dirs}"
        )
        sys.stderr.write(message)
        if done == self.total:
            sys.stderr.write("\n")
        sys.stderr.flush()


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


def _float_or_none(value: Any, *, allow_zero: bool = False) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed > 0 or (allow_zero and parsed >= 0):
        return parsed
    return None


def _duration_from_record(
    record: dict[str, Any],
    n_frames: int,
    base_fps: float,
    explicit_duration: float | None = None,
) -> float:
    if explicit_duration is not None and explicit_duration > 0:
        return explicit_duration
    meta = record.get("metadata") or {}
    extraction = meta.get("offline_frame_extraction") or {}
    shared = meta.get("shared_source_frames") or {}
    if isinstance(shared, dict):
        start = _float_or_none(shared.get("segment_start_sec"), allow_zero=True)
        end = _float_or_none(shared.get("segment_end_sec"))
        if start is not None and end is not None and end > start:
            return end - start
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
    return n_frames / max(base_fps, 1e-6)


def _base_fps_from_record(record: dict[str, Any]) -> float:
    meta = record.get("metadata") or {}
    extraction = meta.get("offline_frame_extraction") or {}
    shared = meta.get("shared_source_frames") or {}
    for value in (
        extraction.get("effective_fps"),
        shared.get("cache_fps") if isinstance(shared, dict) else None,
        meta.get("video_fps_override"),
    ):
        fps = _float_or_none(value)
        if fps is not None:
            return fps
    return 2.0


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_path(path: str | Path) -> Path:
    return Path(str(path)).expanduser().resolve(strict=False)


def _normalize_cache_roots(cache_roots: list[str | Path] | tuple[str | Path, ...] | None) -> tuple[Path, ...]:
    if not cache_roots:
        return ()
    return tuple(_resolve_path(root) for root in cache_roots if str(root).strip())


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _cache_dir_allowed(frame_dir: Path, cache_roots: tuple[Path, ...]) -> bool:
    if not cache_roots:
        return True
    resolved = _resolve_path(frame_dir)
    return any(_is_relative_to(resolved, root) for root in cache_roots)


def default_frame_policy_cache_roots(data_root: str | Path) -> list[str]:
    """Trusted 2fps cache roots used by multi-task experiments.

    These are deliberately explicit so an old 1fps frame list cannot silently
    become the base source for a new experiment.
    """
    root = _resolve_path(data_root)
    return [
        str(root / "offline_frames" / "base_cache_2fps"),
        str(root.parent / "hier_seg_annotation_v1" / "frame_cache" / "source_2fps"),
    ]


def parse_cache_roots(cache_roots: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if cache_roots is None:
        return []
    if isinstance(cache_roots, str):
        raw_parts = cache_roots.replace(",", ":").split(":")
    else:
        raw_parts = list(cache_roots)
    return [part.strip() for part in raw_parts if str(part).strip()]


def _frame_path(frame_dir: Path, prefix: str, first_index: int, zero_based_index: int) -> Path:
    return frame_dir / f"{prefix}{first_index + zero_based_index:06d}.jpg"


def _detect_frame_naming(frame_dir: Path) -> tuple[str, int] | None:
    for prefix, first_index in (("", 1), ("", 0), ("frame_", 1), ("frame_", 0)):
        if _frame_path(frame_dir, prefix, first_index, 0).is_file():
            return prefix, first_index
    return None


def _read_meta_n_frames(frame_dir: Path) -> int | None:
    meta_path = frame_dir / "meta.json"
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = _float_or_none(meta.get("n_frames"))
    if value is None:
        return None
    return int(value)


def _infer_dense_frame_count(frame_dir: Path, prefix: str, first_index: int) -> int:
    """Infer count for dense ffmpeg frame dumps without listing the directory."""
    if not _frame_path(frame_dir, prefix, first_index, 0).is_file():
        return 0

    low = 0
    high = 1
    while _frame_path(frame_dir, prefix, first_index, high - 1).is_file():
        low = high
        high *= 2

    while low < high:
        mid = (low + high + 1) // 2
        if _frame_path(frame_dir, prefix, first_index, mid - 1).is_file():
            low = mid
        else:
            high = mid - 1
    return low


def _frame_dir_info_from_dir(
    frame_dir: str | Path,
    context: FramePolicyContext,
) -> tuple[FrameDirInfo | None, str]:
    frame_dir_path = Path(str(frame_dir))
    if not frame_dir_path.is_absolute():
        return None, f"frame_dir is not absolute: {frame_dir_path}"
    if not _cache_dir_allowed(frame_dir_path, context.cache_roots):
        roots = ", ".join(str(root) for root in context.cache_roots)
        return None, f"frame_dir outside trusted 2fps cache roots: {frame_dir_path} not in [{roots}]"
    resolved_dir = _resolve_path(frame_dir_path)
    cached = context.frame_dir_cache.get(resolved_dir)
    if cached is not None:
        return cached, ""
    if not resolved_dir.is_dir():
        return None, f"frame_dir not found: {resolved_dir}"

    naming = _detect_frame_naming(resolved_dir)
    if naming is None:
        return None, f"frame_dir has no jpg frames: {resolved_dir}"
    prefix, first_index = naming

    n_frames = _read_meta_n_frames(resolved_dir)
    if n_frames is None or n_frames <= 0 or not _frame_path(resolved_dir, prefix, first_index, n_frames - 1).is_file():
        n_frames = _infer_dense_frame_count(resolved_dir, prefix, first_index)
    if n_frames <= 0:
        return None, f"unable to infer frame count: {resolved_dir}"

    info = FrameDirInfo(path=resolved_dir, n_frames=n_frames, prefix=prefix, first_index=first_index)
    context.frame_dir_cache[resolved_dir] = info
    return info, ""


def _offline_cache_sources(
    record: dict[str, Any],
    context: FramePolicyContext,
) -> tuple[list[FrameVideoSource] | None, str]:
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

    cache_videos: list[FrameVideoSource] = []
    for idx, video_meta in enumerate(videos_meta):
        if not isinstance(video_meta, dict):
            return None, f"offline_frame_extraction.videos[{idx}] is not an object"
        frame_dir = video_meta.get("frame_dir")
        if not frame_dir:
            return None, f"offline_frame_extraction.videos[{idx}].frame_dir missing"
        frame_dir_info, reason = _frame_dir_info_from_dir(frame_dir, context)
        if frame_dir_info is None:
            return None, reason
        indices = list(range(frame_dir_info.n_frames))
        cache_videos.append(
            FrameVideoSource(
                frame_dir=frame_dir_info,
                source_indices=indices,
                base_fps=base_fps,
                duration_sec=_duration_from_record(record, len(indices), base_fps),
                source="offline_frame_cache",
            )
        )
    return cache_videos, ""


def _frame_indices_for_span(n_frames: int, source_fps: float, start_sec: float, end_sec: float) -> list[int]:
    if n_frames <= 0:
        return []
    if end_sec <= start_sec:
        return [max(0, min(n_frames - 1, int(math.floor(start_sec * max(source_fps, 1e-6)))))]

    eps = 1e-6
    start_idx = max(0, int(math.ceil(start_sec * source_fps - eps)))
    end_exclusive = min(n_frames, int(math.ceil(end_sec * source_fps - eps)))
    if end_exclusive <= start_idx:
        end_exclusive = min(n_frames, start_idx + 1)
    return list(range(start_idx, end_exclusive))


def _shared_source_cache_sources(
    record: dict[str, Any],
    context: FramePolicyContext,
) -> tuple[list[FrameVideoSource] | None, str]:
    meta = record.get("metadata") or {}
    shared = meta.get("shared_source_frames")
    if not isinstance(shared, dict):
        return None, "shared_source_frames missing"

    cache_dir = shared.get("cache_dir")
    if not cache_dir:
        return None, "shared_source_frames.cache_dir missing"
    cache_fps = _float_or_none(shared.get("cache_fps"))
    if cache_fps is None:
        return None, "shared_source_frames.cache_fps missing"
    if abs(cache_fps - 2.0) > 1e-6:
        return None, f"shared_source_frames.cache_fps is {cache_fps}, expected 2.0"
    start_sec = _float_or_none(shared.get("segment_start_sec"), allow_zero=True)
    end_sec = _float_or_none(shared.get("segment_end_sec"))
    if start_sec is None or end_sec is None or end_sec <= start_sec:
        return None, "shared_source_frames segment bounds invalid"

    frame_dir_info, reason = _frame_dir_info_from_dir(cache_dir, context)
    if frame_dir_info is None:
        return None, reason
    indices = _frame_indices_for_span(frame_dir_info.n_frames, cache_fps, start_sec, end_sec)
    if not indices:
        return None, "shared_source_frames selected no frames"
    return [
        FrameVideoSource(
            frame_dir=frame_dir_info,
            source_indices=indices,
            base_fps=cache_fps,
            duration_sec=end_sec - start_sec,
            source="shared_source_cache",
        )
    ], ""


def _cache_dir_from_frame_list(frames: list[str], context: FramePolicyContext) -> tuple[Path | None, str]:
    for frame in frames:
        path = Path(frame)
        if not path.is_absolute():
            continue
        parent = path.parent
        if _cache_dir_allowed(parent, context.cache_roots):
            return parent, ""
    if context.cache_roots:
        return None, "frame list has no absolute frame under trusted 2fps cache roots"
    return None, "frame list has no absolute frame path"


def _inferred_cache_sources_from_frame_list(
    record: dict[str, Any],
    context: FramePolicyContext,
) -> tuple[list[FrameVideoSource] | None, str]:
    videos = record.get("videos") or []
    if not isinstance(videos, list) or not videos:
        return None, "videos missing"

    frame_videos: list[FrameVideoSource] = []
    for idx, frames in enumerate(videos):
        if not isinstance(frames, list) or not frames:
            return None, f"videos[{idx}] is not a frame list"
        if not all(isinstance(frame, str) and frame for frame in frames):
            return None, f"videos[{idx}] contains non-string frame paths"
        cache_dir, reason = _cache_dir_from_frame_list(list(frames), context)
        if cache_dir is None:
            return None, reason
        frame_dir_info, reason = _frame_dir_info_from_dir(cache_dir, context)
        if frame_dir_info is None:
            return None, reason
        base_fps = 2.0
        indices = list(range(frame_dir_info.n_frames))
        frame_videos.append(
            FrameVideoSource(
                frame_dir=frame_dir_info,
                source_indices=indices,
                base_fps=base_fps,
                duration_sec=_duration_from_record(record, len(indices), base_fps),
                source="inferred_2fps_cache",
            )
        )
    return frame_videos, ""


def _frame_sources_from_record(
    record: dict[str, Any],
    context: FramePolicyContext,
) -> tuple[list[FrameVideoSource] | None, str]:
    reasons: list[str] = []
    for loader in (_offline_cache_sources, _shared_source_cache_sources, _inferred_cache_sources_from_frame_list):
        sources, reason = loader(record, context)
        if sources is not None:
            return sources, ""
        reasons.append(reason)
    return None, "; ".join(reasons)


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


def _downsample_indices_by_fps(indices: list[int], base_fps: float, target_fps: float | None) -> list[int]:
    if target_fps is None or target_fps >= base_fps:
        return list(indices)
    target_count = max(1, math.ceil(len(indices) * target_fps / base_fps))
    return [indices[pos] for pos in _uniform_indices(len(indices), target_count)]


def _cap_uniform_indices(indices: list[int], max_frames: int) -> list[int]:
    if max_frames <= 0 or len(indices) <= max_frames:
        return list(indices)
    return [indices[pos] for pos in _uniform_indices(len(indices), max_frames)]


def apply_frame_policy_to_record(
    record: dict[str, Any],
    rules: list[FramePolicyRule],
    max_frames: int,
    policy: str = "",
    cache_roots: list[str | Path] | tuple[str | Path, ...] | None = None,
    context: FramePolicyContext | None = None,
) -> dict[str, Any]:
    if context is None:
        context = FramePolicyContext(
            cache_roots=_normalize_cache_roots(cache_roots),
            frame_dir_cache={},
        )
    frame_sources, _skip_reason = _frame_sources_from_record(record, context)
    if frame_sources is None:
        return record

    n_videos = len(frame_sources)
    max_frames_per_video = max(1, max_frames // n_videos) if max_frames > 0 and n_videos > 1 else max_frames
    rewritten = copy.deepcopy(record)
    rewritten_videos: list[list[str]] = []
    policy_meta: list[dict[str, Any]] = []

    for source in frame_sources:
        indices = source.source_indices
        base_fps = source.base_fps
        duration_sec = source.duration_sec
        rule = _rule_for_duration(rules, duration_sec)
        target_fps = rule.fps if rule is not None else None
        after_fps = _downsample_indices_by_fps(list(indices), base_fps, target_fps)
        final_indices = _cap_uniform_indices(after_fps, max_frames_per_video)
        final_frames = [source.frame_dir.path_for_index(idx) for idx in final_indices]
        rewritten_videos.append(final_frames)
        policy_meta.append(
            {
                "duration_sec": duration_sec,
                "base_fps": base_fps,
                "target_fps": target_fps,
                "max_frames": max_frames_per_video,
                "input_frames": len(indices),
                "after_fps_frames": len(after_fps),
                "output_frames": len(final_frames),
                "source": source.source,
                "frame_dir": str(source.frame_dir.path),
                "frame_count_method": "indexed",
            }
        )

    meta = dict(rewritten.get("metadata") or {})
    meta["experiment_frame_sampling"] = {
        "policy": policy,
        "max_frames": max_frames,
        "implementation_version": FRAME_POLICY_IMPLEMENTATION_VERSION,
        "trusted_cache_roots": [str(root) for root in context.cache_roots],
        "rules": [
            {
                "min_sec": rule.min_sec,
                "max_sec": None if math.isinf(rule.max_sec) else rule.max_sec,
                "fps": rule.fps,
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
    cache_roots: list[str | Path] | tuple[str | Path, ...] | None = None,
    progress_label: str = "",
    progress_interval: int = 1000,
) -> list[dict[str, Any]]:
    rules = parse_frame_policy(policy)
    if not rules and max_frames <= 0:
        return records
    context = FramePolicyContext(
        cache_roots=_normalize_cache_roots(cache_roots),
        frame_dir_cache={},
    )
    progress = FramePolicyProgress(
        label=progress_label,
        total=len(records),
        interval=progress_interval,
    )
    rewritten: list[dict[str, Any]] = []
    for idx, record in enumerate(records, start=1):
        rewritten.append(apply_frame_policy_to_record(record, rules, max_frames, policy, context=context))
        progress.maybe_update(idx, cache_dirs=len(context.frame_dir_cache))
    return rewritten


def summarize_frame_policy_application(records: list[dict[str, Any]]) -> dict[str, Any]:
    applied = 0
    skipped = 0
    sources: Counter[str] = Counter()
    for record in records:
        meta = record.get("metadata") or {}
        if "experiment_frame_sampling" in meta:
            applied += 1
            for video_meta in (meta.get("experiment_frame_sampling") or {}).get("videos", []):
                if isinstance(video_meta, dict):
                    sources[str(video_meta.get("source") or "unknown")] += 1
        else:
            skipped += 1
    return {"applied": applied, "skipped": skipped, "sources": dict(sources)}
