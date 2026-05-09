#!/usr/bin/env python3
"""Shared source-video frame cache helpers.

This module builds a reusable frame cache from source videos so multiple
downstream tasks can materialize frame-list manifests without re-extracting
duplicated JPEGs for every segment clip.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceVideoInfo:
    clip_key: str
    source_video_path: str
    duration_sec: float | None = None


def _safe_name(text: str, max_len: int = 80) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)
    cleaned = cleaned.strip("_")
    return (cleaned or "clip")[:max_len]


def build_source_cache_dir(
    frames_root: str | Path,
    clip_key: str,
    source_video_path: str,
    fps: float,
) -> Path:
    frames_root = Path(frames_root)
    digest_src = f"{Path(source_video_path).resolve()}__fps={fps:.6f}"
    digest = hashlib.sha1(digest_src.encode("utf-8")).hexdigest()[:12]
    stem = _safe_name(clip_key or Path(source_video_path).stem)
    return frames_root / f"{stem}__{digest}"


def load_cached_frames(cache_dir: str | Path) -> list[Path]:
    return sorted(Path(cache_dir).glob("*.jpg"))


def read_cache_meta(cache_dir: str | Path) -> dict | None:
    meta_path = Path(cache_dir) / "meta.json"
    if not meta_path.is_file():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def write_cache_meta(cache_dir: str | Path, meta: dict) -> None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "meta.json"
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def probe_duration(video_path: str | Path) -> float | None:
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None

    if proc.returncode != 0:
        return None

    value = (proc.stdout or "").strip()
    if not value:
        return None

    try:
        duration = float(value)
    except ValueError:
        return None
    return duration if duration > 0 else None


def extract_source_frames(
    source_video_path: str | Path,
    cache_dir: str | Path,
    fps: float,
    jpeg_quality: int = 2,
    overwrite: bool = False,
) -> list[Path]:
    cache_dir = Path(cache_dir)
    cached = load_cached_frames(cache_dir)
    if cached and not overwrite:
        return cached

    if cache_dir.exists() and overwrite:
        for path in cache_dir.glob("*"):
            if path.is_file():
                path.unlink()

    cache_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(cache_dir / "%06d.jpg")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(source_video_path),
        "-vf",
        f"fps={fps}",
        "-q:v",
        str(jpeg_quality),
        pattern,
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            proc.stderr[-500:] if proc.stderr else f"ffmpeg failed for {source_video_path}"
        )

    frames = load_cached_frames(cache_dir)
    if not frames:
        raise RuntimeError(f"no frames extracted from {source_video_path}")
    return frames


def ensure_source_frame_cache(
    source_info: SourceVideoInfo,
    frames_root: str | Path,
    fps: float,
    jpeg_quality: int = 2,
    overwrite: bool = False,
) -> dict:
    cache_dir = build_source_cache_dir(
        frames_root=frames_root,
        clip_key=source_info.clip_key,
        source_video_path=source_info.source_video_path,
        fps=fps,
    )
    frames = extract_source_frames(
        source_video_path=source_info.source_video_path,
        cache_dir=cache_dir,
        fps=fps,
        jpeg_quality=jpeg_quality,
        overwrite=overwrite,
    )

    duration = source_info.duration_sec
    if duration is None or duration <= 0:
        duration = probe_duration(source_info.source_video_path)

    meta = {
        "clip_key": source_info.clip_key,
        "source_video_path": str(Path(source_info.source_video_path).resolve()),
        "cache_dir": str(cache_dir.resolve()),
        "fps": fps,
        "jpeg_quality": jpeg_quality,
        "duration_sec": duration,
        "n_frames": len(frames),
    }
    write_cache_meta(cache_dir, meta)
    return meta


def select_frame_indices(
    n_frames: int,
    source_fps: float,
    start_sec: float,
    end_sec: float,
) -> list[int]:
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


def downsample_indices(
    indices: list[int],
    source_fps: float,
    target_fps: float,
    start_sec: float,
    end_sec: float,
) -> list[int]:
    if not indices or target_fps <= 0 or target_fps >= source_fps - 1e-6:
        return indices

    step = source_fps / target_fps
    rounded_step = int(round(step))
    if rounded_step > 0 and abs(step - rounded_step) <= 1e-6:
        downsampled = indices[::rounded_step]
        return downsampled or [indices[0]]

    picked: list[int] = []
    t = start_sec
    eps = 1e-6
    max_idx = indices[-1]
    min_idx = indices[0]
    while t < end_sec - eps:
        idx = int(round(t * source_fps))
        idx = max(min_idx, min(max_idx, idx))
        if idx in indices and (not picked or idx != picked[-1]):
            picked.append(idx)
        t += 1.0 / target_fps
    return picked or [indices[0]]


def select_frame_paths_for_span(
    frame_paths: list[str | Path],
    source_fps: float,
    start_sec: float,
    end_sec: float,
    target_fps: float | None = None,
) -> list[Path]:
    paths = [Path(p) for p in frame_paths]
    indices = select_frame_indices(len(paths), source_fps, start_sec, end_sec)
    if target_fps is not None:
        indices = downsample_indices(indices, source_fps, target_fps, start_sec, end_sec)
    return [paths[idx] for idx in indices]
