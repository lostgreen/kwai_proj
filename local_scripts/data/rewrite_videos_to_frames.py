#!/usr/bin/env python3
"""Rewrite video JSONL into frame-list JSONL for offline video training.

The output JSONL keeps the original sample fields, but replaces:

    "videos": ["foo.mp4"]

with:

    "videos": [["clip_key/000001.jpg", "clip_key/000002.jpg", ...]]

and writes:

    metadata.video_fps_override = <extraction_fps>

This matches the current EasyR1 dataset loader, which already supports
``videos`` as a list of frame-path lists.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, help="Source training JSONL.")
    parser.add_argument("--output-jsonl", required=True, help="Rewritten frame-list JSONL.")
    parser.add_argument("--frames-root", required=True, help="Output frame root directory.")
    parser.add_argument(
        "--input-image-dir",
        default="",
        help="Prefix for relative input video paths. Leave empty if input JSONL stores absolute paths.",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=2.0,
        help="High-rate extraction fps for short videos.",
    )
    parser.add_argument(
        "--fallback-fps",
        type=float,
        default=1.0,
        help="Medium-rate extraction fps for videos too long for --target-fps but still within max_frames budget.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=256,
        help="Training max_frames. Extraction fps is capped to respect this budget.",
    )
    parser.add_argument(
        "--min-fps",
        type=float,
        default=0.25,
        help="Lower bound for auto-computed extraction fps.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel extraction workers.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=2,
        help="ffmpeg JPEG quality scale (lower is better).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-extract frames even if the output directory already exists.",
    )
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Write absolute frame paths into JSONL instead of paths relative to --frames-root.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def probe_duration(video_path: Path) -> float | None:
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
        return float(value)
    except ValueError:
        return None


def resolve_video_path(video: str, input_image_dir: Path | None) -> Path:
    path = Path(video)
    if path.is_absolute() or input_image_dir is None:
        return path
    return input_image_dir / path


def infer_clip_key(record: dict[str, Any], index: int) -> str:
    meta = record.get("metadata") or {}
    if meta.get("clip_key"):
        return str(meta["clip_key"])

    videos = record.get("videos") or []
    if videos and isinstance(videos[0], str):
        return Path(videos[0]).stem

    return f"sample_{index:08d}"


def build_cache_key(
    resolved_paths: list[Path],
    shared_fps: float,
    max_frames_per_video: int,
) -> str:
    """Build a stable cache key so identical clip files reuse extracted frames.

    We key by the resolved clip paths plus extraction policy. This deduplicates:
    - repeated use of the same clip across train / val
    - repeated use of the same clip across tasks / datasets

    It does NOT merge different overlapping clips from the same original video.
    That would require a separate full-video extraction pipeline.
    """
    joined = "||".join(str(path.resolve()) for path in resolved_paths)
    payload = f"{joined}__fps={shared_fps:.6f}__maxf={max_frames_per_video}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    stem = resolved_paths[0].stem if len(resolved_paths) == 1 else "multi_video"
    safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stem)[:80]
    return f"{safe_stem}__{digest}"


def infer_record_duration(record: dict[str, Any]) -> float | None:
    meta = record.get("metadata") or {}

    candidates = [
        meta.get("clip_duration"),
        record.get("duration"),
        meta.get("duration"),
    ]

    clip_start = meta.get("clip_start")
    clip_end = meta.get("clip_end")
    if clip_start is not None and clip_end is not None:
        try:
            span = float(clip_end) - float(clip_start)
            if span > 0:
                candidates.insert(0, span)
        except (TypeError, ValueError):
            pass

    for value in candidates:
        try:
            duration = float(value)
        except (TypeError, ValueError):
            continue
        if duration > 0:
            return duration

    return None


def compute_shared_fps(
    duration_sec: float | None,
    n_videos: int,
    target_fps: float,
    fallback_fps: float,
    max_frames: int,
    min_fps: float,
) -> float:
    if duration_sec is None or duration_sec <= 0:
        return round(target_fps, 3)

    max_frames_per_video = max(1, max_frames // max(n_videos, 1))
    high_fps_budget_sec = max_frames_per_video / max(target_fps, 1e-6)
    fallback_fps_budget_sec = max_frames_per_video / max(fallback_fps, 1e-6)

    if duration_sec <= high_fps_budget_sec:
        fps = target_fps
    elif duration_sec <= fallback_fps_budget_sec:
        fps = fallback_fps
    else:
        fps = max_frames_per_video / duration_sec

    fps = max(fps, min_fps)
    return round(float(fps), 3)


def extract_frames(
    video_path: Path,
    output_dir: Path,
    fps: float,
    max_frames: int,
    jpeg_quality: int,
    overwrite: bool,
) -> list[Path]:
    if output_dir.exists() and not overwrite:
        cached = sorted(output_dir.glob("*.jpg"))
        if cached:
            return cached

    if output_dir.exists() and overwrite:
        for path in output_dir.glob("*"):
            if path.is_file():
                path.unlink()

    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(output_dir / "%06d.jpg")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        "-q:v",
        str(jpeg_quality),
    ]
    if max_frames > 0:
        cmd += ["-frames:v", str(max_frames)]
    cmd += [pattern]

    proc = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-500:] if proc.stderr else f"ffmpeg failed for {video_path}")

    frames = sorted(output_dir.glob("*.jpg"))
    if not frames:
        raise RuntimeError(f"no frames extracted from {video_path}")
    return frames


def rewrite_record(
    record: dict[str, Any],
    index: int,
    frames_root: Path,
    input_image_dir: Path | None,
    target_fps: float,
    fallback_fps: float,
    max_frames: int,
    min_fps: float,
    jpeg_quality: int,
    overwrite: bool,
    absolute_paths: bool,
) -> dict[str, Any]:
    videos = record.get("videos") or []
    if not videos:
        raise ValueError("record has no videos")
    if isinstance(videos[0], list):
        raise ValueError("record already uses frame lists")

    clip_key = infer_clip_key(record, index)
    n_videos = len(videos)
    max_frames_per_video = max(1, max_frames // max(n_videos, 1))

    resolved_paths = [resolve_video_path(str(video), input_image_dir) for video in videos]
    for path in resolved_paths:
        if not path.exists():
            raise FileNotFoundError(f"video not found: {path}")

    duration = infer_record_duration(record)
    if duration is None and len(resolved_paths) == 1:
        duration = probe_duration(resolved_paths[0])
    if duration is None and len(resolved_paths) > 1:
        durations = [probe_duration(path) for path in resolved_paths]
        valid_durations = [d for d in durations if d is not None and d > 0]
        if valid_durations:
            duration = max(valid_durations)

    shared_fps = compute_shared_fps(
        duration_sec=duration,
        n_videos=n_videos,
        target_fps=target_fps,
        fallback_fps=fallback_fps,
        max_frames=max_frames,
        min_fps=min_fps,
    )

    rewritten = copy.deepcopy(record)
    rewritten_videos: list[list[str]] = []
    extraction_meta: list[dict[str, Any]] = []
    cache_key = build_cache_key(
        resolved_paths=resolved_paths,
        shared_fps=shared_fps,
        max_frames_per_video=max_frames_per_video,
    )
    for vid_idx, video_path in enumerate(resolved_paths):
        if n_videos == 1:
            out_dir = frames_root / cache_key
        else:
            out_dir = frames_root / cache_key / f"video_{vid_idx}"
        frames = extract_frames(
            video_path=video_path,
            output_dir=out_dir,
            fps=shared_fps,
            max_frames=max_frames_per_video,
            jpeg_quality=jpeg_quality,
            overwrite=overwrite,
        )
        if absolute_paths:
            frame_list = [str(path) for path in frames]
        else:
            frame_list = [str(path.relative_to(frames_root)) for path in frames]
        rewritten_videos.append(frame_list)
        extraction_meta.append(
            {
                "source_video_path": str(video_path),
                "frame_dir": str(out_dir if absolute_paths else out_dir.relative_to(frames_root)),
                "n_frames": len(frame_list),
            }
        )

    meta = dict(rewritten.get("metadata") or {})
    meta["video_fps_override"] = shared_fps
    meta["offline_frame_extraction"] = {
        "clip_key": clip_key,
        "cache_key": cache_key,
        "target_fps": target_fps,
        "fallback_fps": fallback_fps,
        "effective_fps": shared_fps,
        "max_frames": max_frames,
        "max_frames_per_video": max_frames_per_video,
        "n_videos": n_videos,
        "duration_sec": duration,
        "videos": extraction_meta,
    }
    rewritten["metadata"] = meta
    rewritten["videos"] = rewritten_videos
    return rewritten


def main() -> None:
    args = parse_args()
    input_jsonl = Path(args.input_jsonl)
    output_jsonl = Path(args.output_jsonl)
    frames_root = Path(args.frames_root)
    input_image_dir = Path(args.input_image_dir) if args.input_image_dir else None

    records = load_jsonl(input_jsonl)
    rewritten: list[dict[str, Any] | None] = [None] * len(records)
    failures: list[tuple[int, str]] = []

    frames_root.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_map = {
            executor.submit(
                rewrite_record,
                record,
                idx,
                frames_root,
                input_image_dir,
                args.target_fps,
                args.fallback_fps,
                args.max_frames,
                args.min_fps,
                args.jpeg_quality,
                args.overwrite,
                args.absolute_paths,
            ): idx
            for idx, record in enumerate(records)
        }

        for done_idx, future in enumerate(as_completed(future_map), start=1):
            idx = future_map[future]
            try:
                rewritten[idx] = future.result()
            except Exception as exc:
                failures.append((idx, str(exc)))
            if done_idx % 50 == 0 or done_idx == len(records):
                print(f"[rewrite_videos_to_frames] {done_idx}/{len(records)} done  failures={len(failures)}", flush=True)

    if failures:
        preview = "\n".join(f"  - idx={idx}: {msg}" for idx, msg in failures[:20])
        raise RuntimeError(
            f"failed to rewrite {len(failures)} / {len(records)} samples:\n{preview}"
        )

    write_jsonl([record for record in rewritten if record is not None], output_jsonl)
    print(f"[rewrite_videos_to_frames] wrote {output_jsonl}")
    print(f"[rewrite_videos_to_frames] frames root: {frames_root}")
    print("[rewrite_videos_to_frames] training should point data.image_dir to this frames root unless --absolute-paths was used")


if __name__ == "__main__":
    main()
