#!/usr/bin/env python3
"""
extract_frames.py — Extract frames from video clips for hierarchical annotation.

Two operating modes:

Mode 1 — Full-clip extraction (L1+L2 annotation, default):
  Extracts 1fps JPEG frames from each clip. Supports:
    - windowed_clip:       directly use JSONL video paths
    - full_video_prefix:   scan original video dir, extract 0→clip_end
    - full_video:          scan original video dir, extract entire video

  Output layout:
    frames/{clip_key}/
        0001.jpg   ← frame at t=1s relative to clip start
        0002.jpg
        ...
        meta.json

Mode 2 — L3 per-event extraction (--annotation-dir):
  After running `annotate.py --level merged`, use this mode to extract
  per-event frames at higher fps (default 2fps) for L3 annotation.
  Reads merged annotation JSONs, extracts frames for each L2 event
  from the original video at the event's absolute time range.

  Output layout:
    frames_l3/{clip_key}_ev{event_id}/
        0001.jpg   ← frame 1 within the event (0-based relative)
        0002.jpg
        ...
        meta.json  ← includes event_start_sec for absolute timestamp recovery

Usage:
    # Mode 1: Full-clip
    python extract_frames.py \\
        --original-video-root /path/to/videos \\
        --output-dir frames/ \\
        --fps 1 --workers 8

    # Mode 2: L3 per-event (run after annotate.py --level merged)
    python extract_frames.py \\
        --annotation-dir annotations/ \\
        --original-video-root /path/to/videos \\
        --output-dir frames_l3/ \\
        --fps 2 --workers 8
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


WINDOWED_FILE_RE = re.compile(r"^(?P<video_id>.+?)_(?P<clip_start>\d+)_(?P<clip_end>\d+)$")


def parse_record_temporal_info(record: dict) -> dict[str, float | str | None]:
    """从 JSONL record 中解析 video_id / clip_start / clip_end。"""
    meta = record.get("metadata") or {}
    videos = record.get("videos") or []
    original_path = videos[0] if videos else ""

    video_id = meta.get("video_id")
    clip_start = meta.get("clip_start")
    clip_end = meta.get("clip_end")
    original_duration = meta.get("original_duration")

    stem = Path(original_path).stem
    match = WINDOWED_FILE_RE.match(stem)
    if match:
        video_id = video_id or match.group("video_id")
        clip_start = clip_start if clip_start is not None else match.group("clip_start")
        clip_end = clip_end if clip_end is not None else match.group("clip_end")

    def _to_number(value, default=None):
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    clip_start = _to_number(clip_start, 0.0)
    clip_end = _to_number(clip_end)
    original_duration = _to_number(original_duration)

    if clip_end is None:
        clip_duration = _to_number(meta.get("clip_duration"))
        if clip_duration is not None:
            clip_end = clip_duration

    return {
        "video_id": video_id,
        "clip_start": clip_start,
        "clip_end": clip_end,
        "original_duration": original_duration,
        "record_video_path": original_path,
    }


def build_original_video_index(original_video_root: str) -> dict[str, str]:
    """递归索引原始视频目录，key 为 video_id（文件 stem）。"""
    root = Path(original_video_root)
    if not root.exists():
        raise FileNotFoundError(f"original video root not found: {root}")

    index: dict[str, str] = {}
    duplicates: list[str] = []
    for path in root.rglob("*.mp4"):
        stem = path.stem
        prev = index.get(stem)
        if prev is not None and prev != str(path):
            duplicates.append(stem)
            continue
        index[stem] = str(path)

    if duplicates:
        dup_preview = ", ".join(sorted(set(duplicates))[:10])
        raise RuntimeError(
            f"duplicate video ids found under {root}: {dup_preview}"
        )

    return index


def probe_video_duration(video_path: Path, timeout: int = 10) -> float | None:
    """读取视频时长；不可读时返回 None。"""
    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            return None
        value = (proc.stdout or "").strip()
        return float(value) if value else None
    except Exception:
        return None


def build_records_from_original_videos(original_video_index: dict[str, str]) -> list[dict]:
    """直接从原始视频目录构造全视频标注 records。"""
    records: list[dict] = []
    for video_id, path in sorted(original_video_index.items()):
        duration = probe_video_duration(Path(path))
        records.append({
            "videos": [path],
            "metadata": {
                "clip_key": video_id,
                "video_id": video_id,
                "clip_start": 0,
                "clip_end": duration,
                "clip_duration": duration,
                "original_duration": duration,
                "is_full_video": True,
            },
        })
    return records


def resolve_video_path(original_path: str, video_dir: str | None) -> Path:
    """
    Resolve the actual video file path.

    If `video_dir` is given, replace the original directory component with it.
    Otherwise use the original path as-is.
    """
    p = Path(original_path)
    if video_dir:
        return Path(video_dir) / p.name
    return p


def resolve_original_video_path(
    record: dict,
    original_video_root: str | None,
    original_video_index: dict[str, str] | None,
    video_dir: str | None,
) -> tuple[Path, dict[str, float | str | None], str]:
    """
    解析用于抽帧的视频路径。

    返回:
        (video_path, temporal_info, source_mode)
    """
    temporal_info = parse_record_temporal_info(record)
    original_path = temporal_info["record_video_path"] or ""

    if original_video_root:
        if not original_video_index:
            raise ValueError("original_video_index is required when original_video_root is set")
        video_id = temporal_info.get("video_id")
        if not video_id:
            raise ValueError(f"cannot resolve video_id from record path: {original_path}")
        resolved = original_video_index.get(str(video_id))
        if resolved is None:
            raise FileNotFoundError(
                f"original video not found for video_id={video_id} under {original_video_root}"
            )
        source_mode = "full_video" if (record.get("metadata") or {}).get("is_full_video") else "full_video_prefix"
        return Path(resolved), temporal_info, source_mode

    return resolve_video_path(str(original_path), video_dir), temporal_info, "windowed_clip"


def extract_clip_frames(
    video_path: Path,
    output_dir: Path,
    fps: float = 1.0,
    max_frames: int = 0,
    duration_sec: float | None = None,
) -> list[Path]:
    """
    Use ffmpeg to extract frames at `fps` from `video_path`.

    Args:
        video_path:  Path to the video file.
        output_dir:  Directory to write JPEG frames into.
        fps:         Frames per second (default 1.0).
        max_frames:  If > 0, stop after this many frames (0 = no limit).
    Returns:
        Sorted list of extracted frame paths.
    Raises:
        RuntimeError: If ffmpeg exits with a non-zero code.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(output_dir / "%04d.jpg")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
    ]
    if duration_sec is not None and duration_sec > 0:
        cmd += ["-t", f"{duration_sec:.3f}"]
    cmd += [
        "-vf", f"fps={fps}",
        "-q:v", "2",        # JPEG quality
    ]
    if max_frames > 0:
        cmd += ["-frames:v", str(max_frames)]
    cmd += [pattern]

    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {video_path}:\n{result.stderr[-500:]}"
        )

    frames = sorted(output_dir.glob("*.jpg"))
    if not frames:
        raise RuntimeError(f"no frames extracted from {video_path}")
    return frames


def clip_key(video_path: str) -> str:
    """Return 'videoId_start_end' from a windowed clip filename."""
    return Path(video_path).stem


def count_jpg_frames(output_dir: Path) -> int:
    return len(list(output_dir.glob("*.jpg")))


def write_meta(output_dir: Path, meta: dict) -> None:
    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def process_record(
    record: dict,
    video_dir: str | None,
    original_video_root: str | None,
    original_video_index: dict[str, str] | None,
    output_base: Path,
    fps: float,
    max_frames: int,
    min_frames: int,
    overwrite: bool,
) -> dict:
    """
    Extract frames for a single JSONL record.

    Returns a status dict: {clip_key, n_frames, skipped, filtered, error}.
    """
    videos = record.get("videos") or []
    if not videos:
        return {
            "clip_key": "?",
            "n_frames": 0,
            "skipped": False,
            "filtered": True,
            "error": "no videos field",
        }

    orig_path = videos[0]
    key = str((record.get("metadata") or {}).get("clip_key") or clip_key(orig_path))
    out_dir = output_base / key

    if not overwrite and out_dir.exists() and count_jpg_frames(out_dir) > 0:
        n = count_jpg_frames(out_dir)
        if min_frames <= 0 or n >= min_frames:
            return {
                "clip_key": key,
                "n_frames": n,
                "skipped": True,
                "filtered": False,
                "error": None,
            }

    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir, ignore_errors=True)

    try:
        video_path, temporal_info, source_mode = resolve_original_video_path(
            record=record,
            original_video_root=original_video_root,
            original_video_index=original_video_index,
            video_dir=video_dir,
        )
    except (ValueError, FileNotFoundError) as e:
        return {
            "clip_key": key,
            "n_frames": 0,
            "skipped": False,
            "filtered": True,
            "error": str(e),
        }

    if not video_path.exists():
        return {
            "clip_key": key,
            "n_frames": 0,
            "skipped": False,
            "filtered": True,
            "error": f"video not found: {video_path}",
        }

    duration_sec = None
    if source_mode == "full_video_prefix":
        clip_end = temporal_info.get("clip_end")
        if clip_end is None or float(clip_end) <= 0:
            return {
                "clip_key": key,
                "n_frames": 0,
                "skipped": False,
                "filtered": True,
                "error": f"invalid clip_end for record: {orig_path}",
            }
        duration_sec = float(clip_end)
    elif source_mode == "full_video":
        duration_sec = temporal_info.get("clip_end")

    try:
        frames = extract_clip_frames(
            video_path,
            out_dir,
            fps=fps,
            max_frames=max_frames,
            duration_sec=duration_sec,
        )
        if min_frames > 0 and len(frames) < min_frames:
            shutil.rmtree(out_dir, ignore_errors=True)
            return {
                "clip_key": key,
                "n_frames": len(frames),
                "skipped": False,
                "filtered": True,
                "error": f"too few frames: {len(frames)} < min_frames({min_frames})",
            }

        write_meta(out_dir, {
            "clip_key": key,
            "source_mode": source_mode,
            "record_video_path": orig_path,
            "source_video_path": str(video_path),
            "video_id": temporal_info.get("video_id"),
            "window_start_sec": temporal_info.get("clip_start"),
            "window_end_sec": temporal_info.get("clip_end"),
            "annotation_start_sec": 0.0 if source_mode in {"full_video_prefix", "full_video"} else temporal_info.get("clip_start", 0.0),
            "annotation_end_sec": duration_sec if duration_sec is not None else temporal_info.get("clip_end"),
            "original_duration_sec": temporal_info.get("original_duration"),
            "fps": fps,
            "n_frames": len(frames),
        })
        return {
            "clip_key": key,
            "n_frames": len(frames),
            "skipped": False,
            "filtered": False,
            "error": None,
        }
    except RuntimeError as e:
        shutil.rmtree(out_dir, ignore_errors=True)
        return {
            "clip_key": key,
            "n_frames": 0,
            "skipped": False,
            "filtered": True,
            "error": str(e)[:200],
        }


def extract_l3_event_frames(
    source_video: Path,
    event_start_sec: float,
    event_end_sec: float,
    event_id: int,
    parent_clip_key: str,
    output_base: Path,
    fps: float = 2.0,
    overwrite: bool = False,
    min_frames: int = 2,
) -> dict:
    """Extract frames for one L2 event at higher fps for L3 annotation.

    Frames are numbered from 0001.jpg onward (relative to event_start).
    meta.json records event_start_sec so annotate.py can recover absolute timestamps.
    Output dir: {output_base}/{parent_clip_key}_ev{event_id}/
    """
    key = f"{parent_clip_key}_ev{event_id}"
    out_dir = output_base / key

    if not overwrite and out_dir.exists() and count_jpg_frames(out_dir) > 0:
        n = count_jpg_frames(out_dir)
        if min_frames <= 0 or n >= min_frames:
            return {"key": key, "n_frames": n, "skipped": True, "error": None}

    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir, ignore_errors=True)

    duration = event_end_sec - event_start_sec
    if duration <= 0:
        return {"key": key, "n_frames": 0, "skipped": False, "error": "zero-duration event"}

    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "%04d.jpg")

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{event_start_sec:.3f}",
        "-t", f"{duration:.3f}",
        "-i", str(source_video),
        "-vf", f"fps={fps}",
        "-q:v", "2",
        pattern,
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        shutil.rmtree(out_dir, ignore_errors=True)
        return {"key": key, "n_frames": 0, "skipped": False, "error": result.stderr[-300:]}

    frames = sorted(out_dir.glob("*.jpg"))
    n = len(frames)
    if min_frames > 0 and n < min_frames:
        shutil.rmtree(out_dir, ignore_errors=True)
        return {"key": key, "n_frames": n, "skipped": False,
                "error": f"too few frames: {n} < min_frames({min_frames})"}

    write_meta(out_dir, {
        "key": key,
        "parent_clip_key": parent_clip_key,
        "event_id": event_id,
        "event_start_sec": event_start_sec,
        "event_end_sec": event_end_sec,
        "fps": fps,
        "n_frames": n,
        "source_video_path": str(source_video),
    })
    return {"key": key, "n_frames": n, "skipped": False, "error": None}


def run_l3_extraction(
    annotation_dir: Path,
    output_base: Path,
    fps: float,
    workers: int,
    overwrite: bool,
    min_frames: int,
    video_dir: str | None,
    original_video_index: dict[str, str] | None,
    limit: int,
) -> None:
    """Extract per-event L3 frames from merged annotation JSONs.

    Reads every *.json in annotation_dir, collects all L2 events, and
    extracts frames at *fps* from the source video for each event.
    Output layout:  {output_base}/{clip_key}_ev{event_id}/
    """
    ann_paths = sorted(annotation_dir.glob("*.json"))
    if not ann_paths:
        print(f"ERROR: no annotation JSON files in {annotation_dir}", file=sys.stderr)
        sys.exit(1)

    if limit > 0:
        ann_paths = ann_paths[:limit]

    tasks: list[dict] = []
    for ann_path in ann_paths:
        try:
            with open(ann_path, encoding="utf-8") as f:
                ann = json.load(f)
        except Exception as e:
            print(f"WARN: skip {ann_path.name}: {e}")
            continue

        clip_key_str = ann.get("clip_key") or ann_path.stem
        source_video_str = ann.get("source_video_path") or ""

        # Resolve source video path
        if original_video_index:
            resolved = original_video_index.get(clip_key_str)
            if resolved:
                source_video_str = resolved
        if video_dir and source_video_str:
            candidate = Path(video_dir) / Path(source_video_str).name
            if candidate.exists():
                source_video_str = str(candidate)

        source_video = Path(source_video_str)
        if not source_video.exists():
            print(f"WARN: source video not found for {clip_key_str}: {source_video_str}")
            continue

        events = (ann.get("level2") or {}).get("events") or []
        if not events:
            print(f"WARN: no L2 events in {ann_path.name}, skipping")
            continue

        for ev in events:
            ev_id = ev.get("event_id")
            start = ev.get("start_time")
            end = ev.get("end_time")
            if not (isinstance(ev_id, int) and isinstance(start, (int, float))
                    and isinstance(end, (int, float)) and start < end):
                continue
            tasks.append({
                "source_video": source_video,
                "event_start_sec": float(start),
                "event_end_sec": float(end),
                "event_id": ev_id,
                "parent_clip_key": clip_key_str,
            })

    print(f"L3 per-event extraction: {len(tasks)} events from {len(ann_paths)} clips → {output_base}")
    print(f"FPS={fps}  workers={workers}  min_frames={min_frames}")
    output_base.mkdir(parents=True, exist_ok=True)

    done = skipped = errors = 0

    def _task(t: dict) -> dict:
        return extract_l3_event_frames(
            source_video=t["source_video"],
            event_start_sec=t["event_start_sec"],
            event_end_sec=t["event_end_sec"],
            event_id=t["event_id"],
            parent_clip_key=t["parent_clip_key"],
            output_base=output_base,
            fps=fps,
            overwrite=overwrite,
            min_frames=min_frames,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_task, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            if res.get("skipped"):
                skipped += 1
            elif res.get("error"):
                errors += 1
                print(f"[{i}/{len(tasks)}] ERROR  {res['key']}: {res['error']}")
            else:
                done += 1
                if i % 100 == 0 or i == len(tasks):
                    print(f"[{i}/{len(tasks)}] done={done} skipped={skipped} errors={errors}")

    print(f"\nDone: {done} extracted, {skipped} skipped, {errors} errors")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frames from video clips for annotation")
    parser.add_argument("--jsonl", default=None,
                        help="可选：输入 JSONL。若不提供，则直接扫描 --original-video-root 下的原视频。")
    parser.add_argument("--video-dir", default=None,
                        help="兼容旧模式：覆盖 JSONL 中的视频目录。"
                             "如果设置了 --original-video-root，则该参数会被忽略。")
    parser.add_argument("--original-video-root", default=None,
                        help="原始视频根目录。设置后会从原视频 00:00 抽到 metadata.clip_end。")
    parser.add_argument("--annotation-dir", default=None,
                        help="L3 per-event 模式：merged annotation JSON 目录（来自 annotate.py --level merged）。"
                             "设置后为每个 L2 event 独立抽帧，输出到 {output-dir}/{clip_key}_ev{N}/。"
                             "FPS 默认为 2.0，建议配合 --fps 2 使用。")
    parser.add_argument("--output-dir", required=True,
                        help="Root directory to write frame folders under")
    parser.add_argument("--fps", type=float, default=1.0,
                        help="Frames per second to extract (default: 1.0; use 2.0 for L3 per-event mode)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Max frames per clip (0 = no limit, ignored in L3 mode)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel ffmpeg workers (default: 4)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process at most N clips/annotations (0 = all)")
    parser.add_argument("--min-frames", type=int, default=16,
                        help="过滤低于该阈值的样本；<=0 表示不过滤 (L3 mode default: 2)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-extract even if output dir already has frames")
    args = parser.parse_args()

    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    # ── L3 per-event mode ──────────────────────────────────────────────────
    if args.annotation_dir:
        annotation_dir = Path(args.annotation_dir)
        if not annotation_dir.exists():
            print(f"ERROR: annotation-dir not found: {annotation_dir}", file=sys.stderr)
            sys.exit(1)

        fps = args.fps if args.fps != 1.0 else 2.0  # default 2fps for L3
        min_frames = args.min_frames if args.min_frames != 16 else 2

        original_video_index = None
        if args.original_video_root:
            print(f"Indexing original videos under {args.original_video_root} ...")
            original_video_index = build_original_video_index(args.original_video_root)
            print(f"Found {len(original_video_index)} original videos")

        run_l3_extraction(
            annotation_dir=annotation_dir,
            output_base=output_base,
            fps=fps,
            workers=args.workers,
            overwrite=args.overwrite,
            min_frames=min_frames,
            video_dir=args.video_dir,
            original_video_index=original_video_index,
            limit=args.limit,
        )
        return

    # ── Standard per-clip mode ─────────────────────────────────────────────
    original_video_index = None
    if args.original_video_root:
        print(f"Indexing original videos under {args.original_video_root} ...")
        original_video_index = build_original_video_index(args.original_video_root)
        print(f"Found {len(original_video_index)} original videos")

    records = []
    if args.jsonl:
        jsonl_path = Path(args.jsonl)
        if not jsonl_path.exists():
            print(f"ERROR: JSONL not found: {jsonl_path}", file=sys.stderr)
            sys.exit(1)

        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    elif args.original_video_root and original_video_index is not None:
        records = build_records_from_original_videos(original_video_index)
    else:
        print("ERROR: you must provide either --jsonl or --original-video-root", file=sys.stderr)
        sys.exit(1)

    if args.limit > 0:
        records = records[: args.limit]

    print(f"Processing {len(records)} clips → {output_base}")
    print(f"FPS={args.fps}  max_frames={args.max_frames or 'unlimited'}  workers={args.workers}")
    if args.original_video_root and not args.jsonl:
        mode = 'full_videos_from_root'
    elif args.original_video_root:
        mode = 'full_video_prefix'
    else:
        mode = 'windowed_clip'
    print(f"min_frames={args.min_frames}  mode={mode}")

    done = skipped = filtered = errors = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                process_record,
                rec,
                args.video_dir,
                args.original_video_root,
                original_video_index,
                output_base,
                args.fps,
                args.max_frames,
                args.min_frames,
                args.overwrite,
            ): rec
            for rec in records
        }
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            if res.get("filtered"):
                filtered += 1
                print(f"[{i}/{len(records)}] FILTER {res['clip_key']}: {res['error']}")
            elif res["error"]:
                errors += 1
                print(f"[{i}/{len(records)}] ERROR  {res['clip_key']}: {res['error']}")
            elif res["skipped"]:
                skipped += 1
                if i % 100 == 0:
                    print(f"[{i}/{len(records)}] (skip) {res['clip_key']} ({res['n_frames']} frames)")
            else:
                done += 1
                print(f"[{i}/{len(records)}] OK     {res['clip_key']} → {res['n_frames']} frames")

    print(f"\nDone: {done} extracted, {skipped} skipped, {filtered} filtered, {errors} errors")


if __name__ == "__main__":
    main()
