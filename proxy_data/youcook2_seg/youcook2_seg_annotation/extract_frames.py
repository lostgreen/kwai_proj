#!/usr/bin/env python3
"""
extract_frames.py — Extract 1fps frames for YouCook2 segmentation annotation.

默认兼容旧模式：直接从 JSONL 里的 windowed clip 抽帧。

推荐新模式：直接传入 `--original-video-root`，脚本会扫描原始 YouCook2
视频目录，对每个原视频从 00:00 开始抽完整视频帧，用于“从原视频开始重新标注”。

如果同时提供 JSONL，则会读取 JSONL 中的 `metadata.clip_end`，从原视频 00:00
抽到该样本对应的 `clip_end`，作为兼容过渡模式。

同时会自动过滤：
1. 不可读 / 缺失的视频
2. 抽帧后帧数低于阈值的视频

Usage:
    python /home/xuboshen/zgw/EasyR1/proxy_data/youcook2_seg_annotation/extract_frames.py \
        --original-video-root /m2v_intern/xuboshen/zgw/data/YouCook2_mp4 \
        --output-dir /m2v_intern/xuboshen/zgw/data/youcook2_seg_annotation/frames \
        --workers 8 \
        --limit 1000

Output layout:
    frames/
        {clip_key}/
            0001.jpg   ← frame at t=1s relative to clip start
            0002.jpg
            ...
            meta.json
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
    """递归索引原始 YouCook2 视频，key 为 video_id（文件 stem）。"""
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract 1fps frames from YC2 windowed clips")
    parser.add_argument("--jsonl", default=None,
                        help="可选：输入 JSONL。若不提供，则直接扫描 --original-video-root 下的原视频。")
    parser.add_argument("--video-dir", default=None,
                        help="兼容旧模式：覆盖 JSONL 中的视频目录。"
                             "如果设置了 --original-video-root，则该参数会被忽略。")
    parser.add_argument("--original-video-root", default=None,
                        help="原始 YouCook2 视频根目录，例如 /m2v_intern/.../YouCook2_mp4。"
                             "设置后会从原视频 00:00 抽到 metadata.clip_end。")
    parser.add_argument("--output-dir", required=True,
                        help="Root directory to write frame folders under")
    parser.add_argument("--fps", type=float, default=1.0,
                        help="Frames per second to extract (default: 1.0)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Max frames per clip (0 = no limit)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel ffmpeg workers (default: 4)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process at most N clips (0 = all)")
    parser.add_argument("--min-frames", type=int, default=16,
                        help="过滤低于该阈值的样本；<=0 表示不过滤")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-extract even if output dir already has frames")
    args = parser.parse_args()

    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

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
