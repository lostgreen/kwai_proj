#!/usr/bin/env python3
"""Rewrite hier-seg JSONL to frame-list JSONL backed by a shared source cache.

Physical storage:
  - Extract each source video only once at a canonical cache fps (default 2fps)

Logical views:
  - Materialize each training record as a list of frames sliced from the shared
    source cache using annotation timestamps
  - L1 / L2-full can downsample the 2fps cache to a 1fps logical view without
    creating a second physical JPEG copy
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from proxy_data.shared.frame_cache import (
    SourceVideoInfo,
    ensure_source_frame_cache,
    load_cached_frames,
    probe_duration,
    select_frame_paths_for_span,
)
from proxy_data.shared.seg_source import load_annotations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, help="Hier-seg source JSONL (e.g. train_all.jsonl)")
    parser.add_argument("--output-jsonl", required=True, help="Rewritten frame-list JSONL")
    parser.add_argument("--annotation-dir", required=True, help="Annotation JSON directory for source-video lookup")
    parser.add_argument("--frames-root", required=True, help="Shared source frame cache root")
    parser.add_argument("--cache-fps", type=float, default=2.0, help="Canonical fps for the physical shared cache")
    parser.add_argument("--l1-view-fps", type=float, default=1.0, help="Logical fps for L1 frame lists")
    parser.add_argument("--l2-full-view-fps", type=float, default=1.0, help="Logical fps for L2 full-video frame lists")
    parser.add_argument("--default-view-fps", type=float, default=2.0, help="Logical fps for L2 phase / L3 frame lists")
    parser.add_argument("--jpeg-quality", type=int, default=2, help="ffmpeg JPEG quality scale (lower is better)")
    parser.add_argument("--workers", type=int, default=8, help="Parallel source-cache workers")
    parser.add_argument("--overwrite-cache", action="store_true", help="Re-extract source caches even if JPEGs already exist")
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


def build_annotation_lookup(annotation_dir: str | Path) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for ann in load_annotations(annotation_dir, complete_only=False):
        clip_key = str(ann.get("clip_key", "")).strip()
        if not clip_key:
            continue
        source_video = ann.get("source_video_path") or ann.get("video_path") or ""
        if not source_video:
            continue
        lookup[clip_key] = {
            "source_video_path": source_video,
            "clip_duration_sec": ann.get("clip_duration_sec"),
        }
    return lookup


def infer_clip_key(record: dict[str, Any]) -> str:
    meta = record.get("metadata") or {}
    clip_key = str(meta.get("clip_key", "")).strip()
    if clip_key:
        return clip_key

    videos = record.get("videos") or []
    if videos and isinstance(videos[0], str):
        name = Path(videos[0]).name
        if "_L1_" in name:
            return name.split("_L1_", 1)[0]
        if "_L2_" in name:
            return name.split("_L2_", 1)[0]
        if "_L3_" in name:
            return name.split("_L3_", 1)[0]
        return Path(videos[0]).stem
    raise ValueError("unable to infer clip_key from record")


def resolve_source_info(record: dict[str, Any], ann_lookup: dict[str, dict[str, Any]]) -> SourceVideoInfo:
    meta = record.get("metadata") or {}
    clip_key = infer_clip_key(record)
    ann_info = ann_lookup.get(clip_key, {})
    source_video_path = ann_info.get("source_video_path") or meta.get("source_video_path")
    duration_sec = ann_info.get("clip_duration_sec")
    if duration_sec in (None, "", 0):
        duration_sec = meta.get("clip_duration_sec")

    if not source_video_path:
        raise KeyError(f"missing source_video_path for clip_key={clip_key}")
    return SourceVideoInfo(
        clip_key=clip_key,
        source_video_path=str(source_video_path),
        duration_sec=float(duration_sec) if duration_sec not in (None, "") else None,
    )


def infer_segment_bounds(record: dict[str, Any], source_duration_sec: float | None) -> tuple[float, float]:
    meta = record.get("metadata") or {}
    level = meta.get("level")
    if level == "3s":
        level = 3

    clip_duration_sec = meta.get("clip_duration_sec")
    duration = float(clip_duration_sec) if clip_duration_sec not in (None, "") else source_duration_sec
    if duration is None or duration <= 0:
        duration = source_duration_sec

    if level == 1:
        if duration is None:
            raise ValueError("L1 record missing clip duration")
        return 0.0, float(duration)

    if level == 2 and meta.get("l2_mode") == "full":
        if duration is None:
            raise ValueError("L2 full record missing clip duration")
        return 0.0, float(duration)

    if level == 2 and meta.get("phase_start_sec") is not None and meta.get("phase_end_sec") is not None:
        return float(meta["phase_start_sec"]), float(meta["phase_end_sec"])

    if level == 3 and meta.get("clip_start_sec") is not None and meta.get("clip_end_sec") is not None:
        return float(meta["clip_start_sec"]), float(meta["clip_end_sec"])

    if level == 3 and meta.get("event_start_sec") is not None and meta.get("event_end_sec") is not None:
        return float(meta["event_start_sec"]), float(meta["event_end_sec"])

    raise ValueError(
        "unable to infer segment bounds "
        f"(clip_key={meta.get('clip_key')}, level={level}, keys={sorted(meta.keys())})"
    )


def infer_view_fps(record: dict[str, Any], args: argparse.Namespace) -> float:
    meta = record.get("metadata") or {}
    if meta.get("video_fps_override") not in (None, ""):
        return float(meta["video_fps_override"])

    level = meta.get("level")
    if level == "3s":
        level = 3

    if level == 1:
        return float(meta.get("l1_fps", args.l1_view_fps))
    if level == 2 and meta.get("l2_mode") == "full":
        return float(meta.get("l2_fps", args.l2_full_view_fps))
    return float(args.default_view_fps)


def rewrite_record(
    record: dict[str, Any],
    source_info: SourceVideoInfo,
    cache_dir: Path,
    cache_fps: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    cached_frames = load_cached_frames(cache_dir)
    if not cached_frames:
        raise RuntimeError(f"empty cache dir: {cache_dir}")

    source_duration = source_info.duration_sec
    if source_duration is None or source_duration <= 0:
        source_duration = probe_duration(source_info.source_video_path)

    start_sec, end_sec = infer_segment_bounds(record, source_duration)
    if source_duration is not None:
        start_sec = max(0.0, min(float(start_sec), float(source_duration)))
        end_sec = max(start_sec, min(float(end_sec), float(source_duration)))

    view_fps = infer_view_fps(record, args)
    frame_paths = select_frame_paths_for_span(
        frame_paths=cached_frames,
        source_fps=cache_fps,
        start_sec=start_sec,
        end_sec=end_sec,
        target_fps=view_fps,
    )
    if not frame_paths:
        raise RuntimeError(
            f"no frames selected for clip_key={source_info.clip_key} span=[{start_sec}, {end_sec})"
        )

    rewritten = copy.deepcopy(record)
    meta = dict(rewritten.get("metadata") or {})
    meta["source_video_path"] = source_info.source_video_path
    meta["video_fps_override"] = view_fps
    meta["shared_source_frames"] = {
        "cache_dir": str(cache_dir.resolve()),
        "cache_fps": cache_fps,
        "segment_start_sec": start_sec,
        "segment_end_sec": end_sec,
        "target_view_fps": view_fps,
        "n_frames": len(frame_paths),
    }
    rewritten["metadata"] = meta
    rewritten["videos"] = [[str(path.resolve()) for path in frame_paths]]
    return rewritten


def main() -> None:
    args = parse_args()
    input_jsonl = Path(args.input_jsonl)
    output_jsonl = Path(args.output_jsonl)
    frames_root = Path(args.frames_root)

    records = load_jsonl(input_jsonl)
    ann_lookup = build_annotation_lookup(args.annotation_dir)
    frames_root.mkdir(parents=True, exist_ok=True)

    record_source_infos: list[SourceVideoInfo] = []
    source_infos: dict[str, SourceVideoInfo] = {}
    for record in records:
        info = resolve_source_info(record, ann_lookup)
        record_source_infos.append(info)
        source_infos.setdefault(info.clip_key, info)

    print(f"[shared-frames] records: {len(records)}")
    print(f"[shared-frames] unique source clips: {len(source_infos)}")
    print(f"[shared-frames] cache root: {frames_root}")

    cache_dirs: dict[str, Path] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_map = {
            executor.submit(
                ensure_source_frame_cache,
                info,
                frames_root,
                args.cache_fps,
                args.jpeg_quality,
                args.overwrite_cache,
            ): clip_key
            for clip_key, info in source_infos.items()
        }
        for done_idx, future in enumerate(as_completed(future_map), start=1):
            clip_key = future_map[future]
            meta = future.result()
            cache_dirs[clip_key] = Path(meta["cache_dir"])
            if done_idx % 50 == 0 or done_idx == len(future_map):
                print(
                    f"[shared-frames] cache {done_idx}/{len(future_map)} ready",
                    flush=True,
                )

    rewritten: list[dict[str, Any]] = []
    for idx, (record, info) in enumerate(zip(records, record_source_infos), start=1):
        info = source_infos[info.clip_key]
        cache_dir = cache_dirs[info.clip_key]
        rewritten.append(rewrite_record(record, info, cache_dir, args.cache_fps, args))
        if idx % 200 == 0 or idx == len(records):
            print(f"[shared-frames] rewrite {idx}/{len(records)}", flush=True)

    write_jsonl(rewritten, output_jsonl)
    print(f"[shared-frames] wrote {output_jsonl}")


if __name__ == "__main__":
    main()
