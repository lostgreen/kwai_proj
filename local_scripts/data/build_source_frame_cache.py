#!/usr/bin/env python3
"""Prebuild a shared source-video frame cache from annotation JSONs."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from proxy_data.shared.frame_cache import SourceVideoInfo, ensure_source_frame_cache
from proxy_data.shared.seg_source import load_annotations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-dir", required=True, help="Annotation JSON directory")
    parser.add_argument("--frames-root", required=True, help="Shared source frame cache root")
    parser.add_argument("--cache-fps", type=float, default=2.0, help="Canonical physical cache fps")
    parser.add_argument("--jpeg-quality", type=int, default=2, help="ffmpeg JPEG quality scale (lower is better)")
    parser.add_argument("--workers", type=int, default=8, help="Parallel source-cache workers")
    parser.add_argument("--overwrite", action="store_true", help="Re-extract source caches")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames_root = Path(args.frames_root)
    frames_root.mkdir(parents=True, exist_ok=True)

    infos: dict[str, SourceVideoInfo] = {}
    for ann in load_annotations(args.annotation_dir, complete_only=False):
        clip_key = str(ann.get("clip_key", "")).strip()
        source_video_path = ann.get("source_video_path") or ann.get("video_path") or ""
        if not clip_key or not source_video_path:
            continue
        infos.setdefault(
            clip_key,
            SourceVideoInfo(
                clip_key=clip_key,
                source_video_path=str(source_video_path),
                duration_sec=ann.get("clip_duration_sec"),
            ),
        )

    print(f"[source-cache] unique source clips: {len(infos)}")
    print(f"[source-cache] cache root: {frames_root}")
    print(f"[source-cache] cache fps: {args.cache_fps}")

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_map = {
            executor.submit(
                ensure_source_frame_cache,
                info,
                frames_root,
                args.cache_fps,
                args.jpeg_quality,
                args.overwrite,
            ): clip_key
            for clip_key, info in infos.items()
        }
        for done_idx, future in enumerate(as_completed(future_map), start=1):
            future.result()
            if done_idx % 50 == 0 or done_idx == len(future_map):
                print(
                    f"[source-cache] ready {done_idx}/{len(future_map)}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
