#!/usr/bin/env python3
"""
detect_scenes.py — Detect shot boundaries using PySceneDetect.

Preprocessing step for the l2l3_first annotation pipeline.
Reads meta.json from each frame directory to find the source video,
runs scene detection, and writes scenes.json alongside the frames.

Usage:
    python detect_scenes.py \
        --frames-dir frames/ \
        --detector content \
        --threshold 20.0 \
        --workers 4
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def detect_scenes_for_clip(
    frame_dir: Path,
    overwrite: bool = False,
    detector_type: str = "content",
    threshold: float = 20.0,
    min_scene_len: int = 15,
) -> dict:
    """Detect scene boundaries for a single clip.

    Reads meta.json to find the source video, runs PySceneDetect,
    and writes scenes.json into frame_dir.

    Args:
        frame_dir:       Path to the clip's extracted frames directory.
        overwrite:       Re-detect even if scenes.json already exists.
        detector_type:   "content" (ContentDetector) or "adaptive" (AdaptiveDetector).
        threshold:       Detection threshold (20.0 for content, 3.0 for adaptive).
        min_scene_len:   Minimum scene length in source video frames.

    Returns:
        Status dict with clip_key, skipped, n_scenes, error.
    """
    scenes_path = frame_dir / "scenes.json"
    if not overwrite and scenes_path.exists():
        return {"clip_key": frame_dir.name, "skipped": True,
                "n_scenes": -1, "error": None}

    meta_path = frame_dir / "meta.json"
    if not meta_path.exists():
        return {"clip_key": frame_dir.name, "skipped": False,
                "n_scenes": 0, "error": "no meta.json"}

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    source_video = meta.get("source_video_path")
    if not source_video or not Path(source_video).exists():
        return {"clip_key": frame_dir.name, "skipped": False,
                "n_scenes": 0,
                "error": f"source video not found: {source_video}"}

    fps = float(meta.get("fps", 2.0))
    annotation_start = float(meta.get("annotation_start_sec", 0.0))
    annotation_end = meta.get("annotation_end_sec")

    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector, AdaptiveDetector
    except ImportError:
        return {"clip_key": frame_dir.name, "skipped": False,
                "n_scenes": 0,
                "error": "scenedetect not installed: pip install scenedetect[opencv]"}

    try:
        video = open_video(str(source_video))
        scene_manager = SceneManager()

        if detector_type == "adaptive":
            scene_manager.add_detector(AdaptiveDetector(
                adaptive_threshold=threshold,
                min_scene_len=min_scene_len,
            ))
        else:
            scene_manager.add_detector(ContentDetector(
                threshold=threshold,
                min_scene_len=min_scene_len,
            ))

        end_time = float(annotation_end) if annotation_end else None
        scene_manager.detect_scenes(video, end_time=end_time)
        scene_list = scene_manager.get_scene_list()
    except Exception as e:
        return {"clip_key": frame_dir.name, "skipped": False,
                "n_scenes": 0, "error": str(e)[:200]}

    # Extract boundary timestamps — the start of each scene after the first
    boundaries_sec: list[float] = []
    for i, (start, _end) in enumerate(scene_list):
        if i > 0:
            boundaries_sec.append(round(start.get_seconds(), 3))

    # Map to 1-based frame indices at the extraction fps
    boundary_frame_indices: list[int] = []
    for b_sec in boundaries_sec:
        adj_sec = b_sec - annotation_start
        if adj_sec <= 0:
            continue
        frame_idx = int(round(adj_sec * fps)) + 1  # 1-based
        boundary_frame_indices.append(frame_idx)

    scenes_data = {
        "clip_key": frame_dir.name,
        "detector": detector_type,
        "threshold": threshold,
        "min_scene_len": min_scene_len,
        "source_video_path": str(source_video),
        "n_scenes": len(scene_list),
        "boundary_timestamps_sec": boundaries_sec,
        "boundary_frame_indices": boundary_frame_indices,
    }

    with open(scenes_path, "w", encoding="utf-8") as f:
        json.dump(scenes_data, f, indent=2, ensure_ascii=False)

    return {"clip_key": frame_dir.name, "skipped": False,
            "n_scenes": len(scene_list), "error": None}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect scene boundaries using PySceneDetect",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--frames-dir", required=True,
                        help="Root directory of pre-extracted frames")
    parser.add_argument("--detector", choices=["content", "adaptive"],
                        default="content",
                        help="Scene detection algorithm")
    parser.add_argument("--threshold", type=float, default=20.0,
                        help="Detection threshold (20.0 for content, 3.0 for adaptive)")
    parser.add_argument("--min-scene-len", type=int, default=15,
                        help="Minimum scene length in source video frames")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel detection workers")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process at most N clips (0 = all)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-detect even if scenes.json exists")
    args = parser.parse_args()

    frames_base = Path(args.frames_dir)
    if not frames_base.exists():
        print(f"ERROR: frames-dir not found: {frames_base}", file=sys.stderr)
        sys.exit(1)

    frame_dirs = sorted(p for p in frames_base.iterdir() if p.is_dir())
    if args.limit > 0:
        frame_dirs = frame_dirs[:args.limit]

    print(f"Scene detection: {len(frame_dirs)} clips, "
          f"detector={args.detector}, threshold={args.threshold}", flush=True)

    ok = skipped = errors = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                detect_scenes_for_clip,
                fd, args.overwrite, args.detector,
                args.threshold, args.min_scene_len,
            ): fd
            for fd in frame_dirs
        }
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                res = fut.result()
            except Exception as exc:
                errors += 1
                fd = futures[fut]
                print(f"[{i}/{len(frame_dirs)}] CRASH {fd.name}: {exc}", flush=True)
                continue
            if res.get("skipped"):
                skipped += 1
            elif res.get("error"):
                errors += 1
                print(f"[{i}/{len(frame_dirs)}] ERROR {res['clip_key']}: "
                      f"{res['error']}", flush=True)
            else:
                ok += 1
                print(f"[{i}/{len(frame_dirs)}] {res['clip_key']}: "
                      f"{res['n_scenes']} scenes", flush=True)

    print(f"\nDone: {ok} detected, {skipped} skipped, {errors} errors", flush=True)


if __name__ == "__main__":
    main()
