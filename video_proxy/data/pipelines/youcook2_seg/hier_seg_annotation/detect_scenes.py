#!/usr/bin/env python3
"""
detect_scenes.py — Detect shot boundaries using TransNetV2 or PySceneDetect.

Preprocessing step for the scene-first annotation pipeline.
Reads meta.json from each frame directory to find the source video,
runs scene detection, and writes scenes.json alongside the frames.

Usage:
    # TransNetV2 (default, more accurate for gradual transitions / montage):
    python detect_scenes.py \
        --frames-dir frames/ \
        --detector transnet \
        --workers 4

    # PySceneDetect ContentDetector (lighter, no TF dependency):
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


# ─────────────────────────────────────────────────────────────────────────────
# TransNetV2 detection
# ─────────────────────────────────────────────────────────────────────────────

def _detect_transnet(
    source_video: str,
    annotation_start: float,
    annotation_end: float | None,
    fps: float,
    transnet_model,
    threshold: float = 0.5,
) -> tuple[list[float], list[int], int]:
    """Run TransNetV2 on a video and return (boundaries_sec, boundary_frame_indices, n_scenes)."""
    import numpy as np
    import ffmpeg as ffmpeg_lib

    _video_frames, single_frame_preds, _all_frame_preds = transnet_model.predict_video(source_video)
    scenes = transnet_model.predictions_to_scenes(single_frame_preds, threshold=threshold)

    # Get source video fps for frame→sec conversion
    probe = ffmpeg_lib.probe(source_video, select_streams="v:0")
    video_stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
    num, den = map(int, video_stream["r_frame_rate"].split("/"))
    src_fps = num / den if den else 30.0

    # scenes is [n_scenes, 2] with (start_frame, end_frame) inclusive
    # Boundaries are the start of each scene after the first one
    boundaries_sec: list[float] = []
    for i in range(1, len(scenes)):
        boundary_sec = round(float(scenes[i][0]) / src_fps, 3)
        if annotation_end is not None and boundary_sec > float(annotation_end):
            break
        if boundary_sec > annotation_start:
            boundaries_sec.append(boundary_sec)

    # Map to 1-based frame indices at the extraction fps
    boundary_frame_indices: list[int] = []
    for b_sec in boundaries_sec:
        adj_sec = b_sec - annotation_start
        if adj_sec <= 0:
            continue
        frame_idx = int(round(adj_sec * fps)) + 1  # 1-based
        boundary_frame_indices.append(frame_idx)

    n_scenes = len(boundaries_sec) + 1
    return boundaries_sec, boundary_frame_indices, n_scenes


# ─────────────────────────────────────────────────────────────────────────────
# PySceneDetect detection
# ─────────────────────────────────────────────────────────────────────────────

def _detect_pyscenedetect(
    source_video: str,
    annotation_start: float,
    annotation_end: float | None,
    fps: float,
    detector_type: str,
    threshold: float,
    min_scene_len: int,
) -> tuple[list[float], list[int], int]:
    """Run PySceneDetect on a video and return (boundaries_sec, boundary_frame_indices, n_scenes)."""
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector, AdaptiveDetector

    video = open_video(source_video)
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

    boundaries_sec: list[float] = []
    for i, (start, _end) in enumerate(scene_list):
        if i > 0:
            boundaries_sec.append(round(start.get_seconds(), 3))

    boundary_frame_indices: list[int] = []
    for b_sec in boundaries_sec:
        adj_sec = b_sec - annotation_start
        if adj_sec <= 0:
            continue
        frame_idx = int(round(adj_sec * fps)) + 1  # 1-based
        boundary_frame_indices.append(frame_idx)

    return boundaries_sec, boundary_frame_indices, len(scene_list)


# ─────────────────────────────────────────────────────────────────────────────
# Per-clip dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def detect_scenes_for_clip(
    frame_dir: Path,
    overwrite: bool = False,
    detector_type: str = "transnet",
    threshold: float = 0.5,
    min_scene_len: int = 15,
    transnet_model=None,
    annotations_dir: Path | None = None,
) -> dict:
    """Detect scene boundaries for a single clip.

    Args:
        frame_dir:       Path to the clip's extracted frames directory.
        overwrite:       Re-detect even if scenes.json already exists.
        detector_type:   "transnet", "content" (ContentDetector), or "adaptive" (AdaptiveDetector).
        threshold:       Detection threshold (0.5 for transnet, 20.0 for content, 3.0 for adaptive).
        min_scene_len:   Minimum scene length in source video frames (PySceneDetect only).
        transnet_model:  Pre-loaded TransNetV2 model instance (required for transnet mode).
        annotations_dir: If provided, skip clips that already have complete annotations.

    Returns:
        Status dict with clip_key, skipped, n_scenes, error.
    """
    scenes_path = frame_dir / "scenes.json"
    if not overwrite and scenes_path.exists():
        return {"clip_key": frame_dir.name, "skipped": True,
                "n_scenes": -1, "error": None}

    # Skip if this clip already has a complete annotation
    if annotations_dir is not None:
        ann_file = annotations_dir / f"{frame_dir.name}.json"
        if ann_file.exists():
            try:
                with open(ann_file, encoding="utf-8") as f:
                    ann = json.load(f)
                if ann.get("level1") is not None and ann.get("level2") is not None and ann.get("level3") is not None:
                    return {"clip_key": frame_dir.name, "skipped": True,
                            "n_scenes": -1, "error": None}
            except Exception:
                pass

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

    fps = float(meta.get("fps", 1.0))
    annotation_start = float(meta.get("annotation_start_sec", 0.0))
    annotation_end = meta.get("annotation_end_sec")

    try:
        if detector_type == "transnet":
            if transnet_model is None:
                return {"clip_key": frame_dir.name, "skipped": False,
                        "n_scenes": 0, "error": "transnet_model not provided"}
            boundaries_sec, boundary_frame_indices, n_scenes = _detect_transnet(
                str(source_video), annotation_start, annotation_end, fps,
                transnet_model, threshold=threshold,
            )
        else:
            boundaries_sec, boundary_frame_indices, n_scenes = _detect_pyscenedetect(
                str(source_video), annotation_start, annotation_end, fps,
                detector_type, threshold, min_scene_len,
            )
    except Exception as e:
        return {"clip_key": frame_dir.name, "skipped": False,
                "n_scenes": 0, "error": str(e)[:200]}

    scenes_data = {
        "clip_key": frame_dir.name,
        "detector": detector_type,
        "threshold": threshold,
        "source_video_path": str(source_video),
        "n_scenes": n_scenes,
        "boundary_timestamps_sec": boundaries_sec,
        "boundary_frame_indices": boundary_frame_indices,
    }
    if detector_type != "transnet":
        scenes_data["min_scene_len"] = min_scene_len

    with open(scenes_path, "w", encoding="utf-8") as f:
        json.dump(scenes_data, f, indent=2, ensure_ascii=False)

    return {"clip_key": frame_dir.name, "skipped": False,
            "n_scenes": n_scenes, "error": None}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect scene boundaries using TransNetV2 or PySceneDetect",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--frames-dir", required=True,
                        help="Root directory of pre-extracted frames")
    parser.add_argument("--detector", choices=["transnet", "content", "adaptive"],
                        default="transnet",
                        help="Scene detection algorithm")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Detection threshold (default: 0.5 for transnet, 20.0 for content, 3.0 for adaptive)")
    parser.add_argument("--min-scene-len", type=int, default=15,
                        help="Minimum scene length in source video frames (PySceneDetect only)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel detection workers")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process at most N clips (0 = all)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-detect even if scenes.json exists")
    parser.add_argument("--annotations-dir", default=None,
                        help="Skip clips that already have complete annotations in this directory")
    args = parser.parse_args()

    # Resolve default threshold per detector
    if args.threshold is None:
        if args.detector == "transnet":
            args.threshold = 0.5
        elif args.detector == "adaptive":
            args.threshold = 3.0
        else:
            args.threshold = 20.0

    frames_base = Path(args.frames_dir)
    if not frames_base.exists():
        print(f"ERROR: frames-dir not found: {frames_base}", file=sys.stderr)
        sys.exit(1)

    frame_dirs = sorted(p for p in frames_base.iterdir() if p.is_dir())
    if args.limit > 0:
        frame_dirs = frame_dirs[:args.limit]

    print(f"Scene detection: {len(frame_dirs)} clips, "
          f"detector={args.detector}, threshold={args.threshold}", flush=True)

    # Load TransNetV2 model once if needed
    transnet_model = None
    if args.detector == "transnet":
        try:
            from transnetv2 import TransNetV2
            transnet_model = TransNetV2()
            print("TransNetV2 model loaded.", flush=True)
        except ImportError:
            print("ERROR: transnetv2 not installed. Install with:\n"
                  "  pip install tensorflow>=2.0 ffmpeg-python pillow\n"
                  "  pip install git+https://github.com/soCzech/TransNetV2.git",
                  file=sys.stderr)
            sys.exit(1)

    ok = skipped = errors = 0
    annotations_dir = Path(args.annotations_dir) if args.annotations_dir else None

    if args.detector == "transnet":
        # TransNetV2 uses TF model — run sequentially to avoid GPU contention
        for i, fd in enumerate(frame_dirs, 1):
            try:
                res = detect_scenes_for_clip(
                    fd, args.overwrite, args.detector,
                    args.threshold, args.min_scene_len,
                    transnet_model=transnet_model,
                    annotations_dir=annotations_dir,
                )
            except Exception as exc:
                errors += 1
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
    else:
        # PySceneDetect is CPU-only, safe to parallelize
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    detect_scenes_for_clip,
                    fd, args.overwrite, args.detector,
                    args.threshold, args.min_scene_len,
                    annotations_dir=annotations_dir,
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
