#!/usr/bin/env python3
"""
prepare_clips.py — Build EasyR1-ready video inputs for all three levels.

For each L1 record:
  - Re-encode the source video at a fixed frame rate (default 1 fps) via ffmpeg.
  - The resulting video preserves real timestamps, so the model sees a temporally
    faithful but low-fps version of the full clip.
  - Output name: {clip_key}_L1_{fps}fps.mp4
  - Source video is read from metadata["source_video_path"] (set by build_hier_data.py).

For each L2 record:
  - Extract [window_start_sec, window_end_sec] from the source video via ffmpeg
  - Subtract window_start from all event timestamps → 0-based
  - Rebuild prompt with 0-based time range

For each L3 record:
  - Extract [event_start_sec, event_end_sec] from the source video via ffmpeg
  - Timestamps are already 0-based (normalized in build_dataset.py)

Usage:
    python prepare_clips.py \
        --input  datasets/hier_L1_train.jsonl \
        --output datasets/hier_L1_train_clipped.jsonl \
        --clip-dir /path/to/clip_output_dir \
        --workers 8 \
        --l1-fps 1
"""

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
# V4 shot-first prompt for rebuilding prompts after clipping
_ablation_dir = str(Path(__file__).resolve().parent.parent.parent.parent / "local_scripts" / "hier_seg_ablations" / "prompt_ablation")
if _ablation_dir not in sys.path:
    sys.path.insert(0, _ablation_dir)
from prompt_variants_v4 import PROMPT_VARIANTS_V4


def _ffmpeg_fps_resample(src: str, dst: Path, fps: int = 1) -> None:
    """Re-encode *src* at *fps* frames per second and save to *dst*.

    Preserves real timestamps so the model sees a temporally faithful but
    low-fps version of the full clip.  This replaces the old warped-JPEG-concat
    approach for L1 records.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", src,
        "-vf", f"fps={fps}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg fps-resample failed for {src}:\n"
            + result.stderr.decode(errors="replace")
        )


def _process_l1(record: dict, clip_dir: Path, fps: int = 1) -> dict:
    """Re-encode the L1 source video at *fps* frames per second.

    The source video path is read from ``metadata["source_video_path"]``,
    which is set by ``build_hier_data.py``.  This replaces the old approach
    of concatenating warped JPEG frames into a synthetic MP4.

    Output name: ``{clip_key}_L1_{fps}fps.mp4``
    """
    meta     = record["metadata"]
    clip_key = meta["clip_key"]
    fps      = meta.get("l1_fps", fps)

    # source_video_path is always the original full clip, not a prior output
    src_video = meta.get("source_video_path") or record["videos"][0]

    out_name = f"{clip_key}_L1_{fps}fps.mp4"
    out_path = clip_dir / out_name

    if not out_path.exists():
        _ffmpeg_fps_resample(src_video, out_path, fps)

    rec = dict(record)
    rec["videos"] = [str(out_path)]
    return rec


def _ffmpeg_extract(src: str, start: int, end: int, dst: Path) -> None:
    """Extract [start, end] seconds from src and save to dst using stream copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    duration = end - start
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-t",  str(duration),
        "-i",  src,
        "-c",  "copy",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {src} [{start},{end}]:\n"
            + result.stderr.decode(errors="replace")
        )


def _ffmpeg_extract_with_fps(src: str, start: int, end: int, dst: Path, fps: int = 2) -> None:
    """Extract [start, end] seconds and re-encode at *fps* frames per second."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    duration = end - start
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-t",  str(duration),
        "-i",  src,
        "-vf", f"fps={fps}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg extract+fps failed for {src} [{start},{end}] @{fps}fps:\n"
            + result.stderr.decode(errors="replace")
        )


def _process_l2_full(record: dict, clip_dir: Path, fps: int = 1) -> dict:
    """Re-encode full video at *fps* for L2 full-video mode.

    Mirrors _process_l1: same source, same fps-resample pipeline.
    Answer timestamps are already absolute (0-based from video start = 0),
    so no offset correction is needed.
    Output name: ``{clip_key}_L2_full_{fps}fps.mp4``
    """
    meta = record["metadata"]
    clip_key = meta["clip_key"]
    fps = meta.get("l2_fps", fps)
    src_video = meta.get("source_video_path") or record["videos"][0]

    out_name = f"{clip_key}_L2_full_{fps}fps.mp4"
    out_path = clip_dir / out_name

    if not out_path.exists():
        _ffmpeg_fps_resample(src_video, out_path, fps)

    rec = dict(record)
    rec["videos"] = [str(out_path)]
    return rec


def _process_l2(record: dict, clip_dir: Path) -> dict:
    meta = record["metadata"]
    clip_key    = meta["clip_key"]
    win_start   = meta["window_start_sec"]
    win_end     = meta["window_end_sec"]
    src_video   = record["videos"][0]
    offset      = win_start
    duration    = win_end - win_start

    out_name = f"{clip_key}_L2_w{win_start}_{win_end}.mp4"
    out_path = clip_dir / out_name

    if not out_path.exists():
        _ffmpeg_extract(src_video, win_start, win_end, out_path)

    # Normalize answer timestamps from absolute to 0-based
    m = re.search(r"<events>(\[.*?\])</events>", record["answer"], re.DOTALL)
    if m:
        spans = json.loads(m.group(1))
        spans = [[max(0, st - offset), max(0, et - offset)] for st, et in spans]
        new_answer = f"<events>{json.dumps(spans)}</events>"
    else:
        new_answer = record["answer"]

    # Rebuild prompt (0-based duration) using V4 shot-first template
    prompt_body = PROMPT_VARIANTS_V4["L2"]["V1"].format(duration=duration)
    new_user_text = (
        "Watch the following video clip carefully:\n<video>\n\n"
        + prompt_body
    )

    rec = dict(record)
    rec["videos"]   = [str(out_path)]
    rec["prompt"]   = new_user_text
    rec["messages"] = [{"role": "user", "content": new_user_text}]
    rec["answer"]   = new_answer
    rec["metadata"] = dict(meta, clip_offset_sec=offset)
    return rec


def _process_l2_phase(record: dict, clip_dir: Path, fps: int = 2) -> dict:
    """Extract a per-phase L2 clip and re-encode at *fps* frames per second.

    Expects metadata with ``phase_id``, ``phase_start_sec``, ``phase_end_sec``
    (set by ``build_hier_data.py --l2-mode phase``).
    Answer timestamps are already 0-based (built that way in build_l2_phase_records).
    """
    meta      = record["metadata"]
    clip_key  = meta["clip_key"]
    phase_id  = meta["phase_id"]
    ph_start  = meta["phase_start_sec"]
    ph_end    = meta["phase_end_sec"]
    src_video = record["videos"][0]

    out_name = f"{clip_key}_L2_ph{phase_id}_{ph_start}_{ph_end}.mp4"
    out_path = clip_dir / out_name

    if not out_path.exists():
        _ffmpeg_extract_with_fps(src_video, ph_start, ph_end, out_path, fps)

    rec = dict(record)
    rec["videos"]   = [str(out_path)]
    rec["metadata"] = dict(meta, clip_offset_sec=ph_start)
    return rec


def _process_l3(record: dict, clip_dir: Path, fps: int = 0) -> dict:
    meta      = record["metadata"]
    clip_key  = meta["clip_key"]
    event_id  = meta["parent_event_id"]
    src_video = record["videos"][0]

    # New format: clip_start/end from padded window (already 0-based in answer).
    # Fallback to event bounds for backward compatibility.
    clip_start = meta.get("clip_start_sec", meta["event_start_sec"])
    clip_end   = meta.get("clip_end_sec",   meta["event_end_sec"])

    out_name = f"{clip_key}_L3_ev{event_id}_{clip_start}_{clip_end}.mp4"
    out_path = clip_dir / out_name

    if not out_path.exists():
        if fps > 0:
            _ffmpeg_extract_with_fps(src_video, clip_start, clip_end, out_path, fps)
        else:
            _ffmpeg_extract(src_video, clip_start, clip_end, out_path)

    # Timestamps in answer are already 0-based (normalized in build_dataset.py).
    # Prompt is already correct (get_level3_query_prompt with 0-based duration).
    rec = dict(record)
    rec["videos"]   = [str(out_path)]
    rec["metadata"] = dict(meta, clip_offset_sec=clip_start)
    return rec


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract sub-clips for L2/L3 records and normalize timestamps",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",    required=True, help="Input JSONL (from build_hier_data.py)")
    parser.add_argument("--output",   required=True, help="Output JSONL with updated videos/timestamps")
    parser.add_argument("--clip-dir", required=True, help="Directory to write extracted video clips")
    parser.add_argument("--workers",  type=int, default=4, help="Parallel ffmpeg workers")
    parser.add_argument("--l1-fps",   type=int, default=1, help="Frame rate for L1 fps-resampled clips (default: 1)")
    parser.add_argument("--l2l3-fps", type=int, default=0, help="Frame rate for L2/L3 clips (0=stream copy, 2=recommended)")
    parser.add_argument("--dry-run",  action="store_true", help="Skip ffmpeg, only rewrite JSONL (clips must exist)")
    parser.add_argument("--overwrite", action="store_true", help="Re-run even if output JSONL already exists")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)
    clip_dir    = Path(args.clip_dir)

    if output_path.exists() and not args.overwrite:
        print(f"SKIP: output already exists ({output_path}). Use --overwrite to re-run.")
        sys.exit(0)

    clip_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        print("No records found.", file=sys.stderr)
        sys.exit(1)

    raw_level = records[0]["metadata"].get("level")
    # Normalize level: "3s" (L3_seg) → 3
    level = 3 if raw_level in (3, "3s") else raw_level
    if level not in (1, 2, 3):
        print(f"ERROR: prepare_clips supports L1, L2, L3 (got level={raw_level})", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(records)} L{raw_level} records → {output_path}")
    print(f"Clip output dir: {clip_dir}")

    if args.dry_run:
        print("DRY RUN: skipping ffmpeg")

    l1_fps = args.l1_fps
    l2l3_fps = args.l2l3_fps

    # Detect L2 mode: full (l2_mode=="full") / phase (has phase_id) / window
    first_meta = records[0].get("metadata", {})
    is_l2_full  = (level == 2 and first_meta.get("l2_mode") == "full")
    is_l2_phase = (level == 2 and not is_l2_full and "phase_id" in first_meta)

    def _select_l2_fn(rec, d):
        if is_l2_full:
            return _process_l2_full(rec, d, fps=l1_fps)
        if is_l2_phase:
            return _process_l2_phase(rec, d, fps=l2l3_fps if l2l3_fps > 0 else 2)
        return _process_l2(rec, d)

    process_fn = {
        1: lambda rec, d: _process_l1(rec, d, fps=l1_fps),
        2: _select_l2_fn,
        3: lambda rec, d: _process_l3(rec, d, fps=l2l3_fps),
    }[level]

    done = skipped = errors = 0
    results: list[tuple[int, dict | Exception]] = []

    def _task(idx_rec):
        idx, rec = idx_rec
        try:
            if args.dry_run:
                # Patch the video path only, skip ffmpeg
                meta = rec["metadata"]
                if level == 1:
                    fps = meta.get("l1_fps", args.l1_fps)
                    name = f"{meta['clip_key']}_L1_{fps}fps.mp4"
                elif level == 2:
                    if meta.get("l2_mode") == "full":
                        fps = meta.get("l2_fps", args.l1_fps)
                        name = f"{meta['clip_key']}_L2_full_{fps}fps.mp4"
                    elif "phase_id" in meta:
                        name = f"{meta['clip_key']}_L2_ph{meta['phase_id']}_{meta['phase_start_sec']}_{meta['phase_end_sec']}.mp4"
                    else:
                        name = f"{meta['clip_key']}_L2_w{meta['window_start_sec']}_{meta['window_end_sec']}.mp4"
                else:
                    cs = meta.get("clip_start_sec", meta["event_start_sec"])
                    ce = meta.get("clip_end_sec",   meta["event_end_sec"])
                    name = f"{meta['clip_key']}_L3_ev{meta['parent_event_id']}_{cs}_{ce}.mp4"
                rec["videos"] = [str(clip_dir / name)]
                return idx, rec
            return idx, process_fn(rec, clip_dir)
        except Exception as exc:
            return idx, exc

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_task, (i, r)): i for i, r in enumerate(records)}
        for future in as_completed(futures):
            idx, result = future.result()
            results.append((idx, result))
            n = len(results)
            if isinstance(result, Exception):
                errors += 1
                print(f"  [{n}/{len(records)}] ERROR record {idx}: {result}")
            else:
                done += 1
                if n % 50 == 0 or n == len(records):
                    print(f"  [{n}/{len(records)}] done={done} errors={errors}")

    # Write in original order
    results.sort(key=lambda x: x[0])
    with open(output_path, "w", encoding="utf-8") as f:
        for _, result in results:
            if not isinstance(result, Exception):
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            else:
                skipped += 1

    print(f"\nDone. Written: {done - skipped}  Errors/skipped: {errors + skipped}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
