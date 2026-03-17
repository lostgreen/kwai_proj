#!/usr/bin/env python3
"""
prepare_clips.py — Build EasyR1-ready video inputs for all three levels.

For each L1 record:
  - Read warped_mapping (already subsampled to ≤256 frames by build_dataset.py)
  - Collect the corresponding 1fps JPEG frames from the frame directory
  - Concatenate them (1s per frame) into a synthetic mp4 via ffmpeg concat demuxer
  - The resulting video has exactly M frames → warped frame index i = video frame i

For each L2 record:
  - Extract [window_start_sec, window_end_sec] from the source video via ffmpeg
  - Subtract window_start from all event timestamps → 0-based
  - Rebuild prompt with 0-based time range

For each L3 record:
  - Extract [event_start_sec, event_end_sec] from the source video via ffmpeg
  - Timestamps are already 0-based (normalized in build_dataset.py)

Usage:
    python prepare_clips.py \
        --input  datasets/youcook2_hier_L1_train.jsonl \
        --output datasets/youcook2_hier_L1_train_clipped.jsonl \
        --clip-dir /path/to/clip_output_dir \
        --workers 8
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from prompts import get_level2_train_prompt


def _ffmpeg_concat_frames(frame_paths: list[Path], dst: Path) -> None:
    """Create a 1fps video from an ordered list of JPEG frames via ffmpeg concat demuxer.

    Each frame is shown for exactly 1 second, so warped frame index i = video frame i.
    The concat demuxer handles non-consecutive frame selections correctly.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Build concat file: each entry gets `duration 1`; last entry repeated without duration
    # (ffmpeg concat demuxer requirement for accurate last-frame duration)
    lines: list[str] = []
    for p in frame_paths:
        lines.append(f"file '{p.resolve()}'")
        lines.append("duration 1")
    lines.append(f"file '{frame_paths[-1].resolve()}'")

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".txt", dir=dst.parent)
    try:
        with os.fdopen(tmp_fd, "w") as f:
            f.write("\n".join(lines) + "\n")
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", tmp_path,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(dst),
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg concat failed for {dst}:\n"
                + result.stderr.decode(errors="replace")
            )
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


def _process_l1(record: dict, clip_dir: Path) -> dict:
    """Assemble a synthetic video from the warped-selected 1fps frames for L1.

    warped_mapping (already subsampled to ≤256 in build_dataset.py) maps
    warped_idx 1..M → real_sec.  Each frame is stored as {real_sec:04d}.jpg
    inside frame_dir.  The output video has exactly M frames at 1fps, so the
    model's i-th frame exactly corresponds to warped index i.
    """
    meta = record["metadata"]
    clip_key = meta["clip_key"]
    mapping  = meta["warped_mapping"]
    frame_dir = Path(meta["frame_dir"])

    out_name = f"{clip_key}_L1_warped{len(mapping)}f.mp4"
    out_path = clip_dir / out_name

    if not out_path.exists():
        frame_paths = [frame_dir / f"{e['real_sec']:04d}.jpg" for e in mapping]
        _ffmpeg_concat_frames(frame_paths, out_path)

    rec = dict(record)
    rec["videos"]   = [str(out_path)]
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

    # Rebuild prompt (0-based duration)
    new_user_text = (
        "Watch the following cooking video clip carefully:\n<video>\n\n"
        + get_level2_train_prompt(duration)
    )

    rec = dict(record)
    rec["videos"]   = [str(out_path)]
    rec["prompt"]   = new_user_text
    rec["messages"] = [{"role": "user", "content": new_user_text}]
    rec["answer"]   = new_answer
    rec["metadata"] = dict(meta, clip_offset_sec=offset)
    return rec


def _process_l3(record: dict, clip_dir: Path) -> dict:
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
    parser.add_argument("--input",    required=True, help="Input JSONL (L2 or L3 from build_dataset.py)")
    parser.add_argument("--output",   required=True, help="Output JSONL with updated videos/timestamps")
    parser.add_argument("--clip-dir", required=True, help="Directory to write extracted video clips")
    parser.add_argument("--workers",  type=int, default=4, help="Parallel ffmpeg workers")
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

    level = records[0]["metadata"].get("level")
    if level not in (1, 2, 3):
        print(f"ERROR: prepare_clips supports L1, L2, L3 (got level={level})", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(records)} L{level} records → {output_path}")
    print(f"Clip output dir: {clip_dir}")

    if args.dry_run:
        print("DRY RUN: skipping ffmpeg")

    process_fn = {1: _process_l1, 2: _process_l2, 3: _process_l3}[level]

    done = skipped = errors = 0
    results: list[tuple[int, dict | Exception]] = []

    def _task(idx_rec):
        idx, rec = idx_rec
        try:
            if args.dry_run:
                # Patch the video path only, skip ffmpeg
                meta = rec["metadata"]
                if level == 1:
                    name = f"{meta['clip_key']}_L1_warped{meta['n_warped_frames']}f.mp4"
                elif level == 2:
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
