"""
将 Stage A/B 筛选结果转为 segmentation_visualize 可读格式

把 stage_a_results_keep.jsonl 或 stage_b_results_keep.jsonl 里的候选样本
转为 segmentation_visualize 三层标注 JSON 格式，并从视频抽取 1fps 帧，
然后用现有的可视化服务器查看。

思路：
  - ET-Instruct / TimeLens 的事件标注 → L2 segments
  - Stage B 的 phase_sketch → L1 segments（如果有的话）
  - L3 留空（尚未标注）
  - 从视频抽取 1fps 帧到 frames/ 子目录

用法:
    # 转换 ET-Instruct Stage A keep 样本（含抽帧）
    python convert_to_viz.py \\
        --input et_instruct_164k/results/stage_a_results_keep.jsonl \\
        --output et_instruct_164k/results/viz_candidates/ \\
        --data-source et_instruct \\
        --video-root /path/to/et_instruct_videos/

    # 跳过抽帧（仅生成 JSON）
    python convert_to_viz.py \\
        --input et_instruct_164k/results/stage_a_results_keep.jsonl \\
        --output et_instruct_164k/results/viz_candidates/ \\
        --data-source et_instruct \\
        --video-root /path/to/et_instruct_videos/ \\
        --no-frames

    # 然后用 segmentation_visualize 查看
    python data_visualization/segmentation_visualize/server.py \\
        --annotation-dir et_instruct_164k/results/viz_candidates/ \\
        --port 8765
"""

import json
import argparse
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# ── Event Parsers ────────────────────────────────────────

def parse_events_et_instruct(sample: dict) -> list[dict]:
    tgt = sample.get("tgt", [])
    gpt_text = ""
    for turn in sample.get("conversations", []):
        if turn.get("from") == "gpt":
            gpt_text = turn.get("value", "")
            break

    events = []
    n_events = len(tgt) // 2

    pattern = r'([\d.]+)\s*-\s*([\d.]+)\s*seconds?,\s*(.+?)(?=\d+\.?\d*\s*-\s*\d+\.?\d*\s*seconds?|$)'
    matches = re.findall(pattern, gpt_text, re.DOTALL)

    if matches and len(matches) >= n_events:
        for start_s, end_s, desc in matches:
            events.append({
                "start": float(start_s),
                "end": float(end_s),
                "description": desc.strip().rstrip('.').strip(),
            })
    else:
        for i in range(n_events):
            events.append({
                "start": float(tgt[i * 2]),
                "end": float(tgt[i * 2 + 1]),
                "description": f"(event {i+1})",
            })
    return events


def parse_events_timelens(sample: dict) -> list[dict]:
    raw_events = sample.get("events", [])
    parsed = []
    for ev in raw_events:
        desc = ev.get("query", "")
        spans = ev.get("span", [])
        if spans and len(spans[0]) == 2:
            start, end = spans[0][0], spans[0][1]
        else:
            start, end = 0.0, 0.0
        parsed.append({"start": float(start), "end": float(end), "description": desc})
    return parsed


PARSERS = {
    "et_instruct": parse_events_et_instruct,
    "timelens": parse_events_timelens,
}


def get_video_path(sample: dict, data_source: str) -> str:
    """Get video path from sample."""
    if data_source == "et_instruct":
        return sample.get("video", "")
    else:
        return sample.get("video_path", "")


def get_sample_id(sample: dict, data_source: str) -> str:
    """Get a unique, filesystem-safe ID for the sample."""
    path = get_video_path(sample, data_source)
    # Create a safe filename from video path
    return path.replace("/", "__").replace(".mp4", "").replace(".mkv", "").replace(".webm", "")


# ── Frame Extraction ─────────────────────────────────────

def extract_frames(
    video_path: str,
    frame_dir: str,
    fps: float = 1.0,
    duration_sec: float | None = None,
) -> int:
    """Extract 1fps frames from video using ffmpeg.

    Returns number of extracted frames, or 0 on failure.
    """
    os.makedirs(frame_dir, exist_ok=True)
    pattern = os.path.join(frame_dir, "%04d.jpg")

    cmd = ["ffmpeg", "-y", "-i", video_path]
    if duration_sec is not None and duration_sec > 0:
        cmd += ["-t", f"{duration_sec:.3f}"]
    cmd += ["-vf", f"fps={fps}", "-q:v", "2", pattern]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            return 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return 0

    n_frames = len(list(Path(frame_dir).glob("*.jpg")))
    return n_frames


# ── Conversion ───────────────────────────────────────────

def convert_sample(
    sample: dict,
    parse_fn,
    data_source: str,
    video_root: str | None = None,
    frame_dir: str | None = None,
    n_frames: int = 0,
) -> dict:
    """Convert a single curation candidate to segmentation_visualize format."""
    events = parse_fn(sample)
    duration = sample.get("duration", 0)
    video_path = get_video_path(sample, data_source)
    sample_id = get_sample_id(sample, data_source)

    # Resolve full video path
    full_video_path = video_path
    if video_root:
        full_video_path = os.path.join(video_root, video_path)

    # Build L2 segments from events
    l2_events = []
    for i, ev in enumerate(events, 1):
        start_sec = int(ev["start"])
        end_sec = int(ev["end"])
        l2_events.append({
            "event_id": i,
            "parent_phase_id": None,  # Will be filled if phase_sketch exists
            "start_time": start_sec,
            "end_time": end_sec,
            "instruction": ev["description"],
            "visual_keywords": [],
        })

    # Try to build L1 from phase_sketch (Stage B result)
    l1_phases = []
    assessment = sample.get("_assessment", {})
    phase_sketch = assessment.get("phase_sketch", [])

    if phase_sketch:
        for i, sketch in enumerate(phase_sketch, 1):
            if isinstance(sketch, str) and ":" in sketch:
                phase_name, event_range = sketch.split(":", 1)
                phase_name = phase_name.strip()
                event_range = event_range.strip()

                # Try to determine phase time range from referenced events
                referenced_events = _parse_event_range(event_range, len(events))
                if referenced_events:
                    phase_start = min(int(events[j]["start"]) for j in referenced_events)
                    phase_end = max(int(events[j]["end"]) for j in referenced_events)

                    # Assign parent_phase_id to referenced events
                    for j in referenced_events:
                        if j < len(l2_events):
                            l2_events[j]["parent_phase_id"] = i

                    l1_phases.append({
                        "phase_id": i,
                        "start_time": phase_start,
                        "end_time": phase_end,
                        "phase_name": phase_name,
                        "narrative_summary": f"Events: {event_range}",
                    })

    # Build the annotation dict
    annotation = {
        "clip_key": sample_id,
        "video_path": full_video_path,
        "clip_duration_sec": duration,
        "n_frames": n_frames,
        "frame_dir": frame_dir or "",
        "level1": {"macro_phases": l1_phases},
        "level2": {"events": l2_events},
        "level3": {"grounding_results": []},
        # Extra: curation metadata
        "_curation": {
            "source": sample.get("source", "unknown"),
            "data_source": data_source,
            "stage": sample.get("_stage", "?"),
            "assessment": assessment,
        },
    }

    return annotation


def _parse_event_range(range_str: str, n_events: int) -> list[int]:
    """Parse event range like '1-3' or '4,5,6' to 0-based indices."""
    indices = []
    range_str = range_str.strip()

    # Try "1-3" format
    m = re.match(r'(\d+)\s*-\s*(\d+)', range_str)
    if m:
        start = int(m.group(1))
        end = int(m.group(2))
        for i in range(start, end + 1):
            if 1 <= i <= n_events:
                indices.append(i - 1)
        return indices

    # Try "1,2,3" format
    for part in range_str.split(","):
        part = part.strip()
        if part.isdigit():
            i = int(part)
            if 1 <= i <= n_events:
                indices.append(i - 1)

    # Try "1 2 3" format
    if not indices:
        for part in range_str.split():
            if part.isdigit():
                i = int(part)
                if 1 <= i <= n_events:
                    indices.append(i - 1)

    return indices


# ── Main ─────────────────────────────────────────────────

def process_one_sample(
    sample: dict,
    parse_fn,
    data_source: str,
    video_root: str | None,
    output_dir: str,
    extract: bool,
    fps: float,
    overwrite: bool,
) -> dict:
    """Process a single sample: extract frames + write JSON. Returns status dict."""
    sample_id = get_sample_id(sample, data_source)
    video_path = get_video_path(sample, data_source)
    full_video_path = os.path.join(video_root, video_path) if video_root else video_path

    frame_dir_path = os.path.join(output_dir, "frames", sample_id)
    n_frames = 0
    frame_error = None

    if extract:
        # Check if already extracted
        existing = list(Path(frame_dir_path).glob("*.jpg")) if os.path.isdir(frame_dir_path) else []
        if existing and not overwrite:
            n_frames = len(existing)
        else:
            if not os.path.isfile(full_video_path):
                frame_error = f"video not found: {full_video_path}"
            else:
                duration = sample.get("duration")
                n_frames = extract_frames(
                    full_video_path,
                    frame_dir_path,
                    fps=fps,
                    duration_sec=duration,
                )
                if n_frames == 0:
                    frame_error = "ffmpeg extraction failed"

    annotation = convert_sample(
        sample, parse_fn, data_source, video_root,
        frame_dir=frame_dir_path if extract and n_frames > 0 else "",
        n_frames=n_frames,
    )

    json_path = os.path.join(output_dir, f"{sample_id}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(annotation, f, indent=2, ensure_ascii=False)

    return {
        "sample_id": sample_id,
        "n_frames": n_frames,
        "error": frame_error,
    }


def main():
    parser = argparse.ArgumentParser(
        description="将 Stage A/B 结果转为 segmentation_visualize 格式（含抽帧）",
    )
    parser.add_argument("--input", required=True, help="stage_a/b_results_keep.jsonl")
    parser.add_argument("--output", required=True, help="输出目录（每样本一个 JSON + frames/）")
    parser.add_argument("--data-source", required=True, choices=list(PARSERS.keys()))
    parser.add_argument("--video-root", default=None, help="视频文件根目录")
    parser.add_argument("--limit", type=int, default=0, help="最多转换条数（0=全部）")
    parser.add_argument("--no-frames", action="store_true", help="跳过抽帧（仅生成 JSON）")
    parser.add_argument("--fps", type=float, default=1.0, help="抽帧帧率（默认 1fps）")
    parser.add_argument("--workers", type=int, default=4, help="并行抽帧 worker 数")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有帧")
    args = parser.parse_args()

    parse_fn = PARSERS[args.data_source]
    do_extract = not args.no_frames

    # Load candidates
    samples = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    if args.limit > 0:
        samples = samples[:args.limit]

    print(f"加载 {len(samples)} 条候选样本")
    if do_extract:
        print(f"抽帧: fps={args.fps}, workers={args.workers}")
    else:
        print("跳过抽帧（--no-frames）")

    os.makedirs(args.output, exist_ok=True)

    success = 0
    frame_errors = 0

    if do_extract and args.workers > 1:
        # Parallel extraction
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    process_one_sample,
                    sample, parse_fn, args.data_source, args.video_root,
                    args.output, do_extract, args.fps, args.overwrite,
                ): i
                for i, sample in enumerate(samples)
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                res = fut.result()
                if res["error"]:
                    frame_errors += 1
                    if frame_errors <= 10:
                        print(f"  [{idx+1}/{len(samples)}] WARN {res['sample_id']}: {res['error']}")
                else:
                    success += 1
                    if success % 50 == 0:
                        print(f"  [{success}/{len(samples)}] OK  (latest: {res['n_frames']} frames)")
    else:
        # Sequential
        for i, sample in enumerate(samples):
            res = process_one_sample(
                sample, parse_fn, args.data_source, args.video_root,
                args.output, do_extract, args.fps, args.overwrite,
            )
            if res["error"]:
                frame_errors += 1
                if frame_errors <= 10:
                    print(f"  [{i+1}/{len(samples)}] WARN {res['sample_id']}: {res['error']}")
            else:
                success += 1

    print(f"\n转换完成: {success} 成功, {frame_errors} 抽帧失败 -> {args.output}")
    print(f"\n查看命令:")
    print(f"  python data_visualization/segmentation_visualize/server.py \\")
    print(f"      --annotation-dir {args.output} \\")
    print(f"      --port 8765")


if __name__ == "__main__":
    main()
