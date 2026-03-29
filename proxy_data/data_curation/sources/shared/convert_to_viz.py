"""
将 Stage A/B 筛选结果转为 segmentation_visualize 可读格式

把 stage_a_results_keep.jsonl 或 stage_b_results_keep.jsonl 里的候选样本
转为 segmentation_visualize 三层标注 JSON 格式，然后用现有的可视化服务器查看。

思路：
  - ET-Instruct / TimeLens 的事件标注 → L2 segments
  - Stage B 的 phase_sketch → L1 segments（如果有的话）
  - L3 留空（尚未标注）

用法:
    # 转换 ET-Instruct Stage A keep 样本
    python convert_to_viz.py \\
        --input et_instruct_164k/results/stage_a_results_keep.jsonl \\
        --output et_instruct_164k/results/viz_candidates/ \\
        --data-source et_instruct \\
        --video-root /path/to/et_instruct_videos/

    # 转换 TimeLens Stage B keep 样本
    python convert_to_viz.py \\
        --input timelens_100k/results/stage_b_results_keep.jsonl \\
        --output timelens_100k/results/viz_candidates/ \\
        --data-source timelens \\
        --video-root /path/to/timelens_videos/

    # 然后用 segmentation_visualize 查看
    python ../../data_visualization/segmentation_visualize/server.py \\
        --data et_instruct_164k/results/viz_candidates/ \\
        --port 8765
"""

import json
import argparse
import os
import re
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


# ── Conversion ───────────────────────────────────────────

def convert_sample(
    sample: dict,
    parse_fn,
    data_source: str,
    video_root: str | None = None,
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

def main():
    parser = argparse.ArgumentParser(
        description="将 Stage A/B 结果转为 segmentation_visualize 格式",
    )
    parser.add_argument("--input", required=True, help="stage_a/b_results_keep.jsonl")
    parser.add_argument("--output", required=True, help="输出目录（每样本一个 JSON）")
    parser.add_argument("--data-source", required=True, choices=list(PARSERS.keys()))
    parser.add_argument("--video-root", default=None, help="视频文件根目录")
    parser.add_argument("--limit", type=int, default=0, help="最多转换条数（0=全部）")
    args = parser.parse_args()

    parse_fn = PARSERS[args.data_source]

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

    # Convert
    os.makedirs(args.output, exist_ok=True)
    converted = 0
    for sample in samples:
        annotation = convert_sample(sample, parse_fn, args.data_source, args.video_root)
        sample_id = annotation["clip_key"]
        output_path = os.path.join(args.output, f"{sample_id}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(annotation, f, indent=2, ensure_ascii=False)
        converted += 1

    print(f"转换完成: {converted} 个标注文件 -> {args.output}")
    print(f"\n查看命令:")
    print(f"  python data_visualization/segmentation_visualize/server.py \\")
    print(f"      --annotation-dir {args.output} \\")
    print(f"      --port 8765")


if __name__ == "__main__":
    main()
