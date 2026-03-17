#!/usr/bin/env python3
"""
build_dataset.py — Convert hierarchical annotations into EasyR1 training JSONL.

Level 1: One record per clip (warped-time macro phase segmentation).
Level 2: 128s sliding windows containing multiple events (constructed at training time).
Level 3: One record per L2 event (local temporal grounding).

Usage:
    python build_dataset.py \
        --annotation-dir proxy_data/youcook2_seg_annotation/annotations \
        --output proxy_data/youcook2_seg_annotation/youcook2_hier_L1_train.jsonl \
        --level 1
"""

import argparse
import json
import sys
from pathlib import Path

from prompts import get_level1_prompt, get_level2_prompt, get_level3_prompt


_LEVEL_TO_PROBLEM_TYPE = {
    1: "temporal_seg_hier_L1",
    2: "temporal_seg_hier_L2",
    3: "temporal_seg_hier_L3",
}


def _strip_internal_keys(d: dict) -> dict:
    """Remove internal metadata keys (prefixed with '_') from annotation dicts."""
    return {k: v for k, v in d.items() if not k.startswith("_")}


# ─────────────────────────────────────────────────────────────────────────────
# Level 1: one record per clip — warped timeline
# ─────────────────────────────────────────────────────────────────────────────

def build_level1_records(ann: dict) -> list[dict]:
    """Build training record for Level 1 (warped-time macro phase segmentation)."""
    l1 = ann.get("level1")
    if l1 is None or l1.get("_parse_error"):
        return []

    clip_duration = float(ann.get("clip_duration_sec") or 0)
    if clip_duration <= 0:
        return []

    video_path = ann.get("source_video_path") or ann.get("video_path", "")
    mapping = l1.get("_warped_mapping", [])
    n_frames = len(mapping) if mapping else l1.get("_sampling", {}).get("n_sampled_frames", 32)

    # Prompt uses warped frame count
    user_text = get_level1_prompt(n_frames)
    full_user = f"Watch the following cooking video clip carefully:\n<video>\n\n{user_text}"

    # Answer: macro_phases with start_frame/end_frame (warped space)
    # Strip real-time fields and internal keys for a clean answer
    answer_data = _strip_internal_keys(l1)

    return [{
        "messages": [{"role": "user", "content": full_user}],
        "prompt": full_user,
        "answer": json.dumps(answer_data, ensure_ascii=False),
        "videos": [video_path] if video_path else [],
        "data_type": "video",
        "problem_type": _LEVEL_TO_PROBLEM_TYPE[1],
        "metadata": {
            "clip_key": ann.get("clip_key", ""),
            "clip_duration_sec": clip_duration,
            "level": 1,
            "n_warped_frames": n_frames,
            "warped_mapping": mapping,
            "n_frames": int(ann.get("n_frames") or 0),
            "frame_dir": ann.get("frame_dir", ""),
            "source_mode": ann.get("source_mode", ""),
            "annotated_at": ann.get("annotated_at"),
        },
    }]


# ─────────────────────────────────────────────────────────────────────────────
# Level 2: 128s sliding windows with multiple events (constructed at training time)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_sliding_windows(
    total_duration: float,
    window_size: int = 128,
    stride: int = 64,
) -> list[tuple[int, int]]:
    """Generate overlapping (start_sec, end_sec) sliding windows."""
    windows: list[tuple[int, int]] = []
    start = 0
    total = int(total_duration)
    while start < total:
        end = min(start + window_size, total)
        if end - start >= stride // 2:
            windows.append((start, end))
        if end >= total:
            break
        start += stride
    return windows


def _events_in_window(events: list[dict], win_start: int, win_end: int) -> list[dict]:
    """Filter events overlapping with a window, clipping to window boundaries."""
    result = []
    for ev in events:
        st = ev.get("start_time")
        et = ev.get("end_time")
        if not isinstance(st, (int, float)) or not isinstance(et, (int, float)):
            continue
        # Check overlap
        if st >= win_end or et <= win_start:
            continue
        # Clip to window
        clipped = dict(ev)
        clipped["start_time"] = max(int(st), win_start)
        clipped["end_time"] = min(int(et), win_end)
        result.append(clipped)
    # Re-number
    for i, ev in enumerate(result, 1):
        ev["event_id"] = i
    return result


def build_level2_records(
    ann: dict,
    window_size: int = 128,
    stride: int = 64,
    min_events: int = 2,
) -> list[dict]:
    """
    Build training records for Level 2.

    Ignores L1 phase structure. Constructs 128s sliding windows over the full
    clip duration, filters out the all-event annotation from L2, and only keeps
    windows containing >= min_events events.
    """
    l2 = ann.get("level2")
    if l2 is None or l2.get("_parse_error"):
        return []

    clip_duration = float(ann.get("clip_duration_sec") or 0)
    if clip_duration <= 0:
        return []

    video_path = ann.get("source_video_path") or ann.get("video_path", "")
    all_events = l2.get("events", [])
    if not all_events:
        return []

    # Generate sliding windows over the full clip
    windows = _generate_sliding_windows(clip_duration, window_size=window_size, stride=stride)

    records = []
    for win_start, win_end in windows:
        win_events = _events_in_window(all_events, win_start, win_end)

        # Only keep windows with enough events
        if len(win_events) < min_events:
            continue

        user_text = get_level2_prompt(win_start, win_end)
        full_user = f"Watch the following cooking video clip carefully:\n<video>\n\n{user_text}"

        # Answer: events visible in this window
        answer_data = {"events": [_strip_internal_keys(ev) for ev in win_events]}

        records.append({
            "messages": [{"role": "user", "content": full_user}],
            "prompt": full_user,
            "answer": json.dumps(answer_data, ensure_ascii=False),
            "videos": [video_path] if video_path else [],
            "data_type": "video",
            "problem_type": _LEVEL_TO_PROBLEM_TYPE[2],
            "metadata": {
                "clip_key": ann.get("clip_key", ""),
                "clip_duration_sec": clip_duration,
                "level": 2,
                "window_start_sec": win_start,
                "window_end_sec": win_end,
                "n_events_in_window": len(win_events),
                "n_frames": int(ann.get("n_frames") or 0),
                "frame_dir": ann.get("frame_dir", ""),
                "source_mode": ann.get("source_mode", ""),
                "annotated_at": ann.get("annotated_at"),
            },
        })

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Level 3: one record per L2 event (temporal grounding)
# ─────────────────────────────────────────────────────────────────────────────

def build_level3_records(ann: dict, min_actions: int = 3) -> list[dict]:
    """Build training records for Level 3 (one per L2 event)."""
    l2 = ann.get("level2")
    l3 = ann.get("level3")
    if l2 is None or l3 is None or l3.get("_parse_error"):
        return []

    clip_duration = float(ann.get("clip_duration_sec") or 0)
    if clip_duration <= 0:
        return []

    video_path = ann.get("source_video_path") or ann.get("video_path", "")
    events = l2.get("events", [])
    all_results = l3.get("grounding_results", [])

    records = []
    for event in events:
        if not isinstance(event, dict):
            continue
        event_id = event.get("event_id")
        start_time = event.get("start_time")
        end_time = event.get("end_time")
        instruction = event.get("instruction", "")

        if not isinstance(start_time, (int, float)) or not isinstance(end_time, (int, float)):
            continue

        # Gather grounding results for this event
        event_results = [
            _strip_internal_keys(r) for r in all_results
            if isinstance(r, dict) and r.get("parent_event_id") == event_id
        ]
        if len(event_results) < min_actions:
            continue

        user_text = get_level3_prompt(int(start_time), int(end_time), instruction)
        full_user = f"Watch the following cooking video clip carefully:\n<video>\n\n{user_text}"

        answer_data = {"grounding_results": event_results}

        records.append({
            "messages": [{"role": "user", "content": full_user}],
            "prompt": full_user,
            "answer": json.dumps(answer_data, ensure_ascii=False),
            "videos": [video_path] if video_path else [],
            "data_type": "video",
            "problem_type": _LEVEL_TO_PROBLEM_TYPE[3],
            "metadata": {
                "clip_key": ann.get("clip_key", ""),
                "clip_duration_sec": clip_duration,
                "level": 3,
                "parent_event_id": event_id,
                "event_start_sec": int(start_time),
                "event_end_sec": int(end_time),
                "action_query": instruction,
                "n_grounding_results": len(event_results),
                "n_frames": int(ann.get("n_frames") or 0),
                "frame_dir": ann.get("frame_dir", ""),
                "source_mode": ann.get("source_mode", ""),
                "annotated_at": ann.get("annotated_at"),
            },
        })

    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert annotation JSONs to EasyR1 training JSONL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--annotation-dir", required=True,
                        help="Directory with per-clip .json annotation files")
    parser.add_argument("--output", required=True,
                        help="Output JSONL path")
    parser.add_argument("--level", type=int, choices=[1, 2, 3], default=1,
                        help="Which annotation level to build the dataset for")
    parser.add_argument("--l2-window-size", type=int, default=128,
                        help="[Level 2] Sliding window size in seconds for training data")
    parser.add_argument("--l2-stride", type=int, default=64,
                        help="[Level 2] Sliding window stride in seconds for training data")
    parser.add_argument("--l2-min-events", type=int, default=2,
                        help="[Level 2] Minimum events per window to include in training data")
    parser.add_argument("--l3-min-actions", type=int, default=3,
                        help="[Level 3] Minimum grounding actions required per L2 event")
    parser.add_argument("--complete-only", action="store_true",
                        help="Only process clips that have all 3 levels annotated (L1+L2+L3). "
                             "Useful when L1 has more clips than L2/L3.")
    args = parser.parse_args()

    ann_dir = Path(args.annotation_dir)
    if not ann_dir.exists():
        print(f"ERROR: annotation-dir not found: {ann_dir}", file=sys.stderr)
        sys.exit(1)

    ann_files = sorted(ann_dir.glob("*.json"))
    print(f"Found {len(ann_files)} annotation files in total")

    if args.complete_only:
        filtered = []
        for af in ann_files:
            try:
                with open(af, encoding="utf-8") as f:
                    d = json.load(f)
                has_l1 = d.get("level1") and not d["level1"].get("_parse_error")
                has_l2 = d.get("level2") and not d["level2"].get("_parse_error")
                has_l3 = d.get("level3") and not d["level3"].get("_parse_error")
                if has_l1 and has_l2 and has_l3:
                    filtered.append(af)
            except Exception:
                pass
        print(f"  --complete-only: {len(filtered)} clips have all 3 levels annotated (L1+L2+L3)")
        ann_files = filtered
    if args.level == 2:
        print(f"L2 training windows: size={args.l2_window_size}s  stride={args.l2_stride}s  min_events={args.l2_min_events}")
    elif args.level == 3:
        print(f"L3 training filter: min_actions={args.l3_min_actions}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = skipped = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for ann_file in ann_files:
            try:
                with open(ann_file, encoding="utf-8") as f:
                    ann = json.load(f)
            except Exception as e:
                print(f"  SKIP (parse error): {ann_file.name}: {e}")
                skipped += 1
                continue

            if args.level == 1:
                records = build_level1_records(ann)
            elif args.level == 2:
                records = build_level2_records(
                    ann,
                    window_size=args.l2_window_size,
                    stride=args.l2_stride,
                    min_events=args.l2_min_events,
                )
            else:
                records = build_level3_records(ann, min_actions=args.l3_min_actions)

            if not records:
                skipped += 1
                continue

            for record in records:
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += len(records)

    print(f"Written: {written} records  Skipped: {skipped} clips")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
