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
import random
import sys
from pathlib import Path

from prompts import get_level1_train_prompt, get_level2_train_prompt, get_level3_query_prompt, get_level3_seg_prompt


_LEVEL_TO_PROBLEM_TYPE = {
    1: "temporal_seg_hier_L1",
    2: "temporal_seg_hier_L2",
    3: "temporal_seg_hier_L3",
    "3s": "temporal_seg_hier_L3_seg",
}


def _strip_internal_keys(d: dict) -> dict:
    """Remove internal metadata keys (prefixed with '_') from annotation dicts."""
    return {k: v for k, v in d.items() if not k.startswith("_")}


# ─────────────────────────────────────────────────────────────────────────────
# Level 1: one record per clip — warped timeline
# ─────────────────────────────────────────────────────────────────────────────

def _subsample_warped_mapping(
    mapping: list[dict],
    max_frames: int,
) -> tuple[list[dict], dict[int, int]]:
    """
    Uniformly subsample a warped_mapping to at most max_frames entries.

    Returns:
        new_mapping  — subsampled entries with warped_idx re-indexed 1..M
        remap        — dict mapping old warped_idx → new warped_idx
    """
    n = len(mapping)
    if n <= max_frames:
        return mapping, {e["warped_idx"]: e["warped_idx"] for e in mapping}

    m = max_frames
    # Pick m indices evenly spread over [0, n-1]
    selected_positions = [round(i * (n - 1) / (m - 1)) for i in range(m)]
    new_mapping = []
    for new_idx_0, orig_pos in enumerate(selected_positions):
        entry = dict(mapping[orig_pos])
        entry["warped_idx"] = new_idx_0 + 1
        new_mapping.append(entry)

    # Build remap: for any original warped_idx, find nearest new warped_idx
    # (linear interpolation in original space → new space)
    remap: dict[int, int] = {}
    for entry in mapping:
        orig = entry["warped_idx"]  # 1-indexed
        new_f = round((orig - 1) / (n - 1) * (m - 1)) + 1
        remap[orig] = max(1, min(m, new_f))

    return new_mapping, remap


def build_level1_records(ann: dict, max_frames: int = 256) -> list[dict]:
    """Build training record for Level 1 (warped-time macro phase segmentation).

    Args:
        max_frames: Maximum frames the model can see. When the original
                    warped mapping exceeds this limit, it is uniformly
                    subsampled and phase boundaries are remapped accordingly.
    """
    l1 = ann.get("level1")
    if l1 is None or l1.get("_parse_error"):
        return []

    clip_duration = float(ann.get("clip_duration_sec") or 0)
    if clip_duration <= 0:
        return []

    video_path = ann.get("source_video_path") or ann.get("video_path", "")
    mapping = l1.get("_warped_mapping", [])
    if not mapping:
        n_frames = l1.get("_sampling", {}).get("n_sampled_frames", 32)
        remap: dict[int, int] = {}
    else:
        mapping, remap = _subsample_warped_mapping(mapping, max_frames)
        n_frames = len(mapping)

    # Remap phase boundaries when subsampling occurred
    answer_data = _strip_internal_keys(l1)
    if remap:
        for phase in answer_data.get("macro_phases", []):
            if "start_frame" in phase:
                phase["start_frame"] = remap.get(phase["start_frame"], phase["start_frame"])
            if "end_frame" in phase:
                phase["end_frame"] = remap.get(phase["end_frame"], phase["end_frame"])

    phases = answer_data.get("macro_phases", [])
    spans = [[p["start_frame"], p["end_frame"]] for p in phases if "start_frame" in p and "end_frame" in p]
    answer_str = f"<events>{json.dumps(spans)}</events>"

    user_text = get_level1_train_prompt(n_frames)
    full_user = f"Watch the following cooking video clip carefully:\n<video>\n\n{user_text}"

    return [{
        "messages": [{"role": "user", "content": full_user}],
        "prompt": full_user,
        "answer": answer_str,
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

        user_text = get_level2_train_prompt(win_end - win_start)
        full_user = f"Watch the following cooking video clip carefully:\n<video>\n\n{user_text}"

        # Answer: events in this window (absolute timestamps; normalized by prepare_clips.py)
        spans = [[int(ev["start_time"]), int(ev["end_time"])] for ev in win_events]
        answer_str = f"<events>{json.dumps(spans)}</events>"

        records.append({
            "messages": [{"role": "user", "content": full_user}],
            "prompt": full_user,
            "answer": answer_str,
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
# Level 3: query-conditioned atomic grounding
# ─────────────────────────────────────────────────────────────────────────────

_L3_MAX_CLIP_SEC = 128  # hard cap on clip length fed to the model


def build_level3_records(
    ann: dict,
    min_actions: int = 3,
    padding: int = 5,
    order: str = "sequential",
) -> list[dict]:
    """
    Build training records for Level 3 (query-conditioned atomic grounding).

    One record per L2 event (× 2 if order="both").  The model is given a
    short video clip (L2 event ± padding, capped at _L3_MAX_CLIP_SEC) and an
    ordered or shuffled list of action captions. It must output the 0-based
    timestamp for each action in the given order.

    Args:
        min_actions: Skip events with fewer than this many grounding results.
        padding:     Seconds of context to add before/after the event bounds.
        order:       "sequential" | "shuffled" | "both"
    """
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
        event_id   = event.get("event_id")
        ev_start   = event.get("start_time")
        ev_end     = event.get("end_time")
        instruction = event.get("instruction", "")

        if not isinstance(ev_start, (int, float)) or not isinstance(ev_end, (int, float)):
            continue

        ev_start, ev_end = int(ev_start), int(ev_end)

        # Collect grounding results for this event, sorted by start_time
        raw_results = sorted(
            [r for r in all_results
             if isinstance(r, dict) and r.get("parent_event_id") == event_id],
            key=lambda r: r.get("start_time", 0),
        )
        if len(raw_results) < min_actions:
            continue

        # Compute padded clip window, capped at _L3_MAX_CLIP_SEC
        clip_start = max(0, ev_start - padding)
        clip_end   = min(int(clip_duration), ev_end + padding)
        if clip_end - clip_start > _L3_MAX_CLIP_SEC:
            # Shrink symmetrically
            excess = (clip_end - clip_start) - _L3_MAX_CLIP_SEC
            trim_start = min(excess // 2, ev_start - clip_start)
            clip_start += trim_start
            clip_end = clip_start + _L3_MAX_CLIP_SEC
        duration = clip_end - clip_start

        # Build per-action data (0-based timestamps)
        actions = []
        for r in raw_results:
            st = max(0, int(r.get("start_time", 0)) - clip_start)
            et = min(duration, int(r.get("end_time", duration)) - clip_start)
            actions.append({
                "orig_action_id": r.get("action_id"),
                "sub_action": r.get("sub_action", ""),
                "start_time": st,
                "end_time": et,
            })

        def _make_record(ordered_actions: list[dict], is_shuffled: bool) -> dict:
            queries = [a["sub_action"] for a in ordered_actions]
            query_order = [a["orig_action_id"] for a in ordered_actions]

            user_text = (
                "Watch the following cooking video clip carefully:\n<video>\n\n"
                + get_level3_query_prompt(queries, duration)
            )
            answer_str = f"<events>{json.dumps([[a['start_time'], a['end_time']] for a in ordered_actions])}</events>"
            return {
                "messages": [{"role": "user", "content": user_text}],
                "prompt": user_text,
                "answer": answer_str,
                "videos": [video_path] if video_path else [],
                "data_type": "video",
                "problem_type": _LEVEL_TO_PROBLEM_TYPE[3],
                "metadata": {
                    "clip_key": ann.get("clip_key", ""),
                    "clip_duration_sec": clip_duration,
                    "level": 3,
                    "parent_event_id": event_id,
                    "event_start_sec": ev_start,
                    "event_end_sec": ev_end,
                    "clip_start_sec": clip_start,
                    "clip_end_sec": clip_end,
                    "action_query": instruction,
                    "n_grounding_results": len(ordered_actions),
                    "query_order": query_order,
                    "shuffled": is_shuffled,
                    "n_frames": int(ann.get("n_frames") or 0),
                    "frame_dir": ann.get("frame_dir", ""),
                    "source_mode": ann.get("source_mode", ""),
                    "annotated_at": ann.get("annotated_at"),
                },
            }

        if order in ("sequential", "both"):
            records.append(_make_record(actions, is_shuffled=False))

        if order in ("shuffled", "both"):
            shuffled = actions[:]
            random.shuffle(shuffled)
            records.append(_make_record(shuffled, is_shuffled=True))

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Level 3 Seg: atomic action segmentation (no queries, detect all actions)
# ─────────────────────────────────────────────────────────────────────────────

def build_level3_seg_records(
    ann: dict,
    min_actions: int = 3,
    padding: int = 5,
) -> list[dict]:
    """
    Build training records for Level 3 segmentation (no query, detect all actions).

    Same clip extraction as grounding but the prompt only asks to detect all
    atomic actions without providing text queries. Answer segments are sorted
    chronologically. Uses F1-IoU reward (like L1/L2).
    """
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
        ev_start = event.get("start_time")
        ev_end = event.get("end_time")

        if not isinstance(ev_start, (int, float)) or not isinstance(ev_end, (int, float)):
            continue

        ev_start, ev_end = int(ev_start), int(ev_end)

        raw_results = sorted(
            [r for r in all_results
             if isinstance(r, dict) and r.get("parent_event_id") == event_id],
            key=lambda r: r.get("start_time", 0),
        )
        if len(raw_results) < min_actions:
            continue

        clip_start = max(0, ev_start - padding)
        clip_end = min(int(clip_duration), ev_end + padding)
        if clip_end - clip_start > _L3_MAX_CLIP_SEC:
            excess = (clip_end - clip_start) - _L3_MAX_CLIP_SEC
            trim_start = min(excess // 2, ev_start - clip_start)
            clip_start += trim_start
            clip_end = clip_start + _L3_MAX_CLIP_SEC
        duration = clip_end - clip_start

        # Build segments (0-based, chronological order)
        spans = []
        for r in raw_results:
            st = max(0, int(r.get("start_time", 0)) - clip_start)
            et = min(duration, int(r.get("end_time", duration)) - clip_start)
            spans.append([st, et])

        user_text = (
            "Watch the following cooking video clip carefully:\n<video>\n\n"
            + get_level3_seg_prompt(duration)
        )
        answer_str = f"<events>{json.dumps(spans)}</events>"

        records.append({
            "messages": [{"role": "user", "content": user_text}],
            "prompt": user_text,
            "answer": answer_str,
            "videos": [video_path] if video_path else [],
            "data_type": "video",
            "problem_type": _LEVEL_TO_PROBLEM_TYPE["3s"],
            "metadata": {
                "clip_key": ann.get("clip_key", ""),
                "clip_duration_sec": clip_duration,
                "level": "3s",
                "parent_event_id": event_id,
                "event_start_sec": ev_start,
                "event_end_sec": ev_end,
                "clip_start_sec": clip_start,
                "clip_end_sec": clip_end,
                "n_actions": len(spans),
                "n_frames": int(ann.get("n_frames") or 0),
                "frame_dir": ann.get("frame_dir", ""),
                "source_mode": ann.get("source_mode", ""),
                "annotated_at": ann.get("annotated_at"),
            },
        })

    return records
    parser = argparse.ArgumentParser(
        description="Convert annotation JSONs to EasyR1 training JSONL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--annotation-dir", required=True,
                        help="Directory with per-clip .json annotation files")
    parser.add_argument("--output", required=True,
                        help="Output JSONL path")
    parser.add_argument("--level", type=str, choices=["1", "2", "3", "3s"], default="1",
                        help="Which annotation level to build: 1 (phases), 2 (events), "
                             "3 (query grounding), 3s (action segmentation)")
    parser.add_argument("--l2-window-size", type=int, default=128,
                        help="[Level 2] Sliding window size in seconds for training data")
    parser.add_argument("--l2-stride", type=int, default=64,
                        help="[Level 2] Sliding window stride in seconds for training data")
    parser.add_argument("--l2-min-events", type=int, default=2,
                        help="[Level 2] Minimum events per window to include in training data")
    parser.add_argument("--l3-min-actions", type=int, default=3,
                        help="[Level 3] Minimum grounding actions required per L2 event")
    parser.add_argument("--l3-padding", type=int, default=5,
                        help="[Level 3] Seconds of context to add before/after each L2 event "
                             "when extracting the clip (capped at 128s total)")
    parser.add_argument("--l3-order", default="sequential",
                        choices=["sequential", "shuffled", "both"],
                        help="[Level 3] Query order: sequential (annotation order), "
                             "shuffled (random permutation), or both (one record each)")
    parser.add_argument("--complete-only", action="store_true",
                        help="Only process clips that have all 3 levels annotated (L1+L2+L3). "
                             "Useful when L1 has more clips than L2/L3.")
    parser.add_argument("--max-frames", type=int, default=256,
                        help="[Level 1] Max frames the model can see. Warped mapping is "
                             "uniformly subsampled when the original exceeds this limit.")
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
    if args.level == "1":
        print(f"L1 warped compression: max_frames={args.max_frames}")
    elif args.level == "2":
        print(f"L2 training windows: size={args.l2_window_size}s  stride={args.l2_stride}s  min_events={args.l2_min_events}")
    elif args.level == "3":
        print(f"L3 query grounding: min_actions={args.l3_min_actions}  padding={args.l3_padding}s  order={args.l3_order}")
    elif args.level == "3s":
        print(f"L3 segmentation: min_actions={args.l3_min_actions}  padding={args.l3_padding}s")

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

            if args.level == "1":
                records = build_level1_records(ann, max_frames=args.max_frames)
            elif args.level == "2":
                records = build_level2_records(
                    ann,
                    window_size=args.l2_window_size,
                    stride=args.l2_stride,
                    min_events=args.l2_min_events,
                )
            elif args.level == "3s":
                records = build_level3_seg_records(
                    ann,
                    min_actions=args.l3_min_actions,
                    padding=args.l3_padding,
                )
            else:
                records = build_level3_records(
                    ann,
                    min_actions=args.l3_min_actions,
                    padding=args.l3_padding,
                    order=args.l3_order,
                )

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
