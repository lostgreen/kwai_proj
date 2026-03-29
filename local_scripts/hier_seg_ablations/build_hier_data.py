#!/usr/bin/env python3
"""
build_hier_data.py — 直接从原始标注 JSON 一步构建 hier-seg 训练数据。

替代原先的 5 步流水线 (extract_frames → annotate → build_dataset → prepare_clips → prepare_data)。

支持 L1 (macro phase) / L2 (event detection) / L3 (query grounding) / L3_seg (free segmentation)。
所有时间戳在构建时直接转为 0-based 窗口/clip 相对坐标。

用法:
    # 三层全部构建
    python build_hier_data.py \
        --annotation-dir /path/to/annotations \
        --clip-dir-l2 /path/to/clips/L2 \
        --clip-dir-l3 /path/to/clips/L3 \
        --output-dir ./data/hier_seg \
        --levels L1 L2 L3 L3_seg

    # 只构建 L2
    python build_hier_data.py \
        --annotation-dir /path/to/annotations \
        --clip-dir-l2 /path/to/clips/L2 \
        --output-dir ./data/hier_L2_only \
        --levels L2
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

# 添加 repo root 到 sys.path 以便 import prompts + shared
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
PROMPTS_DIR = os.path.join(REPO_ROOT, "proxy_data", "youcook2_seg", "hier_seg_annotation")
PROXY_DATA_DIR = os.path.join(REPO_ROOT, "proxy_data")
for _p in (PROMPTS_DIR, PROXY_DATA_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from prompts import (
    get_level1_train_prompt_temporal,
    get_level2_train_prompt,
    get_level3_query_prompt,
    get_level3_seg_prompt,
)
from shared.seg_source import (
    load_annotations,
    generate_sliding_windows,
    compute_l3_clip as _compute_l3_clip,
    get_l1_clip_path,
    get_l2_clip_path,
    get_l3_clip_path,
    L2_WINDOW_SIZE,
    L2_STRIDE,
    L3_MAX_CLIP_SEC,
    L3_PADDING,
)

PROBLEM_TYPES = {
    "L1": "temporal_seg_hier_L1",
    "L2": "temporal_seg_hier_L2",
    "L3": "temporal_seg_hier_L3",
    "L3_seg": "temporal_seg_hier_L3_seg",
}


# =====================================================================
# JSONL 写入
# =====================================================================
def write_jsonl(records: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# =====================================================================
# L1: Macro Phase Segmentation (全视频 fps 重采样，时间戳模式)
# =====================================================================
def build_l1_records(
    ann: dict,
    clip_dir_l1: str = "",
    l1_fps: int = 1,
) -> list[dict]:
    """从标注 JSON 构建 L1 训练记录 (基于真实时间戳)。

    视频输入: 原始视频以 l1_fps 帧率重采样后的 MP4 (由 prepare_clips.py 生成)。
    当 clip_dir_l1 为空时，videos 指向原始源视频 (可后续由 prepare_clips.py 处理)。
    """
    l1 = ann.get("level1")
    if not l1 or l1.get("_parse_error"):
        return []

    clip_duration = float(ann.get("clip_duration_sec") or 0)
    if clip_duration <= 0:
        return []

    phases = l1.get("macro_phases", [])
    if not phases:
        return []

    source_video = ann.get("source_video_path") or ann.get("video_path", "")
    clip_key = ann.get("clip_key", "")
    duration = int(clip_duration)

    # 提取 phase 边界 (真实秒数)
    spans = []
    for p in phases:
        st = p.get("start_time")
        et = p.get("end_time")
        if not isinstance(st, (int, float)) or not isinstance(et, (int, float)):
            continue
        st, et = int(st), int(et)
        if st >= et or st < 0:
            continue
        st = max(0, st)
        et = min(duration, et)
        if st < et:
            spans.append([st, et])

    if not spans:
        return []

    # 确定视频路径: 优先用 fps 重采样 clip，否则用源视频
    if clip_dir_l1:
        vp = get_l1_clip_path(clip_key, clip_dir_l1, fps=l1_fps)
    else:
        vp = source_video

    prompt = (
        "Watch the following video clip carefully:\n<video>\n\n"
        + get_level1_train_prompt_temporal(duration)
    )
    answer = f"<events>{json.dumps(spans)}</events>"

    return [{
        "messages": [{"role": "user", "content": prompt}],
        "prompt": prompt,
        "answer": answer,
        "videos": [vp] if vp else [],
        "data_type": "video",
        "problem_type": PROBLEM_TYPES["L1"],
        "metadata": {
            "clip_key": clip_key,
            "clip_duration_sec": clip_duration,
            "level": 1,
            "l1_fps": l1_fps,
            "source_video_path": source_video,  # 供 prepare_clips.py 使用
            "n_phases": len(spans),
            "source": "annotation_json",
        },
    }]


# =====================================================================
# L2: Event Detection (128s 滑窗, 直接归零)
# =====================================================================
def build_l2_records(
    ann: dict,
    clip_dir_l2: str = "",
    min_events: int = 2,
    window_size: int = L2_WINDOW_SIZE,
    stride: int = L2_STRIDE,
) -> list[dict]:
    """从标注 JSON 构建 L2 训练记录。

    window_size > 0: 按 (window_size, stride) 滑窗切分，裁剪事件到窗口边界，归零。
    window_size <= 0: 不切窗，直接以完整 clip 作为单一 "窗口" ([0, clip_duration])。
    """
    l2 = ann.get("level2")
    if not l2 or l2.get("_parse_error"):
        return []

    clip_duration = float(ann.get("clip_duration_sec") or 0)
    if clip_duration <= 0:
        return []

    events = l2.get("events", [])
    clip_key = ann.get("clip_key", "")
    video_path = ann.get("source_video_path") or ann.get("video_path", "")

    # 生成窗口列表：-1 / 0 表示不切窗，直接用完整 clip
    if window_size > 0:
        windows = generate_sliding_windows(clip_duration, window_size, stride)
    else:
        windows = [(0, int(clip_duration))]

    records = []
    for ws, we in windows:
        duration = we - ws

        # 收集并裁剪重叠事件到窗口内，直接归零
        matched = []
        for ev in events:
            if not isinstance(ev, dict):
                continue
            ev_start = ev.get("start_time")
            ev_end = ev.get("end_time")
            if not isinstance(ev_start, (int, float)) or not isinstance(ev_end, (int, float)):
                continue
            ev_start, ev_end = int(ev_start), int(ev_end)
            if ev_start >= we or ev_end <= ws:
                continue
            # 裁剪到窗口内，直接归零
            s = max(ev_start, ws) - ws
            e = min(ev_end, we) - ws
            if s < e:
                matched.append([s, e])

        if len(matched) < min_events:
            continue

        # 确定视频路径
        if clip_dir_l2:
            vp = get_l2_clip_path(clip_key, ws, we, clip_dir_l2)
        else:
            vp = video_path

        prompt = (
            "Watch the following video clip carefully:\n<video>\n\n"
            + get_level2_train_prompt(duration)
        )
        answer = f"<events>{json.dumps(matched)}</events>"

        records.append({
            "messages": [{"role": "user", "content": prompt}],
            "prompt": prompt,
            "answer": answer,
            "videos": [vp],
            "data_type": "video",
            "problem_type": PROBLEM_TYPES["L2"],
            "metadata": {
                "clip_key": clip_key,
                "clip_duration_sec": clip_duration,
                "level": 2,
                "window_start_sec": ws,
                "window_end_sec": we,
                "n_events_in_window": len(matched),
                "source": "annotation_json",
            },
        })

    return records


# =====================================================================
# L3: Query-Conditioned Grounding
# =====================================================================
def build_l3_records(
    ann: dict,
    clip_dir_l3: str = "",
    min_actions: int = 3,
    l3_order: str = "sequential",
) -> list[dict]:
    """从标注 JSON 构建 L3 grounding 训练记录。

    每个 L2 event 一条记录 (order=both 时两条)。
    """
    l2 = ann.get("level2")
    l3 = ann.get("level3")
    if not l2 or not l3 or l3.get("_parse_error"):
        return []

    clip_duration = float(ann.get("clip_duration_sec") or 0)
    if clip_duration <= 0:
        return []

    video_path = ann.get("source_video_path") or ann.get("video_path", "")
    clip_key = ann.get("clip_key", "")
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

        # 收集该事件的 grounding results
        raw_results = sorted(
            [r for r in all_results
             if isinstance(r, dict) and r.get("parent_event_id") == event_id],
            key=lambda r: r.get("start_time", 0),
        )
        if len(raw_results) < min_actions:
            continue

        clip_start, clip_end, duration = _compute_l3_clip(
            ev_start, ev_end, int(clip_duration),
        )

        # 构建 action 数据 (0-based)
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

        # 确定视频路径
        if clip_dir_l3:
            vp = get_l3_clip_path(clip_key, event_id, clip_start, clip_end, clip_dir_l3)
        else:
            vp = video_path

        def _make_record(ordered_actions, is_shuffled):
            queries = [a["sub_action"] for a in ordered_actions]
            spans = [[a["start_time"], a["end_time"]] for a in ordered_actions]
            prompt_text = (
                "Watch the following video clip carefully:\n<video>\n\n"
                + get_level3_query_prompt(queries, duration)
            )
            answer_str = f"<events>{json.dumps(spans)}</events>"
            return {
                "messages": [{"role": "user", "content": prompt_text}],
                "prompt": prompt_text,
                "answer": answer_str,
                "videos": [vp],
                "data_type": "video",
                "problem_type": PROBLEM_TYPES["L3"],
                "metadata": {
                    "clip_key": clip_key,
                    "clip_duration_sec": clip_duration,
                    "level": 3,
                    "parent_event_id": event_id,
                    "event_start_sec": ev_start,
                    "event_end_sec": ev_end,
                    "clip_start_sec": clip_start,
                    "clip_end_sec": clip_end,
                    "n_grounding_results": len(ordered_actions),
                    "shuffled": is_shuffled,
                    "source": "annotation_json",
                },
            }

        if l3_order in ("sequential", "both"):
            records.append(_make_record(actions, is_shuffled=False))

        if l3_order in ("shuffled", "both"):
            shuffled = actions[:]
            random.shuffle(shuffled)
            records.append(_make_record(shuffled, is_shuffled=True))

    return records


# =====================================================================
# L3 Seg: Free Segmentation (无 query)
# =====================================================================
def build_l3_seg_records(
    ann: dict,
    clip_dir_l3: str = "",
    min_actions: int = 3,
) -> list[dict]:
    """从标注 JSON 构建 L3 segmentation (无 query) 训练记录。"""
    l2 = ann.get("level2")
    l3 = ann.get("level3")
    if not l2 or not l3 or l3.get("_parse_error"):
        return []

    clip_duration = float(ann.get("clip_duration_sec") or 0)
    if clip_duration <= 0:
        return []

    video_path = ann.get("source_video_path") or ann.get("video_path", "")
    clip_key = ann.get("clip_key", "")
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

        clip_start, clip_end, duration = _compute_l3_clip(
            ev_start, ev_end, int(clip_duration),
        )

        # 构建时间段 (0-based, chronological)
        spans = []
        for r in raw_results:
            st = max(0, int(r.get("start_time", 0)) - clip_start)
            et = min(duration, int(r.get("end_time", duration)) - clip_start)
            spans.append([st, et])

        if clip_dir_l3:
            vp = get_l3_clip_path(clip_key, event_id, clip_start, clip_end, clip_dir_l3)
        else:
            vp = video_path

        prompt = (
            "Watch the following video clip carefully:\n<video>\n\n"
            + get_level3_seg_prompt(duration)
        )
        answer = f"<events>{json.dumps(spans)}</events>"

        records.append({
            "messages": [{"role": "user", "content": prompt}],
            "prompt": prompt,
            "answer": answer,
            "videos": [vp],
            "data_type": "video",
            "problem_type": PROBLEM_TYPES["L3_seg"],
            "metadata": {
                "clip_key": clip_key,
                "clip_duration_sec": clip_duration,
                "level": "3s",
                "parent_event_id": event_id,
                "event_start_sec": ev_start,
                "event_end_sec": ev_end,
                "clip_start_sec": clip_start,
                "clip_end_sec": clip_end,
                "n_actions": len(spans),
                "source": "annotation_json",
            },
        })

    return records


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="从原始标注 JSON 直接构建 hier-seg 训练数据 (L1/L2/L3/L3_seg)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--annotation-dir", required=True,
                        help="标注 JSON 目录 (annotations/*.json)")
    parser.add_argument("--clip-dir-l1", default="",
                        help="L1 fps-重采样 clips 目录 (clips/L1/), 留空则用原始视频路径")
    parser.add_argument("--l1-fps", type=int, default=1,
                        help="L1 视频重采样帧率 (默认: 1fps)")
    parser.add_argument("--clip-dir-l2", default="",
                        help="L2 clips 目录 (clips/L2/), 留空则用原始视频路径")
    parser.add_argument("--clip-dir-l3", default="",
                        help="L3 clips 目录 (clips/L3/), 留空则用原始视频路径")
    parser.add_argument("--output-dir", required=True,
                        help="输出目录")
    parser.add_argument("--levels", nargs="+", required=True,
                        choices=["L1", "L2", "L3", "L3_seg"],
                        help="要构建的层级列表")
    parser.add_argument("--l3-order", default="sequential",
                        choices=["sequential", "shuffled", "both"],
                        help="L3 grounding 的 action query 顺序")
    parser.add_argument("--total-val", type=int, default=200,
                        help="总验证集样本数")
    parser.add_argument("--train-per-level", type=int, default=-1,
                        help="每层最多 train 条数 (-1 = 无限制)")
    parser.add_argument("--l2-window-size", type=int, default=L2_WINDOW_SIZE,
                        help="L2 滑窗大小（秒）。设为 -1 则不切窗，直接用完整 clip（默认: 128）")
    parser.add_argument("--l2-stride", type=int, default=L2_STRIDE,
                        help="L2 滑窗步长（秒），仅在 --l2-window-size > 0 时生效（默认: 64）")
    parser.add_argument("--min-events", type=int, default=2,
                        help="L2 每窗口最少事件数")
    parser.add_argument("--min-actions", type=int, default=3,
                        help="L3 每事件最少 grounding results 数")
    parser.add_argument("--complete-only", action="store_true",
                        help="仅处理 L1+L2+L3 均完整的 clip")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    ann_dir = Path(args.annotation_dir)
    if not ann_dir.exists():
        print(f"ERROR: annotation-dir not found: {ann_dir}")
        return

    # 使用 shared 统一加载接口
    ann_list = load_annotations(ann_dir, complete_only=args.complete_only)
    print(f"Found {len(ann_list)} annotation files"
          + (" (complete-only filtered)" if args.complete_only else ""))

    # 按层级收集记录
    level_records: dict[str, list[dict]] = {lv: [] for lv in args.levels}
    stats = {lv: {"clips": 0, "records": 0} for lv in args.levels}

    for ann in ann_list:
        for lv in args.levels:
            if lv == "L1":
                recs = build_l1_records(ann, args.clip_dir_l1, args.l1_fps)
            elif lv == "L2":
                recs = build_l2_records(
                    ann, args.clip_dir_l2, args.min_events,
                    args.l2_window_size, args.l2_stride,
                )
            elif lv == "L3":
                recs = build_l3_records(ann, args.clip_dir_l3, args.min_actions, args.l3_order)
            elif lv == "L3_seg":
                recs = build_l3_seg_records(ann, args.clip_dir_l3, args.min_actions)
            else:
                continue

            if recs:
                stats[lv]["clips"] += 1
                stats[lv]["records"] += len(recs)
                level_records[lv].extend(recs)

    # 打印提取统计
    print(f"\n=== Extraction Stats ===")
    for lv in args.levels:
        print(f"  {lv}: {stats[lv]['clips']} clips, {stats[lv]['records']} records")

    # 合并、分割 train/val
    train_per_level = None if args.train_per_level < 0 else args.train_per_level
    n_levels = len(args.levels)
    val_per_level = max(1, args.total_val // n_levels)

    all_train = []
    all_val = []

    for lv in args.levels:
        records = level_records[lv]
        rng.shuffle(records)

        n_val = min(val_per_level, len(records) // 5)
        val = records[:n_val]
        train_pool = records[n_val:]

        if train_per_level is not None:
            train = train_pool[:train_per_level]
        else:
            train = train_pool

        all_val.extend(val)
        all_train.extend(train)

        print(f"  {lv}: {len(train)} train + {len(val)} val")

    rng.shuffle(all_train)
    rng.shuffle(all_val)

    # 输出
    train_path = os.path.join(args.output_dir, "train.jsonl")
    val_path = os.path.join(args.output_dir, "val.jsonl")
    write_jsonl(all_train, train_path)
    write_jsonl(all_val, val_path)

    print(f"\n=== Output ===")
    print(f"  Train: {len(all_train)}")
    print(f"  Val:   {len(all_val)}")
    print(f"  Dir:   {args.output_dir}")

    if all_train:
        ex = all_train[0]
        print(f"\n  --- Example (first record) ---")
        print(f"  problem_type: {ex['problem_type']}")
        print(f"  video: {ex['videos'][0] if ex['videos'] else 'N/A'}")
        print(f"  answer: {ex['answer'][:200]}")


if __name__ == "__main__":
    main()
