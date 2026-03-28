#!/usr/bin/env python3
"""
build_aot_from_seg.py — 从三层分割标注 JSON 构建 AOT (Action Ordering Task) MCQ 数据。

直接复用 seg annotation 的 L2 events 和 L3 grounding_results，
消除独立的 VLM captioning 步骤。

两种新任务类型:
  - seg_aot_event_shuffle  (L2-based): 给 L2 window clip，判断哪个事件列表顺序正确
  - seg_aot_action_reverse (L3-based): 给 L3 event clip，判断哪个动作列表顺序正确

用法:
    python proxy_data/temporal_aot/build_aot_from_seg.py \\
        --annotation-dir /path/to/annotations \\
        --clip-dir-l2 /path/to/clips/L2 \\
        --clip-dir-l3 /path/to/clips/L3 \\
        --output-dir /path/to/output \\
        --tasks event_shuffle action_reverse \\
        --complete-only
"""

import argparse
import json
import os
import random
from pathlib import Path


# =====================================================================
# 常量 — 与 build_hier_data.py 保持一致
# =====================================================================
L2_WINDOW_SIZE = 128
L2_STRIDE = 64
L3_MAX_CLIP_SEC = 128
L3_PADDING = 5


# =====================================================================
# Prompt 模板
# =====================================================================
_EVENT_SHUFFLE_PROMPT = """\
Watch this {duration}s cooking video.

Which numbered list correctly describes the temporal order of cooking events visible in this video?

A.
{option_a}

B.
{option_b}

Think step by step inside <think></think> tags, then provide your final answer \
(A or B) inside <answer></answer> tags."""


_ACTION_REVERSE_PROMPT = """\
Watch this {duration}s cooking video clip.

Which numbered list correctly describes the temporal order of atomic cooking actions in this video?

A.
{option_a}

B.
{option_b}

Think step by step inside <think></think> tags, then provide your final answer \
(A or B) inside <answer></answer> tags."""


def _format_list(items: list[str]) -> str:
    """将 list of str 格式化为缩进的编号列表。"""
    return "\n".join(f"   {i + 1}. {item}" for i, item in enumerate(items))


# =====================================================================
# 公用: 滑窗生成（与 build_hier_data.py 相同逻辑）
# =====================================================================
def generate_sliding_windows(
    total_duration: float,
    window_size: int = L2_WINDOW_SIZE,
    stride: int = L2_STRIDE,
) -> list[tuple[int, int]]:
    windows = []
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


# =====================================================================
# 公用: L3 clip 窗口计算（与 build_hier_data.py 相同逻辑）
# =====================================================================
def _compute_l3_clip(
    ev_start: int,
    ev_end: int,
    clip_duration: int,
    padding: int = L3_PADDING,
    max_clip: int = L3_MAX_CLIP_SEC,
) -> tuple[int, int, int]:
    """返回 (clip_start, clip_end, duration)。"""
    clip_start = max(0, ev_start - padding)
    clip_end = min(clip_duration, ev_end + padding)
    if clip_end - clip_start > max_clip:
        excess = (clip_end - clip_start) - max_clip
        trim_start = min(excess // 2, ev_start - clip_start)
        clip_start += trim_start
        clip_end = clip_start + max_clip
    return clip_start, clip_end, clip_end - clip_start


# =====================================================================
# JSONL 写入
# =====================================================================
def write_jsonl(records: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# =====================================================================
# Task 1: seg_aot_event_shuffle (L2-based)
# =====================================================================
def build_event_shuffle_records(
    ann: dict,
    clip_dir_l2: str,
    min_events: int = 3,
    complete_only: bool = False,
    rng: random.Random | None = None,
) -> list[dict]:
    """每个 L2 window 构建一条 event-order MCQ 记录。

    前提: 窗口内有 ≥ min_events 个事件，且打乱后顺序与原始不同。
    """
    if rng is None:
        rng = random.Random(42)

    l2 = ann.get("level2")
    if not l2 or l2.get("_parse_error"):
        return []

    clip_duration = float(ann.get("clip_duration_sec") or 0)
    if clip_duration <= 0:
        return []

    events = l2.get("events", [])
    clip_key = ann.get("clip_key", "")
    video_path = ann.get("source_video_path") or ann.get("video_path", "")
    windows = generate_sliding_windows(clip_duration)

    records = []
    for ws, we in windows:
        duration = we - ws

        # 收集窗口内事件（按 start_time 升序），保留 instruction
        matched_events = []
        for ev in events:
            if not isinstance(ev, dict):
                continue
            ev_start = ev.get("start_time")
            ev_end = ev.get("end_time")
            instruction = ev.get("instruction", "").strip()
            if not isinstance(ev_start, (int, float)) or not isinstance(ev_end, (int, float)):
                continue
            ev_start_i, ev_end_i = int(ev_start), int(ev_end)
            # 事件中心必须落在窗口内
            if ev_start_i >= we or ev_end_i <= ws:
                continue
            if instruction:
                matched_events.append(instruction)

        if len(matched_events) < min_events:
            continue

        # Fisher-Yates shuffle（保证顺序不同于原始）
        shuffled = matched_events[:]
        max_attempts = 10
        for _ in range(max_attempts):
            rng.shuffle(shuffled)
            if shuffled != matched_events:
                break
        if shuffled == matched_events:
            continue   # 极罕见：事件全部相同或长度1

        # 视频路径
        clip_filename = f"{clip_key}_L2_w{ws}_{we}.mp4"
        vp = os.path.join(clip_dir_l2, clip_filename) if clip_dir_l2 else video_path
        if complete_only and clip_dir_l2 and not os.path.exists(vp):
            continue

        # 随机化 A/B 答案
        if rng.random() < 0.5:
            option_a_type = "forward"
            option_a_descs = matched_events
            option_b_descs = shuffled
        else:
            option_a_type = "shuffled"
            option_a_descs = shuffled
            option_b_descs = matched_events

        answer = "A" if option_a_type == "forward" else "B"
        prompt_body = _EVENT_SHUFFLE_PROMPT.format(
            duration=duration,
            option_a=_format_list(option_a_descs),
            option_b=_format_list(option_b_descs),
        )
        prompt = f"<video>\n\n{prompt_body}"

        records.append({
            "messages": [{"role": "user", "content": prompt}],
            "prompt": prompt,
            "answer": answer,
            "videos": [vp],
            "data_type": "video",
            "problem_type": "seg_aot_event_shuffle",
            "metadata": {
                "clip_key": clip_key,
                "window_start_sec": ws,
                "window_end_sec": we,
                "n_events": len(matched_events),
                "forward_descriptions": matched_events,
                "alt_descriptions": shuffled,
                "option_a_type": option_a_type,
                "source": "seg_annotation",
            },
        })

    return records


# =====================================================================
# Task 2: seg_aot_action_reverse (L3-based)
# =====================================================================
def build_action_reverse_records(
    ann: dict,
    clip_dir_l3: str,
    min_actions: int = 3,
    complete_only: bool = False,
    rng: random.Random | None = None,
) -> list[dict]:
    """每个 L2 event 构建一条 action-order MCQ 记录。

    前提: event 至少有 min_actions 个 grounding results。
    错误选项为完全逆序（len≥2 时保证不同）。
    """
    if rng is None:
        rng = random.Random(42)

    l2 = ann.get("level2")
    l3 = ann.get("level3")
    if not l2 or not l3 or l3.get("_parse_error"):
        return []

    clip_duration = float(ann.get("clip_duration_sec") or 0)
    if clip_duration <= 0:
        return []

    clip_key = ann.get("clip_key", "")
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
        ev_start_i, ev_end_i = int(ev_start), int(ev_end)

        # 收集并按时间排序 grounding results
        raw = sorted(
            [r for r in all_results
             if isinstance(r, dict) and r.get("parent_event_id") == event_id],
            key=lambda r: r.get("start_time", 0),
        )
        if len(raw) < min_actions:
            continue

        sub_actions = [r.get("sub_action", "").strip() for r in raw]
        sub_actions = [s for s in sub_actions if s]
        if len(sub_actions) < min_actions:
            continue

        reversed_actions = list(reversed(sub_actions))
        # len≥2 时倒序一定不同，无需检查

        # clip 范围（与 build_hier_data.py 一致）
        clip_start, clip_end, duration = _compute_l3_clip(
            ev_start_i, ev_end_i, int(clip_duration),
        )

        # 视频路径
        clip_filename = f"{clip_key}_L3_ev{event_id}_{clip_start}_{clip_end}.mp4"
        vp = os.path.join(clip_dir_l3, clip_filename) if clip_dir_l3 else video_path
        if complete_only and clip_dir_l3 and not os.path.exists(vp):
            continue

        # 随机化 A/B 答案
        if rng.random() < 0.5:
            option_a_type = "forward"
            option_a_descs = sub_actions
            option_b_descs = reversed_actions
        else:
            option_a_type = "reversed"
            option_a_descs = reversed_actions
            option_b_descs = sub_actions

        answer = "A" if option_a_type == "forward" else "B"
        prompt_body = _ACTION_REVERSE_PROMPT.format(
            duration=duration,
            option_a=_format_list(option_a_descs),
            option_b=_format_list(option_b_descs),
        )
        prompt = f"<video>\n\n{prompt_body}"

        records.append({
            "messages": [{"role": "user", "content": prompt}],
            "prompt": prompt,
            "answer": answer,
            "videos": [vp],
            "data_type": "video",
            "problem_type": "seg_aot_action_reverse",
            "metadata": {
                "clip_key": clip_key,
                "parent_event_id": event_id,
                "event_start_sec": ev_start_i,
                "event_end_sec": ev_end_i,
                "clip_start_sec": clip_start,
                "clip_end_sec": clip_end,
                "n_actions": len(sub_actions),
                "forward_descriptions": sub_actions,
                "alt_descriptions": reversed_actions,
                "option_a_type": option_a_type,
                "source": "seg_annotation",
            },
        })

    return records


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="从三层分割标注 JSON 构建 AOT MCQ 训练数据",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--annotation-dir", required=True,
                        help="标注 JSON 目录 (annotations/*.json)")
    parser.add_argument("--clip-dir-l2", default="",
                        help="L2 clips 目录 (clips/L2/)，留空则用原始视频路径")
    parser.add_argument("--clip-dir-l3", default="",
                        help="L3 clips 目录 (clips/L3/)，留空则用原始视频路径")
    parser.add_argument("--output-dir", required=True,
                        help="输出目录，生成 train.jsonl / val.jsonl / stats.json")
    parser.add_argument(
        "--tasks", nargs="+",
        choices=["event_shuffle", "action_reverse"],
        default=["event_shuffle", "action_reverse"],
        help="要构建的任务类型",
    )
    parser.add_argument("--min-events", type=int, default=3,
                        help="event_shuffle: 每窗口最少事件数")
    parser.add_argument("--min-actions", type=int, default=3,
                        help="action_reverse: 每事件最少 action 数")
    parser.add_argument("--total-val", type=int, default=200,
                        help="总验证集样本数（按 task 均分）")
    parser.add_argument("--train-per-task", type=int, default=-1,
                        help="每个任务最多 train 条数 (-1 = 无限制)")
    parser.add_argument("--complete-only", action="store_true",
                        help="跳过 clip 文件不存在的记录")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    ann_dir = Path(args.annotation_dir)
    if not ann_dir.exists():
        print(f"ERROR: annotation-dir not found: {ann_dir}")
        return

    ann_files = sorted(ann_dir.glob("*.json"))
    print(f"Found {len(ann_files)} annotation files")

    # 按 task 收集记录
    TASK_BUILDERS = {
        "event_shuffle":  lambda ann: build_event_shuffle_records(
            ann, args.clip_dir_l2, args.min_events, args.complete_only,
            random.Random(rng.random()),
        ),
        "action_reverse": lambda ann: build_action_reverse_records(
            ann, args.clip_dir_l3, args.min_actions, args.complete_only,
            random.Random(rng.random()),
        ),
    }

    task_records: dict[str, list[dict]] = {t: [] for t in args.tasks}
    clip_counts: dict[str, int] = {t: 0 for t in args.tasks}

    for af in ann_files:
        try:
            with open(af, encoding="utf-8") as f:
                ann = json.load(f)
        except Exception as e:
            print(f"  SKIP (parse error): {af.name}: {e}")
            continue

        for task in args.tasks:
            recs = TASK_BUILDERS[task](ann)
            if recs:
                clip_counts[task] += 1
                task_records[task].extend(recs)

    # 统计
    print(f"\n=== Extraction Stats ===")
    for task in args.tasks:
        print(f"  {task}: {clip_counts[task]} source clips, {len(task_records[task])} records")

    # 分割 train / val
    n_tasks = len(args.tasks)
    val_per_task = max(1, args.total_val // n_tasks)
    train_per_task = None if args.train_per_task < 0 else args.train_per_task

    all_train: list[dict] = []
    all_val: list[dict] = []

    for task in args.tasks:
        records = task_records[task]
        rng.shuffle(records)

        n_val = min(val_per_task, len(records) // 5)
        val = records[:n_val]
        train_pool = records[n_val:]

        train = train_pool[:train_per_task] if train_per_task is not None else train_pool

        all_val.extend(val)
        all_train.extend(train)
        print(f"  {task}: {len(train)} train + {len(val)} val")

    rng.shuffle(all_train)
    rng.shuffle(all_val)

    # 写出
    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.jsonl")
    val_path = os.path.join(args.output_dir, "val.jsonl")
    write_jsonl(all_train, train_path)
    write_jsonl(all_val, val_path)

    # stats.json
    from collections import Counter
    train_types = dict(Counter(r["problem_type"] for r in all_train))
    val_types = dict(Counter(r["problem_type"] for r in all_val))
    stats = {
        "total_annotation_files": len(ann_files),
        "tasks": args.tasks,
        "train_total": len(all_train),
        "val_total": len(all_val),
        "train_by_type": train_types,
        "val_by_type": val_types,
    }
    stats_path = os.path.join(args.output_dir, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n=== Output ===")
    print(f"  Train: {len(all_train)}  →  {train_path}")
    print(f"  Val:   {len(all_val)}  →  {val_path}")
    print(f"  Stats: {stats_path}")

    if all_train:
        ex = all_train[0]
        print(f"\n=== Example record ===")
        print(f"  problem_type: {ex['problem_type']}")
        print(f"  answer: {ex['answer']}")
        print(f"  video: {ex['videos'][0] if ex['videos'] else 'N/A'}")
        print(f"  prompt (first 300 chars):\n  {ex['prompt'][:300]}")


if __name__ == "__main__":
    main()
