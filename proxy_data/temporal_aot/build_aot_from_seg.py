#!/usr/bin/env python3
"""
build_aot_from_seg.py — 从三层分割标注 JSON 构建 AOT (Action Ordering Task) MCQ 数据。

直接复用 seg annotation 的 L2 events 和 L3 grounding_results，
消除独立的 VLM captioning 步骤。

四种任务类型（2×2 factorial: 粒度 × 方向）:

  L3 (action) 层:
  - seg_aot_action_v2t: 给 L3 event clip，判断哪个动作列表顺序正确     (A/B)
  - seg_aot_action_t2v: 给 forward 动作列表，从两个 L3 clip 中选匹配的 (A/B)

  L2 (event) 层:
  - seg_aot_event_v2t:  给 L2 window clip，判断哪个事件列表顺序正确    (A/B/C)
  - seg_aot_event_t2v:  给 forward 事件列表，从三个 L2 clip 中选匹配的 (A/B/C)

用法:
    python proxy_data/temporal_aot/build_aot_from_seg.py \\
        --annotation-dir /path/to/annotations \\
        --clip-dir-l2 /path/to/clips/L2 \\
        --clip-dir-l3 /path/to/clips/L3 \\
        --output-dir /path/to/output \\
        --tasks action_v2t action_t2v event_v2t event_t2v \\
        --complete-only
"""

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path


# =====================================================================
# 常量 — 与 build_hier_data.py 保持一致
# =====================================================================
L2_WINDOW_SIZE = 128
L2_STRIDE = 64
L3_MAX_CLIP_SEC = 128
L3_PADDING = 5

ANSWER_LETTERS = ["A", "B", "C"]


# =====================================================================
# Prompt 模板
# =====================================================================
_ACTION_V2T_PROMPT = """\
Watch this {duration}s cooking video clip.

Which numbered list correctly describes the temporal order of atomic cooking actions in this video?

A.
{option_a}

B.
{option_b}

Think step by step inside <think></think> tags, then provide your final answer \
(A or B) inside <answer></answer> tags."""


_ACTION_T2V_PROMPT = """\
Here are two {duration}s cooking video clips (Clip A and Clip B).

The atomic actions below were performed in this exact order:
{forward_list}

Which clip (A or B) shows these actions in the listed order?

Think step by step inside <think></think> tags, then provide your final answer \
(A or B) inside <answer></answer> tags."""


_EVENT_V2T_PROMPT = """\
Watch this {duration}s cooking video.

Which numbered list correctly describes the temporal order of cooking events visible in this video?

A.
{option_a}

B.
{option_b}

C.
{option_c}

Think step by step inside <think></think> tags, then provide your final answer \
(A, B, or C) inside <answer></answer> tags."""


_EVENT_T2V_PROMPT = """\
Here are three cooking video clips (Clip A, Clip B, and Clip C).

The cooking events below occurred in this exact order:
{forward_list}

Which clip (A, B, or C) shows these events in the listed order?

Think step by step inside <think></think> tags, then provide your final answer \
(A, B, or C) inside <answer></answer> tags."""


def _format_list(items: list[str]) -> str:
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
# Task 1: seg_aot_action_v2t (L3-based, V2T binary)
# =====================================================================
def build_action_v2t_records(
    ann: dict,
    clip_dir_l3: str,
    min_actions: int = 3,
    complete_only: bool = False,
    rng: random.Random | None = None,
) -> list[dict]:
    """每个 L2 event 构建一条 action-order V2T 二选一记录。

    给视频，判断 forward 还是 reversed 动作列表正确。
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

        raw = sorted(
            [r for r in all_results
             if isinstance(r, dict) and r.get("parent_event_id") == event_id],
            key=lambda r: r.get("start_time", 0),
        )
        sub_actions = [r.get("sub_action", "").strip() for r in raw if r.get("sub_action", "").strip()]
        if len(sub_actions) < min_actions:
            continue

        reversed_actions = list(reversed(sub_actions))

        clip_start, clip_end, duration = _compute_l3_clip(
            ev_start_i, ev_end_i, int(clip_duration),
        )
        clip_filename = f"{clip_key}_L3_ev{event_id}_{clip_start}_{clip_end}.mp4"
        vp = os.path.join(clip_dir_l3, clip_filename) if clip_dir_l3 else ann.get("video_path", "")
        if complete_only and clip_dir_l3 and not os.path.exists(vp):
            continue

        if rng.random() < 0.5:
            option_a_type, option_a_descs, option_b_descs = "forward", sub_actions, reversed_actions
        else:
            option_a_type, option_a_descs, option_b_descs = "reversed", reversed_actions, sub_actions

        answer = "A" if option_a_type == "forward" else "B"
        prompt_body = _ACTION_V2T_PROMPT.format(
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
            "problem_type": "seg_aot_action_v2t",
            "metadata": {
                "clip_key": clip_key,
                "parent_event_id": event_id,
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
# Task 2: seg_aot_action_t2v (L3-based, T2V binary)
# =====================================================================
def build_action_t2v_records(
    ann: dict,
    clip_dir_l3: str,
    min_actions: int = 3,
    complete_only: bool = False,
    rng: random.Random | None = None,
) -> list[dict]:
    """同 clip_key 内两个 L3 event clips 构成一条 action T2V 二选一记录。

    给 forward 动作列表（来自 event_i），从 event_i clip 和 event_j clip 中选匹配的。
    干扰项：同 clip_key 内另一 L2 event 的 L3 clip（actions 不同）。
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
    events = l2.get("events", [])
    all_results = l3.get("grounding_results", [])

    # 先收集所有有效 event 的 actions + clip path
    valid_events = []
    for event in events:
        if not isinstance(event, dict):
            continue
        event_id = event.get("event_id")
        ev_start = event.get("start_time")
        ev_end = event.get("end_time")
        if not isinstance(ev_start, (int, float)) or not isinstance(ev_end, (int, float)):
            continue
        ev_start_i, ev_end_i = int(ev_start), int(ev_end)

        raw = sorted(
            [r for r in all_results
             if isinstance(r, dict) and r.get("parent_event_id") == event_id],
            key=lambda r: r.get("start_time", 0),
        )
        sub_actions = [r.get("sub_action", "").strip() for r in raw if r.get("sub_action", "").strip()]
        if len(sub_actions) < min_actions:
            continue

        clip_start, clip_end, duration = _compute_l3_clip(
            ev_start_i, ev_end_i, int(clip_duration),
        )
        clip_filename = f"{clip_key}_L3_ev{event_id}_{clip_start}_{clip_end}.mp4"
        vp = os.path.join(clip_dir_l3, clip_filename) if clip_dir_l3 else ann.get("video_path", "")
        if complete_only and clip_dir_l3 and not os.path.exists(vp):
            continue

        valid_events.append({
            "event_id": event_id,
            "sub_actions": sub_actions,
            "clip_path": vp,
            "duration": duration,
        })

    # 需要至少 2 个有效 event 才能构建 T2V 对
    if len(valid_events) < 2:
        return []

    records = []
    # 对每个 event_i，选一个不同的 event_j 作为干扰
    for i, target_ev in enumerate(valid_events):
        # 选干扰：随机取 j ≠ i
        other_indices = [j for j in range(len(valid_events)) if j != i]
        j = rng.choice(other_indices)
        distractor_ev = valid_events[j]

        target_clip = target_ev["clip_path"]
        distractor_clip = distractor_ev["clip_path"]
        forward_actions = target_ev["sub_actions"]
        # duration 取两者平均，近似显示给模型
        avg_duration = (target_ev["duration"] + distractor_ev["duration"]) // 2

        # 随机化 A/B 位置
        if rng.random() < 0.5:
            correct_pos, videos = "A", [target_clip, distractor_clip]
        else:
            correct_pos, videos = "B", [distractor_clip, target_clip]

        forward_list_str = _format_list(forward_actions)
        prompt_body = _ACTION_T2V_PROMPT.format(
            duration=avg_duration,
            forward_list=forward_list_str,
        )
        # T2V: multiple video tags
        video_tags = "".join(f"<video>" for _ in videos)
        prompt = f"{video_tags}\n\n{prompt_body}"

        records.append({
            "messages": [{"role": "user", "content": prompt}],
            "prompt": prompt,
            "answer": correct_pos,
            "videos": videos,
            "data_type": "video",
            "problem_type": "seg_aot_action_t2v",
            "metadata": {
                "clip_key": clip_key,
                "target_event_id": target_ev["event_id"],
                "distractor_event_id": distractor_ev["event_id"],
                "n_actions": len(forward_actions),
                "forward_descriptions": forward_actions,
                "correct_position": correct_pos,
                "source": "seg_annotation",
            },
        })

    return records


# =====================================================================
# Task 3: seg_aot_event_v2t (L2-based, V2T 3-way)
# =====================================================================
def build_event_v2t_records(
    ann: dict,
    clip_dir_l2: str,
    min_events: int = 3,
    complete_only: bool = False,
    rng: random.Random | None = None,
) -> list[dict]:
    """每个 L2 window 构建一条 event-order V2T 三选一记录。

    三个选项: forward / shuffle / reversed。
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
    windows = generate_sliding_windows(clip_duration)

    records = []
    for ws, we in windows:
        duration = we - ws

        matched_events = []
        for ev in events:
            if not isinstance(ev, dict):
                continue
            ev_start = ev.get("start_time")
            ev_end = ev.get("end_time")
            instruction = ev.get("instruction", "").strip()
            if not isinstance(ev_start, (int, float)) or not isinstance(ev_end, (int, float)):
                continue
            if int(ev_start) >= we or int(ev_end) <= ws:
                continue
            if instruction:
                matched_events.append(instruction)

        if len(matched_events) < min_events:
            continue

        # 生成 shuffled (Fisher-Yates, 保证 ≠ forward)
        shuffled = matched_events[:]
        for _ in range(10):
            rng.shuffle(shuffled)
            if shuffled != matched_events:
                break
        if shuffled == matched_events:
            continue

        # 生成 reversed (直接倒序，len≥2 时一定 ≠ forward)
        reversed_events = list(reversed(matched_events))

        clip_filename = f"{clip_key}_L2_w{ws}_{we}.mp4"
        vp = os.path.join(clip_dir_l2, clip_filename) if clip_dir_l2 else ann.get("video_path", "")
        if complete_only and clip_dir_l2 and not os.path.exists(vp):
            continue

        # 三选一：随机排列 forward/shuffled/reversed 到 A/B/C
        options = [
            ("forward", matched_events),
            ("shuffled", shuffled),
            ("reversed", reversed_events),
        ]
        rng.shuffle(options)
        correct_letter = ANSWER_LETTERS[[o[0] for o in options].index("forward")]

        prompt_body = _EVENT_V2T_PROMPT.format(
            duration=duration,
            option_a=_format_list(options[0][1]),
            option_b=_format_list(options[1][1]),
            option_c=_format_list(options[2][1]),
        )
        prompt = f"<video>\n\n{prompt_body}"

        records.append({
            "messages": [{"role": "user", "content": prompt}],
            "prompt": prompt,
            "answer": correct_letter,
            "videos": [vp],
            "data_type": "video",
            "problem_type": "seg_aot_event_v2t",
            "metadata": {
                "clip_key": clip_key,
                "window_start_sec": ws,
                "window_end_sec": we,
                "n_events": len(matched_events),
                "forward_descriptions": matched_events,
                "option_a_type": options[0][0],
                "option_b_type": options[1][0],
                "option_c_type": options[2][0],
                "source": "seg_annotation",
            },
        })

    return records


# =====================================================================
# Task 4: seg_aot_event_t2v (L2-based, T2V 3-way)
# =====================================================================
def build_event_t2v_records(
    v2t_pool: list[dict],
    train_per_task: int,
    rng: random.Random,
) -> list[dict]:
    """从全局 event_v2t 池组合三选一 T2V 记录。

    每次取 3 条来自不同 clip_key 的 v2t 记录，
    以第 1 条的 forward 事件描述为问题，3 条的 L2 window clip 为选项。
    """
    # 按 clip_key 分组
    by_clip: dict[str, list[dict]] = defaultdict(list)
    for rec in v2t_pool:
        by_clip[rec["metadata"]["clip_key"]].append(rec)

    clip_keys = list(by_clip.keys())
    if len(clip_keys) < 3:
        return []

    records = []
    # 随机三元组
    rng.shuffle(clip_keys)

    # 最多生成 train_per_task 条（避免过多组合）
    max_records = max(len(v2t_pool), train_per_task if train_per_task > 0 else len(v2t_pool))
    attempts = 0

    while len(records) < max_records and attempts < max_records * 3:
        attempts += 1
        # 不放回随机取 3 个不同的 clip_key
        sampled_keys = rng.sample(clip_keys, k=3)
        recs_3 = [rng.choice(by_clip[k]) for k in sampled_keys]

        target_rec = recs_3[0]
        target_clip = target_rec["videos"][0]
        distractor_clips = [recs_3[1]["videos"][0], recs_3[2]["videos"][0]]
        forward_events = target_rec["metadata"]["forward_descriptions"]

        # 随机化 A/B/C 位置
        clips = [target_clip] + distractor_clips
        slot_order = [0, 1, 2]
        rng.shuffle(slot_order)
        arranged_clips = [clips[s] for s in slot_order]
        correct_pos = ANSWER_LETTERS[slot_order.index(0)]

        forward_list_str = _format_list(forward_events)
        prompt_body = _EVENT_T2V_PROMPT.format(forward_list=forward_list_str)
        video_tags = "".join("<video>" for _ in arranged_clips)
        prompt = f"{video_tags}\n\n{prompt_body}"

        records.append({
            "messages": [{"role": "user", "content": prompt}],
            "prompt": prompt,
            "answer": correct_pos,
            "videos": arranged_clips,
            "data_type": "video",
            "problem_type": "seg_aot_event_t2v",
            "metadata": {
                "target_clip_key": target_rec["metadata"]["clip_key"],
                "distractor_clip_keys": [recs_3[1]["metadata"]["clip_key"], recs_3[2]["metadata"]["clip_key"]],
                "n_events": len(forward_events),
                "forward_descriptions": forward_events,
                "correct_position": correct_pos,
                "source": "seg_annotation",
            },
        })

    return records


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="从三层分割标注 JSON 构建 AOT MCQ 训练数据 (4 种 problem_type)",
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
        choices=["action_v2t", "action_t2v", "event_v2t", "event_t2v"],
        default=["action_v2t", "action_t2v", "event_v2t", "event_t2v"],
        help="要构建的任务类型",
    )
    parser.add_argument("--min-events", type=int, default=3,
                        help="event_*: 每窗口最少事件数")
    parser.add_argument("--min-actions", type=int, default=3,
                        help="action_*: 每事件最少 action 数")
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

    # event_t2v 依赖先建好的 event_v2t 池，确保 event_v2t 先执行
    build_order = []
    if "event_v2t" in args.tasks:
        build_order.append("event_v2t")
    for t in args.tasks:
        if t not in build_order:
            build_order.append(t)

    # 逐文件构建 action/event V2T 记录（T2V 后处理）
    task_records: dict[str, list[dict]] = {t: [] for t in args.tasks}

    for af in ann_files:
        try:
            with open(af, encoding="utf-8") as f:
                ann = json.load(f)
        except Exception as e:
            print(f"  SKIP (parse error): {af.name}: {e}")
            continue

        if "action_v2t" in args.tasks:
            task_records["action_v2t"].extend(
                build_action_v2t_records(
                    ann, args.clip_dir_l3, args.min_actions, args.complete_only,
                    random.Random(rng.random()),
                )
            )
        if "action_t2v" in args.tasks:
            task_records["action_t2v"].extend(
                build_action_t2v_records(
                    ann, args.clip_dir_l3, args.min_actions, args.complete_only,
                    random.Random(rng.random()),
                )
            )
        if "event_v2t" in args.tasks or "event_t2v" in args.tasks:
            task_records.setdefault("event_v2t", []).extend(
                build_event_v2t_records(
                    ann, args.clip_dir_l2, args.min_events, args.complete_only,
                    random.Random(rng.random()),
                )
            )

    # event_t2v 后处理：从 event_v2t 池组合
    if "event_t2v" in args.tasks:
        v2t_pool = task_records.get("event_v2t", [])
        print(f"  Building event_t2v from {len(v2t_pool)} event_v2t records ...")
        task_records["event_t2v"] = build_event_t2v_records(
            v2t_pool,
            args.train_per_task,
            random.Random(rng.random()),
        )
        # 如果 event_v2t 不在输出 tasks 列表里，移除
        if "event_v2t" not in args.tasks:
            del task_records["event_v2t"]

    # 统计
    print(f"\n=== Extraction Stats ===")
    for task in args.tasks:
        print(f"  {task}: {len(task_records.get(task, []))} records")

    # 分割 train / val
    n_tasks = len(args.tasks)
    val_per_task = max(1, args.total_val // n_tasks)
    train_per_task = None if args.train_per_task < 0 else args.train_per_task

    all_train: list[dict] = []
    all_val: list[dict] = []

    for task in args.tasks:
        records = task_records.get(task, [])
        rng.shuffle(records)

        n_val = min(val_per_task, max(1, len(records) // 5))
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
        n_vids = len(ex.get("videos", []))
        print(f"  videos ({n_vids}): {ex['videos'][0] if ex['videos'] else 'N/A'}")
        print(f"  prompt (first 300 chars):\n  {ex['prompt'][:300]}")


if __name__ == "__main__":
    main()
