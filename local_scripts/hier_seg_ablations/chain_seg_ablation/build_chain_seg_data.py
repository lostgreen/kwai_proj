#!/usr/bin/env python3
"""
build_chain_seg_data.py — 直接从原始标注 JSON 构建 Chain-of-Segment 训练数据。

仅支持 V2 (ground-seg): 单 caption grounding + 单事件内 L3 分割。
每个窗口内的每个 matched event 独立拆为一条训练样本。

problem_type: temporal_seg_chain_ground_seg

坐标系统: 全部输出为窗口相对坐标 (0 ~ window_duration)

用法:
    python build_chain_seg_data.py \
        --annotation-dir /path/to/annotations \
        --clip-dir /path/to/clips/L2 \
        --output-dir ./data/chain_seg
"""

import argparse
import json
import os
import random
from pathlib import Path


# =====================================================================
# Prompt 模板 — V2: ground-seg (单 caption, 单事件)
# =====================================================================

def _build_v2_prompt(event_description: str, duration: int) -> str:
    return (
        f"Watch the following video clip carefully:\n<video>\n\n"
        f"You are given a {duration}s video clip of a procedural activity and "
        f"a description of one event that occurs in it.\n\n"
        f'Event description: "{event_description}"\n\n'
        f"Your task has two steps:\n"
        f"1. **Locate** the described event's time segment in the clip.\n"
        f"2. **Decompose** the event into its atomic actions "
        f"(fine-grained sub-segments within the event).\n\n"
        f"Rules:\n"
        f"- L2: output exactly one [start_time, end_time] for the described event\n"
        f"- L3: output the atomic actions as [[start, end], ...] within the event's time range\n"
        f"- all timestamps are integer seconds (0-based, 0 ≤ start < end ≤ {duration})\n"
        f"- atomic actions are brief (2-6s) physical state changes\n"
        f"- skip idle or narration; gaps between atomic actions are fine\n\n"
        f"Output format:\n"
        f"<l2_events>[[start, end]]</l2_events>\n"
        f"<l3_events>[[[start, end], ...]]</l3_events>\n\n"
        f"Example (event located at 10-45s with 3 atomic actions):\n"
        f"<l2_events>[[10, 45]]</l2_events>\n"
        f"<l3_events>[[[12, 18], [20, 30], [35, 43]]]</l3_events>"
    )


# =====================================================================
# Answer 构建
# =====================================================================

def _build_answer(l2_segs: list[list[int]], l3_nested: list[list[list[int]]]) -> str:
    l2_str = "[" + ", ".join(f"[{s}, {e}]" for s, e in l2_segs) + "]"
    l3_parts = []
    for event_segs in l3_nested:
        inner = "[" + ", ".join(f"[{s}, {e}]" for s, e in event_segs) + "]"
        l3_parts.append(inner)
    l3_str = "[" + ", ".join(l3_parts) + "]"
    return f"<l2_events>{l2_str}</l2_events>\n<l3_events>{l3_str}</l3_events>"


# =====================================================================
# 滑窗生成
# =====================================================================

def generate_sliding_windows(
    total_duration: float,
    window_size: int = 128,
    stride: int = 64,
) -> list[tuple[int, int]]:
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


# =====================================================================
# 核心: 从单个标注 JSON 提取 chain seg 数据
# =====================================================================

def extract_chain_data_from_annotation(
    ann: dict,
    window_size: int = 128,
    stride: int = 64,
    min_events: int = 2,
    min_l3_actions: int = 3,
    clip_dir: str = "",
) -> list[dict]:
    """从单个原始标注 JSON 提取所有 matched window 数据。"""
    l2 = ann.get("level2")
    l3 = ann.get("level3")
    if not l2 or not l3 or l2.get("_parse_error") or l3.get("_parse_error"):
        return []

    clip_duration = float(ann.get("clip_duration_sec") or 0)
    if clip_duration <= 0:
        return []

    clip_key = ann.get("clip_key", "")
    l2_events = l2.get("events", [])
    l3_results = l3.get("grounding_results", [])

    # 按 parent_event_id 索引 L3 results
    l3_by_event: dict[int, list[dict]] = {}
    for r in l3_results:
        if not isinstance(r, dict):
            continue
        pid = r.get("parent_event_id")
        if pid is not None:
            l3_by_event.setdefault(pid, []).append(r)

    # 对每个 event 的 L3 results 按时间排序
    for pid in l3_by_event:
        l3_by_event[pid].sort(key=lambda x: x.get("start_time", 0))

    windows = generate_sliding_windows(clip_duration, window_size, stride)
    results = []

    for ws, we in windows:
        duration = we - ws

        matched_events = []
        for ev in l2_events:
            if not isinstance(ev, dict):
                continue
            ev_id = ev.get("event_id")
            ev_start = ev.get("start_time")
            ev_end = ev.get("end_time")
            instruction = ev.get("instruction", "")

            if not isinstance(ev_start, (int, float)) or not isinstance(ev_end, (int, float)):
                continue

            ev_start, ev_end = int(ev_start), int(ev_end)

            # 检查事件是否与窗口重叠
            if ev_start >= we or ev_end <= ws:
                continue

            # 裁剪 L2 事件到窗口边界 → 窗口相对坐标
            clipped_start = max(ev_start, ws)
            clipped_end = min(ev_end, we)
            l2_seg_win = [clipped_start - ws, clipped_end - ws]

            # 获取该事件的 L3 grounding results
            event_l3 = l3_by_event.get(ev_id, [])
            if len(event_l3) < min_l3_actions:
                continue

            # 裁剪 L3 到裁剪后的 L2 边界 → 窗口相对坐标
            l3_segs_win = []
            for r in event_l3:
                a_start = r.get("start_time")
                a_end = r.get("end_time")
                if not isinstance(a_start, (int, float)) or not isinstance(a_end, (int, float)):
                    continue
                a_start, a_end = int(a_start), int(a_end)

                # 裁剪到 L2 事件的裁剪边界
                s = max(a_start, clipped_start)
                e = min(a_end, clipped_end)
                if s < e:
                    l3_segs_win.append([s - ws, e - ws])

            if not l3_segs_win:
                continue

            matched_events.append({
                "l2_seg": l2_seg_win,
                "l3_segs": l3_segs_win,
                "l2_caption": instruction,
            })

        if len(matched_events) < min_events:
            continue

        # 构建视频路径
        if clip_dir:
            video_path = os.path.join(clip_dir, f"{clip_key}_L2_w{ws}_{we}.mp4")
        else:
            video_path = ann.get("source_video_path") or ann.get("video_path", "")

        results.append({
            "clip_key": clip_key,
            "window_start": ws,
            "window_end": we,
            "duration": duration,
            "matched_events": matched_events,
            "video_path": video_path,
        })

    return results


# =====================================================================
# 从 window data → V2 记录
# =====================================================================

def build_v2_records(window_data: list[dict]) -> list[dict]:
    """V2: ground-seg (单 caption, 单事件 per 样本)"""
    records = []
    for w in window_data:
        for ev in w["matched_events"]:
            prompt = _build_v2_prompt(ev["l2_caption"], w["duration"])
            answer = _build_answer([ev["l2_seg"]], [ev["l3_segs"]])

            records.append({
                "messages": [{"role": "user", "content": prompt}],
                "prompt": prompt,
                "answer": answer,
                "videos": [w["video_path"]],
                "data_type": "video",
                "problem_type": "temporal_seg_chain_ground_seg",
                "metadata": {
                    "clip_key": w["clip_key"],
                    "window_start_sec": w["window_start"],
                    "window_end_sec": w["window_end"],
                    "event_description": ev["l2_caption"],
                    "source": "annotation_json",
                },
            })
    return records


# =====================================================================
# Main
# =====================================================================

def write_jsonl(records: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="从原始标注 JSON 直接构建 Chain-of-Segment 训练数据 (V2 ground-seg)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--annotation-dir", required=True,
                        help="标注 JSON 目录 (annotations/*.json)")
    parser.add_argument("--clip-dir", default="",
                        help="已有 L2 clips 目录 (clips/L2/)，留空则使用原始视频路径")
    parser.add_argument("--output-dir", required=True,
                        help="输出目录")
    parser.add_argument("--window-size", type=int, default=128,
                        help="滑窗大小 (秒)")
    parser.add_argument("--stride", type=int, default=64,
                        help="滑窗步长 (秒)")
    parser.add_argument("--min-events", type=int, default=2,
                        help="每个窗口最少 matched 事件数")
    parser.add_argument("--min-l3-actions", type=int, default=3,
                        help="每个 L2 事件最少 L3 动作数")
    parser.add_argument("--total-val", type=int, default=100,
                        help="验证集样本数")
    parser.add_argument("--complete-only", action="store_true",
                        help="仅处理 L1+L2+L3 均完整的 clip")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    ann_dir = Path(args.annotation_dir)
    if not ann_dir.exists():
        print(f"ERROR: annotation-dir not found: {ann_dir}")
        return

    ann_files = sorted(ann_dir.glob("*.json"))
    print(f"Found {len(ann_files)} annotation files")

    # 过滤只保留完整标注
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
        print(f"  --complete-only: {len(filtered)} clips have L1+L2+L3")
        ann_files = filtered

    # 提取所有 window 数据
    all_window_data: list[dict] = []
    stats = {"clips": 0, "windows": 0, "events": 0}

    for af in ann_files:
        try:
            with open(af, encoding="utf-8") as f:
                ann = json.load(f)
        except Exception as e:
            print(f"  SKIP (parse error): {af.name}: {e}")
            continue

        windows = extract_chain_data_from_annotation(
            ann,
            window_size=args.window_size,
            stride=args.stride,
            min_events=args.min_events,
            min_l3_actions=args.min_l3_actions,
            clip_dir=args.clip_dir,
        )
        if windows:
            stats["clips"] += 1
            stats["windows"] += len(windows)
            for w in windows:
                stats["events"] += len(w["matched_events"])
            all_window_data.extend(windows)

    print(f"\n=== Extraction Stats ===")
    print(f"  Clips with data: {stats['clips']}")
    print(f"  Windows: {stats['windows']}")
    print(f"  Total matched events: {stats['events']}")
    print(f"  Avg events/window: {stats['events'] / max(stats['windows'], 1):.1f}")

    # 生成 V2 数据
    records = build_v2_records(all_window_data)
    rng.shuffle(records)

    n_val = min(args.total_val, len(records) // 5)
    val_records = records[:n_val]
    train_records = records[n_val:]

    train_path = os.path.join(args.output_dir, "chain_ground_seg_train.jsonl")
    val_path = os.path.join(args.output_dir, "chain_ground_seg_val.jsonl")
    write_jsonl(train_records, train_path)
    write_jsonl(val_records, val_path)

    print(f"\n=== V2 (ground-seg) ===")
    print(f"  Train: {len(train_records)}")
    print(f"  Val:   {len(val_records)}")

    if train_records:
        ex = train_records[0]
        print(f"  Sample video: {ex['videos'][0]}")
        print(f"  Sample answer (first 200): {ex['answer'][:200]}")

    print(f"\nOutput dir: {args.output_dir}")


if __name__ == "__main__":
    main()
