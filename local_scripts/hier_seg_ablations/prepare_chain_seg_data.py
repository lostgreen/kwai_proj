"""
prepare_chain_seg_data.py — Chain-of-Segment (L2+L3 联合) 消融实验数据准备

将 L2 和 L3 标注按 (clip_key, event 时间) 关联，生成链式层次分割训练数据。

每条样本:
    - 视频: L2 的 128s 窗口视频
    - 输入: 视频 + 事件描述列表（来自 L3 的 action_query）
    - GT:   L2 grounding 段 + L3 per-event 原子动作段（全部为窗口相对坐标）
    - problem_type: "temporal_seg_chain_L2L3"

坐标系统:
    - L2 事件: 窗口相对 (0 ~ window_duration)
    - L3 原始: clip 相对 (0 ~ clip_duration)
    - L3 → 绝对: l3_abs = l3_clip_rel + clip_offset_sec
    - L3 → 窗口相对: l3_win_rel = l3_abs - window_start_sec

用法:
    python prepare_chain_seg_data.py --total-val 200 --output-dir ./data/chain_seg
"""

import argparse
import json
import os
import random
import re
from collections import defaultdict


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DEFAULT_DATA_ROOT = os.path.join(REPO_ROOT, "proxy_data", "youcook2_seg_annotation", "datasets")

SEGMENT_PATTERN = re.compile(
    r"\[\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\]"
)


def parse_segments_from_events(text):
    """从 <events>...</events> 中提取 [start, end] 列表。"""
    m = re.search(r"<events>(.*?)</events>", text, re.DOTALL)
    if not m:
        return []
    segs = []
    for sm in SEGMENT_PATTERN.finditer(m.group(1)):
        s, e = float(sm.group(1)), float(sm.group(2))
        if s < e and s >= 0:
            segs.append([s, e])
    return segs


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_chain_seg_prompt(event_descriptions, duration):
    """生成 Chain-of-Segment 训练 prompt。"""
    event_list = "\n".join(f'{i + 1}. "{d}"' for i, d in enumerate(event_descriptions))
    return (
        f"Watch the following cooking video clip carefully:\n<video>\n\n"
        f"You are given a {duration}s cooking video clip and a list of cooking events that occur in it. "
        f"Your task has two steps:\n"
        f"1. **Locate** each event's time segment in the clip (L2 grounding).\n"
        f"2. **Decompose** each event into its atomic cooking actions (L3 segmentation).\n\n"
        f"Events to locate and decompose:\n{event_list}\n\n"
        f"Rules:\n"
        f"- L2: output one [start_time, end_time] per event, in the given order\n"
        f"- L3: for each event, output the atomic actions as [[start, end], ...] within that event's time range\n"
        f"- all timestamps are integer seconds (0-based, 0 ≤ start < end ≤ {duration})\n"
        f"- atomic actions are brief (2-6s) physical state changes (cutting, pouring, stirring, etc.)\n"
        f"- skip idle/narration within events; gaps between atomic actions are fine\n\n"
        f"Output format:\n"
        f"<l2_events>[[start, end], ...]</l2_events>\n"
        f"<l3_events>[[[start, end], ...], [[start, end], ...], ...]</l3_events>\n\n"
        f"Example (2 events, first has 3 atomic actions, second has 2):\n"
        f"<l2_events>[[5, 30], [35, 55]]</l2_events>\n"
        f"<l3_events>[[[5, 10], [12, 20], [22, 30]], [[35, 42], [45, 55]]]</l3_events>"
    )


def build_chain_seg_answer(l2_segs, l3_segs_nested):
    """
    构建 GT answer 字符串。

    l2_segs: [[s1, e1], [s2, e2], ...] — 窗口相对
    l3_segs_nested: [[[a1,b1], [a2,b2]], [[c1,d1]], ...] — 窗口相对
    """
    l2_str = "[" + ", ".join(f"[{int(s)}, {int(e)}]" for s, e in l2_segs) + "]"
    l3_parts = []
    for event_segs in l3_segs_nested:
        inner = "[" + ", ".join(f"[{int(s)}, {int(e)}]" for s, e in event_segs) + "]"
        l3_parts.append(inner)
    l3_str = "[" + ", ".join(l3_parts) + "]"
    return f"<l2_events>{l2_str}</l2_events>\n<l3_events>{l3_str}</l3_events>"


def main():
    parser = argparse.ArgumentParser(description="Chain-of-Segment 数据准备")
    parser.add_argument("--total-val", type=int, default=200)
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--min-events", type=int, default=2,
                        help="每个样本最少需要的 matched 事件数")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # ========================================
    # 1. 加载 L2 数据
    # ========================================
    l2_path = os.path.join(args.data_root, "youcook2_hier_L2_train_clipped.jsonl")
    l2_records = load_jsonl(l2_path)
    print(f"Loaded {len(l2_records)} L2 records")

    # ========================================
    # 2. 加载 L3 数据 (仅 sequential, 非 shuffled)
    # ========================================
    l3_path = os.path.join(args.data_root, "youcook2_hier_L3_train_clipped.jsonl")
    l3_records = load_jsonl(l3_path)

    # 按 (clip_key, event_start_abs, event_end_abs) 索引 L3
    # L3 的 event_start_sec / event_end_sec 是绝对坐标
    l3_index = {}
    for rec in l3_records:
        m = rec["metadata"]
        if m.get("shuffled", False):
            continue
        key = (m["clip_key"], m["event_start_sec"], m["event_end_sec"])
        l3_index[key] = rec

    print(f"Loaded {len(l3_records)} L3 records, indexed {len(l3_index)} seq entries")

    # ========================================
    # 3. 关联 L2 和 L3，构建 Chain-of-Segment 数据
    # ========================================
    chain_records = []
    match_stats = {"windows": 0, "events_matched": 0, "events_total": 0}

    for l2_rec in l2_records:
        l2_meta = l2_rec["metadata"]
        clip_key = l2_meta["clip_key"]
        ws = l2_meta["window_start_sec"]
        we = l2_meta["window_end_sec"]
        duration = we - ws

        # 解析 L2 事件（窗口相对坐标）
        l2_segs = parse_segments_from_events(l2_rec["answer"])
        if not l2_segs:
            continue

        match_stats["events_total"] += len(l2_segs)

        # 尝试匹配每个 L2 事件到 L3
        matched_events = []
        for l2_seg in l2_segs:
            # L2 窗口相对 → 绝对
            abs_start = ws + l2_seg[0]
            abs_end = ws + l2_seg[1]
            l3_key = (clip_key, abs_start, abs_end)

            l3_rec = l3_index.get(l3_key)
            if l3_rec is None:
                continue

            l3_meta = l3_rec["metadata"]
            clip_offset = l3_meta.get("clip_offset_sec", 0)

            # 解析 L3 片段（L3 clip 相对坐标）
            l3_segs_clip_rel = parse_segments_from_events(l3_rec["answer"])
            if not l3_segs_clip_rel:
                continue

            # L3 clip 相对 → 绝对 → 窗口相对
            l3_segs_win_rel = []
            for seg in l3_segs_clip_rel:
                s_abs = seg[0] + clip_offset
                e_abs = seg[1] + clip_offset
                s_win = s_abs - ws
                e_win = e_abs - ws
                # 裁剪到窗口范围
                s_win = max(0, s_win)
                e_win = min(duration, e_win)
                if s_win < e_win:
                    l3_segs_win_rel.append([s_win, e_win])

            if not l3_segs_win_rel:
                continue

            matched_events.append({
                "l2_seg": l2_seg,
                "l3_segs": l3_segs_win_rel,
                "action_query": l3_meta.get("action_query", ""),
            })

        if len(matched_events) < args.min_events:
            continue

        match_stats["windows"] += 1
        match_stats["events_matched"] += len(matched_events)

        # 构建 Chain-of-Segment 样本
        event_descriptions = [e["action_query"] for e in matched_events]
        l2_segs_matched = [e["l2_seg"] for e in matched_events]
        l3_segs_nested = [e["l3_segs"] for e in matched_events]

        prompt_text = build_chain_seg_prompt(event_descriptions, duration)
        answer_text = build_chain_seg_answer(l2_segs_matched, l3_segs_nested)

        chain_rec = {
            "messages": [{"role": "user", "content": prompt_text}],
            "prompt": prompt_text,
            "answer": answer_text,
            "videos": l2_rec["videos"],
            "data_type": "video",
            "problem_type": "temporal_seg_chain_L2L3",
            "metadata": {
                "clip_key": clip_key,
                "window_start_sec": ws,
                "window_end_sec": we,
                "n_matched_events": len(matched_events),
                "event_descriptions": event_descriptions,
                "source_l2_window": f"w{ws}_{we}",
            },
        }
        chain_records.append(chain_rec)

    print(f"\n=== Match Stats ===")
    print(f"  L2 windows with ≥{args.min_events} matched events: {match_stats['windows']}")
    print(f"  Total matched events: {match_stats['events_matched']}")
    print(f"  Total L2 events scanned: {match_stats['events_total']}")

    # ========================================
    # 4. Split train/val and write
    # ========================================
    rng.shuffle(chain_records)
    n_val = min(args.total_val, len(chain_records) // 5)
    val_records = chain_records[:n_val]
    train_records = chain_records[n_val:]

    train_path = os.path.join(args.output_dir, "train.jsonl")
    val_path = os.path.join(args.output_dir, "val.jsonl")
    write_jsonl(train_records, train_path)
    write_jsonl(val_records, val_path)

    print(f"\n=== Output ===")
    print(f"  Dir: {args.output_dir}")
    print(f"  Train: {len(train_records)}")
    print(f"  Val: {len(val_records)}")
    print(f"  Avg events/sample: {match_stats['events_matched'] / max(match_stats['windows'], 1):.1f}")

    # 打印一条样本示例
    if train_records:
        ex = train_records[0]
        print(f"\n=== Sample ===")
        print(f"  video: {ex['videos'][0]}")
        print(f"  n_events: {ex['metadata']['n_matched_events']}")
        print(f"  descriptions: {ex['metadata']['event_descriptions'][:2]}...")
        print(f"  answer (first 300): {ex['answer'][:300]}")


if __name__ == "__main__":
    main()
