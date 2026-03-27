"""
prepare_chain_ablation_data.py — Chain-of-Segment 消融实验数据准备

V1 (dual-seg):   复用 L2 128s 窗口，去掉 caption，保留多事件结构
V2 (ground-seg): 拆成单事件样本，每条含 1 个 L2 caption + 对应 L3

坐标系统:
    - L2 事件: 窗口相对 (0 ~ window_duration)
    - L3 原始: clip 相对 (0 ~ clip_duration)
    - L3 → 绝对: l3_abs = l3_clip_rel + clip_offset_sec
    - L3 → 窗口相对: l3_win_rel = l3_abs - window_start_sec
"""

import argparse
import json
import os
import random
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
DEFAULT_DATA_ROOT = os.path.join(
    REPO_ROOT, "proxy_data", "youcook2_seg_annotation", "datasets"
)

# Add parent dir so we can import prompt templates
sys.path.insert(0, SCRIPT_DIR)
from prompt_variants_chain import CHAIN_PROMPT_VARIANTS

SEGMENT_PATTERN = re.compile(
    r"\[\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\]"
)


def parse_segments_from_events(text):
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


def build_answer(l2_segs, l3_segs_nested):
    l2_str = "[" + ", ".join(f"[{int(s)}, {int(e)}]" for s, e in l2_segs) + "]"
    l3_parts = []
    for event_segs in l3_segs_nested:
        inner = "[" + ", ".join(f"[{int(s)}, {int(e)}]" for s, e in event_segs) + "]"
        l3_parts.append(inner)
    l3_str = "[" + ", ".join(l3_parts) + "]"
    return f"<l2_events>{l2_str}</l2_events>\n<l3_events>{l3_str}</l3_events>"


# ======================================================================
# L2+L3 关联（复用 prepare_chain_seg_data.py 逻辑）
# ======================================================================
def load_and_associate(data_root, min_events=2):
    """加载 L2+L3 数据并按 (clip_key, event时间) 关联。

    返回: list of dict, 每条代表一个 128s 窗口 + 其 matched 事件列表。
    """
    l2_path = os.path.join(data_root, "youcook2_hier_L2_train_clipped.jsonl")
    l3_path = os.path.join(data_root, "youcook2_hier_L3_train_clipped.jsonl")

    l2_records = load_jsonl(l2_path)
    l3_records = load_jsonl(l3_path)

    # 索引 L3（仅 sequential）
    l3_index = {}
    for rec in l3_records:
        m = rec["metadata"]
        if m.get("shuffled", False):
            continue
        key = (m["clip_key"], m["event_start_sec"], m["event_end_sec"])
        l3_index[key] = rec

    print(f"Loaded {len(l2_records)} L2, indexed {len(l3_index)} L3 seq entries")

    windows = []
    for l2_rec in l2_records:
        l2_meta = l2_rec["metadata"]
        clip_key = l2_meta["clip_key"]
        ws = l2_meta["window_start_sec"]
        we = l2_meta["window_end_sec"]
        duration = we - ws

        l2_segs = parse_segments_from_events(l2_rec["answer"])
        if not l2_segs:
            continue

        matched_events = []
        for l2_seg in l2_segs:
            abs_start = ws + l2_seg[0]
            abs_end = ws + l2_seg[1]
            l3_rec = l3_index.get((clip_key, abs_start, abs_end))
            if l3_rec is None:
                continue

            l3_meta = l3_rec["metadata"]
            clip_offset = l3_meta.get("clip_offset_sec", 0)
            l3_segs_clip_rel = parse_segments_from_events(l3_rec["answer"])
            if not l3_segs_clip_rel:
                continue

            l3_segs_win_rel = []
            for seg in l3_segs_clip_rel:
                s_win = seg[0] + clip_offset - ws
                e_win = seg[1] + clip_offset - ws
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

        if len(matched_events) < min_events:
            continue

        windows.append({
            "l2_rec": l2_rec,
            "matched_events": matched_events,
            "duration": duration,
            "clip_key": clip_key,
            "ws": ws,
            "we": we,
        })

    print(f"Windows with ≥{min_events} matched events: {len(windows)}")
    return windows


# ======================================================================
# V1: Dual-Seg (multi-event, no captions)
# ======================================================================
def build_v1_records(windows):
    template = CHAIN_PROMPT_VARIANTS["V1"]
    records = []
    for w in windows:
        duration = w["duration"]
        prompt_text = template.format(duration=int(duration))
        l2_segs = [e["l2_seg"] for e in w["matched_events"]]
        l3_nested = [e["l3_segs"] for e in w["matched_events"]]
        answer_text = build_answer(l2_segs, l3_nested)

        records.append({
            "messages": [{"role": "user", "content": prompt_text}],
            "prompt": prompt_text,
            "answer": answer_text,
            "videos": w["l2_rec"]["videos"],
            "data_type": "video",
            "problem_type": "temporal_seg_chain_dual_seg",
            "metadata": {
                "clip_key": w["clip_key"],
                "window_start_sec": w["ws"],
                "window_end_sec": w["we"],
                "n_events": len(w["matched_events"]),
                "variant": "V1",
            },
        })
    return records


# ======================================================================
# V2: Ground-Seg (single-caption, one event per sample)
# ======================================================================
def build_v2_records(windows):
    template = CHAIN_PROMPT_VARIANTS["V2"]
    records = []
    for w in windows:
        duration = w["duration"]
        for ev in w["matched_events"]:
            prompt_text = template.format(
                duration=int(duration),
                event_description=ev["action_query"],
            )
            answer_text = build_answer([ev["l2_seg"]], [ev["l3_segs"]])

            records.append({
                "messages": [{"role": "user", "content": prompt_text}],
                "prompt": prompt_text,
                "answer": answer_text,
                "videos": w["l2_rec"]["videos"],
                "data_type": "video",
                "problem_type": "temporal_seg_chain_ground_seg",
                "metadata": {
                    "clip_key": w["clip_key"],
                    "window_start_sec": w["ws"],
                    "window_end_sec": w["we"],
                    "event_description": ev["action_query"],
                    "variant": "V2",
                },
            })
    return records


def main():
    parser = argparse.ArgumentParser(description="Chain-Seg 消融数据准备")
    parser.add_argument("--variant", type=str, required=True, choices=["V1", "V2"])
    parser.add_argument("--total-val", type=int, default=100)
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--min-events", type=int, default=2,
                        help="Min matched events per window (V1 only)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    windows = load_and_associate(args.data_root, min_events=args.min_events)

    if args.variant == "V1":
        records = build_v1_records(windows)
    else:
        records = build_v2_records(windows)

    rng.shuffle(records)
    n_val = min(args.total_val, len(records) // 5)
    val_records = records[:n_val]
    train_records = records[n_val:]

    write_jsonl(train_records, os.path.join(args.output_dir, "train.jsonl"))
    write_jsonl(val_records, os.path.join(args.output_dir, "val.jsonl"))

    print(f"\n=== {args.variant} Output ===")
    print(f"  Dir: {args.output_dir}")
    print(f"  Train: {len(train_records)}")
    print(f"  Val: {len(val_records)}")

    if train_records:
        ex = train_records[0]
        print(f"\n=== Sample ===")
        print(f"  video: {ex['videos'][0]}")
        print(f"  problem_type: {ex['problem_type']}")
        print(f"  answer (first 300): {ex['answer'][:300]}")


if __name__ == "__main__":
    main()
