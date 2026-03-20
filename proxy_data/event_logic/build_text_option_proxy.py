#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 YouCookII 原始标注构造 Event Logic proxy 训练数据（add / replace / sort）。

所有任务输出均为 EasyR1 JSONL 格式，可直接用于 GRPO 训练。
Prompt 从 prompts.py 统一维护，内置 CoT (<think>/<answer>) 指令。

数据生成流程:
    build_text_option_proxy.py → proxy_train_text_options.jsonl
                                  (add + replace + sort，已内置 CoT)
    filter_bad_videos.py       → proxy_train_text_options_clean.jsonl
                                  (用 decord 过滤不可读 / 帧数不足的视频)

示例:
    python proxy_data/event_logic/build_text_option_proxy.py \\
        -a proxy_data/youcookii_annotations_trainval.json \\
        -o proxy_data/event_logic/data/proxy_train_text_options.jsonl \\
        --event-clips-root /m2v_intern/xuboshen/zgw/data/youcook2_event_clips \\
        --add-per-video 1 \\
        --replace-per-video 1 \\
        --sort-per-video 1 \\
        --seed 42 \\
        --shuffle
"""

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict

from prompts import get_add_prompt, get_replace_prompt, get_sort_prompt


# ─────────────────────────────────────────────────────────────────────────────
# Constants / helpers
# ─────────────────────────────────────────────────────────────────────────────

_LETTERS = [chr(ord("A") + i) for i in range(26)]
_EVENT_FILE_RE = re.compile(r"_event\d+_(\d+)_(\d+)\.mp4$")


def _option_labels(n: int) -> list[str]:
    return _LETTERS[:n]


def _normalize_segment(seg):
    """segment [start, end] → (int_start, int_end), or None if invalid."""
    if not isinstance(seg, (list, tuple)) or len(seg) != 2:
        return None
    try:
        s = int(round(float(seg[0])))
        e = int(round(float(seg[1])))
    except (TypeError, ValueError):
        return None
    return (s, e) if s < e else None


def _build_event_path(root: str, subset: str, recipe_type: str, video_id: str,
                      event_id: int, start: int, end: int) -> str:
    fname = f"{video_id}_event{event_id:02d}_{start}_{end}.mp4"
    return os.path.join(root, subset, recipe_type, video_id, fname)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_videos(anno_path: str, event_clips_root: str, min_events: int = 4) -> tuple:
    """
    Parse youcookii_annotations_trainval.json into a list of video dicts.

    Returns:
        (videos, sentence_pool_by_recipe)
        - videos: list of dicts with keys: video_id, subset, recipe_type, events
        - sentence_pool_by_recipe: {recipe_type: [(video_id, event_id, sentence), ...]}
    """
    with open(anno_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    db = raw.get("database", {})
    videos = []
    sentence_pool_by_recipe = defaultdict(list)

    for video_id, item in db.items():
        subset = item.get("subset", "")
        recipe_type = str(item.get("recipe_type", ""))
        ann = item.get("annotations", [])
        if not subset or not recipe_type or not isinstance(ann, list):
            continue

        events = []
        for ev in ann:
            seg = _normalize_segment(ev.get("segment"))
            if seg is None:
                continue
            sentence = (ev.get("sentence") or "").strip()
            if not sentence:
                continue
            event_id = int(ev.get("id", len(events)))
            s, e = seg
            path = _build_event_path(event_clips_root, subset, recipe_type, video_id, event_id, s, e)
            events.append({"id": event_id, "start": s, "end": e, "sentence": sentence, "path": path})
            sentence_pool_by_recipe[recipe_type].append((video_id, event_id, sentence))

        events.sort(key=lambda x: x["id"])
        if len(events) < min_events:
            continue

        videos.append({
            "video_id": video_id,
            "subset": subset,
            "recipe_type": recipe_type,
            "events": events,
        })

    return videos, sentence_pool_by_recipe


def rebuild_sentence_pool(videos: list) -> dict:
    """Rebuild sentence_pool_by_recipe from a filtered video list."""
    pool = defaultdict(list)
    for v in videos:
        for ev in v["events"]:
            pool[v["recipe_type"]].append((v["video_id"], ev["id"], ev["sentence"]))
    return pool


# ─────────────────────────────────────────────────────────────────────────────
# Negative sampling
# ─────────────────────────────────────────────────────────────────────────────

def _sample_negatives(sentence_pool_by_recipe: dict, anchor_recipe_type: str,
                      gt_sentence: str, same_video_other: list[str], k: int = 3):
    """
    Sample k negative text descriptions mixing two difficulty levels:
      1. Cross-recipe (1–2 items): easy distractors from unrelated dishes.
      2. Same-video other events (remaining): harder distractors from the same cooking sequence.

    Returns a list of k negatives, or None if there aren't enough candidates.
    """
    n_cross = random.randint(1, min(2, k))
    n_same = k - n_cross
    used = {gt_sentence}
    result = []

    # --- cross-recipe ---
    cross_cands = sorted({
        sent
        for rt, pool in sentence_pool_by_recipe.items()
        if rt != anchor_recipe_type
        for _, _, sent in pool
        if sent not in used
    })
    if len(cross_cands) < n_cross:
        return None
    sampled_cross = random.sample(cross_cands, n_cross)
    result.extend(sampled_cross)
    used.update(sampled_cross)

    # --- same-video other events ---
    same_vid_cands = list(dict.fromkeys(s for s in same_video_other if s not in used))
    if len(same_vid_cands) < n_same:
        return None
    result.extend(random.sample(same_vid_cands, n_same))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Sample builders
# ─────────────────────────────────────────────────────────────────────────────

def build_add_sample(v: dict, sentence_pool_by_recipe: dict,
                     min_ctx: int = 2, max_ctx: int = 4) -> dict | None:
    """
    Build one 'add' sample: predict the next step from N context clips.

    Context: [event_start … event_start+ctx_len-1]
    Target:  event_start+ctx_len  (shown only as text option)
    """
    events = v["events"]
    if len(events) < min_ctx + 1:
        return None

    ctx_len = random.randint(min_ctx, min(max_ctx, len(events) - 1))
    start = random.randint(0, len(events) - (ctx_len + 1))
    ctx_events = events[start:start + ctx_len]
    gt_event = events[start + ctx_len]

    used_ids = {ev["id"] for ev in ctx_events} | {gt_event["id"]}
    same_video_other = [ev["sentence"] for ev in events if ev["id"] not in used_ids]

    negs = _sample_negatives(sentence_pool_by_recipe, v["recipe_type"],
                              gt_event["sentence"], same_video_other, k=3)
    if negs is None:
        return None

    options = negs + [gt_event["sentence"]]
    random.shuffle(options)
    labels = _option_labels(len(options))
    answer = labels[options.index(gt_event["sentence"])]

    prompt = get_add_prompt(len(ctx_events), options)
    return {
        "messages": [{"role": "user", "content": prompt}],
        "prompt": prompt,
        "answer": answer,
        "videos": [ev["path"] for ev in ctx_events],
        "data_type": "video",
        "problem_type": "add",
        "metadata": {
            "video_id": v["video_id"],
            "recipe_type": v["recipe_type"],
            "context_start_event": ctx_events[0]["id"],
            "context_end_event": ctx_events[-1]["id"],
            "target_event": gt_event["id"],
            "target_sentence": gt_event["sentence"],
            "option_type": "text",
        },
    }


def build_replace_sample(v: dict, sentence_pool_by_recipe: dict,
                         seq_len: int = 5) -> dict | None:
    """
    Build one 'replace' sample: fill in the missing middle step.

    Sequence length: seq_len (including the missing slot).
    Missing position: random interior slot (pos 1 … seq_len-2, zero-indexed).
    """
    events = v["events"]
    if len(events) < seq_len or seq_len < 3:
        return None

    start = random.randint(0, len(events) - seq_len)
    seq_events = events[start:start + seq_len]
    missing_pos = random.randint(1, seq_len - 2)
    gt_event = seq_events[missing_pos]

    seq_ids = {ev["id"] for ev in seq_events}
    same_video_other = [ev["sentence"] for ev in events if ev["id"] not in seq_ids]

    negs = _sample_negatives(sentence_pool_by_recipe, v["recipe_type"],
                              gt_event["sentence"], same_video_other, k=3)
    if negs is None:
        return None

    options = negs + [gt_event["sentence"]]
    random.shuffle(options)
    labels = _option_labels(len(options))
    answer = labels[options.index(gt_event["sentence"])]

    context_events = [ev for i, ev in enumerate(seq_events) if i != missing_pos]
    prompt = get_replace_prompt(seq_len, missing_pos, options)
    return {
        "messages": [{"role": "user", "content": prompt}],
        "prompt": prompt,
        "answer": answer,
        "videos": [ev["path"] for ev in context_events],
        "data_type": "video",
        "problem_type": "replace",
        "metadata": {
            "video_id": v["video_id"],
            "recipe_type": v["recipe_type"],
            "context_start_event": seq_events[0]["id"],
            "context_end_event": seq_events[-1]["id"],
            "missing_pos": missing_pos,
            "target_event": gt_event["id"],
            "target_sentence": gt_event["sentence"],
            "option_type": "text",
        },
    }


def build_sort_sample(v: dict, seq_len: int = 5) -> dict | None:
    """
    Build one 'sort' sample: reorder N shuffled clips into chronological order.

    Ground-truth answer encoding:
        answer[i] = 1-based clip number occupying position i in the correct sequence.
    Example: 3 clips shuffled as [ev2, ev0, ev1] → answer = "312"
        (position 0 = Clip3, position 1 = Clip1, position 2 = Clip2)

    Refusal condition: shuffled order must differ from original.
    """
    events = v["events"]
    if len(events) < seq_len:
        return None

    start = random.randint(0, len(events) - seq_len)
    original = events[start:start + seq_len]

    # Resample until the shuffle is not a no-op
    shuffled_indices = list(range(seq_len))
    for _ in range(10):
        random.shuffle(shuffled_indices)
        if shuffled_indices != list(range(seq_len)):
            break
    else:
        return None  # Could not get a non-trivial shuffle (very unlikely)

    # shuffled_indices[i] = which original event appears as Clip (i+1)
    # Compute inverse permutation: position j in original order → which clip number
    inverse = [0] * seq_len
    for clip_idx, orig_idx in enumerate(shuffled_indices):
        inverse[orig_idx] = clip_idx + 1
    answer = "".join(str(x) for x in inverse)

    shuffled_events = [original[i] for i in shuffled_indices]
    prompt = get_sort_prompt(seq_len)
    return {
        "messages": [{"role": "user", "content": prompt}],
        "prompt": prompt,
        "answer": answer,
        "videos": [ev["path"] for ev in shuffled_events],
        "data_type": "video",
        "problem_type": "sort",
        "metadata": {
            "video_id": v["video_id"],
            "recipe_type": v["recipe_type"],
            "context_start_event": original[0]["id"],
            "context_end_event": original[-1]["id"],
            "shuffled_indices": shuffled_indices,
            "clip_sentences": [ev["sentence"] for ev in shuffled_events],
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="构造 Event Logic proxy 训练数据（add / replace / sort）"
    )
    parser.add_argument("--annotations", "-a", required=True,
                        help="youcookii_annotations_trainval.json 路径")
    parser.add_argument("--output", "-o", required=True,
                        help="输出 JSONL 文件")
    parser.add_argument("--event-clips-root", required=True,
                        help="事件切片根目录，例如 /.../youcook2_event_clips")

    # Per-video generation counts
    parser.add_argument("--add-per-video", type=int, default=1,
                        help="每个视频生成 add 样本数（默认 1）")
    parser.add_argument("--replace-per-video", type=int, default=1,
                        help="每个视频生成 replace 样本数（默认 1）")
    parser.add_argument("--sort-per-video", type=int, default=1,
                        help="每个视频生成 sort 样本数（默认 1）")

    # Task hyper-parameters
    parser.add_argument("--min-events", type=int, default=4,
                        help="一个视频最少需要的事件数（默认 4）")
    parser.add_argument("--min-context", type=int, default=2,
                        help="add 任务最小上下文 clip 数（默认 2）")
    parser.add_argument("--max-context", type=int, default=4,
                        help="add 任务最大上下文 clip 数（默认 4）")
    parser.add_argument("--replace-seq-len", type=int, default=5,
                        help="replace 任务序列总步数，含缺失位（默认 5，输入视频数=4）")
    parser.add_argument("--sort-seq-len", type=int, default=5,
                        help="sort 任务 clip 数（默认 5）")

    # Output control
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子（默认 42）")
    parser.add_argument("--shuffle", action="store_true",
                        help="写出前打乱所有样本")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="最多写出样本数（用于快速测试）")

    args = parser.parse_args()
    random.seed(args.seed)

    if args.replace_seq_len < 3:
        raise ValueError("--replace-seq-len 必须 >= 3")

    # Load annotations
    print(f"📂 加载标注: {args.annotations}")
    videos, sentence_pool = load_videos(
        anno_path=args.annotations,
        event_clips_root=args.event_clips_root,
        min_events=args.min_events,
    )
    if not videos:
        raise RuntimeError("没有可用视频，请检查标注文件或 --min-events 参数")
    print(f"   可用视频数: {len(videos)}")

    # Build samples
    samples = []
    stats = Counter()

    for v in videos:
        for _ in range(args.add_per_video):
            s = build_add_sample(v, sentence_pool, args.min_context, args.max_context)
            if s is not None:
                samples.append(s)
                stats["add"] += 1

        for _ in range(args.replace_per_video):
            s = build_replace_sample(v, sentence_pool, args.replace_seq_len)
            if s is not None:
                samples.append(s)
                stats["replace"] += 1

        for _ in range(args.sort_per_video):
            s = build_sort_sample(v, args.sort_seq_len)
            if s is not None:
                samples.append(s)
                stats["sort"] += 1

    if args.shuffle:
        random.shuffle(samples)

    if args.max_samples and args.max_samples > 0:
        samples = samples[:args.max_samples]

    # Write output
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\n✅ 写出完成: {len(samples)} 条样本 → {args.output}")
    print("📊 统计:")
    for task in ["add", "replace", "sort"]:
        print(f"  {task}: {stats[task]}")


if __name__ == "__main__":
    main()
