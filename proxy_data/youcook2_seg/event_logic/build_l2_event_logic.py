#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_l2_event_logic.py — 从 L2 标注 JSON 构造 Event Logic proxy 训练数据。

数据来源:
    proxy_data/hier_seg_annotation/annotations/{clip_key}.json
    其中每个文件的 level2.events 包含:
        event_id, start_time, end_time, instruction, visual_keywords

与旧版 build_text_option_proxy.py 的区别:
    - 数据来自新人工标注的 L2 层，而非原始 youcookii_annotations_trainval.json
    - 事件切片路径由 L2 clips 目录 + clip_key + 时间范围推导，无需 event_clips 目录
    - 增加 AI 因果有效性过滤 (--filter):
        对每条 add/replace 样本，用 VLM 观察帧判断:
          1. 上下文是否充分
          2. 答案是否唯一 (避免多解)
        仅保留 causal_valid=true 且 confidence >= threshold 的样本
    - sort 任务不需要 AI 过滤（纯物理顺序，无因果歧义）

输出格式:
    EasyR1 JSONL，与现有 proxy 数据保持兼容。

用法:
    python proxy_data/event_logic/build_l2_event_logic.py \\
        --annotation-dir   proxy_data/hier_seg_annotation/annotations \\
        --clips-dir        /m2v_intern/xuboshen/zgw/data/hier_seg_annotation/clips/L2 \\
        --frames-dir       /m2v_intern/xuboshen/zgw/data/hier_seg_annotation/frames \\
        --output           proxy_data/event_logic/data/l2_event_logic.jsonl \\
        --filter \\
        --api-base         https://api.novita.ai/v3/openai \\
        --model            qwen/qwen2.5-vl-72b-instruct \\
        --confidence-threshold 0.75 \\
        --workers          8 \\
        --seed             42
"""

import argparse
import base64
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

# 添加 proxy_data 父目录到 sys.path 以便 import shared
# 脚本位于 proxy_data/youcook2_seg/event_logic/，需上溯三级到 proxy_data/
_PROXY_DATA_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROXY_DATA_DIR not in sys.path:
    sys.path.insert(0, _PROXY_DATA_DIR)

from shared.seg_source import load_annotations as _load_ann_raw, get_l2_clip_path

from prompts import (
    get_add_prompt,
    get_replace_prompt,
    get_sort_prompt,
    CAUSALITY_SYSTEM_PROMPT,
    CAUSALITY_USER_PROMPT,
)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_LETTERS = [chr(ord("A") + i) for i in range(26)]


def _option_labels(n: int) -> list[str]:
    return _LETTERS[:n]


# ─────────────────────────────────────────────────────────────────────────────
# Annotation loading
# ─────────────────────────────────────────────────────────────────────────────

def load_annotations(annotation_dir: Path, min_events: int = 4) -> tuple[list[dict], list[str]]:
    """Load all L2 annotation JSON files and convert to event-logic format.

    Uses ``shared.seg_source.load_annotations`` for raw JSON loading, then
    converts each annotation into the per-video event list needed by this
    pipeline.

    Returns:
        (videos, all_sentences)
        videos: list of dicts with keys: clip_key, video_path, events
            where each event has: id, start, end, sentence, video_path, clip_key
        all_sentences: flat list of event instructions for cross-clip negative sampling
    """
    raw_annotations = _load_ann_raw(annotation_dir, complete_only=False)

    videos: list[dict] = []
    all_sentences: list[str] = []

    for ann in raw_annotations:
        l2 = ann.get("level2")
        if not isinstance(l2, dict):
            continue

        events_raw = l2.get("events")
        if not isinstance(events_raw, list):
            continue

        clip_key = ann.get("clip_key", "")
        video_path = ann.get("video_path") or ann.get("source_video_path") or ""

        events: list[dict] = []
        for ev in events_raw:
            s = ev.get("start_time")
            e = ev.get("end_time")
            instruction = (ev.get("instruction") or "").strip()
            if s is None or e is None or not instruction:
                continue
            try:
                s, e = int(round(float(s))), int(round(float(e)))
            except (TypeError, ValueError):
                continue
            if s >= e:
                continue
            events.append({
                "id": ev.get("event_id", len(events)),
                "start": s,
                "end": e,
                "sentence": instruction,
                "video_path": video_path,
                "clip_key": clip_key,
            })

        events.sort(key=lambda x: x["start"])
        if len(events) < min_events:
            continue

        for ev in events:
            all_sentences.append(ev["sentence"])

        videos.append({
            "clip_key": clip_key,
            "video_path": video_path,
            "events": events,
        })

    return videos, all_sentences


# ─────────────────────────────────────────────────────────────────────────────
# Clip path resolution
# ─────────────────────────────────────────────────────────────────────────────

def find_event_clip(clips_dir: Path, clip_key: str, start: int, end: int) -> str | None:
    """
    Find the L2 window clip that fully contains [start, end].

    L2 clips are named like: {clip_key}_L2_w{ws}_{we}.mp4
    We look for the window where ws <= start and we >= end.
    Returns path string or None if not found.
    """
    pattern = f"{clip_key}_L2_w*.mp4"
    best = None
    for p in clips_dir.glob(pattern):
        stem = p.stem  # e.g. --bv0V6ZjWI_L2_w0_128
        parts = stem.rsplit("_w", 1)
        if len(parts) != 2:
            continue
        try:
            ws_str, we_str = parts[1].split("_")
            ws, we = int(ws_str), int(we_str)
        except ValueError:
            continue
        if ws <= start and we >= end:
            # prefer the tightest window
            if best is None or (we - ws) < (best[1] - best[0]):
                best = (ws, we, str(p))
    return best[2] if best is not None else None


def resolve_event_paths(videos: list[dict], clips_dir: Path) -> list[dict]:
    """
    Attach clip file paths to each event. Events without a resolvable clip are removed.
    Videos that end up with fewer than min_events events are dropped.
    """
    result = []
    for v in videos:
        resolved = []
        for ev in v["events"]:
            path = find_event_clip(clips_dir, v["clip_key"], ev["start"], ev["end"])
            if path is None:
                continue
            ev = dict(ev)
            ev["path"] = path
            resolved.append(ev)
        if len(resolved) >= 4:
            result.append({**v, "events": resolved})
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Negative sampling (text-only, same logic as build_text_option_proxy.py)
# ─────────────────────────────────────────────────────────────────────────────

def _sample_negatives(
    all_sentences: list[str],
    gt_sentence: str,
    same_clip_other: list[str],
    k: int = 3,
    rng: random.Random = random,
) -> list[str] | None:
    """
    Sample k negative text descriptions:
      ~1/3 cross-clip (random from all_sentences minus same clip)
      ~2/3 same-clip other events (harder distractors)
    """
    n_cross = rng.randint(1, min(2, k))
    n_same = k - n_cross
    used = {gt_sentence}
    result = []

    cross_cands = sorted({s for s in all_sentences if s not in used and s not in same_clip_other})
    if len(cross_cands) < n_cross:
        return None
    sampled_cross = rng.sample(cross_cands, n_cross)
    result.extend(sampled_cross)
    used.update(sampled_cross)

    same_cands = list(dict.fromkeys(s for s in same_clip_other if s not in used))
    if len(same_cands) < n_same:
        return None
    result.extend(rng.sample(same_cands, n_same))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Sample builders
# ─────────────────────────────────────────────────────────────────────────────

def build_add_sample(
    v: dict,
    all_sentences: list[str],
    min_ctx: int = 2,
    max_ctx: int = 4,
    rng: random.Random = random,
) -> dict | None:
    events = v["events"]
    if len(events) < min_ctx + 1:
        return None

    ctx_len = rng.randint(min_ctx, min(max_ctx, len(events) - 1))
    start = rng.randint(0, len(events) - (ctx_len + 1))
    ctx_events = events[start : start + ctx_len]
    gt_event = events[start + ctx_len]

    used_ids = {ev["id"] for ev in ctx_events} | {gt_event["id"]}
    same_clip_other = [ev["sentence"] for ev in events if ev["id"] not in used_ids]

    negs = _sample_negatives(all_sentences, gt_event["sentence"], same_clip_other, k=3, rng=rng)
    if negs is None:
        return None

    options = negs + [gt_event["sentence"]]
    rng.shuffle(options)
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
            "clip_key": v["clip_key"],
            "context_event_ids": [ev["id"] for ev in ctx_events],
            "target_event_id": gt_event["id"],
            "target_sentence": gt_event["sentence"],
            "options": options,
            # --- for AI filter ---
            "ctx_event_paths": [ev["path"] for ev in ctx_events],
            "ctx_event_timestamps": [(ev["start"], ev["end"]) for ev in ctx_events],
        },
    }


def build_replace_sample(
    v: dict,
    all_sentences: list[str],
    seq_len: int = 5,
    rng: random.Random = random,
) -> dict | None:
    events = v["events"]
    if len(events) < seq_len or seq_len < 3:
        return None

    start = rng.randint(0, len(events) - seq_len)
    seq_events = events[start : start + seq_len]
    missing_pos = rng.randint(1, seq_len - 2)
    gt_event = seq_events[missing_pos]

    seq_ids = {ev["id"] for ev in seq_events}
    same_clip_other = [ev["sentence"] for ev in events if ev["id"] not in seq_ids]

    negs = _sample_negatives(all_sentences, gt_event["sentence"], same_clip_other, k=3, rng=rng)
    if negs is None:
        return None

    options = negs + [gt_event["sentence"]]
    rng.shuffle(options)
    labels = _option_labels(len(options))
    answer = labels[options.index(gt_event["sentence"])]

    ctx_events = [ev for i, ev in enumerate(seq_events) if i != missing_pos]
    prompt = get_replace_prompt(seq_len, missing_pos, options)
    return {
        "messages": [{"role": "user", "content": prompt}],
        "prompt": prompt,
        "answer": answer,
        "videos": [ev["path"] for ev in ctx_events],
        "data_type": "video",
        "problem_type": "replace",
        "metadata": {
            "clip_key": v["clip_key"],
            "sequence_event_ids": [ev["id"] for ev in seq_events],
            "missing_pos": missing_pos,
            "target_event_id": gt_event["id"],
            "target_sentence": gt_event["sentence"],
            "options": options,
            # --- for AI filter ---
            "ctx_event_paths": [ev["path"] for ev in ctx_events],
            "ctx_event_timestamps": [(ev["start"], ev["end"]) for ev in ctx_events],
        },
    }


def build_sort_sample(
    v: dict,
    seq_len: int = 5,
    rng: random.Random = random,
) -> dict | None:
    events = v["events"]
    if len(events) < seq_len:
        return None

    start = rng.randint(0, len(events) - seq_len)
    original = events[start : start + seq_len]

    shuffled_indices = list(range(seq_len))
    for _ in range(10):
        rng.shuffle(shuffled_indices)
        if shuffled_indices != list(range(seq_len)):
            break
    else:
        return None

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
            "clip_key": v["clip_key"],
            "original_event_ids": [ev["id"] for ev in original],
            "shuffled_indices": shuffled_indices,
            "clip_sentences": [ev["sentence"] for ev in shuffled_events],
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# AI Causality Filter
# ─────────────────────────────────────────────────────────────────────────────

def _encode_frame(frame_path: Path, resize_max_width: int = 480, jpeg_quality: int = 70) -> str:
    with Image.open(frame_path) as img:
        img = img.convert("RGB")
        if resize_max_width > 0 and img.width > resize_max_width:
            new_h = max(1, round(img.height * resize_max_width / img.width))
            img = img.resize((resize_max_width, new_h), Image.Resampling.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        return base64.b64encode(buf.getvalue()).decode()


def sample_frames_from_dir(
    frames_dir: Path,
    clip_key: str,
    start_sec: int,
    end_sec: int,
    n_frames: int = 4,
) -> list[Path]:
    """
    Sample up to n_frames JPEG frames from the pre-extracted frame directory
    that fall within [start_sec, end_sec].
    """
    frame_dir = frames_dir / clip_key
    if not frame_dir.exists():
        return []
    candidates = sorted(
        p for p in frame_dir.glob("*.jpg")
        if start_sec <= int(p.stem) <= end_sec
    )
    if not candidates:
        return []
    if len(candidates) <= n_frames:
        return candidates
    stride = (len(candidates) - 1) / (n_frames - 1)
    return [candidates[round(i * stride)] for i in range(n_frames)]


def call_causality_filter(
    api_base: str,
    api_key: str,
    model: str,
    sample: dict,
    frames_dir: Path,
    n_frames_per_event: int = 3,
    retries: int = 3,
) -> dict | None:
    """
    Call VLM to judge whether the sample has a unique, contextually-supported answer.

    Returns parsed JSON dict or None on failure.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai is required: pip install openai")

    key = api_key or os.environ.get("NOVITA_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    client = OpenAI(api_key=key, base_url=api_base)

    meta = sample["metadata"]
    problem_type = sample["problem_type"]
    clip_key = meta["clip_key"]
    ctx_paths = meta["ctx_event_paths"]
    ctx_timestamps = meta["ctx_event_timestamps"]
    options = meta["options"]

    # Build task description
    if problem_type == "add":
        task_type = "predict the correct NEXT cooking step after the context events"
    else:
        task_type = "identify the correct MISSING step that fills the gap in the sequence"

    options_str = "\n".join(
        f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(options)
    )
    user_text = CAUSALITY_USER_PROMPT.format(task_type=task_type, options_str=options_str)

    # Collect frames from all context events
    content: list[dict] = [{"type": "text", "text": user_text}]
    for i, (path, (s, e)) in enumerate(zip(ctx_paths, ctx_timestamps)):
        frames = sample_frames_from_dir(frames_dir, clip_key, s, e, n_frames=n_frames_per_event)
        if not frames:
            return None  # can't filter without frames
        content.append({"type": "text", "text": f"[Context Event {i + 1}]"})
        for fp in frames:
            try:
                b64 = _encode_frame(fp)
            except Exception:
                continue
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
            })

    messages = [
        {"role": "system", "content": CAUSALITY_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]

    last_err = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=512,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content
            return json.loads(raw)
        except Exception as exc:
            last_err = exc
            if attempt < retries - 1:
                time.sleep(2 ** attempt)

    print(f"  [filter] API error for {clip_key}: {last_err}", file=sys.stderr)
    return None


def filter_samples_with_ai(
    samples: list[dict],
    api_base: str,
    api_key: str,
    model: str,
    frames_dir: Path,
    confidence_threshold: float = 0.75,
    workers: int = 4,
    n_frames_per_event: int = 3,
) -> list[dict]:
    """
    Filter add/replace samples via AI causality check.
    Sort samples pass through unchanged.
    """
    to_filter = [(i, s) for i, s in enumerate(samples) if s["problem_type"] in ("add", "replace")]
    passthrough = [(i, s) for i, s in enumerate(samples) if s["problem_type"] == "sort"]

    kept_flags: dict[int, bool] = {i: True for i, _ in passthrough}
    stats = Counter({"sort_pass": len(passthrough)})

    def check_one(idx_sample):
        idx, s = idx_sample
        result = call_causality_filter(
            api_base, api_key, model, s, frames_dir,
            n_frames_per_event=n_frames_per_event,
        )
        if result is None:
            return idx, False, "api_failed"
        valid = result.get("causal_valid", False)
        conf = float(result.get("confidence", 0.0))
        if valid and conf >= confidence_threshold:
            return idx, True, "kept"
        return idx, False, f"filtered(valid={valid},conf={conf:.2f})"

    print(f"\n[AI Filter] Checking {len(to_filter)} add/replace samples "
          f"with {workers} workers ...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(check_one, item): item for item in to_filter}
        for i, fut in enumerate(as_completed(futures), 1):
            idx, keep, reason = fut.result()
            kept_flags[idx] = keep
            stats[reason] += 1
            if i % 100 == 0 or i == len(to_filter):
                print(f"  [{i}/{len(to_filter)}] {dict(stats)}")

    filtered = [s for i, s in enumerate(samples) if kept_flags.get(i, False)]
    print(f"\n[AI Filter] {len(filtered)}/{len(samples)} samples kept. Stats: {dict(stats)}")
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="从 L2 标注构造 Event Logic 训练数据 (add / replace / sort) + AI 因果过滤"
    )
    parser.add_argument("--annotation-dir", "-a", required=True,
                        help="L2 标注 JSON 目录 (hier_seg_annotation/annotations)")
    parser.add_argument("--clips-dir", required=True,
                        help="L2 clips 目录 (hier_seg_annotation/clips/L2)")
    parser.add_argument("--frames-dir", required=True,
                        help="帧目录 (hier_seg_annotation/frames)，用于 AI 过滤")
    parser.add_argument("--output", "-o", required=True,
                        help="输出 JSONL 文件路径")

    # Per-video counts
    parser.add_argument("--add-per-video", type=int, default=1)
    parser.add_argument("--replace-per-video", type=int, default=1)
    parser.add_argument("--sort-per-video", type=int, default=1)

    # Task hyper-params
    parser.add_argument("--min-events", type=int, default=4,
                        help="最少事件数的视频才参与采样")
    parser.add_argument("--min-context", type=int, default=2)
    parser.add_argument("--max-context", type=int, default=4)
    parser.add_argument("--replace-seq-len", type=int, default=5)
    parser.add_argument("--sort-seq-len", type=int, default=5)

    # AI filter
    parser.add_argument("--filter", action="store_true",
                        help="启用 AI 因果有效性过滤 (需要 --api-base 和 --model)")
    parser.add_argument("--api-base", default="https://api.novita.ai/v3/openai")
    parser.add_argument("--api-key", default="",
                        help="API key（默认读取 NOVITA_API_KEY 或 OPENAI_API_KEY 环境变量）")
    parser.add_argument("--model", default="qwen/qwen2.5-vl-72b-instruct")
    parser.add_argument("--confidence-threshold", type=float, default=0.75,
                        help="保留样本的最低 confidence 分（默认 0.75）")
    parser.add_argument("--filter-workers", type=int, default=4,
                        help="AI 过滤并发线程数（默认 4）")
    parser.add_argument("--frames-per-event", type=int, default=3,
                        help="每个上下文事件采样的帧数（默认 3）")

    # Output control
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)

    args = parser.parse_args()

    rng = random.Random(args.seed)

    # ── Load annotations ──────────────────────────────────────────────────────
    print(f"📂 加载 L2 标注: {args.annotation_dir}")
    annotation_dir = Path(args.annotation_dir)
    clips_dir = Path(args.clips_dir)
    frames_dir = Path(args.frames_dir)

    videos, all_sentences = load_annotations(annotation_dir, min_events=args.min_events)
    print(f"   原始可用视频数: {len(videos)}")

    # ── Resolve clip paths ────────────────────────────────────────────────────
    print(f"🔍 解析事件切片路径: {clips_dir}")
    videos = resolve_event_paths(videos, clips_dir)
    print(f"   路径解析后可用视频数: {len(videos)}")

    if not videos:
        print("❌ 没有可用视频，请检查 --annotation-dir 和 --clips-dir", file=sys.stderr)
        sys.exit(1)

    # ── Build samples ─────────────────────────────────────────────────────────
    samples = []
    stats = Counter()

    for v in videos:
        for _ in range(args.add_per_video):
            s = build_add_sample(v, all_sentences, args.min_context, args.max_context, rng)
            if s is not None:
                samples.append(s)
                stats["add"] += 1

        for _ in range(args.replace_per_video):
            s = build_replace_sample(v, all_sentences, args.replace_seq_len, rng)
            if s is not None:
                samples.append(s)
                stats["replace"] += 1

        for _ in range(args.sort_per_video):
            s = build_sort_sample(v, args.sort_seq_len, rng)
            if s is not None:
                samples.append(s)
                stats["sort"] += 1

    print(f"\n📊 初始样本数: {len(samples)} {dict(stats)}")

    # ── AI causality filter ───────────────────────────────────────────────────
    if args.filter:
        samples = filter_samples_with_ai(
            samples,
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.model,
            frames_dir=frames_dir,
            confidence_threshold=args.confidence_threshold,
            workers=args.filter_workers,
            n_frames_per_event=args.frames_per_event,
        )

    if args.shuffle:
        rng.shuffle(samples)

    if args.max_samples and args.max_samples > 0:
        samples = samples[: args.max_samples]

    # ── Write output ──────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Strip internal AI-filter fields from final output
    def _clean(s: dict) -> dict:
        meta = dict(s.get("metadata", {}))
        meta.pop("ctx_event_paths", None)
        meta.pop("ctx_event_timestamps", None)
        return {**s, "metadata": meta}

    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(_clean(s), ensure_ascii=False) + "\n")

    print(f"\n✅ 写出完成: {len(samples)} 条样本 → {out_path}")
    final_stats = Counter(s["problem_type"] for s in samples)
    print("📊 任务分布:", dict(final_stats))


if __name__ == "__main__":
    main()
