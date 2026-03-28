#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_l2_event_logic_t2v.py — 构造 T→V Event Logic 训练数据 (add_t2v / replace_t2v)。

与 build_l2_event_logic.py (V→T) 的核心区别:
  V→T:  上下文 = N 个视频 clip  |  选项 = 4 条文字描述  →  选正确文字
  T→V:  上下文 = N 条文字描述  |  选项 = 4 个视频 clip  →  选正确视频

数据来源:
  - AoT event manifest (aot_event_manifest.jsonl):
      每条记录包含一个事件的 clip_key / source_video_id / sequence_index /
      forward_video_path / sentence 等字段。通过 source_video_id + sequence_index
      重建各视频的完整事件序列。
  - Step captions JSONL (l2_step_captions.jsonl):
      由 annotate_l2_step_captions.py 生成，以 clip_key 为键存储 recipe-instruction
      风格的文字描述。缺少 caption 的事件将被跳过。

输出格式:
  EasyR1 JSONL，problem_type 分别为 add_t2v / replace_t2v，与现有格式兼容。

用法:
    # Step 1: 先生成 caption（从现有 V→T 数据集提取 manifest）
    python proxy_data/event_logic/annotate_l2_step_captions.py \\
        --from-dataset proxy_data/event_logic/data/proxy_train_text_options.jsonl \\
        --output proxy_data/event_logic/data/l2_step_captions.jsonl \\
        --api-base https://api.novita.ai/v3/openai \\
        --model qwen/qwen2.5-vl-72b-instruct \\
        --workers 8

    # Step 2: 构建 T→V 数据集（可选 AI 因果过滤）
    python proxy_data/event_logic/build_l2_event_logic_t2v.py \\
        --manifest-jsonl proxy_data/temporal_aot/data/aot_event_manifest.jsonl \\
        --captions-jsonl proxy_data/event_logic/data/l2_step_captions.jsonl \\
        --output proxy_data/event_logic/data/l2_event_logic_t2v.jsonl \\
        --filter \\
        --api-base https://api.novita.ai/v3/openai \\
        --model qwen/qwen2.5-vl-72b-instruct \\
        --workers 8 \\
        --seed 42
"""

from __future__ import annotations

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

from prompts import (
    get_add_t2v_prompt,
    get_replace_t2v_prompt,
    CAUSALITY_T2V_SYSTEM_PROMPT,
    CAUSALITY_T2V_USER_PROMPT,
)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_LETTERS = [chr(ord("A") + i) for i in range(26)]


def _option_labels(n: int) -> list[str]:
    return _LETTERS[:n]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_manifest(path: str) -> list[dict]:
    """Load AoT event manifest JSONL. Each line = one event."""
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_captions(path: str) -> dict[str, str]:
    """Load step captions JSONL. Returns {clip_key: caption}."""
    index: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            cap = (item.get("caption") or "").strip()
            if cap:
                index[item["clip_key"]] = cap
    return index


def build_video_sequences(
    records: list[dict],
    captions: dict[str, str],
    min_events: int = 4,
) -> tuple[list[list[dict]], list[dict]]:
    """
    Group manifest records into per-video event sequences.

    Returns:
        (sequences, flat_event_pool)
        sequences: list of video event lists (each sorted by sequence_index)
        flat_event_pool: all individual events with caption, for negative sampling
    """
    by_video: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        vid = r.get("source_video_id") or r.get("video_id") or ""
        if not vid:
            continue
        by_video[vid].append(r)

    sequences: list[list[dict]] = []
    flat_pool: list[dict] = []

    missing_caption = 0

    for vid, events in by_video.items():
        # Sort by sequence position
        events = sorted(events, key=lambda e: (e.get("sequence_index") or e.get("event_id") or 0))

        # Attach caption to each event; skip events without caption
        enriched = []
        for ev in events:
            clip_key = ev.get("clip_key") or Path(ev.get("forward_video_path", "")).stem
            cap = captions.get(clip_key, "")
            if not cap:
                missing_caption += 1
                continue
            enriched.append({
                "clip_key": clip_key,
                "video_path": ev.get("forward_video_path") or ev.get("video_path") or "",
                "sentence": ev.get("sentence", ""),
                "caption": cap,
                "source_video_id": vid,
                "sequence_index": ev.get("sequence_index") or ev.get("event_id") or 0,
            })

        flat_pool.extend(enriched)
        if len(enriched) >= min_events:
            sequences.append(enriched)

    if missing_caption:
        print(f"[load] {missing_caption} events skipped (no caption). "
              f"Run annotate_l2_step_captions.py to generate captions.")

    print(f"[load] {len(sequences)} videos with ≥{min_events} events. "
          f"Total events in pool: {len(flat_pool)}")
    return sequences, flat_pool


# ─────────────────────────────────────────────────────────────────────────────
# Negative video clip sampling
# ─────────────────────────────────────────────────────────────────────────────

def _sample_negative_clips(
    flat_pool: list[dict],
    gt_clip_key: str,
    same_video_others: list[dict],
    k: int = 3,
    rng: random.Random = random,
) -> list[dict] | None:
    """
    Sample k negative event dicts (each with video_path) as distractor video options.
    Prefer ~2/3 same-video (harder) + ~1/3 cross-video (easier) when possible.
    Falls back to all cross-video when same-video pool is too small.
    """
    used_keys = {gt_clip_key}
    result: list[dict] = []
    same_video_keys = {e["clip_key"] for e in same_video_others}

    # Same-video negatives (as many as available, up to k-1)
    same_cands = [e for e in same_video_others if e["clip_key"] not in used_keys]
    n_same = min(len(same_cands), max(0, k - 1))  # leave at least 1 slot for cross-video
    if n_same > 0:
        sampled_same = rng.sample(same_cands, n_same)
        result.extend(sampled_same)
        used_keys.update(e["clip_key"] for e in sampled_same)

    # Cross-video negatives (fill remaining slots)
    n_cross = k - len(result)
    cross_cands = [e for e in flat_pool
                   if e["clip_key"] not in used_keys
                   and e["clip_key"] not in same_video_keys]
    if len(cross_cands) < n_cross:
        return None  # can't fill remaining slots even from cross-video pool
    result.extend(rng.sample(cross_cands, n_cross))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Sample builders
# ─────────────────────────────────────────────────────────────────────────────

def build_add_t2v_sample(
    seq: list[dict],
    flat_pool: list[dict],
    min_ctx: int = 2,
    max_ctx: int = 4,
    rng: random.Random = random,
) -> dict | None:
    """
    T→V Add: context = N text captions (steps 1..N), options = 4 video clips.
    """
    if len(seq) < min_ctx + 1:
        return None

    ctx_len = rng.randint(min_ctx, min(max_ctx, len(seq) - 1))
    start = rng.randint(0, len(seq) - (ctx_len + 1))
    ctx_events = seq[start: start + ctx_len]
    gt_event = seq[start + ctx_len]

    used_ids = {ev["clip_key"] for ev in ctx_events} | {gt_event["clip_key"]}
    same_video_others = [ev for ev in seq if ev["clip_key"] not in used_ids]

    neg_events = _sample_negative_clips(
        flat_pool, gt_event["clip_key"], same_video_others, k=3, rng=rng
    )
    if neg_events is None:
        return None

    # Shuffle options: 1 correct + 3 distractors
    option_events = neg_events + [gt_event]
    rng.shuffle(option_events)
    labels = _option_labels(len(option_events))
    answer = labels[option_events.index(gt_event)]

    context_captions = [ev["caption"] for ev in ctx_events]
    prompt = get_add_t2v_prompt(context_captions, len(option_events))

    return {
        "messages": [{"role": "user", "content": prompt}],
        "prompt": prompt,
        "answer": answer,
        "videos": [ev["video_path"] for ev in option_events],
        "data_type": "video",
        "problem_type": "add_t2v",
        "metadata": {
            "source_video_id": seq[0]["source_video_id"],
            "context_clip_keys": [ev["clip_key"] for ev in ctx_events],
            "context_captions": context_captions,
            "target_clip_key": gt_event["clip_key"],
            "target_caption": gt_event["caption"],
            "option_clip_keys": [ev["clip_key"] for ev in option_events],
            # for AI filter
            "_ctx_captions": context_captions,
            "_correct_option": answer,
            "_option_video_paths": [ev["video_path"] for ev in option_events],
        },
    }


def build_replace_t2v_sample(
    seq: list[dict],
    flat_pool: list[dict],
    seq_len: int = 5,
    rng: random.Random = random,
) -> dict | None:
    """
    T→V Replace: context = N-1 text descriptions + [MISSING], options = 4 video clips.
    """
    if len(seq) < seq_len or seq_len < 3:
        return None

    start = rng.randint(0, len(seq) - seq_len)
    seq_events = seq[start: start + seq_len]
    missing_pos = rng.randint(1, seq_len - 2)  # never first or last
    gt_event = seq_events[missing_pos]

    seq_ids = {ev["clip_key"] for ev in seq_events}
    same_video_others = [ev for ev in seq if ev["clip_key"] not in seq_ids]

    neg_events = _sample_negative_clips(
        flat_pool, gt_event["clip_key"], same_video_others, k=3, rng=rng
    )
    if neg_events is None:
        return None

    option_events = neg_events + [gt_event]
    rng.shuffle(option_events)
    labels = _option_labels(len(option_events))
    answer = labels[option_events.index(gt_event)]

    # Build context: all steps except missing as text; missing_pos as None → [MISSING]
    all_steps: list[str | None] = [
        None if i == missing_pos else ev["caption"]
        for i, ev in enumerate(seq_events)
    ]
    prompt = get_replace_t2v_prompt(all_steps, missing_pos, len(option_events))

    context_captions = [s for s in all_steps if s is not None]  # for metadata/filter

    return {
        "messages": [{"role": "user", "content": prompt}],
        "prompt": prompt,
        "answer": answer,
        "videos": [ev["video_path"] for ev in option_events],
        "data_type": "video",
        "problem_type": "replace_t2v",
        "metadata": {
            "source_video_id": seq[0]["source_video_id"],
            "sequence_clip_keys": [ev["clip_key"] for ev in seq_events],
            "missing_pos": missing_pos,
            "target_clip_key": gt_event["clip_key"],
            "target_caption": gt_event["caption"],
            "option_clip_keys": [ev["clip_key"] for ev in option_events],
            # for AI filter
            "_ctx_captions": context_captions,
            "_all_steps": [s or "[MISSING]" for s in all_steps],
            "_correct_option": answer,
            "_option_video_paths": [ev["video_path"] for ev in option_events],
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# AI Causality Filter (T→V version)
# ─────────────────────────────────────────────────────────────────────────────

def _sample_frames_from_video(
    video_path: str,
    n_frames: int = 3,
    resize_max_width: int = 480,
    jpeg_quality: int = 70,
) -> list[str]:
    """Sample n_frames from a video file; return list of base64 JPEG data URLs."""
    try:
        import decord
    except ImportError as exc:
        raise ImportError("decord is required for AI filter: pip install decord") from exc

    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    total = len(vr)
    if total == 0:
        return []

    if n_frames >= total:
        indices = list(range(total))
    elif n_frames == 1:
        indices = [total // 2]
    else:
        stride = (total - 1) / (n_frames - 1)
        indices = [round(i * stride) for i in range(n_frames)]

    frames = vr.get_batch(indices).asnumpy()
    result = []
    for frame in frames:
        img = Image.fromarray(frame).convert("RGB")
        if resize_max_width > 0 and img.width > resize_max_width:
            new_h = max(1, round(img.height * resize_max_width / img.width))
            img = img.resize((resize_max_width, new_h), Image.Resampling.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode()
        result.append(f"data:image/jpeg;base64,{b64}")
    return result


def call_causality_filter_t2v(
    api_base: str,
    api_key: str,
    model: str,
    sample: dict,
    n_frames_per_option: int = 3,
    retries: int = 3,
) -> dict | None:
    """
    T→V causality filter: checks if text context is sufficient and video options are distinguishable.

    Input (from sample metadata):
      - _ctx_captions: list of text step descriptions (context)
      - _correct_option: the correct answer letter
      - _option_video_paths: 4 video paths (options A/B/C/D)

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
    ctx_captions: list[str] = meta["_ctx_captions"]
    correct_option: str = meta["_correct_option"]
    option_paths: list[str] = meta["_option_video_paths"]

    if problem_type == "add_t2v":
        task_type = "predict the correct NEXT cooking step video after reading the text context"
    else:
        # replace_t2v
        all_steps: list[str] = meta.get("_all_steps", [])
        task_type = f"identify the correct [MISSING] step video given the surrounding text steps"

    text_context_str = "\n".join(
        f"Step {i + 1}: {cap}" for i, cap in enumerate(ctx_captions)
    )

    user_text = CAUSALITY_T2V_USER_PROMPT.format(
        task_type=task_type,
        text_context_str=text_context_str,
        correct_option=correct_option,
    )

    # Build message content: user text + keyframes for each option
    labels = _option_labels(len(option_paths))
    content: list[dict] = [{"type": "text", "text": user_text}]
    for label, vpath in zip(labels, option_paths):
        content.append({"type": "text", "text": f"\n[Option {label} keyframes]"})
        try:
            data_urls = _sample_frames_from_video(vpath, n_frames=n_frames_per_option)
        except Exception as exc:
            print(f"  [filter] frame extraction failed for {vpath}: {exc}", file=sys.stderr)
            return None
        for url in data_urls:
            content.append({
                "type": "image_url",
                "image_url": {"url": url, "detail": "low"},
            })

    messages = [
        {"role": "system", "content": CAUSALITY_T2V_SYSTEM_PROMPT},
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
            return json.loads(resp.choices[0].message.content)
        except Exception as exc:
            last_err = exc
            if attempt < retries - 1:
                time.sleep(2 ** attempt)

    print(f"  [filter] API error: {last_err}", file=sys.stderr)
    return None


def filter_samples_with_ai(
    samples: list[dict],
    api_base: str,
    api_key: str,
    model: str,
    confidence_threshold: float = 0.75,
    workers: int = 4,
    n_frames_per_option: int = 3,
) -> list[dict]:
    """Run T→V causality filter over all samples in parallel."""
    stats = Counter()

    def check_one(idx_sample):
        idx, s = idx_sample
        result = call_causality_filter_t2v(
            api_base, api_key, model, s,
            n_frames_per_option=n_frames_per_option,
        )
        if result is None:
            return idx, False, "api_failed"
        valid = result.get("causal_valid", False)
        conf = float(result.get("confidence", 0.0))
        if valid and conf >= confidence_threshold:
            return idx, True, "kept"
        return idx, False, f"filtered(valid={valid},conf={conf:.2f})"

    print(f"\n[AI Filter] Checking {len(samples)} samples with {workers} workers …")
    kept_flags: dict[int, bool] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(check_one, (i, s)): i for i, s in enumerate(samples)}
        for fi, future in enumerate(as_completed(futures), 1):
            idx, keep, reason = future.result()
            kept_flags[idx] = keep
            stats[reason] += 1
            if fi % 50 == 0 or fi == len(samples):
                print(f"  [{fi}/{len(samples)}] {dict(stats)}")

    filtered = [s for i, s in enumerate(samples) if kept_flags.get(i, False)]
    print(f"\n[AI Filter] {len(filtered)}/{len(samples)} samples kept. Stats: {dict(stats)}")
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# Output cleanup
# ─────────────────────────────────────────────────────────────────────────────

def _clean(s: dict) -> dict:
    """Strip internal AI-filter fields (_xxx) from final output."""
    meta = {k: v for k, v in s.get("metadata", {}).items() if not k.startswith("_")}
    return {**s, "metadata": meta}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "构造 T→V Event Logic 训练数据 (add_t2v / replace_t2v)。\n"
            "上下文 = 文字描述，选项 = 视频 clip（与 V→T 方向互补）。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input
    parser.add_argument("--manifest-jsonl", required=True,
                        help="AoT event manifest JSONL (aot_event_manifest.jsonl)。"
                             "每条记录含 source_video_id / sequence_index / clip_key / forward_video_path。")
    parser.add_argument("--captions-jsonl", required=True,
                        help="Step captions JSONL (l2_step_captions.jsonl)，由 annotate_l2_step_captions.py 生成。")
    parser.add_argument("--output", "-o", required=True, help="输出 JSONL 文件路径")

    # Task hyper-params
    parser.add_argument("--min-events", type=int, default=4, help="视频最少事件数（默认 4）")
    parser.add_argument("--min-context", type=int, default=2, help="Add 任务最少 context 步数（默认 2）")
    parser.add_argument("--max-context", type=int, default=4, help="Add 任务最多 context 步数（默认 4）")
    parser.add_argument("--replace-seq-len", type=int, default=5, help="Replace 序列长度（默认 5）")
    parser.add_argument("--add-per-video", type=int, default=1, help="每视频生成 add 样本数（默认 1）")
    parser.add_argument("--replace-per-video", type=int, default=1, help="每视频生成 replace 样本数（默认 1）")

    # AI filter
    parser.add_argument("--filter", action="store_true",
                        help="启用 T→V AI 因果有效性过滤")
    parser.add_argument("--api-base", default="https://api.novita.ai/v3/openai")
    parser.add_argument("--api-key", default="",
                        help="API key（默认读取 NOVITA_API_KEY / OPENAI_API_KEY 环境变量）")
    parser.add_argument("--model", default="qwen/qwen2.5-vl-72b-instruct")
    parser.add_argument("--confidence-threshold", type=float, default=0.75)
    parser.add_argument("--filter-workers", type=int, default=4)
    parser.add_argument("--frames-per-option", type=int, default=3,
                        help="每个 option 视频采样的关键帧数（默认 3）")

    # Output control
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)

    args = parser.parse_args()
    rng = random.Random(args.seed)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"📂 加载 manifest: {args.manifest_jsonl}")
    records = load_manifest(args.manifest_jsonl)
    print(f"   事件总数: {len(records)}")

    print(f"📂 加载 step captions: {args.captions_jsonl}")
    captions = load_captions(args.captions_jsonl)
    print(f"   已有 caption 的 clip 数: {len(captions)}")

    sequences, flat_pool = build_video_sequences(records, captions, min_events=args.min_events)

    if not sequences:
        print("❌ 无可用视频序列，请检查 manifest 和 captions 的 clip_key 是否匹配。", file=sys.stderr)
        sys.exit(1)

    # ── Build samples ─────────────────────────────────────────────────────────
    samples: list[dict] = []
    stats = Counter()

    for seq in sequences:
        for _ in range(args.add_per_video):
            s = build_add_t2v_sample(seq, flat_pool, args.min_context, args.max_context, rng)
            if s is not None:
                samples.append(s)
                stats["add_t2v"] += 1

        for _ in range(args.replace_per_video):
            s = build_replace_t2v_sample(seq, flat_pool, args.replace_seq_len, rng)
            if s is not None:
                samples.append(s)
                stats["replace_t2v"] += 1

    print(f"\n📊 初始样本数: {len(samples)} {dict(stats)}")

    # ── AI causality filter ───────────────────────────────────────────────────
    if args.filter:
        samples = filter_samples_with_ai(
            samples,
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.model,
            confidence_threshold=args.confidence_threshold,
            workers=args.filter_workers,
            n_frames_per_option=args.frames_per_option,
        )

    if args.shuffle:
        rng.shuffle(samples)

    if args.max_samples and args.max_samples > 0:
        samples = samples[: args.max_samples]

    # ── Write output ──────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(_clean(s), ensure_ascii=False) + "\n")

    print(f"\n✅ 写出完成: {len(samples)} 条样本 → {out_path}")
    final_stats = Counter(s["problem_type"] for s in samples)
    print("📊 任务分布:", dict(final_stats))


if __name__ == "__main__":
    main()
