#!/usr/bin/env python3
"""
build_gseg_data.py — Convert grounding+segmentation annotations into training JSONL.

Reads annotation JSONs produced by annotate_gseg.py and outputs train/val JSONL
in the same format consumed by the EasyR1 training pipeline.

Usage:
    python build_gseg_data.py \
        --annotation-dir /path/to/gseg_annotations \
        --output-dir /path/to/gseg_training_data \
        --video-dir /path/to/source_videos \
        --use-think
"""

import argparse
import json
import os
import random
from pathlib import Path

from prompts_gseg import get_training_prompt, get_training_prompt_with_think

PROBLEM_TYPE = "grounding_seg"
PROBLEM_TYPE_THINK = "grounding_seg_cot"


# ─────────────────────────────────────────────────────────────────────────────
# Annotation loading
# ─────────────────────────────────────────────────────────────────────────────

def load_annotations(ann_dir: str, min_segments: int = 2, max_segments: int = 20) -> list[dict]:
    """Load annotation JSONs and flatten multi-task entries.

    Supports both multi-task format (``tasks: [...]``) and legacy
    single-task format (``query`` + ``segments`` at top level).
    Returns a flat list where each element is a single task dict with
    ``clip_key``, ``source_video_path``, ``clip_duration_sec`` propagated.
    """
    results = []
    ann_path = Path(ann_dir)
    for fp in sorted(ann_path.glob("*.json")):
        try:
            with open(fp, encoding="utf-8") as f:
                ann = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        clip_key = ann.get("clip_key", fp.stem)
        source_video = ann.get("source_video_path", "")
        duration = ann.get("clip_duration_sec", 0)

        # Multi-task format
        tasks = ann.get("tasks", [])
        if not tasks:
            # Legacy single-task fallback
            if ann.get("query") and ann.get("segments"):
                tasks = [ann]

        for i, task in enumerate(tasks):
            if not task.get("query") or not task.get("segments"):
                continue
            n_seg = len(task["segments"])
            if n_seg < min_segments or n_seg > max_segments:
                continue
            # Propagate clip-level fields into each task
            flat = dict(task)
            flat["clip_key"] = clip_key
            flat["source_video_path"] = source_video
            flat["clip_duration_sec"] = duration
            flat["task_index"] = i
            results.append(flat)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Record building
# ─────────────────────────────────────────────────────────────────────────────

def build_record(
    ann: dict,
    video_dir: str,
    use_think: bool = False,
) -> dict | None:
    """Build a single training record from an annotation."""
    clip_key = ann.get("clip_key", "")
    duration = int(ann.get("clip_duration_sec", 0))
    if duration <= 0:
        return None

    query = ann["query"]
    segments = ann["segments"]

    # Build answer spans (chronological order)
    spans = []
    for seg in sorted(segments, key=lambda s: s.get("start_time", 0)):
        s = int(seg.get("start_time", 0))
        e = int(seg.get("end_time", 0))
        if e > s:
            spans.append([s, e])

    if not spans:
        return None

    # Resolve video path
    source_video = ann.get("source_video_path", "")
    if video_dir and source_video:
        video_name = Path(source_video).name
        video_path = os.path.join(video_dir, video_name)
    elif source_video:
        video_path = source_video
    else:
        video_path = ""

    # Build prompt
    if use_think:
        prompt = get_training_prompt_with_think(query)
        ptype = PROBLEM_TYPE_THINK
    else:
        prompt = get_training_prompt(query)
        ptype = PROBLEM_TYPE

    answer = f"<events>{json.dumps(spans)}</events>"

    # Grounding metadata
    grounding = ann.get("grounding", {})

    return {
        "messages": [{"role": "user", "content": prompt}],
        "prompt": prompt,
        "answer": answer,
        "videos": [video_path] if video_path else [],
        "data_type": "video",
        "problem_type": ptype,
        "metadata": {
            "clip_key": clip_key,
            "clip_duration_sec": duration,
            "domain": ann.get("domain", ""),
            "query_style": ann.get("query_style", ""),
            "output_count": len(spans),
            "grounding_start": grounding.get("start_time"),
            "grounding_end": grounding.get("end_time"),
            "noise_ratio": _compute_noise_ratio(grounding, duration),
            "source_video_path": source_video,
            "reorderable": ann.get("reorderable", False),
            "reorder_reason": ann.get("reorder_reason", ""),
        },
    }


def _compute_noise_ratio(grounding: dict, duration: int) -> float:
    """Fraction of the video outside the grounding boundaries."""
    if not grounding or duration <= 0:
        return 0.0
    gs = grounding.get("start_time", 0)
    ge = grounding.get("end_time", duration)
    relevant = max(0, ge - gs)
    return round(1.0 - relevant / duration, 3) if duration > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Balanced sampling (by domain)
# ─────────────────────────────────────────────────────────────────────────────

def balanced_sample(records: list[dict], target: int, rng: random.Random) -> list[dict]:
    """Domain-balanced downsampling."""
    by_domain: dict[str, list[dict]] = {}
    for r in records:
        d = r.get("metadata", {}).get("domain", "other")
        by_domain.setdefault(d, []).append(r)

    n_domains = len(by_domain)
    if n_domains == 0:
        return []

    per_domain = max(1, target // n_domains)
    result = []
    surplus = 0

    for domain in sorted(by_domain):
        pool = by_domain[domain]
        rng.shuffle(pool)
        take = min(len(pool), per_domain)
        result.extend(pool[:take])
        if take < per_domain:
            surplus += per_domain - take

    # Redistribute surplus
    if surplus > 0:
        remaining = []
        for domain in sorted(by_domain):
            pool = by_domain[domain]
            if len(pool) > per_domain:
                remaining.extend(pool[per_domain:])
        rng.shuffle(remaining)
        result.extend(remaining[:surplus])

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build training JSONL from grounding+segmentation annotations.",
    )
    parser.add_argument("--annotation-dir", required=True,
                        help="Directory of annotation JSONs from annotate_gseg.py")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for train.jsonl / val.jsonl")
    parser.add_argument("--video-dir", default="",
                        help="Directory containing source videos (to resolve paths)")
    parser.add_argument("--use-think", action="store_true",
                        help="Use <think> CoT format in prompts")
    parser.add_argument("--min-segments", type=int, default=2)
    parser.add_argument("--max-segments", type=int, default=20)
    parser.add_argument("--val-ratio", type=float, default=0.15,
                        help="Fraction of data for validation")
    parser.add_argument("--balance-target", type=int, default=0,
                        help="Target total samples for domain-balanced sampling (0 = no balancing)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Load annotations
    annotations = load_annotations(args.annotation_dir, args.min_segments, args.max_segments)
    print(f"Loaded {len(annotations)} valid annotations from {args.annotation_dir}")

    if not annotations:
        print("No valid annotations found.")
        return

    # Build records
    records = []
    for ann in annotations:
        rec = build_record(ann, args.video_dir, args.use_think)
        if rec is not None:
            records.append(rec)

    print(f"Built {len(records)} training records")

    if not records:
        print("No valid records produced.")
        return

    # Domain-balanced sampling
    if args.balance_target > 0:
        before = len(records)
        records = balanced_sample(records, args.balance_target, rng)
        print(f"Balanced sampling: {before} → {len(records)} (target={args.balance_target})")

    # Collect stats
    style_counts: dict[str, int] = {}
    domain_counts: dict[str, int] = {}
    for r in records:
        m = r.get("metadata", {})
        qs = m.get("query_style", "?")
        style_counts[qs] = style_counts.get(qs, 0) + 1
        d = m.get("domain", "?")
        domain_counts[d] = domain_counts.get(d, 0) + 1

    print(f"\n=== Query Style Distribution ===")
    for k, v in sorted(style_counts.items()):
        print(f"  {k}: {v}")
    print(f"\n=== Domain Distribution ===")
    for k, v in sorted(domain_counts.items()):
        print(f"  {k}: {v}")

    # Train/val split
    rng.shuffle(records)
    n_val = max(1, int(len(records) * args.val_ratio))
    val_records = records[:n_val]
    train_records = records[n_val:]

    print(f"\n=== Output ===")
    print(f"  Train: {len(train_records)}")
    print(f"  Val:   {len(val_records)}")

    # Write
    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.jsonl")
    val_path = os.path.join(args.output_dir, "val.jsonl")
    _write_jsonl(train_records, train_path)
    _write_jsonl(val_records, val_path)

    print(f"  Dir:   {args.output_dir}")

    # Example
    if train_records:
        ex = train_records[0]
        print(f"\n  --- Example ---")
        print(f"  problem_type: {ex['problem_type']}")
        print(f"  query: {ex['prompt'][:200]}")
        print(f"  answer: {ex['answer'][:200]}")
        print(f"  video: {ex['videos'][0] if ex['videos'] else 'N/A'}")


def _write_jsonl(records: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
