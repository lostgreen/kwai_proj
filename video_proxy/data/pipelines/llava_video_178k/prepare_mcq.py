#!/usr/bin/env python3
"""
Convert LLaVA-Video-178K MCQ data to unified JSONL for offline rollout filtering.

Reads all *_mc_qa_processed.json files from the LLaVA-Video-178K dataset,
extracts each single-turn MCQ conversation and converts to the JSONL format
expected by offline_rollout_filter.py.

Input format (LLaVA-Video-178K):
    {
        "id": "1006-8968804598",
        "conversations": [
            {"from": "human", "value": "<image>\nQuestion?\nOptions:\nA. ...\nE. ...\nPlease provide..."},
            {"from": "gpt", "value": "E. training."}
        ],
        "data_source": "0_30_s_nextqa",
        "video": "NextQA/NExTVideo/1006/8968804598.mp4"
    }

Output format (unified JSONL, one line per QA):
    {
        "prompt": "<video>\nQuestion?\nOptions:\nA. ...\nE. ...\n\nAnswer with the option letter.",
        "answer": "E",
        "videos": ["/abs/path/to/video.mp4"],
        "problem_type": "llava_mcq",
        "data_type": "video",
        "metadata": {
            "id": "...",
            "source": "nextqa",
            "duration_bucket": "0_30_s",
            "data_source": "0_30_s_nextqa"
        }
    }

Usage:
    python prepare_mcq.py \
        --dataset-root /path/to/LLaVA-Video-178K \
        --output mcq_all.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path


# Map data_source to (duration_bucket, source_name)
_DURATION_PATTERN = re.compile(r"^(\d+_\d+_[sm])_(.+)$")


def parse_data_source(data_source: str) -> tuple[str, str]:
    """Extract duration_bucket and source from data_source like '0_30_s_nextqa'."""
    m = _DURATION_PATTERN.match(data_source)
    if m:
        return m.group(1), m.group(2)
    return "unknown", data_source


def extract_answer_letter(gpt_value: str) -> str | None:
    """Extract the answer letter from GPT response like 'E. training.'"""
    gpt_value = gpt_value.strip()
    m = re.match(r"^([A-Z])\.", gpt_value)
    if m:
        return m.group(1)
    # Fallback: single letter
    m = re.match(r"^([A-Z])$", gpt_value)
    if m:
        return m.group(1)
    return None


def rewrite_prompt(human_value: str) -> str:
    """Rewrite LLaVA human prompt to unified format.

    - Replace <image> with <video> (LLaVA-Video uses <image> tag for videos)
    - Strip the original instruction suffix, add a clean one
    """
    # Replace <image> with <video>
    text = human_value.replace("<image>", "<video>")

    # Remove the trailing instruction line if present
    text = re.sub(
        r"\n*Please provide your answer by stating the letter followed by the full option\.\s*$",
        "",
        text,
    )

    # Add clean instruction
    text = text.rstrip() + "\n\nAnswer with the option letter."
    return text


def load_frame_meta(dataset_root: str) -> dict[str, dict]:
    """Load all frame_meta.jsonl files and build a lookup by video filename stem.

    Returns {video_stem: {"video_path": ..., "duration": ..., "frame_count": ...}}
    """
    meta_lookup: dict[str, dict] = {}
    root = Path(dataset_root)
    for meta_path in sorted(root.glob("*/frame_meta.jsonl")):
        with open(meta_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                vpath = entry.get("video_path", "")
                if vpath:
                    stem = Path(vpath).stem
                    meta_lookup[stem] = entry
    if meta_lookup:
        print(f"  Loaded frame_meta: {len(meta_lookup)} entries")
    return meta_lookup


def resolve_video_path(
    video_rel: str,
    data_source: str,
    dataset_root: str,
    frame_meta: dict[str, dict],
) -> tuple[str, float | None]:
    """Resolve video relative path to absolute path with fallback.

    Returns (abs_path, duration_or_None).

    Resolution order:
    1. {dataset_root}/{video_rel}  (works for NextQA, youtube, perceptiontest)
    2. {dataset_root}/{data_source}/{video_rel}  (fallback for academic inside bucket)
    3. frame_meta lookup by video stem  (academic with frame_meta.jsonl)
    """
    root = Path(dataset_root)
    duration = None

    # Try 1: from dataset root
    candidate = root / video_rel
    if candidate.is_file():
        return str(candidate), duration

    # Try 2: from bucket dir
    if data_source:
        candidate2 = root / data_source / video_rel
        if candidate2.is_file():
            return str(candidate2), duration

    # Try 3: frame_meta lookup
    stem = Path(video_rel).stem
    if stem in frame_meta:
        entry = frame_meta[stem]
        meta_path = entry.get("video_path", "")
        duration = entry.get("duration")
        if meta_path:
            return meta_path, duration

    # Fallback: return root-based path (will fail at runtime if not found)
    return str(root / video_rel), duration


def convert_record(
    raw: dict,
    dataset_root: str,
    frame_meta: dict[str, dict] | None = None,
) -> dict | None:
    """Convert one LLaVA-Video-178K MCQ record to unified format.

    Returns None if the record is invalid (no MCQ, bad answer, etc.).
    """
    convs = raw.get("conversations", [])
    if len(convs) < 2:
        return None

    human_msg = convs[0]
    gpt_msg = convs[1]
    if human_msg.get("from") != "human" or gpt_msg.get("from") != "gpt":
        return None

    # Extract answer letter
    answer = extract_answer_letter(gpt_msg["value"])
    if answer is None:
        return None

    # Check this is actually MCQ (has Options: section)
    human_value = human_msg["value"]
    if "Options:" not in human_value and "\nA." not in human_value:
        return None

    # Build prompt
    prompt = rewrite_prompt(human_value)

    # Video path resolution
    video_rel = raw.get("video", "")
    data_source = raw.get("data_source", "")
    video_path, duration = resolve_video_path(
        video_rel, data_source, dataset_root, frame_meta or {},
    )

    # Parse source info
    duration_bucket, source = parse_data_source(data_source)

    metadata: dict = {
        "id": raw.get("id", ""),
        "source": source,
        "duration_bucket": duration_bucket,
        "data_source": data_source,
    }
    if duration is not None:
        metadata["duration"] = duration

    return {
        "prompt": prompt,
        "answer": answer,
        "videos": [video_path],
        "problem_type": "llava_mcq",
        "data_type": "video",
        "metadata": metadata,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert LLaVA-Video-178K MCQ data to unified JSONL"
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Root directory of LLaVA-Video-178K dataset",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL path",
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        default=None,
        help="Only include these sources (e.g., nextqa perceptiontest). Default: all",
    )
    parser.add_argument(
        "--duration-buckets",
        nargs="*",
        default=None,
        help="Only include these duration buckets (e.g., 0_30_s 30_60_s). Default: all",
    )
    parser.add_argument(
        "--verify-videos",
        action="store_true",
        help="Check that each video file exists (skip missing ones)",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root
    stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    records = []
    skipped = 0
    missing_videos = 0

    # Load frame_meta for academic sources
    print("Loading frame_meta files...")
    frame_meta = load_frame_meta(dataset_root)

    # Walk all *_mc_qa_processed.json files
    for entry in sorted(os.listdir(dataset_root)):
        full = os.path.join(dataset_root, entry)
        if not os.path.isdir(full):
            continue
        # Only duration-prefixed directories
        if not re.match(r"\d+_\d+_", entry):
            continue

        for fname in sorted(os.listdir(full)):
            if not fname.endswith(".json") or "_mc_" not in fname:
                continue

            json_path = os.path.join(full, fname)
            print(f"  Loading {entry}/{fname} ...", end="", flush=True)
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                print(f" skip (not a list)")
                continue

            count = 0
            for raw in data:
                rec = convert_record(raw, dataset_root, frame_meta)
                if rec is None:
                    skipped += 1
                    continue

                # Apply filters
                meta = rec["metadata"]
                if args.sources and meta["source"] not in args.sources:
                    skipped += 1
                    continue
                if args.duration_buckets and meta["duration_bucket"] not in args.duration_buckets:
                    skipped += 1
                    continue

                # Verify video exists
                if args.verify_videos:
                    vpath = rec["videos"][0]
                    if not os.path.isfile(vpath):
                        missing_videos += 1
                        if missing_videos <= 5:
                            print(f"\n    WARN: video not found: {vpath}")
                        continue

                records.append(rec)
                stats[meta["duration_bucket"]][meta["source"]] += 1
                count += 1

            print(f" {count} MCQ records")

    # Write output
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"  Total MCQ records: {len(records)}  (skipped: {skipped})")
    if args.verify_videos:
        print(f"  Missing videos: {missing_videos}")
    print(f"  Output: {args.output}")
    print(f"{'='*60}")

    # Print grid
    all_sources = sorted({s for bkt in stats.values() for s in bkt})
    header = f"{'bucket':>15s}" + "".join(f"{s:>18s}" for s in all_sources) + f"{'TOTAL':>10s}"
    print(header)
    for bkt in sorted(stats.keys()):
        row_total = sum(stats[bkt].values())
        row = f"{bkt:>15s}" + "".join(f"{stats[bkt].get(s, 0):>18d}" for s in all_sources) + f"{row_total:>10d}"
        print(row)
    grand_total = sum(sum(bkt.values()) for bkt in stats.values())
    print(f"{'TOTAL':>15s}" + "".join(f"{sum(stats[b].get(s,0) for b in stats):>18d}" for s in all_sources) + f"{grand_total:>10d}")


if __name__ == "__main__":
    main()
