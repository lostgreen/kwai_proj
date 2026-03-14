#!/usr/bin/env python3
"""
build_dataset.py — Convert per-clip Level 1 annotations into training JSONL.

Reads annotation JSON files produced by annotate.py and builds a new training
dataset in EasyR1 format that can be used for RL training with hierarchical
temporal segmentation rewards.

Usage:
    python build_dataset.py \
        --annotation-dir proxy_data/youcook2_seg_annotation/annotations \
        --output proxy_data/youcook2_seg_annotation/youcook2_hier_train.jsonl \
        --level 1 \
        [--split train]

Output format per record:
    {
        "messages": [...],
        "prompt": "...",
        "answer": "{\"macro_phases\": [...]}",
        "videos": ["path/to/clip.mp4"],
        "data_type": "video",
        "problem_type": "temporal_seg_hier_L1",
        "metadata": {
            "clip_key": "...",
            "clip_duration_sec": 84.0,
            "level": 1
        }
    }
"""

import argparse
import json
import sys
from pathlib import Path

from prompts import SYSTEM_PROMPT, get_level1_prompt, get_level2_prompt, get_level3_prompt


_LEVEL_TO_PROBLEM_TYPE = {
    1: "temporal_seg_hier_L1",
    2: "temporal_seg_hier_L2",
    3: "temporal_seg_hier_L3",
}


def build_record(ann: dict, level: int) -> dict | None:
    """Convert an annotation dict into an EasyR1 training record."""
    level_key = f"level{level}"
    annotation = ann.get(level_key)
    if annotation is None or annotation.get("_parse_error"):
        return None

    clip_duration = float(ann.get("clip_duration_sec") or 0)
    if clip_duration <= 0:
        return None

    video_path = ann.get("source_video_path") or ann.get("video_path", "")

    # Build user prompt text
    if level == 1:
        user_text = get_level1_prompt(clip_duration)
    elif level == 2:
        l1 = ann.get("level1")
        if l1 is None:
            return None
        user_text = get_level2_prompt(clip_duration, l1)
    elif level == 3:
        l1 = ann.get("level1")
        l2 = ann.get("level2")
        if l1 is None or l2 is None:
            return None
        user_text = get_level3_prompt(clip_duration, l1, l2)
    else:
        return None

    # Full user message with video placeholder
    full_user = f"Watch the following cooking video clip carefully:\n<video>\n\n{user_text}"

    return {
        "messages": [
            {"role": "user", "content": full_user},
        ],
        "prompt": full_user,
        "answer": json.dumps(annotation, ensure_ascii=False),
        "videos": [video_path] if video_path else [],
        "data_type": "video",
        "problem_type": _LEVEL_TO_PROBLEM_TYPE[level],
        "metadata": {
            "clip_key": ann.get("clip_key", ""),
            "clip_duration_sec": clip_duration,
            "level": level,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert annotation JSONs to EasyR1 training JSONL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--annotation-dir", required=True,
                        help="Directory with per-clip .json annotation files")
    parser.add_argument("--output", required=True,
                        help="Output JSONL path")
    parser.add_argument("--level", type=int, choices=[1, 2, 3], default=1,
                        help="Which annotation level to build the dataset for")
    args = parser.parse_args()

    ann_dir = Path(args.annotation_dir)
    if not ann_dir.exists():
        print(f"ERROR: annotation-dir not found: {ann_dir}", file=sys.stderr)
        sys.exit(1)

    ann_files = sorted(ann_dir.glob("*.json"))
    print(f"Found {len(ann_files)} annotation files")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = skipped = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for ann_file in ann_files:
            try:
                with open(ann_file, encoding="utf-8") as f:
                    ann = json.load(f)
            except Exception as e:
                print(f"  SKIP (parse error): {ann_file.name}: {e}")
                skipped += 1
                continue

            record = build_record(ann, args.level)
            if record is None:
                skipped += 1
                continue

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Written: {written} records  Skipped: {skipped}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
