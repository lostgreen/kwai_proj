#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build V2T/T2V MCQ datasets from event-level manifests and caption pairs.

Example:
python proxy_data/temporal_aot/build_aot_mcq.py \
  --manifest-jsonl /tmp/aot_event_manifest.jsonl \
  --caption-pairs /tmp/aot_annotations/caption_pairs.jsonl \
  --v2t-output /tmp/v2t_train.jsonl \
  --t2v-output /tmp/t2v_train.jsonl \
  --max-samples 500
"""

from __future__ import annotations

import argparse
import json
import os
import random

from prompts import get_t2v_prompt, get_v2t_prompt


def load_jsonl(path: str) -> list[dict]:
    items: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Build event-level AoT V2T/T2V datasets.")
    parser.add_argument("--manifest-jsonl", required=True, help="Manifest from build_event_aot_data.py")
    parser.add_argument("--caption-pairs", required=True, help="caption_pairs.jsonl from annotate_event_captions.py")
    parser.add_argument("--v2t-output", required=True, help="Output JSONL for V2T")
    parser.add_argument("--t2v-output", required=True, help="Output JSONL for T2V")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-samples", type=int, default=0, help="Max number of paired samples to keep")
    parser.add_argument("--min-confidence", type=float, default=0.6, help="Minimum confidence for both captions")
    args = parser.parse_args()

    random.seed(args.seed)
    manifest = {item["clip_key"]: item for item in load_jsonl(args.manifest_jsonl)}
    pairs = load_jsonl(args.caption_pairs)
    random.shuffle(pairs)

    v2t_records: list[dict] = []
    t2v_records: list[dict] = []
    kept = 0

    for pair in pairs:
        if args.max_samples > 0 and kept >= args.max_samples:
            break
        if pair.get("forward_confidence", 0.0) < args.min_confidence:
            continue
        if pair.get("reverse_confidence", 0.0) < args.min_confidence:
            continue
        if not pair.get("is_different", False):
            continue

        clip_key = pair["clip_key"]
        item = manifest.get(clip_key)
        if item is None:
            continue

        forward_caption = pair["forward_caption"].strip()
        reverse_caption = pair["reverse_caption"].strip()
        forward_video = item["forward_video_path"]
        reverse_video = item.get("reverse_video_path") or ""
        composite_video = item.get("composite_video_path") or ""

        v2t_records.append(
            {
                "messages": [{"role": "user", "content": get_v2t_prompt(forward_caption, reverse_caption)}],
                "prompt": get_v2t_prompt(forward_caption, reverse_caption),
                "answer": "A",
                "videos": [forward_video],
                "data_type": "video",
                "problem_type": "aot_v2t",
                "metadata": {
                    "clip_key": clip_key,
                    "forward_caption": forward_caption,
                    "reverse_caption": reverse_caption,
                    "video_direction": "forward",
                },
            }
        )

        if reverse_video:
            v2t_records.append(
                {
                    "messages": [{"role": "user", "content": get_v2t_prompt(forward_caption, reverse_caption)}],
                    "prompt": get_v2t_prompt(forward_caption, reverse_caption),
                    "answer": "B",
                    "videos": [reverse_video],
                    "data_type": "video",
                    "problem_type": "aot_v2t",
                    "metadata": {
                        "clip_key": clip_key,
                        "forward_caption": forward_caption,
                        "reverse_caption": reverse_caption,
                        "video_direction": "reverse",
                    },
                }
            )

        if composite_video:
            t2v_records.append(
                {
                    "messages": [{"role": "user", "content": get_t2v_prompt(forward_caption)}],
                    "prompt": get_t2v_prompt(forward_caption),
                    "answer": "A",
                    "videos": [composite_video],
                    "data_type": "video",
                    "problem_type": "aot_t2v",
                    "metadata": {
                        "clip_key": clip_key,
                        "caption": forward_caption,
                        "first_segment_direction": "forward",
                        "second_segment_direction": "reverse",
                    },
                }
            )

            t2v_records.append(
                {
                    "messages": [{"role": "user", "content": get_t2v_prompt(reverse_caption)}],
                    "prompt": get_t2v_prompt(reverse_caption),
                    "answer": "B",
                    "videos": [composite_video],
                    "data_type": "video",
                    "problem_type": "aot_t2v",
                    "metadata": {
                        "clip_key": clip_key,
                        "caption": reverse_caption,
                        "first_segment_direction": "forward",
                        "second_segment_direction": "reverse",
                    },
                }
            )

        kept += 1

    os.makedirs(os.path.dirname(args.v2t_output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.t2v_output) or ".", exist_ok=True)
    for path, records in ((args.v2t_output, v2t_records), (args.t2v_output, t2v_records)):
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(v2t_records)} V2T records to {args.v2t_output}")
    print(f"Wrote {len(t2v_records)} T2V records to {args.t2v_output}")


if __name__ == "__main__":
    main()
