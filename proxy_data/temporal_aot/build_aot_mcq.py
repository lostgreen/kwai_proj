#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build V2T/T2V/4way MCQ datasets from event-level manifests and caption pairs.

Problem types produced:
  aot_v2t       — binary: given one video (forward or reverse), pick the matching caption (A/B)
  aot_t2v       — binary: given a composite video, pick which segment matches a caption (A/B)
  aot_4way_v2t  — 4-option: given one video, pick the correct caption from
                  {forward, reverse, shuffle, hard-negative from same recipe_type} (A/B/C/D)
                  Only built when caption_pairs contains shuffle_caption entries.

Example:
python proxy_data/temporal_aot/build_aot_mcq.py \
  --manifest-jsonl /tmp/aot_event_manifest.jsonl \
  --caption-pairs /tmp/aot_annotations/caption_pairs.jsonl \
  --v2t-output /tmp/v2t_train.jsonl \
  --t2v-output /tmp/t2v_train.jsonl \
  --fourway-output /tmp/4way_train.jsonl \
  --max-samples 500
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict

from prompts import get_4way_v2t_prompt, get_t2v_prompt, get_v2t_prompt


def load_jsonl(path: str) -> list[dict]:
    items: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _write_jsonl(path: str, records: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _build_hard_negative_index(
    pairs: list[dict],
    manifest: dict[str, dict],
) -> dict[str, list[str]]:
    """Build a mapping from recipe_type → list of forward_captions for hard-negative sampling."""
    index: dict[str, list[str]] = defaultdict(list)
    for pair in pairs:
        clip_key = pair.get("clip_key", "")
        item = manifest.get(clip_key, {})
        recipe_type = item.get("recipe_type") or "unknown"
        caption = pair.get("forward_caption", "").strip()
        if caption:
            index[recipe_type].append(caption)
    return index


def _sample_hard_negative(
    rng: random.Random,
    index: dict[str, list[str]],
    recipe_type: str,
    exclude_captions: set[str],
) -> str:
    """Return a caption from the same recipe_type that is not in exclude_captions.
    Falls back to a random caption from any recipe_type if nothing suitable is found."""
    candidates = [c for c in index.get(recipe_type, []) if c not in exclude_captions]
    if not candidates:
        # Fallback: any caption not in exclude set
        all_caps = [c for caps in index.values() for c in caps if c not in exclude_captions]
        candidates = all_caps
    if not candidates:
        return ""
    return rng.choice(candidates)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build event-level AoT V2T/T2V/4way datasets.")
    parser.add_argument("--manifest-jsonl", required=True, help="Manifest from build_event_aot_data.py")
    parser.add_argument("--caption-pairs", required=True, help="caption_pairs.jsonl from annotate_event_captions.py")
    parser.add_argument("--v2t-output", required=True, help="Output JSONL for binary V2T")
    parser.add_argument("--t2v-output", required=True, help="Output JSONL for binary T2V")
    parser.add_argument("--fourway-output", default="", help="Output JSONL for 4-option V2T (only when shuffle captions exist)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-samples", type=int, default=0, help="Max number of paired samples to keep")
    parser.add_argument("--min-confidence", type=float, default=0.6, help="Minimum confidence for both captions")
    parser.add_argument(
        "--require-direction-clear",
        action="store_true",
        default=True,
        help=(
            "Drop pairs where VLM judged direction_clear=False (cyclic/ambiguous actions such as "
            "stirring, mixing, kneading, shaking). Enabled by default; pass --no-require-direction-clear to disable."
        ),
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    manifest = {item["clip_key"]: item for item in load_jsonl(args.manifest_jsonl)}
    pairs = load_jsonl(args.caption_pairs)
    rng.shuffle(pairs)

    hard_neg_index = _build_hard_negative_index(pairs, manifest)

    v2t_records: list[dict] = []
    t2v_records: list[dict] = []
    fourway_records: list[dict] = []
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
        if args.require_direction_clear:
            if not pair.get("forward_direction_clear", True):
                continue
            if not pair.get("reverse_direction_clear", True):
                continue

        clip_key = pair["clip_key"]
        item = manifest.get(clip_key)
        if item is None:
            continue

        forward_caption = pair["forward_caption"].strip()
        reverse_caption = pair["reverse_caption"].strip()
        shuffle_caption = pair.get("shuffle_caption", "").strip()
        forward_video = item["forward_video_path"]
        reverse_video = item.get("reverse_video_path") or ""
        composite_video = item.get("composite_video_path") or ""
        shuffle_video = item.get("shuffle_video_path") or ""

        # ------------------------------------------------------------------
        # Binary V2T
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # Binary T2V
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # 4-option V2T  (requires shuffle caption + shuffle video)
        # ------------------------------------------------------------------
        if shuffle_caption and shuffle_video and args.fourway_output:
            recipe_type = item.get("recipe_type") or "unknown"
            hard_neg = _sample_hard_negative(
                rng,
                hard_neg_index,
                recipe_type,
                exclude_captions={forward_caption, reverse_caption, shuffle_caption},
            )
            if not hard_neg:
                pass  # skip 4-way for this sample if no hard negative is available
            else:
                # Randomize which position (A/B/C/D) each caption occupies.
                # The correct caption class is always forward (for the forward video).
                slots = [
                    ("forward", forward_caption),
                    ("reverse", reverse_caption),
                    ("shuffle", shuffle_caption),
                    ("hard_neg", hard_neg),
                ]
                rng.shuffle(slots)
                correct_letter = "ABCD"[[s[0] for s in slots].index("forward")]
                opts = {f"option_{chr(65+i)}": s[1] for i, s in enumerate(slots)}
                prompt_text = get_4way_v2t_prompt(slots[0][1], slots[1][1], slots[2][1], slots[3][1])
                fourway_records.append(
                    {
                        "messages": [{"role": "user", "content": prompt_text}],
                        "prompt": prompt_text,
                        "answer": correct_letter,
                        "videos": [forward_video],
                        "data_type": "video",
                        "problem_type": "aot_4way_v2t",
                        "metadata": {
                            "clip_key": clip_key,
                            "video_direction": "forward",
                            "recipe_type": recipe_type,
                            **opts,
                            "option_types": {chr(65+i): s[0] for i, s in enumerate(slots)},
                        },
                    }
                )

                # Also build the shuffle-video variant: correct answer is the shuffle caption
                correct_letter_shuf = "ABCD"[[s[0] for s in slots].index("shuffle")]
                fourway_records.append(
                    {
                        "messages": [{"role": "user", "content": prompt_text}],
                        "prompt": prompt_text,
                        "answer": correct_letter_shuf,
                        "videos": [shuffle_video],
                        "data_type": "video",
                        "problem_type": "aot_4way_v2t",
                        "metadata": {
                            "clip_key": clip_key,
                            "video_direction": "shuffle",
                            "recipe_type": recipe_type,
                            **opts,
                            "option_types": {chr(65+i): s[0] for i, s in enumerate(slots)},
                        },
                    }
                )

        kept += 1

    _write_jsonl(args.v2t_output, v2t_records)
    _write_jsonl(args.t2v_output, t2v_records)
    print(f"Wrote {len(v2t_records)} V2T records to {args.v2t_output}")
    print(f"Wrote {len(t2v_records)} T2V records to {args.t2v_output}")

    if args.fourway_output and fourway_records:
        _write_jsonl(args.fourway_output, fourway_records)
        print(f"Wrote {len(fourway_records)} 4-way V2T records to {args.fourway_output}")
    elif args.fourway_output:
        print("No 4-way records produced (shuffle captions or hard negatives missing)")



if __name__ == "__main__":
    main()
