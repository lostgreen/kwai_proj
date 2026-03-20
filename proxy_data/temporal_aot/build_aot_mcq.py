#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build V2T/T2V/3way MCQ datasets from event-level manifests and caption pairs.

Problem types produced:
  aot_v2t       — binary: given one video (forward or reverse), pick the matching caption (A/B)
  aot_t2v       — binary: given a composite video, pick which segment matches a caption (A/B)
  aot_3way_v2t  — 3-option: given one video, pick the correct caption from
                  {forward, reverse, shuffle} (A/B/C)
  aot_3way_t2v  — 3-option: given a caption, pick the matching video from
                  {forward, reverse, shuffle} (A/B/C)

Example:
python proxy_data/temporal_aot/build_aot_mcq.py \
  --manifest-jsonl /tmp/aot_event_manifest.jsonl \
  --caption-pairs /tmp/aot_annotations/caption_pairs.jsonl \
  --v2t-output /tmp/v2t_train.jsonl \
  --t2v-output /tmp/t2v_train.jsonl \
  --threeway-output /tmp/3way_v2t_train.jsonl \
  --threeway-t2v-output /tmp/3way_t2v_train.jsonl \
  --max-samples 500
"""

from __future__ import annotations

import argparse
import json
import os
import random

from prompts import get_3way_t2v_prompt, get_3way_v2t_prompt, get_t2v_prompt, get_v2t_prompt


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build event-level AoT V2T/T2V/3way datasets.")
    parser.add_argument("--manifest-jsonl", required=True, help="Manifest from build_event_aot_data.py")
    parser.add_argument("--caption-pairs", required=True, help="caption_pairs.jsonl from annotate_event_captions.py")
    parser.add_argument("--v2t-output", default="", help="Output JSONL for binary V2T (empty = skip)")
    parser.add_argument("--t2v-output", default="", help="Output JSONL for binary T2V (empty = skip)")
    parser.add_argument("--threeway-output", default="", help="Output JSONL for 3-option V2T (forward/reverse/shuffle caption)")
    parser.add_argument("--threeway-t2v-output", default="", help="Output JSONL for 3-option T2V (given caption, pick from 3 videos)")
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

    v2t_records: list[dict] = []
    t2v_records: list[dict] = []
    threeway_records: list[dict] = []
    threeway_t2v_records: list[dict] = []
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
        # Randomize which caption occupies slot A vs B to prevent the model
        # from using linguistic naturalness (forward captions sound more natural)
        # as a position shortcut during offline rollout filtering.
        # ------------------------------------------------------------------
        if rng.random() < 0.5:
            opt_a, opt_b = forward_caption, reverse_caption
            fwd_v2t_answer = "A"
        else:
            opt_a, opt_b = reverse_caption, forward_caption
            fwd_v2t_answer = "B"
        rev_v2t_answer = "B" if fwd_v2t_answer == "A" else "A"
        v2t_prompt = get_v2t_prompt(opt_a, opt_b)
        v2t_records.append(
            {
                "messages": [{"role": "user", "content": v2t_prompt}],
                "prompt": v2t_prompt,
                "answer": fwd_v2t_answer,
                "videos": [forward_video],
                "data_type": "video",
                "problem_type": "aot_v2t",
                "metadata": {
                    "clip_key": clip_key,
                    "forward_caption": forward_caption,
                    "reverse_caption": reverse_caption,
                    "option_a_caption": opt_a,
                    "option_b_caption": opt_b,
                    "video_direction": "forward",
                },
            }
        )
        if reverse_video:
            v2t_records.append(
                {
                    "messages": [{"role": "user", "content": v2t_prompt}],
                    "prompt": v2t_prompt,
                    "answer": rev_v2t_answer,
                    "videos": [reverse_video],
                    "data_type": "video",
                    "problem_type": "aot_v2t",
                    "metadata": {
                        "clip_key": clip_key,
                        "forward_caption": forward_caption,
                        "reverse_caption": reverse_caption,
                        "option_a_caption": opt_a,
                        "option_b_caption": opt_b,
                        "video_direction": "reverse",
                    },
                }
            )

        # ------------------------------------------------------------------
        # Binary T2V
        # Randomize which segment label ("first" / "second") maps to option A
        # vs B to prevent the model learning "natural caption → first segment → A".
        # The composite video structure is fixed (forward first, reverse second);
        # only the option labels change.
        # ------------------------------------------------------------------
        if composite_video:
            if rng.random() < 0.5:
                seg_a, seg_b = "first", "second"
                fwd_t2v_answer = "A"  # forward caption → first segment = A
            else:
                seg_a, seg_b = "second", "first"
                fwd_t2v_answer = "B"  # forward caption → first segment = B
            rev_t2v_answer = "B" if fwd_t2v_answer == "A" else "A"
            fwd_t2v_prompt = get_t2v_prompt(
                forward_caption,
                option_a_text=f"The {seg_a} segment",
                option_b_text=f"The {seg_b} segment",
            )
            rev_t2v_prompt = get_t2v_prompt(
                reverse_caption,
                option_a_text=f"The {seg_a} segment",
                option_b_text=f"The {seg_b} segment",
            )
            t2v_records.append(
                {
                    "messages": [{"role": "user", "content": fwd_t2v_prompt}],
                    "prompt": fwd_t2v_prompt,
                    "answer": fwd_t2v_answer,
                    "videos": [composite_video],
                    "data_type": "video",
                    "problem_type": "aot_t2v",
                    "metadata": {
                        "clip_key": clip_key,
                        "caption": forward_caption,
                        "first_segment_direction": "forward",
                        "second_segment_direction": "reverse",
                        "option_a_segment": seg_a,
                        "option_b_segment": seg_b,
                    },
                }
            )
            t2v_records.append(
                {
                    "messages": [{"role": "user", "content": rev_t2v_prompt}],
                    "prompt": rev_t2v_prompt,
                    "answer": rev_t2v_answer,
                    "videos": [composite_video],
                    "data_type": "video",
                    "problem_type": "aot_t2v",
                    "metadata": {
                        "clip_key": clip_key,
                        "caption": reverse_caption,
                        "first_segment_direction": "forward",
                        "second_segment_direction": "reverse",
                        "option_a_segment": seg_a,
                        "option_b_segment": seg_b,
                    },
                }
            )

        # ------------------------------------------------------------------
        # 3-option V2T  (requires shuffle caption + shuffle video)
        # ------------------------------------------------------------------
        recipe_type = item.get("recipe_type") or "unknown"
        if shuffle_caption and shuffle_video and args.threeway_output:
            slots = [
                ("forward", forward_caption),
                ("reverse", reverse_caption),
                ("shuffle", shuffle_caption),
            ]
            rng.shuffle(slots)
            correct_letter = "ABC"[[s[0] for s in slots].index("forward")]
            opts = {f"option_{chr(65+i)}": s[1] for i, s in enumerate(slots)}
            prompt_text = get_3way_v2t_prompt(slots[0][1], slots[1][1], slots[2][1])
            threeway_records.append(
                {
                    "messages": [{"role": "user", "content": prompt_text}],
                    "prompt": prompt_text,
                    "answer": correct_letter,
                    "videos": [forward_video],
                    "data_type": "video",
                    "problem_type": "aot_3way_v2t",
                    "metadata": {
                        "clip_key": clip_key,
                        "video_direction": "forward",
                        "recipe_type": recipe_type,
                        **opts,
                        "option_types": {chr(65+i): s[0] for i, s in enumerate(slots)},
                    },
                }
            )

            # shuffle-video variant: correct answer is the shuffle caption
            correct_letter_shuf = "ABC"[[s[0] for s in slots].index("shuffle")]
            threeway_records.append(
                {
                    "messages": [{"role": "user", "content": prompt_text}],
                    "prompt": prompt_text,
                    "answer": correct_letter_shuf,
                    "videos": [shuffle_video],
                    "data_type": "video",
                    "problem_type": "aot_3way_v2t",
                    "metadata": {
                        "clip_key": clip_key,
                        "video_direction": "shuffle",
                        "recipe_type": recipe_type,
                        **opts,
                        "option_types": {chr(65+i): s[0] for i, s in enumerate(slots)},
                    },
                }
            )

            # reverse-video variant: correct answer is the reverse caption
            if reverse_video:
                correct_letter_rev = "ABC"[[s[0] for s in slots].index("reverse")]
                threeway_records.append(
                    {
                        "messages": [{"role": "user", "content": prompt_text}],
                        "prompt": prompt_text,
                        "answer": correct_letter_rev,
                        "videos": [reverse_video],
                        "data_type": "video",
                        "problem_type": "aot_3way_v2t",
                        "metadata": {
                            "clip_key": clip_key,
                            "video_direction": "reverse",
                            "recipe_type": recipe_type,
                            **opts,
                            "option_types": {chr(65+i): s[0] for i, s in enumerate(slots)},
                        },
                    }
                )

        # ------------------------------------------------------------------
        # 3-option T2V  (given caption, pick from 3 videos)
        # ------------------------------------------------------------------
        if shuffle_video and reverse_video and args.threeway_t2v_output:
            video_slots = [
                ("forward", forward_video),
                ("reverse", reverse_video),
                ("shuffle", shuffle_video),
            ]
            rng.shuffle(video_slots)
            correct_letter_fwd = "ABC"[[s[0] for s in video_slots].index("forward")]
            video_list = [s[1] for s in video_slots]
            prompt_text_t2v = get_3way_t2v_prompt(forward_caption)
            threeway_t2v_records.append(
                {
                    "messages": [{"role": "user", "content": prompt_text_t2v}],
                    "prompt": prompt_text_t2v,
                    "answer": correct_letter_fwd,
                    "videos": video_list,
                    "data_type": "video",
                    "problem_type": "aot_3way_t2v",
                    "metadata": {
                        "clip_key": clip_key,
                        "caption": forward_caption,
                        "caption_direction": "forward",
                        "recipe_type": recipe_type,
                        "video_types": {chr(65+i): s[0] for i, s in enumerate(video_slots)},
                    },
                }
            )

            # reverse-caption variant
            correct_letter_rev = "ABC"[[s[0] for s in video_slots].index("reverse")]
            prompt_text_t2v_rev = get_3way_t2v_prompt(reverse_caption)
            threeway_t2v_records.append(
                {
                    "messages": [{"role": "user", "content": prompt_text_t2v_rev}],
                    "prompt": prompt_text_t2v_rev,
                    "answer": correct_letter_rev,
                    "videos": video_list,
                    "data_type": "video",
                    "problem_type": "aot_3way_t2v",
                    "metadata": {
                        "clip_key": clip_key,
                        "caption": reverse_caption,
                        "caption_direction": "reverse",
                        "recipe_type": recipe_type,
                        "video_types": {chr(65+i): s[0] for i, s in enumerate(video_slots)},
                    },
                }
            )

            # shuffle-caption variant
            correct_letter_shuf = "ABC"[[s[0] for s in video_slots].index("shuffle")]
            prompt_text_t2v_shuf = get_3way_t2v_prompt(shuffle_caption)
            threeway_t2v_records.append(
                {
                    "messages": [{"role": "user", "content": prompt_text_t2v_shuf}],
                    "prompt": prompt_text_t2v_shuf,
                    "answer": correct_letter_shuf,
                    "videos": video_list,
                    "data_type": "video",
                    "problem_type": "aot_3way_t2v",
                    "metadata": {
                        "clip_key": clip_key,
                        "caption": shuffle_caption,
                        "caption_direction": "shuffle",
                        "recipe_type": recipe_type,
                        "video_types": {chr(65+i): s[0] for i, s in enumerate(video_slots)},
                    },
                }
            )

        kept += 1

    if args.v2t_output:
        _write_jsonl(args.v2t_output, v2t_records)
        print(f"Wrote {len(v2t_records)} V2T records to {args.v2t_output}")
    else:
        print(f"Skipped V2T output ({len(v2t_records)} records generated but no --v2t-output specified)")
    if args.t2v_output:
        _write_jsonl(args.t2v_output, t2v_records)
        print(f"Wrote {len(t2v_records)} T2V records to {args.t2v_output}")
    else:
        print(f"Skipped T2V output ({len(t2v_records)} records generated but no --t2v-output specified)")

    if args.threeway_output and threeway_records:
        _write_jsonl(args.threeway_output, threeway_records)
        print(f"Wrote {len(threeway_records)} 3-way V2T records to {args.threeway_output}")
    elif args.threeway_output:
        print("No 3-way V2T records produced (shuffle captions/videos missing)")

    if args.threeway_t2v_output and threeway_t2v_records:
        _write_jsonl(args.threeway_t2v_output, threeway_t2v_records)
        print(f"Wrote {len(threeway_t2v_records)} 3-way T2V records to {args.threeway_t2v_output}")
    elif args.threeway_t2v_output:
        print("No 3-way T2V records produced (shuffle/reverse videos missing)")


if __name__ == "__main__":
    main()
