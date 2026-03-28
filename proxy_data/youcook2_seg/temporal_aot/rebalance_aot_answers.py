#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rebalance A/B answer distribution for filtered Temporal AoT datasets by swapping
option order in a subset of records.

This script does not drop any samples. Instead, it flips some majority-answer
records so the final kept data is as balanced as possible after filtering.

Examples:
python proxy_data/temporal_aot/rebalance_aot_answers.py \
  --input-jsonl proxy_data/temporal_aot/data/mixed_aot_train.offline_filtered.jsonl \
  --output-jsonl proxy_data/temporal_aot/data/mixed_aot_train.offline_filtered.balanced.jsonl

python proxy_data/temporal_aot/rebalance_aot_answers.py \
  --input-jsonl proxy_data/temporal_aot/data/mixed_aot_train.offline_filtered.jsonl \
  --output-jsonl /tmp/mixed_aot_train.balanced.jsonl \
  --balance-scope all \
  --seed 7
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from collections import Counter, defaultdict
from typing import Any

from prompts import get_v2t_prompt, get_t2v_prompt, get_3way_v2t_prompt


SUPPORTED_PROBLEM_TYPES = ("aot_t2v", "aot_v2t", "aot_3way_v2t", "aot_3way_t2v")
BINARY_PROBLEM_TYPES = {"aot_t2v", "aot_v2t"}
THREEWAY_PROBLEM_TYPES = {"aot_3way_v2t", "aot_3way_t2v"}
_THREEWAY_LETTERS = ("A", "B", "C")


def load_jsonl(path: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(path: str, records: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def flip_answer(answer: str) -> str:
    if answer == "A":
        return "B"
    if answer == "B":
        return "A"
    raise ValueError(f"Unsupported answer: {answer!r}")


def sync_prompt(record: dict[str, Any], prompt: str) -> None:
    record["prompt"] = prompt
    messages = record.get("messages")
    if not isinstance(messages, list):
        return
    for message in messages:
        if message.get("role") == "user":
            message["content"] = prompt
            break


def flip_record(record: dict[str, Any]) -> dict[str, Any]:
    problem_type = record.get("problem_type")
    flipped = copy.deepcopy(record)
    metadata = flipped.setdefault("metadata", {})

    if problem_type == "aot_v2t":
        # Use option_a/b_caption if present (randomized builds); fall back to
        # forward/reverse for backwards compatibility with old data.
        opt_a = metadata.get("option_a_caption") or metadata.get("forward_caption")
        opt_b = metadata.get("option_b_caption") or metadata.get("reverse_caption")
        if not isinstance(opt_a, str) or not isinstance(opt_b, str):
            raise ValueError("aot_v2t record is missing caption metadata")
        sync_prompt(flipped, get_v2t_prompt(opt_b, opt_a))
        metadata["option_a_caption"] = opt_b
        metadata["option_b_caption"] = opt_a
    elif problem_type == "aot_t2v":
        caption = metadata.get("caption")
        if not isinstance(caption, str):
            raise ValueError("aot_t2v record is missing caption")
        # Use stored segment labels (randomized builds); fall back to old defaults.
        cur_a_seg = metadata.get("option_a_segment", "first")
        cur_b_seg = metadata.get("option_b_segment", "second")
        sync_prompt(
            flipped,
            get_t2v_prompt(
                caption=caption,
                option_a_text=f"The {cur_b_seg} segment",
                option_b_text=f"The {cur_a_seg} segment",
            ),
        )
        metadata["option_a_segment"] = cur_b_seg
        metadata["option_b_segment"] = cur_a_seg
    else:
        raise ValueError(f"Unsupported problem_type for flipping: {problem_type!r}")

    flipped["answer"] = flip_answer(record["answer"])
    metadata["answer_rebalanced"] = True
    metadata["original_answer_before_rebalance"] = record["answer"]
    return flipped


def permute_3way_v2t_record(record: dict[str, Any], new_letter: str) -> dict[str, Any]:
    """Move the correct answer of a 3-way V2T record to `new_letter` by swapping two slots."""
    old_letter = record["answer"]
    if old_letter == new_letter:
        return copy.deepcopy(record)

    result = copy.deepcopy(record)
    metadata = result.setdefault("metadata", {})

    # Swap option caption texts
    old_cap = metadata.get(f"option_{old_letter}", "")
    new_cap = metadata.get(f"option_{new_letter}", "")
    metadata[f"option_{old_letter}"] = new_cap
    metadata[f"option_{new_letter}"] = old_cap

    # Swap semantic type tags
    ot = dict(metadata.get("option_types") or {})
    ot[old_letter], ot[new_letter] = ot.get(new_letter, ""), ot.get(old_letter, "")
    metadata["option_types"] = ot

    # Rebuild prompt from updated option texts
    new_prompt = get_3way_v2t_prompt(
        metadata.get("option_A", ""),
        metadata.get("option_B", ""),
        metadata.get("option_C", ""),
    )
    sync_prompt(result, new_prompt)

    result["answer"] = new_letter
    metadata["answer_rebalanced"] = True
    metadata["original_answer_before_rebalance"] = old_letter
    return result


def permute_3way_t2v_record(record: dict[str, Any], new_letter: str) -> dict[str, Any]:
    """Move the correct answer of a 3-way T2V record to `new_letter` by swapping two video slots."""
    old_letter = record["answer"]
    if old_letter == new_letter:
        return copy.deepcopy(record)

    result = copy.deepcopy(record)
    metadata = result.setdefault("metadata", {})

    # Swap video paths in the videos list (order = [A, B, C])
    _idx = {"A": 0, "B": 1, "C": 2}
    old_i, new_i = _idx[old_letter], _idx[new_letter]
    videos = list(result.get("videos") or [])
    if len(videos) == 3:
        videos[old_i], videos[new_i] = videos[new_i], videos[old_i]
        result["videos"] = videos

    # Swap semantic type tags
    vt = dict(metadata.get("video_types") or {})
    vt[old_letter], vt[new_letter] = vt.get(new_letter, ""), vt.get(old_letter, "")
    metadata["video_types"] = vt

    # Prompt text is generic ("Video A / Video B / Video C / Video D") — no rebuild needed
    result["answer"] = new_letter
    metadata["answer_rebalanced"] = True
    metadata["original_answer_before_rebalance"] = old_letter
    return result


def count_answers(records: list[dict[str, Any]], target_problem_types: set[str]) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for record in records:
        problem_type = record.get("problem_type")
        answer = record.get("answer")
        if problem_type in target_problem_types and isinstance(answer, str) and answer.upper() in "ABC":
            counts[(problem_type, answer.upper())] += 1
    return counts


def print_stats(title: str, records: list[dict[str, Any]], target_problem_types: set[str]) -> None:
    counts = count_answers(records, target_problem_types)
    print(title)
    for problem_type in sorted(target_problem_types):
        if problem_type in THREEWAY_PROBLEM_TYPES:
            parts = "  ".join(f"{l}={counts[(problem_type, l)]}" for l in _THREEWAY_LETTERS)
            total = sum(counts[(problem_type, l)] for l in _THREEWAY_LETTERS)
            print(f"  {problem_type}: {parts}  total={total}")
        else:
            a_count = counts[(problem_type, "A")]
            b_count = counts[(problem_type, "B")]
            total = a_count + b_count
            print(f"  {problem_type}: A={a_count} B={b_count} total={total}")


def build_group_key(
    record: dict[str, Any],
    balance_scope: str,
    target_problem_types: set[str],
) -> str | None:
    problem_type = record.get("problem_type")
    if problem_type not in target_problem_types:
        return None
    answer = record.get("answer")
    if not isinstance(answer, str):
        return None
    if problem_type in BINARY_PROBLEM_TYPES:
        if answer not in ("A", "B"):
            return None
        return "all" if balance_scope == "all" else str(problem_type)
    elif problem_type in THREEWAY_PROBLEM_TYPES:
        if answer.upper() not in _THREEWAY_LETTERS:
            return None
        return str(problem_type)  # 3-way always groups by type
    return None


def _rebalance_binary_group(
    output: list[dict[str, Any]],
    indices: list[int],
    rng: random.Random,
) -> int:
    answers = Counter(output[idx]["answer"] for idx in indices)
    diff = abs(answers["A"] - answers["B"])
    if diff <= 1:
        return 0
    majority_answer = "A" if answers["A"] > answers["B"] else "B"
    flips_needed = diff // 2
    candidate_indices = [idx for idx in indices if output[idx]["answer"] == majority_answer]
    rng.shuffle(candidate_indices)
    for idx in candidate_indices[:flips_needed]:
        output[idx] = flip_record(output[idx])
    return flips_needed


def _rebalance_3way_group(
    output: list[dict[str, Any]],
    indices: list[int],
    rng: random.Random,
) -> int:
    """Balance A/B/C distribution for 3-way MCQ samples by swapping slot assignments."""
    total = len(indices)
    if total < 3:
        return 0
    target = total // 3
    counts = Counter(output[idx]["answer"] for idx in indices)

    surplus: list[tuple[int, str]] = []  # (sample_idx, old_letter) — samples to reassign
    deficit: list[str] = []             # target letters to assign to

    for letter in _THREEWAY_LETTERS:
        cnt = counts.get(letter, 0)
        if cnt > target:
            over = [idx for idx in indices if output[idx]["answer"] == letter]
            rng.shuffle(over)
            surplus.extend((idx, letter) for idx in over[: cnt - target])
        elif cnt < target:
            deficit.extend([letter] * (target - cnt))

    flipped = 0
    for (sample_idx, _old), new_letter in zip(surplus, deficit):
        pt = output[sample_idx]["problem_type"]
        if pt == "aot_3way_v2t":
            output[sample_idx] = permute_3way_v2t_record(output[sample_idx], new_letter)
        else:
            output[sample_idx] = permute_3way_t2v_record(output[sample_idx], new_letter)
        flipped += 1
    return flipped


def rebalance_records(
    records: list[dict[str, Any]],
    target_problem_types: set[str],
    balance_scope: str,
    seed: int,
) -> tuple[list[dict[str, Any]], int]:
    rng = random.Random(seed)
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        group_key = build_group_key(record, balance_scope, target_problem_types)
        if group_key is not None:
            grouped_indices[group_key].append(idx)

    output = copy.deepcopy(records)
    flipped_count = 0

    for group_key in sorted(grouped_indices):
        indices = grouped_indices[group_key]
        sample_pt = output[indices[0]]["problem_type"] if indices else ""
        if sample_pt in THREEWAY_PROBLEM_TYPES:
            flipped_count += _rebalance_3way_group(output, indices, rng)
        else:
            flipped_count += _rebalance_binary_group(output, indices, rng)

    return output, flipped_count


def parse_problem_types(raw: str) -> set[str]:
    values = {item.strip() for item in raw.split(",") if item.strip()}
    invalid = values - set(SUPPORTED_PROBLEM_TYPES)
    if invalid:
        raise ValueError(f"Unsupported problem types: {sorted(invalid)}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebalance A/B answers in filtered Temporal AoT JSONL by swapping option order."
    )
    parser.add_argument("--input-jsonl", required=True, help="Filtered mixed JSONL input")
    parser.add_argument("--output-jsonl", required=True, help="Balanced JSONL output")
    parser.add_argument(
        "--problem-types",
        default="aot_t2v,aot_v2t,aot_3way_v2t,aot_3way_t2v",
        help="Comma-separated AoT problem types to rebalance",
    )
    parser.add_argument(
        "--balance-scope",
        choices=("problem_type", "all"),
        default="problem_type",
        help="Balance each problem type separately or all selected AoT samples together",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for picking which samples to flip")
    args = parser.parse_args()

    target_problem_types = parse_problem_types(args.problem_types)
    records = load_jsonl(args.input_jsonl)
    print_stats("Before rebalance:", records, target_problem_types)
    balanced_records, flipped_count = rebalance_records(
        records=records,
        target_problem_types=target_problem_types,
        balance_scope=args.balance_scope,
        seed=args.seed,
    )
    write_jsonl(args.output_jsonl, balanced_records)
    print(f"Flipped {flipped_count} records")
    print_stats("After rebalance:", balanced_records, target_problem_types)
    print(f"Wrote balanced dataset to {args.output_jsonl}")


if __name__ == "__main__":
    main()
