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

from prompts import get_v2t_prompt


SUPPORTED_PROBLEM_TYPES = ("aot_t2v", "aot_v2t")


def get_t2v_prompt(
    caption: str,
    option_a_text: str = "The first segment",
    option_b_text: str = "The second segment",
) -> str:
    return (
        "The input video contains two segments separated by a black screen.\n"
        "<video>\n\n"
        f'Which segment best matches the caption "{caption}"?\n'
        f"Options:\nA. {option_a_text}\nB. {option_b_text}\n\n"
        "First, carefully observe both segments and use the black screen as the boundary between them. "
        "Reason about the visible action order in the first segment and in the second segment, "
        "then compare them with the caption to decide which segment matches better.\n\n"
        "Think step by step inside <think> </think> tags, then provide your final answer "
        "(a single letter A or B) inside <answer> </answer> tags."
    )


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
        forward_caption = metadata.get("forward_caption")
        reverse_caption = metadata.get("reverse_caption")
        if not isinstance(forward_caption, str) or not isinstance(reverse_caption, str):
            raise ValueError("aot_v2t record is missing forward_caption/reverse_caption")
        sync_prompt(flipped, get_v2t_prompt(reverse_caption, forward_caption))
        metadata["option_a_caption"] = reverse_caption
        metadata["option_b_caption"] = forward_caption
    elif problem_type == "aot_t2v":
        caption = metadata.get("caption")
        if not isinstance(caption, str):
            raise ValueError("aot_t2v record is missing caption")
        sync_prompt(
            flipped,
            get_t2v_prompt(
                caption=caption,
                option_a_text="The second segment",
                option_b_text="The first segment",
            ),
        )
        metadata["option_a_segment"] = "second"
        metadata["option_b_segment"] = "first"
    else:
        raise ValueError(f"Unsupported problem_type for flipping: {problem_type!r}")

    flipped["answer"] = flip_answer(record["answer"])
    metadata["answer_rebalanced"] = True
    metadata["original_answer_before_rebalance"] = record["answer"]
    return flipped


def count_answers(records: list[dict[str, Any]], target_problem_types: set[str]) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for record in records:
        problem_type = record.get("problem_type")
        answer = record.get("answer")
        if problem_type in target_problem_types and answer in ("A", "B"):
            counts[(problem_type, answer)] += 1
    return counts


def print_stats(title: str, records: list[dict[str, Any]], target_problem_types: set[str]) -> None:
    counts = count_answers(records, target_problem_types)
    total_a = sum(counts[(pt, "A")] for pt in target_problem_types)
    total_b = sum(counts[(pt, "B")] for pt in target_problem_types)
    print(title)
    print(f"  overall: A={total_a} B={total_b} total={total_a + total_b}")
    for problem_type in sorted(target_problem_types):
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
    if record.get("answer") not in ("A", "B"):
        return None
    if balance_scope == "all":
        return "all"
    return str(problem_type)


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
        answers = Counter(output[idx]["answer"] for idx in indices)
        diff = abs(answers["A"] - answers["B"])
        if diff <= 1:
            continue

        majority_answer = "A" if answers["A"] > answers["B"] else "B"
        flips_needed = diff // 2
        candidate_indices = [idx for idx in indices if output[idx]["answer"] == majority_answer]
        rng.shuffle(candidate_indices)

        for idx in candidate_indices[:flips_needed]:
            output[idx] = flip_record(output[idx])
            flipped_count += 1

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
        default="aot_t2v,aot_v2t",
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
