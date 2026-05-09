#!/usr/bin/env python3
"""Rewrite temporal-grounding JSONL prompts/answers to the TG-Bench style.

This is intended for existing TimeRFT/TGBench/base JSONL files whose videos and
metadata are already correct, but whose prompt or answer format needs to be
refreshed.

Usage from train/:
    python video_proxy/data/base_sources/tg/rewrite_format.py \
        --input video_proxy/data/base_sources/tg/data/tg_timerft_max256s_validated.jsonl \
        --output video_proxy/data/base_sources/tg/data/tg_timerft_max256s_validated_reprompt.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from video_proxy.data.base_sources.tg.common import (  # noqa: E402
    format_answer_text,
    load_jsonl,
    prompt_for,
    safe_pair,
    write_jsonl,
)


_TEXTUAL_SENTENCE_PATTERN = re.compile(
    r"textual\s+sentence:\s*['\"“](.*?)['\"”]\s*\.?\s*Please\s+return",
    re.IGNORECASE | re.DOTALL,
)
_SENTENCE_RANGE_PATTERN = re.compile(
    r"The\s+event(?:\s+['\"“].*?['\"”])?\s+happens\s+in\s+(?:the\s+)?"
    r"([0-9]*\.?[0-9]+)\s*-\s*([0-9]*\.?[0-9]+)\s*seconds?",
    re.IGNORECASE | re.DOTALL,
)
_GENERIC_SECONDS_RANGE_PATTERN = re.compile(
    r"([0-9]*\.?[0-9]+)\s*-\s*([0-9]*\.?[0-9]+)\s*seconds?",
    re.IGNORECASE,
)
_ANSWER_TAG_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_TO_RANGE_PATTERN = re.compile(r"([0-9]*\.?[0-9]+)\s+(?:to|and)\s+([0-9]*\.?[0-9]+)", re.IGNORECASE)
_SEGMENT_PATTERN = re.compile(r"\[\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\]")


def pair_from_match(match: re.Match[str] | None) -> tuple[float, float] | None:
    if match is None:
        return None
    try:
        start = float(match.group(1))
        end = float(match.group(2))
    except (TypeError, ValueError, IndexError):
        return None
    if start >= 0 and end > start:
        return start, end
    return None


def parse_span_from_text(text: str) -> tuple[float, float] | None:
    for pattern in (_SENTENCE_RANGE_PATTERN, _GENERIC_SECONDS_RANGE_PATTERN, _TO_RANGE_PATTERN, _SEGMENT_PATTERN):
        pair = pair_from_match(pattern.search(text))
        if pair is not None:
            return pair

    answer_match = _ANSWER_TAG_PATTERN.search(text)
    if answer_match is not None:
        return parse_span_from_text(answer_match.group(1))
    return None


def extract_sentence(record: dict[str, Any]) -> str | None:
    meta = record.get("metadata") or {}
    for key in ("query", "sentence", "caption", "description"):
        value = meta.get(key) or record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    prompt = str(record.get("prompt") or "")
    match = _TEXTUAL_SENTENCE_PATTERN.search(prompt)
    if match is not None:
        return " ".join(match.group(1).split())
    return None


def extract_timestamp(record: dict[str, Any]) -> tuple[float, float] | None:
    meta = record.get("metadata") or {}
    for value in (meta.get("timestamp"), meta.get("span"), record.get("timestamp")):
        pair = safe_pair(value)
        if pair is not None:
            return pair

    answer = str(record.get("answer") or record.get("ground_truth") or "")
    if answer:
        return parse_span_from_text(answer)
    return None


def build_prompt(sentence: str, mode: str) -> str:
    return prompt_for(sentence, mode)


def rewrite_record(record: dict[str, Any], mode: str) -> dict[str, Any]:
    sentence = extract_sentence(record)
    if not sentence:
        raise ValueError("missing sentence/query in metadata or prompt")
    timestamp = extract_timestamp(record)
    if timestamp is None:
        raise ValueError("missing valid timestamp or parseable answer")
    start, end = timestamp

    out = copy.deepcopy(record)
    prompt = build_prompt(sentence, mode)
    out["prompt"] = prompt
    out["messages"] = [{"role": "user", "content": prompt}]
    out["answer"] = format_answer_text(start, end)
    out["data_type"] = out.get("data_type") or "video"
    out["problem_type"] = out.get("problem_type") or "temporal_grounding"

    meta = dict(out.get("metadata") or {})
    meta["sentence"] = sentence
    meta.setdefault("query", sentence)
    meta["timestamp"] = [round(start, 2), round(end, 2)]
    meta["prompt_format"] = "tgbench_natural_language_v1"
    meta["answer_format"] = "tgbench_natural_language_v1"
    out["metadata"] = meta
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite TG JSONL prompt/answer into TG-Bench natural-language style")
    parser.add_argument("--input", required=True, help="Input TG JSONL")
    parser.add_argument("--output", required=True, help="Output rewritten TG JSONL")
    parser.add_argument("--mode", choices=["no_cot", "cot"], default="no_cot")
    parser.add_argument("--skip-bad", action="store_true", help="Skip rows that cannot be rewritten instead of failing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.is_file():
        raise SystemExit(f"Input not found: {input_path}")

    rows = load_jsonl(input_path)
    rewritten: list[dict[str, Any]] = []
    failures: list[tuple[int, str]] = []
    for idx, row in enumerate(rows):
        try:
            rewritten.append(rewrite_record(row, args.mode))
        except Exception as exc:
            if not args.skip_bad:
                raise SystemExit(f"Failed to rewrite row {idx} from {input_path}: {exc}") from exc
            failures.append((idx, str(exc)))

    write_jsonl(output_path, rewritten)
    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "input_records": len(rows),
        "rewritten_records": len(rewritten),
        "skipped_records": len(failures),
        "mode": args.mode,
        "prompt_format": "tgbench_natural_language_v1",
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("==========================================")
    print(" TG prompt rewrite done")
    print(f" Input:      {input_path} ({len(rows)} records)")
    print(f" Output:     {output_path} ({len(rewritten)} records)")
    print(f" Skipped:    {len(failures)}")
    print(" Format:     tgbench_natural_language_v1")
    print("==========================================")


if __name__ == "__main__":
    main()
