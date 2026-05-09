#!/usr/bin/env python3
"""Check whether saved rollout responses obey a tagged CoT budget."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RolloutCoTSummary:
    total: int = 0
    started: int = 0
    closed: int = 0
    missing_start: int = 0
    missing_end: int = 0
    over_budget: int = 0
    max_observed_tokens: int = 0


def _count_text_tokens(text: str) -> int:
    return len(re.findall(r"\S+", text.strip()))


def _count_cot_tokens(text: str, tokenizer: Any | None = None) -> int:
    if tokenizer is None:
        return _count_text_tokens(text)
    return len(tokenizer.encode(text, add_special_tokens=False))


def _cot_span(response: str, start_token: str, end_token: str) -> tuple[str | None, bool, bool]:
    start_idx = response.find(start_token)
    if start_idx < 0:
        return None, False, False

    content_start = start_idx + len(start_token)
    end_idx = response.find(end_token, content_start)
    if end_idx < 0:
        return response[content_start:], True, False

    return response[content_start:end_idx], True, True


def analyze_rollout_file(
    path: Path,
    *,
    start_token: str = "<think>",
    end_token: str = "</think>",
    max_tokens: int = 0,
    tokenizer: Any | None = None,
) -> RolloutCoTSummary:
    summary = RolloutCoTSummary()
    with path.open(encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc

            summary.total += 1
            response = str(record.get("response") or "")
            span, has_start, has_end = _cot_span(response, start_token, end_token)
            if not has_start:
                summary.missing_start += 1
                continue

            summary.started += 1
            if has_end:
                summary.closed += 1
            else:
                summary.missing_end += 1

            token_count = _count_cot_tokens(span or "", tokenizer)
            summary.max_observed_tokens = max(summary.max_observed_tokens, token_count)
            if max_tokens > 0 and token_count > max_tokens:
                summary.over_budget += 1

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", help="Saved rollout JSONL, e.g. rollouts/step_000001.jsonl")
    parser.add_argument("--start-token", default="<think>", help="CoT start tag")
    parser.add_argument("--end-token", default="</think>", help="CoT end tag")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="Expected text-token budget for the CoT span; 0 disables over-budget checks",
    )
    parser.add_argument(
        "--tokenizer",
        default="",
        help="Optional tokenizer/model path for matching the training CoT budget token count",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--require-start",
        action="store_true",
        help="Exit nonzero if any response does not contain the CoT start tag",
    )
    parser.add_argument(
        "--require-closed",
        action="store_true",
        help="Exit nonzero if any started CoT span lacks the end tag",
    )
    parser.add_argument(
        "--require-within-budget",
        action="store_true",
        help="Exit nonzero if any CoT span exceeds --max-tokens",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.jsonl)
    if not path.is_file():
        raise SystemExit(f"Rollout JSONL not found: {path}")

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer,
            trust_remote_code=args.trust_remote_code,
            use_fast=True,
        )

    summary = analyze_rollout_file(
        path,
        start_token=args.start_token,
        end_token=args.end_token,
        max_tokens=args.max_tokens,
        tokenizer=tokenizer,
    )
    print(json.dumps(summary.__dict__, ensure_ascii=False, indent=2, sort_keys=True))

    failures = []
    if args.require_start and summary.missing_start:
        failures.append(f"{summary.missing_start} responses missing {args.start_token}")
    if args.require_closed and summary.missing_end:
        failures.append(f"{summary.missing_end} CoT spans missing {args.end_token}")
    if args.require_within_budget and summary.over_budget:
        failures.append(f"{summary.over_budget} CoT spans over budget {args.max_tokens}")

    if failures:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
