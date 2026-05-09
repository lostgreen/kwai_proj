#!/usr/bin/env python3
"""Convert EasyR1-style JSONL prompts from direct/no-CoT to CoT instructions.

The converter only rewrites ``prompt`` and the user message content in
``messages``. Answers, videos, metadata, and task labels are preserved.

Examples:
    python video_proxy/data/scripts/convert_jsonl_to_cot.py input.jsonl output.jsonl
    python video_proxy/data/scripts/convert_jsonl_to_cot.py input.jsonl output_thought.jsonl --reasoning-tag thought
    python video_proxy/data/scripts/convert_jsonl_to_cot.py input.jsonl --in-place
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


TG_POST_PROMPT = "Please return its start time and end time in seconds."
SUPPORTED_REASONING_TAGS = ("think", "thought")

CHOICE_COT_SUFFIX = "inside <answer></answer> tags."

_HAS_COT_PATTERN = re.compile(
    r"<(?:think|thought)\s*>\s*</(?:think|thought)\s*>|"
    r"<(?:think|thought)>|"
    r"think\s+step\s+by\s+step|first,?\s+think",
    re.IGNORECASE,
)
_ANSWER_TAG_INSTRUCTION = re.compile(
    r"Provide your answer\s*(?P<detail>\([^.\n]*?\))?\s*inside\s*<answer>\s*</answer>\s*tags\.?",
    re.IGNORECASE | re.DOTALL,
)
_FINAL_ANSWER_TAG_INSTRUCTION = re.compile(
    r"Provide your final answer\s*(?P<detail>\([^.\n]*?\))?\s*inside\s*<answer>\s*</answer>\s*tags\.?",
    re.IGNORECASE | re.DOTALL,
)
_ANSWER_SEQUENCE_TAG_INSTRUCTION = re.compile(
    r"Provide your answer\s+(?P<detail>as .*?)\s*inside\s*<answer>\s*</answer>\s*tags\.?",
    re.IGNORECASE | re.DOTALL,
)
_OLD_MCQ_DIRECT = "Answer with the option letter."
_EVENTS_OUTPUT_PATTERN = re.compile(
    r"(?=Output the start and end time .*?<events>)",
    re.IGNORECASE | re.DOTALL,
)
_STRICT_EVENTS_OUTPUT = (
    "Output format (strictly follow this):\n"
    "<events>\n"
    "[start_time, end_time]\n"
    "</events>\n\n"
    "Where start_time and end_time are in seconds "
    "(precise to one decimal place, e.g., [12.5, 17.8])."
)

@dataclass
class ConversionSummary:
    total: int = 0
    converted: int = 0
    unchanged: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    def add(self, changed: bool, reason: str) -> None:
        self.total += 1
        if changed:
            self.converted += 1
        else:
            self.unchanged += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1


def _has_cot(prompt: str) -> bool:
    return bool(_HAS_COT_PATTERN.search(prompt))


def _normalize_answer_tag_spacing(text: str) -> str:
    return re.sub(r"<answer>\s*</answer>", "<answer></answer>", text)


def _reasoning_token(reasoning_tag: str) -> str:
    if reasoning_tag not in SUPPORTED_REASONING_TAGS:
        raise ValueError(
            f"reasoning_tag must be one of {SUPPORTED_REASONING_TAGS}, got {reasoning_tag!r}"
        )
    return f"<{reasoning_tag}></{reasoning_tag}>"


def _choice_prefix(reasoning_tag: str) -> str:
    return (
        f"Think step by step inside {_reasoning_token(reasoning_tag)} tags, "
        "then provide your final answer"
    )


def _events_cot_instruction(reasoning_tag: str) -> str:
    return (
        f"First, think step by step inside {_reasoning_token(reasoning_tag)} tags. "
        "Use the visual evidence and timestamps to decide the temporal boundaries.\n\n"
    )


def _strict_events_cot(reasoning_tag: str) -> str:
    return (
        f"First, think step by step inside {_reasoning_token(reasoning_tag)} tags. "
        "Describe what happens at different time periods in the video "
        "and determine when the target event occurs.\n\n"
        "Then, provide the precise time period in the following format:\n"
        "<events>\n"
        "[start_time, end_time]\n"
        "</events>\n\n"
        "Where start_time and end_time are in seconds "
        "(precise to one decimal place, e.g., [12.5, 17.8])."
    )


def _tg_cot_suffix(reasoning_tag: str) -> str:
    return (
        f" First, think step by step inside {_reasoning_token(reasoning_tag)}, "
        "then give the final sentence only."
    )


def _choice_replacement(match: re.Match[str], reasoning_tag: str) -> str:
    detail = (match.group("detail") or "").strip()
    detail_text = f" {detail}" if detail else ""
    return f"{_choice_prefix(reasoning_tag)}{detail_text} {CHOICE_COT_SUFFIX}"


def _rewrite_choice_prompt(prompt: str, reasoning_tag: str) -> tuple[str, bool]:
    new_prompt, n = _ANSWER_TAG_INSTRUCTION.subn(
        lambda match: _choice_replacement(match, reasoning_tag),
        prompt,
    )
    if n > 0:
        return _normalize_answer_tag_spacing(new_prompt).rstrip(), True

    new_prompt, n = _FINAL_ANSWER_TAG_INSTRUCTION.subn(
        lambda match: _choice_replacement(match, reasoning_tag),
        prompt,
    )
    if n > 0:
        return _normalize_answer_tag_spacing(new_prompt).rstrip(), True

    new_prompt, n = _ANSWER_SEQUENCE_TAG_INSTRUCTION.subn(
        lambda match: _choice_replacement(match, reasoning_tag),
        prompt,
    )
    if n > 0:
        return _normalize_answer_tag_spacing(new_prompt).rstrip(), True

    if _OLD_MCQ_DIRECT in prompt:
        new_prompt = prompt.replace(
            _OLD_MCQ_DIRECT,
            f"{_choice_prefix(reasoning_tag)} (a single letter) {CHOICE_COT_SUFFIX}",
        )
        return new_prompt.rstrip(), True

    return prompt, False


def _rewrite_events_prompt(prompt: str, reasoning_tag: str) -> tuple[str, bool]:
    if _STRICT_EVENTS_OUTPUT in prompt:
        return prompt.replace(_STRICT_EVENTS_OUTPUT, _strict_events_cot(reasoning_tag)).rstrip(), True

    match = _EVENTS_OUTPUT_PATTERN.search(prompt)
    if match is not None:
        idx = match.start()
        new_prompt = prompt[:idx].rstrip() + "\n\n" + _events_cot_instruction(reasoning_tag) + prompt[idx:]
        return new_prompt.rstrip(), True

    if "<events>" in prompt and "Output" in prompt:
        return (prompt.rstrip() + "\n\n" + _events_cot_instruction(reasoning_tag).rstrip()).rstrip(), True

    return prompt, False


def _rewrite_tg_natural_prompt(prompt: str, reasoning_tag: str) -> tuple[str, bool]:
    if TG_POST_PROMPT not in prompt:
        return prompt, False
    return prompt.replace(TG_POST_PROMPT, TG_POST_PROMPT + _tg_cot_suffix(reasoning_tag)).rstrip(), True


def _sync_messages(record: dict[str, Any], prompt: str) -> None:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        record["messages"] = [{"role": "user", "content": prompt}]
        return

    last_user_idx: int | None = None
    for idx, message in enumerate(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            last_user_idx = idx

    if last_user_idx is None:
        messages.append({"role": "user", "content": prompt})
    else:
        messages[last_user_idx]["content"] = prompt


def convert_record(record: dict[str, Any], reasoning_tag: str = "think") -> tuple[dict[str, Any], bool, str]:
    _reasoning_token(reasoning_tag)
    prompt = str(record.get("prompt") or "")
    if not prompt:
        return copy.deepcopy(record), False, "missing_prompt"

    if _has_cot(prompt):
        return copy.deepcopy(record), False, "already_cot"

    for reason, rewriter in (
        ("tg_natural", _rewrite_tg_natural_prompt),
        ("answer_tag", _rewrite_choice_prompt),
        ("events", _rewrite_events_prompt),
    ):
        new_prompt, changed = rewriter(prompt, reasoning_tag)
        if changed:
            out = copy.deepcopy(record)
            out["prompt"] = new_prompt
            _sync_messages(out, new_prompt)
            return out, True, reason

    return copy.deepcopy(record), False, "unsupported"


def convert_jsonl(input_path: Path, output_path: Path, reasoning_tag: str = "think") -> ConversionSummary:
    _reasoning_token(reasoning_tag)
    summary = ConversionSummary()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open(encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {input_path}:{line_no}: {exc}") from exc
            converted, changed, reason = convert_record(record, reasoning_tag=reasoning_tag)
            summary.add(changed, reason)
            fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert direct/no-CoT JSONL prompts to CoT prompts")
    parser.add_argument("input", help="Input JSONL path")
    parser.add_argument("output", nargs="?", help="Output JSONL path")
    parser.add_argument("--in-place", action="store_true", help="Overwrite the input file safely")
    parser.add_argument(
        "--reasoning-tag",
        choices=SUPPORTED_REASONING_TAGS,
        default="think",
        help="Reasoning tag to request in converted prompts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.is_file():
        raise SystemExit(f"Input not found: {input_path}")

    if args.in_place:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".jsonl",
            dir=input_path.parent,
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
        summary = convert_jsonl(input_path, tmp_path, reasoning_tag=args.reasoning_tag)
        tmp_path.replace(input_path)
        output_path = input_path
    else:
        if not args.output:
            raise SystemExit("Output path is required unless --in-place is set")
        output_path = Path(args.output)
        summary = convert_jsonl(input_path, output_path, reasoning_tag=args.reasoning_tag)

    print(
        f"Done: {summary.total} total, {summary.converted} converted, "
        f"{summary.unchanged} unchanged -> {output_path}"
    )
    if summary.reasons:
        reason_text = ", ".join(f"{key}={value}" for key, value in sorted(summary.reasons.items()))
        print(f"Reasons: {reason_text}")


if __name__ == "__main__":
    main()
