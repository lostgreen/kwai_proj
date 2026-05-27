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
_ONLY_ONE_LETTER_INSTRUCTION = re.compile(
    r"Answer with only one letter:\s*(?P<letters>[^\n.]+)\.?",
    re.IGNORECASE,
)
_OLD_MCQ_DIRECT = "Answer with the option letter."
_EVENTS_OUTPUT_PATTERN = re.compile(
    r"(?=Output the start and end time .*?<events>)",
    re.IGNORECASE | re.DOTALL,
)
_EVENTS_OUTPUT_FORMAT_PATTERN = re.compile(
    r"(?=Output format\s*\(strictly follow this\):\s*<events>)",
    re.IGNORECASE | re.DOTALL,
)
_OUTPUT_ONLY_TIMESTAMPS_RULE = "- Output only timestamps, no descriptions."
_EVENTS_EXAMPLE_PATTERN = re.compile(
    r"Example:\s*<events>(?P<events>.*?)</events>",
    re.IGNORECASE | re.DOTALL,
)
_LEGACY_EVENTS_COT_INSTRUCTION_PATTERN = re.compile(
    r"First,?\s+think\s+step\s+by\s+step\s+inside\s+"
    r"<(?P<tag>think|thought)></(?P=tag)>\s+tags\.\s*"
    r"Use\s+the\s+visual\s+evidence\s+and\s+timestamps\s+to\s+decide\s+"
    r"the\s+temporal\s+boundaries\.\s*",
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
_ANSWER_EVENTS_OUTPUT = "<answer>[[start_time, end_time]]</answer>"

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


def _events_cot_instruction(reasoning_tag: str, problem_type: str = "") -> str:
    l3_lines = ""
    if problem_type == "temporal_seg_hier_L3_seg":
        l3_lines = (
            "- L3 policy: For L3, preserve visually distinct shot boundaries first; "
            "long single shots may need additional state/action splits.\n"
            "- L3 caution: Do not collapse multiple clear shots into one broad segment.\n"
        )
    return (
        f"First, think step by step inside {_reasoning_token(reasoning_tag)} tags. "
        "Use concise visual evidence and timestamps to decide the temporal boundaries.\n"
        "Inside the reasoning tags, include:\n"
        "- Shots: sustained visual shot or scene anchors with approximate timestamps.\n"
        "- Decisions: KEEP/MERGE/SPLIT choices that turn shots into the final segments.\n"
        f"{l3_lines}"
        "- Partition check: verify the final segments are chronological, non-overlapping, "
        "gap-free when the task requires full coverage, and cover the requested timeline.\n\n"
        "Then output the final timestamps after the closing reasoning tag inside "
        "<answer></answer> tags.\n\n"
    )


def _strict_events_cot(reasoning_tag: str) -> str:
    return (
        f"First, think step by step inside {_reasoning_token(reasoning_tag)} tags. "
        "Describe what happens at different time periods in the video "
        "and determine when the target event occurs.\n\n"
        "Then, provide the precise time period in the following format:\n"
        f"{_ANSWER_EVENTS_OUTPUT}\n\n"
        "Where start_time and end_time are in seconds "
        "(precise to one decimal place, e.g., [12.5, 17.8])."
    )


def _tg_cot_suffix(reasoning_tag: str) -> str:
    return (
        f" First, think step by step inside {_reasoning_token(reasoning_tag)}, "
        "then give the final sentence only inside <answer></answer> tags."
    )


def _events_cot_example(reasoning_tag: str, events_block: str) -> str:
    return (
        "Example:\n"
        f"<{reasoning_tag}>\n"
        "The video has five shots: [0,8] oil is poured into a pan, [8,14] white seeds are added, "
        "[14,22] dark seeds are added in the same pan, [22,34] green leaves are added followed by "
        "chopped chilies, and [34,42] the ingredients are stirred. Shots [0,8], [8,14], and [14,22] "
        "are merged into [0,22] because they are consecutive views of one seasoning-base task. "
        "The shot [22,34] is split into [22,28] and [28,34] because adding leaves and adding chilies "
        "are distinct actions inside one shot. The shot [34,42] is kept as [34,42] because it is one "
        "coherent stirring task. The final events are chronological, adjacent, non-overlapping, "
        "and cover the full 0-42s clip.\n"
        f"</{reasoning_tag}>\n"
        "<answer>[[0, 22], [22, 28], [28, 34], [34, 42]]</answer>"
    )


def _rewrite_events_examples(prompt: str, reasoning_tag: str) -> tuple[str, bool]:
    new_prompt, count = _EVENTS_EXAMPLE_PATTERN.subn(
        lambda match: _events_cot_example(reasoning_tag, match.group("events")),
        prompt,
    )
    return new_prompt, count > 0


def _upgrade_existing_events_cot_instruction(prompt: str, problem_type: str = "") -> tuple[str, bool]:
    def replace(match: re.Match[str]) -> str:
        tag = match.group("tag").lower()
        return _events_cot_instruction(tag, problem_type=problem_type)

    new_prompt, count = _LEGACY_EVENTS_COT_INSTRUCTION_PATTERN.subn(replace, prompt)
    return new_prompt, count > 0


def _choice_replacement(match: re.Match[str], reasoning_tag: str) -> str:
    detail = (match.group("detail") or "").strip()
    detail_text = f" {detail}" if detail else ""
    return f"{_choice_prefix(reasoning_tag)}{detail_text} {CHOICE_COT_SUFFIX}"


def _only_one_letter_replacement(match: re.Match[str], reasoning_tag: str) -> str:
    raw_letters = (match.group("letters") or "").strip()
    letters = re.findall(r"\b[A-Z]\b", raw_letters.upper())
    detail = ", ".join(letters[:-1]) + f", or {letters[-1]}" if len(letters) > 2 else raw_letters
    return f"{_choice_prefix(reasoning_tag)} (a single letter: {detail}) {CHOICE_COT_SUFFIX}"


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

    new_prompt, n = _ONLY_ONE_LETTER_INSTRUCTION.subn(
        lambda match: _only_one_letter_replacement(match, reasoning_tag),
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


def _rewrite_events_prompt(prompt: str, reasoning_tag: str, problem_type: str = "") -> tuple[str, bool]:
    prompt, example_changed = _rewrite_events_examples(prompt, reasoning_tag)

    if _STRICT_EVENTS_OUTPUT in prompt:
        return prompt.replace(_STRICT_EVENTS_OUTPUT, _strict_events_cot(reasoning_tag)).rstrip(), True

    prompt = prompt.replace(
        _OUTPUT_ONLY_TIMESTAMPS_RULE,
        "- In the final <answer> block, output only timestamps, no descriptions.",
    )

    match = _EVENTS_OUTPUT_PATTERN.search(prompt) or _EVENTS_OUTPUT_FORMAT_PATTERN.search(prompt)
    if match is not None:
        idx = match.start()
        new_prompt = (
            prompt[:idx].rstrip()
            + "\n\n"
            + _events_cot_instruction(reasoning_tag, problem_type=problem_type)
            + prompt[idx:]
        )
        return new_prompt.rstrip(), True

    if "<events>" in prompt and "Output" in prompt:
        return (
            prompt.rstrip()
            + "\n\n"
            + _events_cot_instruction(reasoning_tag, problem_type=problem_type).rstrip()
        ).rstrip(), True

    return prompt.rstrip(), example_changed


def _normalize_existing_cot_events_prompt(prompt: str, problem_type: str = "") -> tuple[str, bool]:
    if "<events>" not in prompt:
        return prompt, False

    new_prompt, instruction_changed = _upgrade_existing_events_cot_instruction(
        prompt,
        problem_type=problem_type,
    )
    tag_match = re.search(r"<(think|thought)></\1>", new_prompt, re.IGNORECASE)
    reasoning_tag = tag_match.group(1).lower() if tag_match else "think"
    new_prompt, example_changed = _rewrite_events_examples(new_prompt, reasoning_tag)
    return new_prompt.rstrip(), instruction_changed or example_changed


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
    problem_type = str(record.get("problem_type") or "")
    if not prompt:
        return copy.deepcopy(record), False, "missing_prompt"

    if _has_cot(prompt):
        new_prompt, changed = _normalize_existing_cot_events_prompt(prompt, problem_type=problem_type)
        if changed:
            out = copy.deepcopy(record)
            out["prompt"] = new_prompt
            _sync_messages(out, new_prompt)
            return out, True, "already_cot_events"
        return copy.deepcopy(record), False, "already_cot"

    for reason, rewriter in (
        ("tg_natural", lambda text, tag: _rewrite_tg_natural_prompt(text, tag)),
        ("answer_tag", lambda text, tag: _rewrite_choice_prompt(text, tag)),
        ("events", lambda text, tag: _rewrite_events_prompt(text, tag, problem_type=problem_type)),
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


def collect_prompt_samples(input_path: Path, per_type: int = 2) -> dict[str, list[str]]:
    if per_type <= 0:
        return {}

    samples: dict[str, list[str]] = {}
    with input_path.open(encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {input_path}:{line_no}: {exc}") from exc

            problem_type = str(record.get("problem_type") or "unknown")
            prompt = str(record.get("prompt") or "")
            bucket = samples.setdefault(problem_type, [])
            if len(bucket) < per_type:
                bucket.append(prompt)

    return {problem_type: samples[problem_type] for problem_type in sorted(samples)}


def print_prompt_samples(
    samples: dict[str, list[str]],
    *,
    max_chars: int = 1600,
) -> None:
    if not samples:
        return

    print("Prompt samples by problem_type:")
    for problem_type, prompts in samples.items():
        print(f"\n## {problem_type}")
        for idx, prompt in enumerate(prompts, start=1):
            shown = prompt
            if max_chars > 0 and len(shown) > max_chars:
                shown = shown[:max_chars].rstrip() + "\n...<truncated>"
            print(f"[{idx}] {shown}")


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
    parser.add_argument(
        "--sample-prompts",
        type=int,
        default=0,
        metavar="N",
        help="After conversion, print up to N prompt fields per problem_type from the output JSONL",
    )
    parser.add_argument(
        "--sample-prompt-max-chars",
        type=int,
        default=1600,
        help="Maximum characters to print per sampled prompt; use 0 for no truncation",
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

    if args.sample_prompts > 0:
        print_prompt_samples(
            collect_prompt_samples(output_path, per_type=args.sample_prompts),
            max_chars=args.sample_prompt_max_chars,
        )


if __name__ == "__main__":
    main()
