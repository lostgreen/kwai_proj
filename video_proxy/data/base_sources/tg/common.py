"""Shared temporal-grounding JSONL and prompt helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PRE_PROMPT = (
    "Please find the visual event described by a sentence in the video, "
    "determining its starting and ending times. "
    "The format should be: 'The event happens in the start time - end time seconds'. "
    "For example: The event 'person turn a light on' happens in the 24.3 - 30.4 seconds. "
    "Now I will give you the textual sentence: "
)

POST_PROMPT = "Please return its start time and end time in seconds."

PROMPT_TEMPLATE_NO_COT = (
    "<video>\n"
    + PRE_PROMPT
    + '"{sentence}". '
    + POST_PROMPT
)

PROMPT_TEMPLATE_COT = (
    "<video>\n"
    + PRE_PROMPT
    + '"{sentence}". '
    + POST_PROMPT
    + " First reason step by step in <think></think>, then give the final sentence only."
)


def round2(value: float) -> float:
    """Round timestamps to the precision used in TG source JSONL files."""
    return round(value, 2)


def format_seconds(value: float) -> str:
    """Format seconds in the compact TG-Bench natural-language style."""
    return f"{round2(value):.2f}".rstrip("0").rstrip(".")


def format_answer_text(start: float, end: float) -> str:
    """Build the TG-Bench natural-language answer string."""
    return f"The event happens in the {format_seconds(start)} - {format_seconds(end)} seconds."


def prompt_for(sentence: str, mode: str = "no_cot") -> str:
    """Build a TG prompt for a sentence."""
    template = PROMPT_TEMPLATE_COT if mode == "cot" else PROMPT_TEMPLATE_NO_COT
    return template.format(sentence=sentence)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL file with line-numbered parse errors."""
    jsonl_path = Path(path)
    rows: list[dict[str, Any]] = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Failed to parse {jsonl_path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write records to JSONL, creating the parent directory."""
    jsonl_path = Path(path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_pair(value: Any) -> tuple[float, float] | None:
    """Parse a positive [start, end] pair."""
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            start = float(value[0])
            end = float(value[1])
        except (TypeError, ValueError):
            return None
        if start >= 0 and end > start:
            return start, end
    return None
