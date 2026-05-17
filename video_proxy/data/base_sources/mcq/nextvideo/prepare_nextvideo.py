#!/usr/bin/env python3
"""Convert NeXTVideo MCQ JSONL into the shared MCQ rollout format.

NeXTVideo ships records as chat-style messages plus a relative ``video.path``:

    {"messages": [...], "video": {"path": "./NExTVideo/1164/3238737531.mp4"}}

This adapter keeps source-specific parsing here and emits the same schema used
by the existing LLaVA MCQ rollout/filter/check pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from video_proxy.data.base_sources.mcq.prepare.convert_to_direct import (
    ensure_messages,
    rewrite_prompt,
)


_ANSWER_LETTER = re.compile(r"^[A-Za-z]$")
_OPTIONS_DIRECT_INSTRUCTION = "Answer with the option's letter from the given choices directly."


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Failed to parse {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    chunks: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text" and item.get("text") is not None:
            chunks.append(str(item.get("text")))
    return "\n".join(chunks).strip()


def _extract_prompt(row: dict[str, Any]) -> str:
    messages = row.get("messages") or []
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            text = _message_text(message.get("content"))
            if text:
                return text
    return ""


def _extract_answer(row: dict[str, Any]) -> str | None:
    answer = row.get("gt")
    if answer is None:
        messages = row.get("messages") or []
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            answer = _message_text(message.get("content"))
            if answer:
                break
    answer = str(answer or "").strip().upper()
    return answer if _ANSWER_LETTER.match(answer) else None


def _normalize_prompt(prompt: str) -> str:
    prompt = prompt.replace(_OPTIONS_DIRECT_INSTRUCTION, "Answer with the option letter.")
    prompt = prompt.replace(" Answer with the option letter.", "\n\nAnswer with the option letter.")
    prompt = prompt.strip()
    if not prompt.startswith("<video>"):
        prompt = "<video>\n" + prompt
    prompt, _changed = rewrite_prompt(prompt)
    return prompt


def _resolve_video_path(dataset_root: Path, raw_path: str) -> Path:
    normalized = raw_path.strip()
    if normalized.startswith("./"):
        normalized = normalized[2:]
    path = Path(normalized)
    if path.is_absolute():
        return path
    return dataset_root / path


def _video_id(raw_path: str) -> str:
    path = Path(raw_path.strip().lstrip("./"))
    if len(path.parts) >= 3:
        return f"{path.parts[-2]}-{path.stem}"
    return path.stem


def _data_source(raw_path: str) -> str:
    path = Path(raw_path.strip().lstrip("./"))
    if len(path.parts) >= 2:
        return f"nextvideo_{path.parts[1]}"
    return "nextvideo_unknown"


def convert_record(
    row: dict[str, Any],
    *,
    dataset_root: Path,
    split: str,
    line_no: int,
    verify_video: bool = False,
) -> dict[str, Any] | None:
    prompt = _extract_prompt(row)
    answer = _extract_answer(row)
    raw_video_path = str((row.get("video") or {}).get("path") or "")
    if not prompt or not answer or not raw_video_path:
        return None

    video_path = _resolve_video_path(dataset_root, raw_video_path)
    if verify_video and not video_path.is_file():
        return None

    prompt = _normalize_prompt(prompt)
    metadata = {
        "id": f"nextvideo_{split}_{line_no:06d}",
        "video_id": _video_id(raw_video_path),
        "source": "nextvideo",
        "dataset": "NeXTVideo",
        "split": split,
        "data_source": _data_source(raw_video_path),
        "raw_video_path": raw_video_path,
        "num_frames_hint": (row.get("video") or {}).get("num_frames"),
    }
    record = {
        "prompt": prompt,
        "answer": answer,
        "videos": [str(video_path)],
        "problem_type": "llava_mcq",
        "data_type": "video",
        "metadata": metadata,
    }
    ensure_messages(record, prompt)
    return record


def convert_file(
    *,
    input_path: Path,
    output_path: Path,
    dataset_root: Path,
    split: str,
    verify_videos: bool = False,
) -> dict[str, Any]:
    rows = load_jsonl(input_path)
    records: list[dict[str, Any]] = []
    skipped = 0
    for line_no, row in enumerate(rows, start=1):
        record = convert_record(
            row,
            dataset_root=dataset_root,
            split=split,
            line_no=line_no,
            verify_video=verify_videos,
        )
        if record is None:
            skipped += 1
            continue
        records.append(record)
    write_jsonl(output_path, records)

    answers = Counter(record["answer"] for record in records)
    data_sources = Counter(record["metadata"]["data_source"] for record in records)
    return {
        "input": str(input_path),
        "output": str(output_path),
        "dataset_root": str(dataset_root),
        "split": split,
        "total": len(rows),
        "converted": len(records),
        "skipped": skipped,
        "verify_videos": verify_videos,
        "answers": dict(answers),
        "data_sources_top20": dict(data_sources.most_common(20)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert NeXTVideo MCQ JSONL to shared MCQ format")
    parser.add_argument("--input", required=True, help="NeXTVideo train.jsonl or val.jsonl")
    parser.add_argument("--output", required=True, help="Output shared MCQ JSONL")
    parser.add_argument(
        "--dataset-root",
        default="",
        help="NeXTVideo root containing NExTVideo/ videos. Defaults to input parent.",
    )
    parser.add_argument("--split", default="", help="Split label for metadata ids. Defaults to input stem.")
    parser.add_argument("--verify-videos", action="store_true", help="Skip records whose video path is missing")
    parser.add_argument("--summary-json", default="", help="Optional conversion summary JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    dataset_root = Path(args.dataset_root) if args.dataset_root else input_path.parent
    split = args.split or input_path.stem
    summary = convert_file(
        input_path=input_path,
        output_path=Path(args.output),
        dataset_root=dataset_root,
        split=split,
        verify_videos=args.verify_videos,
    )
    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("==========================================")
    print(" NeXTVideo MCQ conversion")
    print(f" Input:     {summary['input']}")
    print(f" Output:    {summary['output']}")
    print(f" Converted: {summary['converted']}/{summary['total']} (skipped {summary['skipped']})")
    if args.summary_json:
        print(f" Summary:   {args.summary_json}")
    print("==========================================")


if __name__ == "__main__":
    main()
