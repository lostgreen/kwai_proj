#!/usr/bin/env python3
"""Find validation samples whose CoT changes across rollout checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


VAL_STEP_RE = re.compile(r"val_step_(\d+)\.jsonl$")
THOUGHT_RE = re.compile(r"<thought>(.*?)(?:</thought>|<answer>|<events>|$)", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
EVENTS_RE = re.compile(r"<events>(.*?)</events>", re.DOTALL)


@dataclass(frozen=True)
class ValAttempt:
    step: int
    reward: float | None
    response: str
    thought: str
    answer: str
    cot_repaired: bool | None
    final_token_len: float | None


@dataclass(frozen=True)
class ValCotCase:
    label: str
    uid: str
    display_id: str
    problem_type: str
    prompt: str
    ground_truth: str
    attempts: list[ValAttempt]
    reward_min: float | None
    reward_max: float | None
    cot_changed: bool
    answer_changed: bool


def as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def extract_thought(response: str) -> str:
    match = THOUGHT_RE.search(response or "")
    if not match:
        return ""
    return normalize_text(match.group(1))


def extract_answer(response: str) -> str:
    response = response or ""
    match = ANSWER_RE.search(response)
    if match:
        return normalize_text(match.group(1))
    match = EVENTS_RE.search(response)
    if match:
        return f"<events>{normalize_text(match.group(1))}</events>"
    return ""


def val_step(path: Path) -> int | None:
    match = VAL_STEP_RE.match(path.name)
    return int(match.group(1)) if match else None


def val_rollout_files(exp_dir: Path, max_step: int | None = None) -> list[tuple[int, Path]]:
    rollout_dir = exp_dir / "rollouts" if (exp_dir / "rollouts").is_dir() else exp_dir
    files: list[tuple[int, Path]] = []
    for path in rollout_dir.glob("val_step_*.jsonl"):
        step = val_step(path)
        if step is None:
            continue
        if max_step is not None and step > max_step:
            continue
        files.append((step, path))
    return sorted(files)


def metadata_display_id(row: dict[str, Any]) -> str:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        return ""
    for key in ("uid", "record_id", "clip_key", "video_id", "id"):
        value = metadata.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def stable_question_id(row: dict[str, Any]) -> tuple[str, str]:
    problem_id = row.get("problem_id")
    if problem_id not in (None, ""):
        problem_id_text = str(problem_id)
        return problem_id_text, problem_id_text

    uid = row.get("uid")
    if uid not in (None, "") and not str(uid).startswith("val-"):
        uid_text = str(uid)
        return uid_text, uid_text

    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    fingerprint_obj = {
        "problem_type": row.get("problem_type") or row.get("task_type") or "",
        "prompt": row.get("prompt") or "",
        "ground_truth": row.get("ground_truth") or row.get("gt") or "",
        "metadata": {
            key: metadata.get(key)
            for key in (
                "clip_key",
                "record_id",
                "video_id",
                "parent_event_id",
                "event_id",
                "missing_id",
                "correct_text",
                "query",
                "sentence",
                "ordered_ids",
                "before_ids",
                "after_ids",
                "source",
                "data_source",
            )
            if metadata.get(key) not in (None, "")
        },
    }
    raw = json.dumps(fingerprint_obj, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"fingerprint:{digest}", metadata_display_id(row) or f"fingerprint:{digest}"


def read_val_attempts(exp_dir: Path, max_step: int | None = None) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for step, path in val_rollout_files(exp_dir, max_step=max_step):
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                uid, display_id = stable_question_id(row)
                response = str(row.get("response") or "")
                debug = row.get("cot_budget_debug")
                debug = debug if isinstance(debug, dict) else {}
                if uid not in groups:
                    groups[uid] = {
                        "uid": uid,
                        "display_id": display_id,
                        "problem_type": str(row.get("problem_type") or row.get("task_type") or "unknown"),
                        "prompt": str(row.get("prompt") or ""),
                        "ground_truth": str(row.get("ground_truth") or row.get("gt") or ""),
                        "attempts": [],
                    }
                groups[uid]["attempts"].append(
                    ValAttempt(
                        step=step,
                        reward=as_float(row.get("reward")),
                        response=response,
                        thought=extract_thought(response),
                        answer=extract_answer(response),
                        cot_repaired=debug.get("cot_repaired") if "cot_repaired" in debug else None,
                        final_token_len=as_float(debug.get("final_token_len")),
                    )
                )
    return groups


def has_meaningful_change(attempts: list[ValAttempt]) -> tuple[bool, bool]:
    thoughts = {attempt.thought for attempt in attempts if attempt.thought}
    answers = {attempt.answer for attempt in attempts if attempt.answer}
    return len(thoughts) > 1, len(answers) > 1


def collect_val_cot_change_cases(
    exp_dir: Path,
    *,
    label: str | None = None,
    max_step: int | None = None,
    min_attempts: int = 2,
    task_filter: str | None = None,
) -> list[ValCotCase]:
    label = label or exp_dir.name
    cases: list[ValCotCase] = []
    for group in read_val_attempts(exp_dir, max_step=max_step).values():
        attempts = sorted(group["attempts"], key=lambda item: item.step)
        if task_filter and task_filter.lower() not in group["problem_type"].lower():
            continue
        if len(attempts) < min_attempts:
            continue
        cot_changed, answer_changed = has_meaningful_change(attempts)
        if not cot_changed:
            continue
        rewards = [attempt.reward for attempt in attempts if attempt.reward is not None]
        cases.append(
            ValCotCase(
                label=label,
                uid=group["uid"],
                display_id=group["display_id"],
                problem_type=group["problem_type"],
                prompt=group["prompt"],
                ground_truth=group["ground_truth"],
                attempts=attempts,
                reward_min=min(rewards) if rewards else None,
                reward_max=max(rewards) if rewards else None,
                cot_changed=cot_changed,
                answer_changed=answer_changed,
            )
        )
    return sorted(
        cases,
        key=lambda case: (
            -(case.reward_max - case.reward_min if case.reward_max is not None and case.reward_min is not None else 0.0),
            case.problem_type,
            case.uid,
        ),
    )


def parse_exp_arg(raw: str) -> tuple[str | None, Path]:
    if "=" in raw:
        label, path = raw.split("=", 1)
        return label, Path(path)
    return None, Path(raw)


def format_float(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def truncate(text: str, limit: int) -> str:
    text = normalize_text(text)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def case_to_json(case: ValCotCase) -> dict[str, Any]:
    return {
        "label": case.label,
        "uid": case.uid,
        "display_id": case.display_id,
        "problem_type": case.problem_type,
        "ground_truth": case.ground_truth,
        "reward_min": case.reward_min,
        "reward_max": case.reward_max,
        "cot_changed": case.cot_changed,
        "answer_changed": case.answer_changed,
        "attempts": [
            {
                "step": attempt.step,
                "reward": attempt.reward,
                "thought": attempt.thought,
                "answer": attempt.answer,
                "cot_repaired": attempt.cot_repaired,
                "final_token_len": attempt.final_token_len,
                "response": attempt.response,
            }
            for attempt in case.attempts
        ],
    }


def print_markdown(cases: list[ValCotCase], *, limit: int, thought_chars: int) -> None:
    for idx, case in enumerate(cases[:limit], start=1):
        print(f"## {idx}. {case.label} | {case.problem_type} | {case.display_id}")
        print()
        print(
            f"- uid: `{case.uid}`; reward: {format_float(case.reward_min)} -> {format_float(case.reward_max)}; "
            f"answer_changed: {case.answer_changed}"
        )
        print(f"- gt: `{truncate(case.ground_truth, 240)}`")
        print()
        print("| step | reward | repaired | len | answer | thought |")
        print("| ---: | ---: | --- | ---: | --- | --- |")
        for attempt in case.attempts:
            print(
                "| "
                + " | ".join(
                    [
                        str(attempt.step),
                        format_float(attempt.reward),
                        "" if attempt.cot_repaired is None else str(attempt.cot_repaired),
                        format_float(attempt.final_token_len),
                        f"`{truncate(attempt.answer, 80)}`",
                        truncate(attempt.thought, thought_chars).replace("|", "\\|"),
                    ]
                )
                + " |"
            )
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiments", nargs="+", help="Experiment dirs, or label=/path/to/dir")
    parser.add_argument("--max-step", type=int, default=100)
    parser.add_argument("--min-attempts", type=int, default=2)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--thought-chars", type=int, default=360)
    parser.add_argument("--task-filter", help="Only include problem types containing this text")
    parser.add_argument("--format", choices=("markdown", "jsonl"), default="markdown")
    args = parser.parse_args()

    all_cases: list[ValCotCase] = []
    for item in args.experiments:
        label, exp_dir = parse_exp_arg(item)
        all_cases.extend(
            collect_val_cot_change_cases(
                exp_dir,
                label=label,
                max_step=args.max_step,
                min_attempts=args.min_attempts,
                task_filter=args.task_filter,
            )
        )

    if args.format == "jsonl":
        for case in all_cases[: args.limit]:
            print(json.dumps(case_to_json(case), ensure_ascii=False, sort_keys=True))
    else:
        print_markdown(all_cases, limit=args.limit, thought_chars=args.thought_chars)


if __name__ == "__main__":
    main()
