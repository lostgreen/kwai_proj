#!/usr/bin/env python3
"""Patch legacy hier-seg prompt wording in existing JSONL files.

This only rewrites static prompt text. It does not touch video paths, frame
metadata, fps metadata, answers, or sampling policy metadata.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path


REPLACEMENTS = (
    (
        "IMPORTANT — SPARSE SAMPLING:\n"
        "This clip is sampled at 1-2 fps (not continuous video). "
        "Do NOT rely on single-frame micro-motions, instantaneous contact changes, "
        "or camera cuts to place boundaries. "
        "Create a boundary ONLY when the change is sustained across multiple sampled frames "
        "or when the task/state clearly shifts.",
        "IMPORTANT — SPARSE VISUAL EVIDENCE:\n"
        "This clip is represented by sparsely sampled frames and timestamp markers, "
        "not continuous video. Use the displayed timestamps as the temporal reference; "
        "do not assume every second is visually observed. "
        "Do NOT rely on single-frame micro-motions, instantaneous contact changes, "
        "or camera cuts to place boundaries. "
        "Create a boundary ONLY when the change is visible across multiple sampled frames "
        "or when the task/state clearly shifts.",
    ),
)

REGEX_REPLACEMENTS = (
    (
        re.compile(r"You are given a ([^.\n]+?s video clip \(timestamps 0 to [^)]+\)), sampled at 1-2 fps\."),
        r"You are given a \1 represented by sparsely sampled frames.",
    ),
    (
        re.compile(r"You are given a ([^.\n]+?s video clip), sampled at 1-2 fps\."),
        r"You are given a \1 represented by sparsely sampled frames.",
    ),
)


def patch_text(text: str) -> tuple[str, int]:
    changes = 0
    for old, new in REPLACEMENTS:
        count = text.count(old)
        if count:
            text = text.replace(old, new)
            changes += count
    for pattern, replacement in REGEX_REPLACEMENTS:
        text, count = pattern.subn(replacement, text)
        changes += count
    return text, changes


def patch_record(record: dict) -> tuple[dict, int]:
    changes = 0
    patched = dict(record)

    for key in ("prompt",):
        value = patched.get(key)
        if isinstance(value, str):
            new_value, n = patch_text(value)
            if n:
                patched[key] = new_value
                changes += n

    messages = patched.get("messages")
    if isinstance(messages, list):
        new_messages = []
        for msg in messages:
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                new_msg = dict(msg)
                new_content, n = patch_text(new_msg["content"])
                if n:
                    new_msg["content"] = new_content
                    changes += n
                new_messages.append(new_msg)
            else:
                new_messages.append(msg)
        patched["messages"] = new_messages

    return patched, changes


def patch_jsonl(path: Path, *, dry_run: bool, backup: bool) -> tuple[int, int]:
    records = 0
    changed_records = 0

    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out, path.open("r", encoding="utf-8") as inp:
            for line in inp:
                records += 1
                if not line.strip():
                    out.write(line)
                    continue
                record = json.loads(line)
                patched, changes = patch_record(record)
                if changes:
                    changed_records += 1
                out.write(json.dumps(patched, ensure_ascii=False) + "\n")

        if dry_run:
            os.unlink(tmp_name)
        else:
            if backup:
                backup_path = path.with_suffix(path.suffix + ".bak")
                if not backup_path.exists():
                    os.replace(path, backup_path)
                else:
                    raise FileExistsError(f"backup already exists: {backup_path}")
            os.replace(tmp_name, path)
    except Exception:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise

    return records, changed_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch legacy 'sampled at 1-2 fps' hier-seg prompt wording in JSONL files.",
    )
    parser.add_argument("jsonl", nargs="+", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--backup", action="store_true", help="Keep a .bak copy before replacing")
    args = parser.parse_args()

    total_records = 0
    total_changed = 0
    for path in args.jsonl:
        records, changed = patch_jsonl(path, dry_run=args.dry_run, backup=args.backup)
        total_records += records
        total_changed += changed
        mode = "dry-run" if args.dry_run else "patched"
        print(f"[{mode}] {path}: records={records} changed_records={changed}")

    print(f"[summary] records={total_records} changed_records={total_changed}")


if __name__ == "__main__":
    main()
