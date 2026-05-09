#!/usr/bin/env python3
"""Inspect how target tokenizers split candidate CoT tags."""

from __future__ import annotations

import argparse
import json
from typing import Any

from transformers import AutoTokenizer


DEFAULT_TAGS = ("<think>", "</think>", "<thought>", "</thought>")


def probe_tags(model_name_or_path: str, tags: list[str], trust_remote_code: bool = False) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    results = {}
    for tag in tags:
        token_ids = tokenizer.encode(tag, add_special_tokens=False)
        results[tag] = {
            "token_ids": token_ids,
            "num_tokens": len(token_ids),
            "tokens": tokenizer.convert_ids_to_tokens(token_ids),
            "decoded": tokenizer.decode(token_ids, skip_special_tokens=False),
        }
    return {
        "model": model_name_or_path,
        "tokenizer_class": tokenizer.__class__.__name__,
        "tags": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Tokenizer/model name or local path, e.g. Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument(
        "--tag",
        action="append",
        dest="tags",
        help="Tag string to inspect. Can be passed multiple times. Defaults to common CoT tags.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    tags = args.tags or list(DEFAULT_TAGS)
    print(json.dumps(probe_tags(args.model, tags, args.trust_remote_code), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
