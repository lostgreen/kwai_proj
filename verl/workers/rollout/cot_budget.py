# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for limiting manually tagged chain-of-thought spans during rollout."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Optional, Sequence


def _find_subsequence(seq: Sequence[int], pattern: Sequence[int], start: int = 0) -> int:
    if not pattern:
        return -1
    limit = len(seq) - len(pattern)
    for idx in range(max(start, 0), limit + 1):
        if list(seq[idx : idx + len(pattern)]) == list(pattern):
            return idx
    return -1


def _find_last_subsequence(seq: Sequence[int], pattern: Sequence[int]) -> int:
    if not pattern or len(pattern) > len(seq):
        return -1
    for idx in range(len(seq) - len(pattern), -1, -1):
        if list(seq[idx : idx + len(pattern)]) == list(pattern):
            return idx
    return -1


def _unique_patterns(patterns: Sequence[Sequence[int]]) -> list[list[int]]:
    unique = []
    seen = set()
    for pattern in patterns:
        normalized = tuple(pattern)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(list(normalized))
    return unique


def _find_subsequence_any(
    seq: Sequence[int], patterns: Sequence[Sequence[int]], start: int = 0
) -> tuple[int, Optional[list[int]]]:
    best_idx = -1
    best_pattern = None
    for pattern in patterns:
        idx = _find_subsequence(seq, pattern, start=start)
        if idx >= 0 and (best_idx < 0 or idx < best_idx or (idx == best_idx and len(pattern) > len(best_pattern or []))):
            best_idx = idx
            best_pattern = list(pattern)
    return best_idx, best_pattern


def _find_last_subsequence_any(seq: Sequence[int], patterns: Sequence[Sequence[int]]) -> tuple[int, Optional[list[int]]]:
    best_idx = -1
    best_pattern = None
    for pattern in patterns:
        idx = _find_last_subsequence(seq, pattern)
        if idx >= 0 and (idx > best_idx or (idx == best_idx and len(pattern) > len(best_pattern or []))):
            best_idx = idx
            best_pattern = list(pattern)
    return best_idx, best_pattern


@dataclass(frozen=True)
class CoTBudgetController:
    """State-free controller that decides when a CoT close tag must be forced."""

    start_token_ids: Sequence[int]
    end_token_ids: Sequence[int]
    max_tokens: int
    start_token_id_variants: Optional[Sequence[Sequence[int]]] = None
    end_token_id_variants: Optional[Sequence[Sequence[int]]] = None
    token_pair_variants: Optional[Sequence[tuple[Sequence[int], Sequence[int], Sequence[Sequence[int]]]]] = None
    repair_suffix_token_ids: Optional[Sequence[int]] = None

    def __post_init__(self) -> None:
        if not self.start_token_ids:
            raise ValueError("start_token_ids must not be empty")
        if not self.end_token_ids:
            raise ValueError("end_token_ids must not be empty")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

    def next_forced_token(self, token_ids: Sequence[int]) -> Optional[int]:
        """Return the next forced close-tag token, or None when sampling is free."""

        start_idx, start_pattern, end_token_ids, end_patterns = self.find_latest_start(token_ids)
        if start_idx < 0:
            return None

        repair_close_ids = list(end_token_ids) + list(self.repair_suffix_token_ids or [])
        content_start = start_idx + len(start_pattern or [])
        end_idx, _ = _find_subsequence_any(token_ids, end_patterns, start=content_start)
        if end_idx >= 0:
            generated_after_start = list(token_ids[content_start:])
            close_prefix_len = self._close_prefix_len(generated_after_start, repair_close_ids)
            if close_prefix_len >= len(end_token_ids) and close_prefix_len < len(repair_close_ids):
                return repair_close_ids[close_prefix_len]
            return None

        generated_after_start = list(token_ids[content_start:])
        if len(generated_after_start) >= self.max_tokens:
            close_prefix_len = self._close_prefix_len(generated_after_start, repair_close_ids)
            if close_prefix_len > 0:
                return repair_close_ids[close_prefix_len]
            return repair_close_ids[0]

        return None

    def repaired_prefix(self, token_ids: Sequence[int], max_length: Optional[int] = None) -> Optional[list[int]]:
        """Return a budget-compliant prefix with a closed CoT span, if repair is needed."""

        start_idx, start_pattern, end_token_ids, end_patterns = self.find_latest_start(token_ids)
        repair_close_ids = list(end_token_ids) + list(self.repair_suffix_token_ids or [])
        if start_idx < 0:
            return None

        content_start = start_idx + len(start_pattern or [])
        end_idx, _ = _find_subsequence_any(token_ids, end_patterns, start=content_start)
        if end_idx >= 0 and end_idx - content_start <= self.max_tokens:
            return None

        budgeted_content_end = min(content_start + self.max_tokens, len(token_ids))
        if max_length is not None:
            if max_length <= 0:
                return []
            if max_length < len(repair_close_ids):
                return list(repair_close_ids[:max_length])
            available_content_tokens = max_length - content_start - len(repair_close_ids)
            if available_content_tokens < 0:
                return list(token_ids[: max_length - len(repair_close_ids)]) + repair_close_ids
            budgeted_content_end = min(budgeted_content_end, content_start + available_content_tokens)

        content_end = budgeted_content_end
        return list(token_ids[:content_end]) + repair_close_ids

    def has_start(self, token_ids: Sequence[int]) -> bool:
        start_idx, _, _, _ = self.find_latest_start(token_ids)
        return start_idx >= 0

    def span_status(self, token_ids: Sequence[int]) -> dict[str, Any]:
        start_idx, start_pattern, end_token_ids, end_patterns = self.find_latest_start(token_ids)
        if start_idx < 0:
            return {
                "cot_start_detected": False,
                "cot_start_index": -1,
                "cot_start_token_ids": None,
                "cot_end_detected": False,
                "cot_end_index": -1,
                "cot_end_token_ids": None,
                "cot_end_pattern_ids": None,
            }

        content_start = start_idx + len(start_pattern or [])
        end_idx, end_pattern = _find_subsequence_any(token_ids, end_patterns, start=content_start)
        return {
            "cot_start_detected": True,
            "cot_start_index": start_idx,
            "cot_start_token_ids": list(start_pattern or []),
            "cot_end_detected": end_idx >= 0,
            "cot_end_index": end_idx,
            "cot_end_token_ids": list(end_token_ids),
            "cot_end_pattern_ids": list(end_pattern) if end_pattern is not None else None,
        }

    def find_latest_start(
        self, token_ids: Sequence[int]
    ) -> tuple[int, Optional[list[int]], list[int], list[list[int]]]:
        best_idx = -1
        best_start = None
        best_end = list(self.end_token_ids)
        best_end_patterns = self._end_patterns()
        for start_pattern, end_token_ids, end_patterns in self._token_pattern_groups():
            idx = _find_last_subsequence(token_ids, start_pattern)
            if idx >= 0 and (idx > best_idx or (idx == best_idx and len(start_pattern) > len(best_start or []))):
                best_idx = idx
                best_start = list(start_pattern)
                best_end = list(end_token_ids)
                best_end_patterns = _unique_patterns([end_token_ids, *end_patterns])
        return best_idx, best_start, best_end, best_end_patterns

    def _start_patterns(self) -> list[list[int]]:
        return _unique_patterns([self.start_token_ids, *(self.start_token_id_variants or [])])

    def _end_patterns(self) -> list[list[int]]:
        return _unique_patterns([self.end_token_ids, *(self.end_token_id_variants or [])])

    def _token_pattern_groups(self) -> list[tuple[list[int], list[int], list[list[int]]]]:
        groups = [(pattern, list(self.end_token_ids), self._end_patterns()) for pattern in self._start_patterns()]
        for start_pattern, end_token_ids, end_patterns in self.token_pair_variants or []:
            if not start_pattern or not end_token_ids:
                continue
            groups.append((list(start_pattern), list(end_token_ids), _unique_patterns(end_patterns)))
        return groups

    def _close_prefix_len(self, generated_after_start: Sequence[int], end_token_ids: Sequence[int]) -> int:
        max_prefix_len = min(len(generated_after_start), len(end_token_ids) - 1)
        for prefix_len in range(max_prefix_len, 0, -1):
            if list(generated_after_start[-prefix_len:]) == list(end_token_ids[:prefix_len]):
                return prefix_len
        return 0


class CoTBudgetProcessor:
    """vLLM-compatible logits processor for a configurable CoT budget."""

    def __init__(self, controller: CoTBudgetController):
        self.controller = controller

    def __call__(self, *args: Any) -> Any:
        if len(args) == 2:
            token_ids, logits = args
        elif len(args) == 3:
            _, token_ids, logits = args
        else:
            raise TypeError(f"CoTBudgetProcessor expected 2 or 3 arguments, got {len(args)}")
        forced_token = self.controller.next_forced_token(token_ids)
        if forced_token is None:
            return logits

        return self._force_token(logits, forced_token)

    @staticmethod
    def _force_token(logits: Any, token_id: int) -> Any:
        if hasattr(logits, "fill_"):
            logits.fill_(-math.inf)
            logits[token_id] = 0.0
            return logits

        masked = [-math.inf] * max(len(logits), token_id + 1)
        masked[token_id] = 0.0
        return masked


def configure_vllm_engine_for_cot_budget(cot_budget_enabled: bool) -> None:
    if cot_budget_enabled:
        os.environ["VLLM_USE_V1"] = "1"


def _encode_variants(
    tokenizer: Any,
    token: str,
    suffixes: Sequence[str],
    prefixes: Sequence[str] = ("",),
) -> list[list[int]]:
    variants = []
    for prefix in prefixes:
        for suffix in suffixes:
            try:
                variants.append(tokenizer.encode(prefix + token + suffix, add_special_tokens=False))
            except Exception:
                continue
    return variants


def _paired_reasoning_aliases(start_token: str, end_token: str) -> list[tuple[str, str]]:
    known_pairs = {
        ("<think>", "</think>"): [("<thought>", "</thought>")],
        ("<thought>", "</thought>"): [("<think>", "</think>")],
    }
    return known_pairs.get((start_token, end_token), [])


def make_cot_budget_controller(
    tokenizer: Any,
    *,
    start_token: str,
    end_token: str,
    max_tokens: int,
    repair_suffix: str = "",
) -> CoTBudgetController:
    start_token_ids = tokenizer.encode(start_token, add_special_tokens=False)
    end_token_ids = tokenizer.encode(end_token, add_special_tokens=False)
    start_variants = _encode_variants(tokenizer, start_token, ["\n", "\n\n", " ", "\t", "\r\n"])
    end_variants = _encode_variants(
        tokenizer,
        end_token,
        ["", "\n", "\n\n", " ", "\t", "\r\n", "<answer>", "\n<answer>", " <answer>"],
        prefixes=("", "\n"),
    )
    repair_suffix_token_ids = None
    if repair_suffix:
        repair_suffix_token_ids = tokenizer.encode(repair_suffix, add_special_tokens=False)
    token_pair_variants = []
    for alias_start_token, alias_end_token in _paired_reasoning_aliases(start_token, end_token):
        try:
            alias_start_token_ids = tokenizer.encode(alias_start_token, add_special_tokens=False)
            alias_end_token_ids = tokenizer.encode(alias_end_token, add_special_tokens=False)
        except Exception:
            continue
        alias_start_variants = _encode_variants(
            tokenizer,
            alias_start_token,
            ["", "\n", "\n\n", " ", "\t", "\r\n"],
        )
        alias_end_variants = _encode_variants(
            tokenizer,
            alias_end_token,
            ["", "\n", "\n\n", " ", "\t", "\r\n", "<answer>", "\n<answer>", " <answer>"],
            prefixes=("", "\n"),
        )
        for alias_start_pattern in _unique_patterns([alias_start_token_ids, *alias_start_variants]):
            token_pair_variants.append((alias_start_pattern, alias_end_token_ids, alias_end_variants))
    return CoTBudgetController(
        start_token_ids=start_token_ids,
        end_token_ids=end_token_ids,
        max_tokens=max_tokens,
        start_token_id_variants=start_variants,
        end_token_id_variants=end_variants,
        token_pair_variants=token_pair_variants,
        repair_suffix_token_ids=repair_suffix_token_ids,
    )


def make_cot_budget_processor(
    tokenizer: Any,
    *,
    start_token: str,
    end_token: str,
    max_tokens: int,
    repair_suffix: str = "",
) -> CoTBudgetProcessor:
    return CoTBudgetProcessor(
        make_cot_budget_controller(
            tokenizer,
            start_token=start_token,
            end_token=end_token,
            max_tokens=max_tokens,
            repair_suffix=repair_suffix,
        )
    )
