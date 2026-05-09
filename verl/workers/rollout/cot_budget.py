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


@dataclass(frozen=True)
class CoTBudgetController:
    """State-free controller that decides when a CoT close tag must be forced."""

    start_token_ids: Sequence[int]
    end_token_ids: Sequence[int]
    max_tokens: int

    def __post_init__(self) -> None:
        if not self.start_token_ids:
            raise ValueError("start_token_ids must not be empty")
        if not self.end_token_ids:
            raise ValueError("end_token_ids must not be empty")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

    def next_forced_token(self, token_ids: Sequence[int]) -> Optional[int]:
        """Return the next forced close-tag token, or None when sampling is free."""

        start_idx = _find_last_subsequence(token_ids, self.start_token_ids)
        if start_idx < 0:
            return None

        content_start = start_idx + len(self.start_token_ids)
        end_idx = _find_subsequence(token_ids, self.end_token_ids, start=content_start)
        if end_idx >= 0:
            return None

        generated_after_start = list(token_ids[content_start:])
        if len(generated_after_start) >= self.max_tokens:
            close_prefix_len = self._close_prefix_len(generated_after_start)
            if close_prefix_len > 0:
                return self.end_token_ids[close_prefix_len]
            return self.end_token_ids[0]

        return None

    def _close_prefix_len(self, generated_after_start: Sequence[int]) -> int:
        max_prefix_len = min(len(generated_after_start), len(self.end_token_ids) - 1)
        for prefix_len in range(max_prefix_len, 0, -1):
            if list(generated_after_start[-prefix_len:]) == list(self.end_token_ids[:prefix_len]):
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


def make_cot_budget_processor(
    tokenizer: Any,
    *,
    start_token: str,
    end_token: str,
    max_tokens: int,
) -> CoTBudgetProcessor:
    start_token_ids = tokenizer.encode(start_token, add_special_tokens=False)
    end_token_ids = tokenizer.encode(end_token, add_special_tokens=False)
    return CoTBudgetProcessor(
        CoTBudgetController(
            start_token_ids=start_token_ids,
            end_token_ids=end_token_ids,
            max_tokens=max_tokens,
        )
    )
