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

import importlib.util
import os
import re
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import partial
from typing import Any, Callable, Optional, Tuple, TypedDict

import torch
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from .config import RewardConfig
from .metrics import build_dense_reward_metrics, coerce_reward_metric


class RewardInput(TypedDict, total=False):
    response: str
    response_length: int
    ground_truth: str
    data_type: str
    problem_type: str
    problem: Optional[str]
    problem_id: Optional[str]
    metadata: Any
    cot_budget_debug: Any


class RewardScore(TypedDict):
    overall: float
    format: Optional[float]
    accuracy: Optional[float]


SequentialRewardFunction = Callable[[RewardInput], RewardScore]

BatchRewardFunction = Callable[[list[RewardInput]], list[RewardScore]]


_SEGMENT_TAG_PAIRS = (
    ("<thought>", "</thought>", "thought", True),
    ("<think>", "</think>", "thought", True),
    ("<answer>", "</answer>", "answer", False),
)

_SEGMENT_METRIC_LABELS = (
    (1, "thought"),
    (2, "answer"),
    (0, "default"),
    (3, "format"),
)

_SEGMENT_FORMAT_RE = re.compile(r"</?(?:thought|think|answer)>", re.IGNORECASE)


def _token_count(tokenizer: PreTrainedTokenizer, text: str) -> int:
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return len(tokenizer.encode(text))


def _default_ranges(response: str, spans: list[tuple[int, int, str]]) -> list[tuple[int, int]]:
    ranges = []
    cursor = 0
    for start, end, _ in spans:
        if start > cursor:
            ranges.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < len(response):
        ranges.append((cursor, len(response)))
    return ranges


def _is_format_default(text: str) -> bool:
    if not text.strip():
        return True
    return not _SEGMENT_FORMAT_RE.sub("", text).strip()


def _segment_spans(response: str, *, answer_fallback_after_thought: bool) -> list[tuple[int, int, str]]:
    spans = []
    thought_close_positions = []
    has_answer_span = False
    for open_tag, close_tag, segment_name, enables_fallback in _SEGMENT_TAG_PAIRS:
        search_from = 0
        while True:
            open_pos = response.find(open_tag, search_from)
            if open_pos < 0:
                break
            content_start = open_pos + len(open_tag)
            close_pos = response.find(close_tag, content_start)
            if close_pos < 0:
                break
            if close_pos > content_start:
                spans.append((content_start, close_pos, segment_name))
                has_answer_span = has_answer_span or segment_name == "answer"
                if enables_fallback:
                    thought_close_positions.append(close_pos + len(close_tag))
            search_from = close_pos + len(close_tag)
    if answer_fallback_after_thought and thought_close_positions and not has_answer_span:
        fallback_start = max(thought_close_positions)
        fallback_end = len(response)
        if response[fallback_start:fallback_end].strip():
            spans.append((fallback_start, fallback_end, "answer"))
    spans.sort(key=lambda item: item[0])
    return spans


def build_response_loss_weight_mask(
    response_ids: torch.Tensor,
    response_mask: torch.Tensor,
    response: str,
    tokenizer: PreTrainedTokenizer,
    config: RewardConfig,
) -> torch.Tensor:
    """Build per-response-token loss weights from CoT/answer XML-style spans."""
    weights, _ = build_response_loss_weight_mask_and_metrics(
        response_ids,
        response_mask,
        response,
        tokenizer,
        config,
    )
    return weights


def build_response_loss_weight_mask_and_metrics(
    response_ids: torch.Tensor,
    response_mask: torch.Tensor,
    response: str,
    tokenizer: PreTrainedTokenizer,
    config: RewardConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Build per-token loss weights plus compact audit metrics for CoT weighting."""
    weights = torch.ones_like(response_mask, dtype=torch.float32) * float(config.default_loss_weight)
    weights = weights * response_mask.to(dtype=weights.dtype)
    segment_labels = torch.zeros_like(response_mask, dtype=torch.long)
    spans = _segment_spans(
        response,
        answer_fallback_after_thought=bool(config.answer_fallback_after_thought),
    )
    if not spans:
        fallback = response_mask.to(dtype=torch.float32)
        return fallback, _summarize_response_loss_weights(fallback, response_mask, segment_labels)

    valid_len = int(torch.sum(response_mask).item())
    if config.format_loss_weight is not None:
        for char_start, char_end in _default_ranges(response, spans):
            if not _is_format_default(response[char_start:char_end]):
                continue
            token_start = min(_token_count(tokenizer, response[:char_start]), valid_len)
            token_end = min(_token_count(tokenizer, response[:char_end]), valid_len)
            if token_end <= token_start:
                continue
            weights[token_start:token_end] = float(config.format_loss_weight)
            segment_labels[token_start:token_end] = 3

    for char_start, char_end, segment_name in spans:
        token_start = min(_token_count(tokenizer, response[:char_start]), valid_len)
        token_end = min(_token_count(tokenizer, response[:char_end]), valid_len)
        if token_end <= token_start:
            continue
        segment_weight = (
            config.thought_loss_weight if segment_name == "thought" else config.answer_loss_weight
        )
        weights[token_start:token_end] = float(segment_weight)
        segment_labels[token_start:token_end] = 1 if segment_name == "thought" else 2

    return weights, _summarize_response_loss_weights(weights, response_mask, segment_labels)


def _summarize_response_loss_weights(
    weights: torch.Tensor,
    response_mask: torch.Tensor,
    segment_labels: torch.Tensor,
) -> dict[str, float]:
    mask = response_mask.to(dtype=weights.dtype)
    valid_tokens = torch.sum(mask).item()
    if valid_tokens <= 0:
        return {
            "response_loss_weight/mean": 0.0,
            "response_loss_weight/weighted_token_ratio": 0.0,
            **{f"response_loss_weight/{name}_token_ratio": 0.0 for _, name in _SEGMENT_METRIC_LABELS},
            **{f"response_loss_weight/{name}_effective_ratio": 0.0 for _, name in _SEGMENT_METRIC_LABELS},
        }

    weighted_tokens = torch.sum(weights * mask).item()
    metrics = {
        "response_loss_weight/mean": float(weighted_tokens / valid_tokens),
        "response_loss_weight/weighted_token_ratio": float(weighted_tokens / valid_tokens),
    }
    total_weight = max(float(weighted_tokens), 1e-8)
    for label_id, name in _SEGMENT_METRIC_LABELS:
        segment_mask = (segment_labels == label_id).to(dtype=weights.dtype) * mask
        token_count = torch.sum(segment_mask).item()
        segment_weight = torch.sum(weights * segment_mask).item()
        metrics[f"response_loss_weight/{name}_token_ratio"] = float(token_count / valid_tokens)
        metrics[f"response_loss_weight/{name}_effective_ratio"] = float(segment_weight / total_weight)
    return metrics


class FunctionRewardManager(ABC):
    """Reward manager for rule-based reward."""

    def __init__(self, config: RewardConfig, tokenizer: PreTrainedTokenizer):
        if config.reward_function is None:
            raise ValueError("Reward function is not provided.")

        if not os.path.exists(config.reward_function):
            raise FileNotFoundError(f"Reward function file {config.reward_function} not found.")

        spec = importlib.util.spec_from_file_location("custom_reward_fn", config.reward_function)
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["custom_reward_fn"] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Failed to load reward function: {e}")

        if not hasattr(module, config.reward_function_name):
            raise AttributeError(f"Module {module} does not have function {config.reward_function_name}.")

        reward_fn = getattr(module, config.reward_function_name)
        print(f"Using reward function `{config.reward_function_name}` from `{config.reward_function}`.")
        self.reward_fn = partial(reward_fn, **config.reward_function_kwargs)
        self.config = config
        self.tokenizer = tokenizer

    def _maybe_build_loss_weight_mask(
        self,
        response_ids: torch.Tensor,
        response_mask: torch.Tensor,
        response: str,
    ) -> Optional[torch.Tensor]:
        if not self.config.enable_response_loss_weight_mask:
            return None
        return build_response_loss_weight_mask(response_ids, response_mask, response, self.tokenizer, self.config)

    def _maybe_build_loss_weight_mask_and_metrics(
        self,
        response_ids: torch.Tensor,
        response_mask: torch.Tensor,
        response: str,
    ) -> tuple[Optional[torch.Tensor], dict[str, float]]:
        if not self.config.enable_response_loss_weight_mask:
            return None, {}
        return build_response_loss_weight_mask_and_metrics(
            response_ids,
            response_mask,
            response,
            self.tokenizer,
            self.config,
        )

    @abstractmethod
    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        """Compute reward for a batch of data."""
        ...


class SequentialFunctionRewardManager(FunctionRewardManager):
    reward_fn: SequentialRewardFunction

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        loss_weight_mask = (
            torch.zeros_like(data.batch["response_mask"], dtype=torch.float32)
            if self.config.enable_response_loss_weight_mask
            else None
        )
        reward_metrics = defaultdict(list)
        response_ids = data.batch["responses"]
        response_length = torch.sum(data.batch["response_mask"], dim=-1)
        for i in range(len(data)):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            valid_response_ids = response_ids[i][:cur_response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            score = self.reward_fn(
                {
                    "response": response_str,
                    "response_length": cur_response_length,
                    "ground_truth": data.non_tensor_batch["ground_truth"][i],
                }
            )
            reward_tensor[i, cur_response_length - 1] = score["overall"]
            if loss_weight_mask is not None:
                loss_weight_mask[i], loss_weight_metrics = self._maybe_build_loss_weight_mask_and_metrics(
                    response_ids[i],
                    data.batch["response_mask"][i],
                    response_str,
                )
                for key, value in loss_weight_metrics.items():
                    reward_metrics[key].append(value)
            for key, value in score.items():
                reward_metrics[key].append(value)

        if loss_weight_mask is not None:
            return reward_tensor, reward_metrics, {"response_loss_weight_mask": loss_weight_mask}
        return reward_tensor, reward_metrics


class BatchFunctionRewardManager(FunctionRewardManager):
    reward_fn: BatchRewardFunction

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        reward_inputs = []
        loss_weight_metric_rows = []
        response_ids = data.batch["responses"]
        response_length = torch.sum(data.batch["response_mask"], dim=-1)
        loss_weight_mask = (
            torch.zeros_like(data.batch["response_mask"], dtype=torch.float32)
            if self.config.enable_response_loss_weight_mask
            else None
        )
        cot_budget_debug = data.non_tensor_batch.get("cot_budget_debug", [None] * len(data))
        for i in range(len(data)):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            valid_response_ids = response_ids[i][:cur_response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )


            reward_inputs.append(
                {
                    "response": response_str,
                    "response_length": cur_response_length,
                    "ground_truth": data.non_tensor_batch["ground_truth"][i],
                    "data_type": data.non_tensor_batch["data_type"][i],
                    "problem_type": data.non_tensor_batch["problem_type"][i],
                    "problem": data.non_tensor_batch.get("problem_reserved_text", [None]*len(data))[i],
                    "problem_id": data.non_tensor_batch.get("problem_id", [None]*len(data))[i],
                    "metadata": data.non_tensor_batch.get("metadata", [None]*len(data))[i],
                    "cot_budget_debug": cot_budget_debug[i],
                }
            )
            if loss_weight_mask is not None:
                loss_weight_mask[i], loss_weight_metrics = self._maybe_build_loss_weight_mask_and_metrics(
                    response_ids[i],
                    data.batch["response_mask"][i],
                    response_str,
                )
                loss_weight_metric_rows.append(loss_weight_metrics)


        # print(data)
        # print("\n\n\n\n\n\n\n\n\n")
        # print(reward_inputs)

        scores = self.reward_fn(reward_inputs)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = build_dense_reward_metrics(scores, len(data))
        for loss_weight_metrics in loss_weight_metric_rows:
            for key, value in loss_weight_metrics.items():
                reward_metrics[key].append(value)
        for i in range(len(data)):
            score = scores[i] if i < len(scores) and isinstance(scores[i], dict) else {}
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            reward_tensor[i, cur_response_length - 1] = coerce_reward_metric(score.get("overall", 0.0))

        if loss_weight_mask is not None:
            return reward_tensor, reward_metrics, {"response_loss_weight_mask": loss_weight_mask}
        return reward_tensor, reward_metrics
