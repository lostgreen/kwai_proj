from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_core_algos():
    module_names = (
        "verl",
        "verl.trainer",
        "verl.utils",
        "verl.utils.torch_functional",
        "verl.trainer.core_algos",
    )
    old_modules = {name: sys.modules.get(name) for name in module_names}

    verl_stub = types.ModuleType("verl")
    trainer_stub = types.ModuleType("verl.trainer")
    utils_stub = types.ModuleType("verl.utils")
    torch_functional_stub = types.ModuleType("verl.utils.torch_functional")

    def masked_mean(values, mask, eps=1e-8):
        return (values * mask).sum() / (mask.sum() + eps)

    torch_functional_stub.masked_mean = masked_mean
    sys.modules["verl"] = verl_stub
    sys.modules["verl.trainer"] = trainer_stub
    sys.modules["verl.utils"] = utils_stub
    sys.modules["verl.utils.torch_functional"] = torch_functional_stub

    try:
        spec = __import__("importlib.util").util.spec_from_file_location(
            "verl.trainer.core_algos",
            REPO_ROOT / "verl" / "trainer" / "core_algos.py",
        )
        assert spec is not None and spec.loader is not None
        module = __import__("importlib.util").util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old in old_modules.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def _load_reward_modules():
    module_names = (
        "transformers",
        "verl",
        "verl.protocol",
        "verl.workers",
        "verl.workers.reward",
        "verl.workers.reward.config",
        "verl.workers.reward.metrics",
        "verl.workers.reward.function",
    )
    old_modules = {name: sys.modules.get(name) for name in module_names}

    transformers_stub = types.ModuleType("transformers")
    transformers_stub.PreTrainedTokenizer = object
    verl_stub = types.ModuleType("verl")
    protocol_stub = types.ModuleType("verl.protocol")
    protocol_stub.DataProto = object
    workers_stub = types.ModuleType("verl.workers")
    reward_pkg_stub = types.ModuleType("verl.workers.reward")

    sys.modules["transformers"] = transformers_stub
    sys.modules["verl"] = verl_stub
    sys.modules["verl.protocol"] = protocol_stub
    sys.modules["verl.workers"] = workers_stub
    sys.modules["verl.workers.reward"] = reward_pkg_stub

    try:
        import importlib.util

        config_spec = importlib.util.spec_from_file_location(
            "verl.workers.reward.config",
            REPO_ROOT / "verl" / "workers" / "reward" / "config.py",
        )
        assert config_spec is not None and config_spec.loader is not None
        config_module = importlib.util.module_from_spec(config_spec)
        sys.modules[config_spec.name] = config_module
        config_spec.loader.exec_module(config_module)

        metrics_stub = types.ModuleType("verl.workers.reward.metrics")
        metrics_stub.build_dense_reward_metrics = lambda scores, batch_size: {}
        metrics_stub.coerce_reward_metric = lambda value: float(value)
        sys.modules["verl.workers.reward.metrics"] = metrics_stub

        function_spec = importlib.util.spec_from_file_location(
            "verl.workers.reward.function",
            REPO_ROOT / "verl" / "workers" / "reward" / "function.py",
        )
        assert function_spec is not None and function_spec.loader is not None
        function_module = importlib.util.module_from_spec(function_spec)
        sys.modules[function_spec.name] = function_module
        function_spec.loader.exec_module(function_module)
        return config_module, function_module
    finally:
        for name, old in old_modules.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


class _CharTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(int(token_id)) for token_id in ids)


def test_average_loss_supports_token_weighted_mode():
    core_algos = _load_core_algos()
    values = torch.tensor(
        [
            [1.0, 3.0, 100.0],
            [2.0, 6.0, 100.0],
        ]
    )
    mask = torch.tensor(
        [
            [1, 1, 0],
            [1, 1, 0],
        ],
        dtype=torch.long,
    )
    weights = torch.tensor(
        [
            [0.25, 1.0, 9.0],
            [1.0, 3.0, 9.0],
        ]
    )

    loss = core_algos.average_loss(values, mask, mode="token_weighted", weight_mask=weights)

    expected = (1.0 * 0.25 + 3.0 * 1.0 + 2.0 * 1.0 + 6.0 * 3.0) / (0.25 + 1.0 + 1.0 + 3.0)
    assert torch.isclose(loss, torch.tensor(expected))


def test_average_loss_supports_seq_weighted_mode():
    core_algos = _load_core_algos()
    values = torch.tensor(
        [
            [1.0, 3.0, 100.0],
            [2.0, 6.0, 100.0],
        ]
    )
    mask = torch.tensor(
        [
            [1, 1, 0],
            [1, 1, 0],
        ],
        dtype=torch.long,
    )
    weights = torch.tensor(
        [
            [0.25, 1.0, 9.0],
            [1.0, 3.0, 9.0],
        ]
    )

    loss = core_algos.average_loss(values, mask, mode="seq_weighted", weight_mask=weights)

    seq0 = (1.0 * 0.25 + 3.0 * 1.0) / (0.25 + 1.0)
    seq1 = (2.0 * 1.0 + 6.0 * 3.0) / (1.0 + 3.0)
    assert torch.isclose(loss, torch.tensor((seq0 + seq1) / 2.0))


def test_build_response_loss_weight_mask_marks_thought_and_answer_spans():
    config_module, function_module = _load_reward_modules()
    tokenizer = _CharTokenizer()
    response = "<thought>plan</thought><answer>B</answer>"
    response_ids = torch.tensor(tokenizer.encode(response))
    response_mask = torch.ones_like(response_ids, dtype=torch.float32)
    config = config_module.RewardConfig(
        enable_response_loss_weight_mask=True,
        thought_loss_weight=0.2,
        answer_loss_weight=2.0,
        default_loss_weight=1.0,
    )

    weights = function_module.build_response_loss_weight_mask(response_ids, response_mask, response, tokenizer, config)

    thought_offset = response.index("plan")
    answer_offset = response.index("B")
    assert math.isclose(weights[thought_offset].item(), 0.2, abs_tol=1e-6)
    assert math.isclose(weights[answer_offset].item(), 2.0, abs_tol=1e-6)
    assert math.isclose(weights[0].item(), 1.0, abs_tol=1e-6)


def test_build_response_loss_weight_mask_reports_segment_audit_metrics():
    config_module, function_module = _load_reward_modules()
    tokenizer = _CharTokenizer()
    response = "<thought>abcd</thought><answer>XY</answer>"
    response_ids = torch.tensor(tokenizer.encode(response))
    response_mask = torch.ones_like(response_ids, dtype=torch.float32)
    config = config_module.RewardConfig(
        enable_response_loss_weight_mask=True,
        thought_loss_weight=0.25,
        answer_loss_weight=2.0,
        default_loss_weight=1.0,
    )

    weights, metrics = function_module.build_response_loss_weight_mask_and_metrics(
        response_ids,
        response_mask,
        response,
        tokenizer,
        config,
    )

    valid_len = float(len(response))
    thought_tokens = 4.0
    answer_tokens = 2.0
    default_tokens = valid_len - thought_tokens - answer_tokens
    total_weight = default_tokens * 1.0 + thought_tokens * 0.25 + answer_tokens * 2.0
    assert torch.equal(weights.ne(0).to(dtype=response_mask.dtype), response_mask)
    assert math.isclose(metrics["response_loss_weight/thought_token_ratio"], thought_tokens / valid_len)
    assert math.isclose(metrics["response_loss_weight/answer_token_ratio"], answer_tokens / valid_len)
    assert math.isclose(metrics["response_loss_weight/default_token_ratio"], default_tokens / valid_len)
    assert math.isclose(metrics["response_loss_weight/thought_effective_ratio"], thought_tokens * 0.25 / total_weight)
    assert math.isclose(metrics["response_loss_weight/answer_effective_ratio"], answer_tokens * 2.0 / total_weight)
    assert math.isclose(metrics["response_loss_weight/weighted_token_ratio"], total_weight / valid_len)


def test_build_response_loss_weight_mask_can_downweight_format_tokens_separately():
    config_module, function_module = _load_reward_modules()
    tokenizer = _CharTokenizer()
    response = "<thought>abcd</thought><answer>XY</answer>"
    response_ids = torch.tensor(tokenizer.encode(response))
    response_mask = torch.ones_like(response_ids, dtype=torch.float32)
    config = config_module.RewardConfig(
        enable_response_loss_weight_mask=True,
        thought_loss_weight=0.25,
        answer_loss_weight=2.0,
        default_loss_weight=1.0,
        format_loss_weight=0.1,
    )

    weights, metrics = function_module.build_response_loss_weight_mask_and_metrics(
        response_ids,
        response_mask,
        response,
        tokenizer,
        config,
    )

    assert math.isclose(weights[response.index("<thought>")].item(), 0.1, abs_tol=1e-6)
    assert math.isclose(weights[response.index("abcd")].item(), 0.25, abs_tol=1e-6)
    assert math.isclose(weights[response.index("XY")].item(), 2.0, abs_tol=1e-6)
    assert math.isclose(weights[response.index("</answer>")].item(), 0.1, abs_tol=1e-6)
    assert metrics["response_loss_weight/format_token_ratio"] > 0.0
    assert metrics["response_loss_weight/format_effective_ratio"] < metrics["response_loss_weight/default_effective_ratio"]


def test_build_response_loss_weight_mask_keeps_middle_text_as_default_not_format():
    config_module, function_module = _load_reward_modules()
    tokenizer = _CharTokenizer()
    response = "<thought>draft</thought> extra explanation <answer>B</answer>"
    response_ids = torch.tensor(tokenizer.encode(response))
    response_mask = torch.ones_like(response_ids, dtype=torch.float32)
    config = config_module.RewardConfig(
        enable_response_loss_weight_mask=True,
        thought_loss_weight=0.1,
        answer_loss_weight=0.9,
        default_loss_weight=1.0,
        format_loss_weight=0.2,
    )

    weights, metrics = function_module.build_response_loss_weight_mask_and_metrics(
        response_ids,
        response_mask,
        response,
        tokenizer,
        config,
    )

    assert math.isclose(weights[response.index("extra")].item(), 1.0, abs_tol=1e-6)
    assert math.isclose(weights[response.index("<answer>")].item(), 0.2, abs_tol=1e-6)
    assert metrics["response_loss_weight/default_token_ratio"] > 0.0
    assert metrics["response_loss_weight/format_token_ratio"] > 0.0


def test_build_response_loss_weight_mask_treats_text_after_thought_as_answer_fallback():
    config_module, function_module = _load_reward_modules()
    tokenizer = _CharTokenizer()
    response = "<thought>draft range 1 - 2 seconds</thought>The event happens in the 20 - 30 seconds."
    response_ids = torch.tensor(tokenizer.encode(response))
    response_mask = torch.ones_like(response_ids, dtype=torch.float32)
    config = config_module.RewardConfig(
        enable_response_loss_weight_mask=True,
        thought_loss_weight=0.1,
        answer_loss_weight=0.9,
        default_loss_weight=1.0,
    )

    weights, metrics = function_module.build_response_loss_weight_mask_and_metrics(
        response_ids,
        response_mask,
        response,
        tokenizer,
        config,
    )

    thought_offset = response.index("draft")
    answer_offset = response.index("The event")
    assert math.isclose(weights[thought_offset].item(), 0.1, abs_tol=1e-6)
    assert math.isclose(weights[answer_offset].item(), 0.9, abs_tol=1e-6)
    assert metrics["response_loss_weight/answer_token_ratio"] > 0.0
    assert metrics["response_loss_weight/answer_effective_ratio"] > 0.0


def test_build_response_loss_weight_mask_treats_events_after_thought_as_answer_fallback():
    config_module, function_module = _load_reward_modules()
    tokenizer = _CharTokenizer()
    response = "<thought>draft</thought><events>[[20, 30]]</events>"
    response_ids = torch.tensor(tokenizer.encode(response))
    response_mask = torch.ones_like(response_ids, dtype=torch.float32)
    config = config_module.RewardConfig(
        enable_response_loss_weight_mask=True,
        thought_loss_weight=0.1,
        answer_loss_weight=0.9,
        default_loss_weight=1.0,
    )

    weights, metrics = function_module.build_response_loss_weight_mask_and_metrics(
        response_ids,
        response_mask,
        response,
        tokenizer,
        config,
    )

    event_offset = response.index("<events>")
    assert math.isclose(weights[event_offset].item(), 0.9, abs_tol=1e-6)
    assert metrics["response_loss_weight/answer_token_ratio"] > 0.0


def test_build_response_loss_weight_mask_can_disable_answer_fallback_after_thought():
    config_module, function_module = _load_reward_modules()
    tokenizer = _CharTokenizer()
    response = "<thought>draft</thought>The event happens in the 20 - 30 seconds."
    response_ids = torch.tensor(tokenizer.encode(response))
    response_mask = torch.ones_like(response_ids, dtype=torch.float32)
    config = config_module.RewardConfig(
        enable_response_loss_weight_mask=True,
        thought_loss_weight=0.1,
        answer_loss_weight=0.9,
        default_loss_weight=1.0,
        answer_fallback_after_thought=False,
    )

    weights, metrics = function_module.build_response_loss_weight_mask_and_metrics(
        response_ids,
        response_mask,
        response,
        tokenizer,
        config,
    )

    answer_offset = response.index("The event")
    assert math.isclose(weights[answer_offset].item(), 1.0, abs_tol=1e-6)
    assert math.isclose(metrics["response_loss_weight/answer_token_ratio"], 0.0, abs_tol=1e-6)


def test_build_response_loss_weight_mask_falls_back_to_ones_without_tags():
    config_module, function_module = _load_reward_modules()
    tokenizer = _CharTokenizer()
    response = "direct answer B"
    response_ids = torch.tensor(tokenizer.encode(response))
    response_mask = torch.ones_like(response_ids, dtype=torch.float32)
    config = config_module.RewardConfig(enable_response_loss_weight_mask=True)

    weights = function_module.build_response_loss_weight_mask(response_ids, response_mask, response, tokenizer, config)

    assert torch.equal(weights, response_mask)
