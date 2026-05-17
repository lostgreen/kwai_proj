from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_metrics_module():
    old_modules = {
        name: sys.modules.get(name)
        for name in ("verl", "verl.trainer", "verl.protocol", "torch")
    }
    sys.modules["verl"] = types.ModuleType("verl")
    sys.modules["verl.trainer"] = types.ModuleType("verl.trainer")
    protocol = types.ModuleType("verl.protocol")
    protocol.DataProto = object
    sys.modules["verl.protocol"] = protocol
    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = type("Tensor", (), {})
    sys.modules["torch"] = torch_stub
    try:
        spec = importlib.util.spec_from_file_location(
            "verl.trainer.metrics_under_test",
            REPO_ROOT / "verl" / "trainer" / "metrics.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old in old_modules.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def test_metric_numeric_values_flattens_per_video_lists():
    metrics_mod = _load_metrics_module()

    values = metrics_mod._flatten_numeric_values(
        np.array(
            [
                [2.0, 0.5],
                1.0,
                [64, 32],
                None,
                "bad",
            ],
            dtype=object,
        )
    )

    assert values == [2.0, 0.5, 1.0, 64.0, 32.0]


def test_cot_budget_metrics_summarize_debug_flags_and_lengths():
    metrics_mod = _load_metrics_module()

    metrics = metrics_mod.compute_cot_budget_metrics(
        [
            {
                "cot_budget_enabled": True,
                "cot_start_detected": True,
                "cot_end_detected": True,
                "cot_repaired": False,
                "cot_text_fallback_used": False,
                "remaining_tokens": 12,
                "raw_token_len": 90,
                "final_token_len": 102,
                "max_cot_tokens": 128,
            },
            {
                "cot_budget_enabled": True,
                "cot_start_detected": True,
                "cot_end_detected": False,
                "cot_repaired": True,
                "cot_text_fallback_used": True,
                "remaining_tokens": 0,
                "raw_token_len": 256,
                "final_token_len": 256,
                "max_cot_tokens": 128,
            },
            None,
        ]
    )

    assert metrics["cot_budget/enabled_ratio"] == 1.0
    assert metrics["cot_budget/start_detected_ratio"] == 1.0
    assert metrics["cot_budget/end_detected_ratio"] == 0.5
    assert metrics["cot_budget/repaired_ratio"] == 0.5
    assert metrics["cot_budget/text_fallback_ratio"] == 0.5
    assert metrics["cot_budget/remaining_tokens_mean"] == 6.0
    assert metrics["cot_budget/final_token_len_max"] == 256.0


def test_cot_budget_metrics_return_empty_when_debug_absent():
    metrics_mod = _load_metrics_module()

    assert metrics_mod.compute_cot_budget_metrics([None, {}]) == {}
