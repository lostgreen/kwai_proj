from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


class _FakeScalar:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


class _FakeLengths:
    def __init__(self, values):
        self._values = values

    def __getitem__(self, index):
        return _FakeScalar(self._values[index])


class _FakeTensor:
    def __init__(self, rows):
        self.rows = rows
        self.assigned = {}

    def __getitem__(self, item):
        if isinstance(item, tuple):
            return self.assigned[item]
        if isinstance(item, int):
            return self.rows[item]
        row, col = item
        if isinstance(col, slice):
            return self.rows[row][col]
        return self.rows[row][col]

    def __setitem__(self, item, value):
        self.assigned[item] = value


class _FakeTorch(types.ModuleType):
    Tensor = _FakeTensor

    def zeros_like(self, tensor, dtype=None):
        rows = [[0.0 for _ in row] for row in tensor.rows]
        return _FakeTensor(rows)

    def sum(self, tensor, dim=-1):
        return _FakeLengths([sum(row) for row in tensor.rows])


class _FakeData:
    def __init__(self):
        self.batch = {
            "responses": _FakeTensor([[101, 102, 0]]),
            "response_mask": _FakeTensor([[1, 1, 0]]),
        }
        self.non_tensor_batch = {
            "ground_truth": np.array(["A"], dtype=object),
            "data_type": np.array(["video"], dtype=object),
            "problem_type": np.array(["event_logic_predict_next"], dtype=object),
            "cot_budget_debug": np.array(
                [{"cot_budget_enabled": True, "cot_repaired": True}],
                dtype=object,
            ),
        }

    def __len__(self):
        return 1


class _TinyTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return "decoded-response"


def _load_reward_modules():
    module_names = (
        "torch",
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

    torch_stub = _FakeTorch("torch")
    torch_stub.float32 = "float32"

    transformers_stub = types.ModuleType("transformers")
    transformers_stub.PreTrainedTokenizer = object

    verl_stub = types.ModuleType("verl")
    protocol_stub = types.ModuleType("verl.protocol")
    protocol_stub.DataProto = object
    workers_stub = types.ModuleType("verl.workers")
    reward_pkg_stub = types.ModuleType("verl.workers.reward")

    sys.modules["torch"] = torch_stub
    sys.modules["transformers"] = transformers_stub
    sys.modules["verl"] = verl_stub
    sys.modules["verl.protocol"] = protocol_stub
    sys.modules["verl.workers"] = workers_stub
    sys.modules["verl.workers.reward"] = reward_pkg_stub

    try:
        config_spec = importlib.util.spec_from_file_location(
            "verl.workers.reward.config",
            REPO_ROOT / "verl" / "workers" / "reward" / "config.py",
        )
        assert config_spec is not None and config_spec.loader is not None
        config_module = importlib.util.module_from_spec(config_spec)
        sys.modules[config_spec.name] = config_module
        config_spec.loader.exec_module(config_module)

        metrics_spec = importlib.util.spec_from_file_location(
            "verl.workers.reward.metrics",
            REPO_ROOT / "verl" / "workers" / "reward" / "metrics.py",
        )
        assert metrics_spec is not None and metrics_spec.loader is not None
        metrics_module = importlib.util.module_from_spec(metrics_spec)
        sys.modules[metrics_spec.name] = metrics_module
        metrics_spec.loader.exec_module(metrics_module)

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


def test_batch_reward_manager_passes_cot_budget_debug(tmp_path):
    config_module, function_module = _load_reward_modules()
    reward_file = tmp_path / "cot_debug_reward.py"
    reward_file.write_text(
        """
def compute_score(reward_inputs):
    results = []
    for item in reward_inputs:
        debug = item.get("cot_budget_debug") or {}
        repaired = bool(debug.get("cot_repaired", False))
        results.append({
            "overall": 1.0 if repaired else 0.0,
            "format": 0.0,
            "accuracy": 1.0,
        })
    return results
""",
        encoding="utf-8",
    )

    config = config_module.RewardConfig(
        reward_type="batch",
        reward_function=f"{reward_file}:compute_score",
    )
    config.post_init()
    manager = function_module.BatchFunctionRewardManager(config, _TinyTokenizer())

    reward_tensor, reward_metrics = manager.compute_reward(_FakeData())

    assert reward_tensor[0, 1] == 1.0
    assert reward_metrics["overall"] == [1.0]
