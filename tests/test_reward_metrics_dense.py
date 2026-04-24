from __future__ import annotations

import sys
from pathlib import Path
import importlib.util

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_SPEC = importlib.util.spec_from_file_location(
    "reward_metrics_under_test",
    REPO_ROOT / "verl" / "workers" / "reward" / "metrics.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
build_dense_reward_metrics = _MODULE.build_dense_reward_metrics


def test_dense_reward_metrics_pad_task_specific_keys():
    scores = [
        {"overall": 1.0, "format": 1.0, "accuracy": 1.0},
        {"overall": 0.4, "format": 0.0, "accuracy": 0.4, "r_global": 0.3, "r_local": 0.5},
        {"overall": 0.0, "format": 0.0, "accuracy": 0.0},
    ]

    metrics = build_dense_reward_metrics(scores, batch_size=3)

    assert set(metrics) == {"overall", "format", "accuracy", "r_global", "r_local"}
    assert all(len(values) == 3 for values in metrics.values())
    assert metrics["r_global"] == [0.0, 0.3, 0.0]
    assert metrics["r_local"] == [0.0, 0.5, 0.0]


def test_dense_reward_metrics_fill_missing_scores():
    metrics = build_dense_reward_metrics([{"overall": 0.8}], batch_size=2)

    assert metrics["overall"] == [0.8, 0.0]
