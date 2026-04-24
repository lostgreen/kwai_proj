from __future__ import annotations

from collections import defaultdict
from typing import Any


def coerce_reward_metric(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def build_dense_reward_metrics(scores: list[dict[str, Any]], batch_size: int) -> dict[str, list[float]]:
    metric_keys = set()
    for score in scores[:batch_size]:
        if isinstance(score, dict):
            metric_keys.update(score.keys())
    metric_keys.add("overall")

    reward_metrics = defaultdict(list)
    for i in range(batch_size):
        score = scores[i] if i < len(scores) and isinstance(scores[i], dict) else {}
        for key in metric_keys:
            reward_metrics[key].append(coerce_reward_metric(score.get(key, 0.0)))
    return reward_metrics
