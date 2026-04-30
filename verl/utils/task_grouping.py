from typing import Any


def _as_task_name(value: Any) -> str:
    if hasattr(value, "item"):
        try:
            value = value.item()
        except ValueError:
            pass
    return "" if value is None else str(value)


def resolve_task_homogeneous_bucket(task_value: Any, grouping: str | None = "raw") -> str:
    task_name = _as_task_name(task_value)
    mode = (grouping or "raw").strip().lower()
    if mode in {"raw", "problem_type", "none"}:
        return task_name
    if mode != "opd_task_group":
        raise ValueError(f"Unsupported task homogeneous grouping: {grouping!r}")

    normalized_task = task_name.lower().replace("-", "_")
    if normalized_task in {"base", "temporal_grounding", "tg", "grounding", "llava_mcq", "mcq"}:
        return "base"
    if "aot" in normalized_task:
        return "aot"
    if normalized_task.startswith("temporal_seg") or normalized_task.startswith("hier_seg") or normalized_task == "seg":
        return "seg"
    if normalized_task.startswith("event_logic") or normalized_task in {"eventlogic", "logic"}:
        return "logic"
    return task_name
