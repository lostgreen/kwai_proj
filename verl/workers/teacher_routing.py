from collections.abc import Mapping, Sequence
from typing import Any, Optional


def _as_python_value(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return value
    return value


def _validate_teacher_name(teacher_name: str, teacher_names: Sequence[str], routing_value: str) -> str:
    if teacher_name not in teacher_names:
        raise ValueError(
            f"OPD teacher map routes {routing_value!r} to {teacher_name!r}, "
            f"but configured teachers are {sorted(teacher_names)}."
        )
    return teacher_name


def resolve_opd_teacher_name(
    routing_value: Any,
    teacher_names: Sequence[str],
    task_map: Optional[Mapping[str, str]] = None,
    default_teacher: Optional[str] = None,
) -> str:
    if not teacher_names:
        raise ValueError("No OPD teacher models are configured.")
    if len(teacher_names) == 1:
        return next(iter(teacher_names))

    teacher_names = tuple(teacher_names)
    value = _as_python_value(routing_value)
    task_name = "" if value is None else str(value)

    if task_map:
        if task_name in task_map:
            return _validate_teacher_name(task_map[task_name], teacher_names, task_name)
        for pattern, teacher_name in task_map.items():
            if pattern.endswith("*") and task_name.startswith(pattern[:-1]):
                return _validate_teacher_name(teacher_name, teacher_names, task_name)

    if task_name in teacher_names:
        return task_name

    normalized_task = task_name.lower().replace("-", "_")
    normalized_teachers = {teacher.lower().replace("-", "_"): teacher for teacher in teacher_names}

    if (
        "aot" in normalized_task
        or normalized_task in {"temporal_grounding", "tg", "grounding", "llava_mcq"}
    ) and "aot" in normalized_teachers:
        return normalized_teachers["aot"]

    event_teacher = normalized_teachers.get("eventlogic") or normalized_teachers.get("event_logic")
    if normalized_task.startswith("event_logic") and event_teacher is not None:
        return event_teacher

    seg_teacher = normalized_teachers.get("seg") or normalized_teachers.get("hier_seg")
    if (
        normalized_task.startswith("temporal_seg")
        or normalized_task.startswith("hier_seg")
        or normalized_task == "seg"
    ) and seg_teacher is not None:
        return seg_teacher

    if default_teacher:
        return _validate_teacher_name(default_teacher, teacher_names, task_name)

    raise ValueError(
        f"No OPD teacher configured for routing value {task_name!r}. "
        f"Configured teachers: {sorted(teacher_names)}. "
        "Set worker.ref.teacher_task_map or worker.ref.default_teacher to route this task."
    )
