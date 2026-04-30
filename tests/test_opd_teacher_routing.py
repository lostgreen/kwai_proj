import sys
import types

import pytest


if "codetiming" not in sys.modules:
    codetiming_stub = types.ModuleType("codetiming")
    codetiming_stub.Timer = object
    sys.modules["codetiming"] = codetiming_stub

from verl.workers.teacher_routing import resolve_opd_teacher_name


def test_resolve_opd_teacher_name_routes_known_video_tasks():
    teacher_names = ["aot", "seg", "eventlogic"]

    assert resolve_opd_teacher_name("seg_aot_action_v2t_3way", teacher_names) == "aot"
    assert resolve_opd_teacher_name("temporal_grounding", teacher_names) == "aot"
    assert resolve_opd_teacher_name("llava_mcq", teacher_names) == "aot"
    assert resolve_opd_teacher_name("temporal_seg_hier_L2", teacher_names) == "seg"
    assert resolve_opd_teacher_name("event_logic_sort", teacher_names) == "eventlogic"


def test_resolve_opd_teacher_name_supports_exact_and_prefix_overrides():
    teacher_names = ["aot_teacher", "seg_teacher", "logic_teacher"]
    task_map = {
        "seg_aot_*": "aot_teacher",
        "temporal_seg_hier_L3_seg": "seg_teacher",
        "event_logic_*": "logic_teacher",
    }

    assert resolve_opd_teacher_name("seg_aot_event_dir_binary", teacher_names, task_map) == "aot_teacher"
    assert resolve_opd_teacher_name("temporal_seg_hier_L3_seg", teacher_names, task_map) == "seg_teacher"
    assert resolve_opd_teacher_name("event_logic_fill_blank", teacher_names, task_map) == "logic_teacher"


def test_resolve_opd_teacher_name_fails_closed_for_unknown_multi_teacher_task():
    with pytest.raises(ValueError, match="No OPD teacher configured"):
        resolve_opd_teacher_name("unknown_task", ["aot", "seg", "eventlogic"])
