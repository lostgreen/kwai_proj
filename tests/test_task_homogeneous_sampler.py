import sys
import types


if "codetiming" not in sys.modules:
    codetiming_stub = types.ModuleType("codetiming")
    codetiming_stub.Timer = object
    sys.modules["codetiming"] = codetiming_stub

if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_utils_stub = types.ModuleType("torch.utils")
    torch_data_stub = types.ModuleType("torch.utils.data")

    class Sampler:
        def __class_getitem__(cls, _item):
            return cls

    torch_data_stub.Sampler = Sampler
    torch_utils_stub.data = torch_data_stub
    torch_stub.utils = torch_utils_stub
    sys.modules["torch"] = torch_stub
    sys.modules["torch.utils"] = torch_utils_stub
    sys.modules["torch.utils.data"] = torch_data_stub

from verl.utils.task_grouping import resolve_task_homogeneous_bucket
from verl.utils.task_sampler import TaskHomogeneousBatchSampler


def test_opd_task_grouping_routes_base_aot_seg_logic_buckets():
    assert resolve_task_homogeneous_bucket("temporal_grounding", "opd_task_group") == "base"
    assert resolve_task_homogeneous_bucket("llava_mcq", "opd_task_group") == "base"
    assert resolve_task_homogeneous_bucket("seg_aot_action_v2t_3way", "opd_task_group") == "aot"
    assert resolve_task_homogeneous_bucket("seg_aot_event_t2v_binary", "opd_task_group") == "aot"
    assert resolve_task_homogeneous_bucket("temporal_seg_hier_L2", "opd_task_group") == "seg"
    assert resolve_task_homogeneous_bucket("event_logic_fill_blank", "opd_task_group") == "logic"
    assert resolve_task_homogeneous_bucket("event_logic_sort", "opd_task_group") == "logic"


def test_raw_task_grouping_preserves_problem_type():
    assert resolve_task_homogeneous_bucket("temporal_seg_hier_L2", "raw") == "temporal_seg_hier_L2"


def test_task_homogeneous_sampler_uses_opd_four_bucket_grouping():
    class DatasetWrapper:
        dataset = [
            {"problem_type": "temporal_grounding"},
            {"problem_type": "llava_mcq"},
            {"problem_type": "seg_aot_action_v2t_3way"},
            {"problem_type": "temporal_seg_hier_L1"},
            {"problem_type": "event_logic_sort"},
        ]

    sampler = TaskHomogeneousBatchSampler(
        DatasetWrapper(),
        batch_size=1,
        task_key="problem_type",
        task_grouping="opd_task_group",
        drop_last=False,
    )

    assert {name: len(indices) for name, indices in sampler.task_buckets.items()} == {
        "base": 2,
        "aot": 1,
        "seg": 1,
        "logic": 1,
    }
