from __future__ import annotations

import importlib.util
import inspect
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_core_algos_module():
    module_names = (
        "torch",
        "torch.nn",
        "torch.nn.functional",
        "verl",
        "verl.utils",
        "verl.utils.torch_functional",
        "verl.trainer",
    )
    old_modules = {name: sys.modules.get(name) for name in module_names}

    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = type("Tensor", (), {})
    torch_stub.FloatTensor = torch_stub.Tensor
    torch_stub.no_grad = lambda: (lambda fn: fn)
    torch_stub.is_tensor = lambda value: False

    nn_stub = types.ModuleType("torch.nn")
    functional_stub = types.ModuleType("torch.nn.functional")
    torch_stub.nn = nn_stub
    nn_stub.functional = functional_stub

    verl_stub = types.ModuleType("verl")
    utils_stub = types.ModuleType("verl.utils")
    torch_functional_stub = types.ModuleType("verl.utils.torch_functional")
    trainer_stub = types.ModuleType("verl.trainer")

    sys.modules["torch"] = torch_stub
    sys.modules["torch.nn"] = nn_stub
    sys.modules["torch.nn.functional"] = functional_stub
    sys.modules["verl"] = verl_stub
    sys.modules["verl.utils"] = utils_stub
    sys.modules["verl.utils.torch_functional"] = torch_functional_stub
    sys.modules["verl.trainer"] = trainer_stub

    try:
        spec = importlib.util.spec_from_file_location(
            "verl.trainer.core_algos_under_test",
            REPO_ROOT / "verl" / "trainer" / "core_algos.py",
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


def test_ema_grpo_task_key_groups_proxy_reward_families():
    core_algos = _load_core_algos_module()

    assert core_algos._task_key_of("temporal_grounding", None) == "tg"
    assert core_algos._task_key_of("temporal_seg_hier_L1", None) == "seg"
    assert core_algos._task_key_of("temporal_seg_hier_L3_seg", None) == "seg"

    assert core_algos._task_key_of("llava_mcq", None) == "mcq"
    assert core_algos._task_key_of("seg_aot_action_t2v_binary", None) == "mcq"
    assert core_algos._task_key_of("seg_aot_event_v2t_3way", None) == "mcq"
    assert core_algos._task_key_of("event_logic_predict_next", None) == "mcq"
    assert core_algos._task_key_of("event_logic_fill_blank", None) == "mcq"

    assert core_algos._task_key_of("event_logic_sort", None) == "sort"
    assert core_algos._task_key_of("segmentation", "video") == "segmentation/video"


def test_ema_grpo_uses_single_task_key_helper():
    core_algos = _load_core_algos_module()

    source = inspect.getsource(core_algos.compute_ema_grpo_outcome_advantage)
    assert "def _task_key_of(" not in source
