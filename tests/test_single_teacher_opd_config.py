import ast
from pathlib import Path
import sys
import types


if "codetiming" not in sys.modules:
    codetiming_stub = types.ModuleType("codetiming")
    codetiming_stub.Timer = object
    sys.modules["codetiming"] = codetiming_stub

from verl.trainer.config import PPOConfig


def test_ref_model_defaults_to_actor_model_for_existing_kl_runs():
    config = PPOConfig()
    config.worker.actor.model.model_path = "/models/student"
    config.worker.actor.model.tokenizer_path = "/models/student-tokenizer"
    config.worker.actor.model.trust_remote_code = False
    config.worker.actor.model.freeze_vision_tower = True

    config.deep_post_init()

    assert config.worker.ref.model is not config.worker.actor.model
    assert config.worker.ref.model.model_path == "/models/student"
    assert config.worker.ref.model.tokenizer_path == "/models/student-tokenizer"
    assert config.worker.ref.model.trust_remote_code is False
    assert config.worker.ref.model.freeze_vision_tower is True


def test_ref_model_can_be_overridden_for_single_teacher_opd():
    config = PPOConfig()
    config.worker.actor.model.model_path = "/models/student"
    config.worker.actor.model.tokenizer_path = "/models/student-tokenizer"
    config.worker.ref.model.model_path = "/models/teacher"
    config.worker.ref.model.tokenizer_path = "/models/teacher-tokenizer"
    config.worker.ref.model.trust_remote_code = True

    config.deep_post_init()

    assert config.worker.actor.model.model_path == "/models/student"
    assert config.worker.ref.model.model_path == "/models/teacher"
    assert config.worker.ref.model.tokenizer_path == "/models/teacher-tokenizer"
    assert config.worker.ref.model.trust_remote_code is True


def test_opd_mode_syncs_distillation_knobs_to_actor():
    config = PPOConfig()
    config.algorithm.training_mode = "opd"
    config.algorithm.opd_topk = 16
    config.algorithm.opd_kl_coef = 0.7

    config.deep_post_init()

    assert config.worker.actor.opd_enabled is True
    assert config.worker.actor.opd_topk == 16
    assert config.worker.actor.opd_kl_coef == 0.7


def test_teacher_topk_result_carries_temperature_meta_info():
    source = Path("verl/workers/fsdp_workers.py").read_text()
    module = ast.parse(source)
    fn = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef) and node.name == "compute_ref_topk_log_probs"
    )

    data_proto_calls = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "from_dict"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "DataProto"
    ]

    assert any(keyword.arg == "meta_info" for call in data_proto_calls for keyword in call.keywords)
