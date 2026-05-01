import ast
import sys
import types
from pathlib import Path


if "codetiming" not in sys.modules:
    codetiming_stub = types.ModuleType("codetiming")
    codetiming_stub.Timer = object
    sys.modules["codetiming"] = codetiming_stub

from verl.trainer.config import PPOConfig
from verl.workers.config import ModelConfig


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


def test_multi_teacher_ref_models_do_not_replace_single_ref_with_actor_default():
    config = PPOConfig()
    config.worker.actor.model.model_path = "/models/student"
    config.worker.actor.micro_batch_size_per_device_for_experience = 3
    config.worker.ref.teacher_key = "problem_type"
    config.worker.ref.default_teacher = "seg"
    config.worker.ref.teacher_models = {
        "aot": ModelConfig(model_path="/models/aot"),
        "seg": ModelConfig(model_path="/models/seg"),
        "eventlogic": ModelConfig(model_path="/models/eventlogic", tokenizer_path="/tokenizers/eventlogic"),
    }

    config.deep_post_init()

    assert config.worker.ref.model.model_path is None
    assert set(config.worker.ref.teacher_models) == {"aot", "seg", "eventlogic"}
    assert config.worker.ref.teacher_models["aot"].tokenizer_path == "/models/aot"
    assert config.worker.ref.teacher_models["seg"].tokenizer_path == "/models/seg"
    assert config.worker.ref.teacher_models["eventlogic"].tokenizer_path == "/tokenizers/eventlogic"
    assert config.worker.ref.micro_batch_size_per_device_for_experience == 3
    assert config.worker.ref.teacher_key == "problem_type"
    assert config.worker.ref.default_teacher == "seg"


def test_teacher_topk_result_carries_temperature_meta_info():
    source = Path("verl/workers/fsdp_workers.py").read_text()
    module = ast.parse(source)
    fns = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef)
        and node.name in {"compute_ref_topk_log_probs", "_compute_ref_topk_log_probs_with_module"}
    ]

    data_proto_calls = [
        node
        for fn in fns
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "from_dict"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "DataProto"
    ]

    assert any(keyword.arg == "meta_info" for call in data_proto_calls for keyword in call.keywords)


def test_opd_metrics_path_does_not_require_reward_or_advantage_fields():
    metrics_source = Path("verl/trainer/metrics.py").read_text()
    metrics_module = ast.parse(metrics_source)
    fn = next(
        (
            node
            for node in ast.walk(metrics_module)
            if isinstance(node, ast.FunctionDef) and node.name == "compute_opd_data_metrics"
        ),
        None,
    )

    assert fn is not None
    string_constants = {
        node.value
        for node in ast.walk(fn)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "token_level_scores" not in string_constants
    assert "token_level_rewards" not in string_constants
    assert "advantages" not in string_constants
    assert "returns" not in string_constants

    trainer_source = Path("verl/trainer/ray_trainer.py").read_text()
    assert "compute_opd_data_metrics" in trainer_source


def test_fsdp_worker_has_multi_teacher_topk_routing_and_ref_offload_hooks():
    source = Path("verl/workers/fsdp_workers.py").read_text()

    assert "resolve_opd_teacher_name" in source
    assert "teacher_models" in source
    assert "ref_fsdp_modules" in source
    assert "ref_policies" in source
    assert "offload_fsdp_model(ref_module)" in source


def test_multi_teacher_opd_launcher_wires_three_teachers_and_cpu_offload():
    script = Path("local_scripts/run_multi_teacher_opd.sh").read_text()

    assert "AOT_TEACHER_MODEL_PATH" in script
    assert "SEG_TEACHER_MODEL_PATH" in script
    assert "EVENTLOGIC_TEACHER_MODEL_PATH" in script
    assert "composition_base_logic_el10k_mf256_ema/global_step_272/actor/huggingface" in script
    assert "composition_base_aot_logic_aot10k_el10k_mf256_ema/global_step_300" not in script
    assert "validate_opd_teacher_paths" in script
    assert "REF_OFFLOAD_PARAMS=\"${REF_OFFLOAD_PARAMS:-true}\"" in script
    assert "ACTOR_OFFLOAD_PARAMS=\"${ACTOR_OFFLOAD_PARAMS:-true}\"" in script
    assert "TASKS=\"${TASKS:-tg mcq hier_seg event_logic aot}\"" in script


def test_multi_teacher_cli_overrides_do_not_use_hydra_plus_prefix():
    runner = Path("local_scripts/run_multi_task.sh").read_text()

    assert "+worker.ref.teacher_models" not in runner
    assert 'worker.ref.teacher_models.aot.model_path="${AOT_TEACHER_MODEL_PATH}"' in runner
    assert 'worker.ref.teacher_models.seg.model_path="${SEG_TEACHER_MODEL_PATH}"' in runner
    assert 'worker.ref.teacher_models.eventlogic.model_path="${EVENTLOGIC_TEACHER_MODEL_PATH}"' in runner


def test_multi_teacher_opd_launcher_enables_homogeneous_batching_by_default():
    launcher = Path("local_scripts/run_multi_teacher_opd.sh").read_text()
    runner = Path("local_scripts/run_multi_task.sh").read_text()

    assert 'TASK_HOMOGENEOUS_BATCHING="${TASK_HOMOGENEOUS_BATCHING:-true}"' in launcher
    assert 'TASK_HOMOGENEOUS_GROUPING="${TASK_HOMOGENEOUS_GROUPING:-opd_task_group}"' in launcher
    assert 'data.task_homogeneous_batching="${TASK_HOMOGENEOUS_BATCHING}"' in runner
    assert 'data.task_homogeneous_grouping="${TASK_HOMOGENEOUS_GROUPING}"' in runner


def test_multi_teacher_opd_launcher_preserves_mf256_default():
    launcher = Path("local_scripts/run_multi_teacher_opd.sh").read_text()

    assert 'MAX_FRAMES="${MAX_FRAMES:-256}"' in launcher


def test_single_teacher_opd_launcher_defaults_to_base_aot_mf256_sanity_data():
    launcher = Path("local_scripts/run_single_teacher_opd.sh").read_text()

    assert (
        'TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/'
        "multi_task_4b_lr5e-7_kl0p01_entropy0p005_ablations/"
        'composition_base_aot_aot10k_mf256_ema/global_step_200/actor/huggingface}"'
    ) in launcher
    assert (
        'TRAIN_FILE="${TRAIN_FILE:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/multi_task/'
        'experiments/composition_base_aot_aot10k_mf256_ema/train.jsonl}"'
    ) in launcher
    assert (
        'TEST_FILE="${TEST_FILE:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/multi_task/'
        'experiments/composition_base_aot_aot10k_mf256_ema/val.jsonl}"'
    ) in launcher
    assert 'TASKS="${TASKS:-tg mcq aot}"' in launcher
    assert 'MAX_FRAMES="${MAX_FRAMES:-256}"' in launcher
    assert 'MAX_PIXELS="${MAX_PIXELS:-65536}"' in launcher
    assert 'TP_SIZE="${TP_SIZE:-1}"' in launcher
    assert 'ROLLOUT_BS="${ROLLOUT_BS:-16}"' in launcher
    assert 'GLOBAL_BS="${GLOBAL_BS:-16}"' in launcher
    assert 'ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"' in launcher
    assert 'OPD_TOPK="${OPD_TOPK:-10}"' in launcher
    assert 'SAVE_FREQ="${SAVE_FREQ:-50}"' in launcher
    assert 'SAVE_LIMIT="${SAVE_LIMIT:-3}"' in launcher
    assert 'MAX_STEPS="${MAX_STEPS:-50}"' in launcher


def test_multi_teacher_opd_launcher_defaults_to_full_composition_mf256_data():
    launcher = Path("local_scripts/run_multi_teacher_opd.sh").read_text()

    assert (
        'TRAIN_FILE="${TRAIN_FILE:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/multi_task/'
        'experiments/composition_base_seg_logic_aot_hier10k_el10k_aot10k_mf256_ema/train.jsonl}"'
    ) in launcher
    assert (
        'TEST_FILE="${TEST_FILE:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/multi_task/'
        'experiments/composition_base_seg_logic_aot_hier10k_el10k_aot10k_mf256_ema/val.jsonl}"'
    ) in launcher
    assert 'TASKS="${TASKS:-tg mcq hier_seg event_logic aot}"' in launcher
    assert 'MAX_FRAMES="${MAX_FRAMES:-256}"' in launcher
    assert 'MAX_PIXELS="${MAX_PIXELS:-65536}"' in launcher
    assert 'TP_SIZE="${TP_SIZE:-1}"' in launcher
    assert 'ROLLOUT_BS="${ROLLOUT_BS:-16}"' in launcher
    assert 'GLOBAL_BS="${GLOBAL_BS:-16}"' in launcher
    assert 'ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"' in launcher
    assert 'OPD_TOPK="${OPD_TOPK:-10}"' in launcher
    assert 'SAVE_FREQ="${SAVE_FREQ:-50}"' in launcher
    assert 'SAVE_LIMIT="${SAVE_LIMIT:-3}"' in launcher
    assert 'MAX_STEPS="${MAX_STEPS:-50}"' in launcher
    assert 'HIER_TARGET="${HIER_TARGET:-10000}"' in launcher
    assert 'EL_TRAIN="${EL_TRAIN:-${EL_HARDER_DATA}/train_10k.jsonl}"' in launcher
    assert 'EL_VAL_SOURCE="${EL_VAL_SOURCE:-${EL_HARDER_DATA}/val_logic.jsonl}"' in launcher
    assert 'EL_TARGET="${EL_TARGET:-10000}"' in launcher
    assert 'VAL_EL_N="${VAL_EL_N:-300}"' in launcher
    assert 'AOT_TARGET="${AOT_TARGET:-10000}"' in launcher
    assert 'VAL_AOT_N="${VAL_AOT_N:-300}"' in launcher


def test_opd_comparison_4b_mopd_defaults_to_batch64_and_unlimited_checkpoints():
    common = Path("local_scripts/opd_comparison/common.sh").read_text()
    launcher = Path("local_scripts/opd_comparison/run_mopd_4b_full_epoch.sh").read_text()
    launcher_8b = Path("local_scripts/opd_comparison/run_mopd_8b_from_4b_teachers.sh").read_text()
    runner = Path("local_scripts/run_multi_task.sh").read_text()

    mopd_defaults = common[
        common.index("opd_comparison_mopd_defaults()") : common.index("opd_comparison_validate_rollout_tokens()")
    ]

    assert 'MODEL_PATH="${MODEL_PATH:-${QWEN3_VL_4B_MODEL_PATH}}"' in launcher
    assert 'CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${CHECKPOINT_ROOT_4B_COMPARISON}}"' in launcher
    assert 'EVENTLOGIC_TEACHER_STEP="${EVENTLOGIC_TEACHER_STEP:-272}"' in common
    assert "composition_base_logic_el10k_mf256_ema" in common
    assert "validate_opd_teacher_paths" in common
    assert "validate_opd_teacher_paths" in launcher
    assert "validate_opd_teacher_paths" in launcher_8b
    assert 'ROLLOUT_BS="${ROLLOUT_BS:-64}"' in mopd_defaults
    assert 'GLOBAL_BS="${GLOBAL_BS:-64}"' in mopd_defaults
    assert 'VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}"' in mopd_defaults
    assert 'SAVE_FREQ="${SAVE_FREQ:-50}"' in common
    assert 'SAVE_LIMIT="${SAVE_LIMIT:--1}"' in common
    assert 'trainer.save_freq="${SAVE_FREQ}"' in runner
    assert 'trainer.save_limit="${SAVE_LIMIT}"' in runner


def test_multi_task_runner_checks_raw_sources_only_when_mix_is_needed():
    runner = Path("local_scripts/run_multi_task.sh").read_text()

    mix_gate_index = runner.index('if [[ "${NEEDS_MIX}" == "true" ]]; then')
    source_check_index = runner.index("    check \\")

    assert mix_gate_index < source_check_index
