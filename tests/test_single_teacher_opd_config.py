import ast
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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


def test_experiments_tree_has_teacher_and_opd_entrypoints_for_each_target_model():
    teacher_base = Path("video_proxy/experiments/teacher_train")
    opd_base = Path("video_proxy/experiments/opd")
    expected_models = {
        "qwen3_vl_4b": ("qwen3_vl", "4b", "Qwen3-VL-4B-Instruct"),
        "qwen3_vl_8b": ("qwen3_vl", "8b", "Qwen3-VL-8B-Instruct"),
        "qwen2_5_vl_3b": ("qwen2_5_vl", "3b", "Qwen2.5-VL-3B-Instruct"),
        "qwen2_5_vl_7b": ("qwen2_5_vl", "7b", "Qwen2.5-VL-7B-Instruct"),
    }

    for dirname, (family, size, model_name) in expected_models.items():
        teacher = (teacher_base / dirname / "run.sh").read_text()
        opd = (opd_base / dirname / "run.sh").read_text()

        assert f'MODEL_FAMILY="{family}"' in teacher
        assert f'MODEL_SIZE="{size}"' in teacher
        assert model_name in teacher
        assert "recipes/teacher_train_ema_grpo.sh" in teacher
        assert f'MODEL_FAMILY="{family}"' in opd
        assert f'MODEL_SIZE="{size}"' in opd
        assert model_name in opd
        assert "TEACHER_MODEL_PATH" in opd
        assert "recipes/opd_train.sh" in opd

    assert not Path("video_proxy/training/models").exists()


def test_multi_teacher_cli_overrides_do_not_use_hydra_plus_prefix():
    runner = Path("video_proxy/training/launchers/run_multi_task.sh").read_text()

    assert "+worker.ref.teacher_models" not in runner
    assert 'worker.ref.teacher_models."${_teacher_name}".model_path="${!_teacher_path_var}"' in runner
    assert 'worker.ref.teacher_models."${_teacher_name}".tokenizer_path="${!_teacher_tokenizer_var:-${!_teacher_path_var}}"' in runner
    assert 'worker.ref.teacher_models."${_teacher_name}".trust_remote_code="${!_teacher_trust_remote_code_var}"' in runner


def test_opd_recipe_supports_multi_teacher_paths_and_homogeneous_batching_runner():
    recipe = Path("video_proxy/training/recipes/opd_train.sh").read_text()
    runner = Path("video_proxy/training/launchers/run_multi_task.sh").read_text()

    assert "AOT_TEACHER_MODEL_PATH" in recipe
    assert "TEACHER_MODEL_PATH" in recipe
    assert 'data.task_homogeneous_batching="${TASK_HOMOGENEOUS_BATCHING}"' in runner
    assert 'data.task_homogeneous_grouping="${TASK_HOMOGENEOUS_GROUPING}"' in runner


def test_opd_recipe_preserves_mf256_default():
    launcher = Path("video_proxy/training/recipes/opd_train.sh").read_text()

    assert 'MAX_FRAMES="${MAX_FRAMES:-256}"' in launcher
    assert 'MAX_STEPS="${MAX_STEPS-50}"' in launcher


def test_qwen3_4b_opd_entrypoint_defaults_to_teacher_checkpoint_and_sanity_settings():
    launcher = Path("video_proxy/experiments/opd/qwen3_vl_4b/run.sh").read_text()
    recipe = Path("video_proxy/training/recipes/opd_train.sh").read_text()

    assert (
        'TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/'
        'multi_task/qwen3_vl_4b_teacher_ema_grpo/global_step_200/actor/huggingface}"'
    ) in launcher
    assert 'TASKS="${TASKS:-tg mcq aot}"' in recipe
    assert 'MAX_FRAMES="${MAX_FRAMES:-256}"' in recipe
    assert 'MAX_PIXELS="${MAX_PIXELS:-65536}"' in recipe
    assert 'TP_SIZE="${TP_SIZE:-1}"' in recipe
    assert 'ROLLOUT_BS="${ROLLOUT_BS:-16}"' in recipe
    assert 'GLOBAL_BS="${GLOBAL_BS:-16}"' in recipe
    assert 'ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"' in recipe
    assert 'OPD_TOPK="${OPD_TOPK:-10}"' in recipe
    assert 'SAVE_FREQ="${SAVE_FREQ:-50}"' in recipe
    assert 'SAVE_LIMIT="${SAVE_LIMIT:-3}"' in recipe
    assert 'MAX_STEPS="${MAX_STEPS-50}"' in recipe


def test_opd_full_epoch_presets_keep_batch64_and_checkpoint_controls():
    launcher = Path("video_proxy/experiments/opd/qwen3_vl_4b/run.sh").read_text()
    launcher_8b = Path("video_proxy/experiments/opd/qwen3_vl_8b/run.sh").read_text()
    runner = Path("video_proxy/training/launchers/run_multi_task.sh").read_text()
    rollout_config = Path("verl/workers/rollout/config.py").read_text()
    rollout_impl = Path("verl/workers/rollout/vllm_rollout_spmd.py").read_text()

    assert 'MODEL_PATH="${MODEL_PATH:-/m2v_intern/xuboshen/models/Qwen3-VL-4B-Instruct}"' in launcher
    assert 'EXP_NAME="${EXP_NAME:-qwen3_vl_4b_opd}"' in launcher
    assert 'MODEL_PATH="${MODEL_PATH:-/m2v_intern/xuboshen/models/Qwen3-VL-8B-Instruct}"' in launcher_8b
    assert 'EXP_NAME="${EXP_NAME:-qwen3_vl_8b_opd}"' in launcher_8b
    assert 'SAVE_LIMIT="${SAVE_LIMIT:-1}"' in launcher_8b
    assert 'SAVE_BEST="${SAVE_BEST:-true}"' in launcher_8b
    assert 'ROLLOUT_BS="${ROLLOUT_BS:-64}"' in launcher
    assert 'GLOBAL_BS="${GLOBAL_BS:-64}"' in launcher
    assert 'VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}"' in launcher
    assert 'ENABLE_GPU_FILLER="${ENABLE_GPU_FILLER:-false}"' in launcher_8b
    assert "FILLER_GPUS" not in launcher_8b
    assert 'ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-64}"' in launcher
    assert 'SAVE_FREQ="${SAVE_FREQ:-50}"' in launcher
    assert 'SAVE_LIMIT="${SAVE_LIMIT:--1}"' in launcher
    assert 'trainer.save_freq="${SAVE_FREQ}"' in runner
    assert 'trainer.save_limit="${SAVE_LIMIT}"' in runner
    assert 'worker.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}"' in runner
    assert "max_num_seqs: int = 1024" in rollout_config
    assert "max_num_seqs=config.max_num_seqs" in rollout_impl


def test_mopd_entrypoints_restore_three_and_two_teacher_presets_for_each_target_model():
    opd_base = Path("video_proxy/experiments/opd")
    common = (opd_base / "common_mopd.sh").read_text()
    expected_models = {
        "qwen3_vl_4b": ("qwen3_vl", "4b", "Qwen3-VL-4B-Instruct"),
        "qwen3_vl_8b": ("qwen3_vl", "8b", "Qwen3-VL-8B-Instruct"),
        "qwen2_5_vl_3b": ("qwen2_5_vl", "3b", "Qwen2.5-VL-3B-Instruct"),
        "qwen2_5_vl_7b": ("qwen2_5_vl", "7b", "Qwen2.5-VL-7B-Instruct"),
    }

    assert 'FULL_COMPOSITION_EXP_NAME="${FULL_COMPOSITION_EXP_NAME:-composition_base_seg_logic_aot_hier10k_el10k_aot10k_mf256_ema}"' in common
    assert 'BASE_R1_R2_COMPOSITION_EXP_NAME="${BASE_R1_R2_COMPOSITION_EXP_NAME:-composition_base_seg_aot_hier10k_aot10k_mf256_ema}"' in common
    assert 'TASKS="${TASKS:-tg mcq hier_seg event_logic aot}"' in common
    assert 'TASKS="${TASKS:-tg mcq hier_seg aot}"' in common
    assert 'OPD_TEACHER_SET="${OPD_TEACHER_SET:-aot seg eventlogic}"' in common
    assert 'OPD_TEACHER_SET="${OPD_TEACHER_SET:-aot seg}"' in common
    assert "validate_mopd_teacher_paths" in common

    for dirname, (family, size, model_name) in expected_models.items():
        three_teacher = (opd_base / dirname / "run_mopd_3teachers.sh").read_text()
        two_teacher = (opd_base / dirname / "run_mopd_2teachers.sh").read_text()

        assert f'MODEL_FAMILY="{family}"' in three_teacher
        assert f'MODEL_SIZE="{size}"' in three_teacher
        assert model_name in three_teacher
        assert "mopd_full_composition_data_defaults" in three_teacher
        assert "mopd_three_teacher_defaults" in three_teacher
        assert "source \"${SCRIPT_DIR}/../common_mopd.sh\"" in three_teacher
        assert "recipes/opd_train.sh" in three_teacher

        assert f'MODEL_FAMILY="{family}"' in two_teacher
        assert f'MODEL_SIZE="{size}"' in two_teacher
        assert model_name in two_teacher
        assert "mopd_base_r1_r2_data_defaults" in two_teacher
        assert "mopd_two_teacher_defaults" in two_teacher
        assert "EVENTLOGIC_TEACHER_MODEL_PATH" not in two_teacher
        assert "source \"${SCRIPT_DIR}/../common_mopd.sh\"" in two_teacher
        assert "recipes/opd_train.sh" in two_teacher


def test_qwen3_4b_and_8b_mopd_presets_keep_pre_refactor_paths_and_batch_settings():
    common = Path("video_proxy/experiments/opd/common_mopd.sh").read_text()
    qwen3_4b = Path("video_proxy/experiments/opd/qwen3_vl_4b/run_mopd_3teachers.sh").read_text()
    qwen3_4b_base_r1_r2 = Path("video_proxy/experiments/opd/qwen3_vl_4b/run_mopd_2teachers.sh").read_text()
    qwen3_8b = Path("video_proxy/experiments/opd/qwen3_vl_8b/run_mopd_3teachers.sh").read_text()

    assert (
        'MOPD_TEACHER_CKPT_ROOT="${MOPD_TEACHER_CKPT_ROOT:-/m2v_intern/xuboshen/zgw/'
        'RL-Models/VideoProxyMixed/multi_task_4b_lr5e-7_kl0p01_entropy0p005_ablations}"'
    ) in common
    assert 'AOT_TEACHER_STEP="${AOT_TEACHER_STEP:-200}"' in common
    assert 'SEG_TEACHER_STEP="${SEG_TEACHER_STEP:-250}"' in common
    assert 'EVENTLOGIC_TEACHER_STEP="${EVENTLOGIC_TEACHER_STEP:-272}"' in common
    assert "composition_base_aot_aot10k_mf256_ema" in common
    assert "composition_base_seg_hier10k_mf256_ema" in common
    assert "composition_base_logic_el10k_mf256_ema" in common

    assert 'EXP_NAME="${EXP_NAME:-mopd_qwen3vl4b_full_comp_4b_teachers_bs64_mf256_epoch1_save50}"' in qwen3_4b
    assert 'CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${MOPD_CHECKPOINT_ROOT_4B}}"' in qwen3_4b
    assert 'TP_SIZE="${TP_SIZE:-1}"' in qwen3_4b

    assert (
        'EXP_NAME="${EXP_NAME:-mopd_qwen3vl4b_base_r1_r2_4b_teachers_bs64_mf256_epoch1_save50_keep1}"'
        in qwen3_4b_base_r1_r2
    )
    assert 'SAVE_LIMIT="${SAVE_LIMIT:-1}"' in qwen3_4b_base_r1_r2
    assert 'SAVE_BEST="${SAVE_BEST:-true}"' in qwen3_4b_base_r1_r2

    assert (
        'EXP_NAME="${EXP_NAME:-mopd_qwen3vl8b_full_comp_4b_teachers_bs64_mf256_epoch1_save50_keep1}"'
        in qwen3_8b
    )
    assert 'CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${MOPD_CHECKPOINT_ROOT_8B}}"' in qwen3_8b
    assert 'TP_SIZE="${TP_SIZE:-2}"' in qwen3_8b
    assert 'ENABLE_GPU_FILLER="${ENABLE_GPU_FILLER:-false}"' in qwen3_8b

    for text in (common, qwen3_4b, qwen3_8b):
        assert 'N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"' in common
        assert 'ROLLOUT_BS="${ROLLOUT_BS:-64}"' in common
        assert 'GLOBAL_BS="${GLOBAL_BS:-64}"' in common
        assert 'VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}"' in common
        assert 'ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-64}"' in common
        assert 'MAX_FRAMES="${MAX_FRAMES:-256}"' in common
        assert 'MAX_PIXELS="${MAX_PIXELS:-65536}"' in common
        assert 'SAVE_FREQ="${SAVE_FREQ:-50}"' in common
    assert 'SAVE_LIMIT="${SAVE_LIMIT:--1}"' in common
    assert 'MAX_STEPS=""' in common
    assert 'MAX_STEPS="${MAX_STEPS-50}"' in Path("video_proxy/training/recipes/opd_train.sh").read_text()


def test_qwen2_5_vl_7b_mopd_defaults_use_its_task_teacher_checkpoints():
    three_teacher = Path("video_proxy/experiments/opd/qwen2_5_vl_7b/run_mopd_3teachers.sh").read_text()
    two_teacher = Path("video_proxy/experiments/opd/qwen2_5_vl_7b/run_mopd_2teachers.sh").read_text()

    expected_aot = (
        "/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/multi_task/"
        "qwen2_5_vl_7b_aot_teacher_nocot/global_step_250/actor/huggingface"
    )
    expected_seg = (
        "/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/multi_task/"
        "qwen2_5_vl_7b_seg_teacher_nocot/global_step_272/actor/huggingface"
    )
    expected_logic = (
        "/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/multi_task/"
        "qwen2_5_vl_7b_logic_teacher_nocot/global_step_250/actor/huggingface"
    )

    assert f'AOT_TEACHER_MODEL_PATH="${{AOT_TEACHER_MODEL_PATH:-{expected_aot}}}"' in three_teacher
    assert f'SEG_TEACHER_MODEL_PATH="${{SEG_TEACHER_MODEL_PATH:-{expected_seg}}}"' in three_teacher
    assert (
        f'EVENTLOGIC_TEACHER_MODEL_PATH="${{EVENTLOGIC_TEACHER_MODEL_PATH:-{expected_logic}}}"'
        in three_teacher
    )
    assert f'AOT_TEACHER_MODEL_PATH="${{AOT_TEACHER_MODEL_PATH:-{expected_aot}}}"' in two_teacher
    assert f'SEG_TEACHER_MODEL_PATH="${{SEG_TEACHER_MODEL_PATH:-{expected_seg}}}"' in two_teacher
    assert "EVENTLOGIC_TEACHER_MODEL_PATH" not in two_teacher
    assert 'TP_SIZE="${TP_SIZE:-2}"' in three_teacher
    assert 'TP_SIZE="${TP_SIZE:-2}"' in two_teacher


def test_qwen2_5_vl_7b_mopd_has_2gpu_debug_and_8gpu_train_wrappers():
    base = Path("video_proxy/experiments/opd/qwen2_5_vl_7b")
    debug = (base / "run_debug_2gpu.sh").read_text()
    train = (base / "run_train_8gpu.sh").read_text()

    for launcher in (debug, train):
        assert 'target="${1:-3teachers}"' in launcher
        assert '3teachers|run_mopd_3teachers.sh) target_script="${SCRIPT_DIR}/run_mopd_3teachers.sh" ;;' in launcher
        assert '2teachers|run_mopd_2teachers.sh) target_script="${SCRIPT_DIR}/run_mopd_2teachers.sh" ;;' in launcher
        assert 'exec bash "${target_script}" "$@"' in launcher

    assert 'N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-2}"' in debug
    assert 'TP_SIZE="${TP_SIZE:-2}"' in debug
    assert 'ROLLOUT_BS="${ROLLOUT_BS:-8}"' in debug
    assert 'GLOBAL_BS="${GLOBAL_BS:-${ROLLOUT_BS}}"' in debug
    assert 'VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"' in debug
    assert "MAX_STEPS=" not in debug
    assert 'SAVE_FREQ="${SAVE_FREQ:-10}"' in debug
    assert 'VAL_FREQ="${VAL_FREQ:-10}"' in debug
    assert 'DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"' in debug
    assert 'ENABLE_GPU_FILLER="${ENABLE_GPU_FILLER:-false}"' in debug
    assert 'POST_TRAIN_OCCUPANCY="${POST_TRAIN_OCCUPANCY:-false}"' in debug

    assert 'N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"' in train
    assert 'TP_SIZE="${TP_SIZE:-2}"' in train
    assert 'ROLLOUT_BS="${ROLLOUT_BS:-64}"' in train
    assert 'GLOBAL_BS="${GLOBAL_BS:-64}"' in train
    assert 'VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}"' in train


def test_multi_teacher_runner_builds_teacher_args_from_configured_teacher_set():
    runner = Path("video_proxy/training/launchers/run_multi_task.sh").read_text()

    assert 'read -r -a OPD_TEACHER_SET_EFFECTIVE <<< "${OPD_TEACHER_SET}"' in runner
    assert 'for _teacher_name in "${OPD_TEACHER_SET_EFFECTIVE[@]}"; do' in runner
    assert '_teacher_prefix="$(printf \'%s\' "${_teacher_name}" | tr \'[:lower:]\' \'[:upper:]\')"' in runner
    assert 'worker.ref.teacher_models."${_teacher_name}".model_path="${!_teacher_path_var}"' in runner
    assert 'worker.ref.teacher_models."${_teacher_name}".tokenizer_path="${!_teacher_tokenizer_var:-${!_teacher_path_var}}"' in runner
    assert 'worker.ref.teacher_models."${_teacher_name}".trust_remote_code="${!_teacher_trust_remote_code_var}"' in runner
    assert "for _required_teacher_var in AOT_TEACHER_MODEL_PATH SEG_TEACHER_MODEL_PATH EVENTLOGIC_TEACHER_MODEL_PATH" not in runner


def test_grpo_baseline_does_not_enable_opd_teachers():
    launcher = Path("video_proxy/experiments/baselines/grpo/qwen3_vl_4b/run.sh").read_text()

    assert 'TRAINING_MODE="rl"' in launcher
    assert 'ADV_ESTIMATOR="${ADV_ESTIMATOR:-ema_grpo}"' in launcher
    assert 'AOT_TEACHER_MODEL_PATH=""' in launcher
    assert 'SEG_TEACHER_MODEL_PATH=""' in launcher
    assert 'EVENTLOGIC_TEACHER_MODEL_PATH=""' in launcher
    assert 'ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-512}"' in launcher
    assert "recipes/teacher_train_ema_grpo.sh" not in launcher
    assert "recipes/opd_train.sh" not in launcher
    assert "launchers/run_multi_task.sh" in launcher


def test_multi_task_runner_checks_raw_sources_only_when_mix_is_needed():
    runner = Path("video_proxy/training/launchers/run_multi_task.sh").read_text()

    mix_gate_index = runner.index('if [[ "${NEEDS_MIX}" == "true" ]]; then')
    source_check_index = runner.index("    check \\")

    assert mix_gate_index < source_check_index


def test_multi_task_common_defines_rollout_limits_used_by_runner():
    common = Path("video_proxy/training/common/multi_task_common.sh").read_text()
    runner = Path("video_proxy/training/launchers/run_multi_task.sh").read_text()

    assert 'ROLLOUT_MAX_BATCHED_TOKENS="${ROLLOUT_MAX_BATCHED_TOKENS:-20480}"' in common
    assert 'ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-512}"' in common
    assert 'worker.rollout.max_num_batched_tokens="${ROLLOUT_MAX_BATCHED_TOKENS}"' in runner
    assert 'worker.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}"' in runner


def test_legacy_ablation_tree_removed_from_training_workspace():
    assert not Path("video_proxy/experiments/ablations").exists()
