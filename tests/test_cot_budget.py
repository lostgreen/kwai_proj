from __future__ import annotations

import math
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


_MODULE_PATH = Path(__file__).resolve().parents[1] / "verl" / "workers" / "rollout" / "cot_budget.py"
_SPEC = importlib.util.spec_from_file_location("cot_budget", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
cot_budget = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = cot_budget
_SPEC.loader.exec_module(cot_budget)

CoTBudgetController = cot_budget.CoTBudgetController
CoTBudgetProcessor = cot_budget.CoTBudgetProcessor
make_cot_budget_processor = cot_budget.make_cot_budget_processor


def test_controller_forces_custom_thought_end_after_budget():
    controller = CoTBudgetController(
        start_token_ids=[10],
        end_token_ids=[20],
        max_tokens=3,
    )

    assert controller.next_forced_token([1, 10, 101, 102]) is None
    assert controller.next_forced_token([1, 10, 101, 102, 103]) == 20


def test_controller_supports_multi_token_end_sequence():
    controller = CoTBudgetController(
        start_token_ids=[10],
        end_token_ids=[20, 21, 22],
        max_tokens=2,
    )

    assert controller.next_forced_token([10, 100, 101]) == 20
    assert controller.next_forced_token([10, 100, 101, 20]) == 21
    assert controller.next_forced_token([10, 100, 101, 20, 21]) == 22
    assert controller.next_forced_token([10, 100, 101, 20, 21, 22]) is None


def test_controller_does_not_continue_partial_end_before_budget():
    controller = CoTBudgetController(
        start_token_ids=[10],
        end_token_ids=[20, 21],
        max_tokens=8,
    )

    assert controller.next_forced_token([10, 100, 20]) is None


def test_controller_does_not_force_when_end_was_generated_naturally():
    controller = CoTBudgetController(
        start_token_ids=[10],
        end_token_ids=[20],
        max_tokens=2,
    )

    assert controller.next_forced_token([10, 100, 20, 300, 301]) is None


def test_controller_counts_after_latest_unclosed_start():
    controller = CoTBudgetController(
        start_token_ids=[10],
        end_token_ids=[20],
        max_tokens=2,
    )

    assert controller.next_forced_token([10, 100, 20, 999, 10, 101]) is None
    assert controller.next_forced_token([10, 100, 20, 999, 10, 101, 102]) == 20


def test_processor_masks_logits_to_forced_token():
    controller = CoTBudgetController(
        start_token_ids=[10],
        end_token_ids=[20],
        max_tokens=1,
    )
    processor = CoTBudgetProcessor(controller)

    logits = [0.0, 0.1, 0.2, 0.3]
    masked = processor([10, 100], logits)

    assert masked[:4] == [-math.inf, -math.inf, -math.inf, -math.inf]
    assert masked[20] == 0.0


def test_processor_accepts_prompt_aware_vllm_signature():
    controller = CoTBudgetController(
        start_token_ids=[10],
        end_token_ids=[20],
        max_tokens=1,
    )
    processor = CoTBudgetProcessor(controller)

    masked = processor([777], [10, 100], [0.0, 0.1])

    assert masked[20] == 0.0


def test_controller_rejects_empty_boundaries():
    with pytest.raises(ValueError, match="start_token_ids"):
        CoTBudgetController(start_token_ids=[], end_token_ids=[20], max_tokens=3)
    with pytest.raises(ValueError, match="end_token_ids"):
        CoTBudgetController(start_token_ids=[10], end_token_ids=[], max_tokens=3)
    with pytest.raises(ValueError, match="max_tokens"):
        CoTBudgetController(start_token_ids=[10], end_token_ids=[20], max_tokens=0)


class _FakeTokenizer:
    def __init__(self):
        self.calls = []

    def encode(self, text, add_special_tokens=False):
        self.calls.append((text, add_special_tokens))
        mapping = {
            "<think>": [10],
            "</think>": [20],
            "<thought>": [30, 31],
            "</thought>": [40, 41],
        }
        return mapping[text]


def test_make_processor_uses_configurable_tags():
    tokenizer = _FakeTokenizer()

    processor = make_cot_budget_processor(
        tokenizer,
        start_token="<thought>",
        end_token="</thought>",
        max_tokens=8,
    )

    assert processor.controller.start_token_ids == [30, 31]
    assert processor.controller.end_token_ids == [40, 41]
    assert processor.controller.max_tokens == 8
    assert tokenizer.calls == [("<thought>", False), ("</thought>", False)]


def test_rollout_config_defaults_disable_cot_budget():
    module_path = Path(__file__).resolve().parents[1] / "verl" / "workers" / "rollout" / "config.py"
    spec = importlib.util.spec_from_file_location("rollout_config", module_path)
    assert spec is not None and spec.loader is not None
    rollout_config = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = rollout_config
    spec.loader.exec_module(rollout_config)

    config = rollout_config.RolloutConfig()

    assert config.cot_budget_enabled is False
    assert config.cot_budget_start_token == "<think>"
    assert config.cot_budget_end_token == "</think>"
    assert config.cot_budget_max_tokens == 0


def test_vllm_rollout_wires_cot_processor_into_sampling_params():
    source = (Path(__file__).resolve().parents[1] / "verl" / "workers" / "rollout" / "vllm_rollout_spmd.py").read_text()

    assert "make_cot_budget_processor" in source
    assert "cot_budget_enabled" in source
    assert '"logits_processors"' in source


def test_multi_task_launcher_exposes_cot_budget_flags():
    source = (
        Path(__file__).resolve().parents[1] / "video_proxy" / "training" / "launchers" / "run_multi_task.sh"
    ).read_text()

    assert "COT_BUDGET_START_TOKEN" in source
    assert 'worker.rollout.cot_budget_enabled="${COT_BUDGET_ENABLED}"' in source
    assert 'worker.rollout.cot_budget_start_token="${COT_BUDGET_START_TOKEN}"' in source
    assert 'worker.rollout.cot_budget_end_token="${COT_BUDGET_END_TOKEN}"' in source
    assert 'worker.rollout.cot_budget_max_tokens="${COT_BUDGET_MAX_TOKENS}"' in source


def test_rollout_checker_counts_closed_and_over_budget_cot_spans(tmp_path: Path):
    module_path = Path(__file__).resolve().parents[1] / "video_proxy" / "training" / "tools" / "check_cot_budget_rollout.py"
    spec = importlib.util.spec_from_file_location("check_cot_budget_rollout", module_path)
    checker = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = checker
    spec.loader.exec_module(checker)

    path = tmp_path / "step_000001.jsonl"
    rows = [
        {"response": "<think>one two</think><answer>A</answer>"},
        {"response": "<think>one two three</think><answer>B</answer>"},
        {"response": "<answer>C</answer>"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    summary = checker.analyze_rollout_file(
        path,
        start_token="<think>",
        end_token="</think>",
        max_tokens=2,
    )

    assert summary.total == 3
    assert summary.started == 2
    assert summary.closed == 2
    assert summary.over_budget == 1
    assert summary.missing_start == 1


def test_rollout_checker_can_count_cot_span_with_tokenizer(tmp_path: Path):
    module_path = Path(__file__).resolve().parents[1] / "video_proxy" / "training" / "tools" / "check_cot_budget_rollout.py"
    spec = importlib.util.spec_from_file_location("check_cot_budget_rollout", module_path)
    checker = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = checker
    spec.loader.exec_module(checker)

    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return list(text)

    path = tmp_path / "step_000001.jsonl"
    path.write_text(json.dumps({"response": "<think>abc</think><answer>A</answer>"}) + "\n", encoding="utf-8")

    summary = checker.analyze_rollout_file(
        path,
        start_token="<think>",
        end_token="</think>",
        max_tokens=2,
        tokenizer=FakeTokenizer(),
    )

    assert summary.max_observed_tokens == 3
    assert summary.over_budget == 1


def test_qwen3_cot_2gpu_launcher_converts_data_and_enables_budget_check():
    launcher = Path("video_proxy/experiments/teacher_train/qwen3_vl_4b/run_cot_2gpu.sh").read_text()
    launcher_8b = Path("video_proxy/experiments/teacher_train/qwen3_vl_8b/run_cot_2gpu.sh").read_text()
    runner = Path("video_proxy/training/launchers/run_multi_task.sh").read_text()

    assert "convert_jsonl_to_cot.py" in launcher
    assert "--sample-prompts" in launcher
    assert 'N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-2}"' in launcher
    assert 'COT_BUDGET_ENABLED="${COT_BUDGET_ENABLED:-true}"' in launcher
    assert 'REUSE_EXISTING_DATA="${REUSE_EXISTING_DATA:-true}"' in launcher
    assert "check_cot_budget_rollout.py" in launcher
    assert "--require-start" in launcher
    assert '--tokenizer "${MODEL_PATH}"' in launcher
    assert 'MODEL_SIZE="8b"' in launcher_8b
    assert 'SOURCE_EXP_NAME="${SOURCE_EXP_NAME:-composition_base_seg_logic_aot_hier10k_el10k_aot10k_mf256_ema}"' in launcher
    assert 'SOURCE_EXP_NAME="${SOURCE_EXP_NAME:-composition_base_seg_logic_aot_hier10k_el10k_aot10k_mf256_ema}"' in launcher_8b
    assert 'Qwen3-VL-8B-Instruct' in launcher_8b
    assert "--require-start" in launcher_8b
    assert '--tokenizer "${MODEL_PATH}"' in launcher_8b
    assert 'REUSE_EXISTING_DATA=${REUSE_EXISTING_DATA_EFFECTIVE}' in runner
    assert 'REUSE_EXISTING_DATA_EFFECTIVE' in runner and "missing train/val experiment JSONL" in runner
