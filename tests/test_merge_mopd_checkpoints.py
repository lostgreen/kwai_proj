import json
from pathlib import Path

from local_scripts.opd_comparison.merge_mopd_checkpoints import (
    build_registry,
    discover_checkpoints,
    model_name_for_step,
)


def test_discover_checkpoints_sorts_numeric_steps_and_requires_actor(tmp_path: Path):
    exp_dir = tmp_path / "mopd_run"
    for step in (300, 50, 100):
        actor = exp_dir / f"global_step_{step}" / "actor"
        actor.mkdir(parents=True)
        (actor / "model_world_size_1_rank_0.pt").write_text("stub", encoding="utf-8")
    (exp_dir / "global_step_200").mkdir(parents=True)

    checkpoints = discover_checkpoints(exp_dir)

    assert [item.step for item in checkpoints] == [50, 100, 300]
    assert [item.actor_dir.name for item in checkpoints] == ["actor", "actor", "actor"]


def test_build_registry_uses_step_named_qwen3vl_entries(tmp_path: Path):
    output_dir = tmp_path / "merged"
    entries = [
        (50, output_dir / "Qwen3-VL-4B-Instruct-MOPD-Step50"),
        (100, output_dir / "Qwen3-VL-4B-Instruct-MOPD-Step100"),
    ]

    registry = build_registry(entries, model_prefix="Qwen3-VL-4B-Instruct-MOPD")

    assert model_name_for_step("Qwen3-VL-4B-Instruct-MOPD", 50) == "Qwen3-VL-4B-Instruct-MOPD-Step50"
    assert list(registry) == ["Qwen3-VL-4B-Instruct-MOPD-Step50", "Qwen3-VL-4B-Instruct-MOPD-Step100"]
    assert registry["Qwen3-VL-4B-Instruct-MOPD-Step50"] == {
        "class": "Qwen3VLChat",
        "model_path": str(output_dir / "Qwen3-VL-4B-Instruct-MOPD-Step50"),
        "use_custom_prompt": False,
        "use_vllm": True,
        "temperature": 0,
        "max_new_tokens": 16384,
        "repetition_penalty": 1.0,
        "presence_penalty": 1.5,
        "top_p": 0.8,
        "top_k": 20,
        "min_pixels": 3136,
        "max_pixels": 65536,
    }

    json.dumps(registry, ensure_ascii=False)
