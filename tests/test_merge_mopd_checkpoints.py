import json
from unittest.mock import patch
from pathlib import Path

from local_scripts.opd_comparison.merge_mopd_checkpoints import (
    Checkpoint,
    build_registry,
    discover_checkpoints,
    merge_checkpoint,
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


def test_merge_checkpoint_calls_model_merger_even_when_hf_dir_exists(tmp_path: Path):
    actor_dir = tmp_path / "exp" / "global_step_50" / "actor"
    hf_dir = actor_dir / "huggingface"
    hf_dir.mkdir(parents=True)
    (hf_dir / "config.json").write_text("{}", encoding="utf-8")
    checkpoint = Checkpoint(step=50, step_dir=actor_dir.parent, actor_dir=actor_dir)

    def fake_merge(*_args, **_kwargs):
        hf_dir.mkdir(parents=True)
        (hf_dir / "config.json").write_text("{}", encoding="utf-8")

    with patch("local_scripts.opd_comparison.merge_mopd_checkpoints.subprocess.run") as run:
        run.side_effect = fake_merge
        dest = merge_checkpoint(
            checkpoint,
            output_dir=tmp_path / "merged",
            model_prefix="Qwen3-VL-4B-Instruct-MOPD",
            base_model=tmp_path / "base",
            merger_script=tmp_path / "model_merger.py",
            force=False,
            reuse_existing_hf=False,
            dry_run=False,
        )

    run.assert_called_once()
    assert dest == tmp_path / "merged" / "Qwen3-VL-4B-Instruct-MOPD-Step50"
    assert (dest / "config.json").exists()


def test_merge_checkpoint_can_reuse_existing_hf_dir_when_requested(tmp_path: Path):
    actor_dir = tmp_path / "exp" / "global_step_50" / "actor"
    hf_dir = actor_dir / "huggingface"
    hf_dir.mkdir(parents=True)
    (hf_dir / "config.json").write_text("{}", encoding="utf-8")
    checkpoint = Checkpoint(step=50, step_dir=actor_dir.parent, actor_dir=actor_dir)

    with patch("local_scripts.opd_comparison.merge_mopd_checkpoints.subprocess.run") as run:
        merge_checkpoint(
            checkpoint,
            output_dir=tmp_path / "merged",
            model_prefix="Qwen3-VL-4B-Instruct-MOPD",
            base_model=tmp_path / "base",
            merger_script=tmp_path / "model_merger.py",
            force=False,
            reuse_existing_hf=True,
            dry_run=False,
        )

    run.assert_not_called()
