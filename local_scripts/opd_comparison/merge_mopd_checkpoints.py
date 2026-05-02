#!/usr/bin/env python3
"""Merge every global_step checkpoint from a MOPD run and emit a VLM registry."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_EXPERIMENT_DIR = Path(
    "/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/opd_comparison_4b/"
    "mopd_qwen3vl4b_full_comp_4b_teachers_bs64_mf256_epoch1_save50"
)
DEFAULT_BASE_MODEL = Path("/m2v_intern/xuboshen/models/Qwen3-VL-4B-Instruct")
DEFAULT_OUTPUT_DIR = Path(
    "/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/opd_comparison_4b_merged/"
    "mopd_qwen3vl4b_full_comp_4b_teachers_bs64_mf256_epoch1_save50"
)
DEFAULT_MODEL_PREFIX = "Qwen3-VL-4B-Instruct-MOPD"


@dataclass(frozen=True)
class Checkpoint:
    step: int
    step_dir: Path
    actor_dir: Path


def discover_checkpoints(experiment_dir: Path) -> list[Checkpoint]:
    checkpoints: list[Checkpoint] = []
    for step_dir in experiment_dir.glob("global_step_*"):
        if not step_dir.is_dir():
            continue
        try:
            step = int(step_dir.name.removeprefix("global_step_"))
        except ValueError:
            continue
        actor_dir = step_dir / "actor"
        if not actor_dir.is_dir():
            continue
        if not any(actor_dir.glob("model_world_size_*_rank_0.pt")):
            continue
        checkpoints.append(Checkpoint(step=step, step_dir=step_dir, actor_dir=actor_dir))
    return sorted(checkpoints, key=lambda item: item.step)


def model_name_for_step(model_prefix: str, step: int) -> str:
    return f"{model_prefix}-Step{step}"


def build_registry(entries: Iterable[tuple[int, Path]], *, model_prefix: str) -> dict[str, dict[str, object]]:
    registry: dict[str, dict[str, object]] = {}
    for step, model_path in sorted(entries, key=lambda item: item[0]):
        name = model_name_for_step(model_prefix, step)
        registry[name] = {
            "class": "Qwen3VLChat",
            "model_path": str(model_path),
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
    return registry


def write_registry_py(registry: dict[str, dict[str, object]], path: Path) -> None:
    lines = [
        "from functools import partial",
        "from vlmeval.vlm import Qwen3VLChat",
        "",
        "",
        "MOPD_MODELS = {",
    ]
    for name, cfg in registry.items():
        lines.extend(
            [
                f'    "{name}": partial(',
                "        Qwen3VLChat,",
                f'        model_path="{cfg["model_path"]}",',
                f'        use_custom_prompt={cfg["use_custom_prompt"]},',
                f'        use_vllm={cfg["use_vllm"]},',
                f'        temperature={cfg["temperature"]},',
                f'        max_new_tokens={cfg["max_new_tokens"]},',
                f'        repetition_penalty={cfg["repetition_penalty"]},',
                f'        presence_penalty={cfg["presence_penalty"]},',
                f'        top_p={cfg["top_p"]},',
                f'        top_k={cfg["top_k"]},',
                f'        min_pixels={cfg["min_pixels"]},',
                f'        max_pixels={cfg["max_pixels"]},',
                "    ),",
            ]
        )
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def merge_checkpoint(
    checkpoint: Checkpoint,
    *,
    output_dir: Path,
    model_prefix: str,
    base_model: Path,
    merger_script: Path,
    force: bool,
    dry_run: bool,
) -> Path:
    dest_dir = output_dir / model_name_for_step(model_prefix, checkpoint.step)
    hf_dir = checkpoint.actor_dir / "huggingface"

    print(f"[step {checkpoint.step}] actor: {checkpoint.actor_dir}")
    print(f"[step {checkpoint.step}] dest : {dest_dir}")
    if dry_run:
        return dest_dir

    if dest_dir.exists():
        if not force:
            print(f"[step {checkpoint.step}] skip existing destination")
            return dest_dir
        shutil.rmtree(dest_dir)

    if not hf_dir.is_dir() or force:
        command = [
            sys.executable,
            str(merger_script),
            "--local_dir",
            str(checkpoint.actor_dir),
            "--base_model",
            str(base_model),
        ]
        subprocess.run(command, check=True)

    if not hf_dir.is_dir():
        raise FileNotFoundError(f"merge did not create {hf_dir}")

    shutil.copytree(hf_dir, dest_dir)
    return dest_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--model-prefix", default=DEFAULT_MODEL_PREFIX)
    parser.add_argument("--registry-json", default="model_registry.json")
    parser.add_argument("--registry-py", default="model_meta.py")
    parser.add_argument("--force", action="store_true", help="Re-merge and overwrite existing destinations.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned work without merging or writing files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    merger_script = repo_root / "scripts" / "model_merger.py"
    checkpoints = discover_checkpoints(args.experiment_dir)
    if not checkpoints:
        raise SystemExit(f"No mergeable global_step_* actor checkpoints found in {args.experiment_dir}")

    print(f"Experiment : {args.experiment_dir}")
    print(f"Base model : {args.base_model}")
    print(f"Output dir : {args.output_dir}")
    print(f"Steps      : {', '.join(str(item.step) for item in checkpoints)}")

    entries: list[tuple[int, Path]] = []
    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    for checkpoint in checkpoints:
        dest_dir = merge_checkpoint(
            checkpoint,
            output_dir=args.output_dir,
            model_prefix=args.model_prefix,
            base_model=args.base_model,
            merger_script=merger_script,
            force=args.force,
            dry_run=args.dry_run,
        )
        entries.append((checkpoint.step, dest_dir))

    registry = build_registry(entries, model_prefix=args.model_prefix)
    if args.dry_run:
        print(json.dumps(registry, ensure_ascii=False, indent=2))
        return

    registry_json_path = args.output_dir / args.registry_json
    registry_json_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    registry_py_path = args.output_dir / args.registry_py
    write_registry_py(registry, registry_py_path)

    print(f"[done] registry json: {registry_json_path}")
    print(f"[done] registry py  : {registry_py_path}")


if __name__ == "__main__":
    main()
