# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
python /home/xuboshen/zgw/EasyR1/scripts/model_merger.py \
    --local_dir /m2v_intern/xuboshen/zgw/RL-Models/qwen3_vl_youcook2_temporal_seg_8gpu-48token/global_step_250/actor
"""

import argparse
import os
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch.distributed._tensor import DTensor, Placement, Shard
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForTokenClassification,
    PretrainedConfig,
    PreTrainedModel,
)


def merge_by_placement(tensors: list[torch.Tensor], placement: Placement):
    if placement.is_replicate():
        return tensors[0]
    elif placement.is_partial():
        raise NotImplementedError("Partial placement is not supported yet")
    elif placement.is_shard():
        return torch.cat(tensors, dim=placement.dim).contiguous()
    else:
        raise ValueError(f"Unsupported placement: {placement}")


def upload_model_to_huggingface(local_path: str, remote_path: str):
    # Push to hugging face
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=remote_path, private=False, exist_ok=True)
    api.upload_folder(repo_id=remote_path, folder_path=local_path, repo_type="model")


def _copy_non_weight_files(src_dir: str, dst_dir: str) -> None:
    """Copy config / tokenizer / processor files from *src_dir* to *dst_dir*.

    Only non-weight files are copied (config.json, tokenizer*, preprocessor*,
    generation_config.json, etc.).  Existing weight files (.safetensors, .bin)
    in *dst_dir* are never touched.
    """
    import shutil

    _WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth"}
    _SKIP = {"model.safetensors.index.json"}
    for fname in os.listdir(src_dir):
        src = os.path.join(src_dir, fname)
        if not os.path.isfile(src):
            continue
        if any(fname.endswith(s) for s in _WEIGHT_SUFFIXES):
            continue
        if fname in _SKIP:
            continue
        dst = os.path.join(dst_dir, fname)
        shutil.copy2(src, dst)
        print(f"  copied {fname} from base model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", required=True, type=str, help="The path for your saved model")
    parser.add_argument(
        "--base_model",
        default=None,
        type=str,
        help="Path (or HF hub id) of the ORIGINAL base model. "
        "When provided, config.json / tokenizer files are copied from here "
        "after merge to avoid save_pretrained dropping architecture fields "
        "(e.g. num_experts for MoE / hybrid models).",
    )
    parser.add_argument("--hf_upload_path", default=False, type=str, help="The path of the huggingface repo to upload")
    args = parser.parse_args()
    local_dir: str = args.local_dir

    assert not local_dir.endswith("huggingface"), "The local_dir should not end with huggingface."

    # copy rank zero to find the shape of (dp, fsdp)
    rank = 0
    world_size = 0
    for filename in os.listdir(local_dir):
        match = re.match(r"model_world_size_(\d+)_rank_0\.pt", filename)
        if match:
            world_size = match.group(1)
            break

    assert world_size, "No model file with the proper format."

    rank0_weight_path = os.path.join(local_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
    state_dict = torch.load(rank0_weight_path, map_location="cpu", weights_only=False)
    pivot_key = sorted(state_dict.keys())[0]
    weight = state_dict[pivot_key]
    if isinstance(weight, DTensor):
        # get sharding info
        device_mesh = weight.device_mesh
        mesh = device_mesh.mesh
        mesh_dim_names = device_mesh.mesh_dim_names
    else:
        # for non-DTensor
        mesh = np.array([int(world_size)], dtype=np.int64)
        mesh_dim_names = ("fsdp",)

    print(f"Got device mesh {mesh}, mesh_dim_names {mesh_dim_names}")

    assert mesh_dim_names in (("fsdp",), ("ddp", "fsdp")), f"Unsupported mesh_dim_names {mesh_dim_names}."

    if "tp" in mesh_dim_names:
        # fsdp * tp
        total_shards = mesh.shape[-1] * mesh.shape[-2]
        mesh_shape = (mesh.shape[-2], mesh.shape[-1])
    else:
        # fsdp
        total_shards = mesh.shape[-1]
        mesh_shape = (mesh.shape[-1],)

    print(f"Processing {total_shards} model shards in total.")
    model_state_dict_lst = []
    model_state_dict_lst.append(state_dict)
    model_state_dict_lst.extend([""] * (total_shards - 1))

    def process_one_shard(rank, model_state_dict_lst):
        model_path = os.path.join(local_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
        model_state_dict_lst[rank] = state_dict
        return state_dict

    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count())) as executor:
        for rank in range(1, total_shards):
            executor.submit(process_one_shard, rank, model_state_dict_lst)

    state_dict: dict[str, list[torch.Tensor]] = {}
    param_placements: dict[str, list[Placement]] = {}
    keys = set(model_state_dict_lst[0].keys())
    for key in keys:
        state_dict[key] = []
        for model_state_dict in model_state_dict_lst:
            try:
                tensor = model_state_dict.pop(key)
            except Exception:
                print(f"Cannot find key {key} in rank {rank}.")

            if isinstance(tensor, DTensor):
                state_dict[key].append(tensor._local_tensor.bfloat16())
                placements = tuple(tensor.placements)
                # replicated placement at ddp dimension can be discarded
                if mesh_dim_names[0] == "ddp":
                    placements = placements[1:]

                if key not in param_placements:
                    param_placements[key] = placements
                else:
                    assert param_placements[key] == placements
            else:
                state_dict[key].append(tensor.bfloat16())

    del model_state_dict_lst

    for key in sorted(state_dict):
        if not isinstance(state_dict[key], list):
            print(f"No need to merge key {key}")
            continue

        if key in param_placements:
            # merge shards
            placements: tuple[Shard] = param_placements[key]
            if len(mesh_shape) == 1:
                # 1-D list, FSDP without TP
                assert len(placements) == 1
                shards = state_dict[key]
                state_dict[key] = merge_by_placement(shards, placements[0])
            else:
                # 2-D list, FSDP + TP
                raise NotImplementedError("FSDP + TP is not supported yet.")
        else:
            state_dict[key] = torch.cat(state_dict[key], dim=0)

    print("Merge completed.")
    hf_path = os.path.join(local_dir, "huggingface")

    # ── If --base_model is given, resolve it to a local directory so we can
    #    copy non-weight files from it later.  This handles both local paths
    #    and HuggingFace Hub model ids (e.g. "Qwen/Qwen3-VL-4B-Instruct").
    base_model_dir: str | None = None
    if args.base_model:
        if os.path.isdir(args.base_model):
            base_model_dir = args.base_model
        else:
            # Assume it's a HF hub id → snapshot_download
            from huggingface_hub import snapshot_download

            base_model_dir = snapshot_download(
                args.base_model,
                ignore_patterns=["*.safetensors", "*.bin", "*.pt", "*.pth"],
            )
        print(f"Base model config source: {base_model_dir}")

    config: PretrainedConfig = AutoConfig.from_pretrained(
        base_model_dir if base_model_dir else hf_path
    )
    architectures: list[str] = getattr(config, "architectures", ["Unknown"])

    if "ForTokenClassification" in architectures[0]:
        AutoClass = AutoModelForTokenClassification
    elif "ForConditionalGeneration" in architectures[0]:
        AutoClass = AutoModelForImageTextToText
    elif "ForCausalLM" in architectures[0]:
        AutoClass = AutoModelForCausalLM
    else:
        raise NotImplementedError(f"Unknown architecture {architectures}.")

    with torch.device("meta"):
        model: PreTrainedModel = AutoClass.from_config(config, torch_dtype=torch.bfloat16)

    assert isinstance(model, PreTrainedModel)
    model.to_empty(device="cpu")

    # ── Save merged weights.
    # save_pretrained() also re-serialises config.json from the (meta-device)
    # model object, which can silently drop architecture-specific fields
    # (num_experts, decoder_sparse_step, …).  We fix that below.
    import shutil

    config_path = os.path.join(hf_path, "config.json")
    config_backup = config_path + ".bak"
    if os.path.exists(config_path):
        shutil.copy2(config_path, config_backup)

    print(f"Saving model to {hf_path}...")
    model.save_pretrained(hf_path, state_dict=state_dict)
    del state_dict, model

    if base_model_dir:
        # Best path: copy ALL non-weight files (config, tokenizer, processor,
        # chat_template, …) from the authoritative base model.
        print(f"Restoring config / tokenizer files from base model: {base_model_dir}")
        _copy_non_weight_files(base_model_dir, hf_path)
        if os.path.exists(config_backup):
            os.remove(config_backup)
    elif os.path.exists(config_backup):
        # Fallback: restore the config.json that existed in hf_path before
        # save_pretrained overwrote it.  Works for first-time merges where
        # the training framework wrote a correct config during checkpoint save.
        shutil.move(config_backup, config_path)
        print("Restored config.json from pre-merge backup")
    else:
        print("[WARN] No --base_model provided and no config backup available. "
              "config.json may be incomplete – consider re-running with --base_model.")

    if args.hf_upload_path:
        upload_model_to_huggingface(hf_path, args.hf_upload_path)
