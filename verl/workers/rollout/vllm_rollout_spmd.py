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

import os
from contextlib import contextmanager
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.distributed
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, ProcessorMixin

from ...protocol import DataProto
from ...utils import torch_functional as VF
from ...utils.dataset import process_image, process_video
from ...utils.torch_dtypes import PrecisionType
from .base import BaseRollout
from .config import RolloutConfig
from .cot_budget import CoTBudgetController, configure_vllm_engine_for_cot_budget, make_cot_budget_controller


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray, list], repeats: int) -> Union[torch.Tensor, np.ndarray, list]:
    # repeat the elements, supports tensor, numpy array and list
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    elif isinstance(value, np.ndarray):
        return np.repeat(value, repeats, axis=0)
    elif isinstance(value, list):
        out = []
        for v in value:
            out.extend([v] * repeats)
        return out
    else:
        return np.repeat(value, repeats, axis=0)


def _get_logit_bias(processor: Optional[ProcessorMixin]) -> Optional[dict[int, float]]:
    # enforce vllm to not output image token
    # TODO: add video token
    if processor is not None and hasattr(processor, "image_token"):
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        return {image_token_id: -100}
    else:
        return None


def _process_multi_modal_data(
    multi_modal_data: dict[str, Any], min_pixels: int, max_pixels: int, video_fps: float
) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    """
    兼容原逻辑的最小改动版本：
    - 返回 (mm_data, mm_kwargs)
      * image: ({"image": [processed_images]}, None)
      * video: ({"video": [processed_videos]}, {"fps": sample_fps, "do_sample_frames": False})
    """
    images, videos = [], []
    mm_kwargs = None

    if "images" in multi_modal_data:
        for image in multi_modal_data["images"]:
            images.append(process_image(image, multi_modal_data.get("min_pixels", min_pixels), multi_modal_data.get("max_pixels", max_pixels)))

    if "videos" in multi_modal_data:
        kwargs = {k: v for k, v in multi_modal_data.items() if k not in ["images", "videos", "video_nframes", "video_fps"]}
        sample_fps = multi_modal_data.get("video_fps", video_fps)
        for idx, video in enumerate(multi_modal_data["videos"]):
            # 兼容带 fps 返回；若项目内函数不支持 return_fps，则回退到原始行为
            if isinstance(sample_fps, (list, tuple)):
                kwargs["video_fps"] = sample_fps[min(idx, len(sample_fps) - 1)]
            else:
                kwargs["video_fps"] = sample_fps
            processed, video_fps = process_video(video, return_fps=True, **kwargs)
            videos.append(processed)

        if len(videos) > 0:
            # mm_kwargs = {"fps": fps, "do_sample_frames": False} qwen25_vl
            mm_kwargs = {"do_sample_frames": False, "do_resize": False}



    if len(images) != 0:
        return {"image": images}, None

    if len(videos) != 0:
        return {"video": videos}, mm_kwargs

    print("!!!!!!aaaaa")

    return None, None


class vLLMRollout(BaseRollout):
    def __init__(
        self,
        model_path: str,
        config: RolloutConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
    ):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
        """
        super().__init__()
        self.rank = int(os.getenv("RANK", "0"))
        self.config = config
        self.pad_token_id = tokenizer.pad_token_id
        self.use_tqdm = (self.rank == 0) and (not config.disable_tqdm)
        if config.tensor_parallel_size > torch.distributed.get_world_size():
            raise ValueError("Tensor parallelism size should be less than world size.")

        if config.max_num_batched_tokens < config.prompt_length + config.response_length:
            raise ValueError("max_num_batched_tokens should be greater than prompt_length + response_length.")

        configure_vllm_engine_for_cot_budget(config.cot_budget_enabled)
        from vllm import LLM, SamplingParams

        engine_kwargs = {}
        if processor is not None:  # only VLMs have processor
            engine_kwargs["disable_mm_preprocessor_cache"] = True
            if config.limit_images:
                engine_kwargs["limit_mm_per_prompt"] = {"image": 1, "video": 1}

        self.inference_engine = LLM(
            model=model_path,
            skip_tokenizer_init=False,
            trust_remote_code=config.trust_remote_code,
            load_format="dummy",
            dtype=PrecisionType.to_str(PrecisionType.to_dtype(config.dtype)),
            seed=config.seed,
            max_model_len=config.max_model_len or config.prompt_length + config.response_length,
            distributed_executor_backend="external_launcher",
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_num_batched_tokens=config.max_num_batched_tokens,
            max_num_seqs=config.max_num_seqs,
            disable_log_stats=config.disable_log_stats,
            enforce_eager=config.enforce_eager,
            disable_custom_all_reduce=True,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_sleep_mode=True,
            **engine_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        # vLLM V1 sleep() has a bug where it asserts freed_bytes >= 0, which
        # can spuriously fail when NCCL/CUDA context allocates memory concurrently.
        # Catch AssertionError and fall back gracefully (model stays on GPU).
        try:
            self.inference_engine.sleep(level=1)
        except AssertionError:
            pass

        self.cot_budget_controller: Optional[CoTBudgetController] = None
        if config.cot_budget_enabled:
            if config.cot_budget_max_tokens <= 0:
                raise ValueError("worker.rollout.cot_budget_max_tokens must be positive when cot_budget_enabled=true.")
            self.cot_budget_controller = make_cot_budget_controller(
                tokenizer,
                start_token=config.cot_budget_start_token,
                end_token=config.cot_budget_end_token,
                max_tokens=config.cot_budget_max_tokens,
            )

        sampling_kwargs = {
            "max_tokens": config.response_length,
            "detokenize": False,
            "logit_bias": _get_logit_bias(processor),
        }
        default_sampling_params = SamplingParams()
        for key in config.to_dict().keys():
            if hasattr(default_sampling_params, key):
                sampling_kwargs[key] = getattr(config, key)

        print(f"Sampling params: {sampling_kwargs}.")
        self.sampling_params = SamplingParams(**sampling_kwargs)
        self._last_cot_budget_debug: list[dict[str, Any]] = []

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)

        yield
        # roll back to previous sampling params
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    def _generate_with_cot_budget(self, vllm_inputs: list[dict[str, Any]]) -> list[list[int]]:
        completions = self.inference_engine.generate(
            prompts=vllm_inputs, sampling_params=self.sampling_params, use_tqdm=self.use_tqdm
        )
        response_ids = [list(output.token_ids) for completion in completions for output in completion.outputs]
        debug_info: list[dict[str, Any]] = []
        if self.cot_budget_controller is None:
            self._last_cot_budget_debug = []
            return response_ids

        continuation_requests: list[tuple[int, dict[str, Any], int]] = []
        response_idx = 0
        for prompt_idx, completion in enumerate(completions):
            for output in completion.outputs:
                raw_token_ids = list(output.token_ids)
                cot_start_detected = self.cot_budget_controller.has_start(raw_token_ids)
                debug_entry = {
                    "response_index": response_idx,
                    "prompt_index": prompt_idx,
                    "cot_budget_enabled": True,
                    "cot_start_detected": cot_start_detected,
                    "cot_repaired": False,
                    "raw_token_len": len(raw_token_ids),
                    "repaired_token_len": len(raw_token_ids),
                    "remaining_tokens": 0,
                    "continuation_token_len": 0,
                    "final_token_len": len(raw_token_ids),
                    "max_cot_tokens": self.cot_budget_controller.max_tokens,
                    "max_response_length": self.config.response_length,
                }
                repaired = self.cot_budget_controller.repaired_prefix(
                    raw_token_ids, max_length=self.config.response_length
                )
                if repaired is not None:
                    response_ids[response_idx] = repaired
                    remaining_tokens = self.config.response_length - len(repaired)
                    debug_entry.update(
                        {
                            "cot_repaired": True,
                            "repaired_token_len": len(repaired),
                            "remaining_tokens": remaining_tokens,
                            "final_token_len": len(repaired),
                        }
                    )
                    if remaining_tokens > 0:
                        continuation_input = dict(vllm_inputs[prompt_idx])
                        continuation_input["prompt_token_ids"] = (
                            list(vllm_inputs[prompt_idx]["prompt_token_ids"]) + repaired
                        )
                        continuation_requests.append((response_idx, continuation_input, remaining_tokens))
                debug_info.append(debug_entry)
                response_idx += 1

        if not continuation_requests:
            self._last_cot_budget_debug = debug_info
            return response_ids

        max_remaining_tokens = max(remaining_tokens for _, _, remaining_tokens in continuation_requests)
        continuation_inputs = [request for _, request, _ in continuation_requests]
        with self.update_sampling_params(n=1, max_tokens=max_remaining_tokens):
            continuations = self.inference_engine.generate(
                prompts=continuation_inputs,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )

        for (response_idx, _, remaining_tokens), continuation in zip(continuation_requests, continuations):
            if continuation.outputs:
                continuation_ids = list(continuation.outputs[0].token_ids)[:remaining_tokens]
                response_ids[response_idx] = (
                    response_ids[response_idx] + continuation_ids
                )[: self.config.response_length]
                debug_info[response_idx]["continuation_token_len"] = len(continuation_ids)
                debug_info[response_idx]["final_token_len"] = len(response_ids[response_idx])

        self._last_cot_budget_debug = debug_info
        return response_ids

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        import time as _time
        from verl.utils.timing_logger import tlog
        _rank = torch.distributed.get_rank()
        _t0 = _time.time()

        # left-padded attention_mask
        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        batch_raw_prompt_ids = non_tensor_batch.pop("raw_prompt_ids")
        batch_multi_modal_data = non_tensor_batch.pop("multi_modal_data", None)
        if batch_size != len(batch_raw_prompt_ids):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if batch_multi_modal_data is not None:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(batch_raw_prompt_ids, batch_multi_modal_data):
                mm_data, mm_kwargs = _process_multi_modal_data(
                    multi_modal_data,
                    prompts.meta_info["min_pixels"],
                    prompts.meta_info["max_pixels"],
                    prompts.meta_info["video_fps"],
                )
                item = {
                    "prompt_token_ids": list(raw_prompt_ids),
                }
                if mm_data is not None:
                    item["multi_modal_data"] = mm_data
                    if mm_kwargs is not None:
                        item["mm_processor_kwargs"] = mm_kwargs
                vllm_inputs.append(item)

        else:
            vllm_inputs = [{"prompt_token_ids": list(raw_prompt_ids)} for raw_prompt_ids in batch_raw_prompt_ids]

        _t1 = _time.time()
        tlog(f"[vllm][rank={_rank}] prepare_inputs: {_t1 - _t0:.2f}s, n_prompts={batch_size}, n_per_prompt={self.sampling_params.n}")

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**prompts.meta_info):
            generated_response_ids = self._generate_with_cot_budget(vllm_inputs)
            _t2 = _time.time()
            tlog(f"[vllm][rank={_rank}] engine.generate: {_t2 - _t1:.2f}s")

            response_ids = VF.pad_2d_list_to_length(
                generated_response_ids, self.pad_token_id, max_length=self.config.response_length
            ).to(input_ids.device)

            if self.sampling_params.n > 1:
                batch_size = batch_size * self.sampling_params.n
                input_ids = _repeat_interleave(input_ids, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                if batch_multi_modal_data is not None:
                    batch_multi_modal_data = _repeat_interleave(batch_multi_modal_data, self.sampling_params.n)

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.ndim == 3:  # qwen2vl mrope: (batch_size, 4, seq_length)
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)

        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1 | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3 | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_mask = VF.get_response_mask(
            response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if batch_multi_modal_data is not None:
            non_tensor_batch = {"multi_modal_data": batch_multi_modal_data}
        else:
            non_tensor_batch = {}
        if self._last_cot_budget_debug:
            non_tensor_batch["cot_budget_debug"] = np.array(self._last_cot_budget_debug, dtype=object)

        _t3 = _time.time()
        tlog(f"[vllm][rank={_rank}] post_process: {_t3 - _t2:.2f}s, total: {_t3 - _t0:.2f}s")

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompts.meta_info)
