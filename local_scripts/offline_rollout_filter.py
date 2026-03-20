#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline filter a JSONL training set by rollout reward diversity.

For each sample:
1. Run the current model with `num_rollouts` sampled generations.
2. Score each generation with the configured reward function.
3. Keep the sample only if the rollout rewards are not all identical.

This is intended to replace online filtering for mixed-task DAPO runs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

from verl.utils.dataset import process_image, process_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline rollout filter for RL JSONL datasets.")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--report_jsonl", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--reward_function", required=True, help="Path spec like path/to/reward.py:compute_score")
    parser.add_argument("--backend", choices=["vllm", "transformers"], default="vllm")
    parser.add_argument("--num_rollouts", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--video_fps", type=float, default=2.0)
    parser.add_argument("--max_frames", type=int, default=256)
    parser.add_argument("--max_pixels", type=int, default=49152)
    parser.add_argument("--min_pixels", type=int, default=3136)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--reward_round_digits", type=int, default=6)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--max_model_len", type=int, default=0)
    parser.add_argument("--max_num_batched_tokens", type=int, default=16384)
    parser.add_argument("--dtype", default="bfloat16")
    # Data-parallel sharding: split input across multiple GPU workers
    parser.add_argument("--shard_id", type=int, default=0, help="This worker's shard index (0-based)")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of parallel shards")
    return parser.parse_args()


def load_jsonl(path: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Failed to parse {path}:{line_no}: {exc}") from exc
    return items


def load_reward_function(path_spec: str):
    if ":" in path_spec:
        module_path, func_name = path_spec.rsplit(":", maxsplit=1)
    else:
        module_path, func_name = path_spec, "main"
    module_path = os.path.abspath(module_path)

    spec = importlib.util.spec_from_file_location("offline_reward_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load reward module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["offline_reward_module"] = module
    spec.loader.exec_module(module)
    if not hasattr(module, func_name):
        raise AttributeError(f"{module_path} has no function `{func_name}`")
    return getattr(module, func_name)


def build_messages(example: dict[str, Any]) -> list[dict[str, Any]]:
    prompt_str = example.get("prompt", "")
    if "messages" in example and isinstance(example["messages"], list):
        for msg in example["messages"]:
            if msg.get("role") == "user":
                prompt_str = msg.get("content", prompt_str)

    videos = example.get("videos") or []
    images = example.get("images") or []
    if videos:
        content = []
        for idx, chunk in enumerate(str(prompt_str).split("<video>")):
            if idx != 0:
                content.append({"type": "video"})
            if chunk:
                content.append({"type": "text", "text": chunk})
        return [{"role": "user", "content": content}]
    if images:
        content = []
        for idx, chunk in enumerate(str(prompt_str).split("<image>")):
            if idx != 0:
                content.append({"type": "image"})
            if chunk:
                content.append({"type": "text", "text": chunk})
        return [{"role": "user", "content": content}]
    return [{"role": "user", "content": str(prompt_str)}]


def build_prompt(example: dict[str, Any], processor) -> str:
    messages = build_messages(example)
    return processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def build_multi_modal_data(example: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    videos = example.get("videos") or []
    images = example.get("images") or []
    if videos:
        n_videos = len(videos)
        max_frames_per_video = max(1, args.max_frames // n_videos) if n_videos > 1 else args.max_frames
        return {
            "videos": videos,
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
            "max_frames": max_frames_per_video,
            "video_fps": args.video_fps,
        }
    if images:
        return {
            "images": images,
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
        }
    return None


def prepare_inputs(example: dict[str, Any], processor, args: argparse.Namespace) -> dict[str, torch.Tensor]:
    prompt = build_prompt(example, processor)

    videos = example.get("videos") or []
    images = example.get("images") or []
    if videos:
        processed_videos = []
        video_metadatas = []
        max_frames_per_video = max(1, args.max_frames // len(videos)) if len(videos) > 1 else args.max_frames
        for video in videos:
            processed_video, _video_sample_fps = process_video(
                video,
                min_pixels=args.min_pixels,
                max_pixels=args.max_pixels,
                max_frames=max_frames_per_video,
                video_fps=args.video_fps,
                return_fps=True,
            )
            if processed_video is not None:
                frames, meta = processed_video
                processed_videos.append(frames)
                video_metadatas.append(meta)
        return processor(
            text=[prompt],
            videos=processed_videos or None,
            add_special_tokens=False,
            video_metadata=video_metadatas or None,
            return_tensors="pt",
            do_resize=False,
            do_sample_frames=False,
        )

    if images:
        processed_images = [process_image(image, args.min_pixels, args.max_pixels) for image in images]
        return processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")

    return processor(text=[prompt], add_special_tokens=False, return_tensors="pt")


def get_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise RuntimeError("Model has no parameters") from exc


def init_transformers_backend(args: argparse.Namespace):
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model_config = AutoConfig.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        bos_token_id=processor.tokenizer.bos_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        config=model_config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return processor, model


def init_vllm_backend(args: argparse.Namespace):
    import time as _time

    print("[offline_filter/vllm] importing vllm ...", flush=True)
    _t0 = _time.time()
    from vllm import LLM, SamplingParams
    print(f"[offline_filter/vllm] vllm imported in {_time.time()-_t0:.1f}s", flush=True)

    from verl.workers.rollout.vllm_rollout_spmd import _get_logit_bias

    print("[offline_filter/vllm] loading processor ...", flush=True)
    _t0 = _time.time()
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    print(f"[offline_filter/vllm] processor loaded in {_time.time()-_t0:.1f}s", flush=True)

    llm_kwargs = {
        "model": args.model_path,
        "trust_remote_code": True,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": args.dtype,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "disable_log_stats": True,
        "disable_mm_preprocessor_cache": True,
        "seed": args.seed,
    }
    if args.max_model_len > 0:
        llm_kwargs["max_model_len"] = args.max_model_len
    print(f"[offline_filter/vllm] constructing LLM engine (kwargs: { {k:v for k,v in llm_kwargs.items() if k != 'model'} }) ...", flush=True)
    _t0 = _time.time()
    engine = LLM(**llm_kwargs)
    print(f"[offline_filter/vllm] LLM engine ready in {_time.time()-_t0:.1f}s", flush=True)

    sampling_params = SamplingParams(
        n=args.num_rollouts,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=-1,
        seed=args.seed,
        max_tokens=args.max_new_tokens,
        logit_bias=_get_logit_bias(processor),
    )
    return processor, engine, sampling_params


def generate_with_transformers(example: dict[str, Any], processor, model, args: argparse.Namespace) -> list[str]:
    model_device = get_model_device(model)
    inputs = prepare_inputs(example, processor, args)
    inputs = {k: v.to(model_device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            num_return_sequences=args.num_rollouts,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
        )

    trimmed_ids = generated_ids[:, prompt_len:]
    return processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def generate_with_vllm(example: dict[str, Any], processor, llm, sampling_params, args: argparse.Namespace) -> list[str]:
    """Single-sample fallback (used only for error retry)."""
    request = _build_vllm_request(example, processor, args)
    if request is None:
        return []
    outputs = llm.generate([request], sampling_params=sampling_params, use_tqdm=False)
    return [output.text for output in outputs[0].outputs]


def _build_vllm_request(example: dict[str, Any], processor, args: argparse.Namespace) -> dict[str, Any] | None:
    """Build a single vLLM request dict from a data example."""
    from verl.workers.rollout.vllm_rollout_spmd import _process_multi_modal_data

    prompt = build_prompt(example, processor)
    request = {
        "prompt_token_ids": processor.tokenizer.encode(prompt, add_special_tokens=False),
    }
    multi_modal_data = build_multi_modal_data(example, args)
    if multi_modal_data is not None:
        mm_data, mm_kwargs = _process_multi_modal_data(
            multi_modal_data,
            args.min_pixels,
            args.max_pixels,
            args.video_fps,
        )
        if mm_data is not None:
            request["multi_modal_data"] = mm_data
        if mm_kwargs is not None:
            request["mm_processor_kwargs"] = mm_kwargs
    return request


def generate_batch_with_vllm(
    items: list[dict[str, Any]],
    processor,
    llm,
    sampling_params,
    args: argparse.Namespace,
    batch_size: int = 32,
) -> list[list[str] | Exception]:
    """Batch generate with vLLM in mini-batches to bound CPU memory.

    Video frame preprocessing is CPU/RAM intensive. Processing all items at
    once can OOM when running multiple workers in parallel. Mini-batches
    keep peak RAM usage bounded while still benefiting from vLLM's
    continuous batching within each batch.
    """
    import time as _time

    results: list[list[str] | Exception] = [[] for _ in range(len(items))]
    total_batches = (len(items) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(items))
        batch_items = items[start:end]

        requests = []
        req_to_item = []   # maps request index -> item index within this batch

        for local_idx, item in enumerate(batch_items):
            global_idx = start + local_idx
            try:
                req = _build_vllm_request(item, processor, args)
                if req is None:
                    results[global_idx] = RuntimeError("Failed to build vLLM request")
                    continue
                requests.append(req)
                req_to_item.append(global_idx)
            except Exception as exc:
                results[global_idx] = exc

        if requests:
            _t0 = _time.time()
            batch_outputs = llm.generate(requests, sampling_params=sampling_params, use_tqdm=False)
            _elapsed = _time.time() - _t0
            for req_idx, output in enumerate(batch_outputs):
                results[req_to_item[req_idx]] = [o.text for o in output.outputs]
            print(
                f"[offline_filter] batch {batch_idx+1}/{total_batches}: "
                f"{len(requests)} requests in {_elapsed:.1f}s",
                flush=True,
            )

    return results


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    import time as _time

    print(f"[offline_filter] Loading JSONL: {args.input_jsonl}", flush=True)
    _t0 = _time.time()
    items = load_jsonl(args.input_jsonl)
    print(f"[offline_filter] Loaded {len(items)} samples in {_time.time()-_t0:.1f}s", flush=True)
    if args.max_samples > 0:
        items = items[: args.max_samples]

    # Apply data-parallel sharding
    if args.num_shards > 1:
        items = items[args.shard_id :: args.num_shards]
        print(f"[offline_filter] Shard {args.shard_id}/{args.num_shards}: processing {len(items)} samples", flush=True)

    print(f"[offline_filter] Loading reward function: {args.reward_function}", flush=True)
    _t0 = _time.time()
    reward_fn = load_reward_function(args.reward_function)
    print(f"[offline_filter] Reward function loaded in {_time.time()-_t0:.1f}s", flush=True)

    if args.backend == "vllm":
        print(f"[offline_filter] Initializing vLLM backend (model={args.model_path}, "
              f"tp={args.tensor_parallel_size}, gpu_mem={args.gpu_memory_utilization}, "
              f"max_model_len={args.max_model_len}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')})",
              flush=True)
        _t0 = _time.time()
        processor, llm, sampling_params = init_vllm_backend(args)
        print(f"[offline_filter] vLLM backend ready in {_time.time()-_t0:.1f}s", flush=True)
        model = None
    else:
        print("[offline_filter] Initializing transformers backend ...", flush=True)
        processor, model = init_transformers_backend(args)
        print("[offline_filter] Transformers backend ready", flush=True)
        llm = None
        sampling_params = None

    output_path = Path(args.output_jsonl)
    report_path = Path(args.report_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    dropped = 0
    kept_by_type: dict[str, int] = {}
    dropped_by_type: dict[str, int] = {}

    def _process_result(
        item: dict[str, Any],
        idx: int,
        responses: list[str],
        fout,
        freport,
    ) -> None:
        nonlocal kept, dropped
        reward_inputs = []
        for response in responses:
            reward_inputs.append(
                {
                    "response": response,
                    "response_length": len(response),
                    "ground_truth": item.get("answer", ""),
                    "data_type": item.get("data_type", ""),
                    "problem_type": item.get("problem_type", ""),
                    "problem": item.get("prompt", ""),
                    "problem_id": item.get("metadata", {}).get("clip_key"),
                }
            )
        scores = reward_fn(reward_inputs)
        rewards = [round(float(score["overall"]), args.reward_round_digits) for score in scores]
        unique_rewards = sorted(set(rewards))
        keep = len(unique_rewards) > 1
        ptype = item.get("problem_type", "unknown") or "unknown"
        if keep:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1
            kept_by_type[ptype] = kept_by_type.get(ptype, 0) + 1
        else:
            dropped += 1
            dropped_by_type[ptype] = dropped_by_type.get(ptype, 0) + 1

        report = {
            "index": idx,
            "problem_type": item.get("problem_type", ""),
            "keep": keep,
            "rewards": rewards,
            "unique_rewards": unique_rewards,
            "answer": item.get("answer", ""),
            "prompt": item.get("prompt", ""),
            "responses": responses,
        }
        freport.write(json.dumps(report, ensure_ascii=False) + "\n")

    def _process_error(
        item: dict[str, Any],
        idx: int,
        exc: Exception,
        freport,
    ) -> None:
        nonlocal dropped
        ptype = item.get("problem_type", "unknown") or "unknown"
        dropped += 1
        dropped_by_type[ptype] = dropped_by_type.get(ptype, 0) + 1
        report = {
            "index": idx,
            "problem_type": item.get("problem_type", ""),
            "keep": False,
            "error": str(exc),
            "answer": item.get("answer", ""),
            "prompt": item.get("prompt", ""),
        }
        freport.write(json.dumps(report, ensure_ascii=False) + "\n")

    with output_path.open("w", encoding="utf-8") as fout, report_path.open("w", encoding="utf-8") as freport:
        if args.backend == "vllm":
            # ---- Batched vLLM inference ----
            print(f"[offline_filter] Preprocessing {len(items)} requests for batch vLLM inference ...", flush=True)
            _t0 = _time.time()
            batch_results = generate_batch_with_vllm(items, processor, llm, sampling_params, args)
            print(f"[offline_filter] Batch inference done in {_time.time()-_t0:.1f}s", flush=True)

            for idx, (item, result) in enumerate(zip(items, batch_results)):
                if isinstance(result, Exception):
                    _process_error(item, idx, result, freport)
                else:
                    try:
                        _process_result(item, idx, result, fout, freport)
                    except Exception as exc:
                        _process_error(item, idx, exc, freport)
                if (idx + 1) % args.log_every == 0 or (idx + 1) == len(items):
                    print(
                        f"[offline_filter] scored={idx + 1}/{len(items)} "
                        f"kept={kept} dropped={dropped}",
                        flush=True,
                    )
        else:
            # ---- Sequential transformers inference ----
            progress = tqdm(items, desc="offline_filter", total=len(items), dynamic_ncols=True)
            for idx, item in enumerate(progress):
                try:
                    responses = generate_with_transformers(item, processor, model, args)
                    _process_result(item, idx, responses, fout, freport)
                except Exception as exc:
                    _process_error(item, idx, exc, freport)
                progress.set_postfix(kept=kept, dropped=dropped, refresh=False)
                if (idx + 1) % args.log_every == 0 or (idx + 1) == len(items):
                    print(
                        f"[offline_filter] processed={idx + 1}/{len(items)} "
                        f"kept={kept} dropped={dropped}"
                    )

    print(
        f"[offline_filter] done. input={len(items)} kept={kept} dropped={dropped} "
        f"output={output_path} report={report_path}"
    )

    # Per-task breakdown
    all_types = sorted(set(list(kept_by_type) + list(dropped_by_type)))
    if all_types:
        print("\n[offline_filter] Per-task summary:")
        print(f"  {'task':<30}  {'kept':>6}  {'dropped':>7}  {'total':>6}  {'keep%':>7}")
        print("  " + "-" * 60)
        for t in all_types:
            k = kept_by_type.get(t, 0)
            d = dropped_by_type.get(t, 0)
            total_t = k + d
            pct = f"{100.0 * k / total_t:.1f}%" if total_t else "  n/a"
            print(f"  {t:<30}  {k:>6}  {d:>7}  {total_t:>6}  {pct:>7}")
        print("  " + "-" * 60)
        print(f"  {'TOTAL':<30}  {kept:>6}  {dropped:>7}  {len(items):>6}  {100.0*kept/len(items) if items else 0:.1f}%")


if __name__ == "__main__":
    main()
