#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local VLM pre-screening for hierarchical annotation pipeline.

Uses a local Qwen VL model (via vLLM) to quickly assess whether a video
is suitable for hierarchical temporal segmentation annotation.

Evaluates four dimensions:
  1. L1 Phase structure score (1-5) + estimated phase count
  2. L2 Event structure score (1-5) + estimated event count
  3. Domain classification (domain_l1 + domain_l2)
  4. Visual quality (good/bad)

Decision rule: keep only if L1 >= threshold AND L2 >= threshold,
ensuring the video has at least two layers of temporal structure.

Architecture: Reuses the vLLM batch inference pattern from
offline_rollout_filter.py — supports tensor parallelism and
data-parallel sharding across multiple GPUs.

Usage:
    # Single GPU
    python local_screen.py \
        --input_jsonl sample_dev.jsonl \
        --output_jsonl screen_results.jsonl \
        --keep_jsonl screen_keep.jsonl \
        --reject_jsonl screen_reject.jsonl \
        --model_path /path/to/Qwen3-VL-4B-Instruct

    # 8-GPU data parallel (4B fits on 1 GPU)
    for i in $(seq 0 7); do
        CUDA_VISIBLE_DEVICES=$i python local_screen.py \
            --input_jsonl sample_dev.jsonl \
            --output_jsonl screen_shard${i}.jsonl \
            --keep_jsonl keep_shard${i}.jsonl \
            --reject_jsonl reject_shard${i}.jsonl \
            --model_path /path/to/Qwen3-VL-4B-Instruct \
            --shard_id $i --num_shards 8 &
    done
    wait
    cat keep_shard*.jsonl > screen_keep.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor

# ── Domain taxonomy (imported from annotation prompts) ──
# We inline the valid values here to avoid fragile cross-package imports.
VALID_DOMAIN_L1 = {
    "procedural", "physical", "lifestyle", "entertainment",
    "narrative", "educational", "other",
}
VALID_DOMAIN_L2 = {
    "cooking", "construction_building", "crafting_diy", "repair_maintenance",
    "sports", "fitness_exercise", "music_performance", "beauty_grooming",
    "cleaning_housework", "gardening_outdoor", "vehicle_operation",
    "movie_scene", "reality_show", "animation",
    "vlog", "interview_talk", "documentary", "news_report",
    "science_experiment", "lecture_tutorial", "other",
}


# ─────────────────────────────────────────────────────────────────────────────
# Screening prompt
# ─────────────────────────────────────────────────────────────────────────────

SCREEN_PROMPT_TEMPLATE = """\
<video>
Watch this video clip ({duration:.0f}s) and evaluate its temporal structure at two levels.

**IMPORTANT RULE:** A "phase" must be a largely continuous block of time dedicated to a specific goal or stage. If the video is a **montage** that frequently cuts between many short, different scenes (like a sports highlight reel or a music video), it lacks a true phase structure and should receive a low L1_SCORE (1 or 2), even if it contains a variety of content.

Q1 — L1 Phase Structure (L1_SCORE, integer 1-5):
  Does the video progress through distinct, logically ordered high-level phases?
  5 = 4+ clearly distinct and continuous phases with different goals or narrative functions (e.g., procedural: prep → cook → plate; narrative: setup → conflict → resolution).
  4 = 3 distinct, continuous phases are identifiable.
  3 = 2 distinct, continuous phases are visible.
  2 = The activity has some variation, but no clear phase boundaries or logical progression.
  1 = A single continuous activity, or a montage of short clips with no overarching temporal order.
  Also estimate: how many phases? (EST_PHASES: integer)

Q2 — L2 Event Structure (L2_SCORE, integer 1-5):
  Within potential phases, can you identify distinct sub-events or actions?
  5 = Most segments contain 3+ distinct sub-events.
  4 = Most segments have 2+ identifiable sub-events.
  3 = Some segments have sub-events, others are monolithic.
  2 = Sub-events are barely distinguishable.
  1 = No clear sub-event structure.
  Also estimate: total events across all phases? (EST_EVENTS: integer)

Q3 — Domain classification:
  domain_l1 choices: procedural | physical | lifestyle | entertainment | narrative | educational | other
  domain_l2 choices: cooking | construction_building | crafting_diy | repair_maintenance | sports | fitness_exercise | music_performance | beauty_grooming | cleaning_housework | gardening_outdoor | vehicle_operation | movie_scene | reality_show | animation | vlog | interview_talk | documentary | news_report | science_experiment | lecture_tutorial | other

Q4 — Visual quality (QUALITY):
  good = clear footage with visible physical actions, movements, or narrative content
  bad = dark, blurry, static, screen recording, text-heavy slides, pure talking head, gaming footage

Answer in EXACTLY this format (one field per line):
L1_SCORE: <integer 1-5>
EST_PHASES: <integer>
L2_SCORE: <integer 1-5>
EST_EVENTS: <integer>
DOMAIN_L1: <one word from the list>
DOMAIN_L2: <one word from the list>
QUALITY: <good or bad>
REASON: <one sentence explaining your assessment, especially your reasoning for the L1_SCORE>"""


# ─────────────────────────────────────────────────────────────────────────────
# Response parsing & decision rules
# ─────────────────────────────────────────────────────────────────────────────

def parse_screen_response(text: str) -> dict[str, Any]:
    """Parse key-value screening response. Robust to 4B model quirks."""
    result: dict[str, Any] = {"_raw": text}

    m = re.search(r"L1_SCORE:\s*(\d)", text, re.IGNORECASE)
    result["l1_score"] = int(m.group(1)) if m else None

    m = re.search(r"EST_PHASES:\s*(\d+)", text, re.IGNORECASE)
    result["est_phases"] = int(m.group(1)) if m else None

    m = re.search(r"L2_SCORE:\s*(\d)", text, re.IGNORECASE)
    result["l2_score"] = int(m.group(1)) if m else None

    m = re.search(r"EST_EVENTS:\s*(\d+)", text, re.IGNORECASE)
    result["est_events"] = int(m.group(1)) if m else None

    m = re.search(r"DOMAIN_L1:\s*(\S+)", text, re.IGNORECASE)
    if m:
        val = m.group(1).lower().strip().rstrip(".,;:")
        result["domain_l1"] = val if val in VALID_DOMAIN_L1 else None
    else:
        result["domain_l1"] = None

    m = re.search(r"DOMAIN_L2:\s*(\S+)", text, re.IGNORECASE)
    if m:
        val = m.group(1).lower().strip().rstrip(".,;:")
        result["domain_l2"] = val if val in VALID_DOMAIN_L2 else None
    else:
        result["domain_l2"] = None

    m = re.search(r"QUALITY:\s*(good|bad)", text, re.IGNORECASE)
    result["quality"] = m.group(1).lower() if m else None

    m = re.search(r"REASON:\s*(.+)", text, re.IGNORECASE)
    result["reason"] = m.group(1).strip() if m else None

    return result


def apply_screening_rules(
    parsed: dict[str, Any],
    l1_threshold: int = 3,
    l2_threshold: int = 3,
) -> str:
    """Programmatic decision rules. Returns 'keep' or 'reject'.

    Requires BOTH L1 (phase) and L2 (event) scores to meet thresholds,
    ensuring the video has at least two layers of temporal structure.
    """
    l1 = parsed.get("l1_score")
    l2 = parsed.get("l2_score")

    # Parse failure on critical field → reject
    if l1 is None or l2 is None:
        return "reject"

    # Bad visual quality → hard reject
    if parsed.get("quality") == "bad":
        return "reject"

    # Both levels must meet threshold → at least 2-layer structure
    if l1 < l1_threshold or l2 < l2_threshold:
        return "reject"

    return "keep"


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# vLLM inference (adapted from offline_rollout_filter.py)
# ─────────────────────────────────────────────────────────────────────────────

def build_screen_messages(example: dict[str, Any]) -> list[dict[str, Any]]:
    """Build screening prompt messages with <video> placeholder."""
    duration = example.get("duration", 0)
    prompt_text = SCREEN_PROMPT_TEMPLATE.format(duration=duration)

    # Split on <video> to create interleaved content
    content: list[dict[str, str]] = []
    for idx, chunk in enumerate(prompt_text.split("<video>")):
        if idx != 0:
            content.append({"type": "video"})
        if chunk:
            content.append({"type": "text", "text": chunk})

    return [{"role": "user", "content": content}]


def build_prompt(example: dict[str, Any], processor) -> str:
    messages = build_screen_messages(example)
    return processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )


def init_vllm_backend(args: argparse.Namespace):
    from vllm import LLM, SamplingParams
    from verl.workers.rollout.vllm_rollout_spmd import _get_logit_bias

    print("[screen] Loading processor...", flush=True)
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    print(f"[screen] Processor loaded in {time.time()-t0:.1f}s", flush=True)

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

    print(f"[screen] Constructing vLLM engine (tp={args.tensor_parallel_size})...", flush=True)
    t0 = time.time()
    engine = LLM(**llm_kwargs)
    print(f"[screen] vLLM engine ready in {time.time()-t0:.1f}s", flush=True)

    sampling_params = SamplingParams(
        n=1,  # single generation for screening
        temperature=args.temperature,
        top_p=0.9,
        top_k=-1,
        seed=args.seed,
        max_tokens=args.max_new_tokens,
        logit_bias=_get_logit_bias(processor),
    )
    return processor, engine, sampling_params


def _load_video_frames(
    video_path: str,
    max_frames: int,
    video_fps: float,
    max_pixels: int,
    min_pixels: int,
    image_patch_size: int = 14,
):
    """Load video and return pre-processed frames + metadata for vLLM.

    Uses qwen_vl_utils.fetch_video (same backend as HF processor) to
    decode, sample, and resize frames.  Returns (tensor, metadata, fps).
    """
    from qwen_vl_utils.vision_process import fetch_video

    vision_info = {
        "video": video_path,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "max_frames": max_frames,
        "fps": video_fps,
    }
    # fetch_video returns ((tensor[T,C,H,W], metadata_dict), sample_fps)
    # image_patch_size must match the HF processor's patch_size so that
    # the resize factor (patch_size * merge_size) is consistent.
    result = fetch_video(
        vision_info,
        image_patch_size=image_patch_size,
        return_video_sample_fps=True,
        return_video_metadata=True,
    )
    (video_tensor, metadata), sample_fps = result
    return video_tensor, metadata, sample_fps


def _build_vllm_request(
    example: dict[str, Any],
    processor,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    videos = example.get("videos") or []

    if not videos:
        prompt = build_prompt(example, processor)
        return {
            "prompt_token_ids": processor.tokenizer.encode(prompt, add_special_tokens=False),
        }

    # Read patch_size from the HF processor so fetch_video uses the same
    # resize factor (patch_size * merge_size).  Default 14 for Qwen2.5-VL.
    video_proc = getattr(processor, "video_processor", None) or processor.image_processor
    patch_size = getattr(video_proc, "patch_size", 14)

    # Load video frames → (tensor[T,C,H,W], metadata, sample_fps)
    loaded = [
        _load_video_frames(
            v, args.max_frames, args.video_fps, args.max_pixels, args.min_pixels,
            image_patch_size=patch_size,
        )
        for v in videos
    ]
    # Build (tensor, metadata) tuples for vLLM multi_modal_data
    mm_tuples = [(t, m) for t, m, _fps in loaded]

    # Pass text prompt (not token_ids) so vLLM runs the full
    # _apply_hf_processor_text_mm path end-to-end:
    # correct pad-token expansion + pixel-value computation in one shot,
    # avoiding any mismatch between local and vLLM-internal HF processor.
    prompt = build_prompt(example, processor)

    return {
        "prompt": prompt,
        "multi_modal_data": {"video": mm_tuples},
        "mm_processor_kwargs": {
            "do_sample_frames": False,
            "do_resize": False,
        },
    }


def iter_vllm_batches(
    items: list[dict[str, Any]],
    processor,
    llm,
    sampling_params,
    args: argparse.Namespace,
    batch_size: int = 32,
):
    """Yield (global_idx, item, responses_or_exception) one mini-batch at a time."""
    total_batches = (len(items) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(items))
        batch_items = items[start:end]

        requests = []
        req_positions = []
        errors = {}

        for local_idx, item in enumerate(batch_items):
            try:
                req = _build_vllm_request(item, processor, args)
                if req is None:
                    errors[local_idx] = RuntimeError("Failed to build vLLM request")
                    continue
                requests.append(req)
                req_positions.append(local_idx)
            except Exception as exc:
                import traceback
                print(f"[screen] ERROR building request idx={start+local_idx}: {exc}", flush=True)
                traceback.print_exc()
                errors[local_idx] = exc

        for local_idx, exc in errors.items():
            yield start + local_idx, batch_items[local_idx], exc

        if requests:
            t0 = time.time()
            batch_outputs = llm.generate(
                requests, sampling_params=sampling_params, use_tqdm=False,
            )
            elapsed = time.time() - t0
            print(
                f"[screen] batch {batch_idx+1}/{total_batches}: "
                f"{len(requests)} requests in {elapsed:.1f}s",
                flush=True,
            )
            for req_idx, output in enumerate(batch_outputs):
                local_idx = req_positions[req_idx]
                yield start + local_idx, batch_items[local_idx], [o.text for o in output.outputs]


# ─────────────────────────────────────────────────────────────────────────────
# CLI & Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local VLM pre-screening for hierarchical annotation candidates.",
    )
    # I/O
    parser.add_argument("--input_jsonl", required=True, help="Input JSONL from sample_per_source.py")
    parser.add_argument("--output_jsonl", required=True, help="All results with _screen field")
    parser.add_argument("--keep_jsonl", required=True, help="Kept records (decision=keep)")
    parser.add_argument("--reject_jsonl", required=True, help="Rejected records (decision=reject)")
    parser.add_argument("--report_jsonl", default="", help="Optional detailed report")

    # Model
    parser.add_argument("--model_path", required=True, help="Path to local VLM (e.g. Qwen3-VL-4B-Instruct)")

    # vLLM params
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.80)
    parser.add_argument("--max_model_len", type=int, default=0)
    parser.add_argument("--max_num_batched_tokens", type=int, default=32768)
    parser.add_argument("--dtype", default="bfloat16")

    # Video params
    parser.add_argument("--video_fps", type=float, default=1.0)
    parser.add_argument("--max_frames", type=int, default=128)
    parser.add_argument("--max_pixels", type=int, default=49152)
    parser.add_argument("--min_pixels", type=int, default=3136)

    # Generation
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    # Screening
    parser.add_argument("--l1_threshold", type=int, default=3,
                        help="Min L1 (phase) score to keep (1-5)")
    parser.add_argument("--l2_threshold", type=int, default=3,
                        help="Min L2 (event) score to keep (1-5)")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Process at most N samples (0 = all)")

    # Data-parallel sharding
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)

    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    print(f"[screen] Loading: {args.input_jsonl}", flush=True)
    items = load_jsonl(args.input_jsonl)
    print(f"[screen] Loaded {len(items)} samples", flush=True)

    if args.max_samples > 0:
        items = items[:args.max_samples]

    # Data-parallel sharding
    if args.num_shards > 1:
        items = items[args.shard_id :: args.num_shards]
        print(f"[screen] Shard {args.shard_id}/{args.num_shards}: "
              f"processing {len(items)} samples", flush=True)

    # Init vLLM
    print(f"[screen] Initializing vLLM (model={args.model_path}, "
          f"tp={args.tensor_parallel_size}, "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')})",
          flush=True)
    processor, llm, sampling_params = init_vllm_backend(args)

    # Create output dirs
    for p in [args.output_jsonl, args.keep_jsonl, args.reject_jsonl]:
        Path(p).parent.mkdir(parents=True, exist_ok=True)
    if args.report_jsonl:
        Path(args.report_jsonl).parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    rejected = 0
    errors = 0
    kept_by_source: dict[str, int] = {}
    rejected_by_source: dict[str, int] = {}

    fout = open(args.output_jsonl, "w", encoding="utf-8")
    fkeep = open(args.keep_jsonl, "w", encoding="utf-8")
    freject = open(args.reject_jsonl, "w", encoding="utf-8")
    freport = open(args.report_jsonl, "w", encoding="utf-8") if args.report_jsonl else None

    try:
        processed = 0
        for idx, item, result in iter_vllm_batches(
            items, processor, llm, sampling_params, args,
            batch_size=args.batch_size,
        ):
            source = item.get("source", "unknown")

            if isinstance(result, Exception):
                # Error → reject
                errors += 1
                rejected += 1
                rejected_by_source[source] = rejected_by_source.get(source, 0) + 1
                item["_screen"] = {"error": str(result), "decision": "reject"}
                line = json.dumps(item, ensure_ascii=False) + "\n"
                fout.write(line)
                freject.write(line)
                if freport:
                    freport.write(json.dumps({
                        "index": idx, "source": source,
                        "error": str(result), "decision": "reject",
                    }, ensure_ascii=False) + "\n")
            else:
                response_text = result[0] if result else ""
                parsed = parse_screen_response(response_text)
                decision = apply_screening_rules(parsed, args.l1_threshold, args.l2_threshold)
                parsed["decision"] = decision

                item["_screen"] = parsed
                line = json.dumps(item, ensure_ascii=False) + "\n"
                fout.write(line)

                if decision == "keep":
                    fkeep.write(line)
                    kept += 1
                    kept_by_source[source] = kept_by_source.get(source, 0) + 1
                else:
                    freject.write(line)
                    rejected += 1
                    rejected_by_source[source] = rejected_by_source.get(source, 0) + 1

                if freport:
                    freport.write(json.dumps({
                        "index": idx, "source": source,
                        "decision": decision,
                        "l1_score": parsed.get("l1_score"),
                        "l2_score": parsed.get("l2_score"),
                        "est_phases": parsed.get("est_phases"),
                        "est_events": parsed.get("est_events"),
                        "domain_l1": parsed.get("domain_l1"),
                        "quality": parsed.get("quality"),
                        "reason": parsed.get("reason"),
                        "response": response_text,
                    }, ensure_ascii=False) + "\n")

            processed += 1
            fout.flush()
            fkeep.flush()
            freject.flush()
            if freport:
                freport.flush()

            if processed % args.log_every == 0 or processed == len(items):
                print(
                    f"[screen] processed={processed}/{len(items)} "
                    f"kept={kept} rejected={rejected} errors={errors}",
                    flush=True,
                )

    finally:
        fout.close()
        fkeep.close()
        freject.close()
        if freport:
            freport.close()

    # Summary
    total = kept + rejected
    print(f"\n[screen] Done. total={total} kept={kept} rejected={rejected} errors={errors}")
    if total > 0:
        print(f"[screen] Keep rate: {100.0 * kept / total:.1f}%")

    all_sources = sorted(set(list(kept_by_source) + list(rejected_by_source)))
    if all_sources:
        print(f"\n[screen] Per-source summary:")
        print(f"  {'source':<25}  {'kept':>6}  {'rejected':>8}  {'total':>6}  {'keep%':>7}")
        print("  " + "-" * 60)
        for s in all_sources:
            k = kept_by_source.get(s, 0)
            r = rejected_by_source.get(s, 0)
            t = k + r
            pct = f"{100.0 * k / t:.1f}%" if t else "n/a"
            print(f"  {s:<25}  {k:>6}  {r:>8}  {t:>6}  {pct:>7}")

    print(f"\n[screen] Output: {args.output_jsonl}")
    print(f"[screen] Kept:   {args.keep_jsonl} ({kept} records)")
    print(f"[screen] Reject: {args.reject_jsonl} ({rejected} records)")


if __name__ == "__main__":
    main()
