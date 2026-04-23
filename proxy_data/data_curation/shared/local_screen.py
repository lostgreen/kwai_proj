#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local VLM pre-screening for hierarchical annotation pipeline.

Uses a local Qwen VL model (via vLLM) to assess whether a video
is suitable for hierarchical temporal segmentation annotation.

Unified mode (--unified, recommended for 32B+ models) evaluates all
dimensions in a single pass:
  1. L1 Phase structure score (1-5)
  2. L2 Event structure score (1-5)
  3. Progression type (procedural / narrative / interwoven_or_repetitive)
  4. Order dependency (strict / loose / none)
  5. Domain classification (domain_l1 + domain_l2)
  6. Visual quality (good/bad)

Usage (from train/ directory):
    # 32B model, TP=2, unified mode
    python proxy_data/data_curation/shared/local_screen.py \
        --input_jsonl proxy_data/data_curation/results/et_instruct_164k/sample_dev.jsonl \
        --output_jsonl screen_results.jsonl \
        --keep_jsonl screen_keep.jsonl \
        --reject_jsonl screen_reject.jsonl \
        --model_path /m2v_intern/xuboshen/zgw/models/Qwen3-VL-32B-Instruct \
        --tensor_parallel_size 2 --unified

    # 2-GPU data parallel (4B model)
    for i in 0 1; do
        CUDA_VISIBLE_DEVICES=$i python proxy_data/data_curation/shared/local_screen.py \
            --input_jsonl sample_dev.jsonl \
            --output_jsonl screen_shard${i}.jsonl \
            --keep_jsonl keep_shard${i}.jsonl \
            --reject_jsonl reject_shard${i}.jsonl \
            --model_path /m2v_intern/xuboshen/models/Qwen3-VL-4B-Instruct \
            --shard_id $i --num_shards 2 --unified &
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


SECONDARY_SCREEN_PROMPT = """\
<video>
Watch this video clip ({duration:.0f}s). This video has passed initial quality and structure checks.

Your task is to evaluate the *quality of its temporal progression* — specifically, whether the video showcases diverse, meaningful event transitions (GOOD for training) or just repeats the same action in loops (BAD for training).

Evaluate on three independent dimensions:

Q1 — Progression Type (PROG_TYPE):
  What is the nature of the progression across the video's major segments? Choose ONE:
  procedural = Step-by-step physical progression toward a goal (e.g., crafting, cooking, building). The state of objects changes permanently between segments.
  narrative = Chronological journey or evolving situation (e.g., vlogs, travel, shifting topics). Driven by time, location, or intent shifts.
  repetitive_loop = The segments are identical or near-identical cycles of the same action (e.g., gym sets, repeated tricks, drilling the same movement).

Q2 — Visual Diversity (VISUAL_DIVERSITY):
  How much does the visual content change between the video's major segments?
  high = Drastic changes: different rooms/locations, different objects, different camera setups.
  medium = Same general setting, but clear shifts in tools, focus, or activity.
  low = The scene looks almost identical across all segments (typical for gym, static camera, repetitive drills).

Q3 — Order Dependency (ORDER_DEPENDENCY):
  If you randomly shuffled the chronological order of the major segments, what would happen?
  strict = The video would become physically impossible or logically absurd (e.g., eating before cooking).
  loose = It would feel disjointed but not impossible (e.g., showing the beach before breakfast in a vlog).
  none = Order doesn't matter. Segments are fully interchangeable (e.g., compilation of tricks, gym sets).

Answer in EXACTLY this format (one field per line):
PROG_TYPE: <procedural | narrative | repetitive_loop>
VISUAL_DIVERSITY: <high | medium | low>
ORDER_DEPENDENCY: <strict | loose | none>
REASON: <One sentence explaining how the visual state evolves (or fails to evolve) between segments.>"""


# ─────────────────────────────────────────────────────────────────────────────
# Unified single-pass prompt (for stronger models, merges Stage 1 + Stage 2)
# ─────────────────────────────────────────────────────────────────────────────

VALID_UNIFIED_PROG_TYPES = {"procedural", "narrative", "interwoven_or_repetitive"}

UNIFIED_SCREEN_PROMPT = """\
<video>
Watch this video clip ({duration:.0f}s) and perform a comprehensive analysis of its suitability for hierarchical temporal annotation.

**CRITICAL RULES FOR JUDGMENT:**
1.  **A "Phase" MUST be a CONTINUOUS block of time.** A video that frequently cuts back and forth between two different scenes (e.g., an interview and a sports action) does NOT have multiple phases. Instead, it is a single, interwoven phase and should receive a LOW L1_SCORE (1 or 2).
2.  **DO NOT confuse editing cuts with event boundaries.** A simple camera angle change within the same ongoing action is NOT a new event. We are looking for changes in the *person's goal or the state of objects*.

### PART 1: Basic Structural Assessment

Q1 — L1 Macro-Phase Structure (L1_SCORE, integer 1-5):
  Does the video progress through distinct, logically ordered, and **continuous** high-level phases?
  5 = 4+ very distinct and continuous phases with different goals (e.g., procedural: prep → cook → plate; narrative: setup → conflict → resolution).
  4 = 3 distinct, continuous phases are identifiable.
  3 = 2 distinct, continuous phases are visible.
  2 = The activity has some variation, but lacks clear, continuous phase boundaries. It might be interwoven or a montage.
  1 = A single continuous activity OR a video that constantly cuts back and forth between different content types (interview/action).

Q2 — L2 Sub-Event Structure (L2_SCORE, integer 1-5):
  Within the potential phases, are there identifiable, distinct sub-events based on **goal shifts or object state changes** (not just camera cuts)?
  5 = Most phases are clearly composed of multiple sub-events.
  4 = Most phases have 2+ identifiable sub-events.
  3 = Some phases have sub-events, others are monolithic.
  2 = Sub-events are barely distinguishable.
  1 = No clear sub-event structure within phases.

### PART 2: Deep Progression Analysis

Q3 — Progression Type (PROG_TYPE):
  Based on the video's overall flow, choose ONE primary type:
  procedural = A step-by-step process creating a final product or state. The order is critical.
  narrative = A chronological journey, story, or evolving situation. The order follows time, location, or intent.
  interwoven_or_repetitive = The video constantly switches between different scenes (like interview/action cuts) OR repeats the same action in loops. This is a BAD signal for L1 structure.

Q4 — Order Dependency (ORDER_DEPENDENCY):
  If you shuffled the main logical segments (ignoring interwoven editing cuts), would the video's logic break?
  strict = Yes, it would become physically impossible or narratively absurd.
  loose = It would feel weird or out of order, but still possible to understand.
  none = No, the order of the core actions does not matter at all.

### PART 3: General Information

Q5 — Domain Classification:
  domain_l1 choices: procedural | physical | lifestyle | entertainment | narrative | educational | other
  domain_l2 choices: cooking | construction_building | crafting_diy | repair_maintenance | sports | fitness_exercise | music_performance | beauty_grooming | cleaning_housework | gardening_outdoor | vehicle_operation | movie_scene | reality_show | animation | vlog | interview_talk | documentary | news_report | science_experiment | lecture_tutorial | other

Q6 — Visual Quality (QUALITY):
  good = Clear footage with meaningful, visible actions.
  bad = Blurry, static, screen recording, pure talking head, gaming footage, text-heavy slides.

Answer in EXACTLY this format (one field per line):
L1_SCORE: <integer 1-5>
L2_SCORE: <integer 1-5>
PROG_TYPE: <procedural | narrative | interwoven_or_repetitive>
ORDER_DEPENDENCY: <strict | loose | none>
DOMAIN_L1: <one word from the list>
DOMAIN_L2: <one word from the list>
QUALITY: <good or bad>
REASON: <One sentence explaining your core judgment, especially regarding the PROG_TYPE and ORDER_DEPENDENCY.>"""


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


VALID_PROG_TYPES = {"procedural", "narrative", "repetitive_loop"}
VALID_VISUAL_DIVERSITY = {"high", "medium", "low"}
VALID_ORDER_DEPENDENCY = {"strict", "loose", "none"}


def parse_secondary_response(text: str) -> dict[str, Any]:
    """Parse key-value secondary screening response."""
    result: dict[str, Any] = {"_raw": text}

    m = re.search(r"PROG_TYPE:\s*(\S+)", text, re.IGNORECASE)
    if m:
        val = m.group(1).lower().strip().rstrip(".,;:")
        result["prog_type"] = val if val in VALID_PROG_TYPES else None
    else:
        result["prog_type"] = None

    m = re.search(r"VISUAL_DIVERSITY:\s*(\S+)", text, re.IGNORECASE)
    if m:
        val = m.group(1).lower().strip().rstrip(".,;:")
        result["visual_diversity"] = val if val in VALID_VISUAL_DIVERSITY else None
    else:
        result["visual_diversity"] = None

    m = re.search(r"ORDER_DEPENDENCY:\s*(\S+)", text, re.IGNORECASE)
    if m:
        val = m.group(1).lower().strip().rstrip(".,;:")
        result["order_dependency"] = val if val in VALID_ORDER_DEPENDENCY else None
    else:
        result["order_dependency"] = None

    m = re.search(r"REASON:\s*(.+)", text, re.IGNORECASE)
    result["reason"] = m.group(1).strip() if m else None

    return result


def apply_secondary_rules(parsed: dict[str, Any]) -> str:
    """Decision rules for secondary screening. Returns 'keep' or 'reject'.

    Rules:
    - repetitive_loop → hard reject
    - visual_diversity=low AND order_dependency=none → hard reject
    - procedural/narrative with at least medium diversity OR loose order → keep
    - Parse failure on critical field → reject
    """
    prog = parsed.get("prog_type")
    vis = parsed.get("visual_diversity")
    order = parsed.get("order_dependency")

    # Parse failure → reject
    if prog is None:
        return "reject"

    # Repetitive loop → hard reject
    if prog == "repetitive_loop":
        return "reject"

    # Double-low: no diversity AND no order dependency → reject
    if vis == "low" and order == "none":
        return "reject"

    return "keep"


def parse_unified_response(text: str) -> dict[str, Any]:
    """Parse unified single-pass screening response (all fields in one shot)."""
    result: dict[str, Any] = {"_raw": text}

    m = re.search(r"L1_SCORE:\s*(\d)", text, re.IGNORECASE)
    result["l1_score"] = int(m.group(1)) if m else None

    m = re.search(r"L2_SCORE:\s*(\d)", text, re.IGNORECASE)
    result["l2_score"] = int(m.group(1)) if m else None

    m = re.search(r"PROG_TYPE:\s*(\S+)", text, re.IGNORECASE)
    if m:
        val = m.group(1).lower().strip().rstrip(".,;:")
        result["prog_type"] = val if val in VALID_UNIFIED_PROG_TYPES else None
    else:
        result["prog_type"] = None

    m = re.search(r"ORDER_DEPENDENCY:\s*(\S+)", text, re.IGNORECASE)
    if m:
        val = m.group(1).lower().strip().rstrip(".,;:")
        result["order_dependency"] = val if val in VALID_ORDER_DEPENDENCY else None
    else:
        result["order_dependency"] = None

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


def apply_unified_rules(
    parsed: dict[str, Any],
    l1_threshold: int = 3,
    l2_threshold: int = 3,
) -> str:
    """Unified decision rules (merges Stage 1 + Stage 2 logic). Returns 'keep' or 'reject'.

    Reject if ANY of:
      - Parse failure on L1 or L2 score
      - quality == bad
      - L1 < threshold or L2 < threshold
      - prog_type == repetitive_or_unconnected
      - order_dependency == none (fully interchangeable segments)
    """
    l1 = parsed.get("l1_score")
    l2 = parsed.get("l2_score")
    prog = parsed.get("prog_type")
    order = parsed.get("order_dependency")

    # Parse failure on critical fields → reject
    if l1 is None or l2 is None:
        return "reject"

    # Bad visual quality → hard reject
    if parsed.get("quality") == "bad":
        return "reject"

    # Both levels must meet threshold
    if l1 < l1_threshold or l2 < l2_threshold:
        return "reject"

    # Interwoven or repetitive → hard reject
    if prog == "interwoven_or_repetitive":
        return "reject"

    # No order dependency at all → reject
    if order == "none":
        return "reject"

    return "keep"


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _get_item_key(item: dict[str, Any]) -> str:
    """Extract a unique key for a video record.

    Priority: metadata.clip_key > metadata.video_id > videos[0] stem > hash.
    """
    meta = item.get("metadata") or {}
    if meta.get("clip_key"):
        return meta["clip_key"]
    if meta.get("video_id"):
        return meta["video_id"]
    videos = item.get("videos") or []
    if videos:
        return Path(videos[0]).stem
    # Fallback: deterministic hash of the JSON
    return str(hash(json.dumps(item, sort_keys=True, ensure_ascii=True)))


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


def _load_existing_results(output_path: str) -> tuple[set[str], list[dict[str, Any]]]:
    """Load already-processed results. Returns (processed_keys, existing_items)."""
    if not os.path.exists(output_path):
        return set(), []
    existing = load_jsonl(output_path)
    keys = {_get_item_key(item) for item in existing}
    return keys, existing


# ─────────────────────────────────────────────────────────────────────────────
# vLLM inference (adapted from offline_rollout_filter.py)
# ─────────────────────────────────────────────────────────────────────────────

def build_screen_messages(
    example: dict[str, Any],
    prompt_template: str = SCREEN_PROMPT_TEMPLATE,
) -> list[dict[str, Any]]:
    """Build screening prompt messages with <video> placeholder."""
    duration = example.get("duration", 0)
    prompt_text = prompt_template.format(duration=duration)

    # Split on <video> to create interleaved content
    content: list[dict[str, str]] = []
    for idx, chunk in enumerate(prompt_text.split("<video>")):
        if idx != 0:
            content.append({"type": "video"})
        if chunk:
            content.append({"type": "text", "text": chunk})

    return [{"role": "user", "content": content}]


def build_prompt(
    example: dict[str, Any],
    processor,
    prompt_template: str = SCREEN_PROMPT_TEMPLATE,
) -> str:
    messages = build_screen_messages(example, prompt_template=prompt_template)
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
    prompt_template: str = SCREEN_PROMPT_TEMPLATE,
) -> dict[str, Any] | None:
    videos = example.get("videos") or []

    if not videos:
        prompt = build_prompt(example, processor, prompt_template=prompt_template)
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
    prompt = build_prompt(example, processor, prompt_template=prompt_template)

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
    prompt_template: str = SCREEN_PROMPT_TEMPLATE,
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
                req = _build_vllm_request(item, processor, args, prompt_template=prompt_template)
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
    parser.add_argument("--secondary_screen", action="store_true",
                        help="Run a second-pass screening on stage-1 kept videos "
                             "to filter out repetitive-loop false positives")
    parser.add_argument("--secondary_screen_only", action="store_true",
                        help="Stage-2 only: read s1_keep_jsonl from a previous Stage-1 run, "
                             "run secondary screening, and rewrite outputs. "
                             "Skips Stage 1 entirely.")
    parser.add_argument("--unified", action="store_true",
                        help="Single-pass unified screening (merges Stage 1 + Stage 2 into "
                             "one prompt). Recommended for stronger models (e.g. 32B+).")
    parser.add_argument("--s1_keep_jsonl", default="",
                        help="Stage-1 keep file to read for --secondary_screen_only. "
                             "Defaults to --keep_jsonl if not set.")
    parser.add_argument("--s1_reject_jsonl", default="",
                        help="Stage-1 reject file to read for --secondary_screen_only. "
                             "Defaults to --reject_jsonl if not set.")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Process at most N samples (0 = all)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip videos already in output_jsonl (by clip_key)")
    parser.add_argument("--resume_from", default="",
                        help="Read already-processed results from this file instead of "
                             "output_jsonl. Useful for multi-GPU: point to the merged "
                             "screen_results.jsonl so each shard knows what was done.")

    # Data-parallel sharding
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)

    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)

    return parser.parse_args()


def _run_primary_screen(
    items: list[dict[str, Any]],
    processor,
    llm,
    sampling_params,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run primary screening. Returns (kept_items, rejected_items)."""
    kept_items: list[dict[str, Any]] = []
    rejected_items: list[dict[str, Any]] = []
    errors = 0
    processed = 0

    for idx, item, result in iter_vllm_batches(
        items, processor, llm, sampling_params, args,
        batch_size=args.batch_size,
        prompt_template=SCREEN_PROMPT_TEMPLATE,
    ):
        if isinstance(result, Exception):
            errors += 1
            item["_screen"] = {"error": str(result), "decision": "reject"}
            rejected_items.append(item)
        else:
            response_text = result[0] if result else ""
            parsed = parse_screen_response(response_text)
            decision = apply_screening_rules(parsed, args.l1_threshold, args.l2_threshold)
            parsed["decision"] = decision
            item["_screen"] = parsed

            if decision == "keep":
                kept_items.append(item)
            else:
                rejected_items.append(item)

        processed += 1
        if processed % args.log_every == 0 or processed == len(items):
            print(
                f"[screen-1] processed={processed}/{len(items)} "
                f"kept={len(kept_items)} rejected={len(rejected_items)} errors={errors}",
                flush=True,
            )

    return kept_items, rejected_items


def _run_secondary_screen(
    items: list[dict[str, Any]],
    processor,
    llm,
    sampling_params,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run secondary screening on stage-1 kept items. Returns (kept, rejected)."""
    kept_items: list[dict[str, Any]] = []
    rejected_items: list[dict[str, Any]] = []
    errors = 0
    processed = 0

    for idx, item, result in iter_vllm_batches(
        items, processor, llm, sampling_params, args,
        batch_size=args.batch_size,
        prompt_template=SECONDARY_SCREEN_PROMPT,
    ):
        if isinstance(result, Exception):
            errors += 1
            item["_screen_2"] = {"error": str(result), "decision": "reject"}
            rejected_items.append(item)
        else:
            response_text = result[0] if result else ""
            parsed = parse_secondary_response(response_text)
            decision = apply_secondary_rules(parsed)

            # Contradiction detection: stage-1 said L1>=4 but stage-2 says repetitive
            s1 = item.get("_screen", {})
            if parsed.get("prog_type") == "repetitive_loop" and (s1.get("l1_score") or 0) >= 4:
                parsed["_s1_contradiction"] = True

            parsed["decision"] = decision
            item["_screen_2"] = parsed

            if decision == "keep":
                kept_items.append(item)
            else:
                rejected_items.append(item)

        processed += 1
        if processed % args.log_every == 0 or processed == len(items):
            print(
                f"[screen-2] processed={processed}/{len(items)} "
                f"kept={len(kept_items)} rejected={len(rejected_items)} errors={errors}",
                flush=True,
            )

    return kept_items, rejected_items


def _run_unified_screen(
    items: list[dict[str, Any]],
    processor,
    llm,
    sampling_params,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Single-pass unified screening. Returns (kept_items, rejected_items)."""
    kept_items: list[dict[str, Any]] = []
    rejected_items: list[dict[str, Any]] = []
    errors = 0
    processed = 0

    for idx, item, result in iter_vllm_batches(
        items, processor, llm, sampling_params, args,
        batch_size=args.batch_size,
        prompt_template=UNIFIED_SCREEN_PROMPT,
    ):
        if isinstance(result, Exception):
            errors += 1
            item["_screen"] = {"error": str(result), "decision": "reject"}
            rejected_items.append(item)
        else:
            response_text = result[0] if result else ""
            parsed = parse_unified_response(response_text)
            decision = apply_unified_rules(parsed, args.l1_threshold, args.l2_threshold)
            parsed["decision"] = decision
            item["_screen"] = parsed

            if decision == "keep":
                kept_items.append(item)
            else:
                rejected_items.append(item)

        processed += 1
        if processed % args.log_every == 0 or processed == len(items):
            print(
                f"[screen-unified] processed={processed}/{len(items)} "
                f"kept={len(kept_items)} rejected={len(rejected_items)} errors={errors}",
                flush=True,
            )

    return kept_items, rejected_items


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    # ── Secondary-screen-only mode ──
    if args.secondary_screen_only:
        print(f"[screen] Stage-2 only mode: reading Stage-1 results", flush=True)

        # Stage 2 input: read the merged Stage 1 keep/reject files.
        # --s1_keep_jsonl / --s1_reject_jsonl specify the input explicitly
        # (needed for multi-GPU: input=merged, output=per-shard).
        # Falls back to --keep_jsonl / --reject_jsonl for single-GPU.
        s1_keep_path = args.s1_keep_jsonl or args.keep_jsonl
        s1_reject_path = args.s1_reject_jsonl or args.reject_jsonl

        if not os.path.exists(s1_keep_path):
            print(f"[screen] ERROR: --secondary_screen_only requires existing {s1_keep_path}",
                  flush=True)
            sys.exit(1)
        s1_kept = load_jsonl(s1_keep_path)
        s1_rejected = load_jsonl(s1_reject_path) if os.path.exists(s1_reject_path) else []
        print(f"[screen] Loaded {len(s1_kept)} kept + {len(s1_rejected)} rejected from Stage 1",
              flush=True)

        # Apply sharding to Stage 2 input (for multi-GPU parallelism)
        if args.num_shards > 1:
            s1_kept = s1_kept[args.shard_id :: args.num_shards]
            s1_rejected = s1_rejected[args.shard_id :: args.num_shards]
            print(f"[screen] Shard {args.shard_id}/{args.num_shards}: "
                  f"{len(s1_kept)} kept + {len(s1_rejected)} rejected", flush=True)

        if not s1_kept:
            print("[screen] No kept items to screen. Done.", flush=True)
            return

        # Init vLLM
        print(f"[screen] Initializing vLLM (model={args.model_path}, "
              f"tp={args.tensor_parallel_size})", flush=True)
        processor, llm, sampling_params = init_vllm_backend(args)

        print(f"\n[screen] ═══ Stage 2: Secondary screening ({len(s1_kept)} videos) ═══",
              flush=True)
        s2_kept, s2_rejected = _run_secondary_screen(s1_kept, processor, llm, sampling_params, args)
        final_kept = s2_kept
        final_rejected = s1_rejected + s2_rejected

        _write_outputs(final_kept, final_rejected, args)

        total = len(final_kept) + len(final_rejected)
        print(f"\n[screen] Done. total={total} kept={len(final_kept)} rejected={len(final_rejected)}")
        print(f"[screen] Stage 2: {len(final_kept)}/{len(s1_kept)} kept "
              f"({100.0 * len(final_kept) / len(s1_kept):.1f}%)")
        _print_source_summary(final_kept, final_rejected)
        print(f"\n[screen] Kept:   {args.keep_jsonl} ({len(final_kept)} records)")
        print(f"[screen] Reject: {args.reject_jsonl} ({len(final_rejected)} records)")
        return

    # ── Normal mode (Stage 1 + optional Stage 2) ──
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

    # Resume: skip already-processed videos
    existing_kept: list[dict[str, Any]] = []
    existing_rejected: list[dict[str, Any]] = []
    if args.resume:
        resume_path = args.resume_from or args.output_jsonl
        done_keys, existing_items = _load_existing_results(resume_path)
        if done_keys:
            before = len(items)
            items = [it for it in items if _get_item_key(it) not in done_keys]
            # Split existing items back into kept/rejected based on stored decision
            for ex in existing_items:
                s1 = ex.get("_screen", {})
                s2 = ex.get("_screen_2", {})
                # Final decision: stage-2 overrides stage-1 if present
                final_decision = s2.get("decision") if s2 else s1.get("decision")
                if final_decision == "keep":
                    existing_kept.append(ex)
                else:
                    existing_rejected.append(ex)
            print(f"[screen] Resume from {resume_path}: {len(done_keys)} already done, "
                  f"{before - len(items)} skipped, {len(items)} remaining", flush=True)

    if not items:
        print("[screen] Nothing to process (all done or empty input).", flush=True)
        # Still write merged output in case this is a resume with nothing new
        if args.resume and (existing_kept or existing_rejected):
            _write_outputs(existing_kept, existing_rejected, args)
        return

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

    # ── Unified single-pass screening ──
    if args.unified:
        print(f"\n[screen] ═══ Unified screening ({len(items)} videos) ═══", flush=True)
        new_kept, new_rejected = _run_unified_screen(items, processor, llm, sampling_params, args)
    else:
        # ── Stage 1: Primary screening ──
        print(f"\n[screen] ═══ Stage 1: Primary screening ({len(items)} videos) ═══", flush=True)
        s1_kept, s1_rejected = _run_primary_screen(items, processor, llm, sampling_params, args)

        # ── Stage 2: Secondary screening (optional) ──
        if args.secondary_screen and s1_kept:
            print(f"\n[screen] ═══ Stage 2: Secondary screening ({len(s1_kept)} videos) ═══", flush=True)
            s2_kept, s2_rejected = _run_secondary_screen(s1_kept, processor, llm, sampling_params, args)
            new_kept = s2_kept
            new_rejected = s1_rejected + s2_rejected
        else:
            new_kept = s1_kept
            new_rejected = s1_rejected

    # Merge with existing results (resume mode)
    final_kept = existing_kept + new_kept
    final_rejected = existing_rejected + new_rejected

    _write_outputs(final_kept, final_rejected, args)

    # ── Summary ──
    total = len(final_kept) + len(final_rejected)
    print(f"\n[screen] Done. total={total} kept={len(final_kept)} rejected={len(final_rejected)}")
    if total > 0:
        print(f"[screen] Keep rate: {100.0 * len(final_kept) / total:.1f}%")

    if not args.unified and args.secondary_screen:
        all_input = len(items) + len(existing_kept) + len(existing_rejected)
        s1_all_kept = len([it for it in final_kept + final_rejected
                          if it.get("_screen", {}).get("decision") == "keep"])
        print(f"[screen] Stage 1: {s1_all_kept}/{all_input} kept")
        print(f"[screen] Stage 2: {len(final_kept)}/{s1_all_kept} kept"
              if s1_all_kept > 0 else "")

    _print_source_summary(final_kept, final_rejected)

    print(f"\n[screen] Output: {args.output_jsonl}")
    print(f"[screen] Kept:   {args.keep_jsonl} ({len(final_kept)} records)")
    print(f"[screen] Reject: {args.reject_jsonl} ({len(final_rejected)} records)")


def _write_outputs(
    kept: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    """Write final output files.

    Output schema per record:
      Base fields (for hier_seg_annotation / extract_frames.py):
        videos, metadata{clip_key, video_id, clip_start, clip_end,
        clip_duration, original_duration, is_full_video, source},
        source, dataset, duration
      Screening fields:
        _screen{l1_score, est_phases, l2_score, est_events,
                domain_l1, domain_l2, quality, reason, decision}
        _screen_2{prog_type, visual_diversity, order_dependency,
                  reason, decision}  (only if secondary_screen)
    """
    def _clean_item(item: dict) -> dict:
        """Strip verbose _raw from screening dicts before writing."""
        out = dict(item)
        for key in ("_screen", "_screen_2"):
            if key in out and isinstance(out[key], dict):
                out[key] = {k: v for k, v in out[key].items() if k != "_raw"}
        return out

    for p in [args.output_jsonl, args.keep_jsonl, args.reject_jsonl]:
        Path(p).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_jsonl, "w", encoding="utf-8") as fout, \
         open(args.keep_jsonl, "w", encoding="utf-8") as fkeep, \
         open(args.reject_jsonl, "w", encoding="utf-8") as freject:
        for item in kept + rejected:
            fout.write(json.dumps(_clean_item(item), ensure_ascii=False) + "\n")
        for item in kept:
            fkeep.write(json.dumps(_clean_item(item), ensure_ascii=False) + "\n")
        for item in rejected:
            freject.write(json.dumps(_clean_item(item), ensure_ascii=False) + "\n")

    if args.report_jsonl:
        Path(args.report_jsonl).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report_jsonl, "w", encoding="utf-8") as freport:
            for item in kept + rejected:
                s1 = item.get("_screen", {})
                s2 = item.get("_screen_2", {})
                # In unified mode, prog_type/order_dependency are in _screen.
                # In two-stage mode, they are in _screen_2.
                freport.write(json.dumps({
                    "clip_key": _get_item_key(item),
                    "source": item.get("source", "unknown"),
                    "decision": s2.get("decision") or s1.get("decision"),
                    "l1_score": s1.get("l1_score"),
                    "l2_score": s1.get("l2_score"),
                    "domain_l1": s1.get("domain_l1"),
                    "domain_l2": s1.get("domain_l2"),
                    "quality": s1.get("quality"),
                    "prog_type": s1.get("prog_type") or s2.get("prog_type"),
                    "order_dependency": s1.get("order_dependency") or s2.get("order_dependency"),
                }, ensure_ascii=False) + "\n")


def _print_source_summary(
    kept: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
) -> None:
    kept_by_source: dict[str, int] = {}
    rejected_by_source: dict[str, int] = {}
    for item in kept:
        s = item.get("source", "unknown")
        kept_by_source[s] = kept_by_source.get(s, 0) + 1
    for item in rejected:
        s = item.get("source", "unknown")
        rejected_by_source[s] = rejected_by_source.get(s, 0) + 1

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


if __name__ == "__main__":
    main()
