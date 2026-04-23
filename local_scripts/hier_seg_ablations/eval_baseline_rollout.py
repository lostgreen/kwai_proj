#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_baseline_rollout.py — Baseline rollout evaluation for hier-seg data.

Samples N records per level from built JSONL (val by default), runs model
rollouts, and compares segment vs. segment+hint performance.

Output format is compatible with ablation_comparison/server.py — each result
record preserves original videos/metadata/prompt/answer and adds response/reward.

Usage:
    # Evaluate val set (segment only)
    python eval_baseline_rollout.py \
        --input-dir /path/to/train/ \
        --model-path /m2v_intern/xuboshen/models/Qwen3-VL-4B-Instruct \
        --sample-per-level 50 \
        --output-dir ./eval_results/

    # Evaluate val set (segment + hint comparison)
    python eval_baseline_rollout.py \
        --input-dir /path/to/train/ \
        --model-path /m2v_intern/xuboshen/models/Qwen3-VL-4B-Instruct \
        --sample-per-level 50 \
        --use-hint \
        --output-dir ./eval_results/

    # Then visualize with ablation_comparison:
    python ablation_comparison/server.py \
        --setting segment:./eval_results/segment \
        --setting hint:./eval_results/segment_hint
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# ── Re-use infrastructure from offline_rollout_filter ──
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR.parent))  # for offline_rollout_filter

from offline_rollout_filter import (
    build_prompt,
    build_multi_modal_data,
    init_vllm_backend,
    load_reward_function,
    _build_vllm_request,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baseline rollout evaluation for hier-seg (segment vs. segment+hint)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    parser.add_argument("--input-dir", required=True,
                        help="V1 data dir (contains L1/L2/L3_seg sub-dirs)")
    parser.add_argument("--hint-input-dir", default="",
                        help="Optional separate hint-version data dir. If empty and --use-hint, "
                             "reads *_hint_clipped.jsonl from --input-dir.")
    parser.add_argument("--output-dir", required=True, help="Directory for eval results")
    parser.add_argument("--sample-per-level", type=int, default=50)
    parser.add_argument("--split", default="val", choices=["val", "train"],
                        help="Which split to evaluate (default: val)")
    parser.add_argument("--use-hint", action="store_true",
                        help="Also evaluate hint variant (reads *_hint*.jsonl alongside base)")
    parser.add_argument("--seed", type=int, default=42)

    # Model
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--num-rollouts", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=1024)

    # Video
    parser.add_argument("--video-fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=256)
    parser.add_argument("--max-pixels", type=int, default=49152)
    parser.add_argument("--min-pixels", type=int, default=3136)

    # vLLM
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=0)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--dtype", default="bfloat16")

    # Reward
    parser.add_argument("--reward-function", default="",
                        help="Reward function path:func (default: use hier_seg_reward)")
    return parser.parse_args()


def load_jsonl(path: str | Path) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def sample_per_level(
    input_dir: Path, n_per_level: int, seed: int,
    split: str = "val", suffix: str = "",
) -> dict[str, list[dict]]:
    """Load and sample n records per level from {split}{suffix}_clipped.jsonl files.

    Args:
        split: "val" or "train"
        suffix: "" for base, "_hint" for hint variant
    """
    rng = random.Random(seed)
    sampled: dict[str, list[dict]] = {}

    for level_name in ("L1", "L2", "L3_seg"):
        # Try clipped first, then raw JSONL
        candidates = [
            input_dir / level_name / f"{split}{suffix}_clipped.jsonl",
            input_dir / level_name / f"{split}{suffix}.jsonl",
        ]
        jsonl_path = None
        for p in candidates:
            if p.exists():
                jsonl_path = p
                break
        if jsonl_path is None:
            print(f"  WARN: no {split}{suffix} JSONL found for {level_name}, skipping")
            continue

        records = load_jsonl(jsonl_path)
        rng.shuffle(records)
        sampled[level_name] = records[:n_per_level]
        print(f"  {level_name}: sampled {len(sampled[level_name])}/{len(records)} from {jsonl_path.name}")

    return sampled


def inject_hint(record: dict) -> dict | None:
    """Create a hint variant of a record by appending hint text to the prompt.

    Returns None if no hint is available for this record.
    """
    meta = record.get("metadata", {})
    level = meta.get("level")

    # Try to find hint from metadata or the prompt itself
    # The hint should have been stored in annotations;
    # for evaluation we inject it into the prompt directly
    hint = meta.get("hint", "")
    if not hint:
        return None

    rec = json.loads(json.dumps(record))  # deep copy
    old_prompt = rec["prompt"]
    new_prompt = old_prompt + f"\n\nHint: {hint}"
    rec["prompt"] = new_prompt
    rec["messages"] = [{"role": "user", "content": new_prompt}]
    rec["metadata"] = dict(meta, condition="segment_hint")
    return rec


def run_rollouts_vllm(
    records: list[dict],
    processor,
    llm,
    sampling_params,
    args,
    batch_size: int = 16,
) -> list[dict]:
    """Run rollouts for a list of records using vLLM batch inference.

    Returns list of result dicts with: record, responses, rewards.
    """
    import time as _time
    from offline_rollout_filter import _build_vllm_request

    results = []
    total_batches = (len(records) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(records))
        batch = records[start:end]

        requests = []
        valid_indices = []
        for i, rec in enumerate(batch):
            try:
                req = _build_vllm_request(rec, processor, args)
                if req is not None:
                    requests.append(req)
                    valid_indices.append(i)
            except Exception as exc:
                print(f"  WARN: failed to build request for record {start+i}: {exc}")

        if not requests:
            continue

        _t0 = _time.time()
        batch_outputs = llm.generate(requests, sampling_params=sampling_params, use_tqdm=False)
        elapsed = _time.time() - _t0
        print(f"  batch {batch_idx+1}/{total_batches}: {len(requests)} requests in {elapsed:.1f}s")

        for req_idx, output in enumerate(batch_outputs):
            local_idx = valid_indices[req_idx]
            rec = batch[local_idx]
            responses = [o.text for o in output.outputs]
            results.append({
                "record": rec,
                "responses": responses,
            })

    return results


def score_results(
    results: list[dict], reward_fn, condition: str
) -> list[dict]:
    """Score each rollout response with the reward function.

    Output format is compatible with ablation_comparison/server.py:
    each record preserves original videos/metadata/prompt/answer and
    adds response (best) + reward (best).

    reward_fn is compute_score(reward_inputs: list[dict]) → list[dict]
    (batch interface compatible with EasyR1 BatchFunctionRewardManager).
    """
    scored = []
    for item in results:
        rec = item["record"]
        responses = item["responses"]

        # Build batch reward_inputs for all rollout responses of this record
        reward_inputs = [
            {
                "response": resp,
                "ground_truth": rec.get("answer", ""),
                "problem_type": rec.get("problem_type", ""),
            }
            for resp in responses
        ]

        try:
            score_dicts = reward_fn(reward_inputs)
            rewards = [float(d.get("overall", 0.0)) for d in score_dicts]
        except Exception:
            rewards = [0.0] * len(responses)

        best_idx = int(np.argmax(rewards)) if rewards else 0
        best_response = responses[best_idx] if responses else ""
        best_reward = rewards[best_idx] if rewards else 0.0

        # ablation_comparison compatible format
        scored.append({
            # Original record fields (for ablation_comparison)
            "videos": rec.get("videos", []),
            "metadata": rec.get("metadata", {}),
            "prompt": rec.get("prompt", ""),
            "answer": rec.get("answer", ""),
            "problem_type": rec.get("problem_type", ""),
            "data_type": rec.get("data_type", "video"),
            # Rollout results
            "response": best_response,
            "reward": best_reward,
            "step": 0,  # baseline = step 0
            "phase": "val",
            # Detailed stats
            "condition": condition,
            "n_rollouts": len(responses),
            "rewards": rewards,
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "std_reward": float(np.std(rewards)) if rewards else 0.0,
            "max_reward": best_reward,
        })

    return scored


def print_summary(scored: list[dict], condition: str) -> dict:
    """Print and return summary statistics."""
    by_level: dict[str, list[float]] = defaultdict(list)
    for item in scored:
        level = item.get("metadata", {}).get("level", "")
        by_level[str(level)].append(item["mean_reward"])

    print(f"\n{'='*50}")
    print(f"  Condition: {condition}")
    print(f"{'='*50}")
    summary = {}
    for level in sorted(by_level.keys()):
        rewards = by_level[level]
        mean = np.mean(rewards)
        std = np.std(rewards)
        print(f"  Level {level}: mean={mean:.4f} ± {std:.4f}  (n={len(rewards)})")
        summary[f"L{level}"] = {"mean": float(mean), "std": float(std), "n": len(rewards)}

    all_rewards = [r for rs in by_level.values() for r in rs]
    overall_mean = np.mean(all_rewards)
    print(f"  Overall:  mean={overall_mean:.4f}  (n={len(all_rewards)})")
    summary["overall"] = {"mean": float(overall_mean), "n": len(all_rewards)}
    return summary


def _write_results(scored: list[dict], path: Path) -> None:
    """Write scored results to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in scored:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  → Saved: {path}  ({len(scored)} records)")


def main() -> None:
    args = parse_args()

    # Translate hyphenated CLI args to underscored attributes for offline_rollout_filter compat
    args.video_fps = args.video_fps
    args.max_frames = args.max_frames
    args.max_pixels = args.max_pixels
    args.min_pixels = args.min_pixels
    args.tensor_parallel_size = args.tensor_parallel_size
    args.gpu_memory_utilization = args.gpu_memory_utilization
    args.max_model_len = args.max_model_len
    args.max_num_batched_tokens = args.max_num_batched_tokens
    args.num_rollouts = args.num_rollouts
    args.max_new_tokens = args.max_new_tokens

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load reward function ──
    reward_spec = args.reward_function
    if not reward_spec:
        reward_spec = str(REPO_ROOT / "verl/reward_function/hier_seg_reward.py") + ":compute_score"
    print(f"[eval] Loading reward function: {reward_spec}")
    reward_fn = load_reward_function(reward_spec)

    # ── Sample data (base: no hint) ──
    split = args.split
    print(f"\n[eval] Sampling {args.sample_per_level} records per level from {input_dir} ({split})")
    sampled = sample_per_level(input_dir, args.sample_per_level, args.seed,
                               split=split, suffix="")

    all_segment_records = []
    for level_name, records in sampled.items():
        for rec in records:
            rec.setdefault("metadata", {})["condition"] = "segment"
            all_segment_records.append(rec)

    print(f"\n[eval] Total segment records: {len(all_segment_records)}")

    # ── Sample hint data (if --use-hint) ──
    all_hint_records = []
    if args.use_hint:
        hint_dir = Path(args.hint_input_dir) if args.hint_input_dir else input_dir
        hint_suffix = "_hint" if not args.hint_input_dir else ""
        print(f"\n[eval] Sampling hint records from {hint_dir} ({split}, suffix='{hint_suffix}')")
        hint_sampled = sample_per_level(hint_dir, args.sample_per_level, args.seed,
                                        split=split, suffix=hint_suffix)
        for level_name, records in hint_sampled.items():
            for rec in records:
                rec.setdefault("metadata", {})["condition"] = "segment_hint"
                all_hint_records.append(rec)
        print(f"  Total hint records: {len(all_hint_records)}")

    # ── Initialize vLLM ──
    print(f"\n[eval] Initializing vLLM (model={args.model_path}, tp={args.tensor_parallel_size})")
    processor, llm, sampling_params = init_vllm_backend(args)

    # ── Condition 1: Direct segment ──
    print(f"\n[eval] Running rollouts: SEGMENT (no hint) ...")
    segment_results = run_rollouts_vllm(
        all_segment_records, processor, llm, sampling_params, args
    )
    segment_scored = score_results(segment_results, reward_fn, "segment")
    segment_summary = print_summary(segment_scored, "segment")

    # Save to output_dir/segment/ (ablation_comparison compatible)
    _write_results(segment_scored, output_dir / "segment" / "results.jsonl")

    # ── Condition 2: Segment with hint ──
    hint_summary = {}
    if all_hint_records:
        print(f"\n[eval] Running rollouts: SEGMENT + HINT ...")
        hint_results = run_rollouts_vllm(
            all_hint_records, processor, llm, sampling_params, args
        )
        hint_scored = score_results(hint_results, reward_fn, "segment_hint")
        hint_summary = print_summary(hint_scored, "segment_hint")

        _write_results(hint_scored, output_dir / "segment_hint" / "results.jsonl")

    # ── Save combined summary ──
    summary = {
        "segment": segment_summary,
        "segment_hint": hint_summary,
        "config": {
            "model": args.model_path,
            "sample_per_level": args.sample_per_level,
            "num_rollouts": args.num_rollouts,
            "temperature": args.temperature,
            "seed": args.seed,
            "split": split,
        },
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[eval] Summary saved: {summary_path}")

    # ── Print comparison ──
    if hint_summary:
        print(f"\n{'='*60}")
        print(f"  COMPARISON: segment vs. segment+hint")
        print(f"{'='*60}")
        for level_key in sorted(set(list(segment_summary.keys()) + list(hint_summary.keys()))):
            if level_key == "overall":
                continue
            seg_mean = segment_summary.get(level_key, {}).get("mean", 0)
            hint_mean = hint_summary.get(level_key, {}).get("mean", 0)
            delta = hint_mean - seg_mean
            print(f"  {level_key}: segment={seg_mean:.4f}  hint={hint_mean:.4f}  Δ={delta:+.4f}")
        seg_all = segment_summary.get("overall", {}).get("mean", 0)
        hint_all = hint_summary.get("overall", {}).get("mean", 0)
        print(f"  Overall: segment={seg_all:.4f}  hint={hint_all:.4f}  Δ={hint_all-seg_all:+.4f}")

    # ── Print ablation_comparison launch hint ──
    print(f"\n[eval] To visualize with ablation_comparison:")
    cmd_parts = [f"python ablation_comparison/server.py"]
    cmd_parts.append(f"  --setting segment:{output_dir / 'segment'}")
    if hint_summary:
        cmd_parts.append(f"  --setting hint:{output_dir / 'segment_hint'}")
    print("  " + " \\\n    ".join(cmd_parts))


if __name__ == "__main__":
    main()
