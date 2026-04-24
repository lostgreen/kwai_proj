#!/usr/bin/env python3
"""Orchestrate stages for the temporal AoT hard-QA pipeline."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import random
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from proxy_data.shared.frame_cache import (  # noqa: E402
    SourceVideoInfo,
    build_source_cache_dir,
    ensure_source_frame_cache,
)

REQUIRED_MANIFEST_FIELDS = ("clip_key", "source_video_path")
REQUIRED_CACHE_META_FIELDS = ("fps", "duration_sec", "n_frames", "cache_dir")
DEFAULT_ROLLOUT_MODEL_PATH = "/m2v_intern/xuboshen/models/Qwen3-VL-8B-Instruct"
DEFAULT_REWARD_FUNCTION = f"{(REPO_ROOT / 'verl' / 'reward_function' / 'mixed_proxy_reward.py').resolve()}:compute_score"


def load_manifest_rows(manifest_paths: Iterable[str | Path]) -> list[dict]:
    rows: list[dict] = []
    for manifest_path in manifest_paths:
        path = Path(manifest_path)
        manifest_dir = path.resolve().parent
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    row = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON in {path}:{line_no}: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"expected object in {path}:{line_no}, got {type(row).__name__}")
                row = dict(row)
                row["_manifest_path"] = str(path.resolve())
                row["_manifest_dir"] = str(manifest_dir)
                rows.append(row)
    return rows


def load_jsonl_rows(jsonl_paths: Iterable[str | Path]) -> list[dict]:
    rows: list[dict] = []
    for jsonl_path in jsonl_paths:
        path = Path(jsonl_path)
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    row = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON in {path}:{line_no}: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"expected object in {path}:{line_no}, got {type(row).__name__}")
                rows.append(row)
    return rows


def _normalize_source_video_path(source_video_path: str, manifest_dir: str | Path | None = None) -> str:
    source_path = Path(source_video_path).expanduser()
    if not source_path.is_absolute():
        anchor_dir = Path(manifest_dir) if manifest_dir is not None else Path.cwd()
        source_path = anchor_dir / source_path
    return str(source_path.resolve(strict=False))


def collect_unique_source_infos(rows: Iterable[dict]) -> list[SourceVideoInfo]:
    by_source: dict[str, SourceVideoInfo] = {}
    for row in rows:
        missing = [field for field in REQUIRED_MANIFEST_FIELDS if not str(row.get(field, "")).strip()]
        if missing:
            raise ValueError(f"manifest row missing required fields: {', '.join(missing)}")

        clip_key = str(row["clip_key"]).strip()
        source_video_path = str(row["source_video_path"]).strip()
        source_identity = _normalize_source_video_path(
            source_video_path,
            manifest_dir=row.get("_manifest_dir"),
        )
        duration_sec = row.get("duration_sec", row.get("clip_duration_sec"))

        existing = by_source.get(source_identity)
        if existing is None:
            by_source[source_identity] = SourceVideoInfo(
                clip_key=clip_key,
                source_video_path=source_identity,
                duration_sec=duration_sec,
            )
            continue

        if existing.clip_key != clip_key:
            raise ValueError(
                "conflicting clip_key values for source video "
                f"{source_identity}: {existing.clip_key!r} vs {clip_key!r}"
            )

        if (
            existing.duration_sec is not None
            and duration_sec is not None
            and existing.duration_sec != duration_sec
        ):
            raise ValueError(
                "conflicting duration_sec values for source video "
                f"{source_identity}: {existing.duration_sec!r} vs {duration_sec!r}"
            )

        if existing.duration_sec is None and duration_sec is not None:
            by_source[source_identity] = SourceVideoInfo(
                clip_key=existing.clip_key,
                source_video_path=existing.source_video_path,
                duration_sec=duration_sec,
            )

    return sorted(by_source.values(), key=lambda item: (item.clip_key, item.source_video_path))


def build_expected_cache_record(
    source_info: SourceVideoInfo,
    frames_root: str | Path,
    fps: float,
    jpeg_quality: int,
) -> dict:
    cache_dir = build_source_cache_dir(
        frames_root=frames_root,
        clip_key=source_info.clip_key,
        source_video_path=source_info.source_video_path,
        fps=fps,
    )
    return {
        "clip_key": source_info.clip_key,
        "source_video_path": source_info.source_video_path,
        "cache_dir": str(cache_dir.resolve()),
        "fps": fps,
        "jpeg_quality": jpeg_quality,
        "duration_sec": source_info.duration_sec,
        "n_frames": None,
        "status": "dry-run",
    }


def validate_cache_metadata(meta: dict) -> dict:
    missing = [field for field in REQUIRED_CACHE_META_FIELDS if field not in meta]
    if missing:
        raise ValueError(f"cache metadata missing required fields: {', '.join(missing)}")

    fps = meta["fps"]
    duration_sec = meta["duration_sec"]
    n_frames = meta["n_frames"]
    cache_dir = meta["cache_dir"]

    if not isinstance(fps, (int, float)) or fps <= 0:
        raise ValueError(f"invalid cache metadata fps: {fps!r}")
    if duration_sec is not None and (not isinstance(duration_sec, (int, float)) or duration_sec <= 0):
        raise ValueError(f"invalid cache metadata duration_sec: {duration_sec!r}")
    if not isinstance(n_frames, int) or n_frames <= 0:
        raise ValueError(f"invalid cache metadata n_frames: {n_frames!r}")
    if not str(cache_dir).strip():
        raise ValueError("invalid cache metadata cache_dir: empty")

    validated = dict(meta)
    validated["clip_key"] = str(meta.get("clip_key", "")).strip()
    validated["source_video_path"] = _normalize_source_video_path(str(meta.get("source_video_path", "")).strip())
    if not validated["clip_key"]:
        raise ValueError("invalid cache metadata clip_key: empty")
    if not validated["source_video_path"]:
        raise ValueError("invalid cache metadata source_video_path: empty")
    validated["cache_dir"] = str(Path(cache_dir).expanduser().resolve(strict=False))
    validated["status"] = str(meta.get("status", "ready"))
    return validated


def freeze_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple((str(key), freeze_json_value(val)) for key, val in sorted(value.items(), key=lambda item: str(item[0])))
    if isinstance(value, list):
        return tuple(freeze_json_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(freeze_json_value(item) for item in value)
    return value


def build_raw_record_dedupe_key(record: dict) -> tuple[Any, ...]:
    return (
        str(record.get("problem_type", "")),
        str(record.get("prompt", "")),
        str(record.get("answer", "")),
        freeze_json_value(record.get("videos") or []),
    )


def _normalize_duration_for_stats(record: dict) -> float | None:
    duration = extract_duration_sec(record)
    if duration is None or duration <= 0:
        return None
    return duration


def _stats_relevant_metadata(record: dict) -> dict[str, Any]:
    return {
        "domain_l1": _extract_domain(record, "domain_l1"),
        "domain_l2": _extract_domain(record, "domain_l2"),
        "duration_sec": _normalize_duration_for_stats(record),
    }


def dedupe_raw_records(records: Iterable[dict]) -> tuple[list[dict], int]:
    deduped: list[dict] = []
    seen: dict[tuple[Any, ...], dict[str, Any]] = {}
    duplicate_count = 0
    for record in records:
        key = build_raw_record_dedupe_key(record)
        existing = seen.get(key)
        if existing is not None:
            existing_stats = _stats_relevant_metadata(existing)
            current_stats = _stats_relevant_metadata(record)
            if existing_stats != current_stats:
                raise ValueError(
                    "conflicting stats metadata for duplicate raw record key: "
                    f"{json.dumps({'existing': existing_stats, 'current': current_stats}, ensure_ascii=False, sort_keys=True)}"
                )
            duplicate_count += 1
            continue
        seen[key] = record
        deduped.append(record)
    return deduped, duplicate_count


def _record_sort_key(record: dict) -> tuple[str, str, str]:
    return (
        str(record.get("problem_type", "")),
        json.dumps(build_raw_record_dedupe_key(record), ensure_ascii=False, sort_keys=False),
        json.dumps(freeze_json_value(record), ensure_ascii=False, sort_keys=False),
    )


def _extract_metadata(record: dict) -> dict:
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        return metadata
    return {}


def _extract_domain(record: dict, key: str) -> str:
    metadata = _extract_metadata(record)
    value = metadata.get(key, record.get(key, "other"))
    text = str(value).strip()
    return text or "other"


def _count_video_frames(video_entry: Any) -> int:
    if isinstance(video_entry, (list, tuple)):
        return sum(_count_video_frames(item) for item in video_entry)
    if video_entry is None:
        return 0
    return 1


def extract_frame_count(record: dict) -> int:
    videos = record.get("videos") or []
    if not isinstance(videos, list):
        return 0
    return sum(_count_video_frames(video) for video in videos)


def extract_duration_sec(record: dict) -> float | None:
    metadata = _extract_metadata(record)
    for value in (
        metadata.get("total_duration_sec"),
        metadata.get("duration_sec"),
        record.get("duration_sec"),
    ):
        if isinstance(value, (int, float)):
            return float(value)
    return None


def summarize_raw_records(records: Iterable[dict]) -> dict:
    records = list(records)
    by_problem_type = Counter(str(record.get("problem_type", "unknown")) for record in records)
    by_domain_l1 = Counter(_extract_domain(record, "domain_l1") for record in records)
    by_domain_l2 = Counter(_extract_domain(record, "domain_l2") for record in records)
    frame_counts = [extract_frame_count(record) for record in records]
    durations = [duration for duration in (_normalize_duration_for_stats(record) for record in records) if duration is not None]
    return {
        "total_count": len(records),
        "by_problem_type": dict(sorted(by_problem_type.items())),
        "by_domain_l1": dict(sorted(by_domain_l1.items())),
        "by_domain_l2": dict(sorted(by_domain_l2.items())),
        "average_frame_count": (sum(frame_counts) / len(frame_counts)) if frame_counts else 0.0,
        "average_duration_sec": (sum(durations) / len(durations)) if durations else 0.0,
    }


def split_raw_records(
    records: Iterable[dict],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio!r}")

    records = list(records)
    total_count = len(records)
    if total_count == 0:
        return [], []

    by_problem_type: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_problem_type[str(record.get("problem_type", "unknown"))].append(record)

    rng = random.Random(seed)
    target_val_count = int(math.floor(total_count * val_ratio))
    target_val_count = min(target_val_count, max(total_count - 1, 0))
    if val_ratio > 0 and target_val_count == 0 and total_count > 1:
        target_val_count = 1

    groups: dict[str, list[dict]] = {}
    for problem_type in sorted(by_problem_type):
        group = sorted(by_problem_type[problem_type], key=_record_sort_key)
        rng.shuffle(group)
        groups[problem_type] = group

    val_allocations = {problem_type: 0 for problem_type in groups}
    if target_val_count > 0:
        coverage_candidates = [problem_type for problem_type, group in groups.items() if len(group) >= 2]
        rng.shuffle(coverage_candidates)
        for problem_type in coverage_candidates[:target_val_count]:
            val_allocations[problem_type] = 1

        remaining_budget = target_val_count - sum(val_allocations.values())
        if remaining_budget > 0:
            fractional_needs: list[tuple[float, float, str]] = []
            for problem_type, group in groups.items():
                ideal = len(group) * val_ratio
                remaining_capacity = len(group) - 1 - val_allocations[problem_type]
                if remaining_capacity <= 0:
                    continue
                fractional_needs.append((ideal - val_allocations[problem_type], rng.random(), problem_type))

            fractional_needs.sort(key=lambda item: (-item[0], item[1], item[2]))
            for _, _, problem_type in fractional_needs:
                if remaining_budget <= 0:
                    break
                capacity = len(groups[problem_type]) - 1 - val_allocations[problem_type]
                if capacity <= 0:
                    continue
                take = min(capacity, remaining_budget)
                val_allocations[problem_type] += take
                remaining_budget -= take

    train: list[dict] = []
    val: list[dict] = []
    for problem_type in sorted(groups):
        group = groups[problem_type]
        planned_val = val_allocations[problem_type]
        val.extend(group[:planned_val])
        train.extend(group[planned_val:])

    train.sort(key=_record_sort_key)
    val.sort(key=_record_sort_key)
    return train, val


def write_jsonl_rows(records: Iterable[dict], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def merge_raw_stage(
    input_paths: Iterable[str | Path],
    output_dir: str | Path,
    seed: int = 42,
    val_ratio: float = 0.1,
    dry_run: bool = False,
) -> dict:
    input_paths = [Path(path) for path in input_paths]
    output_dir = Path(output_dir)
    records = load_jsonl_rows(input_paths)
    deduped_records, duplicate_count = dedupe_raw_records(records)
    train_records, val_records = split_raw_records(deduped_records, val_ratio=val_ratio, seed=seed)

    summary = {
        "stage": "merge-raw",
        "input_paths": [str(path.resolve()) for path in input_paths],
        "output_dir": str(output_dir.resolve()),
        "train_output_path": str((output_dir / "train.jsonl").resolve()),
        "val_output_path": str((output_dir / "val.jsonl").resolve()),
        "stats_output_path": str((output_dir / "stats.json").resolve()),
        "seed": seed,
        "val_ratio": val_ratio,
        "dry_run": dry_run,
        "input_record_count": len(records),
        "deduped_record_count": len(deduped_records),
        "duplicate_record_count": duplicate_count,
        "train_count": len(train_records),
        "val_count": len(val_records),
        "all": summarize_raw_records(deduped_records),
        "train": summarize_raw_records(train_records),
        "val": summarize_raw_records(val_records),
    }

    if dry_run:
        return summary

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_rows(train_records, output_dir / "train.jsonl")
    write_jsonl_rows(val_records, output_dir / "val.jsonl")
    write_stats_output(summary, output_dir / "stats.json")
    return summary


def build_source_cache_stage(
    manifest_paths: Iterable[str | Path],
    frames_root: str | Path,
    fps: float = 2.0,
    jpeg_quality: int = 2,
    overwrite: bool = False,
    workers: int = 1,
    dry_run: bool = False,
) -> dict:
    manifest_paths = [Path(path) for path in manifest_paths]
    rows = load_manifest_rows(manifest_paths)
    source_infos = collect_unique_source_infos(rows)
    frames_root = Path(frames_root)

    summary = {
        "stage": "build-source-cache",
        "manifest_paths": [str(path.resolve()) for path in manifest_paths],
        "frames_root": str(frames_root.resolve()),
        "fps": fps,
        "jpeg_quality": jpeg_quality,
        "overwrite": overwrite,
        "dry_run": dry_run,
        "input_row_count": len(rows),
        "unique_source_count": len(source_infos),
        "caches": [],
    }

    if dry_run:
        summary["caches"] = [
            build_expected_cache_record(
                source_info=info,
                frames_root=frames_root,
                fps=fps,
                jpeg_quality=jpeg_quality,
            )
            for info in source_infos
        ]
        return summary

    frames_root.mkdir(parents=True, exist_ok=True)
    worker_count = max(1, workers)
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map = {
            executor.submit(
                ensure_source_frame_cache,
                info,
                frames_root,
                fps,
                jpeg_quality,
                overwrite,
            ): info
            for info in source_infos
        }
        for future in as_completed(future_map):
            result = validate_cache_metadata(future.result())
            results.append(result)

    summary["caches"] = sorted(results, key=lambda item: (item.get("clip_key", ""), item["cache_dir"]))
    return summary


def write_stats_output(summary: dict, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _build_rollout_filter_paths(output_dir: str | Path) -> dict[str, str]:
    output_dir = Path(output_dir)
    return {
        "output_dir": str(output_dir.resolve()),
        "rollout_output_path": str((output_dir / "rollout_output.jsonl").resolve()),
        "rollout_report_path": str((output_dir / "rollout_report.jsonl").resolve()),
        "hard_cases_output_path": str((output_dir / "hard_cases.jsonl").resolve()),
        "hard_cases_stats_output_path": str((output_dir / "hard_cases.stats.json").resolve()),
    }


def _build_rollout_command(
    *,
    input_path: str | Path,
    rollout_output_path: str | Path,
    rollout_report_path: str | Path,
    model_path: str,
    reward_function: str,
    backend: str,
    num_rollouts: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    video_fps: float,
    max_frames: int,
    max_pixels: int,
    min_pixels: int,
    max_samples: int,
    min_frames: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    max_num_batched_tokens: int,
    batch_size: int,
    dtype: str,
    seed: int,
) -> list[str]:
    return [
        sys.executable,
        str((REPO_ROOT / "local_scripts" / "offline_rollout_filter.py").resolve()),
        "--input_jsonl",
        str(Path(input_path).resolve()),
        "--output_jsonl",
        str(Path(rollout_output_path).resolve()),
        "--report_jsonl",
        str(Path(rollout_report_path).resolve()),
        "--model_path",
        model_path,
        "--reward_function",
        reward_function,
        "--backend",
        backend,
        "--num_rollouts",
        str(num_rollouts),
        "--temperature",
        str(temperature),
        "--top_p",
        str(top_p),
        "--max_new_tokens",
        str(max_new_tokens),
        "--video_fps",
        str(video_fps),
        "--max_frames",
        str(max_frames),
        "--max_pixels",
        str(max_pixels),
        "--min_pixels",
        str(min_pixels),
        "--max_samples",
        str(max_samples),
        "--seed",
        str(seed),
        "--tensor_parallel_size",
        str(tensor_parallel_size),
        "--gpu_memory_utilization",
        str(gpu_memory_utilization),
        "--max_model_len",
        str(max_model_len),
        "--max_num_batched_tokens",
        str(max_num_batched_tokens),
        "--batch_size",
        str(batch_size),
        "--dtype",
        dtype,
        "--min_mean_reward",
        "0.0",
        "--max_mean_reward",
        "1.0",
        "--min_frames",
        str(min_frames),
    ]


def _build_filter_command(
    *,
    report_path: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    stats_output_path: str | Path,
    min_mean_reward: float,
    max_mean_reward: float,
    min_success_count: int,
    success_threshold: float,
    target_total: int,
    nested_balance_key: str,
    seed: int,
) -> list[str]:
    return [
        sys.executable,
        str((REPO_ROOT / "proxy_data" / "youcook2_seg" / "temporal_aot" / "filter_rollout_hard_cases.py").resolve()),
        "--report",
        str(Path(report_path).resolve()),
        "--input",
        str(Path(input_path).resolve()),
        "--output",
        str(Path(output_path).resolve()),
        "--stats-output",
        str(Path(stats_output_path).resolve()),
        "--min-mean-reward",
        str(min_mean_reward),
        "--max-mean-reward",
        str(max_mean_reward),
        "--min-success-count",
        str(min_success_count),
        "--success-threshold",
        str(success_threshold),
        "--target-total",
        str(target_total),
        "--nested-balance-key",
        nested_balance_key,
        "--seed",
        str(seed),
    ]


def _read_json_file(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _run_subprocess(command: list[str]) -> None:
    subprocess.run(command, check=True, cwd=str(REPO_ROOT))


def rollout_filter_stage(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    model_path: str = DEFAULT_ROLLOUT_MODEL_PATH,
    reward_function: str = DEFAULT_REWARD_FUNCTION,
    backend: str = "vllm",
    num_rollouts: int = 8,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 1024,
    video_fps: float = 2.0,
    max_frames: int = 256,
    max_pixels: int = 49152,
    min_pixels: int = 3136,
    max_samples: int = 0,
    min_frames: int = 0,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.8,
    max_model_len: int = 0,
    max_num_batched_tokens: int = 16384,
    batch_size: int = 32,
    dtype: str = "bfloat16",
    min_mean_reward: float = 0.125,
    max_mean_reward: float = 0.625,
    min_success_count: int = 1,
    success_threshold: float = 1.0,
    target_total: int = 5000,
    nested_balance_key: str = "domain_l1",
    seed: int = 42,
    dry_run: bool = False,
) -> dict:
    input_path = Path(input_path)
    output_paths = _build_rollout_filter_paths(output_dir)
    input_records = load_jsonl_rows([input_path])

    rollout_command = _build_rollout_command(
        input_path=input_path,
        rollout_output_path=output_paths["rollout_output_path"],
        rollout_report_path=output_paths["rollout_report_path"],
        model_path=model_path,
        reward_function=reward_function,
        backend=backend,
        num_rollouts=num_rollouts,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        video_fps=video_fps,
        max_frames=max_frames,
        max_pixels=max_pixels,
        min_pixels=min_pixels,
        max_samples=max_samples,
        min_frames=min_frames,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        batch_size=batch_size,
        dtype=dtype,
        seed=seed,
    )
    filter_command = _build_filter_command(
        report_path=output_paths["rollout_report_path"],
        input_path=input_path,
        output_path=output_paths["hard_cases_output_path"],
        stats_output_path=output_paths["hard_cases_stats_output_path"],
        min_mean_reward=min_mean_reward,
        max_mean_reward=max_mean_reward,
        min_success_count=min_success_count,
        success_threshold=success_threshold,
        target_total=target_total,
        nested_balance_key=nested_balance_key,
        seed=seed,
    )

    summary = {
        "stage": "rollout-filter",
        "input_path": str(input_path.resolve()),
        "input_record_count": len(input_records),
        "input_summary": summarize_raw_records(input_records),
        "model_path": model_path,
        "reward_function": reward_function,
        "backend": backend,
        "num_rollouts": num_rollouts,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "video_fps": video_fps,
        "max_frames": max_frames,
        "max_pixels": max_pixels,
        "min_pixels": min_pixels,
        "max_samples": max_samples,
        "min_frames": min_frames,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "max_num_batched_tokens": max_num_batched_tokens,
        "batch_size": batch_size,
        "dtype": dtype,
        "min_mean_reward": min_mean_reward,
        "max_mean_reward": max_mean_reward,
        "min_success_count": min_success_count,
        "success_threshold": success_threshold,
        "target_total": target_total,
        "nested_balance_key": nested_balance_key,
        "seed": seed,
        "dry_run": dry_run,
        **output_paths,
        "planned_commands": [
            {"stage": "rollout", "argv": rollout_command, "display": shlex.join(rollout_command)},
            {"stage": "filter", "argv": filter_command, "display": shlex.join(filter_command)},
        ],
    }

    if dry_run:
        return summary

    Path(output_paths["output_dir"]).mkdir(parents=True, exist_ok=True)
    _run_subprocess(rollout_command)
    _run_subprocess(filter_command)

    rollout_records = load_jsonl_rows([output_paths["rollout_output_path"]])
    report_records = load_jsonl_rows([output_paths["rollout_report_path"]])
    hard_case_records = load_jsonl_rows([output_paths["hard_cases_output_path"]])
    summary.update(
        {
            "rollout_output_record_count": len(rollout_records),
            "rollout_output_summary": summarize_raw_records(rollout_records),
            "report_record_count": len(report_records),
            "hard_case_count": len(hard_case_records),
            "hard_case_summary": summarize_raw_records(hard_case_records),
            "hard_case_filter_summary": _read_json_file(output_paths["hard_cases_stats_output_path"]),
        }
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_cache = subparsers.add_parser(
        "build-source-cache",
        help="Build or inspect the shared 2fps source-frame cache",
    )
    build_cache.add_argument(
        "--manifest",
        dest="manifests",
        action="append",
        required=True,
        help="Task 1 manifest JSONL path. Repeat for multiple manifests.",
    )
    build_cache.add_argument("--frames-root", required=True, help="Shared source frame-cache root")
    build_cache.add_argument("--cache-fps", type=float, default=2.0, help="Canonical physical cache fps")
    build_cache.add_argument("--jpeg-quality", type=int, default=2, help="ffmpeg JPEG quality scale")
    build_cache.add_argument("--workers", type=int, default=1, help="Parallel source-cache workers")
    build_cache.add_argument("--overwrite", action="store_true", help="Re-extract existing caches")
    build_cache.add_argument("--dry-run", action="store_true", help="Report expected cache directories only")
    build_cache.add_argument(
        "--stats-output",
        help="Optional JSON summary output for downstream pipeline stages",
    )
    build_cache.set_defaults(handler=_run_build_source_cache)

    merge_raw = subparsers.add_parser(
        "merge-raw",
        help="Merge raw frame-list JSONLs into a deduplicated train/val pool with stats",
    )
    merge_raw.add_argument(
        "--input",
        dest="inputs",
        action="append",
        required=True,
        help="Raw task JSONL path. Repeat for multiple task-family outputs.",
    )
    merge_raw.add_argument("--output-dir", required=True, help="Output directory for merged raw pool files")
    merge_raw.add_argument("--seed", type=int, default=42, help="Deterministic split seed")
    merge_raw.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    merge_raw.add_argument("--dry-run", action="store_true", help="Report planned outputs without writing files")
    merge_raw.set_defaults(handler=_run_merge_raw)

    rollout_filter = subparsers.add_parser(
        "rollout-filter",
        help="Run rollout scoring and hard-case filtering for a merged raw JSONL",
    )
    rollout_filter.add_argument("--input", required=True, help="Merged raw JSONL path to roll out")
    rollout_filter.add_argument("--output-dir", required=True, help="Output directory for rollout artifacts")
    rollout_filter.add_argument("--model-path", default=DEFAULT_ROLLOUT_MODEL_PATH, help="Model path for offline rollout")
    rollout_filter.add_argument(
        "--reward-function",
        default=DEFAULT_REWARD_FUNCTION,
        help="Reward function path spec for offline rollout",
    )
    rollout_filter.add_argument("--backend", choices=["vllm", "transformers"], default="vllm")
    rollout_filter.add_argument("--num-rollouts", type=int, default=8, help="Number of sampled generations per example")
    rollout_filter.add_argument("--temperature", type=float, default=0.7)
    rollout_filter.add_argument("--top-p", type=float, default=0.9)
    rollout_filter.add_argument("--max-new-tokens", type=int, default=1024)
    rollout_filter.add_argument("--video-fps", type=float, default=2.0)
    rollout_filter.add_argument("--max-frames", type=int, default=256)
    rollout_filter.add_argument("--max-pixels", type=int, default=49152)
    rollout_filter.add_argument("--min-pixels", type=int, default=3136)
    rollout_filter.add_argument("--max-samples", type=int, default=0)
    rollout_filter.add_argument("--min-frames", type=int, default=0)
    rollout_filter.add_argument("--tensor-parallel-size", type=int, default=1)
    rollout_filter.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    rollout_filter.add_argument("--max-model-len", type=int, default=0)
    rollout_filter.add_argument("--max-num-batched-tokens", type=int, default=16384)
    rollout_filter.add_argument("--batch-size", type=int, default=32)
    rollout_filter.add_argument("--dtype", default="bfloat16")
    rollout_filter.add_argument("--min-mean-reward", type=float, default=0.125)
    rollout_filter.add_argument("--max-mean-reward", type=float, default=0.625)
    rollout_filter.add_argument("--min-success-count", type=int, default=1)
    rollout_filter.add_argument("--success-threshold", type=float, default=1.0)
    rollout_filter.add_argument("--target-total", type=int, default=5000)
    rollout_filter.add_argument("--nested-balance-key", default="domain_l1")
    rollout_filter.add_argument("--seed", type=int, default=42)
    rollout_filter.add_argument("--dry-run", action="store_true", help="Report planned commands without running rollout")
    rollout_filter.add_argument(
        "--stats-output",
        help="Optional JSON summary output for this rollout/filter stage",
    )
    rollout_filter.set_defaults(handler=_run_rollout_filter)
    return parser


def _run_build_source_cache(args: argparse.Namespace) -> dict:
    summary = build_source_cache_stage(
        manifest_paths=args.manifests,
        frames_root=args.frames_root,
        fps=args.cache_fps,
        jpeg_quality=args.jpeg_quality,
        overwrite=args.overwrite,
        workers=args.workers,
        dry_run=args.dry_run,
    )

    print(
        f"[build-source-cache] unique sources: {summary['unique_source_count']} "
        f"(from {summary['input_row_count']} manifest rows)"
    )
    for cache in summary["caches"]:
        status = cache.get("status", "ready")
        print(f"[build-source-cache] {status}: {cache['cache_dir']}")

    if args.stats_output:
        write_stats_output(summary, args.stats_output)
        print(f"[build-source-cache] wrote summary: {Path(args.stats_output).resolve()}")

    return summary


def _run_merge_raw(args: argparse.Namespace) -> dict:
    summary = merge_raw_stage(
        input_paths=args.inputs,
        output_dir=args.output_dir,
        seed=args.seed,
        val_ratio=args.val_ratio,
        dry_run=args.dry_run,
    )

    print(
        f"[merge-raw] deduped records: {summary['deduped_record_count']} "
        f"(from {summary['input_record_count']} input rows, duplicates={summary['duplicate_record_count']})"
    )
    print(
        f"[merge-raw] train={summary['train_count']} val={summary['val_count']} "
        f"seed={summary['seed']} val_ratio={summary['val_ratio']}"
    )
    print(f"[merge-raw] train output: {summary['train_output_path']}")
    print(f"[merge-raw] val output: {summary['val_output_path']}")
    print(f"[merge-raw] stats output: {summary['stats_output_path']}")
    if args.dry_run:
        print("[merge-raw] dry-run: no files written")
    else:
        print(f"[merge-raw] wrote merged raw pool: {summary['output_dir']}")
    return summary


def _run_rollout_filter(args: argparse.Namespace) -> dict:
    summary = rollout_filter_stage(
        input_path=args.input,
        output_dir=args.output_dir,
        model_path=args.model_path,
        reward_function=args.reward_function,
        backend=args.backend,
        num_rollouts=args.num_rollouts,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        video_fps=args.video_fps,
        max_frames=args.max_frames,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
        max_samples=args.max_samples,
        min_frames=args.min_frames,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        batch_size=args.batch_size,
        dtype=args.dtype,
        min_mean_reward=args.min_mean_reward,
        max_mean_reward=args.max_mean_reward,
        min_success_count=args.min_success_count,
        success_threshold=args.success_threshold,
        target_total=args.target_total,
        nested_balance_key=args.nested_balance_key,
        seed=args.seed,
        dry_run=args.dry_run,
    )

    print(
        f"[rollout-filter] input records: {summary['input_record_count']} "
        f"model={summary['model_path']} num_rollouts={summary['num_rollouts']}"
    )
    print(
        f"[rollout-filter] hard-case config: mean_reward in "
        f"[{summary['min_mean_reward']}, {summary['max_mean_reward']}], "
        f"min_success_count={summary['min_success_count']}, target_total={summary['target_total']}"
    )
    print(f"[rollout-filter] rollout output: {summary['rollout_output_path']}")
    print(f"[rollout-filter] rollout report: {summary['rollout_report_path']}")
    print(f"[rollout-filter] hard cases output: {summary['hard_cases_output_path']}")
    print(f"[rollout-filter] hard cases stats: {summary['hard_cases_stats_output_path']}")

    if args.dry_run:
        for planned_command in summary["planned_commands"]:
            print(f"[rollout-filter] dry-run {planned_command['stage']}: {planned_command['display']}")
        print("[rollout-filter] dry-run: no commands executed")
    else:
        print(
            f"[rollout-filter] rollout_kept={summary['rollout_output_record_count']} "
            f"report_rows={summary['report_record_count']} hard_cases={summary['hard_case_count']}"
        )
        for problem_type, count in summary["hard_case_summary"]["by_problem_type"].items():
            print(f"[rollout-filter] kept {problem_type}: {count}")

    if args.stats_output:
        write_stats_output(summary, args.stats_output)
        print(f"[rollout-filter] wrote summary: {Path(args.stats_output).resolve()}")

    return summary


def main(argv: list[str] | None = None) -> dict:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    main()
