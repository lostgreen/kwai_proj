#!/usr/bin/env python3
"""
Build binary forward/reverse direction data from event_logic_sort JSONL.

Input records are the existing Event Logic sort samples. For each event-level
sort sample we:
  1. Recover the true forward clip order from `answer`
  2. Concatenate the forward-order event clips
  3. Create a true reverse-playback version via ffmpeg `reverse`
  4. Emit one binary MCQ asking whether the shown video is forward or reverse

This keeps the stronger causal signal of the curated sort data while avoiding
the old "reverse-order concat" shortcut.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DEFAULT_VIDEO_FPS = 2.0
DEFAULT_MAX_FRAMES = 256
MIN_VIDEO_FPS = 0.25

_PROMPT = """\
Watch the video carefully.

This clip shows a multi-step process with a meaningful causal order. Is the process unfolding in its natural forward direction or in reverse?

A. Forward

B. Reverse

Think step by step inside <think></think> tags, then provide your final answer \
(A or B) inside <answer></answer> tags."""


def load_jsonl(path: str) -> list[dict]:
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def ffmpeg_concat_clips(clip_paths: list[str], output_path: str, overwrite: bool = False) -> bool:
    if not clip_paths:
        return False
    if os.path.exists(output_path) and not overwrite:
        return True

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if len(clip_paths) == 1:
        try:
            shutil.copy2(clip_paths[0], output_path)
            return True
        except OSError:
            return False

    list_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, dir=os.path.dirname(output_path)
        ) as f:
            for p in clip_paths:
                escaped = p.replace("'", "'\\''")
                f.write(f"file '{escaped}'\n")
            list_path = f.name

        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", "-an", output_path],
            check=True,
            capture_output=True,
            timeout=120,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        log.warning("Concat failed for %s: %s", output_path, exc)
        if os.path.exists(output_path):
            os.unlink(output_path)
        return False
    finally:
        if list_path and os.path.exists(list_path):
            os.unlink(list_path)


def ffmpeg_reverse_video(input_path: str, output_path: str, overwrite: bool = False) -> bool:
    if not os.path.exists(input_path):
        return False
    if os.path.exists(output_path) and not overwrite:
        return True

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                input_path,
                "-vf",
                "reverse",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                output_path,
            ],
            check=True,
            capture_output=True,
            timeout=180,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        log.warning("Reverse failed for %s: %s", output_path, exc)
        if os.path.exists(output_path):
            os.unlink(output_path)
        return False


def _parse_sort_answer(answer: str) -> list[int] | None:
    digits = [int(d) for d in re.findall(r"\d", answer or "")]
    if len(digits) < 2:
        return None
    order = [d - 1 for d in digits]
    if min(order) < 0:
        return None
    return order


def _clip_duration_from_path(path: str) -> int | None:
    stem = Path(path).stem
    match = re.search(r"_(\d+)_(\d+)$", stem)
    if not match:
        return None
    start = int(match.group(1))
    end = int(match.group(2))
    return max(0, end - start)


def _build_video_fps_override(
    total_duration_sec: int | float,
    default_fps: float = DEFAULT_VIDEO_FPS,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> float | None:
    duration = float(total_duration_sec or 0.0)
    if duration <= 0:
        return None
    fps = min(default_fps, max_frames / duration)
    fps = max(fps, MIN_VIDEO_FPS)
    fps = round(fps, 3)
    if fps >= default_fps:
        return None
    return fps


def _record_uid(record: dict) -> str:
    payload = {
        "prompt": record.get("prompt", ""),
        "answer": record.get("answer", ""),
        "videos": record.get("videos", []),
        "ordered_ids": (record.get("metadata") or {}).get("ordered_ids", []),
    }
    return hashlib.md5(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def _recover_forward_payload(record: dict) -> dict | None:
    if record.get("problem_type") != "event_logic_sort":
        return None
    meta = record.get("metadata") or {}
    if meta.get("granularity") != "Event":
        return None

    order = _parse_sort_answer(record.get("answer", ""))
    videos = record.get("videos") or []
    instructions = meta.get("instructions") or []
    if order is None or len(order) != len(videos) or len(set(order)) != len(order):
        return None
    if any(idx < 0 or idx >= len(videos) for idx in order):
        return None

    fwd_videos = [videos[idx] for idx in order]
    if not all(os.path.exists(path) for path in fwd_videos):
        return None

    fwd_instructions = []
    for idx in order:
        if idx < len(instructions):
            fwd_instructions.append(str(instructions[idx]).strip())

    total_duration = 0
    valid_duration = True
    for path in fwd_videos:
        duration = _clip_duration_from_path(path)
        if duration is None:
            valid_duration = False
            break
        total_duration += duration

    clip_key = meta.get("clip_key", "unknown")
    uid = _record_uid(record)
    return {
        "clip_key": clip_key,
        "uid": uid,
        "forward_videos": fwd_videos,
        "forward_instructions": fwd_instructions,
        "ordered_ids": meta.get("ordered_ids", []),
        "domain_l1": meta.get("domain_l1", "other"),
        "domain_l2": meta.get("domain_l2", "other"),
        "total_duration_sec": total_duration if valid_duration else 0,
    }


def _build_direction_record(info: dict, query_variant: str, forward_path: str, reverse_path: str) -> dict:
    answer = "A" if query_variant == "forward" else "B"
    video_path = forward_path if query_variant == "forward" else reverse_path
    meta = {
        "clip_key": info["clip_key"],
        "ordered_ids": info["ordered_ids"],
        "n_events": len(info["ordered_ids"]),
        "total_duration_sec": info["total_duration_sec"],
        "query_variant": query_variant,
        "forward_descriptions": info["forward_instructions"],
        "domain_l1": info["domain_l1"],
        "domain_l2": info["domain_l2"],
        "source": "event_logic_sort",
    }
    video_fps_override = _build_video_fps_override(info["total_duration_sec"])
    if video_fps_override is not None:
        meta["video_fps_override"] = video_fps_override
    return {
        "messages": [{"role": "user", "content": f"<video>\n\n{_PROMPT}"}],
        "prompt": f"<video>\n\n{_PROMPT}",
        "answer": answer,
        "videos": [video_path],
        "data_type": "video",
        "problem_type": "seg_aot_sort_event_dir_binary",
        "metadata": meta,
    }


def _prepare_records(records: list[dict], concat_dir: str) -> tuple[list[dict], dict[str, tuple[list[str], str]], dict[str, str]]:
    prepared: list[dict] = []
    concat_jobs: dict[str, tuple[list[str], str]] = {}
    reverse_jobs: dict[str, str] = {}

    for record in records:
        info = _recover_forward_payload(record)
        if info is None:
            continue
        clip_key = info["clip_key"]
        uid = info["uid"]
        forward_path = os.path.join(concat_dir, "forward", f"{clip_key}_{uid}_fwd.mp4")
        reverse_path = os.path.join(concat_dir, "reverse", f"{clip_key}_{uid}_revplay.mp4")
        info["forward_path"] = forward_path
        info["reverse_path"] = reverse_path
        prepared.append(info)
        concat_jobs.setdefault(forward_path, (info["forward_videos"], forward_path))
        reverse_jobs.setdefault(reverse_path, forward_path)

    return prepared, concat_jobs, reverse_jobs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build forward/reverse direction data from event_logic_sort JSONL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-input", required=True)
    parser.add_argument("--val-input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--concat-dir", default="")
    parser.add_argument("--concat-workers", type=int, default=8)
    parser.add_argument("--overwrite-concat", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output_dir = os.path.abspath(args.output_dir)
    concat_dir = args.concat_dir or os.path.join(output_dir, "concat_videos")

    train_records = load_jsonl(args.train_input)
    val_records = load_jsonl(args.val_input)
    train_infos, concat_jobs, reverse_jobs = _prepare_records(train_records, concat_dir)
    val_infos, val_concat_jobs, val_reverse_jobs = _prepare_records(val_records, concat_dir)
    concat_jobs.update(val_concat_jobs)
    reverse_jobs.update(val_reverse_jobs)

    log.info(
        "Prepared %d train + %d val sort-direction candidates (%d concat jobs)",
        len(train_infos),
        len(val_infos),
        len(concat_jobs),
    )

    failed_paths: set[str] = set()
    with ThreadPoolExecutor(max_workers=args.concat_workers) as pool:
        futures = {
            pool.submit(ffmpeg_concat_clips, clip_paths, out_path, args.overwrite_concat): out_path
            for clip_paths, out_path in concat_jobs.values()
        }
        for fut in as_completed(futures):
            out_path = futures[fut]
            try:
                if not fut.result():
                    failed_paths.add(out_path)
            except Exception:
                failed_paths.add(out_path)

    reverse_inputs = {out_path: in_path for out_path, in_path in reverse_jobs.items() if in_path not in failed_paths}
    if reverse_inputs:
        with ThreadPoolExecutor(max_workers=max(1, args.concat_workers // 2)) as pool:
            reverse_futures = {
                pool.submit(ffmpeg_reverse_video, in_path, out_path, args.overwrite_concat): out_path
                for out_path, in_path in reverse_inputs.items()
            }
            for fut in as_completed(reverse_futures):
                out_path = reverse_futures[fut]
                try:
                    if not fut.result():
                        failed_paths.add(out_path)
                except Exception:
                    failed_paths.add(out_path)

    def build_split(infos: list[dict]) -> list[dict]:
        built: list[dict] = []
        for idx, info in enumerate(infos):
            forward_path = info["forward_path"]
            reverse_path = info["reverse_path"]
            if forward_path in failed_paths or reverse_path in failed_paths:
                continue
            query_variant = "forward" if idx % 2 == 0 else "reverse"
            built.append(_build_direction_record(info, query_variant, forward_path, reverse_path))
        return built

    train_out = build_split(train_infos)
    val_out = build_split(val_infos)

    os.makedirs(output_dir, exist_ok=True)
    write_jsonl(train_out, os.path.join(output_dir, "train.jsonl"))
    write_jsonl(val_out, os.path.join(output_dir, "val.jsonl"))

    stats = {
        "train_input": args.train_input,
        "val_input": args.val_input,
        "train_total": len(train_out),
        "val_total": len(val_out),
        "concat_dir": concat_dir,
        "concat_total": len(concat_jobs),
        "failed_paths": len(failed_paths),
        "train_by_domain_l1": {},
        "val_by_domain_l1": {},
    }
    for split_name, records in (("train_by_domain_l1", train_out), ("val_by_domain_l1", val_out)):
        counter: dict[str, int] = {}
        for rec in records:
            domain = (rec.get("metadata") or {}).get("domain_l1", "other")
            counter[domain] = counter.get(domain, 0) + 1
        stats[split_name] = counter
    with open(os.path.join(output_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    log.info("Output train=%d val=%d -> %s", len(train_out), len(val_out), output_dir)
    if train_out:
        ex = train_out[0]
        log.info("Example: answer=%s video=%s", ex["answer"], ex["videos"][0])


if __name__ == "__main__":
    main()
