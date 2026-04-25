#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from transformers import AutoProcessor

from verl.utils.dataset import process_video
from verl.utils.video_fps import resolve_video_fps_list


TIME_RE = re.compile(r"<([0-9]+(?:\.[0-9]+)?) seconds>")


def _json_blob(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False)


def load_record(path: Path, contains: str) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if contains in _json_blob(record):
                record["_line_no"] = line_no
                return record
    raise SystemExit(f"no record containing {contains!r} in {path}")


def build_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    prompt = str(record.get("prompt") or "")
    content: list[dict[str, Any]] = []
    for idx, part in enumerate(prompt.split("<video>")):
        if idx != 0:
            content.append({"type": "video"})
        if part:
            content.append({"type": "text", "text": part})
    return [{"role": "user", "content": content}]


def expected_duration(record: dict[str, Any], video_idx: int = 0) -> float:
    sampling = (record.get("metadata") or {}).get("experiment_frame_sampling") or {}
    sampling_videos = sampling.get("videos") or []
    if sampling_videos:
        meta = sampling_videos[min(video_idx, len(sampling_videos) - 1)]
        return float(meta["duration_sec"])

    videos = record.get("videos") or []
    metadata = record.get("metadata") or {}
    fps_values = resolve_video_fps_list(metadata, 2.0, n_videos=max(1, len(videos)))
    fps = fps_values[min(video_idx, len(fps_values) - 1)]
    if videos and isinstance(videos[video_idx], list):
        return len(videos[video_idx]) / max(float(fps), 1e-6)
    for key in ("duration", "clip_duration"):
        value = metadata.get(key) or record.get(key)
        if value:
            return float(value)
    raise SystemExit("unable to infer expected duration from record")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check decoded Qwen timestamp tokens against frame-policy fps.")
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--contains", default="person start undressing")
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--min-pixels", type=int, default=3136)
    parser.add_argument("--max-pixels", type=int, default=65536)
    parser.add_argument("--default-fps", type=float, default=2.0)
    parser.add_argument("--tolerance", type=float, default=1.5)
    args = parser.parse_args()

    record = load_record(Path(args.jsonl), args.contains)
    videos = record.get("videos") or []
    if not videos:
        raise SystemExit("selected record has no videos")

    fps_list = resolve_video_fps_list(record.get("metadata") or {}, args.default_fps, n_videos=len(videos))
    processed_videos = []
    video_metadatas = []
    for idx, video in enumerate(videos):
        processed, _sample_fps = process_video(
            video,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            max_frames=args.max_frames,
            video_fps=fps_list[min(idx, len(fps_list) - 1)],
            return_fps=True,
        )
        frames, metadata = processed
        processed_videos.append(frames)
        video_metadatas.append(metadata)

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    text = processor.apply_chat_template(build_messages(record), add_generation_prompt=True, tokenize=False)
    inputs = processor(
        text=[text],
        videos=processed_videos,
        video_metadata=video_metadatas,
        return_tensors="pt",
        do_resize=False,
        do_sample_frames=False,
        add_special_tokens=False,
    )
    decoded = processor.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    times = [float(value) for value in TIME_RE.findall(decoded)]
    if not times:
        raise SystemExit("decoded prompt contains no '<x seconds>' timestamps")

    max_seen = max(times)
    duration = expected_duration(record)
    result = {
        "line_no": record["_line_no"],
        "fps_list": fps_list,
        "max_decoded_time": max_seen,
        "expected_duration": duration,
        "delta": abs(max_seen - duration),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["delta"] > args.tolerance:
        raise SystemExit(f"timestamp mismatch: decoded={max_seen}, expected={duration}")


if __name__ == "__main__":
    main()
