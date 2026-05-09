#!/usr/bin/env python3
"""Build event-level forward/reverse frame-list JSONL from Task 1 event_dir manifests."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import hashlib
from pathlib import Path
from typing import Any

_PROXY_DATA_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROXY_DATA_DIR not in sys.path:
    sys.path.insert(0, _PROXY_DATA_DIR)

from shared.frame_cache import (  # noqa: E402
    build_source_cache_dir,
    load_cached_frames,
    select_frame_paths_for_span,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DEFAULT_VIDEO_FPS = 2.0
DEFAULT_MAX_FRAMES = 256
MIN_VIDEO_FPS = 0.25
SAMPLE_MODES = ("one_per_event", "two_per_event")

_PROMPT = """\
Watch the video carefully.

This clip shows a multi-step process with a meaningful causal order. Is the process unfolding in its natural forward direction or in reverse?

A. Forward

B. Reverse

Think step by step inside <think></think> tags, then provide your final answer \
(A or B) inside <answer></answer> tags."""


def _normalize_source_video_path(source_video_path: str, manifest_dir: str | Path | None = None) -> str:
    source_path = Path(source_video_path).expanduser()
    if not source_path.is_absolute():
        anchor_dir = Path(manifest_dir) if manifest_dir is not None else Path.cwd()
        source_path = anchor_dir / source_path
    return str(source_path.resolve(strict=False))


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    manifest_path = Path(path)
    manifest_dir = manifest_path.resolve().parent
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = dict(json.loads(line))
            row["_manifest_path"] = str(manifest_path.resolve())
            row["_manifest_dir"] = str(manifest_dir)
            source_video_path = str(row.get("source_video_path") or "").strip()
            if source_video_path:
                row["source_video_path"] = _normalize_source_video_path(
                    source_video_path,
                    manifest_dir=manifest_dir,
                )
            rows.append(row)
    return rows


def write_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _build_video_fps_override(
    total_duration_sec: int | float,
    num_videos: int,
    default_fps: float = DEFAULT_VIDEO_FPS,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> float | None:
    duration = float(total_duration_sec or 0.0)
    if duration <= 0:
        return None
    max_frames_per_video = max(1, max_frames // max(num_videos, 1))
    fps = min(default_fps, max_frames_per_video / duration)
    fps = max(fps, MIN_VIDEO_FPS)
    fps = round(fps, 3)
    if fps >= default_fps:
        return None
    return fps


def _resolve_cached_frames(row: dict[str, Any], frames_root: str | Path, cache_fps: float) -> tuple[Path, list[Path]]:
    cache_dir = build_source_cache_dir(
        frames_root=frames_root,
        clip_key=str(row.get("clip_key") or ""),
        source_video_path=str(row.get("source_video_path") or ""),
        fps=cache_fps,
    )
    cached_frames = load_cached_frames(cache_dir)
    if not cached_frames:
        raise FileNotFoundError(
            f"no cached frames for clip_key={row.get('clip_key')} cache_dir={cache_dir}"
        )
    return cache_dir, cached_frames


def _extract_event(row: dict[str, Any]) -> dict[str, Any]:
    events = list(row.get("events") or [])
    if len(events) != 1:
        raise ValueError(
            f"event_dir rows must contain exactly one event span, got {len(events)} for clip_key={row.get('clip_key')}"
        )
    event = dict(events[0])
    start = event.get("start")
    end = event.get("end")
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        raise ValueError(f"event span missing numeric start/end for clip_key={row.get('clip_key')}")
    start_sec = float(start)
    end_sec = float(end)
    if end_sec <= start_sec:
        raise ValueError(
            f"event span end must be greater than start for clip_key={row.get('clip_key')} event_id={event.get('event_id')}"
        )
    event["start"] = int(start_sec) if float(int(start_sec)) == start_sec else start_sec
    event["end"] = int(end_sec) if float(int(end_sec)) == end_sec else end_sec
    return event


def _build_event_payload(
    row: dict[str, Any],
    frames_root: str | Path,
    cache_fps: float,
    target_fps: float | None,
) -> dict[str, Any]:
    event = _extract_event(row)
    cache_dir, cached_frames = _resolve_cached_frames(row, frames_root, cache_fps)
    selected = select_frame_paths_for_span(
        frame_paths=cached_frames,
        source_fps=cache_fps,
        start_sec=float(event["start"]),
        end_sec=float(event["end"]),
        target_fps=target_fps,
    )
    if not selected:
        raise RuntimeError(
            f"no frames selected for clip_key={row.get('clip_key')} event_id={event.get('event_id')}"
        )
    forward_frames = [str(path.resolve()) for path in selected]
    return {
        "event": event,
        "cache_dir": str(cache_dir.resolve()),
        "forward_frames": forward_frames,
        "reverse_frames": list(reversed(forward_frames)),
        "total_duration_sec": float(event["end"]) - float(event["start"]),
    }


def _base_metadata(
    row: dict[str, Any],
    payload: dict[str, Any],
    cache_fps: float,
    sample_mode: str,
    query_variant: str,
) -> dict[str, Any]:
    event = payload["event"]
    event_text = str(event.get("text") or row.get("event_text") or "").strip()
    metadata = {
        "clip_key": row.get("clip_key"),
        "phase_id": row.get("phase_id", event.get("parent_phase_id")),
        "event_id": event.get("event_id", row.get("event_id")),
        "event_text": event_text,
        "forward_descriptions": [event_text] if event_text else [],
        "total_duration_sec": payload["total_duration_sec"],
        "query_variant": query_variant,
        "sample_mode": sample_mode,
        "domain_l1": row.get("domain_l1", "other"),
        "domain_l2": row.get("domain_l2", "other"),
        "source_video_path": row.get("source_video_path", ""),
        "source": "event_dir_manifest_shared_frames",
        "manifest_type": row.get("manifest_type"),
        "span_start_sec": event.get("start"),
        "span_end_sec": event.get("end"),
        "shared_source_frames": {
            "cache_dir": payload["cache_dir"],
            "cache_fps": cache_fps,
        },
    }
    manifest_path = row.get("_manifest_path")
    if manifest_path:
        metadata["manifest_path"] = manifest_path
    annotation_meta = row.get("annotation_meta")
    if isinstance(annotation_meta, dict):
        metadata["annotation_meta"] = dict(annotation_meta)
    return metadata


def _build_record(
    row: dict[str, Any],
    payload: dict[str, Any],
    cache_fps: float,
    sample_mode: str,
    query_variant: str,
) -> dict[str, Any]:
    metadata = _base_metadata(row, payload, cache_fps, sample_mode, query_variant)
    if query_variant == "forward":
        answer = "A"
        frames = payload["forward_frames"]
    elif query_variant == "reverse":
        answer = "B"
        frames = payload["reverse_frames"]
    else:
        raise ValueError(f"unsupported query_variant: {query_variant}")

    video_fps_override = _build_video_fps_override(payload["total_duration_sec"], num_videos=1)
    if video_fps_override is not None:
        metadata["video_fps_override"] = video_fps_override

    prompt = f"<video>\n\n{_PROMPT}"
    return {
        "messages": [{"role": "user", "content": prompt}],
        "prompt": prompt,
        "answer": answer,
        "videos": [frames],
        "data_type": "video",
        "problem_type": "seg_aot_event_dir_binary",
        "metadata": metadata,
    }


def _row_identity(row: dict[str, Any], payload: dict[str, Any]) -> str:
    event = payload["event"]
    identity = {
        "clip_key": row.get("clip_key"),
        "source_video_path": row.get("source_video_path", ""),
        "phase_id": row.get("phase_id", event.get("parent_phase_id")),
        "event_id": event.get("event_id", row.get("event_id")),
        "span_start_sec": event.get("start"),
        "span_end_sec": event.get("end"),
        "event_text": str(event.get("text") or row.get("event_text") or "").strip(),
    }
    payload_json = json.dumps(identity, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha1(payload_json.encode("utf-8")).hexdigest()
    return f"{digest}:{payload_json}"


def _is_distinguishable_binary_example(payload: dict[str, Any]) -> bool:
    forward_frames = payload["forward_frames"]
    reverse_frames = payload["reverse_frames"]
    return len(forward_frames) >= 2 and forward_frames != reverse_frames


def build_records(
    rows: list[dict[str, Any]],
    frames_root: str | Path,
    cache_fps: float,
    sample_mode: str = "one_per_event",
) -> list[dict[str, Any]]:
    if sample_mode not in SAMPLE_MODES:
        raise ValueError(f"sample_mode must be one of {SAMPLE_MODES}, got {sample_mode!r}")

    prepared: list[tuple[dict[str, Any], dict[str, Any], str]] = []

    for row in rows:
        initial_payload = _build_event_payload(
            row=row,
            frames_root=frames_root,
            cache_fps=cache_fps,
            target_fps=None,
        )
        fps_override = _build_video_fps_override(initial_payload["total_duration_sec"], num_videos=1)
        payload = initial_payload
        if fps_override is not None:
            payload = _build_event_payload(
                row=row,
                frames_root=frames_root,
                cache_fps=cache_fps,
                target_fps=fps_override,
            )
        if not _is_distinguishable_binary_example(payload):
            continue
        prepared.append((row, payload, _row_identity(row, payload)))

    if sample_mode == "one_per_event":
        sorted_identities = sorted(identity for _, _, identity in prepared)
        variant_by_identity = {
            identity: ("forward" if idx % 2 == 0 else "reverse")
            for idx, identity in enumerate(sorted_identities)
        }
    else:
        variant_by_identity = {}

    records: list[dict[str, Any]] = []
    for row, payload, identity in prepared:
        variants = (
            [variant_by_identity[identity]]
            if sample_mode == "one_per_event"
            else ["forward", "reverse"]
        )
        for query_variant in variants:
            records.append(
                _build_record(
                    row=row,
                    payload=payload,
                    cache_fps=cache_fps,
                    sample_mode=sample_mode,
                    query_variant=query_variant,
                )
            )

    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build event forward/reverse frame-list JSONL from Task 1 event_dir manifests."
    )
    parser.add_argument("--event-manifest", required=True, help="Task 1 event_dir manifest JSONL")
    parser.add_argument("--frames-root", required=True, help="Shared source-frame cache root")
    parser.add_argument("--cache-fps", type=float, default=2.0, help="Canonical fps of the shared cache")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--sample-mode", choices=SAMPLE_MODES, default="one_per_event")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _load_jsonl(args.event_manifest)
    records = build_records(
        rows=rows,
        frames_root=args.frames_root,
        cache_fps=args.cache_fps,
        sample_mode=args.sample_mode,
    )
    write_jsonl(records, args.output)
    log.info("Wrote %d event_dir binary records to %s", len(records), args.output)


if __name__ == "__main__":
    main()
