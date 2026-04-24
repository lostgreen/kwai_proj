#!/usr/bin/env python3
"""Build AoT frame-list JSONL from Task 1 manifests plus shared source frames."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterable

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

ANSWER_LETTERS = ["A", "B", "C"]
ACTION_VARIANTS = ["forward", "reversed", "shuffled"]
EVENT_BINARY_TEXT_ORDERS = ["forward", "reversed"]
DEFAULT_VIDEO_FPS = 2.0
DEFAULT_MAX_FRAMES = 256
MIN_VIDEO_FPS = 0.25

_EVENT_V2T_3WAY_PROMPT = """\
Watch the video carefully.

Which numbered list correctly describes the temporal order of events visible in this video?

A.
{option_a}

B.
{option_b}

C.
{option_c}

Think step by step inside <think></think> tags, then provide your final answer \
(A, B, or C) inside <answer></answer> tags."""

_EVENT_T2V_BINARY_PROMPT = """\
Here are two video clips (Clip A and Clip B).

The events below occurred in this exact order:
{forward_list}

Which clip (A or B) shows these events in the listed order?

Think step by step inside <think></think> tags, then provide your final answer \
(A or B) inside <answer></answer> tags."""

_ACTION_V2T_3WAY_PROMPT = """\
Watch the video clip carefully.

Which paragraph best matches the temporal order observed in this video?

A. {option_a}

B. {option_b}

C. {option_c}

Think step by step inside <think></think> tags, then provide your final answer \
(A, B, or C) inside <answer></answer> tags."""

_ACTION_T2V_BINARY_PROMPT = """\
Here are two video clips (Clip A and Clip B).

The paragraph below describes the observed temporal order of atomic actions in one clip:
{action_paragraph}

Which clip (A or B) best matches this paragraph?

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
            if line:
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


def _clean_step_text(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    return text.rstrip(" .;,:")


def _format_list(items: list[str]) -> str:
    return "\n".join(f"   {idx + 1}. {item}" for idx, item in enumerate(items))


def _format_paragraph(items: list[str]) -> str:
    cleaned = [_clean_step_text(item) for item in items if _clean_step_text(item)]
    if not cleaned:
        return ""

    sentences: list[str] = []
    last_idx = len(cleaned) - 1
    for idx, item in enumerate(cleaned):
        if len(cleaned) == 1:
            cue = "In this clip"
        elif idx == 0:
            cue = "First"
        elif idx == last_idx:
            cue = "Finally"
        elif idx == 1:
            cue = "Then"
        elif idx == 2:
            cue = "Next"
        else:
            cue = "After that"
        sentences.append(f"{cue}, {item}.")
    return " ".join(sentences)


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


def _non_identity_shuffle_indices(count: int, rng: random.Random) -> list[int] | None:
    if count < 3:
        return None
    identity = list(range(count))
    reversed_indices = list(reversed(identity))
    shuffled = identity[:]
    for _ in range(32):
        rng.shuffle(shuffled)
        if shuffled != identity and shuffled != reversed_indices:
            return shuffled[:]
    return None


def _sum_child_duration(children: list[dict[str, Any]]) -> int:
    total = 0
    for child in children:
        total += max(0, int(child["end"]) - int(child["start"]))
    return total


def _normalize_children(row: dict[str, Any], child_key: str) -> list[dict[str, Any]] | None:
    raw_children = list(row.get(child_key) or [])
    if len(raw_children) < 2:
        return None

    normalized: list[dict[str, Any]] = []
    for child in raw_children:
        start = child.get("start")
        end = child.get("end")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            return None
        start_sec = int(start)
        end_sec = int(end)
        if end_sec <= start_sec:
            return None
        normalized_child = dict(child)
        normalized_child["start"] = start_sec
        normalized_child["end"] = end_sec
        normalized.append(normalized_child)

    normalized.sort(key=lambda child: (child["start"], child["end"]))
    prev_end: int | None = None
    for child in normalized:
        if prev_end is not None and child["start"] < prev_end:
            return None
        prev_end = child["end"]
    return normalized


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


def _select_child_frame_paths(
    child: dict[str, Any],
    cached_frames: list[Path],
    cache_fps: float,
    target_fps: float | None,
) -> list[str]:
    selected = select_frame_paths_for_span(
        frame_paths=cached_frames,
        source_fps=cache_fps,
        start_sec=float(child["start"]),
        end_sec=float(child["end"]),
        target_fps=target_fps,
    )
    if not selected:
        raise RuntimeError(
            f"no frames selected for child span [{child['start']}, {child['end']})"
        )
    return [str(path.resolve()) for path in selected]


def _build_variant_payload(
    row: dict[str, Any],
    child_key: str,
    frames_root: str | Path,
    cache_fps: float,
    target_fps: float | None,
    rng: random.Random,
    require_shuffled: bool,
) -> dict[str, Any] | None:
    children = _normalize_children(row, child_key)
    if children is None:
        return None

    cache_dir, cached_frames = _resolve_cached_frames(row, frames_root, cache_fps)
    forward_child_frames = [
        _select_child_frame_paths(child, cached_frames, cache_fps, target_fps)
        for child in children
    ]
    reversed_indices = list(reversed(range(len(children))))
    variant_indices = {
        "forward": list(range(len(children))),
        "reversed": reversed_indices,
    }
    shuffle_indices = _non_identity_shuffle_indices(len(children), rng)
    if shuffle_indices is not None:
        variant_indices["shuffled"] = shuffle_indices
    elif require_shuffled:
        return None

    texts = [_clean_step_text(str(child.get("text") or "")) for child in children]
    ids: list[Any] = []
    for child in children:
        ids.append(child.get("action_id", child.get("event_id")))

    variant_texts = {
        name: [texts[idx] for idx in indices]
        for name, indices in variant_indices.items()
    }
    variant_child_ids = {
        name: [ids[idx] for idx in indices]
        for name, indices in variant_indices.items()
    }
    variant_videos = {
        name: [
            frame_path
            for idx in indices
            for frame_path in forward_child_frames[idx]
        ]
        for name, indices in variant_indices.items()
    }

    return {
        "cache_dir": str(cache_dir.resolve()),
        "variant_indices": variant_indices,
        "variant_texts": variant_texts,
        "variant_child_ids": variant_child_ids,
        "variant_videos": variant_videos,
        "children": children,
        "total_duration_sec": _sum_child_duration(children),
    }


def _base_metadata(
    row: dict[str, Any],
    children: list[dict[str, Any]],
    cache_dir: str,
    cache_fps: float,
    child_key: str,
    total_duration_sec: int,
) -> dict[str, Any]:
    meta = {
        "clip_key": row.get("clip_key"),
        "source_video_path": row.get("source_video_path", ""),
        "domain_l1": row.get("domain_l1", "other"),
        "domain_l2": row.get("domain_l2", "other"),
        "source": "aot_group_manifest_shared_frames",
        "manifest_type": row.get("manifest_type"),
        "group_id": (
            f"{row.get('clip_key')}__"
            f"{'event' if child_key == 'actions' else 'phase'}_"
            f"{row.get('event_id', row.get('phase_id'))}"
        ),
        "total_duration_sec": total_duration_sec,
        "shared_source_frames": {
            "cache_dir": cache_dir,
            "cache_fps": cache_fps,
        },
        "child_spans": [
            {
                "id": child.get("action_id", child.get("event_id")),
                "text": child.get("text"),
                "start_sec": child.get("start"),
                "end_sec": child.get("end"),
            }
            for child in children
        ],
    }
    if child_key == "actions":
        meta["event_id"] = row.get("event_id")
        meta["parent_phase_id"] = row.get("parent_phase_id")
        meta["event_text"] = row.get("event_text", "")
        meta["n_actions"] = len(row.get("actions", []))
    else:
        meta["phase_id"] = row.get("phase_id")
        meta["phase_text"] = row.get("phase_text", "")
        meta["n_events"] = len(row.get("events", []))
    annotation_meta = row.get("annotation_meta")
    if isinstance(annotation_meta, dict):
        meta["annotation_meta"] = dict(annotation_meta)
    manifest_path = row.get("_manifest_path")
    if manifest_path:
        meta["manifest_path"] = manifest_path
    return meta


def _build_event_v2t_record(row: dict[str, Any], payload: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    options = [
        ("forward", payload["variant_texts"]["forward"]),
        ("shuffled", payload["variant_texts"]["shuffled"]),
        ("reversed", payload["variant_texts"]["reversed"]),
    ]
    rng.shuffle(options)
    correct = ANSWER_LETTERS[[name for name, _ in options].index("forward")]
    body = _EVENT_V2T_3WAY_PROMPT.format(
        option_a=_format_list(options[0][1]),
        option_b=_format_list(options[1][1]),
        option_c=_format_list(options[2][1]),
    )
    prompt = f"<video>\n\n{body}"
    metadata = _base_metadata(
        row=row,
        children=payload["children"],
        cache_dir=payload["cache_dir"],
        cache_fps=payload["cache_fps"],
        child_key="events",
        total_duration_sec=payload["total_duration_sec"],
    )
    metadata["forward_descriptions"] = payload["variant_texts"]["forward"]
    metadata["ordering_variant_indices"] = payload["variant_indices"]
    metadata["ordering_child_ids"] = payload["variant_child_ids"]
    for idx, (variant, _) in enumerate(options):
        metadata[f"option_{chr(97 + idx)}_type"] = variant
    video_fps_override = _build_video_fps_override(payload["total_duration_sec"], num_videos=1)
    if video_fps_override is not None:
        metadata["video_fps_override"] = video_fps_override
    return {
        "messages": [{"role": "user", "content": prompt}],
        "prompt": prompt,
        "answer": correct,
        "videos": [payload["variant_videos"]["forward"]],
        "data_type": "video",
        "problem_type": "seg_aot_event_v2t_3way",
        "metadata": metadata,
    }


def _build_event_t2v_record(
    row: dict[str, Any],
    payload: dict[str, Any],
    rng: random.Random,
    text_order: str,
) -> dict[str, Any]:
    video_options = [
        ("forward", payload["variant_videos"]["forward"]),
        ("reversed", payload["variant_videos"]["reversed"]),
    ]
    rng.shuffle(video_options)
    correct = ANSWER_LETTERS[[name for name, _ in video_options].index(text_order)]
    text_list = payload["variant_texts"][text_order]
    body = _EVENT_T2V_BINARY_PROMPT.format(forward_list=_format_list(text_list))
    prompt = f"<video><video>\n\n{body}"
    metadata = _base_metadata(
        row=row,
        children=payload["children"],
        cache_dir=payload["cache_dir"],
        cache_fps=payload["cache_fps"],
        child_key="events",
        total_duration_sec=payload["total_duration_sec"],
    )
    metadata["forward_descriptions"] = payload["variant_texts"]["forward"]
    metadata["ordering_variant_indices"] = payload["variant_indices"]
    metadata["ordering_child_ids"] = payload["variant_child_ids"]
    metadata["text_order"] = text_order
    metadata["correct_position"] = correct
    for idx, (variant, _) in enumerate(video_options):
        metadata[f"video_{chr(97 + idx)}_type"] = variant
    video_fps_override = _build_video_fps_override(payload["total_duration_sec"], num_videos=2)
    if video_fps_override is not None:
        metadata["video_fps_override"] = video_fps_override
    return {
        "messages": [{"role": "user", "content": prompt}],
        "prompt": prompt,
        "answer": correct,
        "videos": [video for _, video in video_options],
        "data_type": "video",
        "problem_type": "seg_aot_event_t2v_binary",
        "metadata": metadata,
    }


def _build_action_v2t_record(
    row: dict[str, Any],
    payload: dict[str, Any],
    rng: random.Random,
    query_variant: str,
) -> dict[str, Any]:
    text_map = {
        "forward": _format_paragraph(payload["variant_texts"]["forward"]),
        "reversed": _format_paragraph(payload["variant_texts"]["reversed"]),
        "shuffled": _format_paragraph(payload["variant_texts"]["shuffled"]),
    }
    options = [
        ("forward", text_map["forward"]),
        ("shuffled", text_map["shuffled"]),
        ("reversed", text_map["reversed"]),
    ]
    rng.shuffle(options)
    correct = ANSWER_LETTERS[[name for name, _ in options].index(query_variant)]
    body = _ACTION_V2T_3WAY_PROMPT.format(
        option_a=options[0][1],
        option_b=options[1][1],
        option_c=options[2][1],
    )
    prompt = f"<video>\n\n{body}"
    metadata = _base_metadata(
        row=row,
        children=payload["children"],
        cache_dir=payload["cache_dir"],
        cache_fps=payload["cache_fps"],
        child_key="actions",
        total_duration_sec=payload["total_duration_sec"],
    )
    metadata["forward_descriptions"] = payload["variant_texts"]["forward"]
    metadata["forward_paragraph"] = text_map["forward"]
    metadata["reversed_paragraph"] = text_map["reversed"]
    metadata["shuffled_paragraph"] = text_map["shuffled"]
    metadata["ordering_variant_indices"] = payload["variant_indices"]
    metadata["ordering_child_ids"] = payload["variant_child_ids"]
    metadata["query_variant"] = query_variant
    for idx, (variant, _) in enumerate(options):
        metadata[f"option_{chr(97 + idx)}_type"] = variant
    video_fps_override = _build_video_fps_override(payload["total_duration_sec"], num_videos=1)
    if video_fps_override is not None:
        metadata["video_fps_override"] = video_fps_override
    return {
        "messages": [{"role": "user", "content": prompt}],
        "prompt": prompt,
        "answer": correct,
        "videos": [payload["variant_videos"][query_variant]],
        "data_type": "video",
        "problem_type": "seg_aot_action_v2t_3way",
        "metadata": metadata,
    }


def _build_action_t2v_record(
    row: dict[str, Any],
    payload: dict[str, Any],
    rng: random.Random,
    text_order: str,
    negative_variant: str,
) -> dict[str, Any]:
    text_map = {
        "forward": _format_paragraph(payload["variant_texts"]["forward"]),
        "reversed": _format_paragraph(payload["variant_texts"]["reversed"]),
    }
    if "shuffled" in payload["variant_texts"]:
        text_map["shuffled"] = _format_paragraph(payload["variant_texts"]["shuffled"])
    video_options = [
        (text_order, payload["variant_videos"][text_order]),
        (negative_variant, payload["variant_videos"][negative_variant]),
    ]
    rng.shuffle(video_options)
    correct = ANSWER_LETTERS[[name for name, _ in video_options].index(text_order)]
    body = _ACTION_T2V_BINARY_PROMPT.format(action_paragraph=text_map[text_order])
    prompt = f"<video><video>\n\n{body}"
    metadata = _base_metadata(
        row=row,
        children=payload["children"],
        cache_dir=payload["cache_dir"],
        cache_fps=payload["cache_fps"],
        child_key="actions",
        total_duration_sec=payload["total_duration_sec"],
    )
    metadata["forward_descriptions"] = payload["variant_texts"]["forward"]
    metadata["forward_paragraph"] = text_map["forward"]
    metadata["reversed_paragraph"] = text_map["reversed"]
    metadata["ordering_variant_indices"] = payload["variant_indices"]
    metadata["ordering_child_ids"] = payload["variant_child_ids"]
    metadata["text_order"] = text_order
    metadata["negative_variant"] = negative_variant
    metadata["correct_position"] = correct
    if "shuffled" in text_map:
        metadata["shuffled_paragraph"] = text_map["shuffled"]
    for idx, (variant, _) in enumerate(video_options):
        metadata[f"video_{chr(97 + idx)}_type"] = variant
    video_fps_override = _build_video_fps_override(payload["total_duration_sec"], num_videos=2)
    if video_fps_override is not None:
        metadata["video_fps_override"] = video_fps_override
    return {
        "messages": [{"role": "user", "content": prompt}],
        "prompt": prompt,
        "answer": correct,
        "videos": [video for _, video in video_options],
        "data_type": "video",
        "problem_type": "seg_aot_action_t2v_binary",
        "metadata": metadata,
    }


def build_event_records(
    rows: list[dict[str, Any]],
    frames_root: str | Path,
    cache_fps: float,
    seed: int = 0,
    build_v2t: bool = False,
    build_t2v: bool = False,
    t2v_max_duration: int = 60,
) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    records: dict[str, list[dict[str, Any]]] = {
        "event_v2t": [],
        "event_t2v": [],
    }
    t2v_idx = 0

    for row in rows:
        payload_binary = _build_variant_payload(
            row=row,
            child_key="events",
            frames_root=frames_root,
            cache_fps=cache_fps,
            target_fps=None,
            rng=random.Random(rng.random()),
            require_shuffled=False,
        )
        if payload_binary is None:
            continue
        payload_binary["cache_fps"] = cache_fps
        fps_override_single = _build_video_fps_override(payload_binary["total_duration_sec"], num_videos=1)
        if fps_override_single is not None:
            payload_binary = _build_variant_payload(
                row=row,
                child_key="events",
                frames_root=frames_root,
                cache_fps=cache_fps,
                target_fps=fps_override_single,
                rng=random.Random(rng.random()),
                require_shuffled=False,
            )
            if payload_binary is None:
                continue
            payload_binary["cache_fps"] = cache_fps

        if build_v2t:
            payload_3way = _build_variant_payload(
                row=row,
                child_key="events",
                frames_root=frames_root,
                cache_fps=cache_fps,
                target_fps=fps_override_single,
                rng=random.Random(rng.random()),
                require_shuffled=True,
            )
            if payload_3way is not None:
                payload_3way["cache_fps"] = cache_fps
                records["event_v2t"].append(
                    _build_event_v2t_record(row, payload_3way, random.Random(rng.random()))
                )

        if build_t2v and payload_binary["total_duration_sec"] <= t2v_max_duration:
            fps_override_t2v = _build_video_fps_override(payload_binary["total_duration_sec"], num_videos=2)
            t2v_payload = payload_binary
            if fps_override_t2v != fps_override_single:
                t2v_payload = _build_variant_payload(
                    row=row,
                    child_key="events",
                    frames_root=frames_root,
                    cache_fps=cache_fps,
                    target_fps=fps_override_t2v,
                    rng=random.Random(rng.random()),
                    require_shuffled=False,
                )
                if t2v_payload is None:
                    continue
                t2v_payload["cache_fps"] = cache_fps
            text_order = EVENT_BINARY_TEXT_ORDERS[t2v_idx % len(EVENT_BINARY_TEXT_ORDERS)]
            records["event_t2v"].append(
                _build_event_t2v_record(row, t2v_payload, random.Random(rng.random()), text_order)
            )
            t2v_idx += 1

    return records


def build_action_records(
    rows: list[dict[str, Any]],
    frames_root: str | Path,
    cache_fps: float,
    seed: int = 0,
    build_v2t: bool = False,
    build_t2v: bool = False,
    t2v_max_duration: int = 90,
) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    records: dict[str, list[dict[str, Any]]] = {
        "action_v2t": [],
        "action_t2v": [],
    }
    v2t_idx = 0
    t2v_idx = 0

    for row in rows:
        payload_binary = _build_variant_payload(
            row=row,
            child_key="actions",
            frames_root=frames_root,
            cache_fps=cache_fps,
            target_fps=None,
            rng=random.Random(rng.random()),
            require_shuffled=False,
        )
        if payload_binary is None:
            continue
        payload_binary["cache_fps"] = cache_fps
        fps_override_single = _build_video_fps_override(payload_binary["total_duration_sec"], num_videos=1)
        if fps_override_single is not None:
            payload_binary = _build_variant_payload(
                row=row,
                child_key="actions",
                frames_root=frames_root,
                cache_fps=cache_fps,
                target_fps=fps_override_single,
                rng=random.Random(rng.random()),
                require_shuffled=False,
            )
            if payload_binary is None:
                continue
            payload_binary["cache_fps"] = cache_fps

        if build_v2t:
            payload_3way = _build_variant_payload(
                row=row,
                child_key="actions",
                frames_root=frames_root,
                cache_fps=cache_fps,
                target_fps=fps_override_single,
                rng=random.Random(rng.random()),
                require_shuffled=True,
            )
            if payload_3way is not None:
                payload_3way["cache_fps"] = cache_fps
                query_variant = ACTION_VARIANTS[v2t_idx % len(ACTION_VARIANTS)]
                records["action_v2t"].append(
                    _build_action_v2t_record(row, payload_3way, random.Random(rng.random()), query_variant)
                )
                v2t_idx += 1

        if build_t2v and payload_binary["total_duration_sec"] <= t2v_max_duration:
            fps_override_t2v = _build_video_fps_override(payload_binary["total_duration_sec"], num_videos=2)
            t2v_payload = payload_binary
            if fps_override_t2v != fps_override_single:
                t2v_payload = _build_variant_payload(
                    row=row,
                    child_key="actions",
                    frames_root=frames_root,
                    cache_fps=cache_fps,
                    target_fps=fps_override_t2v,
                    rng=random.Random(rng.random()),
                    require_shuffled=False,
                )
                if t2v_payload is None:
                    continue
                t2v_payload["cache_fps"] = cache_fps
            available_variants = list(t2v_payload["variant_videos"].keys())
            text_order = available_variants[t2v_idx % len(available_variants)]
            remaining = [variant for variant in available_variants if variant != text_order]
            negative_variant = remaining[(t2v_idx // len(available_variants)) % len(remaining)]
            records["action_t2v"].append(
                _build_action_t2v_record(
                    row,
                    t2v_payload,
                    random.Random(rng.random()),
                    text_order=text_order,
                    negative_variant=negative_variant,
                )
            )
            t2v_idx += 1

    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build action/event AoT frame-list JSONL from Task 1 manifests."
    )
    parser.add_argument("--frames-root", required=True, help="Shared source-frame cache root")
    parser.add_argument("--cache-fps", type=float, default=2.0, help="Canonical fps of the shared cache")
    parser.add_argument("--action-manifest", default="", help="Task 1 action group manifest JSONL")
    parser.add_argument("--event-manifest", default="", help="Task 1 event group manifest JSONL")
    parser.add_argument("--action-v2t-output", default="", help="Output JSONL for action_v2t 3-way")
    parser.add_argument("--action-t2v-output", default="", help="Output JSONL for action_t2v binary")
    parser.add_argument("--event-v2t-output", default="", help="Output JSONL for event_v2t 3-way")
    parser.add_argument("--event-t2v-output", default="", help="Output JSONL for event_t2v binary")
    parser.add_argument("--action-t2v-max-duration", type=int, default=90)
    parser.add_argument("--event-t2v-max-duration", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    wants_action = bool(args.action_v2t_output or args.action_t2v_output)
    wants_event = bool(args.event_v2t_output or args.event_t2v_output)
    if not wants_action and not wants_event:
        raise SystemExit(
            "At least one output path is required: --action-v2t-output, --action-t2v-output, "
            "--event-v2t-output, or --event-t2v-output"
        )
    if wants_action and not args.action_manifest:
        raise SystemExit("--action-manifest is required when writing action outputs")
    if wants_event and not args.event_manifest:
        raise SystemExit("--event-manifest is required when writing event outputs")

    if wants_action:
        action_rows = _load_jsonl(args.action_manifest)
        action_records = build_action_records(
            rows=action_rows,
            frames_root=args.frames_root,
            cache_fps=args.cache_fps,
            seed=args.seed,
            build_v2t=bool(args.action_v2t_output),
            build_t2v=bool(args.action_t2v_output),
            t2v_max_duration=args.action_t2v_max_duration,
        )
        if args.action_v2t_output:
            write_jsonl(action_records["action_v2t"], args.action_v2t_output)
            log.info("Wrote %d action_v2t records to %s", len(action_records["action_v2t"]), args.action_v2t_output)
        if args.action_t2v_output:
            write_jsonl(action_records["action_t2v"], args.action_t2v_output)
            log.info("Wrote %d action_t2v records to %s", len(action_records["action_t2v"]), args.action_t2v_output)

    if wants_event:
        event_rows = _load_jsonl(args.event_manifest)
        event_records = build_event_records(
            rows=event_rows,
            frames_root=args.frames_root,
            cache_fps=args.cache_fps,
            seed=args.seed,
            build_v2t=bool(args.event_v2t_output),
            build_t2v=bool(args.event_t2v_output),
            t2v_max_duration=args.event_t2v_max_duration,
        )
        if args.event_v2t_output:
            write_jsonl(event_records["event_v2t"], args.event_v2t_output)
            log.info("Wrote %d event_v2t records to %s", len(event_records["event_v2t"]), args.event_v2t_output)
        if args.event_t2v_output:
            write_jsonl(event_records["event_t2v"], args.event_t2v_output)
            log.info("Wrote %d event_t2v records to %s", len(event_records["event_t2v"]), args.event_t2v_output)


if __name__ == "__main__":
    main()
