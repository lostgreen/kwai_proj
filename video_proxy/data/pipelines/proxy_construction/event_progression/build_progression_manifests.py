#!/usr/bin/env python3
"""
Build unified AoT group manifests directly from seg annotations.

This layer sits between hierarchical seg annotations and later frame-list QA
generation. It preserves temporal structure without depending on atomic clips
or concat mp4 artifacts.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

_PROXY_DATA_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROXY_DATA_DIR not in sys.path:
    sys.path.insert(0, _PROXY_DATA_DIR)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MANIFEST_ACTION = "action"
MANIFEST_EVENT = "event"
MANIFEST_EVENT_DIR = "event_dir"
ANNOTATION_META_KEYS = (
    "video_path",
    "source_mode",
    "annotation_start_sec",
    "annotation_end_sec",
    "window_start_sec",
    "window_end_sec",
    "clip_duration_sec",
    "n_frames",
    "frame_dir",
    "summary",
    "video_caption",
    "archetype",
    "topology_type",
    "global_phase_criterion",
)


def _safe_text(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _as_int_time(value: object) -> int | None:
    if not isinstance(value, (int, float)):
        return None
    return int(value)


def _normalize_span(start: object, end: object) -> tuple[int, int] | None:
    start_sec = _as_int_time(start)
    end_sec = _as_int_time(end)
    if start_sec is None or end_sec is None or start_sec >= end_sec:
        return None
    return start_sec, end_sec


def _duration_ok(
    start_sec: int,
    end_sec: int,
    min_duration: int | None = None,
    max_duration: int | None = None,
) -> bool:
    duration = end_sec - start_sec
    if min_duration is not None and duration < min_duration:
        return False
    if max_duration is not None and duration > max_duration:
        return False
    return True


def _load_annotations_with_paths(annotation_dir: str | Path) -> list[dict]:
    ann_dir = Path(annotation_dir)
    annotations: list[dict] = []
    for ann_path in sorted(ann_dir.glob("*.json")):
        try:
            with ann_path.open("r", encoding="utf-8") as handle:
                ann = json.load(handle)
        except Exception as exc:  # pragma: no cover - defensive for large noisy annotation dirs
            log.warning("Skipping unreadable annotation %s: %s", ann_path, exc)
            continue
        if not isinstance(ann, dict):
            log.warning("Skipping non-object annotation %s", ann_path)
            continue
        ann["_annotation_path"] = str(ann_path.resolve())
        ann["_annotation_dir"] = str(ann_path.resolve().parent)
        annotations.append(ann)
    return annotations


def _raw_source_video_path(ann: dict) -> str:
    return str(ann.get("source_video_path") or ann.get("video_path") or "").strip()


def _source_video_path(ann: dict, source_root: str | Path | None = None) -> str:
    raw_path = _raw_source_video_path(ann)
    if not raw_path:
        return ""

    source_path = Path(raw_path).expanduser()
    if not source_path.is_absolute():
        if source_root is not None:
            anchor_dir = Path(source_root)
        elif ann.get("_annotation_dir"):
            anchor_dir = Path(str(ann["_annotation_dir"]))
        elif ann.get("_annotation_path"):
            anchor_dir = Path(str(ann["_annotation_path"])).parent
        else:
            anchor_dir = Path.cwd()
        source_path = anchor_dir / source_path
    return str(source_path.resolve(strict=False))


def _annotation_meta(ann: dict) -> dict:
    meta = {
        key: ann[key]
        for key in ANNOTATION_META_KEYS
        if key in ann and ann[key] is not None
    }
    raw_source_video_path = _raw_source_video_path(ann)
    if raw_source_video_path:
        meta["raw_source_video_path"] = raw_source_video_path
    if ann.get("_annotation_path"):
        meta["annotation_path"] = str(ann["_annotation_path"])
    if ann.get("_annotation_dir"):
        meta["annotation_dir"] = str(ann["_annotation_dir"])
    return meta


def _flatten_l3_actions(grounding_results: object) -> list[dict]:
    if not isinstance(grounding_results, list):
        return []

    actions: list[dict] = []
    for item in grounding_results:
        if not isinstance(item, dict):
            continue

        sub_actions = item.get("sub_actions")
        if isinstance(sub_actions, list):
            parent_event_id = item.get("event_id", item.get("parent_event_id"))
            for sub_action in sub_actions:
                if not isinstance(sub_action, dict):
                    continue
                action = dict(sub_action)
                action.setdefault("parent_event_id", parent_event_id)
                action.setdefault("parent_phase_id", item.get("parent_phase_id"))
                action.setdefault("event_start", item.get("event_start"))
                action.setdefault("event_end", item.get("event_end"))
                action.setdefault("event_instruction", item.get("event_instruction"))
                actions.append(action)
            continue

        actions.append(item)
    return actions


def _build_action_child(action: dict) -> dict:
    start_sec, end_sec = _normalize_span(action.get("start_time"), action.get("end_time")) or (None, None)
    if start_sec is None or end_sec is None:
        raise ValueError("invalid action span")
    child = {
        "action_id": action.get("action_id"),
        "parent_event_id": action.get("parent_event_id"),
        "parent_phase_id": action.get("parent_phase_id"),
        "text": _safe_text(action.get("sub_action")),
        "start": start_sec,
        "end": end_sec,
    }
    caption = _safe_text(action.get("caption"))
    if caption:
        child["caption"] = caption
    parent_event_start = _as_int_time(action.get("event_start"))
    if parent_event_start is not None:
        child["parent_event_start_sec"] = parent_event_start
    parent_event_end = _as_int_time(action.get("event_end"))
    if parent_event_end is not None:
        child["parent_event_end_sec"] = parent_event_end
    parent_event_instruction = _safe_text(action.get("event_instruction"))
    if parent_event_instruction:
        child["parent_event_instruction"] = parent_event_instruction
    return child


def _build_event_child(event: dict) -> dict:
    start_sec, end_sec = _normalize_span(event.get("start_time"), event.get("end_time")) or (None, None)
    if start_sec is None or end_sec is None:
        raise ValueError("invalid event span")
    return {
        "event_id": event.get("event_id"),
        "parent_phase_id": event.get("parent_phase_id"),
        "text": _safe_text(event.get("instruction")),
        "start": start_sec,
        "end": end_sec,
    }


def collect_action_group_manifests(
    annotations: list[dict],
    min_actions: int = 2,
    max_actions: int = 8,
    complete_only: bool = False,
    filter_order: bool = False,
    min_duration: int | None = None,
    max_duration: int | None = None,
    source_root: str | Path | None = None,
) -> list[dict]:
    rows: list[dict] = []
    stats = Counter()

    for ann in annotations:
        l2 = ann.get("level2")
        l3 = ann.get("level3")
        if not l2 or not l3 or l2.get("_parse_error") or l3.get("_parse_error"):
            stats["skip_no_l2_l3"] += 1
            continue

        clip_key = ann.get("clip_key", "")
        source_video_path = _source_video_path(ann, source_root=source_root)
        annotation_meta = _annotation_meta(ann)
        events = l2.get("events", [])
        actions_all = _flatten_l3_actions(l3.get("grounding_results", []))

        for event in events:
            if not isinstance(event, dict):
                continue

            stats["events_total"] += 1
            if complete_only and event.get("_complete") is False:
                stats["events_incomplete"] += 1
                continue
            if filter_order and not event.get("_order_distinguishable", False):
                stats["events_not_distinguishable"] += 1
                continue

            event_span = _normalize_span(event.get("start_time"), event.get("end_time"))
            if not event_span:
                stats["events_invalid_span"] += 1
                continue

            ev_id = event.get("event_id")
            children = []
            for action in sorted(
                [
                    action for action in actions_all
                    if isinstance(action, dict)
                    and action.get("parent_event_id") == ev_id
                    and _safe_text(action.get("sub_action"))
                    and _normalize_span(action.get("start_time"), action.get("end_time"))
                ],
                key=lambda item: _as_int_time(item.get("start_time")) or 0,
            ):
                if complete_only and action.get("_complete") is False:
                    continue
                children.append(_build_action_child(action))

            if len(children) < min_actions:
                stats["events_too_few_actions"] += 1
                continue
            if len(children) > max_actions:
                stats["events_too_many_actions"] += 1
                continue

            span_start_sec, span_end_sec = event_span
            if not _duration_ok(span_start_sec, span_end_sec, min_duration, max_duration):
                stats["events_duration_filtered"] += 1
                continue

            rows.append({
                "manifest_type": MANIFEST_ACTION,
                "clip_key": clip_key,
                "source_video_path": source_video_path,
                "domain_l1": ann.get("domain_l1", "other"),
                "domain_l2": ann.get("domain_l2", "other"),
                "annotation_meta": annotation_meta,
                "span_start_sec": span_start_sec,
                "span_end_sec": span_end_sec,
                "event_id": ev_id,
                "parent_phase_id": event.get("parent_phase_id"),
                "event_text": _safe_text(event.get("instruction")),
                "actions": children,
            })
            stats["events_kept"] += 1

    log.info("Collected %d action groups with stats: %s", len(rows), dict(stats))
    return rows


def collect_event_group_manifests(
    annotations: list[dict],
    min_events: int = 2,
    max_events: int = 8,
    complete_only: bool = False,
    filter_order: bool = False,
    min_duration: int | None = None,
    max_duration: int | None = None,
    source_root: str | Path | None = None,
) -> list[dict]:
    rows: list[dict] = []
    stats = Counter()

    for ann in annotations:
        l1 = ann.get("level1")
        l2 = ann.get("level2")
        if not l1 or not l2 or l1.get("_parse_error") or l2.get("_parse_error"):
            stats["skip_no_l1_l2"] += 1
            continue

        clip_key = ann.get("clip_key", "")
        source_video_path = _source_video_path(ann, source_root=source_root)
        annotation_meta = _annotation_meta(ann)
        phases = l1.get("macro_phases", [])
        events_all = l2.get("events", [])

        for phase in sorted(
            [
                phase for phase in phases
                if isinstance(phase, dict)
                and _normalize_span(phase.get("start_time"), phase.get("end_time"))
            ],
            key=lambda item: _as_int_time(item.get("start_time")) or 0,
        ):
            stats["phases_total"] += 1
            if complete_only and phase.get("_complete") is False:
                stats["phases_incomplete"] += 1
                continue
            if filter_order and not phase.get("_order_distinguishable", False):
                stats["phases_not_distinguishable"] += 1
                continue

            phase_span = _normalize_span(phase.get("start_time"), phase.get("end_time"))
            if not phase_span:
                stats["phases_invalid_span"] += 1
                continue

            phase_id = phase.get("phase_id")
            children = []
            for event in sorted(
                [
                    event for event in events_all
                    if isinstance(event, dict)
                    and event.get("parent_phase_id") == phase_id
                    and _safe_text(event.get("instruction"))
                    and _normalize_span(event.get("start_time"), event.get("end_time"))
                ],
                key=lambda item: _as_int_time(item.get("start_time")) or 0,
            ):
                if complete_only and event.get("_complete") is False:
                    continue
                children.append(_build_event_child(event))

            if len(children) < min_events:
                stats["phases_too_few_events"] += 1
                continue
            if len(children) > max_events:
                stats["phases_too_many_events"] += 1
                continue

            span_start_sec, span_end_sec = phase_span
            if not _duration_ok(span_start_sec, span_end_sec, min_duration, max_duration):
                stats["phases_duration_filtered"] += 1
                continue

            rows.append({
                "manifest_type": MANIFEST_EVENT,
                "clip_key": clip_key,
                "source_video_path": source_video_path,
                "domain_l1": ann.get("domain_l1", "other"),
                "domain_l2": ann.get("domain_l2", "other"),
                "annotation_meta": annotation_meta,
                "span_start_sec": span_start_sec,
                "span_end_sec": span_end_sec,
                "phase_id": phase_id,
                "phase_text": _safe_text(phase.get("phase_name")),
                "events": children,
            })
            stats["phases_kept"] += 1

    log.info("Collected %d event groups with stats: %s", len(rows), dict(stats))
    return rows


def collect_event_reverse_group_manifests(
    annotations: list[dict],
    complete_only: bool = False,
    filter_order: bool = False,
    min_duration: int | None = None,
    max_duration: int | None = None,
    source_root: str | Path | None = None,
) -> list[dict]:
    rows: list[dict] = []
    stats = Counter()

    for ann in annotations:
        l2 = ann.get("level2")
        if not l2 or l2.get("_parse_error"):
            stats["skip_no_l2"] += 1
            continue

        clip_key = ann.get("clip_key", "")
        source_video_path = _source_video_path(ann, source_root=source_root)
        annotation_meta = _annotation_meta(ann)

        for event in sorted(
            [
                event for event in l2.get("events", [])
                if isinstance(event, dict)
                and _safe_text(event.get("instruction"))
                and _normalize_span(event.get("start_time"), event.get("end_time"))
            ],
            key=lambda item: _as_int_time(item.get("start_time")) or 0,
        ):
            stats["events_total"] += 1
            if complete_only and event.get("_complete") is False:
                stats["events_incomplete"] += 1
                continue
            if filter_order and not event.get("_order_distinguishable", False):
                stats["events_not_distinguishable"] += 1
                continue

            span_start_sec, span_end_sec = _normalize_span(event.get("start_time"), event.get("end_time")) or (None, None)
            if span_start_sec is None or span_end_sec is None:
                stats["events_invalid_span"] += 1
                continue
            if not _duration_ok(span_start_sec, span_end_sec, min_duration, max_duration):
                stats["events_duration_filtered"] += 1
                continue

            event_child = _build_event_child(event)
            rows.append({
                "manifest_type": MANIFEST_EVENT_DIR,
                "clip_key": clip_key,
                "source_video_path": source_video_path,
                "domain_l1": ann.get("domain_l1", "other"),
                "domain_l2": ann.get("domain_l2", "other"),
                "annotation_meta": annotation_meta,
                "span_start_sec": span_start_sec,
                "span_end_sec": span_end_sec,
                "phase_id": event.get("parent_phase_id"),
                "events": [event_child],
            })
            stats["events_kept"] += 1

    log.info("Collected %d event-dir groups with stats: %s", len(rows), dict(stats))
    return rows


def build_group_manifests(
    annotations: list[dict],
    include_action: bool = False,
    include_event: bool = False,
    include_event_dir: bool = False,
    min_actions: int = 2,
    max_actions: int = 8,
    min_events: int = 2,
    max_events: int = 8,
    complete_only: bool = False,
    filter_order: bool = False,
    min_duration: int | None = None,
    max_duration: int | None = None,
    source_root: str | Path | None = None,
) -> dict[str, list[dict]]:
    manifests: dict[str, list[dict]] = {}
    if include_action:
        manifests[MANIFEST_ACTION] = collect_action_group_manifests(
            annotations=annotations,
            min_actions=min_actions,
            max_actions=max_actions,
            complete_only=complete_only,
            filter_order=filter_order,
            min_duration=min_duration,
            max_duration=max_duration,
            source_root=source_root,
        )
    if include_event:
        manifests[MANIFEST_EVENT] = collect_event_group_manifests(
            annotations=annotations,
            min_events=min_events,
            max_events=max_events,
            complete_only=complete_only,
            filter_order=filter_order,
            min_duration=min_duration,
            max_duration=max_duration,
            source_root=source_root,
        )
    if include_event_dir:
        manifests[MANIFEST_EVENT_DIR] = collect_event_reverse_group_manifests(
            annotations=annotations,
            complete_only=complete_only,
            filter_order=filter_order,
            min_duration=min_duration,
            max_duration=max_duration,
            source_root=source_root,
        )
    return manifests


def write_jsonl(records: list[dict], output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build AoT group manifests from seg annotations")
    parser.add_argument("--annotation-dir", "-a", required=True, help="Directory of annotation JSON files")
    parser.add_argument("--action-output", default="", help="Output JSONL path for action-group manifest")
    parser.add_argument("--event-output", default="", help="Output JSONL path for event-group manifest")
    parser.add_argument("--event-dir-output", default="", help="Output JSONL path for event reverse-group manifest")
    parser.add_argument(
        "--source-root",
        default="",
        help="Base directory for resolving relative source_video_path/video_path values",
    )
    parser.add_argument("--complete-only", action="store_true", help="Keep only groups marked complete")
    parser.add_argument("--filter-order", action="store_true", help="Keep only _order_distinguishable groups")
    parser.add_argument("--min-actions", type=int, default=2, help="Minimum child actions per action group")
    parser.add_argument("--max-actions", type=int, default=8, help="Maximum child actions per action group")
    parser.add_argument("--min-events", type=int, default=2, help="Minimum child events per event group")
    parser.add_argument("--max-events", type=int, default=8, help="Maximum child events per event group")
    parser.add_argument("--min-duration", type=int, default=None, help="Minimum group span duration in seconds")
    parser.add_argument("--max-duration", type=int, default=None, help="Maximum group span duration in seconds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not any([args.action_output, args.event_output, args.event_dir_output]):
        raise SystemExit("At least one of --action-output, --event-output, --event-dir-output is required")

    annotations = _load_annotations_with_paths(args.annotation_dir)
    manifests = build_group_manifests(
        annotations=annotations,
        include_action=bool(args.action_output),
        include_event=bool(args.event_output),
        include_event_dir=bool(args.event_dir_output),
        min_actions=args.min_actions,
        max_actions=args.max_actions,
        min_events=args.min_events,
        max_events=args.max_events,
        complete_only=args.complete_only,
        filter_order=args.filter_order,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        source_root=args.source_root or None,
    )

    if args.action_output:
        write_jsonl(manifests[MANIFEST_ACTION], args.action_output)
        log.info("Wrote %d action groups to %s", len(manifests[MANIFEST_ACTION]), args.action_output)
    if args.event_output:
        write_jsonl(manifests[MANIFEST_EVENT], args.event_output)
        log.info("Wrote %d event groups to %s", len(manifests[MANIFEST_EVENT]), args.event_output)
    if args.event_dir_output:
        write_jsonl(manifests[MANIFEST_EVENT_DIR], args.event_dir_output)
        log.info("Wrote %d event-dir groups to %s", len(manifests[MANIFEST_EVENT_DIR]), args.event_dir_output)


if __name__ == "__main__":
    main()
