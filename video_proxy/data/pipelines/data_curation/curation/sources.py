from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .io import read_jsonl


DATASET_LABELS = {
    "et_instruct_164k": "ET-Instruct-164K",
    "timelens_100k": "TimeLens-100K",
}


def normalize_dataset(dataset: str) -> str:
    key = dataset.strip().lower().replace("-", "_")
    aliases = {
        "et": "et_instruct_164k",
        "et_instruct": "et_instruct_164k",
        "et_instruct_164k": "et_instruct_164k",
        "timelens": "timelens_100k",
        "timelens_100k": "timelens_100k",
    }
    if key not in aliases:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return aliases[key]


def load_records(dataset: str, input_path: str | Path) -> list[dict[str, Any]]:
    dataset = normalize_dataset(dataset)
    path = Path(input_path)
    if dataset == "et_instruct_164k":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"ET-Instruct input must be a JSON array: {path}")
        return data
    return read_jsonl(path)


def duration(raw: dict[str, Any]) -> float:
    try:
        return float(raw.get("duration") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def source_key(raw: dict[str, Any]) -> str:
    return str(raw.get("source") or "unknown")


def _join_video(video_root: str | Path | None, rel_path: str) -> str:
    if not video_root:
        return rel_path
    return str(Path(video_root) / rel_path)


def _clip_key(rel_path: str) -> str:
    return Path(rel_path).stem


def to_unified_record(
    dataset: str,
    raw: dict[str, Any],
    video_root: str | Path | None = None,
) -> dict[str, Any]:
    dataset = normalize_dataset(dataset)
    if dataset == "et_instruct_164k":
        return _et_to_unified(raw, video_root)
    return _timelens_to_unified(raw, video_root)


def _base_unified(
    *,
    dataset: str,
    raw: dict[str, Any],
    video_rel: str,
    video_root: str | Path | None,
) -> dict[str, Any]:
    dur = duration(raw)
    source = source_key(raw)
    clip_key = _clip_key(video_rel)
    return {
        "videos": [_join_video(video_root, video_rel)],
        "metadata": {
            "clip_key": clip_key,
            "video_id": clip_key,
            "clip_start": 0,
            "clip_end": dur,
            "clip_duration": dur,
            "original_duration": dur,
            "is_full_video": True,
            "source": source,
        },
        "source": source,
        "dataset": DATASET_LABELS[dataset],
        "duration": dur,
        "_origin": raw.get("_origin"),
    }


def _et_to_unified(raw: dict[str, Any], video_root: str | Path | None) -> dict[str, Any]:
    video_rel = str(raw.get("video") or "")
    record = _base_unified(
        dataset="et_instruct_164k",
        raw=raw,
        video_rel=video_rel,
        video_root=video_root,
    )
    tgt = raw.get("tgt") or []
    record["_et_raw"] = {
        "video": video_rel,
        "task": raw.get("task"),
        "tgt": tgt,
        "n_events": len(tgt) // 2 if isinstance(tgt, list) else 0,
    }
    return record


def _timelens_to_unified(raw: dict[str, Any], video_root: str | Path | None) -> dict[str, Any]:
    video_rel = str(raw.get("video_path") or "")
    record = _base_unified(
        dataset="timelens_100k",
        raw=raw,
        video_rel=video_rel,
        video_root=video_root,
    )
    events = raw.get("events") or []
    record["_tl_raw"] = {
        "video_path": video_rel,
        "n_events": len(events) if isinstance(events, list) else 0,
        "events_summary": [
            {"query": ev.get("query", ""), "span": ev.get("span", [])}
            for ev in events[:10]
            if isinstance(ev, dict)
        ] if isinstance(events, list) else [],
    }
    return record
