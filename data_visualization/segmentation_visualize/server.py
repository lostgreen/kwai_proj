import argparse
import base64
import io
import json
import mimetypes
import re
import shutil
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, unquote, urlparse

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_mmss(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return max(0, int(value))
    if not isinstance(value, str):
        return None
    parts = value.strip().split(":")
    if len(parts) != 2:
        return None
    try:
        minutes = int(parts[0])
        seconds = int(parts[1])
    except ValueError:
        return None
    if minutes < 0 or seconds < 0:
        return None
    return minutes * 60 + seconds


def format_mmss(total_seconds: Any) -> str:
    total = max(0, int(safe_float(total_seconds)))
    minutes, seconds = divmod(total, 60)
    return f"{minutes:02d}:{seconds:02d}"


def image_to_data_url(image: Image.Image, max_width: int = 320) -> str:
    image = image.convert("RGB")
    if max_width > 0 and image.width > max_width:
        ratio = max_width / float(image.width)
        image = image.resize((max_width, max(1, int(image.height * ratio))))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    payload = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{payload}"


def parse_json_field(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def print_progress(prefix: str, current: int, total: int) -> None:
    total = max(total, 1)
    current = max(0, min(current, total))
    width = 28
    filled = int(width * current / total)
    bar = "#" * filled + "-" * (width - filled)
    percent = 100.0 * current / total
    end = "\n" if current >= total else ""
    print(f"\r{prefix} [{bar}] {current}/{total} ({percent:5.1f}%)", end=end, flush=True)


_EVENTS_RE = re.compile(r"<events>(.*?)</events>", re.DOTALL)
_SEG_PAIR_RE = re.compile(r"\[\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\]")


def parse_events_answer(text: str) -> list[list[float]]:
    """Parse '<events>[[s, e], ...]</events>' answer string → list of [s, e] float pairs."""
    if not text:
        return []
    m = _EVENTS_RE.search(str(text))
    if not m:
        return []
    pairs: list[list[float]] = []
    for pm in _SEG_PAIR_RE.finditer(m.group(1)):
        pairs.append([float(pm.group(1)), float(pm.group(2))])
    return pairs


def build_segments_from_events(
    pairs: list[list[float]],
    level: int,
    offset: int = 0,
    label_prefix: str = "Segment",
) -> list[dict[str, Any]]:
    """Convert [[start, end], ...] pairs to segment dicts.

    start/end are clip-local coordinates:
      - L1: warped frame indices (1..N)
      - L2/L3: 0-based seconds within the clip window
    offset converts to display space (0 for L1; clip_offset_sec for L2/L3).
    """
    segments: list[dict[str, Any]] = []
    for idx, pair in enumerate(pairs, 1):
        if not (isinstance(pair, (list, tuple)) and len(pair) >= 2):
            continue
        local_start = int(pair[0])
        local_end = int(pair[1])
        abs_start = local_start + offset
        abs_end = local_end + offset
        segments.append({
            "id": f"l{level}-{idx}",
            "level": level,
            "numeric_id": idx,
            "parent_numeric_id": None,
            "parent_id": None,
            "start_time": format_mmss(abs_start),
            "end_time": format_mmss(abs_end),
            "start_sec": abs_start,
            "end_sec": abs_end,
            "frame_start": abs_start,
            "frame_end": abs_end,
            "label": f"{label_prefix} {idx}",
            "subtitle": "",
            "details": {"clip_local_start": local_start, "clip_local_end": local_end},
        })
    return segments


def build_subset_summary(
    base_summary: dict[str, Any],
    selected_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "loaded": base_summary.get("loaded", False),
        "annotation_dir": base_summary.get("annotation_dir", ""),
        "data_path": base_summary.get("data_path", ""),
        "data_kind": base_summary.get("data_kind", "annotation_json"),
        "clip_count": len(selected_summaries),
        "level1_count": sum(1 for clip in selected_summaries if clip.get("has_level1")),
        "level2_count": sum(1 for clip in selected_summaries if clip.get("has_level2")),
        "level3_count": sum(1 for clip in selected_summaries if clip.get("has_level3")),
        "clips": selected_summaries,
        "total_available_clip_count": base_summary.get("clip_count", len(selected_summaries)),
    }


def normalize_frame_range(start_sec: int, end_sec: int, n_frames: int) -> tuple[int, int]:
    start_idx = max(1, int(start_sec))
    end_idx = max(start_idx, int(end_sec))
    if n_frames > 0:
        start_idx = min(start_idx, n_frames)
        end_idx = min(max(start_idx, end_idx), n_frames)
    return start_idx, end_idx


def sort_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        segments,
        key=lambda item: (
            item.get("start_sec", 10**9),
            item.get("end_sec", 10**9),
            item.get("id", ""),
        ),
    )


def resolve_path(root: Path, path_text: str) -> Path:
    candidate = Path(path_text).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    resolved = (root / candidate).resolve()
    if root not in resolved.parents and resolved != root:
        raise ValueError(f"Relative path must stay inside repo root: {resolved}")
    return resolved


def build_l1_segments(raw_level: dict[str, Any], n_frames: int) -> list[dict[str, Any]]:
    segments = []
    for idx, item in enumerate(raw_level.get("macro_phases") or [], 1):
        if not isinstance(item, dict):
            continue
        phase_id = item.get("phase_id", idx)
        start_sec = parse_mmss(item.get("start_time"))
        end_sec = parse_mmss(item.get("end_time"))
        if start_sec is None or end_sec is None:
            continue
        frame_start, frame_end = normalize_frame_range(start_sec, end_sec, n_frames)
        segments.append(
            {
                "id": f"l1-{phase_id}",
                "level": 1,
                "numeric_id": phase_id,
                "parent_numeric_id": None,
                "parent_id": None,
                "start_time": item.get("start_time") or format_mmss(start_sec),
                "end_time": item.get("end_time") or format_mmss(end_sec),
                "start_sec": start_sec,
                "end_sec": end_sec,
                "frame_start": frame_start,
                "frame_end": frame_end,
                "label": item.get("phase_name") or f"Phase {phase_id}",
                "subtitle": item.get("narrative_summary") or "",
                "details": {
                    "phase_name": item.get("phase_name"),
                    "narrative_summary": item.get("narrative_summary"),
                },
            }
        )
    return sort_segments(segments)


def build_l2_segments(raw_level: dict[str, Any], n_frames: int) -> list[dict[str, Any]]:
    segments = []
    for idx, item in enumerate(raw_level.get("events") or [], 1):
        if not isinstance(item, dict):
            continue
        event_id = item.get("event_id", idx)
        parent_phase_id = item.get("parent_phase_id")
        start_sec = parse_mmss(item.get("start_time"))
        end_sec = parse_mmss(item.get("end_time"))
        if start_sec is None or end_sec is None:
            continue
        frame_start, frame_end = normalize_frame_range(start_sec, end_sec, n_frames)
        segments.append(
            {
                "id": f"l2-{event_id}",
                "level": 2,
                "numeric_id": event_id,
                "parent_numeric_id": parent_phase_id,
                "parent_id": f"l1-{parent_phase_id}" if parent_phase_id is not None else None,
                "start_time": item.get("start_time") or format_mmss(start_sec),
                "end_time": item.get("end_time") or format_mmss(end_sec),
                "start_sec": start_sec,
                "end_sec": end_sec,
                "frame_start": frame_start,
                "frame_end": frame_end,
                "label": item.get("instruction") or f"Event {event_id}",
                "subtitle": ", ".join(item.get("visual_keywords") or []),
                "details": {
                    "instruction": item.get("instruction"),
                    "visual_keywords": item.get("visual_keywords") or [],
                },
            }
        )
    return sort_segments(segments)


def build_l3_segments(raw_level: dict[str, Any], n_frames: int) -> list[dict[str, Any]]:
    segments = []
    for idx, item in enumerate(raw_level.get("grounding_results") or [], 1):
        if not isinstance(item, dict):
            continue
        action_id = item.get("action_id", idx)
        parent_event_id = item.get("parent_event_id")
        start_sec = parse_mmss(item.get("start_time"))
        end_sec = parse_mmss(item.get("end_time"))
        if start_sec is None or end_sec is None:
            continue
        frame_start, frame_end = normalize_frame_range(start_sec, end_sec, n_frames)
        segments.append(
            {
                "id": f"l3-{action_id}",
                "level": 3,
                "numeric_id": action_id,
                "parent_numeric_id": parent_event_id,
                "parent_id": f"l2-{parent_event_id}" if parent_event_id is not None else None,
                "start_time": item.get("start_time") or format_mmss(start_sec),
                "end_time": item.get("end_time") or format_mmss(end_sec),
                "start_sec": start_sec,
                "end_sec": end_sec,
                "frame_start": frame_start,
                "frame_end": frame_end,
                "label": item.get("sub_action") or f"Action {action_id}",
                "subtitle": item.get("post_state") or "",
                "details": {
                    "sub_action": item.get("sub_action"),
                    "pre_state": item.get("pre_state"),
                    "post_state": item.get("post_state"),
                },
            }
        )
    return sort_segments(segments)


def compute_level_diagnostics(segments: list[dict[str, Any]], duration_sec: int) -> dict[str, Any]:
    gaps = []
    overlaps = []
    union_ranges: list[tuple[int, int]] = []
    previous: Optional[dict[str, Any]] = None

    for segment in segments:
        start_sec = int(segment["start_sec"])
        end_sec = int(segment["end_sec"])
        union_ranges.append((start_sec, end_sec))
        if previous is not None:
            if start_sec > int(previous["end_sec"]) + 1:
                gaps.append(
                    {
                        "after": previous["id"],
                        "before": segment["id"],
                        "start_time": format_mmss(int(previous["end_sec"]) + 1),
                        "end_time": format_mmss(start_sec - 1),
                        "duration_sec": start_sec - int(previous["end_sec"]) - 1,
                    }
                )
            if start_sec <= int(previous["end_sec"]):
                overlaps.append(
                    {
                        "left": previous["id"],
                        "right": segment["id"],
                        "start_time": format_mmss(start_sec),
                        "end_time": format_mmss(min(end_sec, int(previous["end_sec"]))),
                    }
                )
        previous = segment

    merged: list[tuple[int, int]] = []
    for start_sec, end_sec in sorted(union_ranges):
        if not merged or start_sec > merged[-1][1] + 1:
            merged.append((start_sec, end_sec))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end_sec))

    covered = sum(end - start + 1 for start, end in merged)
    total = max(1, int(duration_sec))
    return {
        "count": len(segments),
        "gap_count": len(gaps),
        "overlap_count": len(overlaps),
        "gaps": gaps,
        "overlaps": overlaps,
        "coverage_ratio": round(min(1.0, covered / total), 4),
    }


def compute_child_violations(
    child_segments: list[dict[str, Any]],
    parent_segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    parent_map = {segment["numeric_id"]: segment for segment in parent_segments}
    violations = []
    for child in child_segments:
        parent_id = child.get("parent_numeric_id")
        if parent_id is None:
            continue
        parent = parent_map.get(parent_id)
        if parent is None:
            violations.append({"child_id": child["id"], "reason": "missing_parent"})
            continue
        if child["start_sec"] < parent["start_sec"] or child["end_sec"] > parent["end_sec"]:
            violations.append(
                {
                    "child_id": child["id"],
                    "parent_id": parent["id"],
                    "reason": "outside_parent",
                }
            )
    return violations


def build_frame_hits(
    n_frames: int,
    levels: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    hits: dict[str, list[dict[str, Any]]] = {str(idx): [] for idx in range(1, max(1, n_frames) + 1)}
    for segments in levels.values():
        for segment in segments:
            for frame_idx in range(segment["frame_start"], segment["frame_end"] + 1):
                if str(frame_idx) not in hits:
                    hits[str(frame_idx)] = []
                hits[str(frame_idx)].append(
                    {
                        "id": segment["id"],
                        "level": segment["level"],
                        "label": segment["label"],
                        "start_time": segment["start_time"],
                        "end_time": segment["end_time"],
                    }
                )
    return hits


class SegmentationStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self._lock = threading.RLock()
        self.clear()

    def clear(self) -> None:
        self.annotation_dir: Optional[Path] = None
        self.data_kind: str = ""
        self.clips: dict[str, dict[str, Any]] = {}
        self.clip_order: list[str] = []
        self.frame_cache: dict[str, list[dict[str, Any]]] = {}

    def load(self, annotation_dir_text: str) -> dict[str, Any]:
        annotation_dir = resolve_path(self.root, annotation_dir_text)
        data_kind = "annotation_json"
        if annotation_dir.is_file() and annotation_dir.suffix.lower() == ".jsonl":
            clips = self._load_dataset_jsonl(annotation_dir)
            annotation_root = annotation_dir
            data_kind = "dataset_jsonl"
        elif annotation_dir.is_file():
            files = [annotation_dir]
            annotation_root = annotation_dir.parent
        elif annotation_dir.is_dir():
            files = sorted(annotation_dir.glob("*.json"))
            annotation_root = annotation_dir
        else:
            raise FileNotFoundError(f"annotation path not found: {annotation_dir}")

        if data_kind != "dataset_jsonl" and not files:
            raise FileNotFoundError(f"No json files under {annotation_dir}")

        if data_kind != "dataset_jsonl":
            clips = {}
            for file_path in files:
                try:
                    with open(file_path, encoding="utf-8") as f:
                        raw = json.load(f)
                except (OSError, json.JSONDecodeError):
                    continue
                if not isinstance(raw, dict):
                    continue
                clip = self._build_clip(raw, file_path)
                clips[clip["summary"]["clip_key"]] = clip

        with self._lock:
            self.clear()
            self.annotation_dir = annotation_root
            self.data_kind = data_kind
            self.clips = clips
            self.clip_order = sorted(clips.keys())
            return self.summary()

    def summary(self) -> dict[str, Any]:
        clips = [self.clips[key] for key in self.clip_order]
        return {
            "loaded": bool(self.annotation_dir),
            "annotation_dir": str(self.annotation_dir) if self.annotation_dir else "",
            "data_path": str(self.annotation_dir) if self.annotation_dir else "",
            "data_kind": self.data_kind or "annotation_json",
            "clip_count": len(clips),
            "level1_count": sum(1 for clip in clips if clip["summary"]["has_level1"]),
            "level2_count": sum(1 for clip in clips if clip["summary"]["has_level2"]),
            "level3_count": sum(1 for clip in clips if clip["summary"]["has_level3"]),
            "clips": [clip["summary"] for clip in clips],
        }

    def list_clips(self, query: str = "") -> list[dict[str, Any]]:
        query_lower = query.strip().lower()
        clips = [self.clips[key]["summary"] for key in self.clip_order]
        if not query_lower:
            return clips
        return [clip for clip in clips if query_lower in clip["clip_key"].lower()]

    def list_clip_keys(self) -> list[str]:
        return list(self.clip_order)

    def select_preload_clip_keys(
        self,
        max_samples: int = 0,
        prefer_complete: bool = True,
    ) -> list[str]:
        ranked = list(self.clip_order)
        if prefer_complete:
            ranked.sort(
                key=lambda clip_key: (
                    0 if self.clips[clip_key]["summary"]["has_level3"] else 1,
                    0 if self.clips[clip_key]["summary"]["has_level2"] else 1,
                    0 if self.clips[clip_key]["summary"]["has_level1"] else 1,
                    clip_key,
                )
            )
        if max_samples and max_samples > 0:
            ranked = ranked[:max_samples]
        return ranked

    def get_clip(self, clip_key: str) -> Optional[dict[str, Any]]:
        clip = self.clips.get(clip_key)
        if clip is None:
            return None
        payload = clip["payload"]
        if "frame_strip" not in payload:
            payload["frame_strip"] = self._build_frame_strip(payload)
        return payload

    def get_frame_bytes(self, clip_key: str, frame_idx: int, mode: str = "thumb") -> Optional[bytes]:
        clip = self.clips.get(clip_key)
        if clip is None:
            return None

        frame_path = self._resolve_frame_path(clip, frame_idx)
        if frame_path is None or not frame_path.exists():
            return None

        try:
            with Image.open(frame_path) as image:
                image = image.convert("RGB")
                if mode != "full" and image.width > 320:
                    ratio = 320 / float(image.width)
                    image = image.resize((320, max(1, int(image.height * ratio))))
                buf = io.BytesIO()
                image.save(buf, format="JPEG", quality=85)
                payload = buf.getvalue()
        except Exception:
            return None
        return payload

    def approve_clip(self, clip_key: str) -> dict[str, Any]:
        """Move annotation JSON to a sibling 'checked_human' directory."""
        with self._lock:
            clip = self.clips.get(clip_key)
            if clip is None:
                return {"ok": False, "error": f"clip not found: {clip_key}"}

            file_path: Path = clip.get("file_path")
            if file_path is None or not file_path.exists():
                return {"ok": False, "error": f"annotation file not found for {clip_key}"}

            if self.data_kind != "annotation_json":
                return {"ok": False, "error": "approve is only supported for annotation JSON mode"}

            # Destination: sibling directory named 'checked_human'
            src_dir = file_path.parent
            dest_dir = src_dir.parent / "checked_human"
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / file_path.name

            shutil.copy2(str(file_path), str(dest_path))

            # Remove from store
            del self.clips[clip_key]
            self.clip_order = sorted(self.clips.keys())
            self.frame_cache.pop(clip_key, None)

            return {
                "ok": True,
                "clip_key": clip_key,
                "moved_to": str(dest_path),
                "remaining": len(self.clips),
            }

    def _load_dataset_jsonl(self, file_path: Path) -> dict[str, dict[str, Any]]:
        clips: dict[str, dict[str, Any]] = {}
        with open(file_path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(raw, dict):
                    continue
                clip = self._build_dataset_clip(raw, file_path, line_no)
                clips[clip["summary"]["clip_key"]] = clip
        if not clips:
            raise FileNotFoundError(f"No valid dataset records found in {file_path}")
        return clips

    def _build_clip(self, raw: dict[str, Any], file_path: Path) -> dict[str, Any]:
        clip_key = str(raw.get("clip_key") or file_path.stem)
        duration_sec = int(safe_float(raw.get("clip_duration_sec") or raw.get("annotation_end_sec") or 0))
        n_frames = int(safe_float(raw.get("n_frames") or 0))

        level1 = build_l1_segments(raw.get("level1") or {}, n_frames)
        level2 = build_l2_segments(raw.get("level2") or {}, n_frames)
        level3 = build_l3_segments(raw.get("level3") or {}, n_frames)
        levels = {"level1": level1, "level2": level2, "level3": level3}

        diagnostics = {
            "level1": compute_level_diagnostics(level1, duration_sec),
            "level2": compute_level_diagnostics(level2, duration_sec),
            "level3": compute_level_diagnostics(level3, duration_sec),
            "level2_parent_violations": compute_child_violations(level2, level1),
            "level3_parent_violations": compute_child_violations(level3, level2),
        }

        summary = {
            "clip_key": clip_key,
            "duration_sec": duration_sec,
            "duration_label": format_mmss(duration_sec),
            "n_frames": n_frames,
            "annotation_file": str(file_path),
            "frame_dir": raw.get("frame_dir") or "",
            "has_level1": bool(level1),
            "has_level2": bool(level2),
            "has_level3": bool(level3),
            "level1_segments": len(level1),
            "level2_segments": len(level2),
            "level3_segments": len(level3),
            "warning_count": (
                diagnostics["level1"]["overlap_count"]
                + diagnostics["level2"]["overlap_count"]
                + diagnostics["level3"]["overlap_count"]
                + len(diagnostics["level2_parent_violations"])
                + len(diagnostics["level3_parent_violations"])
            ),
        }

        payload = {
            "clip_key": clip_key,
            "video_path": raw.get("video_path"),
            "source_video_path": raw.get("source_video_path"),
            "source_mode": raw.get("source_mode"),
            "annotation_start_sec": raw.get("annotation_start_sec"),
            "annotation_end_sec": raw.get("annotation_end_sec"),
            "window_start_sec": raw.get("window_start_sec"),
            "window_end_sec": raw.get("window_end_sec"),
            "clip_duration_sec": duration_sec,
            "n_frames": n_frames,
            "frame_dir": raw.get("frame_dir") or "",
            "annotated_at": raw.get("annotated_at"),
            "data_kind": "annotation_json",
            "problem_type": "",
            "scope_start_sec": raw.get("annotation_start_sec"),
            "scope_end_sec": raw.get("annotation_end_sec"),
            "levels": levels,
            "sampling": (raw.get("level1") or {}).get("_sampling") or {},
            "warped_mapping": (raw.get("level1") or {}).get("_warped_mapping") or [],
            "segment_calls": {
                "level2": (raw.get("level2") or {}).get("_phase_calls") or [],
                "level3": (raw.get("level3") or {}).get("_segment_calls") or [],
            },
            "diagnostics": diagnostics,
            "frame_hits": build_frame_hits(n_frames, levels),
            "raw": raw,
        }

        return {
            "file_path": file_path,
            "summary": summary,
            "payload": payload,
        }

    def _build_dataset_clip(self, raw: dict[str, Any], file_path: Path, line_no: int) -> dict[str, Any]:
        metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
        answer_data = parse_json_field(raw.get("answer"))
        problem_type = str(raw.get("problem_type") or "")
        base_clip_key = str(metadata.get("clip_key") or file_path.stem)
        video_path = (raw.get("videos") or [None])[0]

        clip_duration_sec = int(safe_float(metadata.get("clip_duration_sec") or 0))
        n_frames = int(safe_float(metadata.get("n_frames") or clip_duration_sec or 0))
        level1: list[dict[str, Any]] = []
        level2: list[dict[str, Any]] = []
        level3: list[dict[str, Any]] = []
        segment_calls = {"level2": [], "level3": []}
        scope_start_sec = metadata.get("window_start_sec")
        scope_end_sec = metadata.get("window_end_sec")
        record_scope_label = ""

        if problem_type == "temporal_seg_hier_L1":
            level = 1
        elif problem_type == "temporal_seg_hier_L2":
            level = 2
        elif problem_type == "temporal_seg_hier_L3":
            level = 3
        else:
            level = 0

        events_pairs = parse_events_answer(raw.get("answer", ""))
        warped_display_frames: list[dict[str, Any]] = []
        action_query: str = ""

        if problem_type == "temporal_seg_hier_L1":
            if events_pairs:
                # NEW clipped format: pairs are warped frame indices (1..n_warped)
                n_warped = int(safe_float(metadata.get("n_warped_frames") or n_frames or 256))
                level1 = build_segments_from_events(events_pairs, level=1, offset=0, label_prefix="Phase")
                warped_display_frames = [
                    {"warped_idx": int(e["warped_idx"]), "real_sec": int(e["real_sec"])}
                    for e in (metadata.get("warped_mapping") or [])
                    if isinstance(e, dict)
                ]
                scope_start_sec = 1
                scope_end_sec = n_warped
                n_frames = n_warped  # frame strip uses warped indices 1..N
            else:
                level1 = build_l1_segments(answer_data, n_frames)
            record_key = base_clip_key
            record_scope_label = f"L1 macro phases · warped frames 1-{n_frames}"

        elif problem_type == "temporal_seg_hier_L2":
            win_start = int(safe_float(metadata.get("window_start_sec") or 0))
            win_end = int(safe_float(metadata.get("window_end_sec") or clip_duration_sec))
            if events_pairs:
                # NEW clipped format: pairs are 0-based seconds within the 128s window
                clip_offset = int(safe_float(metadata.get("clip_offset_sec") or win_start))
                level2 = build_segments_from_events(
                    events_pairs, level=2, offset=clip_offset, label_prefix="Event"
                )
                scope_start_sec = clip_offset
                scope_end_sec = clip_offset + (win_end - win_start)
            else:
                raw_events = answer_data.get("events") or []
                normalized_events = []
                for idx, event in enumerate(raw_events, 1):
                    if not isinstance(event, dict):
                        continue
                    normalized = dict(event)
                    normalized["event_id"] = idx
                    normalized.pop("parent_phase_id", None)
                    normalized_events.append(normalized)
                level2 = build_l2_segments({"events": normalized_events}, n_frames)
                scope_start_sec = win_start
                scope_end_sec = win_end
            record_key = f"{base_clip_key}__L2_{win_start}_{win_end}"
            record_scope_label = f"L2 window {format_mmss(win_start)}-{format_mmss(win_end)}"

        elif problem_type == "temporal_seg_hier_L3":
            event_start = int(safe_float(metadata.get("event_start_sec") or 0))
            event_end = int(safe_float(metadata.get("event_end_sec") or 0))
            action_query = str(metadata.get("action_query") or "")
            if events_pairs:
                # NEW clipped format: pairs are 0-based seconds within the clip+padding window
                clip_start = int(safe_float(
                    metadata.get("clip_start_sec") or metadata.get("clip_offset_sec") or 0
                ))
                clip_end = int(safe_float(metadata.get("clip_end_sec") or event_end))
                level3 = build_segments_from_events(
                    events_pairs, level=3, offset=clip_start, label_prefix="Action"
                )
                # Show parent event as a level-2 box for context
                level2 = [{
                    "id": "l2-1",
                    "level": 2,
                    "numeric_id": 1,
                    "parent_numeric_id": None,
                    "parent_id": None,
                    "start_time": format_mmss(event_start),
                    "end_time": format_mmss(event_end),
                    "start_sec": event_start,
                    "end_sec": event_end,
                    "frame_start": event_start,
                    "frame_end": event_end,
                    "label": action_query or f"Event {event_start}-{event_end}s",
                    "subtitle": "",
                    "details": {"action_query": action_query},
                }]
                scope_start_sec = clip_start
                scope_end_sec = clip_end
            else:
                parent_label = action_query or "Event"
                parent_event = {
                    "event_id": 1,
                    "start_time": event_start,
                    "end_time": event_end,
                    "instruction": parent_label,
                    "visual_keywords": [],
                }
                normalized_results = []
                for idx, result in enumerate(answer_data.get("grounding_results") or [], 1):
                    if not isinstance(result, dict):
                        continue
                    normalized = dict(result)
                    normalized["action_id"] = idx
                    normalized["parent_event_id"] = 1
                    normalized_results.append(normalized)
                level2 = build_l2_segments({"events": [parent_event]}, n_frames)
                level3 = build_l3_segments({"grounding_results": normalized_results}, n_frames)
                scope_start_sec = event_start
                scope_end_sec = event_end
            parent_event_id = int(safe_float(metadata.get("parent_event_id") or 0))
            record_key = f"{base_clip_key}__L3_E{parent_event_id}_{event_start}_{event_end}"
            record_scope_label = (
                f"L3 · {action_query or 'event'} "
                f"{format_mmss(event_start)}-{format_mmss(event_end)}"
            )

        else:
            record_key = f"{base_clip_key}__record_{line_no}"
            record_scope_label = problem_type or "dataset record"

        # For scoped records (L2 window / L3 event), use scope duration for diagnostics
        # and restrict frame_hits to scope frame indices only.
        if (
            isinstance(scope_start_sec, (int, float))
            and isinstance(scope_end_sec, (int, float))
            and scope_end_sec > scope_start_sec
        ):
            diag_duration = int(scope_end_sec) - int(scope_start_sec)
            scope_n_frames = int(scope_end_sec)  # 1fps: frame_idx == second
        else:
            diag_duration = clip_duration_sec or max(1, n_frames)
            scope_n_frames = n_frames

        levels = {"level1": level1, "level2": level2, "level3": level3}
        diagnostics = {
            "level1": compute_level_diagnostics(level1, diag_duration),
            "level2": compute_level_diagnostics(level2, diag_duration),
            "level3": compute_level_diagnostics(level3, diag_duration),
            "level2_parent_violations": compute_child_violations(level2, level1),
            "level3_parent_violations": compute_child_violations(level3, level2),
        }

        summary = {
            "clip_key": record_key,
            "base_clip_key": base_clip_key,
            "duration_sec": clip_duration_sec,
            "duration_label": format_mmss(clip_duration_sec),
            "n_frames": n_frames,
            "annotation_file": f"{file_path}:{line_no}",
            "frame_dir": metadata.get("frame_dir") or "",
            "has_level1": bool(level1),
            "has_level2": bool(level2),
            "has_level3": bool(level3),
            "level1_segments": len(level1),
            "level2_segments": len(level2),
            "level3_segments": len(level3),
            "warning_count": (
                diagnostics["level1"]["overlap_count"]
                + diagnostics["level2"]["overlap_count"]
                + diagnostics["level3"]["overlap_count"]
                + len(diagnostics["level2_parent_violations"])
                + len(diagnostics["level3_parent_violations"])
            ),
        }

        payload = {
            "clip_key": record_key,
            "base_clip_key": base_clip_key,
            "video_path": video_path,
            "source_video_path": video_path,
            "source_mode": metadata.get("source_mode") or "dataset_jsonl",
            "annotation_start_sec": 0,
            "annotation_end_sec": clip_duration_sec,
            "window_start_sec": metadata.get("window_start_sec"),
            "window_end_sec": metadata.get("window_end_sec"),
            "clip_duration_sec": clip_duration_sec,
            "n_frames": n_frames,
            "frame_dir": metadata.get("frame_dir") or "",
            "annotated_at": metadata.get("annotated_at"),
            "data_kind": "dataset_jsonl",
            "problem_type": problem_type,
            "scope_start_sec": scope_start_sec,
            "scope_end_sec": scope_end_sec,
            "record_scope_label": record_scope_label,
            "levels": levels,
            "sampling": {"n_sampled_frames": metadata.get("n_warped_frames")},
            "warped_mapping": metadata.get("warped_mapping") or [],
            "warped_display_frames": warped_display_frames,
            "action_query": action_query,
            "segment_calls": segment_calls,
            "diagnostics": diagnostics,
            "frame_hits": build_frame_hits(scope_n_frames, levels),
            "raw": raw,
        }

        return {
            "file_path": file_path,
            "summary": summary,
            "payload": payload,
        }

    def _build_frame_strip(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        clip_key = str(payload.get("clip_key") or "")
        if clip_key in self.frame_cache:
            return self.frame_cache[clip_key]

        frame_dir_text = str(payload.get("frame_dir") or "")
        n_frames = int(payload.get("n_frames") or 0)
        sampled = set((payload.get("sampling") or {}).get("sampled_frame_indices") or [])
        if not sampled:
            sampled = {
                int(entry.get("real_sec"))
                for entry in (payload.get("warped_mapping") or [])
                if isinstance(entry, dict) and safe_float(entry.get("real_sec"), -1) >= 0
            }

        # For dataset_jsonl records, limit the frame strip to the annotated scope
        # (e.g. L3 event window or L2 sliding window) instead of the full clip.
        # 1fps frames: frame_idx == second, so scope maps directly to frame indices.
        scope_start = payload.get("scope_start_sec")
        scope_end = payload.get("scope_end_sec")
        if (
            payload.get("data_kind") == "dataset_jsonl"
            and isinstance(scope_start, (int, float))
            and isinstance(scope_end, (int, float))
            and scope_end > scope_start
        ):
            frame_start = max(1, int(scope_start))
            frame_end = min(n_frames if n_frames > 0 else int(scope_end), int(scope_end))
        else:
            frame_start = 1
            frame_end = n_frames

        strip: list[dict[str, Any]] = []

        frame_dir: Optional[Path] = None
        if frame_dir_text:
            try:
                frame_dir = resolve_path(self.root, frame_dir_text)
            except ValueError:
                frame_dir = None

        # L1 warped display: use warped_display_frames to map warped_idx → real frame file
        warped_display = payload.get("warped_display_frames")
        if warped_display:
            strip: list[dict[str, Any]] = []
            for entry in warped_display:
                warped_idx = int(entry.get("warped_idx", 0))
                real_sec = int(entry.get("real_sec", warped_idx))
                frame_path = self._resolve_frame_path_from_dir(frame_dir, real_sec) if frame_dir else None
                src = None
                if frame_path is not None and frame_path.exists():
                    try:
                        with Image.open(frame_path) as image:
                            src = image_to_data_url(image, max_width=320)
                    except Exception:
                        src = None
                strip.append({
                    "frame_idx": warped_idx,
                    "timestamp": format_mmss(real_sec),
                    "sampled": True,
                    "src": src,
                })
            self.frame_cache[clip_key] = strip
            return strip

        for frame_idx in range(frame_start, frame_end + 1):
            frame_path = self._resolve_frame_path_from_dir(frame_dir, frame_idx) if frame_dir else None
            src = None
            if frame_path is not None and frame_path.exists():
                try:
                    with Image.open(frame_path) as image:
                        src = image_to_data_url(image, max_width=320)
                except Exception:
                    src = None
            strip.append(
                {
                    "frame_idx": frame_idx,
                    "timestamp": format_mmss(frame_idx),
                    "sampled": frame_idx in sampled,
                    "src": src,
                }
            )

        self.frame_cache[clip_key] = strip
        return strip

    def _resolve_frame_path(self, clip: dict[str, Any], frame_idx: int) -> Optional[Path]:
        frame_dir_text = str(clip["payload"].get("frame_dir") or "")
        if not frame_dir_text:
            return None
        try:
            frame_dir = resolve_path(self.root, frame_dir_text)
        except ValueError:
            return None
        if not frame_dir.exists():
            return None

        primary = frame_dir / f"{frame_idx:04d}.jpg"
        if primary.exists():
            return primary

        for ext in sorted(IMAGE_EXTS):
            candidate = frame_dir / f"{frame_idx:04d}{ext}"
            if candidate.exists():
                return candidate

        for candidate in frame_dir.glob(f"{frame_idx:04d}.*"):
            if candidate.suffix.lower() in IMAGE_EXTS:
                return candidate
        return None

    def _resolve_frame_path_from_dir(self, frame_dir: Path, frame_idx: int) -> Optional[Path]:
        primary = frame_dir / f"{frame_idx:04d}.jpg"
        if primary.exists():
            return primary
        for ext in sorted(IMAGE_EXTS):
            candidate = frame_dir / f"{frame_idx:04d}{ext}"
            if candidate.exists():
                return candidate
        for candidate in frame_dir.glob(f"{frame_idx:04d}.*"):
            if candidate.suffix.lower() in IMAGE_EXTS:
                return candidate
        return None


class SegmentationHandler(BaseHTTPRequestHandler):
    server_version = "SegmentationVisualize/0.1"

    @property
    def store(self) -> SegmentationStore:
        return self.server.store  # type: ignore[attr-defined]

    @property
    def static_dir(self) -> Path:
        return self.server.static_dir  # type: ignore[attr-defined]

    @property
    def preloaded_html(self) -> Optional[bytes]:
        return self.server.preloaded_html  # type: ignore[attr-defined]

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self._handle_api(parsed)
            return
        self._serve_static(parsed.path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/approve/"):
            clip_key = unquote(parsed.path[len("/api/approve/"):])
            result = self.store.approve_clip(clip_key)
            status = HTTPStatus.OK if result.get("ok") else HTTPStatus.BAD_REQUEST
            self._json(result, status)
            return
        self._json({"error": f"unknown POST path: {parsed.path}"}, HTTPStatus.NOT_FOUND)

    def _handle_api(self, parsed) -> None:  # type: ignore[no-untyped-def]
        query = parse_qs(parsed.query)
        path = parsed.path

        try:
            if path == "/api/load-data":
                annotation_dir = (query.get("data_path") or query.get("annotation_dir") or [""])[0]
                if not annotation_dir:
                    self._json({"ok": False, "error": "data_path is required"}, HTTPStatus.BAD_REQUEST)
                    return
                summary = self.store.load(annotation_dir)
                self._json({"ok": True, "summary": summary})
                return

            if path == "/api/state":
                self._json(self.store.summary())
                return

            if path == "/api/clips":
                search = (query.get("search") or [""])[0]
                self._json({"clips": self.store.list_clips(search)})
                return

            if path.startswith("/api/clip/"):
                clip_key = unquote(path[len("/api/clip/"):])
                clip = self.store.get_clip(clip_key)
                if clip is None:
                    self._json({"error": f"clip not found: {clip_key}"}, HTTPStatus.NOT_FOUND)
                    return
                self._json(clip)
                return

            if path.startswith("/api/frame/"):
                parts = path.split("/")
                if len(parts) < 5:
                    self._json({"error": "invalid frame path"}, HTTPStatus.BAD_REQUEST)
                    return
                clip_key = unquote(parts[3])
                try:
                    frame_idx = int(parts[4])
                except ValueError:
                    self._json({"error": "frame_idx must be int"}, HTTPStatus.BAD_REQUEST)
                    return
                mode = (query.get("mode") or ["thumb"])[0]
                payload = self.store.get_frame_bytes(clip_key, frame_idx, mode=mode)
                if payload is None:
                    self.send_error(HTTPStatus.NOT_FOUND, "frame not found")
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return

            self._json({"error": f"unknown api path: {path}"}, HTTPStatus.NOT_FOUND)
        except FileNotFoundError as exc:
            self._json({"ok": False, "error": str(exc)}, HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self._json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # noqa: BLE001
            self._json({"ok": False, "error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def _serve_static(self, path_text: str) -> None:
        clean_path = path_text or "/"
        if clean_path == "/":
            clean_path = "/index.html"

        target = (self.static_dir / clean_path.lstrip("/")).resolve()
        if self.static_dir not in target.parents and target != self.static_dir:
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not target.exists() or not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        mime_type, _ = mimetypes.guess_type(str(target))
        payload = target.read_bytes()
        if target.name == "index.html" and self.preloaded_html:
            marker = b"</head>"
            if marker in payload:
                payload = payload.replace(marker, self.preloaded_html + marker, 1)
            else:
                payload = self.preloaded_html + payload
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format_str: str, *args: Any) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Segmentation annotation visualization server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8787, help="Bind port")
    parser.add_argument(
        "--static-dir",
        default=str(Path(__file__).resolve().parent),
        help="Directory for index.html and assets",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Pre-load annotation directory, annotation json, or built dataset jsonl on startup",
    )
    parser.add_argument(
        "--annotation-dir",
        default=None,
        help="Backward-compatible alias for --data-path",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Preload at most N clip details into HTML (0 = all clips)",
    )
    parser.add_argument(
        "--prefer-complete",
        action="store_true",
        help="Prioritize clips with level1+level2+level3 when selecting preloaded samples",
    )
    args = parser.parse_args()

    static_dir = Path(args.static_dir).expanduser().resolve()
    if not static_dir.exists():
        raise FileNotFoundError(f"static-dir not found: {static_dir}")

    store = SegmentationStore(static_dir.parents[1])
    preloaded_html: Optional[bytes] = None
    preload_path = args.data_path or args.annotation_dir
    if preload_path:
        try:
            summary = store.load(preload_path)
            print(f"Pre-loaded data from: {preload_path}")
            selected_clip_keys = store.select_preload_clip_keys(
                max_samples=args.max_samples,
                prefer_complete=args.prefer_complete,
            )
            n_complete = sum(
                1
                for clip_key in selected_clip_keys
                if store.clips[clip_key]["summary"]["has_level1"]
                and store.clips[clip_key]["summary"]["has_level2"]
                and store.clips[clip_key]["summary"]["has_level3"]
            )
            print(
                "  Building embedded frame strips and clip details..."
                f" selected={len(selected_clip_keys)}"
                f" complete_l123={n_complete}"
                f" total_available={summary['clip_count']}"
            )
            all_details = {}
            total_selected = len(selected_clip_keys)
            for i, clip_key in enumerate(selected_clip_keys, 1):
                clip = store.get_clip(clip_key)
                if clip is not None:
                    all_details[clip_key] = clip
                print_progress("  Preloading clips", i, total_selected)
            selected_summaries = [
                store.clips[clip_key]["summary"]
                for clip_key in selected_clip_keys
                if clip_key in store.clips
            ]
            preload_summary = build_subset_summary(summary, selected_summaries)
            preload = {
                "summary": preload_summary,
                "all_details": all_details,
                "annotation_dir": preload_path,
                "data_path": preload_path,
                "preloaded_clip_keys": selected_clip_keys,
                "max_samples": args.max_samples,
                "prefer_complete": args.prefer_complete,
            }
            preloaded_html = (
                b'<script>window.__PRELOADED__='
                + json.dumps(preload, ensure_ascii=False).encode("utf-8")
                + b";</script>\n"
            )
            print(f"  Ready. Preloaded HTML inject size: {len(preloaded_html) / 1024:.0f} KB")
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] Failed to pre-load data: {exc}")

    server = ThreadingHTTPServer((args.host, args.port), SegmentationHandler)
    server.static_dir = static_dir  # type: ignore[attr-defined]
    server.store = store  # type: ignore[attr-defined]
    server.preloaded_html = preloaded_html  # type: ignore[attr-defined]

    print(f"Segmentation visualization server running at http://{args.host}:{args.port}/")
    print(f"Static dir: {static_dir}")
    server.serve_forever()


if __name__ == "__main__":
    main()
