#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import base64
import json
import math
import re
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, unquote, urlparse

from PIL import Image


_SEGMENT_PATTERN = re.compile(r"\[\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\]")
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpeg", ".mpg"}
_CHOICE_TASKS = {"add", "delete", "replace", "aot_v2t", "aot_t2v"}
_SEG_TIMELINE_MAX_FRAMES = 64


def _phase_rank(phase: str) -> int:
    return 0 if phase == "train" else 1


def _make_step_key(phase: str, step: int) -> str:
    return f"{phase}:{step}"


def _format_step_label(phase: str, step: int) -> str:
    return f"Val {step}" if phase == "val" else f"Step {step}"


def _format_mmss(total_seconds: Any) -> str:
    total = max(0, int(_safe_float(total_seconds, 0.0)))
    minutes, seconds = divmod(total, 60)
    return f"{minutes:02d}:{seconds:02d}"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_segments(text: str) -> list[list[float]]:
    if not text:
        return []
    segments = []
    for start_raw, end_raw in _SEGMENT_PATTERN.findall(str(text)):
        start = _safe_float(start_raw, -1)
        end = _safe_float(end_raw, -1)
        if start < 0 or end <= start:
            continue
        segments.append([start, end])
    return segments


def _sample_evenly(items: list[Any], max_n: int) -> list[Any]:
    if len(items) <= max_n:
        return items
    stride = (len(items) - 1) / float(max_n - 1)
    return [items[round(i * stride)] for i in range(max_n)]


def _image_to_data_url(image: Image.Image, max_width: int = 220) -> str:
    image = image.convert("RGB")
    if image.width > max_width:
        ratio = max_width / float(image.width)
        image = image.resize((max_width, max(1, int(image.height * ratio))))
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=85)
    payload = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{payload}"


def _path_to_data_url(path: Path) -> Optional[str]:
    try:
        with Image.open(path) as image:
            return _image_to_data_url(image)
    except Exception:
        return None


def _looks_like_base64(value: str) -> bool:
    if len(value) < 48 or " " in value or "\n" in value:
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9+/=]+", value))


def _parse_answer_tag(text: str) -> str:
    """Return the trimmed content of the first <answer>…</answer> tag, or ''."""
    m = _ANSWER_RE.search(text or "")
    return m.group(1).strip() if m else ""


def _compact_detail_for_preload(detail: dict[str, Any]) -> dict[str, Any]:
    task = str(detail.get("problem_type") or "")
    compact = {
        "uid": detail.get("uid"),
        "step": detail.get("step"),
        "phase": detail.get("phase"),
        "step_key": detail.get("step_key"),
        "step_label": detail.get("step_label"),
        "problem_type": task,
        "mean_reward": detail.get("mean_reward"),
        "prompt": detail.get("prompt"),
        "ground_truth": detail.get("ground_truth"),
        "missing_clip_index": detail.get("missing_clip_index"),
        "clip_boundaries": detail.get("clip_boundaries") or [],
        "timeline_max_t": detail.get("timeline_max_t"),
        "choice_meta": detail.get("choice_meta"),
        "sort_meta": detail.get("sort_meta"),
        "attempts": [],
    }

    if task == "temporal_seg":
        # frames are loaded lazily via /api/group/<uid>/frames
        compact["timeline_frame_strip"] = []
        compact["frame_strip"] = []
        compact["clip_boundaries"] = []
    else:
        # frames are loaded lazily via /api/group/<uid>/frames
        compact["frame_strip"] = []
    for attempt in detail.get("attempts") or []:
        compact_attempt = {
            "reward": attempt.get("reward"),
            "response": attempt.get("response"),
        }
        if task == "temporal_seg":
            compact_attempt["temporal_segments"] = attempt.get("temporal_segments") or {}
        if "pred_letter" in attempt:
            compact_attempt["pred_letter"] = attempt.get("pred_letter")
        if "pred_order" in attempt:
            compact_attempt["pred_order"] = attempt.get("pred_order")
        compact["attempts"].append(compact_attempt)

    return compact


def _select_preload_step_keys(
    steps: list[dict[str, Any]],
    train_step_interval: int,
) -> set[str]:
    interval = max(1, int(train_step_interval))
    selected: set[str] = set()
    train_steps = [s for s in steps if str(s.get("phase")) != "val"]
    val_steps = [s for s in steps if str(s.get("phase")) == "val"]

    for step_info in val_steps:
        step_key = str(step_info.get("step_key") or "")
        if step_key:
            selected.add(step_key)

    if not train_steps:
        return selected

    max_train_step = max(int(s.get("step", 0)) for s in train_steps)
    min_train_step = min(int(s.get("step", 0)) for s in train_steps)
    for step_info in train_steps:
        step = int(step_info.get("step", 0))
        step_key = str(step_info.get("step_key") or "")
        if not step_key:
            continue
        if step == min_train_step or step == max_train_step or step % interval == 0:
            selected.add(step_key)

    return selected


class RolloutStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self._lock = threading.RLock()
        self.clear()

    def clear(self) -> None:
        self.rollout_dir: Optional[Path] = None
        self.log_file: Optional[Path] = None
        self.groups: dict[str, dict[str, Any]] = {}
        self.group_order: list[str] = []
        self.task_counts: dict[str, int] = {}
        self.step_curve: list[dict[str, float]] = []
        self.training_metrics: list[dict[str, Any]] = []
        self.frame_cache: dict[str, list[str]] = {}
        self.timeline_frame_cache: dict[str, list[dict[str, Any]]] = {}
        self.total_samples = 0

    def _resolve_local_path(self, path_text: str) -> Path:
        candidate = Path(path_text).expanduser()
        if not candidate.is_absolute():
            # Relative paths are resolved against repo root; prevent traversal.
            candidate = (self.root / candidate).resolve()
            if self.root not in candidate.parents and candidate != self.root:
                raise ValueError(f"Relative path must stay inside repo root: {candidate}")
        else:
            candidate = candidate.resolve()
        return candidate

    def load(self, rollout_dir_text: str, log_file_text: Optional[str]) -> dict[str, Any]:
        rollout_dir = self._resolve_local_path(rollout_dir_text)
        if not rollout_dir.exists() or not rollout_dir.is_dir():
            raise FileNotFoundError(f"rollout_dir not found: {rollout_dir}")

        log_file = None
        if log_file_text:
            log_file = self._resolve_local_path(log_file_text)
            if not log_file.exists():
                raise FileNotFoundError(f"log_file not found: {log_file}")

        rollout_files = sorted(rollout_dir.glob("step_*.jsonl")) + sorted(rollout_dir.glob("val_step_*.jsonl"))
        if not rollout_files:
            rollout_files = sorted(rollout_dir.glob("*.jsonl"))
        if not rollout_files:
            raise FileNotFoundError(f"No jsonl files under {rollout_dir}")

        with self._lock:
            self.clear()
            self.rollout_dir = rollout_dir
            self.log_file = log_file

            for file_path in rollout_files:
                with open(file_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        self._consume_record(record)

            if self.log_file:
                self.step_curve = self._load_step_curve(self.log_file)
            else:
                self.step_curve = self._build_curve_from_groups()

            self.group_order = sorted(
                self.groups.keys(),
                key=lambda uid: (
                    self.groups[uid].get("step", 0),
                    _phase_rank(str(self.groups[uid].get("phase", "train"))),
                    -self.groups[uid].get("mean_reward", 0.0),
                ),
            )
            return self.summary()

    def _consume_record(self, record: dict[str, Any]) -> None:
        step = int(record.get("step", 0))
        phase = str(record.get("phase") or "train")
        uid = str(record.get("uid") or f"step-{step}-sample-{self.total_samples}")
        task = str(record.get("problem_type") or "unknown")
        reward = _safe_float(record.get("reward"), 0.0)
        response = str(record.get("response") or "")
        ground_truth = str(record.get("ground_truth") or "")
        temporal_segments = record.get("temporal_segments")
        if not isinstance(temporal_segments, dict):
            temporal_segments = {
                "predicted": _extract_segments(response),
                "ground_truth": _extract_segments(ground_truth),
            }

        attempt = {
            "attempt_index": len(self.groups.get(uid, {}).get("attempts", [])),
            "reward": reward,
            "response": response,
            "temporal_segments": temporal_segments,
        }

        group = self.groups.get(uid)
        if group is None:
            group = {
                "uid": uid,
                "step": step,
                "phase": phase,
                "step_key": _make_step_key(phase, step),
                "step_label": _format_step_label(phase, step),
                "problem_type": task,
                "data_type": record.get("data_type"),
                "prompt": str(record.get("prompt") or ""),
                "ground_truth": ground_truth,
                "problem_id": record.get("problem_id"),
                "problem": record.get("problem"),
                "video_paths": record.get("video_paths") or [],
                "image_paths": record.get("image_paths") or [],
                "multi_modal_source": record.get("multi_modal_source"),
                "attempts": [],
                "mean_reward": 0.0,
            }
            self.groups[uid] = group
            self.task_counts[task] = self.task_counts.get(task, 0) + 1

        group["attempts"].append(attempt)
        rewards = [a["reward"] for a in group["attempts"]]
        group["mean_reward"] = sum(rewards) / float(len(rewards))
        self.total_samples += 1

    def _load_step_curve(self, log_file: Path) -> list[dict[str, float]]:
        points = []
        self.training_metrics: list[dict[str, Any]] = []
        with open(log_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Skip entries where reward.overall is missing or null —
                # the last in-progress step often has no reward written yet,
                # which would otherwise appear as a spurious 0 on the chart.
                overall = (row.get("reward") or {}).get("overall")
                if overall is None:
                    continue
                points.append(
                    {
                        "step": _safe_float(row.get("step")),
                        "reward": _safe_float(overall),
                    }
                )
                # Extract training metrics for visualization
                actor = row.get("actor") or {}
                critic = row.get("critic") or {}
                perf = row.get("perf") or {}
                resp_len = row.get("response_length") or {}
                advantages = critic.get("advantages") or {}
                self.training_metrics.append({
                    "step": _safe_float(row.get("step")),
                    "kl_loss": _safe_float(actor.get("kl_loss")),
                    "ppo_kl": _safe_float(actor.get("ppo_kl")),
                    "pg_loss": _safe_float(actor.get("pg_loss")),
                    "entropy_loss": _safe_float(actor.get("entropy_loss")),
                    "pg_clipfrac_higher": _safe_float(actor.get("pg_clipfrac_higher")),
                    "pg_clipfrac_lower": _safe_float(actor.get("pg_clipfrac_lower")),
                    "grad_norm": _safe_float(actor.get("grad_norm")),
                    "lr": _safe_float(actor.get("lr")),
                    "advantage_mean": _safe_float(advantages.get("mean")),
                    "advantage_max": _safe_float(advantages.get("max")),
                    "advantage_min": _safe_float(advantages.get("min")),
                    "response_length_mean": _safe_float(resp_len.get("mean")),
                    "throughput": _safe_float(perf.get("throughput")),
                })
        self.training_metrics.sort(key=lambda x: x["step"])
        return sorted(points, key=lambda x: x["step"])

    def _build_curve_from_groups(self) -> list[dict[str, float]]:
        step_map: dict[str, dict[str, Any]] = {}
        for group in self.groups.values():
            step_key = str(group.get("step_key"))
            if step_key not in step_map:
                step_map[step_key] = {
                    "step": int(group.get("step", 0)),
                    "phase": str(group.get("phase", "train")),
                    "step_label": str(group.get("step_label") or ""),
                    "rewards": [],
                }
            step_map[step_key]["rewards"].append(float(group.get("mean_reward", 0)))
        points = []
        for step_key, info in step_map.items():
            vals = info["rewards"]
            points.append(
                {
                    "step": float(info["step"]),
                    "phase": info["phase"],
                    "step_key": step_key,
                    "step_label": info["step_label"],
                    "reward": sum(vals) / max(len(vals), 1),
                }
            )
        return sorted(points, key=lambda x: (x["step"], _phase_rank(str(x.get("phase", "train")))))

    def _build_task_curves(self) -> dict[str, list[dict[str, float]]]:
        """Per-task mean-reward curve over training steps."""
        task_step: dict[str, dict[int, list[float]]] = {}
        for g in self.groups.values():
            task = str(g.get("problem_type") or "unknown")
            step = int(g.get("step", 0))
            task_step.setdefault(task, {}).setdefault(step, []).append(float(g.get("mean_reward", 0)))
        return {
            task: [
                {"step": float(s), "reward": sum(v) / max(len(v), 1)}
                for s, v in sorted(step_map.items())
            ]
            for task, step_map in task_step.items()
        }

    def get_steps_summary(self) -> list[dict[str, Any]]:
        """Per-step summary: step, mean_reward, task_counts, n_groups."""
        step_data: dict[str, dict[str, Any]] = {}
        for g in self.groups.values():
            step = int(g.get("step", 0))
            phase = str(g.get("phase", "train"))
            step_key = _make_step_key(phase, step)
            if step_key not in step_data:
                step_data[step_key] = {
                    "step": step,
                    "phase": phase,
                    "step_key": step_key,
                    "step_label": _format_step_label(phase, step),
                    "rewards": [],
                    "tasks": {},
                    "n_groups": 0,
                }
            step_data[step_key]["rewards"].append(float(g.get("mean_reward", 0)))
            task = str(g.get("problem_type") or "unknown")
            step_data[step_key]["tasks"][task] = step_data[step_key]["tasks"].get(task, 0) + 1
            step_data[step_key]["n_groups"] += 1
        result = []
        for _, d in sorted(step_data.items(), key=lambda item: (item[1]["step"], _phase_rank(item[1]["phase"]))):
            result.append({
                "step": d["step"],
                "phase": d["phase"],
                "step_key": d["step_key"],
                "step_label": d["step_label"],
                "mean_reward": sum(d["rewards"]) / max(len(d["rewards"]), 1),
                "n_groups": d["n_groups"],
                "task_counts": d["tasks"],
            })
        return result

    def _build_choice_meta(self, detail: dict[str, Any]) -> dict[str, Any]:
        """
        For add/delete/replace tasks: determine GT letter and option_type,
        and annotate each attempt with pred_letter.
        """
        gt_raw = str(detail.get("ground_truth") or "")
        m = re.search(r"\b([A-D])\b", gt_raw, re.IGNORECASE)
        gt_letter = m.group(1).upper() if m else ""

        # Heuristic: video options when multi_modal_source carries multiple media entries
        mm = detail.get("multi_modal_source")
        option_type = "text"
        if isinstance(mm, dict):
            n_videos = len(mm.get("videos") or [])
            n_images = len(mm.get("images") or [])
            if n_videos > 1 or n_images > 1:
                option_type = "video"

        for attempt in detail.get("attempts", []):
            raw = _parse_answer_tag(attempt.get("response") or "")
            m2 = re.search(r"[A-D]", raw, re.IGNORECASE)
            attempt["pred_letter"] = m2.group(0).upper() if m2 else ""

        return {"gt_letter": gt_letter, "option_type": option_type}

    def _build_sort_meta(self, detail: dict[str, Any]) -> dict[str, Any]:
        """
        For sort tasks: extract GT clip order and annotate each attempt with pred_order.
        Both GT and predictions are digit sequences (e.g., '12345' or '1 2 3 4 5').
        """
        gt_raw = str(detail.get("ground_truth") or "")
        gt_ans = _parse_answer_tag(gt_raw) or gt_raw.strip()
        gt_order = [int(d) for d in re.findall(r"[1-9]", gt_ans)]

        for attempt in detail.get("attempts", []):
            resp = attempt.get("response") or ""
            pred_raw = _parse_answer_tag(resp)
            attempt["pred_order"] = [int(d) for d in re.findall(r"[1-9]", pred_raw)] if pred_raw else []

        return {"gt_order": gt_order}

    def summary(self) -> dict[str, Any]:
        group_count = len(self.groups)
        temporal_count = sum(1 for g in self.groups.values() if g.get("problem_type") == "temporal_seg")
        mean_reward = sum(g.get("mean_reward", 0.0) for g in self.groups.values()) / max(group_count, 1)
        return {
            "group_count": group_count,
            "sample_count": self.total_samples,
            "step_count": len({g.get("step_key") for g in self.groups.values()}),
            "mean_group_reward": mean_reward,
            "temporal_ratio": temporal_count / max(group_count, 1),
            "task_counts": self.task_counts,
            "step_curve": self.step_curve,
            "task_curves": self._build_task_curves(),
            "training_metrics": self.training_metrics,
            "rollout_dir": str(self.rollout_dir) if self.rollout_dir else None,
            "log_file": str(self.log_file) if self.log_file else None,
        }

    def query_groups(
        self,
        step: Optional[int],
        step_key: Optional[str],
        task: Optional[str],
        query: Optional[str],
        limit: int,
    ) -> list[dict[str, Any]]:
        query_lc = (query or "").strip().lower()
        rows = []
        for uid in self.group_order:
            g = self.groups[uid]
            if step_key is not None and str(g.get("step_key")) != step_key:
                continue
            if step_key is None and step is not None and int(g.get("step", -1)) != step:
                continue
            if task and task != "all" and str(g.get("problem_type")) != task:
                continue
            if query_lc:
                text_blob = " ".join([str(g.get("uid")), str(g.get("prompt")), str(g.get("ground_truth"))]).lower()
                if query_lc not in text_blob:
                    continue
            rows.append(
                {
                    "uid": g["uid"],
                    "step": g["step"],
                    "phase": g.get("phase"),
                    "step_key": g.get("step_key"),
                    "step_label": g.get("step_label"),
                    "problem_type": g["problem_type"],
                    "mean_reward": g["mean_reward"],
                    "n_rollouts": len(g["attempts"]),
                    "prompt_preview": str(g.get("prompt") or "")[:180],
                }
            )
            if len(rows) >= limit:
                break
        return rows

    def get_group_detail(
        self,
        uid: str,
        max_frames: int = 30,
        timeline_fps: int = 0,
        text_only: bool = False,
    ) -> dict[str, Any]:
        group = self.groups.get(uid)
        if group is None:
            raise KeyError(f"uid not found: {uid}")
        detail = dict(group)
        detail["attempts"] = sorted(group["attempts"], key=lambda a: a["reward"], reverse=True)
        if text_only:
            # Skip frame extraction entirely (frames are served lazily via /frames endpoint)
            detail["frame_strip"] = []
        else:
            detail["frame_strip"] = self._get_frame_strip(uid, max_frames=max_frames)
        detail["clip_boundaries"] = group.get("_clip_boundaries", [])
        detail["timeline_max_t"] = self._timeline_max_t(detail)
        # Detect [MISSING] position for replace/add/delete tasks
        detail["missing_clip_index"] = self._detect_missing_position(detail)
        task = str(detail.get("problem_type") or "")
        if task == "temporal_seg" and timeline_fps >= 1 and not text_only:
            detail["timeline_frame_strip"] = self._get_timeline_frame_strip(
                uid,
                detail["timeline_max_t"],
                timeline_fps,
                max_frames=_SEG_TIMELINE_MAX_FRAMES,
            )
        if task in _CHOICE_TASKS:
            detail["choice_meta"] = self._build_choice_meta(detail)
        elif task == "sort":
            detail["sort_meta"] = self._build_sort_meta(detail)
        detail.pop("_clip_boundaries", None)
        return detail

    @staticmethod
    def _detect_missing_position(detail: dict[str, Any]) -> Optional[int]:
        """Parse prompt to find which step index is [MISSING]. Returns clip index (0-based)."""
        prompt = detail.get("prompt") or ""
        steps = re.findall(r'Step\s+(\d+)\s*:\s*(.*?)(?=Step\s+\d+\s*:|$)', prompt, re.DOTALL)
        if not steps:
            return None
        for i, (step_num, content) in enumerate(steps):
            if "[MISSING]" in content.upper():
                return i
        return None

    def _timeline_max_t(self, detail: dict[str, Any]) -> float:
        max_t = 0.0
        for attempt in detail.get("attempts", []):
            segs = attempt.get("temporal_segments") or {}
            for part in ("predicted", "ground_truth"):
                for seg in segs.get(part, []) or []:
                    if isinstance(seg, list) and len(seg) == 2:
                        max_t = max(max_t, _safe_float(seg[1], 0.0))
        return max_t

    def _get_frame_strip(self, uid: str, max_frames: int) -> list[str]:
        if uid in self.frame_cache:
            return self.frame_cache[uid]
        # Try disk cache
        disk_data = self._load_frame_disk_cache(uid)
        if disk_data is not None:
            frames, boundaries = disk_data
            self.frame_cache[uid] = frames
            self.groups[uid]["_clip_boundaries"] = boundaries
            return frames
        group = self.groups[uid]
        frames = self._extract_frames(group, max_frames=max_frames)
        self.frame_cache[uid] = frames
        self._save_frame_disk_cache(uid, frames, group.get("_clip_boundaries", []))
        return frames

    def _get_timeline_frame_strip(
        self,
        uid: str,
        timeline_max_t: float,
        timeline_fps: int = 1,
        max_frames: int = _SEG_TIMELINE_MAX_FRAMES,
    ) -> list[dict[str, Any]]:
        cache_key = f"{uid}@{max(1, int(timeline_fps))}fps_{max(1, int(max_frames))}f"
        if cache_key in self.timeline_frame_cache:
            return self.timeline_frame_cache[cache_key]
        disk_frames = self._load_timeline_frame_disk_cache(uid, timeline_fps=timeline_fps, max_frames=max_frames)
        if disk_frames is not None:
            self.timeline_frame_cache[cache_key] = disk_frames
            return disk_frames
        group = self.groups[uid]
        frames = self._extract_timeline_frames(
            group,
            timeline_max_t=timeline_max_t,
            timeline_fps=timeline_fps,
            max_frames=max_frames,
        )
        self.timeline_frame_cache[cache_key] = frames
        self._save_timeline_frame_disk_cache(uid, frames, timeline_fps=timeline_fps, max_frames=max_frames)
        return frames

    def _frame_cache_dir(self) -> Optional[Path]:
        if not self.rollout_dir:
            return None
        cache_dir = self.rollout_dir / ".frame_cache"
        return cache_dir

    def _timeline_frame_cache_dir(self, timeline_fps: int = 1, max_frames: int = _SEG_TIMELINE_MAX_FRAMES) -> Optional[Path]:
        if not self.rollout_dir:
            return None
        fps = max(1, int(timeline_fps))
        frame_cap = max(1, int(max_frames))
        return self.rollout_dir / f".timeline_frame_cache_{fps}fps_{frame_cap}f"

    def _load_frame_disk_cache(self, uid: str) -> Optional[tuple[list[str], list[int]]]:
        cache_dir = self._frame_cache_dir()
        if cache_dir is None:
            return None
        safe_name = re.sub(r'[^\w\-.]', '_', uid) + ".json"
        cache_file = cache_dir / safe_name
        if not cache_file.exists():
            return None
        try:
            with open(cache_file, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return (data.get("frames", []), data.get("clip_boundaries", []))
            if isinstance(data, list):
                # Legacy format: just frames, no boundaries
                return (data, [])
        except Exception:
            pass
        return None

    def _save_frame_disk_cache(self, uid: str, frames: list[str], clip_boundaries: list[int] = None) -> None:
        cache_dir = self._frame_cache_dir()
        if cache_dir is None:
            return
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            safe_name = re.sub(r'[^\w\-.]', '_', uid) + ".json"
            cache_file = cache_dir / safe_name
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump({"frames": frames, "clip_boundaries": clip_boundaries or []}, f)
        except Exception:
            pass

    def _load_timeline_frame_disk_cache(
        self,
        uid: str,
        timeline_fps: int = 1,
        max_frames: int = _SEG_TIMELINE_MAX_FRAMES,
    ) -> Optional[list[dict[str, Any]]]:
        cache_dir = self._timeline_frame_cache_dir(timeline_fps=timeline_fps, max_frames=max_frames)
        if cache_dir is None:
            return None
        safe_name = re.sub(r'[^\w\-.]', '_', uid) + ".json"
        cache_file = cache_dir / safe_name
        if not cache_file.exists():
            return None
        try:
            with open(cache_file, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except Exception:
            pass
        return None

    def _save_timeline_frame_disk_cache(
        self,
        uid: str,
        frames: list[dict[str, Any]],
        timeline_fps: int = 1,
        max_frames: int = _SEG_TIMELINE_MAX_FRAMES,
    ) -> None:
        cache_dir = self._timeline_frame_cache_dir(timeline_fps=timeline_fps, max_frames=max_frames)
        if cache_dir is None:
            return
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            safe_name = re.sub(r'[^\w\-.]', '_', uid) + ".json"
            cache_file = cache_dir / safe_name
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(frames, f)
        except Exception:
            pass

    def _diag_paths(self, uid: str) -> dict[str, Any]:
        """Return diagnostic info: which video paths were tried and whether they exist on disk."""
        group = self.groups.get(uid)
        if not group:
            return {"error": "uid not found"}

        raw_paths: list[str] = []
        mm = group.get("multi_modal_source")
        has_base64 = False
        if isinstance(mm, dict):
            if "frames_base64" in mm:
                has_base64 = True
            for v in mm.get("videos") or []:
                if isinstance(v, str):
                    raw_paths.append(v)
            for v in mm.get("images") or []:
                if isinstance(v, str):
                    raw_paths.append(v)
        for vp in group.get("video_paths") or []:
            p = str(vp)
            if p not in raw_paths:
                raw_paths.append(p)
        for ip in group.get("image_paths") or []:
            p = str(ip)
            if p not in raw_paths:
                raw_paths.append(p)

        checked = []
        for p_text in raw_paths:
            path = Path(p_text)
            if not path.is_absolute():
                path = (self.root / path).resolve()
            resolved = str(path)
            exists = path.exists()
            checked.append({
                "path": p_text,
                "resolved": resolved,
                "exists": exists,
                "is_file": path.is_file() if exists else False,
                "suffix": path.suffix.lower(),
            })

        # Check decord availability
        decord_ok = False
        try:
            import decord  # noqa: F401
            decord_ok = True
        except ImportError:
            pass

        return {
            "has_base64": has_base64,
            "paths": checked,
            "decord_available": decord_ok,
        }

    def warm_frame_cache(self, max_frames: int = 30) -> None:
        """Pre-extract and cache frames for all groups. Prints progress."""
        total = len(self.group_order)
        cached = 0
        extracted = 0
        with_frames = 0
        for i, uid in enumerate(self.group_order):
            if uid in self.frame_cache:
                cached += 1
                if self.frame_cache[uid]:
                    with_frames += 1
                continue
            disk_data = self._load_frame_disk_cache(uid)
            if disk_data is not None:
                frames, boundaries = disk_data
                self.frame_cache[uid] = frames
                self.groups[uid]["_clip_boundaries"] = boundaries
                cached += 1
                if frames:
                    with_frames += 1
                continue
            print(f"\r  Extracting frames: {i+1}/{total} ...", end="", flush=True)
            try:
                frames = self._get_frame_strip(uid, max_frames)
                if frames:
                    with_frames += 1
                extracted += 1
            except Exception:
                self.frame_cache[uid] = []
                self._save_frame_disk_cache(uid, [], [])
        if total > 0:
            print(f"\r  Frame cache: {total} groups, {cached} from cache, {extracted} extracted, {with_frames} with frames")

    def warm_timeline_frame_cache(
        self,
        timeline_fps: int = 1,
        max_frames: int = _SEG_TIMELINE_MAX_FRAMES,
    ) -> None:
        """Pre-extract and cache timeline frames for temporal_seg groups."""
        target_uids = [uid for uid in self.group_order if str(self.groups[uid].get("problem_type")) == "temporal_seg"]
        total = len(target_uids)
        if total <= 0:
            return
        cached = 0
        extracted = 0
        with_frames = 0
        fps = max(1, int(timeline_fps))
        frame_cap = max(1, int(max_frames))
        for i, uid in enumerate(target_uids):
            cache_key = f"{uid}@{fps}fps_{frame_cap}f"
            if cache_key in self.timeline_frame_cache:
                cached += 1
                if self.timeline_frame_cache[cache_key]:
                    with_frames += 1
                continue
            disk_frames = self._load_timeline_frame_disk_cache(uid, timeline_fps=fps, max_frames=frame_cap)
            if disk_frames is not None:
                self.timeline_frame_cache[cache_key] = disk_frames
                cached += 1
                if disk_frames:
                    with_frames += 1
                continue
            print(f"\r  Extracting {fps}fps seg timeline frames: {i+1}/{total} ...", end="", flush=True)
            try:
                detail = self.get_group_detail(uid, max_frames=30, timeline_fps=fps)
                frames = detail.get("timeline_frame_strip", []) or []
                if frames:
                    with_frames += 1
                extracted += 1
            except Exception:
                self.timeline_frame_cache[cache_key] = []
                self._save_timeline_frame_disk_cache(uid, [], timeline_fps=fps, max_frames=frame_cap)
        print(
            f"\r  {fps}fps seg timeline cache ({frame_cap} max frames): {total} groups, "
            f"{cached} from cache, {extracted} extracted, {with_frames} with frames"
        )

    def _extract_frames(self, group: dict[str, Any], max_frames: int) -> list[str]:
        frames: list[str] = []
        clip_boundaries: list[int] = []
        candidates: list[Any] = []
        seen_paths: set[str] = set()
        mm = group.get("multi_modal_source")
        if isinstance(mm, dict):
            if "frames_base64" in mm:
                frames_base64 = mm.get("frames_base64")
                if isinstance(frames_base64, list):
                    candidates.extend(frames_base64)
                elif isinstance(frames_base64, str):
                    candidates.append(frames_base64)
            for v in mm.get("videos") or []:
                if isinstance(v, str):
                    seen_paths.add(v)
                candidates.append(v)
            for v in mm.get("images") or []:
                if isinstance(v, str):
                    seen_paths.add(v)
                candidates.append(v)
        for vp in group.get("video_paths") or []:
            if str(vp) not in seen_paths:
                candidates.append(vp)
        for ip in group.get("image_paths") or []:
            if str(ip) not in seen_paths:
                candidates.append(ip)

        # Distribute max_frames evenly across clips
        n_candidates = max(len(candidates), 1)
        per_clip = max(1, max_frames // n_candidates)

        for candidate in candidates:
            if len(frames) >= max_frames:
                break
            clip_boundaries.append(len(frames))
            budget = min(per_clip, max_frames - len(frames))
            new_frames = self._candidate_to_frames(candidate, max_frames=budget)
            frames.extend(new_frames)

        # Remove boundaries for clips that produced 0 frames
        clip_boundaries = [b for b in clip_boundaries if b < len(frames)]
        group["_clip_boundaries"] = clip_boundaries
        return frames[:max_frames]

    def _extract_timeline_frames(
        self,
        group: dict[str, Any],
        timeline_max_t: float,
        timeline_fps: int = 1,
        max_frames: int = _SEG_TIMELINE_MAX_FRAMES,
    ) -> list[dict[str, Any]]:
        fps = max(1, int(timeline_fps))
        frame_cap = max(1, int(max_frames))
        max_second = max(0, int(math.ceil(_safe_float(timeline_max_t, 0.0))))
        candidates: list[Any] = []
        seen_paths: set[str] = set()
        mm = group.get("multi_modal_source")
        if isinstance(mm, dict):
            for v in mm.get("videos") or []:
                if isinstance(v, str):
                    seen_paths.add(v)
                candidates.append(v)
        for vp in group.get("video_paths") or []:
            if str(vp) not in seen_paths:
                candidates.append(vp)

        for candidate in candidates:
            frames = self._candidate_to_timeline_frames(
                candidate,
                max_second=max_second,
                timeline_fps=fps,
                max_frames=frame_cap,
            )
            if frames:
                return frames

        fallback = self._get_frame_strip(str(group.get("uid")), max_frames=min(frame_cap, 200, max_second + 1))
        if not fallback:
            return []
        if len(fallback) == 1:
            return [{"second": 0, "timestamp": _format_mmss(0), "src": fallback[0]}]

        if len(fallback) > frame_cap:
            fallback = _sample_evenly(fallback, frame_cap)
        mapped = []
        for idx, src in enumerate(fallback):
            second = int(round((idx / max(len(fallback) - 1, 1)) * max_second))
            mapped.append({"second": second, "timestamp": _format_mmss(second), "src": src})
        return mapped

    def _candidate_to_timeline_frames(
        self,
        candidate: Any,
        max_second: int,
        timeline_fps: int = 1,
        max_frames: int = _SEG_TIMELINE_MAX_FRAMES,
    ) -> list[dict[str, Any]]:
        if candidate is None:
            return []
        if isinstance(candidate, str):
            value = candidate.strip()
            path = Path(value)
            if not path.is_absolute():
                path = (self.root / path).resolve()
            if not path.exists():
                return []
            if path.suffix.lower() in _VIDEO_EXTS:
                return self._video_file_to_timeline_frames(
                    path,
                    max_second=max_second,
                    timeline_fps=timeline_fps,
                    max_frames=max_frames,
                )
            return []
        return []

    def _video_file_to_timeline_frames(
        self,
        video_path: Path,
        max_second: int,
        timeline_fps: int = 1,
        max_frames: int = _SEG_TIMELINE_MAX_FRAMES,
    ) -> list[dict[str, Any]]:
        try:
            import decord
            decord.bridge.set_bridge("native")
            vr = decord.VideoReader(str(video_path))
            total = len(vr)
            if total <= 0:
                return []
            avg_fps = _safe_float(getattr(vr, "get_avg_fps", lambda: 0.0)(), 0.0)
            if avg_fps <= 0:
                avg_fps = 1.0
            step = 1.0 / float(max(1, timeline_fps))
            ticks = []
            t = 0.0
            limit = max_second + 1e-6
            while t <= limit:
                ticks.append(round(t, 3))
                t += step
            if len(ticks) > max_frames:
                ticks = _sample_evenly(ticks, max_frames)
            frames = []
            for tick in ticks:
                frame_idx = min(total - 1, max(0, int(round(tick * avg_fps))))
                frame_np = vr[frame_idx].asnumpy()
                image = Image.fromarray(frame_np)
                image.thumbnail((240, 140), Image.LANCZOS)
                whole_second = int(round(tick))
                frames.append(
                    {
                        "second": whole_second if timeline_fps == 1 else tick,
                        "timestamp": _format_mmss(whole_second) if timeline_fps == 1 else f"{tick:.1f}s",
                        "src": _image_to_data_url(image),
                    }
                )
            return frames
        except Exception:
            return []

    def _candidate_to_frames(self, candidate: Any, max_frames: int) -> list[str]:
        if max_frames <= 0 or candidate is None:
            return []

        if isinstance(candidate, str):
            value = candidate.strip()
            if value.startswith("data:image/"):
                return [value]
            if _looks_like_base64(value):
                return [f"data:image/jpeg;base64,{value}"]

            path = Path(value)
            if not path.is_absolute():
                path = (self.root / path).resolve()
            if not path.exists():
                return []
            suffix = path.suffix.lower()
            if suffix in _IMAGE_EXTS:
                image_data = _path_to_data_url(path)
                return [image_data] if image_data else []
            if suffix in _VIDEO_EXTS:
                return self._video_file_to_frames(path, max_frames=max_frames)
            return []

        if isinstance(candidate, list):
            if candidate and all(isinstance(x, str) for x in candidate):
                sampled = _sample_evenly(candidate, max(1, max_frames))
                frames = []
                for item in sampled:
                    frames.extend(self._candidate_to_frames(item, max_frames=1))
                    if len(frames) >= max_frames:
                        break
                return frames
            return []

        return []

    def _video_file_to_frames(self, video_path: Path, max_frames: int) -> list[str]:
        # Try decord first (lightweight, more reliable)
        try:
            import decord
            decord.bridge.set_bridge("native")
            vr = decord.VideoReader(str(video_path))
            total = len(vr)
            if total <= 0:
                return []
            n = min(max_frames, total)
            indices = _sample_evenly(list(range(total)), n)
            frames = []
            for idx in indices:
                frame_np = vr[idx].asnumpy()  # HWC uint8
                image = Image.fromarray(frame_np)
                image.thumbnail((320, 180), Image.LANCZOS)
                frames.append(_image_to_data_url(image))
            return frames
        except Exception:
            pass

        # Fallback: qwen_vl_utils
        try:
            from qwen_vl_utils.vision_process import fetch_video
        except Exception:
            return []
        try:
            vision_info = {
                "video": str(video_path),
                "min_pixels": 128 * 128,
                "max_pixels": 320 * 180,
                "max_frames": max_frames,
                "fps": 2.0,
            }
            result = fetch_video(vision_info, image_patch_size=16)
            if isinstance(result, tuple):
                result = result[0]
            if result is None:
                return []
            if hasattr(result, 'numpy'):
                import numpy as np
                arr = result.cpu().numpy() if hasattr(result, 'cpu') else result.numpy()
            elif hasattr(result, 'shape'):
                arr = result
            elif isinstance(result, list):
                frames = []
                sampled = _sample_evenly(result, min(max_frames, len(result)))
                for img in sampled:
                    if hasattr(img, 'save'):
                        frames.append(_image_to_data_url(img))
                return frames
            else:
                return []
            if len(arr.shape) == 4:
                total = int(arr.shape[0])
            elif len(arr.shape) == 3:
                total = 1
                import numpy as np
                arr = np.expand_dims(arr, 0)
            else:
                return []
            if total <= 0:
                return []
            indices = _sample_evenly(list(range(total)), min(max_frames, total))
            frames = []
            for idx in indices:
                frame = arr[idx]
                if hasattr(frame, 'numpy'):
                    frame = frame.numpy()
                import numpy as np
                frame = np.uint8(frame)
                if len(frame.shape) == 3 and frame.shape[0] in (1, 3):
                    frame = frame.transpose(1, 2, 0)
                    if frame.shape[2] == 1:
                        frame = frame.squeeze(2)
                image = Image.fromarray(frame)
                frames.append(_image_to_data_url(image))
            return frames
        except Exception as e:
            print(f"    [warn] frame extraction failed for {video_path.name}: {e}")
            return []


class DashboardHandler(BaseHTTPRequestHandler):
    store: RolloutStore = None
    static_dir: Path = None
    preloaded_html: Optional[bytes] = None

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, status: int, content: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self) -> None:
        if self.path != "/api/load-data":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid json"})
            return
        self._handle_load_data(payload.get("rollout_dir", ""), payload.get("log_file"))

    def _handle_load_data(self, rollout_dir: str, log_file: Optional[str]) -> None:
        if not rollout_dir:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "rollout_dir is required"})
            return
        try:
            summary = self.store.load(rollout_dir_text=rollout_dir, log_file_text=log_file or None)
        except Exception as e:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(e)})
            return
        self._send_json(HTTPStatus.OK, {"ok": True, "summary": summary})

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/state":
            self._send_json(HTTPStatus.OK, {"ok": True, "summary": self.store.summary()})
            return

        if parsed.path == "/api/load-data":
            q = parse_qs(parsed.query)
            rollout_dir = unquote(q.get("rollout_dir", [""])[0])
            log_file = unquote(q.get("log_file", [""])[0]) or None
            self._handle_load_data(rollout_dir, log_file)
            return

        if parsed.path == "/api/steps":
            self._send_json(HTTPStatus.OK, {"ok": True, "steps": self.store.get_steps_summary()})
            return

        if parsed.path == "/api/groups":
            q = parse_qs(parsed.query)
            step = q.get("step", [None])[0]
            step_key = q.get("step_key", [None])[0]
            task = q.get("task", [None])[0]
            query_text = q.get("q", [None])[0]
            limit = int(q.get("limit", ["500"])[0])
            step_int = int(step) if step and step != "all" else None
            rows = self.store.query_groups(
                step=step_int,
                step_key=step_key,
                task=task,
                query=query_text,
                limit=max(1, min(limit, 5000)),
            )
            self._send_json(HTTPStatus.OK, {"ok": True, "groups": rows})
            return

        if parsed.path.startswith("/api/group/"):
            remaining = unquote(parsed.path[len("/api/group/"):])
            q = parse_qs(parsed.query)

            # /api/group/<uid>/frames — frames-only endpoint for lazy loading
            if remaining.endswith("/frames"):
                uid = remaining[: -len("/frames")]
                if not uid:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "uid required"})
                    return
                max_frames = int(q.get("max_frames", ["30"])[0])
                max_frames = max(4, min(max_frames, 200))
                timeline_fps = int(q.get("timeline_fps", ["0"])[0] or 0)
                try:
                    frames = self.store._get_frame_strip(uid, max_frames=max_frames)
                    clip_boundaries = self.store.groups.get(uid, {}).get("_clip_boundaries", [])
                    # Always include diagnostic info so the frontend can explain why frames are empty
                    diag = self.store._diag_paths(uid)
                    payload: dict[str, Any] = {
                        "ok": True,
                        "frames": frames,
                        "clip_boundaries": clip_boundaries,
                        "timeline_frame_strip": [],
                        "timeline_max_t": None,
                        "diag": diag,
                    }
                    if timeline_fps >= 1:
                        detail = self.store.get_group_detail(
                            uid, max_frames=max_frames, timeline_fps=timeline_fps
                        )
                        payload["timeline_frame_strip"] = detail.get("timeline_frame_strip") or []
                        payload["timeline_max_t"] = detail.get("timeline_max_t")
                except KeyError as e:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": str(e)})
                    return
                self._send_json(HTTPStatus.OK, payload)
                return

            # /api/group/<uid> — full group detail
            uid = remaining
            if not uid:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "uid required"})
                return
            max_frames = int(q.get("max_frames", ["30"])[0])
            max_frames = max(4, min(max_frames, 200))
            timeline_fps = int(q.get("timeline_fps", ["0"])[0] or 0)
            try:
                detail = self.store.get_group_detail(uid, max_frames=max_frames, timeline_fps=timeline_fps)
            except KeyError as e:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": str(e)})
                return
            self._send_json(HTTPStatus.OK, {"ok": True, "group": detail})
            return

        self._serve_static(parsed.path)

    def _serve_static(self, path: str) -> None:
        if path == "/":
            file_path = self.static_dir / "index.html"
        else:
            target = path.lstrip("/")
            file_path = (self.static_dir / target).resolve()
            if self.static_dir not in file_path.parents and file_path != self.static_dir:
                self._send_json(HTTPStatus.FORBIDDEN, {"error": "invalid path"})
                return

        if not file_path.exists() or not file_path.is_file():
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "file not found"})
            return

        content = file_path.read_bytes()

        # Inject pre-computed data blob into index.html
        if file_path.name == "index.html" and self.preloaded_html:
            marker = b'</head>'
            if marker in content:
                content = content.replace(marker, self.preloaded_html + marker, 1)
            else:
                content = self.preloaded_html + content

        if file_path.suffix == ".html":
            content_type = "text/html; charset=utf-8"
        elif file_path.suffix == ".js":
            content_type = "application/javascript; charset=utf-8"
        elif file_path.suffix == ".css":
            content_type = "text/css; charset=utf-8"
        else:
            content_type = "application/octet-stream"
        self._send_text(HTTPStatus.OK, content, content_type)

    def log_message(self, format: str, *args: Any) -> None:
        # Keep server output concise
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Rollout dashboard backend server")
    parser.add_argument("--host", default="0.0.0.0", help="bind host")
    parser.add_argument("--port", type=int, default=8765, help="bind port")
    parser.add_argument(
        "--static-dir",
        default="assets/rollout_dashboard",
        help="directory containing index.html",
    )
    parser.add_argument("--rollout-dir", default=None, help="pre-load rollout directory on startup")
    parser.add_argument("--log-file", default=None, help="pre-load experiment log file on startup")
    parser.add_argument(
        "--preload-train-step-interval",
        type=int,
        default=20,
        help="when preloading HTML, keep every N train steps; val steps are all kept",
    )
    args = parser.parse_args()

    root = Path.cwd().resolve()
    static_dir = (root / args.static_dir).resolve()
    if not static_dir.exists():
        raise FileNotFoundError(f"Static directory not found: {static_dir}")

    store = RolloutStore(root=root)
    preloaded_html: Optional[bytes] = None
    if args.rollout_dir:
        try:
            store.load(rollout_dir_text=args.rollout_dir, log_file_text=args.log_file)
            print(f"Pre-loaded rollout data from: {args.rollout_dir}")
            # Frames are now loaded lazily via /api/group/<uid>/frames — skip warm_frame_cache
            # to avoid extracting thousands of frames on every startup.
            # Pre-build the injected HTML blob
            all_steps = store.get_steps_summary()
            selected_step_keys = _select_preload_step_keys(
                all_steps,
                train_step_interval=args.preload_train_step_interval,
            )
            selected_steps = [s for s in all_steps if str(s.get("step_key") or "") in selected_step_keys]
            print(
                "  Preload steps: "
                f"{len(selected_steps)}/{len(all_steps)} kept "
                f"(train every {max(1, int(args.preload_train_step_interval))} steps, val all)"
            )
            all_groups_list = [
                {
                    "uid": g["uid"],
                    "step": g["step"],
                    "phase": g.get("phase"),
                    "step_key": g.get("step_key"),
                    "step_label": g.get("step_label"),
                    "problem_type": g["problem_type"],
                    "mean_reward": g["mean_reward"],
                    "n_rollouts": len(g["attempts"]),
                    "prompt_preview": str(g.get("prompt") or "")[:180],
                }
                for uid, g in ((uid, store.groups[uid]) for uid in store.group_order)
                if str(g.get("step_key") or "") in selected_step_keys
            ]
            all_group_details = {}
            for uid in store.group_order:
                try:
                    group = store.groups.get(uid) or {}
                    if str(group.get("step_key") or "") not in selected_step_keys:
                        continue
                    # text_only=True: skip frame extraction — frames are loaded lazily on demand
                    detail = store.get_group_detail(uid, text_only=True)
                    all_group_details[uid] = _compact_detail_for_preload(detail)
                except Exception:
                    pass
            preload = {
                "summary": store.summary(),
                "steps": selected_steps,
                "all_groups": all_groups_list,
                "all_details": all_group_details,
            }
            preloaded_html = (
                b'<script>window.__PRELOADED__='
                + json.dumps(preload, ensure_ascii=False).encode("utf-8")
                + b';</script>\n'
            )
            print(f"  Ready. Preloaded HTML inject size: {len(preloaded_html) / 1024:.0f} KB")
        except Exception as e:
            print(f"[warn] Failed to pre-load rollout data: {e}")

    DashboardHandler.store = store
    DashboardHandler.static_dir = static_dir
    DashboardHandler.preloaded_html = preloaded_html
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Rollout dashboard server running at http://{args.host}:{args.port}")
    print(f"Static dir: {static_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
