#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt Ablation Comparison Server
==================================
Compare segmentation rollouts from 4 prompt variants (V1–V4) side-by-side.
Reuses frame extraction & segment parsing from rollout_visualization.
"""

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

# ── Constants & regex ──────────────────────────────────────────────

_SEGMENT_PATTERN = re.compile(r"\[\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\]")
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpeg", ".mpg"}
_TIMELINE_MAX_FRAMES = 64


# ── Helpers ────────────────────────────────────────────────────────

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_mmss(total_seconds: Any) -> str:
    total = max(0, int(_safe_float(total_seconds, 0.0)))
    m, s = divmod(total, 60)
    return f"{m:02d}:{s:02d}"


def _extract_segments(text: str) -> list[list[float]]:
    if not text:
        return []
    segs = []
    for s_raw, e_raw in _SEGMENT_PATTERN.findall(str(text)):
        s = _safe_float(s_raw, -1)
        e = _safe_float(e_raw, -1)
        if s < 0 or e <= s:
            continue
        segs.append([s, e])
    return segs


def _sample_evenly(items: list, max_n: int) -> list:
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


def _looks_like_base64(value: str) -> bool:
    if len(value) < 48 or " " in value or "\n" in value:
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9+/=]+", value))


def _phase_rank(phase: str) -> int:
    return 0 if phase == "train" else 1


def _make_step_key(phase: str, step: int) -> str:
    return f"{phase}:{step}"


def _format_step_label(phase: str, step: int) -> str:
    return f"Val {step}" if phase == "val" else f"Step {step}"


# ── Per-experiment rollout store ───────────────────────────────────

class ExperimentStore:
    """Loads and indexes rollout data for a single experiment variant."""

    def __init__(self, name: str, root: Path):
        self.name = name
        self.root = root.resolve()
        self._lock = threading.RLock()
        self.groups: dict[str, dict[str, Any]] = {}
        self.group_order: list[str] = []
        self.task_counts: dict[str, int] = {}
        self.total_samples = 0
        self.step_curve: list[dict[str, float]] = []
        self.training_metrics: list[dict[str, Any]] = []
        self.frame_cache: dict[str, list[str]] = {}
        self.timeline_frame_cache: dict[str, list[dict[str, Any]]] = {}

    def _resolve_local_path(self, path_text: str) -> Path:
        candidate = Path(path_text).expanduser()
        if not candidate.is_absolute():
            candidate = (self.root / candidate).resolve()
        else:
            candidate = candidate.resolve()
        return candidate

    def load(self, rollout_dir_text: str, log_file_text: Optional[str] = None) -> dict[str, Any]:
        rollout_dir = Path(rollout_dir_text).expanduser().resolve()
        if not rollout_dir.exists() or not rollout_dir.is_dir():
            raise FileNotFoundError(f"rollout_dir not found: {rollout_dir}")

        rollout_files = sorted(rollout_dir.glob("step_*.jsonl")) + sorted(
            rollout_dir.glob("val_step_*.jsonl")
        )
        if not rollout_files:
            rollout_files = sorted(rollout_dir.glob("*.jsonl"))
        if not rollout_files:
            raise FileNotFoundError(f"No jsonl files under {rollout_dir}")

        self.rollout_dir = rollout_dir

        with self._lock:
            for fp in rollout_files:
                with open(fp, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        self._consume_record(record)

            if log_file_text:
                log_file = Path(log_file_text).expanduser().resolve()
                if log_file.exists():
                    self.step_curve, self.training_metrics = self._load_log(log_file)
            if not self.step_curve:
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
                "prompt": str(record.get("prompt") or ""),
                "ground_truth": ground_truth,
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

    @staticmethod
    def _load_log(log_file: Path) -> tuple[list[dict], list[dict]]:
        points = []
        metrics = []
        with open(log_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                overall = (row.get("reward") or {}).get("overall")
                if overall is None:
                    continue
                points.append({"step": _safe_float(row.get("step")), "reward": _safe_float(overall)})
                actor = row.get("actor") or {}
                critic = row.get("critic") or {}
                advantages = critic.get("advantages") or {}
                resp_len = row.get("response_length") or {}
                metrics.append({
                    "step": _safe_float(row.get("step")),
                    "kl_loss": _safe_float(actor.get("kl_loss")),
                    "pg_loss": _safe_float(actor.get("pg_loss")),
                    "entropy_loss": _safe_float(actor.get("entropy_loss")),
                    "grad_norm": _safe_float(actor.get("grad_norm")),
                    "advantage_mean": _safe_float(advantages.get("mean")),
                    "response_length_mean": _safe_float(resp_len.get("mean")),
                })
        metrics.sort(key=lambda x: x["step"])
        return sorted(points, key=lambda x: x["step"]), metrics

    def _build_curve_from_groups(self) -> list[dict[str, float]]:
        step_map: dict[str, dict[str, Any]] = {}
        for g in self.groups.values():
            sk = str(g.get("step_key"))
            if sk not in step_map:
                step_map[sk] = {
                    "step": int(g.get("step", 0)),
                    "phase": str(g.get("phase", "train")),
                    "step_label": str(g.get("step_label") or ""),
                    "rewards": [],
                }
            step_map[sk]["rewards"].append(float(g.get("mean_reward", 0)))
        pts = []
        for _, info in step_map.items():
            vals = info["rewards"]
            pts.append({
                "step": float(info["step"]),
                "phase": info["phase"],
                "step_label": info["step_label"],
                "reward": sum(vals) / max(len(vals), 1),
            })
        return sorted(pts, key=lambda x: (x["step"], _phase_rank(str(x.get("phase", "train")))))

    def build_task_curves(self) -> dict[str, list[dict[str, float]]]:
        task_step: dict[str, dict[int, list[float]]] = {}
        for g in self.groups.values():
            task = str(g.get("problem_type") or "unknown")
            step = int(g.get("step", 0))
            task_step.setdefault(task, {}).setdefault(step, []).append(float(g.get("mean_reward", 0)))
        return {
            task: [
                {"step": float(s), "reward": sum(v) / max(len(v), 1)}
                for s, v in sorted(sm.items())
            ]
            for task, sm in task_step.items()
        }

    def get_steps_summary(self) -> list[dict[str, Any]]:
        step_data: dict[str, dict[str, Any]] = {}
        for g in self.groups.values():
            step = int(g.get("step", 0))
            phase = str(g.get("phase", "train"))
            sk = _make_step_key(phase, step)
            if sk not in step_data:
                step_data[sk] = {
                    "step": step,
                    "phase": phase,
                    "step_key": sk,
                    "step_label": _format_step_label(phase, step),
                    "rewards": [],
                    "tasks": {},
                    "n_groups": 0,
                }
            step_data[sk]["rewards"].append(float(g.get("mean_reward", 0)))
            task = str(g.get("problem_type") or "unknown")
            step_data[sk]["tasks"][task] = step_data[sk]["tasks"].get(task, 0) + 1
            step_data[sk]["n_groups"] += 1
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

    def summary(self) -> dict[str, Any]:
        gc = len(self.groups)
        mr = sum(g.get("mean_reward", 0.0) for g in self.groups.values()) / max(gc, 1)
        return {
            "name": self.name,
            "group_count": gc,
            "sample_count": self.total_samples,
            "step_count": len({g.get("step_key") for g in self.groups.values()}),
            "mean_reward": mr,
            "task_counts": self.task_counts,
            "step_curve": self.step_curve,
            "task_curves": self.build_task_curves(),
            "training_metrics": self.training_metrics,
        }

    # ── Frame extraction (reused from rollout visualization) ──────

    def get_frames(self, uid: str, max_frames: int = 30) -> list[str]:
        if uid in self.frame_cache:
            return self.frame_cache[uid]
        group = self.groups.get(uid)
        if not group:
            return []
        frames = self._extract_frames(group, max_frames)
        self.frame_cache[uid] = frames
        return frames

    def get_timeline_frames(
        self,
        uid: str,
        max_t: float,
        timeline_fps: int = 1,
        max_frames: int = _TIMELINE_MAX_FRAMES,
    ) -> list[dict[str, Any]]:
        cache_key = f"{uid}@{timeline_fps}fps_{max_frames}f"
        if cache_key in self.timeline_frame_cache:
            return self.timeline_frame_cache[cache_key]
        group = self.groups.get(uid)
        if not group:
            return []
        frames = self._extract_timeline_frames(group, max_t, timeline_fps, max_frames)
        self.timeline_frame_cache[cache_key] = frames
        return frames

    def _extract_frames(self, group: dict[str, Any], max_frames: int) -> list[str]:
        candidates: list[Any] = []
        seen: set[str] = set()
        mm = group.get("multi_modal_source")
        if isinstance(mm, dict):
            if "frames_base64" in mm:
                fb = mm["frames_base64"]
                if isinstance(fb, list):
                    candidates.extend(fb)
                elif isinstance(fb, str):
                    candidates.append(fb)
            for v in mm.get("videos") or []:
                if isinstance(v, str):
                    seen.add(v)
                candidates.append(v)
            for v in mm.get("images") or []:
                if isinstance(v, str):
                    seen.add(v)
                candidates.append(v)
        for vp in group.get("video_paths") or []:
            if str(vp) not in seen:
                candidates.append(vp)
        for ip in group.get("image_paths") or []:
            if str(ip) not in seen:
                candidates.append(ip)

        nclips = max(len(candidates), 1)
        per_clip = max(1, max_frames // nclips)
        frames: list[str] = []
        for c in candidates:
            if len(frames) >= max_frames:
                break
            budget = min(per_clip, max_frames - len(frames))
            frames.extend(self._candidate_to_frames(c, budget))
        return frames[:max_frames]

    def _extract_timeline_frames(
        self,
        group: dict[str, Any],
        max_t: float,
        fps: int = 1,
        max_frames: int = _TIMELINE_MAX_FRAMES,
    ) -> list[dict[str, Any]]:
        candidates: list[Any] = []
        seen: set[str] = set()
        mm = group.get("multi_modal_source")
        if isinstance(mm, dict):
            for v in mm.get("videos") or []:
                if isinstance(v, str):
                    seen.add(v)
                candidates.append(v)
        for vp in group.get("video_paths") or []:
            if str(vp) not in seen:
                candidates.append(vp)

        max_second = max(0, int(math.ceil(_safe_float(max_t, 0.0))))
        for c in candidates:
            if not isinstance(c, str):
                continue
            path = Path(c.strip())
            if not path.is_absolute():
                path = (self.root / path).resolve()
            if not path.exists() or path.suffix.lower() not in _VIDEO_EXTS:
                continue
            frames = self._video_to_timeline_frames(path, max_second, fps, max_frames)
            if frames:
                return frames

        # fallback: map existing frames to seconds
        fallback = self.get_frames(str(group.get("uid")), max_frames=min(max_frames, max_second + 1))
        if not fallback:
            return []
        if len(fallback) > max_frames:
            fallback = _sample_evenly(fallback, max_frames)
        mapped = []
        for idx, src in enumerate(fallback):
            sec = int(round((idx / max(len(fallback) - 1, 1)) * max_second))
            mapped.append({"second": sec, "timestamp": _format_mmss(sec), "src": src})
        return mapped

    def _video_to_timeline_frames(
        self,
        video_path: Path,
        max_second: int,
        fps: int = 1,
        max_frames: int = _TIMELINE_MAX_FRAMES,
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
            step = 1.0 / float(max(1, fps))
            ticks = []
            t = 0.0
            limit = max_second + 1e-6
            while t <= limit:
                ticks.append(round(t, 3))
                t += step
            if len(ticks) > max_frames:
                ticks = _sample_evenly(ticks, max_frames)
            result = []
            for tick in ticks:
                fi = min(total - 1, max(0, int(round(tick * avg_fps))))
                frame_np = vr[fi].asnumpy()
                img = Image.fromarray(frame_np)
                img.thumbnail((240, 140), Image.LANCZOS)
                sec = int(round(tick))
                result.append({
                    "second": sec if fps == 1 else tick,
                    "timestamp": _format_mmss(sec) if fps == 1 else f"{tick:.1f}s",
                    "src": _image_to_data_url(img),
                })
            return result
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
                try:
                    with Image.open(path) as img:
                        return [_image_to_data_url(img)]
                except Exception:
                    return []
            if suffix in _VIDEO_EXTS:
                return self._video_to_frames(path, max_frames)
            return []
        if isinstance(candidate, list):
            if candidate and all(isinstance(x, str) for x in candidate):
                sampled = _sample_evenly(candidate, max(1, max_frames))
                frames = []
                for item in sampled:
                    frames.extend(self._candidate_to_frames(item, 1))
                    if len(frames) >= max_frames:
                        break
                return frames
        return []

    def _video_to_frames(self, video_path: Path, max_frames: int) -> list[str]:
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
                frame_np = vr[idx].asnumpy()
                img = Image.fromarray(frame_np)
                img.thumbnail((320, 180), Image.LANCZOS)
                frames.append(_image_to_data_url(img))
            return frames
        except Exception:
            return []


# ── Multi-experiment comparison store ─────────────────────────────

class AblationStore:
    """Manages 4 experiment variants and provides cross-experiment comparison."""

    VARIANT_LABELS = {
        "V1": "Minimal",
        "V2": "Granularity-Enhanced",
        "V3": "Structured-CoT",
        "V4": "Gran + CoT",
    }

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.experiments: dict[str, ExperimentStore] = {}
        # Shared frame cache (same video across variants)
        self._shared_frame_cache: dict[str, list[str]] = {}
        self._shared_timeline_cache: dict[str, list[dict[str, Any]]] = {}

    def load_experiment(
        self,
        variant: str,
        rollout_dir: str,
        log_file: Optional[str] = None,
    ) -> dict[str, Any]:
        store = ExperimentStore(variant, self.root)
        summary = store.load(rollout_dir, log_file)
        self.experiments[variant] = store
        print(f"  [{variant}] Loaded {summary['group_count']} groups, "
              f"{summary['sample_count']} samples, avg reward={summary['mean_reward']:.4f}")
        return summary

    def get_variants(self) -> list[str]:
        return sorted(self.experiments.keys())

    def get_overview(self) -> dict[str, Any]:
        variants = {}
        for v, exp in self.experiments.items():
            variants[v] = exp.summary()
            variants[v]["label"] = self.VARIANT_LABELS.get(v, v)
        return {"variants": variants}

    def get_common_steps(self) -> list[dict[str, Any]]:
        """Return steps that exist in ALL experiments."""
        if not self.experiments:
            return []
        all_step_keys = None
        step_info_map: dict[str, dict[str, Any]] = {}
        for v, exp in self.experiments.items():
            steps = exp.get_steps_summary()
            keys = set()
            for s in steps:
                sk = s["step_key"]
                keys.add(sk)
                if sk not in step_info_map:
                    step_info_map[sk] = {
                        "step": s["step"],
                        "phase": s["phase"],
                        "step_key": sk,
                        "step_label": s["step_label"],
                        "per_variant": {},
                    }
                step_info_map[sk]["per_variant"][v] = {
                    "mean_reward": s["mean_reward"],
                    "n_groups": s["n_groups"],
                    "task_counts": s["task_counts"],
                }
            if all_step_keys is None:
                all_step_keys = keys
            else:
                all_step_keys &= keys

        if not all_step_keys:
            # Fall back to union if no common steps
            all_step_keys = set(step_info_map.keys())

        result = []
        for sk in all_step_keys:
            info = step_info_map[sk]
            result.append(info)
        return sorted(result, key=lambda x: (x["step"], _phase_rank(x["phase"])))

    def get_aligned_samples(
        self,
        step_key: str,
        task_filter: Optional[str] = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Find UIDs that appear in all variants at the given step, with comparison data."""
        uid_data: dict[str, dict[str, Any]] = {}

        for v, exp in self.experiments.items():
            for uid, g in exp.groups.items():
                if str(g.get("step_key")) != step_key:
                    continue
                if task_filter and task_filter != "all":
                    if str(g.get("problem_type")) != task_filter:
                        continue
                if uid not in uid_data:
                    uid_data[uid] = {
                        "uid": uid,
                        "step": g["step"],
                        "phase": g.get("phase"),
                        "problem_type": g.get("problem_type"),
                        "ground_truth": g.get("ground_truth", ""),
                        "variants": {},
                    }
                uid_data[uid]["variants"][v] = {
                    "mean_reward": g["mean_reward"],
                    "n_rollouts": len(g["attempts"]),
                    "prompt_preview": str(g.get("prompt") or "")[:200],
                }

        # Sort by reward variance (most interesting first = biggest gap between variants)
        result = []
        for uid, data in uid_data.items():
            rewards = [vd["mean_reward"] for vd in data["variants"].values()]
            if len(rewards) >= 2:
                data["reward_spread"] = max(rewards) - min(rewards)
                data["max_reward"] = max(rewards)
                data["min_reward"] = min(rewards)
            else:
                data["reward_spread"] = 0.0
                data["max_reward"] = rewards[0] if rewards else 0.0
                data["min_reward"] = rewards[0] if rewards else 0.0
            result.append(data)

        result.sort(key=lambda x: -x["reward_spread"])
        return result[:limit]

    def get_sample_comparison(self, uid: str) -> dict[str, Any]:
        """Return full comparison data for a single UID across all variants."""
        comparison: dict[str, Any] = {
            "uid": uid,
            "variants": {},
            "shared_frames": [],
            "shared_timeline_frames": [],
        }

        max_t = 0.0
        gt_segs = []

        for v, exp in self.experiments.items():
            g = exp.groups.get(uid)
            if g is None:
                continue
            attempts = sorted(g["attempts"], key=lambda a: a["reward"], reverse=True)
            for a in attempts:
                ts = a.get("temporal_segments") or {}
                for part in ("predicted", "ground_truth"):
                    for seg in ts.get(part, []) or []:
                        if isinstance(seg, list) and len(seg) == 2:
                            max_t = max(max_t, _safe_float(seg[1], 0.0))
                if not gt_segs:
                    gt_segs = ts.get("ground_truth", [])

            comparison["variants"][v] = {
                "name": v,
                "label": self.VARIANT_LABELS.get(v, v),
                "mean_reward": g["mean_reward"],
                "prompt": g.get("prompt", ""),
                "ground_truth": g.get("ground_truth", ""),
                "problem_type": g.get("problem_type", ""),
                "attempts": [
                    {
                        "reward": a["reward"],
                        "response": a["response"],
                        "temporal_segments": a.get("temporal_segments") or {},
                    }
                    for a in attempts
                ],
            }

        comparison["timeline_max_t"] = max_t
        comparison["ground_truth_segments"] = gt_segs

        # Shared frames: use the first available experiment's video
        frames = self._get_shared_frames(uid, max_frames=30)
        comparison["shared_frames"] = frames

        if max_t > 0:
            tl_frames = self._get_shared_timeline_frames(uid, max_t)
            comparison["shared_timeline_frames"] = tl_frames

        return comparison

    def _get_shared_frames(self, uid: str, max_frames: int = 30) -> list[str]:
        if uid in self._shared_frame_cache:
            return self._shared_frame_cache[uid]
        for exp in self.experiments.values():
            frames = exp.get_frames(uid, max_frames)
            if frames:
                self._shared_frame_cache[uid] = frames
                return frames
        return []

    def _get_shared_timeline_frames(
        self,
        uid: str,
        max_t: float,
        fps: int = 1,
        max_frames: int = _TIMELINE_MAX_FRAMES,
    ) -> list[dict[str, Any]]:
        cache_key = f"{uid}@{fps}fps_{max_frames}f"
        if cache_key in self._shared_timeline_cache:
            return self._shared_timeline_cache[cache_key]
        for exp in self.experiments.values():
            frames = exp.get_timeline_frames(uid, max_t, fps, max_frames)
            if frames:
                self._shared_timeline_cache[cache_key] = frames
                return frames
        return []

    def get_reward_curves(self) -> dict[str, Any]:
        """Return reward curves for all variants, overall and per-task."""
        curves = {}
        for v, exp in self.experiments.items():
            curves[v] = {
                "overall": exp.step_curve,
                "per_task": exp.build_task_curves(),
                "training_metrics": exp.training_metrics,
            }
        return curves


# ── HTTP handler ──────────────────────────────────────────────────

class ComparisonHandler(BaseHTTPRequestHandler):
    ablation_store: AblationStore = None
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

    def do_GET(self) -> None:
        try:
            self._handle_get()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            try:
                self._send_json(500, {"ok": False, "error": str(exc)})
            except Exception:
                pass

    def _handle_get(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        q = parse_qs(parsed.query)

        print(f"[REQ] {self.command} {self.path}")

        # API endpoints
        if path == "/api/overview":
            self._send_json(HTTPStatus.OK, {
                "ok": True,
                **self.ablation_store.get_overview(),
            })
            return

        if path == "/api/steps":
            self._send_json(HTTPStatus.OK, {
                "ok": True,
                "steps": self.ablation_store.get_common_steps(),
            })
            return

        if path == "/api/samples":
            step_key = q.get("step_key", [None])[0]
            task = q.get("task", [None])[0]
            limit = int(q.get("limit", ["200"])[0])
            if not step_key:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "step_key required"})
                return
            try:
                samples = self.ablation_store.get_aligned_samples(step_key, task, min(limit, 500))
                self._send_json(HTTPStatus.OK, {"ok": True, "samples": samples})
            except Exception as e:
                print(f"[ERROR] /api/samples: {e}")
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(e)})
            return

        if path.startswith("/api/sample/"):
            uid = unquote(path[len("/api/sample/"):])
            if not uid:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "uid required"})
                return
            try:
                comparison = self.ablation_store.get_sample_comparison(uid)
                self._send_json(HTTPStatus.OK, {"ok": True, "comparison": comparison})
            except Exception as e:
                print(f"[ERROR] /api/sample/{uid}: {e}")
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(e)})
            return

        if path == "/api/reward_curves":
            curves = self.ablation_store.get_reward_curves()
            self._send_json(HTTPStatus.OK, {"ok": True, "curves": curves})
            return

        if path.startswith("/api/frames/"):
            uid = unquote(path[len("/api/frames/"):])
            if not uid:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "uid required"})
                return
            max_frames = int(q.get("max_frames", ["30"])[0])
            timeline_fps = int(q.get("timeline_fps", ["1"])[0])
            frames = self.ablation_store._get_shared_frames(uid, min(max_frames, 200))
            tl_frames = []
            max_t = 0.0
            # Compute max_t from any variant
            for exp in self.ablation_store.experiments.values():
                g = exp.groups.get(uid)
                if g:
                    for a in g["attempts"]:
                        ts = a.get("temporal_segments") or {}
                        for part in ("predicted", "ground_truth"):
                            for seg in ts.get(part, []) or []:
                                if isinstance(seg, list) and len(seg) == 2:
                                    max_t = max(max_t, _safe_float(seg[1], 0.0))
                    break
            if max_t > 0 and timeline_fps >= 1:
                tl_frames = self.ablation_store._get_shared_timeline_frames(uid, max_t, timeline_fps)
            self._send_json(HTTPStatus.OK, {
                "ok": True,
                "frames": frames,
                "timeline_frames": tl_frames,
                "timeline_max_t": max_t,
            })
            return

        # Static files
        self._serve_static(path)

    def _serve_static(self, path: str) -> None:
        if path == "/":
            file_path = self.static_dir / "index.html"
        else:
            target = path.lstrip("/")
            file_path = (self.static_dir / target).resolve()
            if self.static_dir not in file_path.parents and file_path != self.static_dir:
                self._send_json(HTTPStatus.FORBIDDEN, {"error": "invalid path"})
                return

        print(f"[STATIC] {path!r} -> {file_path} (exists={file_path.exists()})")
        if not file_path.exists() or not file_path.is_file():
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "file not found"})
            return

        content = file_path.read_bytes()

        if file_path.name == "index.html" and self.preloaded_html:
            marker = b"</head>"
            if marker in content:
                content = content.replace(marker, self.preloaded_html + marker, 1)
            else:
                content = self.preloaded_html + content

        ext_map = {".html": "text/html", ".js": "application/javascript", ".css": "text/css"}
        ct = ext_map.get(file_path.suffix, "application/octet-stream") + "; charset=utf-8"
        self._send_text(HTTPStatus.OK, content, ct)

    def log_message(self, fmt: str, *args: Any) -> None:
        # Log errors but not every request
        if args and len(args) >= 2:
            status = str(args[1]) if len(args) > 1 else ""
            if status.startswith("4") or status.startswith("5"):
                print(f"[HTTP] {args[0]} → {status}")
        return


# ── Entry point ───────────────────────────────────────────────────

def _parse_kv_arg(text: str) -> dict[str, str]:
    """Parse 'V1=/path,V2=/path' into dict."""
    result = {}
    for part in text.split(","):
        part = part.strip()
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        result[k.strip()] = v.strip()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt Ablation Comparison Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8891)
    parser.add_argument("--static-dir", default="prompt_ablation_visualization")
    parser.add_argument(
        "--exp-dirs",
        required=True,
        help="Comma-separated variant=path pairs, e.g. V1=/path/to/rollouts,V2=/path/to/rollouts",
    )
    parser.add_argument(
        "--log-files",
        default="",
        help="Comma-separated variant=path pairs for log files (optional)",
    )
    args = parser.parse_args()

    root = Path.cwd().resolve()
    static_dir = (root / args.static_dir).resolve()
    if not static_dir.exists():
        raise FileNotFoundError(f"Static directory not found: {static_dir}")

    exp_dirs = _parse_kv_arg(args.exp_dirs)
    log_files = _parse_kv_arg(args.log_files) if args.log_files else {}

    if not exp_dirs:
        raise ValueError("At least one experiment directory required in --exp-dirs")

    store = AblationStore(root=root)
    print(f"Loading {len(exp_dirs)} experiments...")
    for variant, rollout_path in sorted(exp_dirs.items()):
        log_path = log_files.get(variant)
        try:
            store.load_experiment(variant, rollout_path, log_path)
        except Exception as e:
            print(f"  [WARN] Failed to load {variant}: {e}")

    if not store.experiments:
        raise RuntimeError("No experiments loaded successfully")

    # Build preloaded data
    preload = {
        "overview": store.get_overview(),
        "steps": store.get_common_steps(),
        "reward_curves": store.get_reward_curves(),
    }
    preloaded_html = (
        b"<script>window.__PRELOADED__="
        + json.dumps(preload, ensure_ascii=False).encode("utf-8")
        + b";</script>\n"
    )
    print(f"Preloaded HTML inject size: {len(preloaded_html) / 1024:.0f} KB")

    ComparisonHandler.ablation_store = store
    ComparisonHandler.static_dir = static_dir
    ComparisonHandler.preloaded_html = preloaded_html

    print(f"Static dir: {static_dir}")
    print(f"index.html exists: {(static_dir / 'index.html').exists()}")

    server = ThreadingHTTPServer((args.host, args.port), ComparisonHandler)
    print(f"\nPrompt Ablation Comparison server running at http://{args.host}:{args.port}")
    print(f"Variants: {', '.join(store.get_variants())}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
