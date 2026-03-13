#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import base64
import json
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
        self.frame_cache: dict[str, list[str]] = {}
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

        rollout_files = sorted(rollout_dir.glob("step_*.jsonl"))
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
                    -self.groups[uid].get("mean_reward", 0.0),
                ),
            )
            return self.summary()

    def _consume_record(self, record: dict[str, Any]) -> None:
        step = int(record.get("step", 0))
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
        with open(log_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                points.append(
                    {
                        "step": _safe_float(row.get("step")),
                        "reward": _safe_float((row.get("reward") or {}).get("overall")),
                    }
                )
        return sorted(points, key=lambda x: x["step"])

    def _build_curve_from_groups(self) -> list[dict[str, float]]:
        step_map: dict[int, list[float]] = {}
        for group in self.groups.values():
            step_map.setdefault(int(group["step"]), []).append(float(group["mean_reward"]))
        points = []
        for step, vals in step_map.items():
            points.append({"step": float(step), "reward": sum(vals) / max(len(vals), 1)})
        return sorted(points, key=lambda x: x["step"])

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
        step_data: dict[int, dict[str, Any]] = {}
        for g in self.groups.values():
            step = int(g.get("step", 0))
            if step not in step_data:
                step_data[step] = {"step": step, "rewards": [], "tasks": {}, "n_groups": 0}
            step_data[step]["rewards"].append(float(g.get("mean_reward", 0)))
            task = str(g.get("problem_type") or "unknown")
            step_data[step]["tasks"][task] = step_data[step]["tasks"].get(task, 0) + 1
            step_data[step]["n_groups"] += 1
        result = []
        for s in sorted(step_data):
            d = step_data[s]
            result.append({
                "step": s,
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
            "step_count": len({g.get("step") for g in self.groups.values()}),
            "mean_group_reward": mean_reward,
            "temporal_ratio": temporal_count / max(group_count, 1),
            "task_counts": self.task_counts,
            "step_curve": self.step_curve,
            "task_curves": self._build_task_curves(),
            "rollout_dir": str(self.rollout_dir) if self.rollout_dir else None,
            "log_file": str(self.log_file) if self.log_file else None,
        }

    def query_groups(self, step: Optional[int], task: Optional[str], query: Optional[str], limit: int) -> list[dict[str, Any]]:
        query_lc = (query or "").strip().lower()
        rows = []
        for uid in self.group_order:
            g = self.groups[uid]
            if step is not None and int(g.get("step", -1)) != step:
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
                    "problem_type": g["problem_type"],
                    "mean_reward": g["mean_reward"],
                    "n_rollouts": len(g["attempts"]),
                    "prompt_preview": str(g.get("prompt") or "")[:180],
                }
            )
            if len(rows) >= limit:
                break
        return rows

    def get_group_detail(self, uid: str, max_frames: int = 30) -> dict[str, Any]:
        group = self.groups.get(uid)
        if group is None:
            raise KeyError(f"uid not found: {uid}")
        detail = dict(group)
        detail["attempts"] = sorted(group["attempts"], key=lambda a: a["reward"], reverse=True)
        detail["frame_strip"] = self._get_frame_strip(uid, max_frames=max_frames)
        detail["timeline_max_t"] = self._timeline_max_t(detail)
        task = str(detail.get("problem_type") or "")
        if task in ("add", "delete", "replace"):
            detail["choice_meta"] = self._build_choice_meta(detail)
        elif task == "sort":
            detail["sort_meta"] = self._build_sort_meta(detail)
        return detail

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
        group = self.groups[uid]
        frames = self._extract_frames(group, max_frames=max_frames)
        self.frame_cache[uid] = frames
        return frames

    def _extract_frames(self, group: dict[str, Any], max_frames: int) -> list[str]:
        frames: list[str] = []
        candidates: list[Any] = []
        mm = group.get("multi_modal_source")
        if isinstance(mm, dict):
            # Online deployment preferred schema: multi_modal_source.frames_base64
            if "frames_base64" in mm:
                frames_base64 = mm.get("frames_base64")
                if isinstance(frames_base64, list):
                    candidates.extend(frames_base64)
                elif isinstance(frames_base64, str):
                    candidates.append(frames_base64)
            if "videos" in mm:
                candidates.extend(mm.get("videos") or [])
            if "images" in mm:
                candidates.extend(mm.get("images") or [])
        candidates.extend(group.get("video_paths") or [])
        candidates.extend(group.get("image_paths") or [])

        for candidate in candidates:
            if len(frames) >= max_frames:
                break
            frames.extend(self._candidate_to_frames(candidate, max_frames=max_frames - len(frames)))

        return frames[:max_frames]

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
        try:
            from qwen_vl_utils.vision_process import fetch_video
        except Exception:
            return []

        try:
            vision_info = {
                "video": str(video_path),
                "min_pixels": 4 * 32 * 32,
                "max_pixels": 48 * 32 * 32,
                "max_frames": max_frames,
                "fps": 2.0,
            }
            result = fetch_video(vision_info, image_patch_size=16)
            if isinstance(result, tuple):
                result = result[0]
            if result is None:
                return []
            if hasattr(result, "shape"):
                total = int(result.shape[0]) if len(result.shape) >= 4 else 0
                if total <= 0:
                    return []
                indices = _sample_evenly(list(range(total)), min(max_frames, total))
                frames = []
                for idx in indices:
                    frame = result[idx]
                    image = Image.fromarray(frame.astype("uint8"))
                    frames.append(_image_to_data_url(image))
                return frames
        except Exception:
            return []
        return []


class DashboardHandler(BaseHTTPRequestHandler):
    store: RolloutStore = None
    static_dir: Path = None

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
            task = q.get("task", [None])[0]
            query_text = q.get("q", [None])[0]
            limit = int(q.get("limit", ["500"])[0])
            step_int = int(step) if step and step != "all" else None
            rows = self.store.query_groups(step=step_int, task=task, query=query_text, limit=max(1, min(limit, 5000)))
            self._send_json(HTTPStatus.OK, {"ok": True, "groups": rows})
            return

        if parsed.path.startswith("/api/group/"):
            uid = unquote(parsed.path[len("/api/group/") :])
            if not uid:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "uid required"})
                return
            q = parse_qs(parsed.query)
            max_frames = int(q.get("max_frames", ["30"])[0])
            max_frames = max(4, min(max_frames, 200))
            try:
                detail = self.store.get_group_detail(uid, max_frames=max_frames)
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
    args = parser.parse_args()

    root = Path.cwd().resolve()
    static_dir = (root / args.static_dir).resolve()
    if not static_dir.exists():
        raise FileNotFoundError(f"Static directory not found: {static_dir}")

    DashboardHandler.store = RolloutStore(root=root)
    DashboardHandler.static_dir = static_dir
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
