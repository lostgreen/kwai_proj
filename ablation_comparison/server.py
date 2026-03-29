#!/usr/bin/env python3
"""Ablation Comparison Server — compare temporal segmentation across settings.

Usage:
    python server.py \\
        --setting PA1:/path/to/pa1/rollout_or_data \\
        --setting PA2:/path/to/pa2/rollout_or_data \\
        --setting R1:/path/to/r1/rollout_or_data \\
        --setting R2:/path/to/r2/rollout_or_data \\
        --port 8790

Each --setting takes NAME:DIR where DIR contains JSONL files (step_*.jsonl or train.jsonl).
Samples are matched across settings by video filename.
"""

import argparse
import base64
import json
import math
import os
import re
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

from PIL import Image

_SEGMENT_RE = re.compile(r"\[\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\]")


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _extract_segments(text: str) -> list[list[float]]:
    if not text:
        return []
    out = []
    for s, e in _SEGMENT_RE.findall(str(text)):
        s_f, e_f = _safe_float(s, -1), _safe_float(e, -1)
        if s_f >= 0 and e_f > s_f:
            out.append([s_f, e_f])
    return out


def _video_key(paths: list[str]) -> str:
    """Extract a stable key from video paths (just the filename, no directory)."""
    for p in paths:
        name = Path(p).name
        if name:
            return name
    return ""


def _image_to_data_url(image: Image.Image, max_w: int = 180) -> str:
    image = image.convert("RGB")
    if image.width > max_w:
        r = max_w / float(image.width)
        image = image.resize((max_w, max(1, int(image.height * r))))
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=80)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"


def _extract_frames_from_video(video_path: str, fps: float = 1.0, max_frames: int = 64) -> list[dict]:
    """Try to extract frames using decord. Returns list of {second, src}."""
    try:
        import decord
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(video_path)
        total_frames = len(vr)
        video_fps = float(vr.get_avg_fps())
        duration = total_frames / video_fps
        step = max(1, int(video_fps / fps))
        indices = list(range(0, total_frames, step))[:max_frames]
        frames_out = []
        for idx in indices:
            frame = vr[idx].asnumpy()
            pil = Image.fromarray(frame)
            sec = round(idx / video_fps)
            frames_out.append({"second": sec, "src": _image_to_data_url(pil)})
        return frames_out
    except Exception:
        return []


# ─── Data Store ────────────────────────────────────────────────

class ComparisonStore:
    """Holds loaded data from multiple ablation settings."""

    def __init__(self):
        self._lock = threading.RLock()
        self.settings: dict[str, dict] = {}   # name -> {dir, records, steps}
        self.video_index: dict[str, dict] = {}  # video_key -> {gt, duration, video_path, settings: {name: [records]}}
        self.video_keys: list[str] = []
        self._frame_cache: dict[str, list[dict]] = {}

    def load_setting(self, name: str, dir_path: str) -> int:
        d = Path(dir_path).resolve()
        if not d.exists():
            raise FileNotFoundError(f"Not found: {d}")

        files = sorted(d.glob("step_*.jsonl")) + sorted(d.glob("val_step_*.jsonl"))
        if not files:
            files = sorted(d.glob("*.jsonl"))
        if not files:
            raise FileNotFoundError(f"No JSONL files in {d}")

        records = []
        for f in files:
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    records.append(rec)

        with self._lock:
            self.settings[name] = {"dir": str(d), "records": records, "count": len(records)}
            self._rebuild_index()
        return len(records)

    def _rebuild_index(self) -> None:
        self.video_index.clear()
        for setting_name, info in self.settings.items():
            for rec in info["records"]:
                vpaths = rec.get("video_paths") or rec.get("videos") or []
                if isinstance(vpaths, str):
                    vpaths = [vpaths]
                # Also check multi_modal_source
                mms = rec.get("multi_modal_source") or {}
                if not vpaths:
                    vpaths = mms.get("videos") or []

                vk = _video_key(vpaths)
                if not vk:
                    # fallback: use uid
                    vk = str(rec.get("uid") or rec.get("problem_id") or id(rec))

                if vk not in self.video_index:
                    # Determine GT & duration from any available record
                    gt_text = str(rec.get("ground_truth") or rec.get("answer") or "")
                    ts = rec.get("temporal_segments")
                    gt_segs = (ts or {}).get("ground_truth", []) if isinstance(ts, dict) else []
                    if not gt_segs:
                        gt_segs = _extract_segments(gt_text)

                    dur = _safe_float(rec.get("duration") or (rec.get("metadata") or {}).get("duration"), 0)
                    if dur <= 0 and gt_segs:
                        dur = max(e for _, e in gt_segs) * 1.1

                    self.video_index[vk] = {
                        "video_key": vk,
                        "gt_segments": gt_segs,
                        "duration": dur,
                        "video_path": vpaths[0] if vpaths else "",
                        "prompt": {},
                        "settings": {},
                    }

                entry = self.video_index[vk]
                # Update GT from any setting that has it
                if not entry["gt_segments"]:
                    gt_text = str(rec.get("ground_truth") or rec.get("answer") or "")
                    ts = rec.get("temporal_segments")
                    gt_segs = (ts or {}).get("ground_truth", []) if isinstance(ts, dict) else []
                    if not gt_segs:
                        gt_segs = _extract_segments(gt_text)
                    if gt_segs:
                        entry["gt_segments"] = gt_segs

                # Parse predicted segments
                response = str(rec.get("response") or "")
                ts = rec.get("temporal_segments")
                pred_segs = (ts or {}).get("predicted", []) if isinstance(ts, dict) else []
                if not pred_segs:
                    pred_segs = _extract_segments(response)

                reward = _safe_float(rec.get("reward"), 0.0)
                step = int(rec.get("step", 0))
                phase = str(rec.get("phase") or "train")
                prompt = str(rec.get("prompt") or "")
                problem_type = str(rec.get("problem_type") or "")
                level = str((rec.get("metadata") or {}).get("level", ""))

                record_info = {
                    "predicted": pred_segs,
                    "reward": reward,
                    "step": step,
                    "phase": phase,
                    "response": response,
                    "prompt": prompt,
                    "problem_type": problem_type,
                    "level": level,
                }

                entry["settings"].setdefault(setting_name, []).append(record_info)
                # Store prompt per setting (use first seen)
                if setting_name not in entry["prompt"]:
                    entry["prompt"][setting_name] = prompt

                # Update duration
                if entry["duration"] <= 0:
                    dur = _safe_float(rec.get("duration") or (rec.get("metadata") or {}).get("duration"), 0)
                    if dur > 0:
                        entry["duration"] = dur

        # Sort by video key
        self.video_keys = sorted(self.video_index.keys())

    def get_frames(self, video_key: str, fps: float = 1.0) -> list[dict]:
        if video_key in self._frame_cache:
            return self._frame_cache[video_key]
        entry = self.video_index.get(video_key)
        if not entry:
            return []
        vpath = entry.get("video_path", "")
        if not vpath or not Path(vpath).exists():
            return []
        frames = _extract_frames_from_video(vpath, fps=fps)
        if frames:
            self._frame_cache[video_key] = frames
        return frames

    def summary(self) -> dict:
        return {
            "settings": {
                name: {"dir": info["dir"], "count": info["count"]}
                for name, info in self.settings.items()
            },
            "total_videos": len(self.video_keys),
            "setting_names": list(self.settings.keys()),
        }

    def list_samples(self, offset: int = 0, limit: int = 50, level_filter: str = "") -> dict:
        keys = self.video_keys
        if level_filter:
            filtered = []
            for vk in keys:
                entry = self.video_index[vk]
                for recs in entry["settings"].values():
                    if any(r.get("level") == level_filter for r in recs):
                        filtered.append(vk)
                        break
            keys = filtered

        total = len(keys)
        page = keys[offset:offset + limit]
        items = []
        for vk in page:
            entry = self.video_index[vk]
            n_settings = len(entry["settings"])
            n_gt = len(entry["gt_segments"])
            duration = entry["duration"]
            levels = set()
            for recs in entry["settings"].values():
                for r in recs:
                    if r.get("level"):
                        levels.add(r["level"])
            items.append({
                "video_key": vk,
                "n_settings": n_settings,
                "n_gt_segments": n_gt,
                "duration": duration,
                "levels": sorted(levels),
            })
        return {"total": total, "offset": offset, "items": items}

    def get_sample(self, video_key: str, step: Optional[int] = None) -> Optional[dict]:
        entry = self.video_index.get(video_key)
        if not entry:
            return None

        result = {
            "video_key": video_key,
            "gt_segments": entry["gt_segments"],
            "duration": entry["duration"],
            "video_path": entry["video_path"],
            "prompts": entry["prompt"],
            "settings": {},
        }

        for setting_name, records in entry["settings"].items():
            # Filter by step if specified; otherwise use the best (highest reward) or latest
            if step is not None:
                candidates = [r for r in records if r["step"] == step]
            else:
                candidates = records

            if not candidates:
                continue

            # Group by step, pick latest or best
            best = max(candidates, key=lambda r: (r["reward"], r["step"]))
            result["settings"][setting_name] = {
                "predicted": best["predicted"],
                "reward": best["reward"],
                "step": best["step"],
                "phase": best["phase"],
                "response": best["response"],
                "prompt": best["prompt"],
                "level": best["level"],
                "all_records": [
                    {
                        "predicted": r["predicted"],
                        "reward": r["reward"],
                        "step": r["step"],
                        "level": r["level"],
                    }
                    for r in records
                ],
            }

        return result


# ─── HTTP Handler ──────────────────────────────────────────────

store = ComparisonStore()
_static_dir = Path(__file__).resolve().parent


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # quiet

    def _json(self, data: Any, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _html(self, path: Path):
        if not path.resolve().is_relative_to(_static_dir):
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        ct = "text/html; charset=utf-8"
        if path.suffix == ".css":
            ct = "text/css; charset=utf-8"
        elif path.suffix == ".js":
            ct = "application/javascript; charset=utf-8"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        qs = parse_qs(parsed.query)

        if path in ("", "/", "/index.html"):
            self._html(_static_dir / "index.html")
        elif path == "/api/summary":
            self._json(store.summary())
        elif path == "/api/samples":
            offset = int(qs.get("offset", ["0"])[0])
            limit = int(qs.get("limit", ["50"])[0])
            level = qs.get("level", [""])[0]
            self._json(store.list_samples(offset, limit, level))
        elif path.startswith("/api/sample/"):
            vk = path[len("/api/sample/"):]
            step_str = qs.get("step", [None])[0]
            step = int(step_str) if step_str else None
            data = store.get_sample(vk, step)
            if data:
                self._json(data)
            else:
                self._json({"error": "not found"}, 404)
        elif path.startswith("/api/frames/"):
            vk = path[len("/api/frames/"):]
            fps = _safe_float(qs.get("fps", ["1"])[0], 1.0)
            frames = store.get_frames(vk, fps)
            self._json({"video_key": vk, "frames": frames})
        else:
            self.send_error(HTTPStatus.NOT_FOUND)


def main():
    parser = argparse.ArgumentParser(description="Ablation comparison server")
    parser.add_argument(
        "--setting", action="append", required=True,
        help="NAME:DIR — setting name and directory of JSONL data. Can be repeated.",
    )
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    for s in args.setting:
        if ":" not in s:
            parser.error(f"Invalid --setting format '{s}', expected NAME:DIR")
        name, dir_path = s.split(":", 1)
        count = store.load_setting(name.strip(), dir_path.strip())
        print(f"  [{name}] loaded {count} records from {dir_path}")

    summary = store.summary()
    print(f"\n  Total videos: {summary['total_videos']}")
    print(f"  Settings:     {summary['setting_names']}")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"\n  → http://localhost:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
