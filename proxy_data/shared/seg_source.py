#!/usr/bin/env python3
"""
seg_source.py — Unified annotation loading and clip-path resolution.

Shared by all three proxy-data pipelines:
  - hier_seg   (local_scripts/hier_seg_ablations/build_hier_data.py)
  - temporal_aot (proxy_data/temporal_aot/build_aot_from_seg.py)
  - event_logic  (proxy_data/event_logic/build_l2_event_logic.py)

Responsibilities
----------------
1. Constants (single source of truth for window / padding sizes).
2. Raw annotation JSON loading + optional completeness filtering.
3. Geometry helpers: sliding windows, L3 clip-bound computation.
4. Clip-path naming conventions for L1 / L2 / L3.

NOT responsible for
-------------------
- ffmpeg execution  → stays in prepare_clips.py
- Prompt templates  → stays in youcook2_seg_annotation/prompts.py
- Task-specific record building → stays in each pipeline's builder script

Import pattern (add proxy_data/ parent to sys.path first):
    import sys, os
    sys.path.insert(0, "/path/to/proxy_data")
    from shared.seg_source import (
        load_annotations,
        generate_sliding_windows, compute_l3_clip,
        get_l1_clip_path, get_l2_clip_path, get_l3_clip_path,
        L2_WINDOW_SIZE, L2_STRIDE, L3_PADDING, L3_MAX_CLIP_SEC,
    )
"""

from __future__ import annotations

import json
from pathlib import Path

# ── Constants (single definition — all builders import from here) ─────────────
L2_WINDOW_SIZE: int = 128   # seconds
L2_STRIDE: int      = 64    # seconds
L3_PADDING: int     = 5     # seconds added around each L3 event clip
L3_MAX_CLIP_SEC: int = 128  # maximum L3 clip length in seconds


# ── Annotation loading ────────────────────────────────────────────────────────

def load_annotations(
    ann_dir: str | Path,
    complete_only: bool = False,
) -> list[dict]:
    """Load all annotation JSONs from *ann_dir*.

    Args:
        ann_dir:       Directory containing ``*.json`` annotation files.
        complete_only: When True, skip any clip whose L1, L2, or L3 block
                       is missing or has a ``_parse_error`` flag.

    Returns:
        List of raw annotation dicts, sorted by filename (clip_key order).
        Files that cannot be parsed are silently skipped.
    """
    ann_dir = Path(ann_dir)
    annotations: list[dict] = []

    for af in sorted(ann_dir.glob("*.json")):
        try:
            with open(af, encoding="utf-8") as fh:
                ann = json.load(fh)
        except Exception:
            continue

        if complete_only:
            ok_l1 = ann.get("level1") and not ann["level1"].get("_parse_error")
            ok_l2 = ann.get("level2") and not ann["level2"].get("_parse_error")
            ok_l3 = ann.get("level3") and not ann["level3"].get("_parse_error")
            if not (ok_l1 and ok_l2 and ok_l3):
                continue

        annotations.append(ann)

    return annotations


# ── Geometry helpers ──────────────────────────────────────────────────────────

def generate_sliding_windows(
    total_duration: float,
    window_size: int = L2_WINDOW_SIZE,
    stride: int = L2_STRIDE,
) -> list[tuple[int, int]]:
    """Generate non-overlapping (by *stride*) windows over [0, total_duration].

    The last window may be shorter than *window_size* but is kept as long as
    it is at least ``stride // 2`` seconds long.

    Returns:
        List of ``(window_start, window_end)`` pairs in seconds.
    """
    windows: list[tuple[int, int]] = []
    start = 0
    total = int(total_duration)
    while start < total:
        end = min(start + window_size, total)
        if end - start >= stride // 2:
            windows.append((start, end))
        if end >= total:
            break
        start += stride
    return windows


def compute_l3_clip(
    ev_start: int,
    ev_end: int,
    clip_duration: int,
    padding: int = L3_PADDING,
    max_clip: int = L3_MAX_CLIP_SEC,
) -> tuple[int, int, int]:
    """Compute padded clip bounds for an L3 event.

    Adds *padding* seconds on each side of ``[ev_start, ev_end]``, clamped to
    ``[0, clip_duration]``.  If the result would exceed *max_clip* seconds the
    excess is trimmed symmetrically while keeping the event within the clip.

    Returns:
        ``(clip_start, clip_end, clip_duration_sec)``
    """
    clip_start = max(0, ev_start - padding)
    clip_end   = min(clip_duration, ev_end + padding)
    if clip_end - clip_start > max_clip:
        excess     = (clip_end - clip_start) - max_clip
        trim_start = min(excess // 2, ev_start - clip_start)
        clip_start += trim_start
        clip_end    = clip_start + max_clip
    return clip_start, clip_end, clip_end - clip_start


# ── Clip-path naming conventions ─────────────────────────────────────────────

def get_l1_clip_path(
    clip_key: str,
    clip_dir_l1: str | Path,
    fps: int = 1,
) -> str:
    """Return the expected path for an L1 fps-resampled clip.

    Naming convention::

        {clip_dir_l1}/{clip_key}_L1_{fps}fps.mp4

    The file is created by ``prepare_clips.py`` using::

        ffmpeg -i <source_video> -vf fps={fps} -c:v libx264 <output>
    """
    return str(Path(clip_dir_l1) / f"{clip_key}_L1_{fps}fps.mp4")


def get_l2_clip_path(
    clip_key: str,
    ws: int,
    we: int,
    clip_dir_l2: str | Path,
) -> str:
    """Return the expected path for an L2 sliding-window clip.

    Naming convention::

        {clip_dir_l2}/{clip_key}_L2_w{ws}_{we}.mp4
    """
    return str(Path(clip_dir_l2) / f"{clip_key}_L2_w{ws}_{we}.mp4")


def get_l3_clip_path(
    clip_key: str,
    event_id: int,
    clip_start: int,
    clip_end: int,
    clip_dir_l3: str | Path,
) -> str:
    """Return the expected path for an L3 event clip.

    Naming convention::

        {clip_dir_l3}/{clip_key}_L3_ev{event_id}_{clip_start}_{clip_end}.mp4
    """
    return str(
        Path(clip_dir_l3) / f"{clip_key}_L3_ev{event_id}_{clip_start}_{clip_end}.mp4"
    )
