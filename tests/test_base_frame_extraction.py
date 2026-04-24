from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from local_scripts.data.rewrite_videos_to_frames import compute_shared_fps


def test_uncapped_extraction_keeps_target_fps_for_long_videos():
    fps = compute_shared_fps(
        duration_sec=240.0,
        n_videos=1,
        target_fps=2.0,
        fallback_fps=1.0,
        max_frames=256,
        min_fps=0.25,
        respect_max_frames=False,
    )

    assert fps == 2.0


def test_capped_extraction_preserves_legacy_max_frame_budget_behavior():
    fps = compute_shared_fps(
        duration_sec=240.0,
        n_videos=1,
        target_fps=2.0,
        fallback_fps=1.0,
        max_frames=256,
        min_fps=0.25,
        respect_max_frames=True,
    )

    assert fps == 1.0
