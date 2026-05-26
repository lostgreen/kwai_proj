from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPT_SPEC = importlib.util.spec_from_file_location(
    "boundary_prompts_under_test",
    REPO_ROOT
    / "video_proxy"
    / "data"
    / "pipelines"
    / "proxy_construction"
    / "event_boundary"
    / "boundary_prompts.py",
)
assert PROMPT_SPEC is not None and PROMPT_SPEC.loader is not None
boundary_prompts = importlib.util.module_from_spec(PROMPT_SPEC)
PROMPT_SPEC.loader.exec_module(boundary_prompts)


def test_l3_v2_preserves_shot_first_priority():
    prompt = boundary_prompts.PROMPT_VARIANTS_V4["L3"]["V2"].format(duration=44)

    assert "SHOT-FIRST" in prompt
    assert "Do not merge visually distinct shots into one segment" in prompt
    assert "A clear camera cut, angle change, framing change, subject change" in prompt
    assert "long single shot may produce multiple L3 segments" in prompt
    assert "Do NOT rely on single-frame flicker" in prompt
    assert "Do NOT rely on single-frame micro-motions" in prompt
    assert "Do NOT rely on single-frame micro-motions, instantaneous contact changes, or camera cuts" not in prompt


def test_l3_v1_kept_for_old_prompt_comparison():
    assert "V1" in boundary_prompts.PROMPT_VARIANTS_V4["L3"]
    assert "V2" in boundary_prompts.PROMPT_VARIANTS_V4["L3"]
    assert boundary_prompts.PROMPT_VARIANTS_V4["L3"]["V1"] != boundary_prompts.PROMPT_VARIANTS_V4["L3"]["V2"]
