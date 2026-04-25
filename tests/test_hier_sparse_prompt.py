from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPT_SPEC = importlib.util.spec_from_file_location(
    "prompt_variants_v4_under_test",
    REPO_ROOT / "local_scripts" / "hier_seg_ablations" / "prompt_ablation" / "prompt_variants_v4.py",
)
assert PROMPT_SPEC is not None and PROMPT_SPEC.loader is not None
PROMPT_MODULE = importlib.util.module_from_spec(PROMPT_SPEC)
PROMPT_SPEC.loader.exec_module(PROMPT_MODULE)

PATCH_SPEC = importlib.util.spec_from_file_location(
    "patch_hier_sparse_prompt_under_test",
    REPO_ROOT / "local_scripts" / "data" / "patch_hier_sparse_prompt.py",
)
assert PATCH_SPEC is not None and PATCH_SPEC.loader is not None
PATCH_MODULE = importlib.util.module_from_spec(PATCH_SPEC)
PATCH_SPEC.loader.exec_module(PATCH_MODULE)


def test_hier_prompt_templates_do_not_claim_fixed_one_to_two_fps():
    for level in ("L1", "L2", "L3"):
        prompt = PROMPT_MODULE.PROMPT_VARIANTS_V4[level]["V1"].format(duration=54)
        assert "sampled at 1-2 fps" not in prompt
        assert "sparsely sampled frames" in prompt
        assert "displayed timestamps" in prompt


def test_existing_jsonl_prompt_text_can_be_patched_without_regeneration():
    text = (
        "You are given a 54s video clip, sampled at 1-2 fps.\n\n"
        "IMPORTANT — SPARSE SAMPLING:\n"
        "This clip is sampled at 1-2 fps (not continuous video). "
        "Do NOT rely on single-frame micro-motions, instantaneous contact changes, "
        "or camera cuts to place boundaries. "
        "Create a boundary ONLY when the change is sustained across multiple sampled frames "
        "or when the task/state clearly shifts."
    )

    patched, changes = PATCH_MODULE.patch_text(text)

    assert changes == 2
    assert "sampled at 1-2 fps" not in patched
    assert "represented by sparsely sampled frames" in patched
    assert "SPARSE VISUAL EVIDENCE" in patched
