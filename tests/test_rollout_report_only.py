from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OFFLINE_FILTER = REPO_ROOT / "video_proxy" / "training" / "tools" / "offline_rollout_filter.py"
NEXTVIDEO_PIPELINE = (
    REPO_ROOT / "video_proxy" / "data" / "base_sources" / "mcq" / "nextvideo" / "run_pipeline.sh"
)


def _argparse_call_for_flag(tree: ast.AST, flag: str) -> ast.Call:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        if any(isinstance(arg, ast.Constant) and arg.value == flag for arg in node.args):
            return node
    raise AssertionError(f"Missing argparse flag {flag}")


def test_offline_rollout_filter_allows_report_only_output():
    tree = ast.parse(OFFLINE_FILTER.read_text(encoding="utf-8"))
    call = _argparse_call_for_flag(tree, "--output_jsonl")

    required_values = [
        kw.value.value
        for kw in call.keywords
        if kw.arg == "required" and isinstance(kw.value, ast.Constant)
    ]
    assert required_values != [True]


def test_nextvideo_pipeline_defaults_to_rollout_report_only_layout():
    text = NEXTVIDEO_PIPELINE.read_text(encoding="utf-8")

    assert "ROLLOUT_ROOT=\"${ROLLOUT_ROOT:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/rollouts}\"" in text
    assert "WRITE_KEPT_JSONL=\"${WRITE_KEPT_JSONL:-false}\"" in text
    assert "ENABLE_GPU_FILLER=\"${ENABLE_GPU_FILLER:-false}\"" in text
    rollout_common = text.split("ROLLOUT_COMMON=(", maxsplit=1)[1].split("\n    )", maxsplit=1)[0]
    assert "--output_jsonl" not in rollout_common
    assert "--report_jsonl" not in rollout_common
    assert "ROLLOUT_REPORT_ARGS=(--report_jsonl \"${ROLLOUT_REPORT}\")" in text
    assert "SHARD_OUTPUT_ARGS=()" in text
    assert "SHARD_OUTPUT_ARGS=(--output_jsonl \"${OUTPUT_ROOT}/_shard${i}_kept.jsonl\")" in text
