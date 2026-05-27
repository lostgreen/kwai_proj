from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parents[1] / "video_proxy" / "data" / "scripts" / "convert_jsonl_to_cot.py"
_SPEC = importlib.util.spec_from_file_location("convert_jsonl_to_cot_l3_under_test", _MODULE_PATH)
convert_jsonl_to_cot = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = convert_jsonl_to_cot
_SPEC.loader.exec_module(convert_jsonl_to_cot)


def test_l3_events_cot_instruction_keeps_shot_boundaries_primary():
    prompt = (
        "Watch the following video clip carefully:\n"
        "<video>\n\n"
        "Detect all fine-grained L3 sub-actions using a SHOT-FIRST policy:\n\n"
        "STEP 1 - FIND SHOT / SCENE BOUNDARIES:\n"
        "Do not merge visually distinct shots into one segment unless they are static.\n\n"
        "Output the start and end time (integer seconds, 0-based) for each segment in chronological order:\n"
        "<events>[[start_time, end_time], ...]</events>\n\n"
        "Example: <events>[[2, 6], [9, 13], [15, 20]]</events>"
    )
    record = {
        "prompt": prompt,
        "messages": [{"role": "user", "content": prompt}],
        "answer": "<events>[[2, 6], [9, 13], [15, 20]]</events>",
        "problem_type": "temporal_seg_hier_L3_seg",
    }

    converted, changed, reason = convert_jsonl_to_cot.convert_record(record, "thought")

    assert changed is True
    assert reason == "events"
    assert "For L3, preserve visually distinct shot boundaries first" in converted["prompt"]
    assert "long single shots may need additional state/action splits" in converted["prompt"]
    assert "Do not collapse multiple clear shots into one broad segment" in converted["prompt"]
    assert "<events>" not in converted["prompt"]
    assert "<answer>[[start_time, end_time], ...]</answer>" in converted["prompt"]
