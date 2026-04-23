import sys
import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


MODULE_PATH = REPO_ROOT / "verl" / "reward_function" / "temporal_grounding_reward.py"
SPEC = importlib.util.spec_from_file_location("temporal_grounding_reward_test_module", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
temporal_grounding_reward = MODULE.temporal_grounding_reward


def test_plain_sentence_format_exact_match():
    gt = "The event happens in the 24.3 - 30.4 seconds."
    pred = "The event happens in the 24.3 - 30.4 seconds."
    result = temporal_grounding_reward(pred, gt, metadata={})
    assert result["overall"] == 1.0
    assert result["accuracy"] == 1.0


def test_plain_sentence_with_query_text_is_supported():
    gt = "The event happens in the 24.3 - 30.4 seconds."
    pred = "The event 'person turn a light on' happens in the 24.3 - 30.4 seconds."
    result = temporal_grounding_reward(pred, gt, metadata={})
    assert result["overall"] == 1.0


def test_plain_sentence_partial_overlap_uses_raw_iou_without_metadata():
    gt = "The event happens in the 10 - 20 seconds."
    pred = "The event happens in the 15 - 25 seconds."
    result = temporal_grounding_reward(pred, gt, metadata={})
    assert abs(result["overall"] - (5.0 / 15.0)) < 1e-6


def test_legacy_answer_tag_is_still_supported():
    gt = "<answer>12.54 to 17.83</answer>"
    pred = "<answer>12.54 to 17.83</answer>"
    result = temporal_grounding_reward(pred, gt, metadata={})
    assert result["overall"] == 1.0


def test_legacy_events_tag_is_still_supported():
    gt = "<events>[[12.5, 17.8]]</events>"
    pred = "<events>[[12.5, 17.8]]</events>"
    result = temporal_grounding_reward(pred, gt, metadata={})
    assert result["overall"] == 1.0
