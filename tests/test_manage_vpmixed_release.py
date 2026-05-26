from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "manage_vpmixed_release.py"
_SPEC = importlib.util.spec_from_file_location("manage_vpmixed_release_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
manage_release = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = manage_release
_SPEC.loader.exec_module(manage_release)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _row(old: Path, problem_type: str, clip: str) -> dict:
    old_frame = str(old / "hier_seg_annotation_v1/frame_cache/source_2fps" / clip / "000001.jpg")
    return {
        "problem_type": problem_type,
        "videos": [[old_frame]],
        "metadata": {
            "experiment_frame_sampling": {
                "trusted_cache_roots": [str(old / "hier_seg_annotation_v1/frame_cache/source_2fps")]
            }
        },
    }


def test_build_task_jsonl_splits_proxy_train_contract_from_full_mix(tmp_path: Path):
    rel = tmp_path / "rel"
    old = tmp_path / "old" / "VideoProxyMixed"
    rel.mkdir(parents=True)

    full_mix = old / "multi_task/experiments/composition_base_seg_logic_aot_hier10k_el10k_aot10k_mf256_ema"
    _write_jsonl(
        full_mix / "train.jsonl",
        [
            _row(old, "temporal_grounding", "tg"),
            _row(old, "llava_mcq", "mcq"),
            _row(old, "temporal_seg_hier_L3_seg", "seg_a"),
            _row(old, "temporal_seg_hier_L3_seg", "seg_b"),
            _row(old, "seg_aot_x", "aot"),
            _row(old, "event_logic_x", "logic"),
        ],
    )
    _write_jsonl(
        full_mix / "val.jsonl",
        [
            _row(old, "temporal_grounding", "tg_val"),
            _row(old, "llava_mcq", "mcq_val"),
            _row(old, "temporal_seg_hier_L1", "seg_val"),
            _row(old, "seg_aot_x", "aot_val"),
            _row(old, "event_logic_x", "logic_val"),
        ],
    )

    report = manage_release.build_task_jsonl(rel, old)

    seg_train = [json.loads(line) for line in (rel / "tasks/seg/train.jsonl").read_text().splitlines()]
    logic_train = [json.loads(line) for line in (rel / "tasks/logic/train.jsonl").read_text().splitlines()]
    assert report["seg/train"]["rows"] == 2
    assert len(seg_train) == 2
    assert len(logic_train) == 1
    assert "/old/VideoProxyMixed/" not in json.dumps(seg_train)
    assert str(rel / "frames/seg/source_2fps") in json.dumps(seg_train)


def test_frame_path_report_does_not_stat_every_frame_path(tmp_path: Path):
    rel = tmp_path / "rel"
    rows = [{"videos": [[str(rel / "frames/seg/source_2fps/clip/000001.jpg")]]}]

    report = manage_release.frame_path_report(rows, rel)

    assert report == {
        "frame_paths": 1,
        "release_local": 1,
        "old_paths": 0,
        "bad_sample": [],
    }
