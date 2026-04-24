import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from proxy_data.youcook2_seg.temporal_aot.build_aot_group_manifest import (
    MANIFEST_ACTION,
    MANIFEST_EVENT,
    MANIFEST_EVENT_DIR,
    build_group_manifests,
    collect_action_group_manifests,
    collect_event_group_manifests,
    collect_event_reverse_group_manifests,
)


def _base_annotation() -> dict:
    return {
        "clip_key": "clip_001",
        "source_video_path": "",
        "domain_l1": "task_howto",
        "domain_l2": "food_cooking",
        "level1": {
            "macro_phases": [
                {
                    "phase_id": 1,
                    "phase_name": "prep ingredients",
                    "start_time": 0,
                    "end_time": 20,
                    "_order_distinguishable": True,
                },
                {
                    "phase_id": 2,
                    "phase_name": "cook",
                    "start_time": 20,
                    "end_time": 35,
                    "_order_distinguishable": False,
                },
            ],
        },
        "level2": {
            "events": [
                {
                    "event_id": 101,
                    "parent_phase_id": 1,
                    "instruction": "slice onions",
                    "start_time": 0,
                    "end_time": 8,
                    "_order_distinguishable": True,
                },
                {
                    "event_id": 102,
                    "parent_phase_id": 1,
                    "instruction": "mix seasoning",
                    "start_time": 8,
                    "end_time": 20,
                    "_order_distinguishable": True,
                },
                {
                    "event_id": 201,
                    "parent_phase_id": 2,
                    "instruction": "saute onions",
                    "start_time": 20,
                    "end_time": 35,
                    "_order_distinguishable": False,
                },
            ],
        },
        "level3": {
            "grounding_results": [
                {
                    "action_id": 1001,
                    "parent_event_id": 101,
                    "parent_phase_id": 1,
                    "sub_action": "grab knife",
                    "start_time": 0,
                    "end_time": 2,
                },
                {
                    "action_id": 1002,
                    "parent_event_id": 101,
                    "parent_phase_id": 1,
                    "sub_action": "halve onion",
                    "start_time": 2,
                    "end_time": 5,
                },
                {
                    "action_id": 1003,
                    "parent_event_id": 101,
                    "parent_phase_id": 1,
                    "sub_action": "slice strips",
                    "start_time": 5,
                    "end_time": 8,
                },
                {
                    "action_id": 1004,
                    "parent_event_id": 102,
                    "parent_phase_id": 1,
                    "sub_action": "add salt",
                    "start_time": 8,
                    "end_time": 12,
                },
                {
                    "action_id": 1005,
                    "parent_event_id": 102,
                    "parent_phase_id": 1,
                    "sub_action": "add pepper",
                    "start_time": 12,
                    "end_time": 16,
                },
                {
                    "action_id": 1006,
                    "parent_event_id": 102,
                    "parent_phase_id": 1,
                    "sub_action": "stir bowl",
                    "start_time": 16,
                    "end_time": 20,
                },
                {
                    "action_id": 2001,
                    "parent_event_id": 201,
                    "parent_phase_id": 2,
                    "sub_action": "add oil",
                    "start_time": 20,
                    "end_time": 24,
                },
                {
                    "action_id": 2002,
                    "parent_event_id": 201,
                    "parent_phase_id": 2,
                    "sub_action": "add onion",
                    "start_time": 24,
                    "end_time": 29,
                },
                {
                    "action_id": 2003,
                    "parent_event_id": 201,
                    "parent_phase_id": 2,
                    "sub_action": "stir pan",
                    "start_time": 29,
                    "end_time": 35,
                },
            ],
        },
    }


def _nested_l3_annotation() -> dict:
    ann = _base_annotation()
    ann.update(
        {
            "source_video_path": "videos/source.mp4",
            "video_path": "videos/fallback.mp4",
            "source_mode": "windowed_clip",
            "annotation_start_sec": 0.0,
            "annotation_end_sec": 35.0,
            "window_start_sec": 0.0,
            "window_end_sec": 35.0,
            "clip_duration_sec": 35.0,
            "n_frames": 70,
            "frame_dir": "frames/clip_001",
            "summary": "Cooking onions.",
            "video_caption": "A person prepares and cooks onions.",
            "archetype": "tutorial",
            "topology_type": "procedural",
        }
    )
    ann["level3"] = {
        "grounding_results": [
            {
                "event_id": 101,
                "parent_phase_id": 1,
                "event_start": 0,
                "event_end": 8,
                "event_instruction": "slice onions",
                "sub_actions": [
                    {
                        "action_id": 1,
                        "start_time": 0,
                        "end_time": 2,
                        "sub_action": "grab onion",
                        "caption": "A hand reaches for an onion.",
                    },
                    {
                        "action_id": 2,
                        "start_time": 2,
                        "end_time": 5,
                        "sub_action": "halve onion",
                        "caption": "The onion is cut in half.",
                    },
                    {
                        "action_id": 3,
                        "start_time": 5,
                        "end_time": 8,
                        "sub_action": "slice strips",
                        "caption": "The onion is sliced into strips.",
                    },
                ],
            },
            {
                "event_id": 102,
                "parent_phase_id": 1,
                "event_start": 8,
                "event_end": 20,
                "event_instruction": "mix seasoning",
                "sub_actions": [
                    {
                        "action_id": 4,
                        "start_time": 8,
                        "end_time": 12,
                        "sub_action": "add salt",
                    },
                    {
                        "action_id": 5,
                        "start_time": 12,
                        "end_time": 16,
                        "sub_action": "add pepper",
                    },
                    {
                        "action_id": 6,
                        "start_time": 16,
                        "end_time": 20,
                        "sub_action": "stir bowl",
                    },
                ],
            },
        ]
    }
    return ann


def test_collect_action_group_manifests_keeps_expected_fields():
    rows = collect_action_group_manifests([_base_annotation()], min_actions=3, max_actions=3)

    assert len(rows) == 3
    first = rows[0]
    assert first["manifest_type"] == MANIFEST_ACTION
    assert first["clip_key"] == "clip_001"
    assert first["source_video_path"] == ""
    assert first["event_id"] == 101
    assert first["parent_phase_id"] == 1
    assert first["span_start_sec"] == 0
    assert first["span_end_sec"] == 8
    assert [child["action_id"] for child in first["actions"]] == [1001, 1002, 1003]
    assert first["actions"][0]["text"] == "grab knife"
    assert first["actions"][0]["parent_event_id"] == 101
    assert first["actions"][0]["start"] == 0
    assert first["actions"][0]["end"] == 2


def test_collect_action_group_manifests_supports_nested_l3_sub_actions(tmp_path: Path):
    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir()
    ann = _nested_l3_annotation()
    ann["_annotation_path"] = str((ann_dir / "clip_001.json").resolve())
    ann["_annotation_dir"] = str(ann_dir.resolve())

    rows = collect_action_group_manifests([ann], min_actions=3, max_actions=3)

    assert len(rows) == 2
    first = rows[0]
    assert first["source_video_path"] == str((ann_dir / "videos/source.mp4").resolve())
    assert first["annotation_meta"]["source_mode"] == "windowed_clip"
    assert first["annotation_meta"]["clip_duration_sec"] == 35.0
    assert first["annotation_meta"]["frame_dir"] == "frames/clip_001"
    assert first["annotation_meta"]["video_caption"] == "A person prepares and cooks onions."
    assert first["annotation_meta"]["raw_source_video_path"] == "videos/source.mp4"
    assert [child["parent_event_id"] for child in first["actions"]] == [101, 101, 101]
    assert first["actions"][0]["caption"] == "A hand reaches for an onion."
    assert first["actions"][0]["parent_event_start_sec"] == 0
    assert first["actions"][0]["parent_event_end_sec"] == 8
    assert first["actions"][0]["parent_event_instruction"] == "slice onions"


def test_collect_action_group_manifests_applies_action_filters():
    rows = collect_action_group_manifests(
        [_base_annotation()],
        min_actions=3,
        max_actions=3,
        filter_order=True,
        min_duration=9,
    )

    assert len(rows) == 1
    assert rows[0]["event_id"] == 102


def test_collect_event_group_manifests_applies_filters():
    ann = _base_annotation()
    ann["level2"]["events"][1]["_complete"] = False

    filtered = collect_event_group_manifests(
        [ann],
        min_events=2,
        max_events=3,
        complete_only=True,
    )
    assert filtered == []

    ordered_only = collect_event_group_manifests(
        [_base_annotation()],
        min_events=2,
        max_events=3,
        filter_order=True,
    )
    assert len(ordered_only) == 1
    assert ordered_only[0]["phase_id"] == 1
    assert [event["event_id"] for event in ordered_only[0]["events"]] == [101, 102]


def test_collect_event_reverse_group_manifests_applies_duration_and_order_filters():
    rows = collect_event_reverse_group_manifests(
        [_base_annotation()],
        filter_order=True,
        min_duration=9,
    )

    assert len(rows) == 1
    assert rows[0]["manifest_type"] == MANIFEST_EVENT_DIR
    assert "event_id" not in rows[0]
    assert "event_text" not in rows[0]
    assert rows[0]["events"][0]["text"] == "mix seasoning"
    assert rows[0]["events"][0]["event_id"] == 102


def test_source_video_path_can_be_resolved_against_explicit_source_root(tmp_path: Path):
    ann = _nested_l3_annotation()
    source_root = tmp_path / "source_root"

    rows = collect_event_reverse_group_manifests([ann], source_root=source_root)

    assert rows[0]["source_video_path"] == str((source_root / "videos/source.mp4").resolve())
    assert rows[0]["annotation_meta"]["raw_source_video_path"] == "videos/source.mp4"


def test_collectors_skip_parse_error_annotations():
    action_ann = _base_annotation()
    action_ann["level2"]["_parse_error"] = True
    assert collect_action_group_manifests([action_ann]) == []

    action_ann = _base_annotation()
    action_ann["level3"]["_parse_error"] = True
    assert collect_action_group_manifests([action_ann]) == []

    event_ann = _base_annotation()
    event_ann["level1"]["_parse_error"] = True
    assert collect_event_group_manifests([event_ann]) == []

    event_ann = _base_annotation()
    event_ann["level2"]["_parse_error"] = True
    assert collect_event_group_manifests([event_ann]) == []

    event_dir_ann = _base_annotation()
    event_dir_ann["level2"]["_parse_error"] = True
    assert collect_event_reverse_group_manifests([event_dir_ann]) == []


def test_build_group_manifests_returns_requested_subset():
    manifests = build_group_manifests(
        annotations=[_base_annotation()],
        include_action=True,
        include_event=False,
        include_event_dir=True,
        min_actions=3,
        max_actions=3,
    )

    assert set(manifests.keys()) == {MANIFEST_ACTION, MANIFEST_EVENT_DIR}
    assert len(manifests[MANIFEST_ACTION]) == 3
    assert len(manifests[MANIFEST_EVENT_DIR]) == 3


def test_cli_resolves_nested_annotation_source_and_meta(tmp_path: Path):
    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir()
    (ann_dir / "clip_001.json").write_text(json.dumps(_nested_l3_annotation()), encoding="utf-8")
    action_out = tmp_path / "group_manifest_action.jsonl"

    subprocess.run(
        [
            sys.executable,
            "proxy_data/youcook2_seg/temporal_aot/build_aot_group_manifest.py",
            "--annotation-dir",
            str(ann_dir),
            "--action-output",
            str(action_out),
            "--min-actions",
            "3",
            "--max-actions",
            "3",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    action_rows = [json.loads(line) for line in action_out.read_text(encoding="utf-8").splitlines()]

    assert len(action_rows) == 2
    assert action_rows[0]["source_video_path"] == str((ann_dir / "videos/source.mp4").resolve())
    assert action_rows[0]["annotation_meta"]["annotation_path"] == str((ann_dir / "clip_001.json").resolve())
    assert action_rows[0]["annotation_meta"]["raw_source_video_path"] == "videos/source.mp4"
    assert action_rows[0]["actions"][0]["caption"] == "A hand reaches for an onion."


def test_cli_writes_requested_outputs(tmp_path: Path):
    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir()
    (ann_dir / "clip_001.json").write_text(json.dumps(_base_annotation()), encoding="utf-8")

    action_out = tmp_path / "group_manifest_action.jsonl"
    event_dir_out = tmp_path / "group_manifest_event_dir.jsonl"

    subprocess.run(
        [
            sys.executable,
            "proxy_data/youcook2_seg/temporal_aot/build_aot_group_manifest.py",
            "--annotation-dir",
            str(ann_dir),
            "--action-output",
            str(action_out),
            "--event-dir-output",
            str(event_dir_out),
            "--min-actions",
            "3",
            "--max-actions",
            "3",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    action_rows = [json.loads(line) for line in action_out.read_text(encoding="utf-8").splitlines()]
    event_dir_rows = [json.loads(line) for line in event_dir_out.read_text(encoding="utf-8").splitlines()]

    assert len(action_rows) == 3
    assert len(event_dir_rows) == 3
    assert all(row["manifest_type"] == MANIFEST_ACTION for row in action_rows)
    assert all(row["manifest_type"] == MANIFEST_EVENT_DIR for row in event_dir_rows)
