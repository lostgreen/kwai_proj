import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from proxy_data.shared.frame_cache import build_source_cache_dir
from proxy_data.youcook2_seg.temporal_aot.build_aot_from_frames import (
    _build_variant_payload,
    _build_video_fps_override,
    _load_jsonl,
    _normalize_children,
    build_action_records,
    build_event_records,
)


def _make_fake_cache(tmp_path: Path, clip_key: str, source_video_path: str, fps: float, n_frames: int) -> Path:
    cache_dir = build_source_cache_dir(tmp_path / "frames", clip_key, source_video_path, fps)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(1, n_frames + 1):
        (cache_dir / f"{idx:06d}.jpg").write_bytes(b"jpg")
    return cache_dir


def _action_row(source_video_path: str) -> dict:
    return {
        "manifest_type": "action",
        "clip_key": "clip_actions",
        "source_video_path": source_video_path,
        "domain_l1": "task_howto",
        "domain_l2": "food_cooking",
        "event_id": 11,
        "parent_phase_id": 3,
        "event_text": "prepare onions",
        "actions": [
            {"action_id": 101, "text": "grab onion", "start": 0, "end": 2},
            {"action_id": 102, "text": "cut onion", "start": 2, "end": 4},
            {"action_id": 103, "text": "move slices", "start": 4, "end": 6},
        ],
    }


def _event_row(source_video_path: str, end_times: tuple[int, int, int] = (2, 4, 6)) -> dict:
    return {
        "manifest_type": "event",
        "clip_key": "clip_events",
        "source_video_path": source_video_path,
        "domain_l1": "task_howto",
        "domain_l2": "food_cooking",
        "phase_id": 7,
        "phase_text": "prep",
        "events": [
            {"event_id": 201, "text": "wash vegetables", "start": 0, "end": end_times[0]},
            {"event_id": 202, "text": "slice vegetables", "start": end_times[0], "end": end_times[1]},
            {"event_id": 203, "text": "plate vegetables", "start": end_times[1], "end": end_times[2]},
        ],
    }


def _two_action_row(source_video_path: str) -> dict:
    return {
        "manifest_type": "action",
        "clip_key": "clip_actions_2",
        "source_video_path": source_video_path,
        "domain_l1": "task_howto",
        "domain_l2": "food_cooking",
        "event_id": 12,
        "parent_phase_id": 3,
        "event_text": "mix sauce",
        "actions": [
            {"action_id": 111, "text": "add sauce", "start": 0, "end": 2},
            {"action_id": 112, "text": "stir sauce", "start": 2, "end": 4},
        ],
    }


def _two_event_row(source_video_path: str) -> dict:
    return {
        "manifest_type": "event",
        "clip_key": "clip_events_2",
        "source_video_path": source_video_path,
        "domain_l1": "task_howto",
        "domain_l2": "food_cooking",
        "phase_id": 8,
        "phase_text": "finish",
        "events": [
            {"event_id": 211, "text": "serve food", "start": 0, "end": 2},
            {"event_id": 212, "text": "clear table", "start": 2, "end": 4},
        ],
    }


def test_build_variant_payload_uses_child_durations_and_concat_order(tmp_path: Path):
    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"")
    _make_fake_cache(tmp_path, "clip_actions", str(source_video), 2.0, n_frames=12)

    payload = _build_variant_payload(
        row=_action_row(str(source_video)),
        child_key="actions",
        frames_root=tmp_path / "frames",
        cache_fps=2.0,
        target_fps=None,
        rng=__import__("random").Random(0),
        require_shuffled=True,
    )

    assert payload is not None
    assert payload["total_duration_sec"] == 6
    assert payload["variant_child_ids"]["forward"] == [101, 102, 103]
    assert payload["variant_child_ids"]["reversed"] == [103, 102, 101]
    assert payload["variant_child_ids"]["shuffled"] not in ([101, 102, 103], [103, 102, 101])
    assert len(payload["variant_videos"]["forward"]) == 12
    assert payload["variant_videos"]["forward"][0].endswith("000001.jpg")
    assert payload["variant_videos"]["forward"][-1].endswith("000012.jpg")


def test_build_video_fps_override_matches_frame_budget_logic():
    assert _build_video_fps_override(30, num_videos=1) is None
    assert _build_video_fps_override(200, num_videos=2) == 0.64
    assert _build_video_fps_override(2000, num_videos=1) == 0.25


def test_build_action_and_event_records_from_fake_cache(tmp_path: Path):
    action_source = tmp_path / "action_source.mp4"
    event_source = tmp_path / "event_source.mp4"
    action_source.write_bytes(b"")
    event_source.write_bytes(b"")
    _make_fake_cache(tmp_path, "clip_actions", str(action_source), 2.0, n_frames=12)
    _make_fake_cache(tmp_path, "clip_events", str(event_source), 2.0, n_frames=12)
    action_row = _action_row(str(action_source))
    action_row["annotation_meta"] = {"clip_duration_sec": 6.0, "frame_dir": "/frames/action"}
    event_row = _event_row(str(event_source))
    event_row["annotation_meta"] = {"clip_duration_sec": 6.0, "frame_dir": "/frames/event"}

    action_records = build_action_records(
        rows=[action_row],
        frames_root=tmp_path / "frames",
        cache_fps=2.0,
        seed=0,
        build_v2t=True,
        build_t2v=True,
        t2v_max_duration=90,
    )
    event_records = build_event_records(
        rows=[event_row],
        frames_root=tmp_path / "frames",
        cache_fps=2.0,
        seed=0,
        build_v2t=True,
        build_t2v=True,
        t2v_max_duration=60,
    )

    assert len(action_records["action_v2t"]) == 1
    assert len(action_records["action_t2v"]) == 1
    assert len(event_records["event_v2t"]) == 1
    assert len(event_records["event_t2v"]) == 1

    action_v2t = action_records["action_v2t"][0]
    assert action_v2t["problem_type"] == "seg_aot_action_v2t_3way"
    assert len(action_v2t["videos"]) == 1
    assert len(action_v2t["videos"][0]) == 12
    assert action_v2t["metadata"]["query_variant"] == "forward"
    assert action_v2t["metadata"]["ordering_child_ids"]["forward"] == [101, 102, 103]
    assert action_v2t["metadata"]["annotation_meta"]["frame_dir"] == "/frames/action"

    action_t2v = action_records["action_t2v"][0]
    assert action_t2v["problem_type"] == "seg_aot_action_t2v_binary"
    assert len(action_t2v["videos"]) == 2
    assert action_t2v["metadata"]["text_order"] == "forward"
    assert action_t2v["metadata"]["negative_variant"] in {"reversed", "shuffled"}

    event_v2t = event_records["event_v2t"][0]
    assert event_v2t["problem_type"] == "seg_aot_event_v2t_3way"
    assert len(event_v2t["videos"]) == 1
    assert event_v2t["metadata"]["ordering_child_ids"]["forward"] == [201, 202, 203]

    event_t2v = event_records["event_t2v"][0]
    assert event_t2v["problem_type"] == "seg_aot_event_t2v_binary"
    assert len(event_t2v["videos"]) == 2
    assert event_t2v["metadata"]["text_order"] == "forward"
    assert event_t2v["metadata"]["annotation_meta"]["frame_dir"] == "/frames/event"


def test_event_t2v_uses_summed_child_duration_budget(tmp_path: Path):
    source_video = tmp_path / "event_source.mp4"
    source_video.write_bytes(b"")
    _make_fake_cache(tmp_path, "clip_events", str(source_video), 2.0, n_frames=20)
    row = _event_row(str(source_video), end_times=(2, 4, 6))
    row["span_start_sec"] = 0
    row["span_end_sec"] = 20

    records = build_event_records(
        rows=[row],
        frames_root=tmp_path / "frames",
        cache_fps=2.0,
        seed=0,
        build_v2t=False,
        build_t2v=True,
        t2v_max_duration=6,
    )
    assert len(records["event_t2v"]) == 1

    blocked = build_event_records(
        rows=[row],
        frames_root=tmp_path / "frames",
        cache_fps=2.0,
        seed=0,
        build_v2t=False,
        build_t2v=True,
        t2v_max_duration=5,
    )
    assert blocked["event_t2v"] == []


def test_two_child_groups_support_binary_t2v_but_not_three_way(tmp_path: Path):
    action_source = tmp_path / "action_source_2.mp4"
    event_source = tmp_path / "event_source_2.mp4"
    action_source.write_bytes(b"")
    event_source.write_bytes(b"")
    _make_fake_cache(tmp_path, "clip_actions_2", str(action_source), 2.0, n_frames=8)
    _make_fake_cache(tmp_path, "clip_events_2", str(event_source), 2.0, n_frames=8)

    action_records = build_action_records(
        rows=[_two_action_row(str(action_source))],
        frames_root=tmp_path / "frames",
        cache_fps=2.0,
        seed=0,
        build_v2t=True,
        build_t2v=True,
        t2v_max_duration=90,
    )
    event_records = build_event_records(
        rows=[_two_event_row(str(event_source))],
        frames_root=tmp_path / "frames",
        cache_fps=2.0,
        seed=0,
        build_v2t=True,
        build_t2v=True,
        t2v_max_duration=60,
    )

    assert action_records["action_v2t"] == []
    assert len(action_records["action_t2v"]) == 1
    assert action_records["action_t2v"][0]["metadata"]["ordering_child_ids"]["reversed"] == [112, 111]

    assert event_records["event_v2t"] == []
    assert len(event_records["event_t2v"]) == 1
    assert event_records["event_t2v"][0]["metadata"]["ordering_child_ids"]["reversed"] == [212, 211]


def test_multiple_two_child_action_rows_cycle_only_valid_binary_variants(tmp_path: Path):
    rows = []
    for idx in range(4):
        source = tmp_path / f"action_source_2_{idx}.mp4"
        source.write_bytes(b"")
        clip_key = f"clip_actions_2_{idx}"
        _make_fake_cache(tmp_path, clip_key, str(source), 2.0, n_frames=8)
        row = _two_action_row(str(source))
        row["clip_key"] = clip_key
        row["event_id"] = 100 + idx
        rows.append(row)

    action_records = build_action_records(
        rows=rows,
        frames_root=tmp_path / "frames",
        cache_fps=2.0,
        seed=0,
        build_v2t=False,
        build_t2v=True,
        t2v_max_duration=90,
    )

    assert len(action_records["action_t2v"]) == 4
    for record in action_records["action_t2v"]:
        assert record["metadata"]["text_order"] in {"forward", "reversed"}
        assert record["metadata"]["negative_variant"] in {"forward", "reversed"}
        assert record["metadata"]["text_order"] != record["metadata"]["negative_variant"]


def test_load_jsonl_resolves_relative_source_video_path_from_manifest_dir(tmp_path: Path):
    manifest_dir = tmp_path / "nested" / "manifests"
    manifest_dir.mkdir(parents=True)
    source = tmp_path / "videos" / "source.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"")
    _make_fake_cache(tmp_path, "clip_rel", str(source), 2.0, n_frames=8)

    manifest = manifest_dir / "action_manifest.jsonl"
    row = _two_action_row("../../videos/source.mp4")
    row["clip_key"] = "clip_rel"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")

    loaded = _load_jsonl(manifest)
    assert loaded[0]["source_video_path"] == str(source.resolve())

    action_records = build_action_records(
        rows=loaded,
        frames_root=tmp_path / "frames",
        cache_fps=2.0,
        seed=0,
        build_v2t=False,
        build_t2v=True,
        t2v_max_duration=90,
    )
    assert len(action_records["action_t2v"]) == 1


def test_normalize_children_sorts_and_rejects_overlaps():
    row = {
        "actions": [
            {"action_id": 2, "text": "second", "start": 2, "end": 4},
            {"action_id": 1, "text": "first", "start": 0, "end": 2},
        ]
    }
    normalized = _normalize_children(row, "actions")
    assert [child["action_id"] for child in normalized] == [1, 2]

    overlap_row = {
        "actions": [
            {"action_id": 1, "text": "first", "start": 0, "end": 3},
            {"action_id": 2, "text": "second", "start": 2, "end": 4},
        ]
    }
    assert _normalize_children(overlap_row, "actions") is None


def test_child_spans_metadata_uses_normalized_child_order(tmp_path: Path):
    source = tmp_path / "action_unsorted.mp4"
    source.write_bytes(b"")
    _make_fake_cache(tmp_path, "clip_actions_unsorted", str(source), 2.0, n_frames=8)

    row = {
        "manifest_type": "action",
        "clip_key": "clip_actions_unsorted",
        "source_video_path": str(source),
        "domain_l1": "task_howto",
        "domain_l2": "food_cooking",
        "event_id": 51,
        "parent_phase_id": 9,
        "event_text": "assemble",
        "actions": [
            {"action_id": 302, "text": "second step", "start": 2, "end": 4},
            {"action_id": 301, "text": "first step", "start": 0, "end": 2},
        ],
    }

    records = build_action_records(
        rows=[row],
        frames_root=tmp_path / "frames",
        cache_fps=2.0,
        seed=0,
        build_v2t=False,
        build_t2v=True,
        t2v_max_duration=90,
    )

    record = records["action_t2v"][0]
    assert [span["id"] for span in record["metadata"]["child_spans"]] == [301, 302]
    assert record["metadata"]["ordering_child_ids"]["forward"] == [301, 302]


def test_cli_writes_requested_outputs(tmp_path: Path):
    action_source = tmp_path / "action_source.mp4"
    event_source = tmp_path / "event_source.mp4"
    action_source.write_bytes(b"")
    event_source.write_bytes(b"")
    _make_fake_cache(tmp_path, "clip_actions", str(action_source), 2.0, n_frames=12)
    _make_fake_cache(tmp_path, "clip_events", str(event_source), 2.0, n_frames=12)

    action_manifest = tmp_path / "action_manifest.jsonl"
    event_manifest = tmp_path / "event_manifest.jsonl"
    action_manifest.write_text(json.dumps(_action_row(str(action_source))) + "\n", encoding="utf-8")
    event_manifest.write_text(json.dumps(_event_row(str(event_source))) + "\n", encoding="utf-8")

    action_v2t_out = tmp_path / "action_v2t.jsonl"
    event_t2v_out = tmp_path / "event_t2v.jsonl"

    subprocess.run(
        [
            sys.executable,
            "proxy_data/youcook2_seg/temporal_aot/build_aot_from_frames.py",
            "--frames-root",
            str(tmp_path / "frames"),
            "--cache-fps",
            "2.0",
            "--action-manifest",
            str(action_manifest),
            "--event-manifest",
            str(event_manifest),
            "--action-v2t-output",
            str(action_v2t_out),
            "--event-t2v-output",
            str(event_t2v_out),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    action_rows = [json.loads(line) for line in action_v2t_out.read_text(encoding="utf-8").splitlines()]
    event_rows = [json.loads(line) for line in event_t2v_out.read_text(encoding="utf-8").splitlines()]

    assert len(action_rows) == 1
    assert len(event_rows) == 1
    assert action_rows[0]["problem_type"] == "seg_aot_action_v2t_3way"
    assert event_rows[0]["problem_type"] == "seg_aot_event_t2v_binary"
