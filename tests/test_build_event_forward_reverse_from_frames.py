import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from proxy_data.shared.frame_cache import build_source_cache_dir, select_frame_paths_for_span
from proxy_data.youcook2_seg.temporal_aot.build_event_forward_reverse_from_frames import (
    _build_video_fps_override,
    _load_jsonl,
    build_records,
)


def _make_fake_cache(tmp_path: Path, clip_key: str, source_video_path: str, fps: float, n_frames: int) -> Path:
    cache_dir = build_source_cache_dir(tmp_path / "frames", clip_key, source_video_path, fps)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(1, n_frames + 1):
        (cache_dir / f"{idx:06d}.jpg").write_bytes(b"jpg")
    return cache_dir


def _event_dir_row(
    clip_key: str,
    source_video_path: str,
    start: int,
    end: int,
    event_id: int,
    phase_id: int = 3,
    event_text: str = "stir the soup",
) -> dict:
    return {
        "manifest_type": "event_dir",
        "clip_key": clip_key,
        "source_video_path": source_video_path,
        "domain_l1": "task_howto",
        "domain_l2": "food_cooking",
        "span_start_sec": start,
        "span_end_sec": end,
        "phase_id": phase_id,
        "events": [
            {
                "event_id": event_id,
                "parent_phase_id": phase_id,
                "text": event_text,
                "start": start,
                "end": end,
            }
        ],
    }


def test_true_reverse_frame_list_construction(tmp_path: Path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"")
    cache_dir = _make_fake_cache(tmp_path, "clip_a", str(source), 2.0, n_frames=12)
    row = _event_dir_row("clip_a", str(source), start=1, end=4, event_id=101)
    row["annotation_meta"] = {"clip_duration_sec": 3.0, "frame_dir": "/frames/event_dir"}

    records = build_records(
        rows=[row],
        frames_root=tmp_path / "frames",
        cache_fps=2.0,
        sample_mode="two_per_event",
    )

    assert len(records) == 2
    forward, reverse = records
    expected_forward = [str(path.resolve()) for path in sorted(cache_dir.glob("*.jpg"))][2:8]
    assert forward["metadata"]["query_variant"] == "forward"
    assert reverse["metadata"]["query_variant"] == "reverse"
    assert forward["answer"] == "A"
    assert reverse["answer"] == "B"
    assert len(forward["videos"]) == 1
    assert len(reverse["videos"]) == 1
    assert forward["videos"][0] == expected_forward
    assert reverse["videos"][0] == list(reversed(forward["videos"][0]))
    assert forward["metadata"]["annotation_meta"]["frame_dir"] == "/frames/event_dir"


def test_one_per_event_alternates_query_variants(tmp_path: Path):
    rows = []
    for idx in range(3):
        source = tmp_path / f"source_{idx}.mp4"
        source.write_bytes(b"")
        clip_key = f"clip_{idx}"
        _make_fake_cache(tmp_path, clip_key, str(source), 2.0, n_frames=6)
        rows.append(_event_dir_row(clip_key, str(source), start=0, end=3, event_id=200 + idx))

    records = build_records(
        rows=rows,
        frames_root=tmp_path / "frames",
        cache_fps=2.0,
        sample_mode="one_per_event",
    )

    variants = [record["metadata"]["query_variant"] for record in records]
    assert variants.count("forward") == 2
    assert variants.count("reverse") == 1
    assert [record["answer"] for record in records] == [
        "A" if record["metadata"]["query_variant"] == "forward" else "B" for record in records
    ]


def test_two_per_event_doubles_rows(tmp_path: Path):
    rows = []
    for idx in range(2):
        source = tmp_path / f"pair_source_{idx}.mp4"
        source.write_bytes(b"")
        clip_key = f"pair_clip_{idx}"
        _make_fake_cache(tmp_path, clip_key, str(source), 2.0, n_frames=8)
        rows.append(_event_dir_row(clip_key, str(source), start=0, end=4, event_id=300 + idx))

    records = build_records(
        rows=rows,
        frames_root=tmp_path / "frames",
        cache_fps=2.0,
        sample_mode="two_per_event",
    )

    assert len(records) == 4
    assert [record["metadata"]["query_variant"] for record in records] == [
        "forward",
        "reverse",
        "forward",
        "reverse",
    ]


def test_degenerate_single_frame_event_is_skipped(tmp_path: Path):
    source = tmp_path / "degenerate_source.mp4"
    source.write_bytes(b"")
    _make_fake_cache(tmp_path, "clip_deg", str(source), 2.0, n_frames=1)

    records = build_records(
        rows=[_event_dir_row("clip_deg", str(source), start=0, end=1, event_id=350)],
        frames_root=tmp_path / "frames",
        cache_fps=2.0,
        sample_mode="two_per_event",
    )

    assert records == []


def test_one_per_event_assignment_is_stable_across_input_order(tmp_path: Path):
    rows = []
    for idx in range(4):
        source = tmp_path / f"stable_source_{idx}.mp4"
        source.write_bytes(b"")
        clip_key = f"stable_clip_{idx}"
        _make_fake_cache(tmp_path, clip_key, str(source), 2.0, n_frames=6)
        rows.append(
            _event_dir_row(
                clip_key,
                str(source),
                start=0,
                end=3,
                event_id=600 + idx,
                phase_id=20 + idx,
                event_text=f"event {idx}",
            )
        )

    forward_order_records = build_records(
        rows=rows,
        frames_root=tmp_path / "frames",
        cache_fps=2.0,
        sample_mode="one_per_event",
    )
    reverse_order_records = build_records(
        rows=list(reversed(rows)),
        frames_root=tmp_path / "frames",
        cache_fps=2.0,
        sample_mode="one_per_event",
    )

    def assignment_map(records: list[dict]) -> dict[tuple[str, int], str]:
        return {
            (record["metadata"]["clip_key"], record["metadata"]["event_id"]): record["metadata"]["query_variant"]
            for record in records
        }

    assert assignment_map(forward_order_records) == assignment_map(reverse_order_records)


def test_load_jsonl_resolves_relative_source_path_against_manifest_dir(tmp_path: Path):
    manifest_dir = tmp_path / "nested" / "manifests"
    manifest_dir.mkdir(parents=True)
    source = tmp_path / "videos" / "source.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"")
    _make_fake_cache(tmp_path, "clip_rel", str(source), 2.0, n_frames=8)

    manifest = manifest_dir / "event_dir_manifest.jsonl"
    manifest.write_text(
        json.dumps(_event_dir_row("clip_rel", "../../videos/source.mp4", start=0, end=4, event_id=401)) + "\n",
        encoding="utf-8",
    )

    rows = _load_jsonl(manifest)
    assert rows[0]["source_video_path"] == str(source.resolve())

    records = build_records(
        rows=rows,
        frames_root=tmp_path / "frames",
        cache_fps=2.0,
        sample_mode="one_per_event",
    )
    assert len(records) == 1
    assert records[0]["metadata"]["source_video_path"] == str(source.resolve())


def test_cli_writes_records_for_tiny_fake_cache(tmp_path: Path):
    source = tmp_path / "cli_source.mp4"
    source.write_bytes(b"")
    _make_fake_cache(tmp_path, "clip_cli", str(source), 2.0, n_frames=2)

    manifest = tmp_path / "event_dir_manifest.jsonl"
    manifest.write_text(
        json.dumps(_event_dir_row("clip_cli", str(source), start=0, end=1, event_id=501)) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "event_dir_binary.jsonl"

    subprocess.run(
        [
            sys.executable,
            "proxy_data/youcook2_seg/temporal_aot/build_event_forward_reverse_from_frames.py",
            "--event-manifest",
            str(manifest),
            "--frames-root",
            str(tmp_path / "frames"),
            "--cache-fps",
            "2.0",
            "--output",
            str(output),
            "--sample-mode",
            "two_per_event",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["problem_type"] == "seg_aot_event_dir_binary"
    assert rows[0]["videos"][0] == rows[1]["videos"][0][::-1]


def test_fps_override_downsampling_preserves_true_reverse_frames(tmp_path: Path):
    source = tmp_path / "long_source.mp4"
    source.write_bytes(b"")
    cache_dir = _make_fake_cache(tmp_path, "clip_long", str(source), 2.0, n_frames=400)

    records = build_records(
        rows=[_event_dir_row("clip_long", str(source), start=0, end=200, event_id=701)],
        frames_root=tmp_path / "frames",
        cache_fps=2.0,
        sample_mode="two_per_event",
    )

    assert len(records) == 2
    forward, reverse = records
    expected_forward = [
        str(path.resolve())
        for path in select_frame_paths_for_span(
            frame_paths=sorted(cache_dir.glob("*.jpg")),
            source_fps=2.0,
            start_sec=0,
            end_sec=200,
            target_fps=1.28,
        )
    ]
    assert forward["metadata"]["video_fps_override"] == 1.28
    assert len(forward["videos"][0]) == len(expected_forward)
    assert forward["videos"][0] == expected_forward
    assert reverse["videos"][0] == list(reversed(expected_forward))


def test_video_fps_override_uses_shared_frame_budget_logic():
    assert _build_video_fps_override(30, num_videos=1) is None
    assert _build_video_fps_override(200, num_videos=1) == 1.28
