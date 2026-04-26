import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from proxy_data.shared.frame_cache import build_source_cache_dir
from proxy_data.youcook2_seg.temporal_aot import (
    filter_rollout_hard_cases,
    hard_qa_pipeline,
    merge_rollout_resume_outputs,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _raw_record(
    *,
    prompt: str,
    answer: str,
    problem_type: str,
    videos: list[list[str]],
    domain_l1: str,
    domain_l2: str,
    duration: float,
) -> dict:
    return {
        "messages": [{"role": "user", "content": prompt}],
        "prompt": prompt,
        "answer": answer,
        "videos": videos,
        "data_type": "video",
        "problem_type": problem_type,
        "metadata": {
            "domain_l1": domain_l1,
            "domain_l2": domain_l2,
            "total_duration_sec": duration,
        },
    }


def _rollout_report(
    *,
    prompt: str,
    answer: str,
    problem_type: str,
    rewards: list[float],
    index: int,
    keep: bool = True,
) -> dict:
    return {
        "index": index,
        "prompt": prompt,
        "answer": answer,
        "problem_type": problem_type,
        "keep": keep,
        "mean_reward": sum(rewards) / len(rewards),
        "rewards": rewards,
        "unique_rewards": sorted(set(rewards)),
    }


def test_load_manifest_rows_reads_multiple_files(tmp_path: Path):
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    _write_jsonl(first, [{"clip_key": "clip_a", "source_video_path": "/tmp/a.mp4"}])
    _write_jsonl(second, [{"clip_key": "clip_b", "source_video_path": "/tmp/b.mp4"}])

    rows = hard_qa_pipeline.load_manifest_rows([first, second])

    assert [row["clip_key"] for row in rows] == ["clip_a", "clip_b"]
    assert rows[0]["_manifest_dir"] == str(first.resolve().parent)
    assert rows[1]["_manifest_path"] == str(second.resolve())


def test_collect_unique_source_infos_anchors_relative_paths_to_manifest_dir(tmp_path: Path):
    manifest_dir = tmp_path / "nested" / "manifests"
    manifest_dir.mkdir(parents=True)
    rows = [
        {
            "clip_key": "clip_a",
            "source_video_path": "../videos/source_a.mp4",
            "_manifest_dir": str(manifest_dir.resolve()),
        }
    ]

    infos = hard_qa_pipeline.collect_unique_source_infos(rows)

    assert infos[0].source_video_path == str((manifest_dir / "../videos/source_a.mp4").resolve())


def test_collect_unique_source_infos_deduplicates_shared_sources():
    rows = [
        {"clip_key": "clip_a", "source_video_path": "/tmp/source_a.mp4", "duration_sec": 12.5},
        {"clip_key": "clip_a", "source_video_path": "/tmp/source_a.mp4"},
        {"clip_key": "clip_b", "source_video_path": "/tmp/source_b.mp4"},
    ]

    infos = hard_qa_pipeline.collect_unique_source_infos(rows)

    assert len(infos) == 2
    assert infos[0].clip_key == "clip_a"
    assert infos[0].duration_sec == 12.5
    assert infos[1].clip_key == "clip_b"


def test_collect_unique_source_infos_raises_on_conflicting_clip_keys():
    rows = [
        {"clip_key": "clip_a", "source_video_path": "/tmp/shared.mp4"},
        {"clip_key": "clip_b", "source_video_path": "/tmp/shared.mp4"},
    ]

    with pytest.raises(ValueError, match="conflicting clip_key"):
        hard_qa_pipeline.collect_unique_source_infos(rows)


def test_collect_unique_source_infos_raises_on_conflicting_durations():
    rows = [
        {"clip_key": "clip_a", "source_video_path": "/tmp/shared.mp4", "duration_sec": 12.0},
        {"clip_key": "clip_a", "source_video_path": "/tmp/shared.mp4", "duration_sec": 13.0},
    ]

    with pytest.raises(ValueError, match="conflicting duration_sec"):
        hard_qa_pipeline.collect_unique_source_infos(rows)


def test_build_source_cache_stage_dry_run_reports_expected_cache_dirs(tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {"clip_key": "clip_a", "source_video_path": str(tmp_path / "videos" / "a.mp4")},
        {"clip_key": "clip_a", "source_video_path": str(tmp_path / "videos" / "a.mp4")},
        {"clip_key": "clip_b", "source_video_path": str(tmp_path / "videos" / "b.mp4")},
    ]
    _write_jsonl(manifest, rows)

    frames_root = tmp_path / "frames"
    summary = hard_qa_pipeline.build_source_cache_stage(
        manifest_paths=[manifest],
        frames_root=frames_root,
        dry_run=True,
    )

    expected_dir = build_source_cache_dir(
        frames_root=frames_root,
        clip_key="clip_a",
        source_video_path=str((tmp_path / "videos" / "a.mp4").resolve()),
        fps=2.0,
    )
    assert summary["dry_run"] is True
    assert summary["unique_source_count"] == 2
    assert summary["caches"][0]["status"] == "dry-run"
    assert summary["caches"][0]["cache_dir"] == str(expected_dir.resolve())
    assert summary["caches"][0]["n_frames"] is None


def test_build_source_cache_stage_calls_ensure_and_validates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(
        manifest,
        [{"clip_key": "clip_a", "source_video_path": str(tmp_path / "videos" / "a.mp4"), "duration_sec": 9.0}],
    )

    calls: list[tuple] = []

    def fake_ensure(source_info, frames_root, fps, jpeg_quality, overwrite):
        calls.append((source_info, Path(frames_root), fps, jpeg_quality, overwrite))
        return {
            "clip_key": source_info.clip_key,
            "source_video_path": source_info.source_video_path,
            "cache_dir": str(Path(frames_root) / "clip_a__abc123"),
            "fps": fps,
            "duration_sec": 9.0,
            "n_frames": 18,
        }

    monkeypatch.setattr(hard_qa_pipeline, "ensure_source_frame_cache", fake_ensure)

    summary = hard_qa_pipeline.build_source_cache_stage(
        manifest_paths=[manifest],
        frames_root=tmp_path / "frames",
        fps=2.0,
        jpeg_quality=4,
        overwrite=True,
        workers=2,
        dry_run=False,
    )

    assert len(calls) == 1
    assert calls[0][0].clip_key == "clip_a"
    assert calls[0][2:] == (2.0, 4, True)
    assert summary["dry_run"] is False
    assert summary["caches"][0]["clip_key"] == "clip_a"
    assert summary["caches"][0]["source_video_path"] == str((tmp_path / "videos" / "a.mp4").resolve())
    assert summary["caches"][0]["status"] == "ready"
    assert summary["caches"][0]["n_frames"] == 18
    assert summary["caches"][0]["cache_dir"] == str((tmp_path / "frames" / "clip_a__abc123").resolve())


def test_summary_cache_shape_matches_between_dry_run_and_real(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    source_path = tmp_path / "videos" / "a.mp4"
    _write_jsonl(manifest, [{"clip_key": "clip_a", "source_video_path": str(source_path)}])

    def fake_ensure(source_info, frames_root, fps, jpeg_quality, overwrite):
        return {
            "clip_key": source_info.clip_key,
            "source_video_path": source_info.source_video_path,
            "cache_dir": str(Path(frames_root) / "clip_a__abc123"),
            "fps": fps,
            "jpeg_quality": jpeg_quality,
            "duration_sec": 9.0,
            "n_frames": 18,
        }

    monkeypatch.setattr(hard_qa_pipeline, "ensure_source_frame_cache", fake_ensure)

    dry_run = hard_qa_pipeline.build_source_cache_stage(
        manifest_paths=[manifest],
        frames_root=tmp_path / "frames",
        dry_run=True,
    )
    real_run = hard_qa_pipeline.build_source_cache_stage(
        manifest_paths=[manifest],
        frames_root=tmp_path / "frames",
        dry_run=False,
    )

    assert set(dry_run["caches"][0]) == set(real_run["caches"][0])
    assert dry_run["caches"][0]["jpeg_quality"] == 2
    assert real_run["caches"][0]["jpeg_quality"] == 2


def test_validate_cache_metadata_rejects_missing_fields():
    with pytest.raises(ValueError, match="duration_sec"):
        hard_qa_pipeline.validate_cache_metadata(
            {
                "fps": 2.0,
                "n_frames": 3,
                "cache_dir": "/tmp/cache",
            }
        )


def test_main_dry_run_writes_stats_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(
        manifest,
        [{"clip_key": "clip_a", "source_video_path": str(tmp_path / "videos" / "a.mp4")}],
    )
    stats_output = tmp_path / "stats.json"

    summary = hard_qa_pipeline.main(
        [
            "build-source-cache",
            "--manifest",
            str(manifest),
            "--frames-root",
            str(tmp_path / "frames"),
            "--dry-run",
            "--stats-output",
            str(stats_output),
        ]
    )

    stdout = capsys.readouterr().out
    saved = json.loads(stats_output.read_text(encoding="utf-8"))
    assert "[build-source-cache] unique sources: 1" in stdout
    assert summary["unique_source_count"] == 1
    assert saved["stage"] == "build-source-cache"
    assert saved["caches"][0]["status"] == "dry-run"


def test_cli_dry_run_subprocess(tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(
        manifest,
        [{"clip_key": "clip_a", "source_video_path": str(tmp_path / "videos" / "a.mp4")}],
    )
    stats_output = tmp_path / "stats.json"

    proc = subprocess.run(
        [
            sys.executable,
            "proxy_data/youcook2_seg/temporal_aot/hard_qa_pipeline.py",
            "build-source-cache",
            "--manifest",
            str(manifest),
            "--frames-root",
            str(tmp_path / "frames"),
            "--dry-run",
            "--stats-output",
            str(stats_output),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert "[build-source-cache] unique sources: 1" in proc.stdout
    saved = json.loads(stats_output.read_text(encoding="utf-8"))
    assert saved["unique_source_count"] == 1


def test_build_raw_record_dedupe_key_handles_nested_frame_lists():
    record_a = _raw_record(
        prompt="nested",
        answer="A",
        problem_type="seg_aot_action_v2t_3way",
        videos=[["f1.jpg", "f2.jpg"], ["g1.jpg"]],
        domain_l1="task_howto",
        domain_l2="food_cooking",
        duration=6.0,
    )
    record_b = _raw_record(
        prompt="nested",
        answer="A",
        problem_type="seg_aot_action_v2t_3way",
        videos=[["f1.jpg", "f2.jpg"], ["g1.jpg"]],
        domain_l1="task_howto",
        domain_l2="food_cooking",
        duration=6.0,
    )

    deduped, duplicate_count = hard_qa_pipeline.dedupe_raw_records([record_a, record_b])

    assert len(deduped) == 1
    assert duplicate_count == 1


def test_dedupe_raw_records_raises_on_conflicting_stats_metadata():
    record_a = _raw_record(
        prompt="same",
        answer="A",
        problem_type="seg_aot_action_v2t_3way",
        videos=[["f1.jpg", "f2.jpg"]],
        domain_l1="task_howto",
        domain_l2="food_cooking",
        duration=6.0,
    )
    record_b = _raw_record(
        prompt="same",
        answer="A",
        problem_type="seg_aot_action_v2t_3way",
        videos=[["f1.jpg", "f2.jpg"]],
        domain_l1="knowledge",
        domain_l2="food_cooking",
        duration=9.0,
    )

    with pytest.raises(ValueError, match="conflicting stats metadata"):
        hard_qa_pipeline.dedupe_raw_records([record_a, record_b])


def test_split_raw_records_is_deterministic_and_stratified():
    records = []
    for idx in range(4):
        records.append(
            _raw_record(
                prompt=f"action-{idx}",
                answer="A",
                problem_type="seg_aot_action_t2v_binary",
                videos=[[f"a{idx}_0.jpg", f"a{idx}_1.jpg"]],
                domain_l1="task_howto",
                domain_l2="food_cooking",
                duration=10.0 + idx,
            )
        )
        records.append(
            _raw_record(
                prompt=f"event-{idx}",
                answer="B",
                problem_type="seg_aot_event_t2v_binary",
                videos=[[f"e{idx}_0.jpg", f"e{idx}_1.jpg", f"e{idx}_2.jpg"]],
                domain_l1="knowledge",
                domain_l2="science",
                duration=20.0 + idx,
            )
        )

    train_a, val_a = hard_qa_pipeline.split_raw_records(records, val_ratio=0.25, seed=7)
    train_b, val_b = hard_qa_pipeline.split_raw_records(records, val_ratio=0.25, seed=7)

    assert train_a == train_b
    assert val_a == val_b
    assert {row["problem_type"] for row in val_a} == {
        "seg_aot_action_t2v_binary",
        "seg_aot_event_t2v_binary",
    }
    assert len(val_a) == 2


def test_split_raw_records_respects_global_budget_on_long_tail():
    records = []
    for idx in range(6):
        records.append(
            _raw_record(
                prompt=f"type_{idx}_0",
                answer="A",
                problem_type=f"ptype_{idx}",
                videos=[[f"{idx}_0.jpg"]],
                domain_l1="task_howto",
                domain_l2="food_cooking",
                duration=5.0,
            )
        )
        records.append(
            _raw_record(
                prompt=f"type_{idx}_1",
                answer="B",
                problem_type=f"ptype_{idx}",
                videos=[[f"{idx}_1.jpg"]],
                domain_l1="task_howto",
                domain_l2="food_cooking",
                duration=6.0,
            )
        )

    train_rows, val_rows = hard_qa_pipeline.split_raw_records(records, val_ratio=0.25, seed=13)

    assert len(val_rows) == 3
    assert len(train_rows) == 9
    assert len({row["problem_type"] for row in val_rows}) == 3
    assert len({row["problem_type"] for row in train_rows}) == 6


def test_summarize_raw_records_computes_problem_domain_and_averages():
    records = [
        _raw_record(
            prompt="one",
            answer="A",
            problem_type="seg_aot_action_v2t_3way",
            videos=[["f1.jpg", "f2.jpg", "f3.jpg"]],
            domain_l1="task_howto",
            domain_l2="food_cooking",
            duration=9.0,
        ),
        _raw_record(
            prompt="two",
            answer="B",
            problem_type="seg_aot_event_dir_binary",
            videos=[["g1.jpg"], ["h1.jpg", "h2.jpg"]],
            domain_l1="task_howto",
            domain_l2="home",
            duration=3.0,
        ),
    ]

    stats = hard_qa_pipeline.summarize_raw_records(records)

    assert stats["total_count"] == 2
    assert stats["by_problem_type"] == {
        "seg_aot_action_v2t_3way": 1,
        "seg_aot_event_dir_binary": 1,
    }
    assert stats["by_domain_l1"] == {"task_howto": 2}
    assert stats["by_domain_l2"] == {"food_cooking": 1, "home": 1}
    assert stats["average_frame_count"] == 3.0
    assert stats["average_duration_sec"] == 6.0


def test_merge_raw_stage_dry_run_returns_summary_without_writing_files(tmp_path: Path):
    first = tmp_path / "action.jsonl"
    second = tmp_path / "event.jsonl"
    output_dir = tmp_path / "merged_raw"
    _write_jsonl(
        first,
        [
            _raw_record(
                prompt="shared",
                answer="A",
                problem_type="seg_aot_action_v2t_3way",
                videos=[["f1.jpg", "f2.jpg"]],
                domain_l1="task_howto",
                domain_l2="food_cooking",
                duration=8.0,
            )
        ],
    )
    _write_jsonl(
        second,
        [
            _raw_record(
                prompt="shared",
                answer="A",
                problem_type="seg_aot_action_v2t_3way",
                videos=[["f1.jpg", "f2.jpg"]],
                domain_l1="task_howto",
                domain_l2="food_cooking",
                duration=8.0,
            ),
            _raw_record(
                prompt="unique",
                answer="B",
                problem_type="seg_aot_event_dir_binary",
                videos=[["g1.jpg", "g2.jpg", "g3.jpg"]],
                domain_l1="knowledge",
                domain_l2="science",
                duration=12.0,
            ),
        ],
    )

    summary = hard_qa_pipeline.merge_raw_stage(
        input_paths=[first, second],
        output_dir=output_dir,
        seed=3,
        val_ratio=0.5,
        dry_run=True,
    )

    assert summary["dry_run"] is True
    assert summary["input_record_count"] == 3
    assert summary["deduped_record_count"] == 2
    assert summary["duplicate_record_count"] == 1
    assert summary["train_output_path"].endswith("train.jsonl")
    assert not (output_dir / "train.jsonl").exists()
    assert not (output_dir / "val.jsonl").exists()
    assert not (output_dir / "stats.json").exists()


def test_merge_raw_stage_writes_outputs_and_stats(tmp_path: Path):
    raw_path = tmp_path / "raw.jsonl"
    output_dir = tmp_path / "merged_raw"
    _write_jsonl(
        raw_path,
        [
            _raw_record(
                prompt="a0",
                answer="A",
                problem_type="seg_aot_action_t2v_binary",
                videos=[["a0_0.jpg", "a0_1.jpg"]],
                domain_l1="task_howto",
                domain_l2="food_cooking",
                duration=5.0,
            ),
            _raw_record(
                prompt="a1",
                answer="A",
                problem_type="seg_aot_action_t2v_binary",
                videos=[["a1_0.jpg", "a1_1.jpg"]],
                domain_l1="task_howto",
                domain_l2="food_cooking",
                duration=7.0,
            ),
            _raw_record(
                prompt="e0",
                answer="B",
                problem_type="seg_aot_event_dir_binary",
                videos=[["e0_0.jpg", "e0_1.jpg", "e0_2.jpg"]],
                domain_l1="knowledge",
                domain_l2="science",
                duration=9.0,
            ),
            _raw_record(
                prompt="e1",
                answer="B",
                problem_type="seg_aot_event_dir_binary",
                videos=[["e1_0.jpg", "e1_1.jpg", "e1_2.jpg"]],
                domain_l1="knowledge",
                domain_l2="science",
                duration=11.0,
            ),
        ],
    )

    summary = hard_qa_pipeline.merge_raw_stage(
        input_paths=[raw_path],
        output_dir=output_dir,
        seed=11,
        val_ratio=0.25,
        dry_run=False,
    )

    train_rows = hard_qa_pipeline.load_jsonl_rows([output_dir / "train.jsonl"])
    val_rows = hard_qa_pipeline.load_jsonl_rows([output_dir / "val.jsonl"])
    saved_stats = json.loads((output_dir / "stats.json").read_text(encoding="utf-8"))

    assert summary["train_count"] == len(train_rows)
    assert summary["val_count"] == len(val_rows)
    assert saved_stats["all"]["by_problem_type"] == {
        "seg_aot_action_t2v_binary": 2,
        "seg_aot_event_dir_binary": 2,
    }
    assert saved_stats["all"]["by_domain_l1"] == {"knowledge": 2, "task_howto": 2}
    assert saved_stats["all"]["average_frame_count"] == 2.5
    assert saved_stats["all"]["average_duration_sec"] == 8.0
    assert len(val_rows) == 1
    assert len({row["problem_type"] for row in val_rows}) == 1


def test_merge_raw_cli_subprocess(tmp_path: Path):
    first = tmp_path / "action.jsonl"
    second = tmp_path / "event.jsonl"
    output_dir = tmp_path / "merged_raw"
    _write_jsonl(
        first,
        [
            _raw_record(
                prompt="a0",
                answer="A",
                problem_type="seg_aot_action_v2t_3way",
                videos=[["a0_0.jpg", "a0_1.jpg"]],
                domain_l1="task_howto",
                domain_l2="food_cooking",
                duration=4.0,
            ),
            _raw_record(
                prompt="a1",
                answer="A",
                problem_type="seg_aot_action_v2t_3way",
                videos=[["a1_0.jpg", "a1_1.jpg"]],
                domain_l1="task_howto",
                domain_l2="food_cooking",
                duration=5.0,
            ),
        ],
    )
    _write_jsonl(
        second,
        [
            _raw_record(
                prompt="e0",
                answer="B",
                problem_type="seg_aot_event_v2t_3way",
                videos=[["e0_0.jpg", "e0_1.jpg", "e0_2.jpg"]],
                domain_l1="knowledge",
                domain_l2="science",
                duration=7.0,
            ),
            _raw_record(
                prompt="e1",
                answer="B",
                problem_type="seg_aot_event_v2t_3way",
                videos=[["e1_0.jpg", "e1_1.jpg", "e1_2.jpg"]],
                domain_l1="knowledge",
                domain_l2="science",
                duration=8.0,
            ),
        ],
    )

    proc = subprocess.run(
        [
            sys.executable,
            "proxy_data/youcook2_seg/temporal_aot/hard_qa_pipeline.py",
            "merge-raw",
            "--input",
            str(first),
            "--input",
            str(second),
            "--output-dir",
            str(output_dir),
            "--seed",
            "17",
            "--val-ratio",
            "0.25",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert "[merge-raw] deduped records: 4" in proc.stdout
    assert (output_dir / "train.jsonl").exists()
    assert (output_dir / "val.jsonl").exists()
    saved_stats = json.loads((output_dir / "stats.json").read_text(encoding="utf-8"))
    assert saved_stats["stage"] == "merge-raw"
    assert saved_stats["val_count"] == 1


def test_filter_rollout_hard_cases_applies_success_threshold_range_and_balancing(tmp_path: Path):
    input_path = tmp_path / "merged_train.jsonl"
    report_path = tmp_path / "rollout_report.jsonl"
    output_path = tmp_path / "hard_cases.jsonl"
    stats_path = tmp_path / "hard_cases.stats.json"
    input_rows = [
        _raw_record(
            prompt="action-task-0",
            answer="A",
            problem_type="seg_aot_action_v2t_3way",
            videos=[["a0_0.jpg", "a0_1.jpg"]],
            domain_l1="task_howto",
            domain_l2="food_cooking",
            duration=5.0,
        ),
        _raw_record(
            prompt="action-knowledge-0",
            answer="A",
            problem_type="seg_aot_action_v2t_3way",
            videos=[["a1_0.jpg", "a1_1.jpg", "a1_2.jpg"]],
            domain_l1="knowledge",
            domain_l2="science",
            duration=6.0,
        ),
        _raw_record(
            prompt="action-task-1",
            answer="A",
            problem_type="seg_aot_action_v2t_3way",
            videos=[["a2_0.jpg", "a2_1.jpg", "a2_2.jpg", "a2_3.jpg"]],
            domain_l1="task_howto",
            domain_l2="food_cooking",
            duration=7.0,
        ),
        _raw_record(
            prompt="event-task-0",
            answer="B",
            problem_type="seg_aot_event_dir_binary",
            videos=[["e0_0.jpg", "e0_1.jpg"]],
            domain_l1="task_howto",
            domain_l2="home",
            duration=8.0,
        ),
        _raw_record(
            prompt="event-knowledge-0",
            answer="B",
            problem_type="seg_aot_event_dir_binary",
            videos=[["e1_0.jpg", "e1_1.jpg", "e1_2.jpg"]],
            domain_l1="knowledge",
            domain_l2="science",
            duration=9.0,
        ),
        _raw_record(
            prompt="event-knowledge-1",
            answer="B",
            problem_type="seg_aot_event_dir_binary",
            videos=[["e2_0.jpg", "e2_1.jpg", "e2_2.jpg", "e2_3.jpg"]],
            domain_l1="knowledge",
            domain_l2="science",
            duration=10.0,
        ),
        _raw_record(
            prompt="too-easy",
            answer="A",
            problem_type="seg_aot_action_v2t_3way",
            videos=[["x0.jpg"]],
            domain_l1="task_howto",
            domain_l2="food_cooking",
            duration=4.0,
        ),
        _raw_record(
            prompt="no-success",
            answer="B",
            problem_type="seg_aot_event_dir_binary",
            videos=[["y0.jpg", "y1.jpg"]],
            domain_l1="knowledge",
            domain_l2="science",
            duration=11.0,
        ),
        _raw_record(
            prompt="upstream-rejected",
            answer="A",
            problem_type="seg_aot_action_v2t_3way",
            videos=[["z0.jpg", "z1.jpg"]],
            domain_l1="task_howto",
            domain_l2="food_cooking",
            duration=12.0,
        ),
    ]
    report_rows = [
        _rollout_report(
            prompt="action-task-0",
            answer="A",
            problem_type="seg_aot_action_v2t_3way",
            rewards=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            index=0,
        ),
        _rollout_report(
            prompt="action-knowledge-0",
            answer="A",
            problem_type="seg_aot_action_v2t_3way",
            rewards=[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            index=1,
        ),
        _rollout_report(
            prompt="action-task-1",
            answer="A",
            problem_type="seg_aot_action_v2t_3way",
            rewards=[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            index=2,
        ),
        _rollout_report(
            prompt="event-task-0",
            answer="B",
            problem_type="seg_aot_event_dir_binary",
            rewards=[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            index=3,
        ),
        _rollout_report(
            prompt="event-knowledge-0",
            answer="B",
            problem_type="seg_aot_event_dir_binary",
            rewards=[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            index=4,
        ),
        _rollout_report(
            prompt="event-knowledge-1",
            answer="B",
            problem_type="seg_aot_event_dir_binary",
            rewards=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            index=5,
        ),
        _rollout_report(
            prompt="too-easy",
            answer="A",
            problem_type="seg_aot_action_v2t_3way",
            rewards=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            index=6,
        ),
        _rollout_report(
            prompt="no-success",
            answer="B",
            problem_type="seg_aot_event_dir_binary",
            rewards=[0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
            index=7,
        ),
        _rollout_report(
            prompt="upstream-rejected",
            answer="A",
            problem_type="seg_aot_action_v2t_3way",
            rewards=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            index=8,
            keep=False,
        ),
    ]
    _write_jsonl(input_path, input_rows)
    _write_jsonl(report_path, report_rows)

    summary = filter_rollout_hard_cases.filter_rollout_hard_cases(
        report_path=report_path,
        input_path=input_path,
        output_path=output_path,
        stats_output_path=stats_path,
        min_mean_reward=0.125,
        max_mean_reward=0.625,
        min_success_count=1,
        success_threshold=1.0,
        target_total=4,
        nested_balance_key="domain_l1",
        seed=9,
    )

    selected_rows = hard_qa_pipeline.load_jsonl_rows([output_path])
    saved_stats = json.loads(stats_path.read_text(encoding="utf-8"))

    assert summary["candidate_count"] == 6
    assert summary["filtered_out_by_upstream_keep"] == 1
    assert summary["filtered_out_by_mean_reward"] == 1
    assert summary["filtered_out_by_success_count"] == 1
    assert len(selected_rows) == 4
    assert saved_stats["total_count"] == 4
    assert saved_stats["by_problem_type"] == {
        "seg_aot_action_v2t_3way": 2,
        "seg_aot_event_dir_binary": 2,
    }
    assert saved_stats["by_domain_l1"] == {
        "knowledge": 2,
        "task_howto": 2,
    }
    assert saved_stats["average_frame_count"] > 0
    assert saved_stats["average_duration_sec"] > 0


def test_filter_rollout_hard_cases_prefers_index_identity_over_prompt_answer(tmp_path: Path):
    input_path = tmp_path / "merged_train.jsonl"
    report_path = tmp_path / "rollout_report.jsonl"
    output_path = tmp_path / "hard_cases.jsonl"
    _write_jsonl(
        input_path,
        [
            _raw_record(
                prompt="shared-prompt",
                answer="A",
                problem_type="seg_aot_action_v2t_3way",
                videos=[["a0.jpg", "a1.jpg"]],
                domain_l1="task_howto",
                domain_l2="food_cooking",
                duration=5.0,
            ),
            _raw_record(
                prompt="shared-prompt",
                answer="A",
                problem_type="seg_aot_action_v2t_3way",
                videos=[["b0.jpg", "b1.jpg", "b2.jpg"]],
                domain_l1="knowledge",
                domain_l2="science",
                duration=6.0,
            ),
        ],
    )
    _write_jsonl(
        report_path,
        [
            _rollout_report(
                prompt="shared-prompt",
                answer="A",
                problem_type="seg_aot_action_v2t_3way",
                rewards=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                index=0,
            ),
            _rollout_report(
                prompt="shared-prompt",
                answer="A",
                problem_type="seg_aot_action_v2t_3way",
                rewards=[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                index=1,
            ),
        ],
    )

    summary = filter_rollout_hard_cases.filter_rollout_hard_cases(
        report_path=report_path,
        input_path=input_path,
        output_path=output_path,
        min_mean_reward=0.125,
        max_mean_reward=0.625,
        min_success_count=1,
        success_threshold=1.0,
        target_total=0,
        seed=3,
    )

    selected_rows = hard_qa_pipeline.load_jsonl_rows([output_path])

    assert summary["candidate_count"] == 2
    assert len(selected_rows) == 2
    assert selected_rows[0]["videos"] != selected_rows[1]["videos"]


def test_filter_rollout_hard_cases_falls_back_when_index_mismatches_content(tmp_path: Path):
    input_path = tmp_path / "merged_train.jsonl"
    report_path = tmp_path / "rollout_report.jsonl"
    output_path = tmp_path / "hard_cases.jsonl"
    input_rows = [
        _raw_record(
            prompt="first",
            answer="A",
            problem_type="seg_aot_action_v2t_3way",
            videos=[["a0.jpg", "a1.jpg"]],
            domain_l1="task_howto",
            domain_l2="food_cooking",
            duration=5.0,
        ),
        _raw_record(
            prompt="second",
            answer="B",
            problem_type="seg_aot_event_dir_binary",
            videos=[["b0.jpg", "b1.jpg", "b2.jpg"]],
            domain_l1="knowledge",
            domain_l2="science",
            duration=6.0,
        ),
    ]
    _write_jsonl(input_path, input_rows)
    _write_jsonl(
        report_path,
        [
            _rollout_report(
                prompt="second",
                answer="B",
                problem_type="seg_aot_event_dir_binary",
                rewards=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                index=0,
            )
        ],
    )

    summary = filter_rollout_hard_cases.filter_rollout_hard_cases(
        report_path=report_path,
        input_path=input_path,
        output_path=output_path,
        min_mean_reward=0.125,
        max_mean_reward=0.625,
        min_success_count=1,
        success_threshold=1.0,
        target_total=0,
        seed=5,
    )

    selected_rows = hard_qa_pipeline.load_jsonl_rows([output_path])

    assert summary["candidate_count"] == 1
    assert len(selected_rows) == 1
    assert selected_rows[0]["prompt"] == "second"
    assert selected_rows[0]["videos"] == [["b0.jpg", "b1.jpg", "b2.jpg"]]


def test_filter_rollout_hard_cases_keeps_multiple_fallback_matches_from_sharded_like_indices(tmp_path: Path):
    input_path = tmp_path / "merged_train.jsonl"
    report_path = tmp_path / "rollout_report.jsonl"
    output_path = tmp_path / "hard_cases.jsonl"
    input_rows = [
        _raw_record(
            prompt="first",
            answer="A",
            problem_type="seg_aot_action_v2t_3way",
            videos=[["a0.jpg", "a1.jpg"]],
            domain_l1="task_howto",
            domain_l2="food_cooking",
            duration=5.0,
        ),
        _raw_record(
            prompt="second",
            answer="B",
            problem_type="seg_aot_event_dir_binary",
            videos=[["b0.jpg", "b1.jpg", "b2.jpg"]],
            domain_l1="knowledge",
            domain_l2="science",
            duration=6.0,
        ),
    ]
    _write_jsonl(input_path, input_rows)
    _write_jsonl(
        report_path,
        [
            _rollout_report(
                prompt="first",
                answer="A",
                problem_type="seg_aot_action_v2t_3way",
                rewards=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                index=0,
            ),
            _rollout_report(
                prompt="second",
                answer="B",
                problem_type="seg_aot_event_dir_binary",
                rewards=[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                index=0,
            ),
        ],
    )

    summary = filter_rollout_hard_cases.filter_rollout_hard_cases(
        report_path=report_path,
        input_path=input_path,
        output_path=output_path,
        min_mean_reward=0.125,
        max_mean_reward=0.625,
        min_success_count=1,
        success_threshold=1.0,
        target_total=0,
        seed=6,
    )

    selected_rows = hard_qa_pipeline.load_jsonl_rows([output_path])

    assert summary["candidate_count"] == 2
    assert len(selected_rows) == 2
    assert {row["prompt"] for row in selected_rows} == {"first", "second"}


def test_merge_rollout_resume_outputs_dedupes_reports_and_kept_records(tmp_path: Path):
    rollout_dir = tmp_path / "rollout"
    output_dir = tmp_path / "merged"
    rollout_dir.mkdir()

    base_kept = _raw_record(
        prompt="first",
        answer="A",
        problem_type="seg_aot_action_v2t_3way",
        videos=[["a0.jpg", "a1.jpg"]],
        domain_l1="task_howto",
        domain_l2="food_cooking",
        duration=5.0,
    )
    resume_kept = _raw_record(
        prompt="second",
        answer="B",
        problem_type="seg_aot_event_dir_binary",
        videos=[["b0.jpg", "b1.jpg"]],
        domain_l1="knowledge",
        domain_l2="science",
        duration=6.0,
    )

    _write_jsonl(
        rollout_dir / "_shard0_report.jsonl",
        [
            _rollout_report(
                prompt="first",
                answer="A",
                problem_type="seg_aot_action_v2t_3way",
                rewards=[1.0, 0.0],
                index=0,
                keep=True,
            ),
            _rollout_report(
                prompt="second",
                answer="B",
                problem_type="seg_aot_event_dir_binary",
                rewards=[0.0, 0.0],
                index=1,
                keep=False,
            ),
        ],
    )
    _write_jsonl(
        rollout_dir / "_shard0_resume2_report.jsonl",
        [
            _rollout_report(
                prompt="second",
                answer="B",
                problem_type="seg_aot_event_dir_binary",
                rewards=[1.0, 0.0],
                index=0,
                keep=True,
            )
        ],
    )
    _write_jsonl(rollout_dir / "_shard0_report_global.jsonl", [{"should": "be ignored"}])
    _write_jsonl(rollout_dir / "_shard0_kept.jsonl", [base_kept])
    _write_jsonl(rollout_dir / "_shard0_resume2_kept.jsonl", [base_kept, resume_kept])

    report_files = merge_rollout_resume_outputs.discover_files(rollout_dir, merge_rollout_resume_outputs.REPORT_RE)
    kept_files = merge_rollout_resume_outputs.discover_files(rollout_dir, merge_rollout_resume_outputs.KEPT_RE)
    reports, report_merge_summary = merge_rollout_resume_outputs.merge_reports(report_files)
    kept_records, kept_merge_summary = merge_rollout_resume_outputs.merge_kept_records(kept_files)
    report_summary = merge_rollout_resume_outputs.summarize_reports(reports, success_threshold=1.0)
    plot_paths = merge_rollout_resume_outputs.write_plots(output_dir, reports, report_summary, kept_records)

    assert [path["path"].name for path in report_files] == ["_shard0_report.jsonl", "_shard0_resume2_report.jsonl"]
    assert report_merge_summary["report_attempt_count"] == 3
    assert report_merge_summary["report_unique_count"] == 2
    assert kept_merge_summary["kept_input_record_count"] == 3
    assert kept_merge_summary["kept_unique_record_count"] == 2
    assert report_summary["report_kept"] == 2
    assert report_summary["by_problem_type"]["seg_aot_event_dir_binary"]["kept"] == 1
    assert all(Path(path).exists() for path in plot_paths.values())


def test_rollout_filter_stage_dry_run_returns_planned_commands(tmp_path: Path):
    input_path = tmp_path / "merged_train.jsonl"
    _write_jsonl(
        input_path,
        [
            _raw_record(
                prompt="a0",
                answer="A",
                problem_type="seg_aot_action_v2t_3way",
                videos=[["a0_0.jpg", "a0_1.jpg"]],
                domain_l1="task_howto",
                domain_l2="food_cooking",
                duration=4.0,
            )
        ],
    )

    summary = hard_qa_pipeline.rollout_filter_stage(
        input_path=input_path,
        output_dir=tmp_path / "rollout",
        dry_run=True,
    )

    assert summary["dry_run"] is True
    assert summary["model_path"] == hard_qa_pipeline.DEFAULT_ROLLOUT_MODEL_PATH
    assert summary["num_rollouts"] == 8
    assert summary["target_total"] == 5000
    assert summary["min_mean_reward"] == 0.125
    assert summary["max_mean_reward"] == 0.625
    assert summary["min_success_count"] == 1
    assert len(summary["planned_commands"]) == 2
    assert summary["rollout_output_path"].endswith("rollout_output.jsonl")
    assert "offline_rollout_filter.py" in summary["planned_commands"][0]["display"]
    assert "filter_rollout_hard_cases.py" in summary["planned_commands"][1]["display"]
    assert summary["hard_cases_output_path"].endswith("hard_cases.jsonl")
    assert summary["hard_cases_stats_output_path"].endswith("hard_cases.stats.json")


def test_main_rollout_filter_dry_run_writes_stats_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    input_path = tmp_path / "merged_train.jsonl"
    stats_output = tmp_path / "summary.json"
    _write_jsonl(
        input_path,
        [
            _raw_record(
                prompt="a0",
                answer="A",
                problem_type="seg_aot_action_v2t_3way",
                videos=[["a0_0.jpg", "a0_1.jpg"]],
                domain_l1="task_howto",
                domain_l2="food_cooking",
                duration=4.0,
            )
        ],
    )

    summary = hard_qa_pipeline.main(
        [
            "rollout-filter",
            "--input",
            str(input_path),
            "--output-dir",
            str(tmp_path / "rollout"),
            "--dry-run",
            "--stats-output",
            str(stats_output),
        ]
    )

    stdout = capsys.readouterr().out
    saved = json.loads(stats_output.read_text(encoding="utf-8"))
    assert "[rollout-filter] dry-run rollout:" in stdout
    assert "[rollout-filter] dry-run filter:" in stdout
    assert summary["dry_run"] is True
    assert saved["stage"] == "rollout-filter"
    assert saved["planned_commands"][0]["stage"] == "rollout"


def test_rollout_filter_stage_runs_planned_commands_with_monkeypatched_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    input_path = tmp_path / "merged_train.jsonl"
    output_dir = tmp_path / "rollout"
    input_rows = [
        _raw_record(
            prompt="keep-action",
            answer="A",
            problem_type="seg_aot_action_v2t_3way",
            videos=[["a0_0.jpg", "a0_1.jpg"]],
            domain_l1="task_howto",
            domain_l2="food_cooking",
            duration=5.0,
        ),
        _raw_record(
            prompt="keep-event",
            answer="B",
            problem_type="seg_aot_event_dir_binary",
            videos=[["e0_0.jpg", "e0_1.jpg", "e0_2.jpg"]],
            domain_l1="knowledge",
            domain_l2="science",
            duration=6.0,
        ),
        _raw_record(
            prompt="drop-no-success",
            answer="B",
            problem_type="seg_aot_event_dir_binary",
            videos=[["d0_0.jpg", "d0_1.jpg"]],
            domain_l1="knowledge",
            domain_l2="science",
            duration=7.0,
        ),
    ]
    _write_jsonl(input_path, input_rows)
    calls: list[list[str]] = []

    def _arg(command: list[str], flag: str) -> str:
        return command[command.index(flag) + 1]

    def fake_run(command, check, cwd):
        assert check is True
        assert cwd == str(Path(__file__).resolve().parents[1])
        calls.append(list(command))
        script_name = Path(command[1]).name
        if script_name == "offline_rollout_filter.py":
            rollout_output = Path(_arg(command, "--output_jsonl"))
            report_output = Path(_arg(command, "--report_jsonl"))
            rollout_output.parent.mkdir(parents=True, exist_ok=True)
            report_output.parent.mkdir(parents=True, exist_ok=True)
            _write_jsonl(rollout_output, input_rows[:2])
            _write_jsonl(
                report_output,
                [
                    _rollout_report(
                        prompt="keep-action",
                        answer="A",
                        problem_type="seg_aot_action_v2t_3way",
                        rewards=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        index=0,
                    ),
                    _rollout_report(
                        prompt="keep-event",
                        answer="B",
                        problem_type="seg_aot_event_dir_binary",
                        rewards=[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        index=1,
                    ),
                    _rollout_report(
                        prompt="drop-no-success",
                        answer="B",
                        problem_type="seg_aot_event_dir_binary",
                        rewards=[0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        index=2,
                    ),
                ],
            )
        elif script_name == "filter_rollout_hard_cases.py":
            filter_rollout_hard_cases.filter_rollout_hard_cases(
                report_path=_arg(command, "--report"),
                input_path=_arg(command, "--input"),
                output_path=_arg(command, "--output"),
                stats_output_path=_arg(command, "--stats-output"),
                min_mean_reward=float(_arg(command, "--min-mean-reward")),
                max_mean_reward=float(_arg(command, "--max-mean-reward")),
                min_success_count=int(_arg(command, "--min-success-count")),
                success_threshold=float(_arg(command, "--success-threshold")),
                target_total=int(_arg(command, "--target-total")),
                nested_balance_key=_arg(command, "--nested-balance-key"),
                seed=int(_arg(command, "--seed")),
            )
        else:
            raise AssertionError(f"Unexpected command: {command}")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(hard_qa_pipeline.subprocess, "run", fake_run)

    summary = hard_qa_pipeline.rollout_filter_stage(
        input_path=input_path,
        output_dir=output_dir,
        dry_run=False,
    )

    assert len(calls) == 2
    assert Path(calls[0][1]).name == "offline_rollout_filter.py"
    assert Path(calls[1][1]).name == "filter_rollout_hard_cases.py"
    assert summary["rollout_output_record_count"] == 2
    assert summary["report_record_count"] == 3
    assert summary["hard_case_count"] == 2
    assert summary["hard_case_summary"]["by_problem_type"] == {
        "seg_aot_action_v2t_3way": 1,
        "seg_aot_event_dir_binary": 1,
    }
    assert summary["hard_case_filter_summary"]["min_success_count"] == 1
    assert summary["hard_case_filter_summary"]["total_count"] == 2


def test_rollout_filter_cli_dry_run_subprocess(tmp_path: Path):
    input_path = tmp_path / "merged_train.jsonl"
    _write_jsonl(
        input_path,
        [
            _raw_record(
                prompt="a0",
                answer="A",
                problem_type="seg_aot_action_v2t_3way",
                videos=[["a0_0.jpg", "a0_1.jpg"]],
                domain_l1="task_howto",
                domain_l2="food_cooking",
                duration=4.0,
            )
        ],
    )

    proc = subprocess.run(
        [
            sys.executable,
            "proxy_data/youcook2_seg/temporal_aot/hard_qa_pipeline.py",
            "rollout-filter",
            "--input",
            str(input_path),
            "--output-dir",
            str(tmp_path / "rollout"),
            "--dry-run",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert "[rollout-filter] dry-run rollout:" in proc.stdout
    assert "[rollout-filter] dry-run filter:" in proc.stdout
    assert "hard_cases.stats.json" in proc.stdout
