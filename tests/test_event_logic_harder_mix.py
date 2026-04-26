from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from proxy_data.youcook2_seg.event_logic.build_harder_training_mix import build_harder_mix


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _mcq_record(problem_type: str, sample_id: str) -> dict:
    return {
        "prompt": f"<video>\nQuestion {sample_id}?\nOptions:\nA. one\nB. two",
        "answer": "A",
        "videos": [[f"/frames/{sample_id}/000001.jpg"]],
        "data_type": "video",
        "problem_type": problem_type,
        "metadata": {"id": sample_id, "clip_key": sample_id},
    }


def _sort_record(sample_id: str, ordered_len: int) -> dict:
    return {
        "prompt": f"<video>\nSort {sample_id}",
        "answer": "".join(str(i) for i in range(1, ordered_len + 1)),
        "videos": [[f"/frames/{sample_id}/{idx:06d}.jpg"] for idx in range(1, ordered_len + 1)],
        "data_type": "video",
        "problem_type": "event_logic_sort",
        "metadata": {
            "id": sample_id,
            "clip_key": sample_id,
            "ordered_ids": [f"Event {idx}" for idx in range(1, ordered_len + 1)],
        },
    }


def test_build_harder_mix_keeps_hard_pn_fb_and_fills_with_longest_sort(tmp_path: Path):
    pn_path = tmp_path / "pn_hard.jsonl"
    fb_path = tmp_path / "fb_hard.jsonl"
    sort_path = tmp_path / "sort_frames.jsonl"
    output_path = tmp_path / "train_10k.jsonl"
    stats_path = tmp_path / "stats.json"

    _write_jsonl(
        pn_path,
        [
            _mcq_record("event_logic_predict_next", "pn0"),
            _mcq_record("event_logic_predict_next", "pn1"),
        ],
    )
    _write_jsonl(fb_path, [_mcq_record("event_logic_fill_blank", "fb0")])
    _write_jsonl(
        sort_path,
        [
            _sort_record("sort_short", 3),
            _sort_record("sort_long", 5),
            _sort_record("sort_mid", 4),
        ],
    )

    stats = build_harder_mix(
        hard_paths=[pn_path, fb_path],
        sort_path=sort_path,
        output_path=output_path,
        stats_path=stats_path,
        target_total=5,
        seed=7,
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    ids = {row["metadata"]["id"] for row in rows}
    assert len(rows) == 5
    assert {"pn0", "pn1", "fb0", "sort_long", "sort_mid"} == ids
    assert "sort_short" not in ids
    assert stats["selected_sort_lengths"] == [5, 4]
    assert stats["selected_by_problem_type"] == {
        "event_logic_fill_blank": 1,
        "event_logic_predict_next": 2,
        "event_logic_sort": 2,
    }
    assert json.loads(stats_path.read_text(encoding="utf-8"))["target_total"] == 5


def test_build_harder_mix_deduplicates_hard_cases_before_sort_fill(tmp_path: Path):
    pn_path = tmp_path / "pn_hard.jsonl"
    fb_path = tmp_path / "fb_hard.jsonl"
    sort_path = tmp_path / "sort_frames.jsonl"
    output_path = tmp_path / "train.jsonl"

    dup = _mcq_record("event_logic_predict_next", "dup")
    _write_jsonl(pn_path, [dup, dup])
    _write_jsonl(fb_path, [])
    _write_jsonl(sort_path, [_sort_record("sort_long", 5)])

    stats = build_harder_mix(
        hard_paths=[pn_path, fb_path],
        sort_path=sort_path,
        output_path=output_path,
        stats_path=None,
        target_total=2,
        seed=7,
    )

    assert stats["hard_duplicate_count"] == 1
    assert stats["selected_before_shuffle_by_source"] == {"hard": 1, "sort": 1}
