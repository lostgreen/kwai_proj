from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from local_scripts.data import mcq, tg
from proxy_data.llava_video_178k.convert_mcq_to_direct import DIRECT_INSTRUCTION
from proxy_data.llava_video_178k.select_mcq_from_rollout_shards import (
    select_records_from_reports,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _mcq_record(
    *,
    record_id: str,
    prompt: str,
    answer: str = "A",
    source: str = "nextqa",
) -> dict:
    return {
        "prompt": prompt,
        "answer": answer,
        "videos": [f"/videos/{record_id}.mp4"],
        "problem_type": "llava_mcq",
        "data_type": "video",
        "metadata": {
            "id": record_id,
            "source": source,
            "duration_bucket": "0_30_s",
            "data_source": f"0_30_s_{source}",
        },
    }


def _report(
    *,
    metadata_id: str = "",
    prompt: str,
    answer: str = "A",
    mean_reward: float | None = None,
    rewards: list[float] | None = None,
    index: int = 0,
) -> dict:
    row = {
        "index": index,
        "metadata_id": metadata_id,
        "prompt": prompt,
        "answer": answer,
        "problem_type": "llava_mcq",
    }
    if mean_reward is not None:
        row["mean_reward"] = mean_reward
    if rewards is not None:
        row["rewards"] = rewards
    return row


def test_select_records_from_reports_keeps_reward_leq_threshold_and_rewrites_direct_prompt(tmp_path: Path):
    old_prompt = "<video>\nQuestion?\nOptions:\nA. yes\nB. no\n\nAnswer with the option letter."
    high_prompt = "<video>\nHard?\nOptions:\nA. yes\nB. no\n\nAnswer with the option letter."
    input_path = tmp_path / "mcq_all.jsonl"
    report_a = tmp_path / "_shard0_report.jsonl"
    report_b = tmp_path / "_shard1_report.jsonl"

    _write_jsonl(
        input_path,
        [
            _mcq_record(record_id="keep", prompt=old_prompt),
            _mcq_record(record_id="too-high", prompt=high_prompt),
        ],
    )
    _write_jsonl(
        report_a,
        [
            _report(metadata_id="keep", prompt=old_prompt, mean_reward=0.375, rewards=[1, 1, 1, 0, 0, 0, 0, 0]),
            _report(metadata_id="too-high", prompt=high_prompt, mean_reward=0.5, rewards=[1, 1, 1, 1, 0, 0, 0, 0]),
        ],
    )
    _write_jsonl(
        report_b,
        [
            _report(metadata_id="keep", prompt=old_prompt, mean_reward=0.125, rewards=[1, 0, 0, 0, 0, 0, 0, 0]),
            _report(metadata_id="missing-score", prompt="bad", rewards=[]),
        ],
    )

    selected, summary = select_records_from_reports(
        input_paths=[input_path],
        report_paths=[report_a, report_b],
        min_mean_reward=0.0,
        max_mean_reward=0.375,
        seed=11,
    )

    assert summary["candidate_count"] == 2
    assert summary["selected_count"] == 1
    assert summary["deduped_count"] == 1
    assert selected[0]["metadata"]["id"] == "keep"
    assert selected[0]["metadata"]["llava_rollout_mean_reward"] == 0.375
    assert DIRECT_INSTRUCTION in selected[0]["prompt"]
    assert "Answer with the option letter." not in selected[0]["prompt"]
    assert selected[0]["messages"] == [{"role": "user", "content": selected[0]["prompt"]}]


def test_select_records_from_reports_falls_back_to_prompt_answer_for_sharded_local_indices(tmp_path: Path):
    first_prompt = "<video>\nFirst?\nOptions:\nA. yes\nB. no\n\nAnswer with the option letter."
    second_prompt = "<video>\nSecond?\nOptions:\nA. yes\nB. no\n\nAnswer with the option letter."
    input_path = tmp_path / "mcq_all.jsonl"
    report_path = tmp_path / "_shard3_report.jsonl"

    _write_jsonl(
        input_path,
        [
            _mcq_record(record_id="first", prompt=first_prompt, answer="A"),
            _mcq_record(record_id="second", prompt=second_prompt, answer="B"),
        ],
    )
    _write_jsonl(
        report_path,
        [
            _report(prompt=second_prompt, answer="B", mean_reward=0.25, index=0),
        ],
    )

    selected, summary = select_records_from_reports(
        input_paths=[input_path],
        report_paths=[report_path],
        min_mean_reward=0.0,
        max_mean_reward=0.375,
    )

    assert summary["selected_count"] == 1
    assert selected[0]["metadata"]["id"] == "second"


def test_mcq_load_val_prefers_requested_val_size_over_stale_files(tmp_path: Path):
    val_dir = tmp_path / "val"
    _write_jsonl(val_dir / "mcq_val_150_frames.jsonl", [{"prompt": "stale"}])
    _write_jsonl(val_dir / "mcq_val_600.jsonl", [{"prompt": "new-0"}, {"prompt": "new-1"}])

    records = mcq.load_val(str(tmp_path), SimpleNamespace(val_mcq_n=600))

    assert [row["prompt"] for row in records] == ["new-0", "new-1"]


def test_tg_and_mcq_default_val_sizes_are_600():
    parser = ArgumentParser()
    tg.add_cli_args(parser)
    mcq.add_cli_args(parser)

    args = parser.parse_args([])

    assert args.val_tg_n == 600
    assert args.val_mcq_n == 600
