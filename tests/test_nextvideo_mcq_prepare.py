from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from video_proxy.data.base_sources.mcq.nextvideo.prepare_nextvideo import (  # noqa: E402
    convert_record,
    load_jsonl,
)
from video_proxy.data.base_sources.mcq.prepare.convert_to_direct import DIRECT_INSTRUCTION  # noqa: E402


def _nextvideo_row(*, answer: str | None = "D", gt: str | None = None) -> dict:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "text": None},
                {
                    "type": "text",
                    "text": (
                        "how many children are in the video?\n"
                        "A. one\n"
                        "B. three\n"
                        "C. seven\n"
                        "D. two\n"
                        "E. five\n"
                        " Answer with the option's letter from the given choices directly."
                    ),
                },
            ],
        },
    ]
    if answer is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            }
        )

    row = {
        "messages": messages,
        "video": {"path": "./NExTVideo/1164/3238737531.mp4", "num_frames": 8},
    }
    if gt is not None:
        row["gt"] = gt
    return row


def test_convert_record_normalizes_nextvideo_train_row(tmp_path: Path):
    dataset_root = tmp_path / "NeXTVideo"
    video_path = dataset_root / "NExTVideo" / "1164" / "3238737531.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"fake-video")

    record = convert_record(
        _nextvideo_row(answer="d"),
        dataset_root=dataset_root,
        split="train",
        line_no=7,
        verify_video=True,
    )

    assert record is not None
    assert record["problem_type"] == "llava_mcq"
    assert record["data_type"] == "video"
    assert record["answer"] == "D"
    assert record["videos"] == [str(video_path)]
    assert record["prompt"].startswith("<video>\nhow many children")
    assert "Answer with the option's letter" not in record["prompt"]
    assert DIRECT_INSTRUCTION in record["prompt"]
    assert record["messages"] == [{"role": "user", "content": record["prompt"]}]
    assert record["metadata"]["id"] == "nextvideo_train_000007"
    assert record["metadata"]["video_id"] == "1164-3238737531"
    assert record["metadata"]["source"] == "nextvideo"
    assert record["metadata"]["data_source"] == "nextvideo_1164"
    assert record["metadata"]["raw_video_path"] == "./NExTVideo/1164/3238737531.mp4"


def test_convert_record_uses_gt_for_val_rows(tmp_path: Path):
    dataset_root = tmp_path / "NeXTVideo"
    video_path = dataset_root / "NExTVideo" / "1164" / "3238737531.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"fake-video")

    record = convert_record(
        _nextvideo_row(answer=None, gt="b"),
        dataset_root=dataset_root,
        split="val",
        line_no=3,
        verify_video=True,
    )

    assert record is not None
    assert record["answer"] == "B"
    assert record["metadata"]["id"] == "nextvideo_val_000003"


def test_convert_record_skips_missing_video_when_verification_enabled(tmp_path: Path):
    record = convert_record(
        _nextvideo_row(answer="A"),
        dataset_root=tmp_path / "NeXTVideo",
        split="train",
        line_no=1,
        verify_video=True,
    )

    assert record is None


def test_load_jsonl_reports_bad_line_with_context(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(_nextvideo_row()) + "\n{bad json}\n", encoding="utf-8")

    try:
        load_jsonl(path)
    except SystemExit as exc:
        assert "bad.jsonl:2" in str(exc)
    else:
        raise AssertionError("load_jsonl should reject malformed JSONL")
