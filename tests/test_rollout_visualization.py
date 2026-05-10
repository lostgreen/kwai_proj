from __future__ import annotations

import sys
import types
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from video_proxy.insights.rollout_viewer.server import RolloutStore


class _FakeFrame:
    def asnumpy(self):
        import numpy as np

        return np.zeros((8, 8, 3), dtype=np.uint8)


class _FakeVideoReader:
    requested_indices: list[int] = []

    def __init__(self, _path: str):
        self._fps = 4.0
        self._total = 13

    def __len__(self) -> int:
        return self._total

    def get_avg_fps(self) -> float:
        return self._fps

    def __getitem__(self, idx: int) -> _FakeFrame:
        self.requested_indices.append(idx)
        return _FakeFrame()


def test_rollout_video_frame_strip_samples_at_one_fps(monkeypatch, tmp_path: Path):
    fake_decord = types.SimpleNamespace(
        VideoReader=_FakeVideoReader,
        bridge=types.SimpleNamespace(set_bridge=lambda _name: None),
    )
    monkeypatch.setitem(sys.modules, "decord", fake_decord)
    _FakeVideoReader.requested_indices = []

    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"not a real video; fake decord handles it")
    store = RolloutStore(root=tmp_path)

    frames = store._video_file_to_frames(video_path, max_frames=10)

    assert len(frames) == 4
    assert _FakeVideoReader.requested_indices == [0, 4, 8, 12]


def test_choice_meta_uses_answer_tag_choice_letter(tmp_path: Path):
    store = RolloutStore(root=tmp_path)
    detail = {
        "ground_truth": "<answer>B</answer>",
        "attempts": [
            {"response": "Reasoning text. <answer>C</answer>"},
        ],
    }

    meta = store._build_choice_meta(detail)

    assert meta["gt_letter"] == "B"
    assert detail["attempts"][0]["pred_letter"] == "C"


def test_store_loads_trainer_rollout_jsonl_contract(tmp_path: Path):
    rollout_dir = tmp_path / "rollouts"
    rollout_dir.mkdir()
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake video")
    record = {
        "phase": "train",
        "step": 12,
        "uid": "sample-1",
        "problem_type": "event_logic_predict_next",
        "data_type": "event_logic",
        "problem_id": "p-1",
        "problem": "reserved problem text",
        "prompt": "<video>\nA. chop onions\nB. pour oil",
        "response": "The visual next step is <answer>B</answer>",
        "ground_truth": "B",
        "reward": 1.0,
        "video_paths": [str(video_path)],
        "image_paths": [],
        "video_nframes": [8],
        "video_fps": [2.0],
        "multi_modal_source": {"videos": [str(video_path)], "video_fps": [2.0]},
    }
    (rollout_dir / "step_000012.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

    store = RolloutStore(root=tmp_path)
    summary = store.load(str(rollout_dir), None)
    detail = store.get_group_detail("sample-1", text_only=True)

    assert summary["group_count"] == 1
    assert summary["sample_count"] == 1
    assert detail["step_key"] == "train:12"
    assert detail["prompt"] == record["prompt"]
    assert detail["ground_truth"] == "B"
    assert detail["choice_meta"]["gt_letter"] == "B"
    assert detail["attempts"][0]["pred_letter"] == "B"
    assert detail["frame_strip"] == []


def test_store_loads_rollout_media_sidecar_contract(tmp_path: Path):
    rollout_dir = tmp_path / "rollouts"
    rollout_dir.mkdir()
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake video")
    media_record = {
        "phase": "train",
        "step": 12,
        "index": 0,
        "uid": "sample-1",
        "problem_id": "p-1",
        "problem_type": "add",
        "video_paths": [str(video_path)],
        "image_paths": [],
        "video_nframes": [8],
        "video_fps": [2.0],
        "multi_modal_source": {"videos": [str(video_path), str(video_path)], "video_fps": [2.0, 2.0]},
    }
    record = {
        "phase": "train",
        "step": 12,
        "uid": "sample-1",
        "problem_type": "add",
        "data_type": "event_logic",
        "problem_id": "p-1",
        "problem": "reserved problem text",
        "prompt": "<video>\nA. chop onions\nB. pour oil",
        "response": "The visual next step is <answer>B</answer>",
        "ground_truth": "B",
        "reward": 1.0,
        "media_ref": {"file": "step_000012.media.jsonl", "index": 0},
    }
    (rollout_dir / "step_000012.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    (rollout_dir / "step_000012.media.jsonl").write_text(
        json.dumps(media_record) + "\n",
        encoding="utf-8",
    )

    store = RolloutStore(root=tmp_path)
    summary = store.load(str(rollout_dir), None)
    detail = store.get_group_detail("sample-1", text_only=True)

    assert summary["group_count"] == 1
    assert summary["sample_count"] == 1
    assert detail["video_paths"] == [str(video_path)]
    assert detail["video_nframes"] == [8]
    assert detail["video_fps"] == [2.0]
    assert detail["multi_modal_source"] == media_record["multi_modal_source"]
    assert detail["choice_meta"]["option_type"] == "video"


def test_text_only_group_detail_does_not_extract_frames(monkeypatch, tmp_path: Path):
    store = RolloutStore(root=tmp_path)
    store.groups["sample-1"] = {
        "uid": "sample-1",
        "step": 1,
        "phase": "train",
        "step_key": "train:1",
        "step_label": "Step 1",
        "problem_type": "llava_mcq",
        "prompt": "A. one\nB. two",
        "ground_truth": "A",
        "attempts": [{"reward": 0.0, "response": "<answer>A</answer>"}],
        "mean_reward": 0.0,
    }

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("frame extraction should be lazy")

    monkeypatch.setattr(store, "_get_frame_strip", fail_if_called)

    detail = store.get_group_detail("sample-1", text_only=True)

    assert detail["frame_strip"] == []


def test_frontend_requests_text_only_group_detail_before_lazy_frames():
    html = (REPO_ROOT / "video_proxy" / "insights" / "rollout_viewer" / "index.html").read_text(
        encoding="utf-8"
    )

    assert "text_only=1" in html
    assert "/frames?max_frames=200" in html


def test_trainer_rollout_writer_includes_dataset_metadata():
    trainer_source = (REPO_ROOT / "verl" / "trainer" / "ray_trainer.py").read_text(encoding="utf-8")

    assert 'metadata = batch.non_tensor_batch.get("metadata"' in trainer_source
    assert '"metadata": _to_jsonable(metadata[i])' in trainer_source


def test_trainer_rollout_writer_splits_media_sidecar_from_main_jsonl():
    trainer_source = (REPO_ROOT / "verl" / "trainer" / "ray_trainer.py").read_text(encoding="utf-8")

    assert "_rollout_media_filepath" in trainer_source
    assert "media_ref" in trainer_source
    assert "media_record" in trainer_source
    assert "media_line_offset" in trainer_source
    assert '"batch_index": i' in trainer_source
    assert '"multi_modal_source": mm_source_json' not in trainer_source
    assert '"video_paths": video_paths' not in trainer_source
