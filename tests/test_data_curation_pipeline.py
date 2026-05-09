import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from video_proxy.data.pipelines.data_curation.curation.duration_filter import curate_dataset, select_records
from video_proxy.data.pipelines.data_curation.curation.io import read_jsonl
from video_proxy.data.pipelines.data_curation.curation.sources import load_records, to_unified_record


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def test_et_duration_filter_outputs_unified_schema(tmp_path):
    source = tmp_path / "et.json"
    video_root = tmp_path / "videos"
    write_json(
        source,
        [
            {
                "task": "slc",
                "source": "coin",
                "video": "coin/good.mp4",
                "duration": 80,
                "tgt": [1, 2],
            },
            {
                "task": "slc",
                "source": "coin",
                "video": "coin/short.mp4",
                "duration": 10,
                "tgt": [1, 2, 3, 4],
            },
        ],
    )

    summary = curate_dataset(
        dataset="et_instruct_164k",
        input_path=source,
        output_dir=tmp_path / "out",
        video_root=video_root,
        min_duration=60,
        max_duration=240,
    )

    assert summary["kept"] == 1
    records = read_jsonl(tmp_path / "out" / "duration_keep.jsonl")
    assert len(records) == 1
    assert records[0]["videos"] == [str(video_root / "coin/good.mp4")]
    assert records[0]["metadata"]["clip_key"] == "good"
    assert records[0]["metadata"]["clip_duration"] == 80
    assert records[0]["dataset"] == "ET-Instruct-164K"
    assert records[0]["_et_raw"]["task"] == "slc"


def test_timelens_duration_filter_outputs_compat_screen_keep(tmp_path):
    source = tmp_path / "timelens.jsonl"
    video_root = tmp_path / "video_shards"
    write_jsonl(
        source,
        [
            {
                "source": "hirest_step",
                "video_path": "hirest_step/good.mp4",
                "duration": 59,
                "events": [{"query": "a", "span": [[0, 5]]}],
            },
            {
                "source": "hirest_step",
                "video_path": "hirest_step/keep.mp4",
                "duration": 60,
                "events": [{"query": "b", "span": [[0, 5]]}],
            },
        ],
    )

    summary = curate_dataset(
        dataset="timelens_100k",
        input_path=source,
        output_dir=tmp_path / "out",
        video_root=video_root,
        min_duration=60,
        max_duration=240,
        write_screen_keep=True,
    )

    assert summary["total"] == 2
    assert summary["kept"] == 1
    duration_keep = read_jsonl(tmp_path / "out" / "duration_keep.jsonl")
    screen_keep = read_jsonl(tmp_path / "out" / "screen_keep.jsonl")
    assert duration_keep == screen_keep
    assert screen_keep[0]["videos"] == [str(video_root / "hirest_step/keep.mp4")]
    assert screen_keep[0]["dataset"] == "TimeLens-100K"
    assert screen_keep[0]["_tl_raw"]["n_events"] == 1


def test_sampling_is_deterministic_and_can_balance_sources():
    records = [
        {"source": "a", "duration": 80, "video": f"a/{idx}.mp4"}
        for idx in range(4)
    ] + [
        {"source": "b", "duration": 90, "video": f"b/{idx}.mp4"}
        for idx in range(4)
    ]

    selected = select_records(
        records,
        dataset="et_instruct_164k",
        min_duration=60,
        max_duration=240,
        per_source=0,
        target_total=4,
        balanced_total=True,
        seed=7,
    )

    counts = {}
    for item in selected:
        counts[item["source"]] = counts.get(item["source"], 0) + 1
    assert counts == {"a": 2, "b": 2}

    selected_again = select_records(
        records,
        dataset="et_instruct_164k",
        min_duration=60,
        max_duration=240,
        per_source=0,
        target_total=4,
        balanced_total=True,
        seed=7,
    )
    assert selected == selected_again


def test_source_adapters_load_and_convert_records(tmp_path):
    et_path = tmp_path / "et.json"
    tl_path = tmp_path / "tl.jsonl"
    write_json(et_path, [{"source": "coin", "video": "coin/x.mp4", "duration": 70}])
    write_jsonl(tl_path, [{"source": "queryd", "video_path": "queryd/y.mp4", "duration": 71}])

    et = load_records("et_instruct_164k", et_path)
    tl = load_records("timelens_100k", tl_path)

    assert to_unified_record("et_instruct_164k", et[0], "/videos")["videos"] == ["/videos/coin/x.mp4"]
    assert to_unified_record("timelens_100k", tl[0], "/videos")["videos"] == ["/videos/queryd/y.mp4"]
