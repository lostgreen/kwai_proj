#!/usr/bin/env python3
"""
Expand raw TimeLens records into query-level temporal-grounding rollout input.

Each source video can yield multiple TG questions, one per event query/span.
The prompt/answer style is aligned with proxy_data/temporal_grounding/build_dataset.py.

Usage:
    python proxy_data/data_curation/timelens_100k/build_tg_rollout_dataset.py \
        --input proxy_data/data_curation/results/timelens_100k_short/short_pool_raw.jsonl \
        --output proxy_data/data_curation/results/timelens_100k_short/tg_rollout_input.jsonl \
        --video-root /m2v_intern/xuboshen/zgw/data/VideoProxyMixed/TimeLens-100K/video_shards
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from proxy_data.temporal_grounding.build_dataset import (  # noqa: E402
    POST_PROMPT,
    PRE_PROMPT,
    format_answer_text,
)


def duration_bucket(duration: float) -> str:
    if duration < 15:
        return "[0,15)"
    if duration < 30:
        return "[15,30)"
    if duration < 45:
        return "[30,45)"
    return "[45,60]"


def build_prompt(sentence: str) -> str:
    return f'<video>\n{PRE_PROMPT}"{sentence}". {POST_PROMPT}'


def load_jsonl(path: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def resolve_video_path(record: dict[str, Any], video_root: str | None) -> tuple[str, str]:
    videos = record.get("videos") or []
    if videos:
        first = str(videos[0])
        return first, Path(first).stem

    video_rel = str(record.get("video_path") or "")
    if not video_rel:
        raise ValueError("record missing both videos and video_path")
    if video_root:
        return str(Path(video_root) / video_rel), Path(video_rel).stem
    return video_rel, Path(video_rel).stem


def iter_valid_spans(event: dict[str, Any]):
    for span in event.get("span") or []:
        if not isinstance(span, list) or len(span) != 2:
            continue
        try:
            start = float(span[0])
            end = float(span[1])
        except (TypeError, ValueError):
            continue
        if start < 0 or end <= start:
            continue
        yield start, end


def to_query_records(record: dict[str, Any], video_root: str | None) -> list[dict[str, Any]]:
    source = str(record.get("source") or "unknown")
    duration = float(record.get("duration") or 0.0)
    video_path, clip_key = resolve_video_path(record, video_root)
    video_rel = str(record.get("video_path") or Path(video_path).name)
    events = record.get("events") or []

    query_records: list[dict[str, Any]] = []
    for event_idx, event in enumerate(events):
        query_text = str(event.get("query") or "").strip()
        if not query_text:
            continue
        valid_spans = list(iter_valid_spans(event))
        if not valid_spans:
            continue

        for span_idx, (start, end) in enumerate(valid_spans):
            query_id = f"{clip_key}::q{event_idx}"
            if span_idx > 0:
                query_id = f"{query_id}_s{span_idx}"

            prompt = build_prompt(query_text)
            answer = format_answer_text(start, end)
            metadata = {
                "id": query_id,
                "clip_key": clip_key,
                "video_id": clip_key,
                "video_uid": clip_key,
                "video_path": video_rel,
                "source": source,
                "duration": duration,
                "duration_bucket": duration_bucket(duration),
                "timestamp": [start, end],
                "sentence": query_text,
                "query": query_text,
                "query_idx": event_idx,
                "span_idx": span_idx,
                "n_queries_in_video": len(events),
                "dataset": "TimeLens-100K",
            }
            query_records.append(
                {
                    "messages": [{"role": "user", "content": prompt}],
                    "prompt": prompt,
                    "answer": answer,
                    "videos": [video_path],
                    "data_type": "video",
                    "problem_type": "temporal_grounding",
                    "metadata": metadata,
                }
            )
    return query_records


def main():
    parser = argparse.ArgumentParser(description="Build query-level TimeLens TG rollout dataset")
    parser.add_argument("--input", required=True, help="Raw TimeLens short-pool JSONL")
    parser.add_argument("--output", required=True, help="Output query-level TG JSONL")
    parser.add_argument("--video-root", default=None, help="Video root joined with raw video_path")
    parser.add_argument("--max-videos", type=int, default=0, help="Limit videos for quick pilot (0 = all)")
    parser.add_argument("--max-queries", type=int, default=0, help="Limit total expanded queries (0 = all)")
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    if args.max_videos > 0:
        rows = rows[: args.max_videos]

    output_rows: list[dict[str, Any]] = []
    source_counter: Counter[str] = Counter()
    query_counter_by_source: defaultdict[str, int] = defaultdict(int)
    skipped_videos = 0

    for row in rows:
        query_rows = to_query_records(row, args.video_root)
        if not query_rows:
            skipped_videos += 1
            continue
        source = str(row.get("source") or "unknown")
        source_counter[source] += 1
        query_counter_by_source[source] += len(query_rows)
        output_rows.extend(query_rows)
        if args.max_queries > 0 and len(output_rows) >= args.max_queries:
            output_rows = output_rows[: args.max_queries]
            break

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Input videos          : {len(rows)}")
    print(f"Skipped videos        : {skipped_videos}")
    print(f"Expanded query items  : {len(output_rows)}")
    print(f"Output                : {out_path}")
    print("\nPer source:")
    for source in sorted(source_counter.keys()):
        print(
            f"  {source:<20} videos={source_counter[source]:>6} "
            f"queries={query_counter_by_source[source]:>6}"
        )


if __name__ == "__main__":
    main()
