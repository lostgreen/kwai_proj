#!/usr/bin/env python3
"""
Explore short-video candidates in TimeLens-100K.

This script is meant for quick inspection before running the heavier VLM
screening pipeline. It reports how many <=60s candidates exist, what sources
they come from, what their event structure looks like, and can optionally
export a 3k-style subset for downstream use.

Example:
    python explore_short_videos.py \
        --input /path/to/timelens-100k.jsonl \
        --config ../configs/timelens_100k.yaml \
        --video-root /path/to/video_shards \
        --total 3000 \
        --balanced-total
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from sample_per_source import to_unified_record
from text_filter import count_valid_events, load_config


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = (SCRIPT_DIR.parent / "results" / "timelens_100k_short").resolve()


def flatten_filter_config(config_path: str | None) -> dict:
    if not config_path:
        return {}
    raw_cfg = load_config(config_path)
    flattened: dict = {}
    if "text_filter" in raw_cfg:
        flattened.update(raw_cfg["text_filter"])
    if "domain_balance" in raw_cfg:
        flattened.update({"priority_domains": raw_cfg["domain_balance"].get("priority_domains", [])})
    if not flattened:
        flattened = raw_cfg
    return flattened


def build_filter_cfg(args: argparse.Namespace) -> dict:
    cfg = {
        "min_duration_sec": 0.0,
        "max_duration_sec": 60.0,
        "min_events": 5,
        "min_event_span": 2.0,
    }
    cfg.update(flatten_filter_config(args.config))
    if args.min_duration is not None:
        cfg["min_duration_sec"] = args.min_duration
    else:
        cfg["min_duration_sec"] = 0.0
    if args.max_duration is not None:
        cfg["max_duration_sec"] = args.max_duration
    else:
        cfg["max_duration_sec"] = 60.0
    if args.min_events is not None:
        cfg["min_events"] = args.min_events
    if args.min_event_span is not None:
        cfg["min_event_span"] = args.min_event_span
    return cfg


def duration_bucket(duration: float) -> str:
    if duration < 15:
        return "[0,15)"
    if duration < 30:
        return "[15,30)"
    if duration < 45:
        return "[30,45)"
    return "[45,60]"


def valid_event_bucket(n_events: int) -> str:
    if n_events <= 4:
        return "<=4"
    if n_events <= 6:
        return "5-6"
    if n_events <= 8:
        return "7-8"
    if n_events <= 12:
        return "9-12"
    return "13+"


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = ratio * (len(ordered) - 1)
    low = int(idx)
    high = min(low + 1, len(ordered) - 1)
    if low == high:
        return ordered[low]
    frac = idx - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def summary_stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "min": 0.0, "p25": 0.0, "median": 0.0, "mean": 0.0, "p75": 0.0, "max": 0.0}
    ordered = sorted(values)
    total = sum(ordered)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p25": percentile(ordered, 0.25),
        "median": percentile(ordered, 0.5),
        "mean": total / len(ordered),
        "p75": percentile(ordered, 0.75),
        "max": ordered[-1],
    }


def rank_pool(pool: list[dict], rng: random.Random) -> list[dict]:
    ranked = list(pool)
    rng.shuffle(ranked)
    ranked.sort(
        key=lambda item: (
            -int(item.get("_n_valid_events", 0)),
            safe_float(item.get("duration"), 0.0),
            item.get("video_path") or "",
        )
    )
    return ranked


def allocate_balanced_total(by_source: dict[str, list[dict]], total: int) -> list[dict]:
    sources = sorted(by_source.keys())
    if total <= 0 or not sources:
        return []

    base = total // len(sources)
    quota = {source: base for source in sources}
    remainder = total - base * len(sources)
    by_size = sorted(sources, key=lambda source: len(by_source[source]), reverse=True)
    for idx in range(remainder):
        quota[by_size[idx % len(by_size)]] += 1

    selected: list[dict] = []
    taken = {source: 0 for source in sources}
    overflow = 0

    for source in sources:
        pool = by_source[source]
        n_take = min(quota[source], len(pool))
        selected.extend(pool[:n_take])
        taken[source] = n_take
        overflow += quota[source] - n_take

    if overflow > 0:
        for source in by_size:
            if overflow <= 0:
                break
            pool = by_source[source]
            remaining = len(pool) - taken[source]
            if remaining <= 0:
                continue
            extra = min(overflow, remaining)
            selected.extend(pool[taken[source]:taken[source] + extra])
            taken[source] += extra
            overflow -= extra

    return selected


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def strip_internal_fields(row: dict) -> dict:
    clean = dict(row)
    clean.pop("_n_valid_events", None)
    return clean


def build_preview(by_source: dict[str, list[dict]], preview_per_source: int) -> dict[str, list[dict]]:
    preview: dict[str, list[dict]] = {}
    for source in sorted(by_source.keys()):
        rows = []
        for item in by_source[source][:preview_per_source]:
            events = item.get("events") or []
            rows.append(
                {
                    "video_path": item.get("video_path"),
                    "duration": safe_float(item.get("duration"), 0.0),
                    "n_events": len(events),
                    "n_valid_events": int(item.get("_n_valid_events", 0)),
                    "event_queries": [event.get("query", "") for event in events[:3]],
                    "event_spans": [event.get("span", []) for event in events[:3]],
                }
            )
        preview[source] = rows
    return preview


def main():
    parser = argparse.ArgumentParser(description="Explore <=60s TimeLens candidates")
    parser.add_argument("--input", required=True, help="timelens-100k.jsonl 路径")
    parser.add_argument("--config", default=None, help="可选 TimeLens YAML 配置")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="分析输出目录")
    parser.add_argument("--video-root", default=None, help="导出 unified JSONL 时使用的视频根目录")
    parser.add_argument("--min-duration", type=float, default=None, help="最小时长，默认 0")
    parser.add_argument("--max-duration", type=float, default=None, help="最大时长，默认 60")
    parser.add_argument("--min-events", type=int, default=None, help="最少 valid events")
    parser.add_argument("--min-event-span", type=float, default=None, help="event span 最短秒数")
    parser.add_argument("--total", type=int, default=0, help="导出候选总量；0 = 仅导出完整池")
    parser.add_argument("--balanced-total", action="store_true", help="当 --total > 0 时，按 source 尽量均衡导出")
    parser.add_argument("--preview-per-source", type=int, default=3, help="每个 source 预览条数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    cfg = build_filter_cfg(args)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    total_rows = 0
    raw_in_window = 0
    raw_pass_events = 0
    raw_by_source: Counter[str] = Counter()
    raw_duration_buckets: Counter[str] = Counter()
    raw_event_buckets: Counter[str] = Counter()

    best_by_video: dict[str, dict] = {}

    print(f"Reading {args.input} ...")
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            total_rows += 1

            duration = safe_float(sample.get("duration"), 0.0)
            if duration < cfg["min_duration_sec"] or duration > cfg["max_duration_sec"]:
                continue

            raw_in_window += 1
            source = sample.get("source", "unknown")
            raw_by_source[source] += 1
            raw_duration_buckets[duration_bucket(duration)] += 1

            n_valid_events = count_valid_events(sample.get("events") or [], cfg["min_event_span"])
            raw_event_buckets[valid_event_bucket(n_valid_events)] += 1
            if n_valid_events < cfg["min_events"]:
                continue

            raw_pass_events += 1
            sample_copy = dict(sample)
            sample_copy["_n_valid_events"] = n_valid_events
            video_path = sample_copy.get("video_path") or f"missing::{total_rows}"
            existing = best_by_video.get(video_path)
            if existing is None:
                best_by_video[video_path] = sample_copy
                continue

            existing_valid = int(existing.get("_n_valid_events", 0))
            if n_valid_events > existing_valid:
                best_by_video[video_path] = sample_copy
                continue
            if n_valid_events == existing_valid and duration < safe_float(existing.get("duration"), 0.0):
                best_by_video[video_path] = sample_copy

    filtered_pool = list(best_by_video.values())
    ranked_pool = rank_pool(filtered_pool, rng)

    by_source_ranked: dict[str, list[dict]] = defaultdict(list)
    for item in ranked_pool:
        by_source_ranked[item.get("source", "unknown")].append(item)

    selected_pool: list[dict]
    if args.total > 0:
        if args.balanced_total:
            selected_pool = allocate_balanced_total(by_source_ranked, args.total)
        else:
            selected_pool = ranked_pool[:args.total]
    else:
        selected_pool = ranked_pool

    pool_durations = [safe_float(item.get("duration"), 0.0) for item in ranked_pool]
    pool_valid_events = [int(item.get("_n_valid_events", 0)) for item in ranked_pool]

    source_rows = []
    for source in sorted(by_source_ranked.keys()):
        items = by_source_ranked[source]
        durations = [safe_float(item.get("duration"), 0.0) for item in items]
        valid_events = [int(item.get("_n_valid_events", 0)) for item in items]
        source_rows.append(
            {
                "source": source,
                "count": len(items),
                "duration_sec": summary_stats(durations),
                "valid_events": summary_stats([float(x) for x in valid_events]),
            }
        )

    selected_by_source: Counter[str] = Counter(item.get("source", "unknown") for item in selected_pool)
    summary = {
        "input_path": str(Path(args.input).resolve()),
        "config_path": str(Path(args.config).resolve()) if args.config else None,
        "output_dir": str(output_dir),
        "filter": cfg,
        "stats": {
            "total_rows": total_rows,
            "raw_in_duration_window": raw_in_window,
            "raw_pass_event_threshold": raw_pass_events,
            "after_dedup": len(ranked_pool),
            "selected_total": len(selected_pool),
            "requested_total": args.total,
            "can_meet_requested_total": (len(ranked_pool) >= args.total) if args.total > 0 else None,
        },
        "distribution": {
            "raw_by_source": dict(raw_by_source),
            "raw_duration_buckets": dict(raw_duration_buckets),
            "raw_valid_event_buckets": dict(raw_event_buckets),
            "pool_duration_sec": summary_stats(pool_durations),
            "pool_valid_events": summary_stats([float(x) for x in pool_valid_events]),
            "pool_by_source": source_rows,
            "selected_by_source": dict(selected_by_source),
        },
        "outputs": {
            "summary_json": str((output_dir / "summary.json").resolve()),
            "preview_json": str((output_dir / "preview_samples.json").resolve()),
            "pool_raw_jsonl": str((output_dir / "short_pool_raw.jsonl").resolve()),
            "selected_raw_jsonl": str((output_dir / "short_selected_raw.jsonl").resolve()) if args.total > 0 else None,
            "selected_unified_jsonl": str((output_dir / "short_selected_unified.jsonl").resolve())
            if args.total > 0 and args.video_root
            else None,
        },
    }

    preview = {
        "filter": cfg,
        "preview_per_source": args.preview_per_source,
        "sources": build_preview(by_source_ranked, args.preview_per_source),
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(output_dir / "preview_samples.json", "w", encoding="utf-8") as f:
        json.dump(preview, f, indent=2, ensure_ascii=False)

    write_jsonl(output_dir / "short_pool_raw.jsonl", [strip_internal_fields(row) for row in ranked_pool])

    if args.total > 0:
        write_jsonl(output_dir / "short_selected_raw.jsonl", [strip_internal_fields(row) for row in selected_pool])
        if args.video_root:
            unified_rows = [to_unified_record(strip_internal_fields(row), args.video_root) for row in selected_pool]
            write_jsonl(output_dir / "short_selected_unified.jsonl", unified_rows)

    print("\n=== Short-video summary ===")
    print(f"Input rows                 : {total_rows}")
    print(f"In duration window         : {raw_in_window}")
    print(f"Pass event threshold       : {raw_pass_events}")
    print(f"After dedup                : {len(ranked_pool)}")
    if args.total > 0:
        print(f"Selected for export        : {len(selected_pool)} / requested {args.total}")
    print("\nPool by source:")
    for row in source_rows:
        print(f"  {row['source']:<20} {row['count']:>6}")

    print(f"\nWrote summary              : {output_dir / 'summary.json'}")
    print(f"Wrote preview              : {output_dir / 'preview_samples.json'}")
    print(f"Wrote pool raw JSONL       : {output_dir / 'short_pool_raw.jsonl'}")
    if args.total > 0:
        print(f"Wrote selected raw JSONL   : {output_dir / 'short_selected_raw.jsonl'}")
        if args.video_root:
            print(f"Wrote selected unified JSONL: {output_dir / 'short_selected_unified.jsonl'}")


if __name__ == "__main__":
    main()
