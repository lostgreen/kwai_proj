"""
TimeLens-100K text filter

Usage:
    # Dry run (stats only, no output)
    python text_filter.py --input /path/to/timelens-100k.jsonl --dry-run

    # Filter and write output
    python text_filter.py \
        --input /path/to/timelens-100k.jsonl \
        --output results/passed_timelens.jsonl \
        --config ../../configs/timelens_100k.yaml

Format (input):
    {
        "source": "cosmo_cap",
        "video_path": "cosmo_cap/BVs52yd-RUQ.mp4",
        "duration": 117.4,
        "events": [{"query": "...", "span": [[0.0, 5.0]]}, ...]
    }
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def count_valid_events(events: list, min_span: float) -> int:
    """Count events whose span width is >= min_span."""
    count = 0
    for ev in events:
        spans = ev.get("span", [])
        for seg in spans:
            if len(seg) == 2 and (seg[1] - seg[0]) >= min_span:
                count += 1
                break  # one valid span per event is enough
    return count


def passes_filters(sample: dict, cfg: dict) -> tuple[bool, str]:
    """Return (pass, reason). reason is non-empty when filtered out."""
    dur = sample.get("duration", 0)
    if dur < cfg["min_duration_sec"]:
        return False, f"duration {dur:.1f}s < {cfg['min_duration_sec']}s"
    if dur > cfg["max_duration_sec"]:
        return False, f"duration {dur:.1f}s > {cfg['max_duration_sec']}s"

    events = sample.get("events", [])
    n_valid = count_valid_events(events, cfg.get("min_event_span", 2.0))
    if n_valid < cfg["min_events"]:
        return False, f"only {n_valid} valid events (need {cfg['min_events']})"

    return True, ""


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TimeLens-100K text filter")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--min-duration", type=float, default=60.0)
    parser.add_argument("--max-duration", type=float, default=240.0)
    parser.add_argument("--min-events", type=int, default=5)
    parser.add_argument("--min-event-span", type=float, default=2.0)
    parser.add_argument("--max-per-domain", type=int, default=3000)
    parser.add_argument(
        "--priority-domains",
        nargs="+",
        default=["hirest_step", "hirest_grounding", "hirest"],
        help="Domains exempt from max-per-domain cap",
    )
    args = parser.parse_args()

    # Build config dict (YAML as base, CLI args override)
    cfg: dict = {
        "min_duration_sec": args.min_duration,
        "max_duration_sec": args.max_duration,
        "min_events": args.min_events,
        "min_event_span": args.min_event_span,
        "max_per_domain": args.max_per_domain,
        "priority_domains": args.priority_domains,
    }
    if args.config:
        raw_cfg = load_config(args.config)
        # Flatten nested YAML: extract text_filter + domain_balance sections
        file_cfg: dict = {}
        if "text_filter" in raw_cfg:
            file_cfg.update(raw_cfg["text_filter"])
        if "domain_balance" in raw_cfg:
            db = raw_cfg["domain_balance"]
            file_cfg["max_per_domain"] = db.get("max_per_domain", 3000)
            file_cfg["priority_domains"] = db.get("priority_domains", [])
        # Fall back: if YAML is already flat, use it directly
        if not file_cfg:
            file_cfg = raw_cfg
        # YAML as base, CLI overrides (only override if user explicitly set)
        cli_defaults = {
            "min_duration_sec": 60.0, "max_duration_sec": 240.0,
            "min_events": 5, "min_event_span": 2.0,
            "max_per_domain": 3000,
            "priority_domains": ["hirest_step", "hirest_grounding", "hirest"],
        }
        merged = dict(file_cfg)
        for k, v in cfg.items():
            if v != cli_defaults.get(k):  # user explicitly changed this CLI arg
                merged[k] = v
        cfg = merged

    # --- Pass 1: basic filters + dedup by video_path (keep most events) ---
    video_best: dict[str, dict] = {}  # video_path -> best sample
    total = skipped_dur = skipped_events = 0

    print(f"Reading {args.input} …")
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            total += 1

            ok, reason = passes_filters(sample, cfg)
            if not ok:
                if "duration" in reason:
                    skipped_dur += 1
                else:
                    skipped_events += 1
                continue

            vp = sample["video_path"]
            existing = video_best.get(vp)
            n_new = count_valid_events(sample["events"], cfg.get("min_event_span", 2.0))
            if existing is None:
                video_best[vp] = sample
            else:
                n_old = count_valid_events(existing["events"], cfg.get("min_event_span", 2.0))
                if n_new > n_old:
                    video_best[vp] = sample

    passed_dedup = list(video_best.values())
    print(f"\n--- Filter stats ---")
    print(f"Total input       : {total:>8,}")
    print(f"Skipped (duration): {skipped_dur:>8,}")
    print(f"Skipped (events)  : {skipped_events:>8,}")
    print(f"After dedup       : {len(passed_dedup):>8,}")

    # --- Pass 2: domain cap ---
    domain_counts: dict[str, int] = defaultdict(int)
    passed_cap: list[dict] = []
    priority_set = set(cfg["priority_domains"])

    # Sort so priority domains are preserved first; then by event count desc
    passed_dedup.sort(
        key=lambda s: (
            0 if s.get("source", "") in priority_set else 1,
            -count_valid_events(s["events"], cfg.get("min_event_span", 2.0)),
        )
    )

    for sample in passed_dedup:
        domain = sample.get("source", "unknown")
        if domain not in priority_set and domain_counts[domain] >= cfg["max_per_domain"]:
            continue
        domain_counts[domain] += 1
        passed_cap.append(sample)

    # --- Summary ---
    print(f"After domain cap  : {len(passed_cap):>8,}")
    print("\n--- Domain breakdown (after cap) ---")
    by_domain: dict[str, list] = defaultdict(list)
    for s in passed_cap:
        by_domain[s.get("source", "unknown")].append(s)
    for domain, items in sorted(by_domain.items(), key=lambda x: -len(x[1])):
        flag = " [priority]" if domain in priority_set else ""
        print(f"  {domain:<25} {len(items):>5}{flag}")

    if args.dry_run:
        print("\n[dry-run] No file written.")
        return

    # --- Write output ---
    if not args.output:
        print("No --output specified, skipping write.")
        return

    # 添加溯源元数据
    origin_meta = {
        "dataset": "TimeLens-100K",
        "source_file": str(Path(args.input).resolve()),
        "filter_config": str(Path(args.config).resolve()) if args.config else None,
        "filter_params": cfg,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for s in passed_cap:
            s["_origin"] = origin_meta
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(passed_cap)} records → {out_path}")

    # 写出筛选 summary
    summary = {
        "total_input": total,
        "skipped_duration": skipped_dur,
        "skipped_events": skipped_events,
        "after_dedup": len(passed_dedup),
        "after_domain_cap": len(passed_cap),
        "domain_breakdown": {d: len(items) for d, items in by_domain.items()},
        "config_used": cfg,
        "origin": origin_meta,
    }
    summary_path = out_path.parent / "filter_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"✅ summary → {summary_path}")


if __name__ == "__main__":
    main()
