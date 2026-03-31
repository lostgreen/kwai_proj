#!/usr/bin/env python3
"""
annotate_check.py — Standalone annotation quality audit for hierarchical segmentation.

Reads existing annotation JSONs, runs quality checks using a (potentially stronger)
VLM model, and writes revised annotations to a separate output directory.

Two-step check pipeline (mirrors the annotation pipeline):
  Step 1: merged_c  — L1+L2 simultaneous check using 1fps full-video frames
  Step 2: 3c        — L3 check using 2fps leaf-node frames (--l3-frames-dir)

Usage:
    # Step 1: Merged L1+L2 check (1fps frames, like annotation Step 1+2)
    python annotate_check.py \\
        --frames-dir frames/ \\
        --annotation-dir annotations/ \\
        --output-dir annotations_checked/ \\
        --levels merged_c \\
        --model gpt-4o \\
        --workers 4

    # Step 2: L3 check (2fps leaf-node frames, like annotation Step 3+4)
    python annotate_check.py \\
        --frames-dir frames/ \\
        --l3-frames-dir frames_l3/ \\
        --annotation-dir annotations_checked/ \\
        --output-dir annotations_checked/ \\
        --levels 3c \\
        --model gpt-4o \\
        --workers 4

    # Legacy: L2-only check (per-phase, old behavior)
    python annotate_check.py ... --levels 2c

    # Dry run (scan only, no API calls)
    python annotate_check.py ... --dry-run
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from annotate import (
    SYSTEM_PROMPT,
    _apply_l2_check_results,
    _apply_l3_check_results,
    _check_level2,
    _check_level3,
    _check_merged_l1l2,
    call_and_parse,
    count_extracted_frames,
    encode_frame_files,
    format_mmss,
    frame_stem_to_index,
    get_all_frame_files,
    get_frames_in_time_range,
    load_frame_meta,
    sample_uniform,
)


def _remove_orphaned_l3_results(
    l3_results: list[dict],
    valid_event_ids: set[int],
    valid_phase_ids: set[int] | None = None,
) -> list[dict]:
    """Remove L3 results whose parent no longer exists after L1+L2 check.

    Checks both parent_event_id (normal events) and parent_phase_id
    (eventless phases used as L3 parents).
    """
    cleaned = []
    for r in l3_results:
        peid = r.get("parent_event_id")
        ppid = r.get("parent_phase_id")
        # If linked to an event, check event validity
        if peid is not None:
            if peid in valid_event_ids:
                cleaned.append(r)
            continue
        # If linked directly to a phase (eventless leaf), check phase validity
        if ppid is not None and valid_phase_ids is not None:
            if ppid in valid_phase_ids:
                cleaned.append(r)
            continue
        # No parent link — keep (shouldn't happen, but safe)
        cleaned.append(r)
    return cleaned


def _remove_out_of_bounds_l3(
    l3_results: list[dict],
    l2_events: list[dict],
) -> list[dict]:
    """Remove L3 results whose timestamps fall outside their parent event's new boundaries."""
    event_bounds = {}
    for e in l2_events:
        eid = e.get("event_id")
        if eid is not None:
            event_bounds[eid] = (e.get("start_time", 0), e.get("end_time", float("inf")))

    cleaned = []
    for r in l3_results:
        peid = r.get("parent_event_id")
        if peid is not None and peid in event_bounds:
            ev_start, ev_end = event_bounds[peid]
            r_start = r.get("start_time", 0)
            r_end = r.get("end_time", 0)
            # Drop if L3 action is completely outside event bounds
            if r_end <= ev_start or r_start >= ev_end:
                continue
        cleaned.append(r)
    return cleaned


def check_clip(
    annotation_path: Path,
    frames_base: Path,
    output_dir: Path,
    levels: list[str],
    api_base: str, api_key: str, model: str,
    max_frames: int, resize_max_width: int, jpeg_quality: int,
    overwrite: bool,
    dry_run: bool,
    l3_frames_base: Path | None = None,
) -> dict:
    """
    Run quality checks on a single clip's annotation.

    Supports levels: "merged_c" (L1+L2 simultaneous), "2c" (L2 only, legacy),
    "3c" (L3 with optional per-event 2fps frames).

    Returns status dict: {clip_key, ok, error, skipped, stats}.
    """
    try:
        with open(annotation_path, encoding="utf-8") as f:
            ann = json.load(f)
    except Exception as e:
        return {"clip_key": annotation_path.stem, "ok": False,
                "error": f"failed to load: {e}", "skipped": False, "stats": {}}

    clip_key = ann.get("clip_key", annotation_path.stem)
    frame_dir = frames_base / clip_key

    if not frame_dir.exists():
        return {"clip_key": clip_key, "ok": False,
                "error": f"frame_dir not found: {frame_dir}", "skipped": False, "stats": {}}

    frame_meta = load_frame_meta(frame_dir)
    clip_duration = float(
        frame_meta.get("annotation_end_sec")
        or ann.get("clip_duration_sec")
        or count_extracted_frames(frame_dir)
    )

    combined_stats: dict[str, dict] = {}

    # ---- Merged L1+L2 Check ----
    if "merged_c" in levels:
        l1 = ann.get("level1")
        l2 = ann.get("level2")
        if l1 is None or l2 is None:
            return {"clip_key": clip_key, "ok": False,
                    "error": "level1+level2 required for merged check", "skipped": False, "stats": {}}

        already_checked = (
            l1.get("_check_stats") is not None and l2.get("_check_stats") is not None
        )
        if not overwrite and already_checked:
            combined_stats["l1"] = l1["_check_stats"]
            combined_stats["l2"] = l2["_check_stats"]
        elif dry_run:
            n_phases = len(l1.get("macro_phases", []))
            n_events = len(l2.get("events", []))
            combined_stats["l1"] = {"dry_run": True, "n_phases": n_phases}
            combined_stats["l2"] = {"dry_run": True, "n_events": n_events}
        else:
            checked_l1, checked_l2 = _check_merged_l1l2(
                frame_dir, clip_duration, l1, l2,
                summary=ann.get("summary", ""),
                topology_type=ann.get("topology_type", "procedural"),
                topology_confidence=float(ann.get("topology_confidence", 0.5)),
                api_base=api_base, api_key=api_key, model=model,
                max_frames=max_frames,
                resize_max_width=resize_max_width, jpeg_quality=jpeg_quality,
                global_phase_criterion=ann.get("global_phase_criterion", ""),
            )
            ann["level1"] = checked_l1
            ann["level2"] = checked_l2
            combined_stats["l1"] = checked_l1.get("_check_stats", {})
            combined_stats["l2"] = checked_l2.get("_check_stats", {})

            # Orphan cleanup: remove L3 results for deleted events/phases
            if ann.get("level3") is not None:
                valid_eids = {e.get("event_id") for e in checked_l2.get("events", [])}
                valid_pids = {p.get("phase_id") for p in checked_l1.get("macro_phases", [])}
                old_l3 = ann["level3"].get("grounding_results", [])

                # Step 1: remove orphans (parent deleted)
                cleaned = _remove_orphaned_l3_results(old_l3, valid_eids, valid_pids)
                # Step 2: remove out-of-bounds (parent boundary changed)
                cleaned = _remove_out_of_bounds_l3(cleaned, checked_l2.get("events", []))

                if len(cleaned) != len(old_l3):
                    ann["level3"]["grounding_results"] = cleaned
                    combined_stats["l3_orphans_removed"] = len(old_l3) - len(cleaned)

    # ---- Legacy L2-only Check ----
    if "2c" in levels:
        l1 = ann.get("level1")
        l2 = ann.get("level2")
        if l1 is None or l2 is None:
            return {"clip_key": clip_key, "ok": False,
                    "error": "level1+level2 required for L2 check", "skipped": False, "stats": {}}
        if not overwrite and l2.get("_check_stats") is not None:
            combined_stats["l2"] = l2["_check_stats"]
        elif dry_run:
            n_events = len(l2.get("events", []))
            combined_stats["l2"] = {"dry_run": True, "n_events": n_events}
        else:
            _, checked_l2 = _check_level2(
                frame_dir, clip_duration, l1, l2,
                api_base, api_key, model,
                max_frames, resize_max_width, jpeg_quality,
            )
            ann["level2"] = checked_l2
            combined_stats["l2"] = checked_l2.get("_check_stats", {})

            # Orphan cleanup: remove L3 results for deleted L2 events
            if ann.get("level3") is not None:
                valid_ids = {e.get("event_id") for e in checked_l2.get("events", [])}
                old_l3 = ann["level3"].get("grounding_results", [])
                cleaned = _remove_orphaned_l3_results(old_l3, valid_ids)
                if len(cleaned) != len(old_l3):
                    ann["level3"]["grounding_results"] = cleaned
                    combined_stats["l2_orphans_removed"] = len(old_l3) - len(cleaned)

    # ---- L3 Check ----
    if "3c" in levels:
        l2 = ann.get("level2")
        l3 = ann.get("level3")
        if l2 is None or l3 is None:
            if "merged_c" not in levels and "2c" not in levels:
                return {"clip_key": clip_key, "ok": False,
                        "error": "level2+level3 required for L3 check", "skipped": False, "stats": {}}
            # L3 doesn't exist but upper check was run — skip L3 check
            combined_stats["l3"] = {"skipped": True, "reason": "no level3 data"}
        elif not overwrite and l3.get("_check_stats") is not None:
            combined_stats["l3"] = l3["_check_stats"]
        elif dry_run:
            n_actions = len(l3.get("grounding_results", []))
            combined_stats["l3"] = {"dry_run": True, "n_actions": n_actions}
        else:
            _, checked_l3 = _check_level3(
                frame_dir, clip_duration, l2, l3,
                api_base, api_key, model,
                max_frames, resize_max_width, jpeg_quality,
                l3_base=l3_frames_base,
                clip_key_str=clip_key,
            )
            ann["level3"] = checked_l3
            combined_stats["l3"] = checked_l3.get("_check_stats", {})

    # Write output
    if not dry_run:
        ann["_audit_meta"] = {
            "audit_model": model,
            "audit_levels": levels,
            "audited_at": datetime.now(timezone.utc).isoformat(),
            "original_annotation": str(annotation_path),
        }
        out_file = output_dir / f"{clip_key}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(ann, f, ensure_ascii=False, indent=2)

    return {"clip_key": clip_key, "ok": True, "error": None, "skipped": False, "stats": combined_stats}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone quality audit for hierarchical annotations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--frames-dir", required=True,
                        help="Root directory of pre-extracted 1fps frames")
    parser.add_argument("--l3-frames-dir", default="",
                        help="Root directory of 2fps leaf-node frames for L3 check "
                             "(falls back to --frames-dir if not provided)")
    parser.add_argument("--annotation-dir", required=True,
                        help="Input directory with existing annotation JSONs")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for checked annotations (separate from input)")
    parser.add_argument("--levels", default="merged_c,3c",
                        help="Comma-separated checks to run: merged_c, 2c, 3c")
    parser.add_argument("--api-base", default="https://api.novita.ai/v3/openai",
                        help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", default="",
                        help="API key (prefers NOVITA_API_KEY env var, then OPENAI_API_KEY)")
    parser.add_argument("--model", default="pa/gmn-2.5-pr",
                        help="Model for quality review (can differ from annotation model)")
    parser.add_argument("--max-frames-per-call", type=int, default=32,
                        help="Max frames per API call")
    parser.add_argument("--resize-max-width", type=int, default=384,
                        help="Resize frames before upload; <=0 disables resizing")
    parser.add_argument("--jpeg-quality", type=int, default=60,
                        help="JPEG quality for recompressing frames before upload")
    parser.add_argument("--workers", type=int, default=2,
                        help="Parallel audit workers")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process at most N clips (0 = all)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-check even if already checked")
    parser.add_argument("--dry-run", action="store_true",
                        help="Scan and report without calling API")
    args = parser.parse_args()

    # Validate levels
    levels = [l.strip() for l in args.levels.split(",")]
    valid_levels = {"merged_c", "2c", "3c"}
    for l in levels:
        if l not in valid_levels:
            print(f"ERROR: unsupported level '{l}', must be one of {valid_levels}", file=sys.stderr)
            sys.exit(1)

    # Warn if both merged_c and 2c are used (redundant)
    if "merged_c" in levels and "2c" in levels:
        print("WARNING: both 'merged_c' and '2c' specified — merged_c already covers L2 check, "
              "2c will re-check L2 events per-phase after merged check", file=sys.stderr)

    api_key = args.api_key or os.environ.get("NOVITA_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""

    ann_dir = Path(args.annotation_dir)
    if not ann_dir.exists():
        print(f"ERROR: annotation-dir not found: {ann_dir}", file=sys.stderr)
        sys.exit(1)

    frames_base = Path(args.frames_dir)
    if not frames_base.exists():
        print(f"ERROR: frames-dir not found: {frames_base}", file=sys.stderr)
        sys.exit(1)

    l3_frames_base = Path(args.l3_frames_dir) if args.l3_frames_dir else None
    if l3_frames_base and not l3_frames_base.exists():
        print(f"WARNING: l3-frames-dir not found: {l3_frames_base}, "
              "L3 check will fall back to 1fps frames", file=sys.stderr)
        l3_frames_base = None

    ann_files = sorted(ann_dir.glob("*.json"))
    if args.limit > 0:
        ann_files = ann_files[:args.limit]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mode_str = "DRY RUN" if args.dry_run else "AUDIT"
    print(f"[{mode_str}] {len(ann_files)} clips, levels={levels}")
    print(f"API: {args.api_base}  model: {args.model}  workers: {args.workers}")
    print(f"Input: {ann_dir}  Output: {output_dir}")
    if l3_frames_base:
        print(f"L3 frames: {l3_frames_base}")
    print()

    ok_count = skipped_count = error_count = 0
    agg_stats: dict[str, int] = {}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                check_clip,
                ann_path, frames_base, output_dir, levels,
                args.api_base, api_key, args.model,
                args.max_frames_per_call,
                args.resize_max_width,
                args.jpeg_quality,
                args.overwrite,
                args.dry_run,
                l3_frames_base,
            ): ann_path
            for ann_path in ann_files
        }
        total = len(futures)
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                res = fut.result()
            except Exception as e:
                error_count += 1
                ann_path = futures[fut]
                print(f"[{i}/{total}] EXCEPTION  {ann_path.stem}: {e}")
                continue

            if res["skipped"]:
                skipped_count += 1
            elif res["ok"]:
                ok_count += 1
                # Aggregate stats
                for level_key, st in res.get("stats", {}).items():
                    if isinstance(st, dict):
                        for k, v in st.items():
                            if isinstance(v, (int, float)):
                                agg_stats[f"{level_key}.{k}"] = agg_stats.get(f"{level_key}.{k}", 0) + v
                print(f"[{i}/{total}] OK     {res['clip_key']}")
            else:
                error_count += 1
                print(f"[{i}/{total}] ERROR  {res['clip_key']}: {res['error']}")

    print(f"\nFinished: {ok_count} checked, {skipped_count} skipped, {error_count} errors")
    if agg_stats:
        print("\nAggregate stats:")
        for k, v in sorted(agg_stats.items()):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
