#!/usr/bin/env python3
"""
fix_annotations.py — Post-hoc repair for hierarchical segmentation annotations.

Problem: run_pipeline.sh Step 5 (leaf_c) correctly shrinks L2 events to tightly
cover L3 children, but Step 6 (2c_shrink) TASK 1 revises L2 events again,
potentially making them smaller than their L3 children.

Fix strategy (bottom-up):
  1. Drop L3 actions with start >= end
  2. Recompute L2 event boundaries from L3 children tight cover
     (undoes 2c_shrink's double-shrink; restores leaf_c intent)
  3. Drop L2 events with start >= end; orphan their L3
  4. Expand L1 phase boundaries to cover L2 events + direct L3
  5. Drop L1 phases with start >= end; orphan L2+L3
  6. Clamp all to [0, clip_duration], renumber IDs

Usage:
    # Dry-run to see what would be fixed
    python fix_annotations.py \\
        --annotation-dir /path/to/annotations_checked \\
        --output-dir /path/to/annotations_fixed \\
        --dry-run

    # Actually fix (writes to output-dir; can be same as input for in-place)
    python fix_annotations.py \\
        --annotation-dir /path/to/annotations_checked \\
        --output-dir /path/to/annotations_fixed
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path


def fix_annotation(ann: dict) -> tuple[dict, dict]:
    """Fix a single annotation. Returns (fixed_ann, stats)."""
    ann = deepcopy(ann)
    clip_dur = ann.get("clip_duration_sec", float("inf"))

    stats: dict[str, int] = {
        "l3_removed_invalid": 0,
        "l2_removed_invalid": 0,
        "l1_removed_invalid": 0,
        "l2_recomputed": 0,       # L2 boundary recomputed from L3 tight cover
        "l1_expanded": 0,         # L1 boundary expanded to cover L2 children
        "l3_orphaned": 0,
    }

    phases = (ann.get("level1") or {}).get("macro_phases") or []
    events = (ann.get("level2") or {}).get("events") or []
    actions = (ann.get("level3") or {}).get("grounding_results") or []

    # ── Step 1: Drop L3 actions with start >= end ──────────────────────
    valid_actions = []
    for a in actions:
        s, e = a.get("start_time", 0), a.get("end_time", 0)
        if s < e:
            a["start_time"] = max(0, s)
            a["end_time"] = min(clip_dur, e)
            valid_actions.append(a)
        else:
            stats["l3_removed_invalid"] += 1
    actions = valid_actions

    # ── Step 2: Recompute L2 event boundaries from L3 tight cover ─────
    #
    # leaf_c (Step 5) set L2 boundaries = VLM-suggested tight cover of L3.
    # 2c_shrink (Step 6) may have revised L2 boundaries again (TASK 1),
    # causing children to overflow.
    #
    # Strategy: set L2 = [min(L3.start), max(L3.end)] for its children.
    # This undoes any incorrect 2c_shrink revision and restores containment.
    #
    l3_by_event: dict[int, list[dict]] = {}
    l3_by_phase: dict[int, list[dict]] = {}
    for a in actions:
        peid = a.get("parent_event_id")
        ppid = a.get("parent_phase_id")
        if peid is not None:
            l3_by_event.setdefault(peid, []).append(a)
        elif ppid is not None:
            l3_by_phase.setdefault(ppid, []).append(a)

    for ev in events:
        eid = ev.get("event_id")
        children = l3_by_event.get(eid, [])
        if not children:
            continue
        child_min = min(c["start_time"] for c in children)
        child_max = max(c["end_time"] for c in children)
        ev_s = ev.get("start_time", 0)
        ev_e = ev.get("end_time", 0)
        # Recompute: ensure parent covers all children
        new_s = min(ev_s, child_min)
        new_e = max(ev_e, child_max)
        if new_s != ev_s or new_e != ev_e:
            ev["_fixed_from"] = [ev_s, ev_e]
            ev["start_time"] = max(0, new_s)
            ev["end_time"] = min(clip_dur, new_e)
            stats["l2_recomputed"] += 1

    # ── Step 3: Drop L2 events with start >= end ──────────────────────
    valid_events = []
    removed_eids: set[int] = set()
    for ev in events:
        s, e = ev.get("start_time", 0), ev.get("end_time", 0)
        if s < e:
            valid_events.append(ev)
        else:
            stats["l2_removed_invalid"] += 1
            eid = ev.get("event_id")
            if eid is not None:
                removed_eids.add(eid)
    events = valid_events

    # Orphan L3 whose parent event was removed
    if removed_eids:
        kept = []
        for a in actions:
            if a.get("parent_event_id") in removed_eids:
                stats["l3_orphaned"] += 1
            else:
                kept.append(a)
        actions = kept

    # ── Step 4: Expand L1 phases to cover L2 children + direct L3 ─────
    events_by_phase: dict[int, list[dict]] = {}
    for ev in events:
        ppid = ev.get("parent_phase_id")
        if ppid is not None:
            events_by_phase.setdefault(ppid, []).append(ev)

    for ph in phases:
        pid = ph.get("phase_id")
        child_bounds: list[tuple[int, int]] = []
        for ev in events_by_phase.get(pid, []):
            child_bounds.append((ev["start_time"], ev["end_time"]))
        # Eventless phases: L3 actions directly parented to this phase
        for a in l3_by_phase.get(pid, []):
            child_bounds.append((a["start_time"], a["end_time"]))

        if not child_bounds:
            continue
        child_min = min(b[0] for b in child_bounds)
        child_max = max(b[1] for b in child_bounds)
        ph_s = ph.get("start_time", 0)
        ph_e = ph.get("end_time", 0)
        new_s = min(ph_s, child_min)
        new_e = max(ph_e, child_max)
        if new_s != ph_s or new_e != ph_e:
            ph["_fixed_from"] = [ph_s, ph_e]
            ph["start_time"] = max(0, new_s)
            ph["end_time"] = min(clip_dur, new_e)
            stats["l1_expanded"] += 1

    # ── Step 5: Drop L1 phases with start >= end ──────────────────────
    valid_phases = []
    removed_pids: set[int] = set()
    for ph in phases:
        s, e = ph.get("start_time", 0), ph.get("end_time", 0)
        if s < e:
            valid_phases.append(ph)
        else:
            stats["l1_removed_invalid"] += 1
            pid = ph.get("phase_id")
            if pid is not None:
                removed_pids.add(pid)
    phases = valid_phases

    # Orphan L2+L3 whose parent phase was removed
    if removed_pids:
        kept_events = []
        for ev in events:
            if ev.get("parent_phase_id") in removed_pids:
                eid = ev.get("event_id")
                if eid is not None:
                    removed_eids.add(eid)
                stats["l2_removed_invalid"] += 1
            else:
                kept_events.append(ev)
        events = kept_events

        kept = []
        for a in actions:
            peid = a.get("parent_event_id")
            ppid = a.get("parent_phase_id")
            if (peid is not None and peid in removed_eids) or \
               (ppid is not None and ppid in removed_pids):
                stats["l3_orphaned"] += 1
            else:
                kept.append(a)
        actions = kept

    # ── Step 6: Renumber IDs (preserve sort order) ────────────────────
    phases.sort(key=lambda p: (p.get("start_time", 0), p.get("end_time", 0)))
    old_to_new_pid: dict[int, int] = {}
    for i, ph in enumerate(phases, 1):
        old_to_new_pid[ph.get("phase_id")] = i
        ph["phase_id"] = i

    events.sort(key=lambda e: (e.get("start_time", 0), e.get("end_time", 0)))
    old_to_new_eid: dict[int, int] = {}
    for i, ev in enumerate(events, 1):
        old_to_new_eid[ev.get("event_id")] = i
        ev["event_id"] = i
        ppid = ev.get("parent_phase_id")
        if ppid in old_to_new_pid:
            ev["parent_phase_id"] = old_to_new_pid[ppid]

    actions.sort(key=lambda a: (a.get("start_time", 0), a.get("end_time", 0)))
    for i, a in enumerate(actions, 1):
        a["action_id"] = i
        peid = a.get("parent_event_id")
        if peid in old_to_new_eid:
            a["parent_event_id"] = old_to_new_eid[peid]
        ppid = a.get("parent_phase_id")
        if ppid in old_to_new_pid:
            a["parent_phase_id"] = old_to_new_pid[ppid]

    # ── Write back ────────────────────────────────────────────────────
    if ann.get("level1") is not None:
        ann["level1"]["macro_phases"] = phases
    if ann.get("level2") is not None:
        ann["level2"]["events"] = events
    if ann.get("level3") is not None:
        ann["level3"]["grounding_results"] = actions

    return ann, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--annotation-dir", required=True,
                        help="Input directory with annotation JSONs")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for fixed annotations")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report issues without writing fixed files")
    args = parser.parse_args()

    ann_dir = Path(args.annotation_dir)
    out_dir = Path(args.output_dir)

    if not ann_dir.exists():
        print(f"ERROR: annotation-dir not found: {ann_dir}", file=sys.stderr)
        sys.exit(1)

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    ann_files = sorted(ann_dir.glob("*.json"))
    if not ann_files:
        print(f"No JSON files found in {ann_dir}")
        return

    print(f"Scanning {len(ann_files)} annotation files ...\n")

    agg: dict[str, int] = {}
    changed = 0

    for ann_path in ann_files:
        with open(ann_path, encoding="utf-8") as f:
            ann = json.load(f)

        clip_key = ann.get("clip_key", ann_path.stem)
        fixed, stats = fix_annotation(ann)

        has_fixes = any(v > 0 for v in stats.values())
        if has_fixes:
            changed += 1
            detail_parts = [f"{k}={v}" for k, v in stats.items() if v > 0]
            print(f"  FIX  {clip_key}: {', '.join(detail_parts)}")

        for k, v in stats.items():
            agg[k] = agg.get(k, 0) + v

        if not args.dry_run:
            out_file = out_dir / f"{clip_key}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(fixed, f, ensure_ascii=False, indent=2)

    # ── Summary ──
    mode = "DRY RUN" if args.dry_run else "FIXED"
    print(f"\n[{mode}] {len(ann_files)} files scanned, {changed} had issues")
    if agg:
        print("\nAggregate fixes:")
        for k, v in sorted(agg.items()):
            if v > 0:
                print(f"  {k}: {v}")

    if not args.dry_run:
        if changed > 0:
            print(f"\nFixed annotations written to: {out_dir}")
        else:
            print(f"\nNo issues found. Clean copies written to: {out_dir}")


if __name__ == "__main__":
    main()
