#!/usr/bin/env python3
"""
Analyze annotation JSON distribution for training data construction decisions.

Computes per-video and aggregate statistics:
- Phases per video, events per video, events per phase
- Leaf-node distribution (events vs empty-event phases)
- L3 micro-actions per leaf node
- Duration distributions at each level
- Topology type breakdown

Usage:
    python analyze_annotations.py \
        --annotation-dir /path/to/annotations \
        [--complete-only]  # only include annotations with L3
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def load_annotations(ann_dir: Path, complete_only: bool = False) -> list[dict]:
    anns = []
    for p in sorted(ann_dir.glob("*.json")):
        with open(p) as f:
            ann = json.load(f)
        if complete_only and not ann.get("level3"):
            continue
        anns.append(ann)
    return anns


def _stat_summary(values: list[float | int], label: str) -> str:
    if not values:
        return f"  {label}: (no data)"
    n = len(values)
    mn, mx = min(values), max(values)
    avg = statistics.mean(values)
    med = statistics.median(values)
    if n > 1:
        std = statistics.stdev(values)
    else:
        std = 0.0
    return (
        f"  {label}: n={n}  "
        f"min={mn:.1f}  max={mx:.1f}  "
        f"avg={avg:.1f}  med={med:.1f}  std={std:.1f}"
    )


def validate_and_clean(anns: list[dict]) -> tuple[list[dict], list[str]]:
    """Detect and report bad annotations. Returns (clean_anns, issues)."""
    issues: list[str] = []
    clean = []

    for ann in anns:
        key = ann.get("clip_key", "?")
        bad = False

        # Check phase durations
        for phase in (ann.get("level1") or {}).get("macro_phases") or []:
            s, e = phase.get("start_time", 0), phase.get("end_time", 0)
            if e < s:
                issues.append(
                    f"  [NEG_DUR] {key}: phase {phase.get('phase_id')} "
                    f"start={s} end={e} (dur={e - s}s)"
                )
                bad = True
            if e > ann.get("clip_duration_sec", 999) + 5:
                issues.append(
                    f"  [OVERFLOW] {key}: phase {phase.get('phase_id')} "
                    f"end={e} > clip_duration={ann.get('clip_duration_sec')}"
                )

        # Check event durations
        for ev in (ann.get("level2") or {}).get("events") or []:
            s, e = ev.get("start_time", 0), ev.get("end_time", 0)
            if e < s:
                issues.append(
                    f"  [NEG_DUR] {key}: event {ev.get('event_id')} "
                    f"start={s} end={e} (dur={e - s}s)"
                )
                bad = True

        # Check L3 action durations
        for act in (ann.get("level3") or {}).get("grounding_results") or []:
            s, e = act.get("start_time", 0), act.get("end_time", 0)
            if e < s:
                issues.append(
                    f"  [NEG_DUR] {key}: action {act.get('action_id')} "
                    f"start={s} end={e} (dur={e - s}s)"
                )
                bad = True
            if e - s == 0:
                issues.append(
                    f"  [ZERO_DUR] {key}: action {act.get('action_id')} "
                    f"start={s} end={e}"
                )

        # Check overlapping phases
        phases = sorted(
            ((ann.get("level1") or {}).get("macro_phases") or []),
            key=lambda p: p.get("start_time", 0),
        )
        for i in range(len(phases) - 1):
            cur_end = phases[i].get("end_time", 0)
            nxt_start = phases[i + 1].get("start_time", 0)
            gap = nxt_start - cur_end
            if gap < -2:  # >2s overlap
                issues.append(
                    f"  [OVERLAP] {key}: phase {phases[i].get('phase_id')} "
                    f"end={cur_end} > phase {phases[i+1].get('phase_id')} "
                    f"start={nxt_start} (overlap={-gap}s)"
                )

        if not bad:
            clean.append(ann)

    return clean, issues


def analyze(anns: list[dict]):
    # ── Topology ──
    topo_counter = Counter()
    # ── Per-video counts ──
    phases_per_video = []
    events_per_video = []
    events_per_phase = []          # per phase that HAS events
    empty_phases_per_video = []    # phases with events=[]
    leaf_nodes_per_video = []
    l3_actions_per_video = []
    l3_actions_per_leaf = []

    # ── Duration distributions ──
    video_durations = []
    phase_durations = []
    event_durations = []
    l3_action_durations = []
    leaf_durations = []            # leaf = event or empty-event phase

    # ── L3 completeness ──
    has_l3_count = 0

    for ann in anns:
        topo = ann.get("topology_type", "unknown")
        topo_counter[topo] += 1

        duration = ann.get("clip_duration_sec", 0)
        video_durations.append(duration)

        phases = (ann.get("level1") or {}).get("macro_phases") or []
        events = (ann.get("level2") or {}).get("events") or []
        l3 = ann.get("level3") or {}
        actions = l3.get("grounding_results") or []

        n_phases = len(phases)
        n_events = len(events)
        phases_per_video.append(n_phases)
        events_per_video.append(n_events)

        if actions:
            has_l3_count += 1

        # Build phase → events mapping
        phase_event_map: dict[int, list[dict]] = defaultdict(list)
        for ev in events:
            pid = ev.get("parent_phase_id")
            if pid is not None:
                phase_event_map[pid].append(ev)

        n_empty = 0
        n_leaf = 0
        for phase in phases:
            pid = phase.get("phase_id")
            children = phase_event_map.get(pid, [])
            ph_dur = (phase.get("end_time", 0) - phase.get("start_time", 0))
            phase_durations.append(ph_dur)

            if children:
                events_per_phase.append(len(children))
                n_leaf += len(children)
                for ev in children:
                    ev_dur = ev.get("end_time", 0) - ev.get("start_time", 0)
                    event_durations.append(ev_dur)
                    leaf_durations.append(ev_dur)
            else:
                n_empty += 1
                n_leaf += 1  # phase itself is a leaf
                leaf_durations.append(ph_dur)

        empty_phases_per_video.append(n_empty)
        leaf_nodes_per_video.append(n_leaf)

        # L3 actions
        n_actions = len(actions)
        l3_actions_per_video.append(n_actions)

        # L3 actions per leaf: group by parent
        actions_by_parent: dict[str, int] = Counter()
        for act in actions:
            peid = act.get("parent_event_id")
            ppid = act.get("parent_phase_id")
            if peid is not None:
                actions_by_parent[f"ev_{peid}"] += 1
            elif ppid is not None:
                actions_by_parent[f"ph_{ppid}"] += 1
        for cnt in actions_by_parent.values():
            l3_actions_per_leaf.append(cnt)

        for act in actions:
            act_dur = act.get("end_time", 0) - act.get("start_time", 0)
            l3_action_durations.append(act_dur)

    # ── Print Report ──
    print("=" * 70)
    print(f"  ANNOTATION DISTRIBUTION ANALYSIS  ({len(anns)} videos)")
    print("=" * 70)

    # Topology
    print("\n[1] Topology Type Distribution")
    for topo, cnt in topo_counter.most_common():
        pct = cnt / len(anns) * 100
        print(f"  {topo:15s}: {cnt:4d} ({pct:.1f}%)")

    # L3 completeness
    print(f"\n[2] L3 Completeness: {has_l3_count}/{len(anns)} "
          f"({has_l3_count / len(anns) * 100:.1f}%) have L3 annotations")

    # Video duration
    print(f"\n[3] Video Duration (sec)")
    print(_stat_summary(video_durations, "duration"))

    # Phases per video
    print(f"\n[4] Phases per Video (L1)")
    print(_stat_summary(phases_per_video, "phases"))
    if phases_per_video:
        dist = Counter(phases_per_video)
        print("  distribution:", dict(sorted(dist.items())))

    # Events per video
    print(f"\n[5] Events per Video (L2)")
    print(_stat_summary(events_per_video, "events"))
    if events_per_video:
        dist = Counter(events_per_video)
        print("  distribution:", dict(sorted(dist.items())))

    # Events per phase (only phases WITH events)
    print(f"\n[6] Events per Phase (only phases with events)")
    print(_stat_summary(events_per_phase, "events/phase"))
    if events_per_phase:
        dist = Counter(events_per_phase)
        print("  distribution:", dict(sorted(dist.items())))

    # Empty phases (events=[])
    print(f"\n[7] Empty-Event Phases per Video")
    print(_stat_summary(empty_phases_per_video, "empty_phases"))
    total_phases = sum(phases_per_video)
    total_empty = sum(empty_phases_per_video)
    if total_phases > 0:
        print(f"  total: {total_empty}/{total_phases} phases have events=[] "
              f"({total_empty / total_phases * 100:.1f}%)")

    # Leaf nodes per video
    print(f"\n[8] Leaf Nodes per Video (events + empty-phases)")
    print(_stat_summary(leaf_nodes_per_video, "leaves"))

    # Phase durations
    print(f"\n[9] Phase Duration (sec)")
    print(_stat_summary(phase_durations, "phase_dur"))

    # Event durations
    print(f"\n[10] Event Duration (sec)")
    print(_stat_summary(event_durations, "event_dur"))

    # Leaf durations
    print(f"\n[11] Leaf Node Duration (sec) — what L3 receives as input")
    print(_stat_summary(leaf_durations, "leaf_dur"))

    # L3 actions per video
    print(f"\n[12] L3 Micro-Actions per Video")
    print(_stat_summary(l3_actions_per_video, "actions"))

    # L3 actions per leaf
    print(f"\n[13] L3 Micro-Actions per Leaf Node")
    print(_stat_summary(l3_actions_per_leaf, "actions/leaf"))
    if l3_actions_per_leaf:
        dist = Counter(l3_actions_per_leaf)
        print("  distribution:", dict(sorted(dist.items())))

    # L3 action durations
    print(f"\n[14] L3 Micro-Action Duration (sec)")
    print(_stat_summary(l3_action_durations, "action_dur"))

    # ── Training difficulty analysis ──
    print("\n" + "=" * 70)
    print("  TRAINING DIFFICULTY ANALYSIS")
    print("=" * 70)

    # How many phases/events would a full-video L2 task have?
    if events_per_video:
        few_events = sum(1 for e in events_per_video if e <= 2)
        print(f"\n  Videos with <=2 events: {few_events}/{len(anns)} "
              f"({few_events / len(anns) * 100:.1f}%)")
    if phases_per_video:
        few_phases = sum(1 for p in phases_per_video if p <= 2)
        print(f"  Videos with <=2 phases: {few_phases}/{len(anns)} "
              f"({few_phases / len(anns) * 100:.1f}%)")

    # How many leaves have only 1-2 L3 actions?
    if l3_actions_per_leaf:
        trivial = sum(1 for a in l3_actions_per_leaf if a <= 2)
        print(f"  Leaf nodes with <=2 L3 actions: {trivial}/{len(l3_actions_per_leaf)} "
              f"({trivial / len(l3_actions_per_leaf) * 100:.1f}%)")

    # Short segment warning
    if leaf_durations:
        short_leaves = sum(1 for d in leaf_durations if d < 10)
        print(f"  Leaf nodes <10sec: {short_leaves}/{len(leaf_durations)} "
              f"({short_leaves / len(leaf_durations) * 100:.1f}%)")

    if event_durations:
        short_events = sum(1 for d in event_durations if d < 10)
        print(f"  Events <10sec: {short_events}/{len(event_durations)} "
              f"({short_events / len(event_durations) * 100:.1f}%)")

    print()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--annotation-dir", required=True,
                        help="Directory with annotation JSONs")
    parser.add_argument("--complete-only", action="store_true",
                        help="Only include annotations with L3 results")
    args = parser.parse_args()

    ann_dir = Path(args.annotation_dir)
    anns = load_annotations(ann_dir, args.complete_only)
    if not anns:
        print(f"No annotation files found in {ann_dir}")
        return

    # ── Data validation ──
    clean_anns, issues = validate_and_clean(anns)

    print("=" * 70)
    print("  DATA QUALITY CHECK")
    print("=" * 70)
    print(f"\n  Total annotations: {len(anns)}")
    print(f"  Clean annotations: {len(clean_anns)} "
          f"({len(clean_anns) / len(anns) * 100:.1f}%)")
    print(f"  Annotations with issues: {len(anns) - len(clean_anns)}")
    print(f"  Total issues found: {len(issues)}")

    if issues:
        print(f"\n  Bad cases:")
        for issue in issues:
            print(issue)

    # ── Run analysis on clean data ──
    print(f"\n  (Running distribution analysis on {len(clean_anns)} clean annotations)\n")
    analyze(clean_anns)


if __name__ == "__main__":
    main()
