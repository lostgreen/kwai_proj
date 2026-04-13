#!/usr/bin/env python3
"""
Recover overwritten rollout report by reconstructing old processed items.

Logic:
  - pilot_sample.jsonl - _remaining.jsonl = items that WERE in old rollout_report.jsonl
  - Create stub entries (mean_reward=-1) for these items so resume_helper skips them
  - Merge stubs + current shard-based rollout_report.jsonl → recovered report
  - Backup old train_final.jsonl (it contains the valid filtered results from old report)

After pipeline finishes:
  cat train_final_old.jsonl train_final.jsonl > train_final_combined.jsonl

Usage:
    python recover_report.py --results-dir ./results [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def load_jsonl(path: str | Path) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def content_key(item: dict) -> tuple[str, str]:
    return (item.get("prompt", ""), item.get("answer", ""))


def main():
    parser = argparse.ArgumentParser(description="Recover overwritten rollout report")
    parser.add_argument("--results-dir", required=True, help="Path to results/ directory")
    parser.add_argument("--dry-run", action="store_true", help="Print stats only, don't write")
    args = parser.parse_args()

    d = Path(args.results_dir)
    pilot_path = d / "pilot_sample.jsonl"
    remaining_path = d / "_remaining.jsonl"
    report_path = d / "rollout_report.jsonl"
    kept_path = d / "rollout_kept.jsonl"
    train_final_path = d / "train_final.jsonl"

    # Validate required files exist
    for p in [pilot_path, remaining_path, report_path]:
        if not p.is_file():
            print(f"ERROR: Required file not found: {p}")
            return

    # ── 1. Load remaining keys (items NOT in old report) ──
    remaining = load_jsonl(remaining_path)
    remaining_keys = {content_key(item) for item in remaining}
    print(f"_remaining.jsonl:      {len(remaining):>8d} items (were NOT in old report)")

    # ── 2. Load pilot ──
    pilot = load_jsonl(pilot_path)
    print(f"pilot_sample.jsonl:    {len(pilot):>8d} items (full pilot set)")

    # ── 3. Compute old processed items: pilot - remaining ──
    old_processed = []
    for item in pilot:
        if content_key(item) not in remaining_keys:
            old_processed.append(item)
    print(f"Old processed (pilot - remaining): {len(old_processed):>5d} items (the lost report)")

    # ── 4. Load current report (shard-only) ──
    current_report = load_jsonl(report_path)
    current_keys = {content_key(entry) for entry in current_report}
    print(f"rollout_report.jsonl:  {len(current_report):>8d} items (shard results, current)")

    # ── 5. Check overlap ──
    overlap = current_keys & {content_key(item) for item in old_processed}
    print(f"Overlap (should be 0): {len(overlap):>8d}")

    # ── 6. Load old train_final if exists ──
    if train_final_path.is_file():
        train_final = load_jsonl(train_final_path)
        print(f"train_final.jsonl:     {len(train_final):>8d} items (old filtered results)")
    else:
        train_final = []
        print(f"train_final.jsonl:     NOT FOUND (no backup needed)")

    # ── 7. Create stubs for old processed items ──
    stubs = []
    for item in old_processed:
        key = content_key(item)
        if key in current_keys:
            continue  # already in shard report, skip
        stub = {
            "prompt": key[0],
            "answer": key[1],
            "keep": False,
            "mean_reward": -1.0,  # will NOT pass [0.25, 0.5] filter → excluded from new train_final
            "_recovered_stub": True,
        }
        stubs.append(stub)

    # ── Summary ──
    total = len(current_report) + len(stubs)
    new_remaining = len(pilot) - total
    print(f"\n{'='*60}")
    print(f"Recovery plan:")
    print(f"  Current shard report:  {len(current_report):>8d}")
    print(f"  + Recovered stubs:     {len(stubs):>8d}")
    print(f"  = Merged report:       {total:>8d}")
    print(f"  Remaining for resume:  {new_remaining:>8d} / {len(pilot)}")
    print(f"{'='*60}")

    if args.dry_run:
        print("\n[DRY RUN] No files modified.")
        return

    # ── 8. Backup old train_final ──
    if train_final_path.is_file():
        backup = d / "train_final_old.jsonl"
        shutil.copy2(train_final_path, backup)
        print(f"\nBacked up: {train_final_path.name} → {backup.name}")

    # ── 9. Write recovered report (overwrite current shard-only report) ──
    merged = current_report + stubs
    with open(report_path, "w", encoding="utf-8") as f:
        for entry in merged:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Wrote recovered report: {report_path} ({len(merged)} entries)")

    # ── 10. Instructions ──
    print(f"""
Next steps:
  1. Re-run pipeline:
     bash proxy_data/llava_video_178k/run_pipeline.sh

     → Step 3 will resume from {new_remaining} remaining items
     → Stubs (mean_reward=-1) are auto-excluded from filter

  2. After pipeline finishes, merge old + new train_final:
     cat {d}/train_final_old.jsonl {d}/train_final.jsonl > {d}/train_final_combined.jsonl

  3. Use train_final_combined.jsonl for training
""")


if __name__ == "__main__":
    main()
