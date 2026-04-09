#!/usr/bin/env python3
"""
stage1_classify.py — Stage 1: paradigm + domain + feasibility classification.

Classifies videos using the v4 taxonomy (7 paradigms, 10 domain_l1 / ~40 domain_l2)
with feasibility filtering. Outputs classification JSONL and distribution charts.

Prerequisites:
    # Extract 64 frames per video (1fps, capped at 64)
    python proxy_data/youcook2_seg/hier_seg_annotation/extract_frames.py \
        --jsonl screen_keep.jsonl \
        --output-dir $DATA_ROOT/frames_stage1 \
        --fps 1 --max-frames 64 --workers 8

Usage:
    # Classify + visualize
    python proxy_data/youcook2_seg/hier_seg_annotation/stage1_classify.py \
        --jsonl screen_keep.jsonl \
        --frames-dir $DATA_ROOT/frames_stage1 \
        --output-dir $DATA_ROOT/stage1_output \
        --model pa/gemini-3.1-pro-preview --workers 4

    # Visualize only (from existing results)
    python proxy_data/youcook2_seg/hier_seg_annotation/stage1_classify.py \
        --jsonl screen_keep.jsonl \
        --frames-dir unused \
        --output-dir $DATA_ROOT/stage1_output \
        --visualize-only

Output:
    {output-dir}/
        classify_results.jsonl   # all records
        classify_keep.jsonl      # feasibility.skip == false
        classify_reject.jsonl    # feasibility.skip == true
        figures/
            paradigm_distribution.png
            domain_l1_distribution.png
            domain_paradigm_heatmap.png
            feasibility_histogram.png
            skip_reason_pie.png
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Imports from annotate.py (same directory)
from annotate import (
    call_and_parse,
    encode_frame_files,
    format_mmss,
    frame_stem_to_index,
    get_all_frame_files,
    load_frame_meta,
    clip_key_from_path,
    sample_uniform,
    get_token_usage,
    reset_token_usage,
)

# Imports from archetypes.py (same directory)
from archetypes import (
    DOMAIN_L1_ALL,
    DOMAIN_L2_ALL,
    PARADIGM_IDS,
    SYSTEM_PROMPT,
    get_classification_prompt,
    resolve_domain_l1,
)


# ─────────────────────────────────────────────────────────────────────────────
# Validation / Normalization
# ─────────────────────────────────────────────────────────────────────────────

VALID_SKIP_REASONS = {
    "talk_dominant", "ambient_static", "low_visual_dynamics",
    "too_short", "low_feasibility",
}
VALID_CAMERA_STYLES = {"static_tripod", "handheld", "multi_angle", "first_person"}
VALID_EDITING_STYLES = {"continuous", "jump_cut", "montage", "mixed"}


def validate_classification(parsed: dict, clip_duration: float) -> dict:
    """Normalize VLM response + apply programmatic feasibility rules."""
    result: dict[str, Any] = {}

    # paradigm
    paradigm = parsed.get("paradigm", "tutorial")
    if paradigm not in PARADIGM_IDS:
        paradigm = "tutorial"
    result["paradigm"] = paradigm

    # paradigm_confidence
    conf = parsed.get("paradigm_confidence")
    if not isinstance(conf, (int, float)):
        conf = 0.5
    result["paradigm_confidence"] = max(0.0, min(1.0, float(conf)))

    # paradigm_reason
    result["paradigm_reason"] = str(parsed.get("paradigm_reason", ""))

    # domain
    domain_l2 = parsed.get("domain_l2", "other")
    if domain_l2 not in DOMAIN_L2_ALL:
        domain_l2 = "other"
    result["domain_l2"] = domain_l2
    result["domain_l1"] = resolve_domain_l1(domain_l2)

    # feasibility
    feas = parsed.get("feasibility", {})
    if not isinstance(feas, dict):
        feas = {}

    score = feas.get("score")
    if not isinstance(score, (int, float)):
        score = 0.5
    score = max(0.0, min(1.0, float(score)))

    skip = bool(feas.get("skip", False))
    skip_reason = feas.get("skip_reason")
    if skip_reason not in VALID_SKIP_REASONS:
        skip_reason = None

    est_phases = feas.get("estimated_n_phases", 0)
    if not isinstance(est_phases, int):
        est_phases = 0
    est_events = feas.get("estimated_n_events", 0)
    if not isinstance(est_events, int):
        est_events = 0

    dynamics = feas.get("visual_dynamics", "medium")
    if dynamics not in {"high", "medium", "low"}:
        dynamics = "medium"

    # Programmatic override rules (DESIGN.md §3.3)
    if clip_duration < 15:
        skip, skip_reason = True, "too_short"
    elif dynamics == "low" and est_events < 2 and not skip:
        skip, skip_reason = True, "low_visual_dynamics"
    elif score < 0.4 and not skip:
        skip, skip_reason = True, "low_feasibility"

    result["feasibility"] = {
        "score": score,
        "skip": skip,
        "skip_reason": skip_reason,
        "estimated_n_phases": est_phases,
        "estimated_n_events": est_events,
        "visual_dynamics": dynamics,
    }

    # video_metadata
    vmeta = parsed.get("video_metadata", {})
    if not isinstance(vmeta, dict):
        vmeta = {}
    result["video_metadata"] = {
        "has_text_overlay": bool(vmeta.get("has_text_overlay", False)),
        "has_narration": bool(vmeta.get("has_narration", False)),
        "camera_style": vmeta.get("camera_style") if vmeta.get("camera_style") in VALID_CAMERA_STYLES else "unknown",
        "editing_style": vmeta.get("editing_style") if vmeta.get("editing_style") in VALID_EDITING_STYLES else "unknown",
    }

    # video_caption
    result["video_caption"] = str(parsed.get("video_caption", ""))

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Single-video classification
# ─────────────────────────────────────────────────────────────────────────────

def classify_one(
    record: dict,
    frames_base: Path,
    done_keys: set[str],
    api_base: str, api_key: str, model: str,
    max_frames: int,
    resize_max_width: int, jpeg_quality: int,
    overwrite: bool,
) -> dict:
    """Classify one video. Returns {clip_key, ok, skipped, error, classification}."""
    videos = record.get("videos") or []
    if not videos:
        return {"clip_key": "?", "ok": False, "skipped": False, "error": "no videos field"}

    meta = record.get("metadata") or {}
    key = str(meta.get("clip_key") or clip_key_from_path(videos[0]))

    # Idempotency
    if not overwrite and key in done_keys:
        return {"clip_key": key, "ok": True, "skipped": True, "error": None}

    frame_dir = frames_base / key
    if not frame_dir.is_dir():
        return {"clip_key": key, "ok": False, "skipped": False,
                "error": f"frame dir not found: {frame_dir}"}

    all_frames = get_all_frame_files(frame_dir)
    if not all_frames:
        return {"clip_key": key, "ok": False, "skipped": False,
                "error": f"no frames in {frame_dir}"}

    # Resolve clip duration
    frame_meta = load_frame_meta(frame_dir)
    clip_duration = float(
        frame_meta.get("annotation_end_sec")
        or meta.get("clip_end")
        or meta.get("clip_duration")
        or record.get("duration")
        or len(all_frames)
    )

    try:
        sampled = sample_uniform(all_frames, max_frames)
        frame_b64 = encode_frame_files(sampled, resize_max_width=resize_max_width, jpeg_quality=jpeg_quality)

        frame_labels = []
        for fp in sampled:
            idx = frame_stem_to_index(fp, 0)
            frame_labels.append(f"[Timestamp {format_mmss(idx)} | Frame {idx}]")

        prompt_text = get_classification_prompt(n_frames=len(sampled), duration_sec=int(clip_duration))
        parsed = call_and_parse(api_base, api_key, model, SYSTEM_PROMPT, prompt_text, frame_b64, frame_labels)

        if parsed is None:
            return {"clip_key": key, "ok": False, "skipped": False,
                    "error": "VLM parse failed after retries"}

        classification = validate_classification(parsed, clip_duration)
        classification["classified_at"] = datetime.now(timezone.utc).isoformat()

        print(f"  [{key}] paradigm={classification['paradigm']} "
              f"domain={classification['domain_l1']}/{classification['domain_l2']} "
              f"feas={classification['feasibility']['score']:.2f} "
              f"skip={classification['feasibility']['skip']}", flush=True)

        return {"clip_key": key, "ok": True, "skipped": False, "error": None,
                "classification": classification}

    except Exception as e:
        return {"clip_key": key, "ok": False, "skipped": False, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_existing_results(output_dir: Path) -> dict[str, dict]:
    """Load existing classify_results.jsonl into {clip_key: record}."""
    results_path = output_dir / "classify_results.jsonl"
    if not results_path.exists():
        return {}
    existing = {}
    for rec in load_jsonl(results_path):
        meta = rec.get("metadata") or {}
        key = meta.get("clip_key") or ""
        if key:
            existing[key] = rec
    return existing


def build_output_record(original: dict, classification: dict) -> dict:
    """Merge original record with _classify payload."""
    out = dict(original)
    out["_classify"] = classification
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def _get_classify_records(records: list[dict]) -> list[dict]:
    """Extract _classify dicts from output records."""
    return [r["_classify"] for r in records if "_classify" in r]


# ── Color palettes ──

PARADIGM_COLORS: dict[str, str] = {
    "tutorial": "#4C72B0",
    "educational": "#55A868",
    "cinematic": "#C44E52",
    "vlog": "#8172B2",
    "sports_match": "#CCB974",
    "cyclical": "#64B5CD",
    "continuous": "#D98880",
}

DOMAIN_L1_COLORS: dict[str, str] = {
    "knowledge_education": "#4C72B0",
    "film_entertainment": "#C44E52",
    "sports_esports": "#CCB974",
    "lifestyle_vlog": "#8172B2",
    "arts_performance": "#55A868",
    "task_howto": "#64B5CD",
}


def run_visualization(records: list[dict], outdir: str) -> None:
    """Generate 2 donut charts + text summary."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    fig_dir = Path(outdir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    classifies = _get_classify_records(records)
    if not classifies:
        print("No classification results to visualize.", flush=True)
        return

    total = len(classifies)
    keep = [c for c in classifies if not c["feasibility"]["skip"]]
    skip = [c for c in classifies if c["feasibility"]["skip"]]

    # ── Text summary ──
    print(f"\n{'=' * 50}", flush=True)
    print(f"  Classification Summary (N={total})", flush=True)
    print(f"{'=' * 50}", flush=True)
    print(f"  Keep:   {len(keep):>5} ({100*len(keep)/total:.1f}%)", flush=True)
    print(f"  Skip:   {len(skip):>5} ({100*len(skip)/total:.1f}%)", flush=True)

    paradigm_counts = Counter(c["paradigm"] for c in classifies)
    print(f"\n  Paradigm Distribution:", flush=True)
    for p, cnt in paradigm_counts.most_common():
        print(f"    {p:<16} {cnt:>5} ({100*cnt/total:.1f}%)", flush=True)

    domain_l1_counts = Counter(c["domain_l1"] for c in classifies)
    print(f"\n  Domain L1 Distribution:", flush=True)
    for d, cnt in domain_l1_counts.most_common():
        print(f"    {d:<22} {cnt:>5} ({100*cnt/total:.1f}%)", flush=True)

    skip_reasons = Counter(c["feasibility"]["skip_reason"] or "not_skipped" for c in classifies)
    print(f"\n  Skip Reasons:", flush=True)
    for r, cnt in skip_reasons.most_common():
        print(f"    {r:<24} {cnt:>5} ({100*cnt/total:.1f}%)", flush=True)
    print(flush=True)

    # ── Chart 1: Paradigm donut ──
    _plot_paradigm_donut(classifies, total, fig_dir / "paradigm_distribution.png", mpl=mpl, plt=plt)

    # ── Chart 2: Domain nested donut (L1 inner, L2 outer) ──
    _plot_domain_donut(classifies, total, fig_dir / "domain_distribution.png", mpl=mpl, plt=plt)

    print(f"  Figures saved to: {fig_dir}/", flush=True)


def _plot_paradigm_donut(classifies: list[dict], total: int, outpath: Path, mpl, plt):
    """Donut chart for paradigm distribution."""
    counts = Counter(c["paradigm"] for c in classifies)
    labels_sorted = sorted(counts.keys(), key=lambda x: -counts[x])
    sizes = [counts[p] for p in labels_sorted]
    colors = [PARADIGM_COLORS.get(p, "#999") for p in labels_sorted]
    display_labels = [
        f"{p}\n({counts[p]}, {100 * counts[p] / total:.0f}%)" for p in labels_sorted
    ]

    fig, ax = plt.subplots(figsize=(8, 8))
    wedge_kwargs = dict(edgecolor="white", linewidth=1.5, width=0.4)
    wedges, texts = ax.pie(
        sizes, radius=1.0, colors=colors,
        labels=display_labels, labeldistance=1.15,
        textprops={"fontsize": 9, "fontweight": "bold"},
        wedgeprops=wedge_kwargs, startangle=90,
    )
    ax.set_title(f"Paradigm Distribution (N={total})", fontsize=13, pad=20)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def _plot_domain_donut(classifies: list[dict], total: int, outpath: Path, mpl, plt):
    """Nested donut: inner = domain_l1, outer = domain_l2."""
    l1_counter: Counter = Counter()
    l2_counter: Counter = Counter()
    for c in classifies:
        d1 = c.get("domain_l1", "other")
        d2 = c.get("domain_l2", "other")
        l1_counter[d1] += 1
        l2_counter[(d1, d2)] += 1

    l1_sorted = sorted(l1_counter.keys(), key=lambda x: -l1_counter[x])

    inner_sizes, inner_colors, inner_labels = [], [], []
    outer_sizes, outer_colors, outer_labels = [], [], []

    for d1 in l1_sorted:
        inner_sizes.append(l1_counter[d1])
        inner_colors.append(DOMAIN_L1_COLORS.get(d1, "#999"))
        pct = l1_counter[d1] / total * 100
        inner_labels.append(f"{d1}\n({l1_counter[d1]}, {pct:.0f}%)")

        subs = sorted(
            [(k, v) for k, v in l2_counter.items() if k[0] == d1],
            key=lambda x: -x[1],
        )
        base_color = mpl.colors.to_rgba(DOMAIN_L1_COLORS.get(d1, "#999"))
        for i, ((_, d2), cnt) in enumerate(subs):
            outer_sizes.append(cnt)
            alpha = 0.5 + 0.5 * (1 - i / max(len(subs), 1))
            c = (*base_color[:3], alpha)
            outer_colors.append(c)
            pct2 = cnt / total * 100
            outer_labels.append(f"{d2}\n({cnt})" if pct2 >= 3 else "")

    fig, ax = plt.subplots(figsize=(10, 10))
    wedge_kwargs = dict(edgecolor="white", linewidth=1.5)
    ax.pie(inner_sizes, radius=0.65, colors=inner_colors,
           labels=inner_labels, labeldistance=0.35,
           textprops={"fontsize": 8, "fontweight": "bold"},
           wedgeprops=wedge_kwargs, startangle=90)
    ax.pie(outer_sizes, radius=1.0, colors=outer_colors,
           labels=outer_labels, labeldistance=1.12,
           textprops={"fontsize": 7},
           wedgeprops={**wedge_kwargs, "width": 0.3}, startangle=90)
    ax.set_title(f"Domain Distribution (N={total})", fontsize=13, pad=20)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 1: paradigm + domain + feasibility classification")
    parser.add_argument("--jsonl", required=True, help="Input JSONL (screen_keep format)")
    parser.add_argument("--frames-dir", required=True, help="Root dir of pre-extracted frame dirs")
    parser.add_argument("--output-dir", required=True, help="Output dir for results + figures")
    parser.add_argument("--api-base", default="https://api.novita.ai/v3/openai")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default="pa/gemini-3.1-pro-preview")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--resize-max-width", type=int, default=0)
    parser.add_argument("--jpeg-quality", type=int, default=60)
    parser.add_argument("--limit", type=int, default=0, help="Max clips to process (0=all)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--visualize-only", action="store_true",
                        help="Skip classification, only generate charts from existing results")
    parser.add_argument("--skip-viz", action="store_true", help="Skip visualization after classification")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_base = Path(args.frames_dir)

    # ── Visualize-only mode ──
    if args.visualize_only:
        existing = load_existing_results(output_dir)
        if not existing:
            print("ERROR: no existing classify_results.jsonl found.", file=sys.stderr)
            sys.exit(1)
        print(f"Loaded {len(existing)} existing results.", flush=True)
        run_visualization(list(existing.values()), args.output_dir)
        return

    # ── Load input ──
    records = load_jsonl(args.jsonl)
    if args.limit > 0:
        records = records[:args.limit]
    print(f"Input: {len(records)} records from {args.jsonl}", flush=True)

    # ── Load existing results for idempotency ──
    existing = load_existing_results(output_dir)
    done_keys = set(existing.keys())
    print(f"Existing results: {len(done_keys)} (will skip unless --overwrite)", flush=True)

    # ── Classify ──
    reset_token_usage()
    new_results: dict[str, dict] = {}  # clip_key → output record
    ok_count = skip_count = err_count = 0
    t0 = time.time()

    print(f"\nClassifying with model={args.model} workers={args.workers} "
          f"max_frames={args.max_frames}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                classify_one,
                rec, frames_base, done_keys,
                args.api_base, args.api_key, args.model,
                args.max_frames, args.resize_max_width, args.jpeg_quality,
                args.overwrite,
            ): rec
            for rec in records
        }

        for future in as_completed(futures):
            rec = futures[future]
            try:
                result = future.result()
            except Exception as e:
                err_count += 1
                print(f"  [ERROR] {e}", flush=True)
                continue

            key = result["clip_key"]
            if result["skipped"]:
                skip_count += 1
                continue
            elif result["ok"]:
                ok_count += 1
                new_results[key] = build_output_record(rec, result["classification"])
            else:
                err_count += 1
                print(f"  [{key}] ERROR: {result['error']}", flush=True)

    elapsed = time.time() - t0
    usage = get_token_usage()
    print(f"\nDone: {ok_count} classified, {skip_count} skipped, {err_count} errors "
          f"({elapsed:.1f}s)", flush=True)
    print(f"Token usage: prompt={usage['prompt_tokens']:,} "
          f"completion={usage['completion_tokens']:,} "
          f"calls={usage['api_calls']}", flush=True)

    # ── Merge existing + new and write output ──
    all_results = dict(existing)
    all_results.update(new_results)

    all_records = list(all_results.values())
    keep_records = [r for r in all_records if not r.get("_classify", {}).get("feasibility", {}).get("skip", True)]
    reject_records = [r for r in all_records if r.get("_classify", {}).get("feasibility", {}).get("skip", False)]

    write_jsonl(output_dir / "classify_results.jsonl", all_records)
    write_jsonl(output_dir / "classify_keep.jsonl", keep_records)
    write_jsonl(output_dir / "classify_reject.jsonl", reject_records)

    print(f"\nWritten: {len(all_records)} total, {len(keep_records)} keep, "
          f"{len(reject_records)} reject", flush=True)

    # ── Visualization ──
    if not args.skip_viz:
        run_visualization(all_records, args.output_dir)


if __name__ == "__main__":
    main()
