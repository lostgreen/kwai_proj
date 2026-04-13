#!/usr/bin/env python3
"""
reclassify_domain.py — Re-classify domain_l2 from video_caption for annotations stuck on "other".

Reads annotation JSONs, finds those with domain_l2=="other", sends the video_caption
to an LLM (text-only, no images) to pick the best domain_l2, and writes updated JSONs
to a separate output directory (original files are never modified).

Usage:
    # Dry-run (no writes, print what would change)
    python reclassify_domain.py \
        --ann-dir /path/to/annotations \
        --model pa/gemini-3.1-pro-preview \
        --dry-run

    # Apply changes (writes to {ann-dir}_reclassified/ by default)
    python reclassify_domain.py \
        --ann-dir /path/to/annotations \
        --model pa/gemini-3.1-pro-preview \
        --workers 8

    # Specify output dir
    python reclassify_domain.py \
        --ann-dir /path/to/annotations \
        --output-dir /path/to/output \
        --workers 8

    # Reclassify ALL annotations (not just "other")
    python reclassify_domain.py \
        --ann-dir /path/to/annotations \
        --all --workers 8
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from archetypes import (
    DOMAIN_L2_ALL,
    DOMAIN_L2_TO_L1,
    DOMAIN_L1_TO_L2,
    DOMAIN_L1_ALL,
    resolve_domain_l1,
)

# ─────────────────────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────────────────────

def _build_domain_list() -> str:
    """Format the 2-level domain taxonomy for the prompt."""
    lines = []
    for l1 in sorted(DOMAIN_L1_ALL):
        l2_list = sorted(DOMAIN_L1_TO_L2.get(l1, []))
        lines.append(f"  {l1}:")
        for l2 in l2_list:
            lines.append(f"    - {l2}")
    return "\n".join(lines)


_RECLASSIFY_PROMPT = """\
You are a video content classifier. Given a video caption, classify it into the \
2-level domain taxonomy below.

## Domain Taxonomy (L1 → L2)
{domain_list}

## Rules
1. Choose domain_l1 from: {domain_l1_list}
2. Choose domain_l2 from the L2 categories under that L1.
3. If the video clearly fits a category, choose it — do NOT default to "other".
4. Only use domain_l2="other" if the video genuinely does not fit ANY of the 22 categories.
5. If you pick domain_l2="other", you MUST provide a domain_l2_suggestion — a short free-text \
label describing what the domain should be (e.g. "puzzle_toys", "medical_procedure").
6. Consider the primary topic, not incidental elements. E.g. a cooking vlog is "food_cooking" \
(task_howto), not "daily_vlog" (lifestyle_vlog).

## Video Caption
{caption}

## Output
Reply with ONLY a JSON object:
{{"domain_l1": "<L1 category>", "domain_l2": "<L2 category or other>", "domain_l2_suggestion": "<only if domain_l2 is other, else empty string>"}}"""


def build_prompt(caption: str) -> str:
    return _RECLASSIFY_PROMPT.format(
        domain_list=_build_domain_list(),
        domain_l1_list=", ".join(sorted(DOMAIN_L1_ALL)),
        caption=caption,
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLM call (text-only, OpenAI-compatible)
# ─────────────────────────────────────────────────────────────────────────────

def call_llm(
    api_base: str, api_key: str, model: str,
    prompt: str,
    max_tokens: int = 128,
    temperature: float = 0.0,
    retries: int = 3,
) -> str:
    from openai import OpenAI

    # Provider-specific key resolution
    if api_key:
        key = api_key
    elif "novita.ai" in api_base.lower():
        key = os.environ.get("NOVITA_API_KEY", "")
    elif "openrouter.ai" in api_base.lower():
        key = os.environ.get("OPENROUTER_API_KEY", "")
    else:
        key = os.environ.get("OPENAI_API_KEY", "")

    client = OpenAI(api_key=key, base_url=api_base)
    messages = [{"role": "user", "content": prompt}]

    last_error = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages,
                max_tokens=max_tokens, temperature=temperature,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            last_error = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise last_error


def parse_domain_response(text: str) -> dict | None:
    """Extract domain_l1, domain_l2, domain_l2_suggestion from LLM JSON response."""
    import re
    text = text.strip()
    # Try direct JSON parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "domain_l2" in obj:
            return obj
    except json.JSONDecodeError:
        pass
    # Try extracting from markdown code block
    m = re.search(r"```(?:json)?\s*(\{[\s\S]+?\})\s*```", text)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict) and "domain_l2" in obj:
                return obj
        except json.JSONDecodeError:
            pass
    # Last resort: find any JSON object
    m2 = re.search(r"\{[\s\S]+\}", text)
    if m2:
        try:
            obj = json.loads(m2.group(0))
            if isinstance(obj, dict) and "domain_l2" in obj:
                return obj
        except json.JSONDecodeError:
            pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Single annotation reclassification
# ─────────────────────────────────────────────────────────────────────────────

def reclassify_one(
    json_path: Path,
    api_base: str, api_key: str, model: str,
    dry_run: bool,
    output_dir: Path | None = None,
) -> dict:
    """Reclassify domain for one annotation JSON. Returns status dict."""
    clip_key = json_path.stem
    try:
        with open(json_path, encoding="utf-8") as f:
            ann = json.load(f)
    except Exception as e:
        return {"key": clip_key, "status": "error", "error": f"load failed: {e}"}

    caption = ann.get("video_caption", "").strip()
    if not caption:
        return {"key": clip_key, "status": "skip", "reason": "no caption"}

    old_l2 = ann.get("domain_l2", "other")
    old_l1 = ann.get("domain_l1", "other")

    prompt = build_prompt(caption)
    try:
        raw = call_llm(api_base, api_key, model, prompt)
    except Exception as e:
        return {"key": clip_key, "status": "error", "error": f"LLM call failed: {e}"}

    parsed = parse_domain_response(raw)
    if parsed is None:
        return {"key": clip_key, "status": "error",
                "error": f"invalid response: {raw[:200]}"}

    new_l2 = parsed.get("domain_l2", "other")
    if new_l2 not in DOMAIN_L2_ALL:
        new_l2 = "other"

    # Validate/resolve domain_l1
    llm_l1 = parsed.get("domain_l1", "")
    new_l1 = resolve_domain_l1(new_l2)  # authoritative L1 from L2 mapping
    if new_l2 == "other" and llm_l1 in DOMAIN_L1_ALL:
        new_l1 = llm_l1  # trust LLM's L1 when L2 is "other"

    suggestion = str(parsed.get("domain_l2_suggestion", "")).strip()

    if new_l2 == old_l2 and new_l1 == old_l1:
        return {"key": clip_key, "status": "unchanged",
                "domain_l2": old_l2, "domain_l1": old_l1}

    if dry_run:
        info = {"key": clip_key, "status": "would_change",
                "old": f"{old_l1}/{old_l2}", "new": f"{new_l1}/{new_l2}",
                "caption": caption[:120]}
        if suggestion:
            info["suggestion"] = suggestion
        return info

    # Write updated JSON to output dir
    ann["domain_l2"] = new_l2
    ann["domain_l1"] = new_l1
    if new_l2 == "other" and suggestion:
        ann["domain_l2_suggestion"] = suggestion
    else:
        ann.pop("domain_l2_suggestion", None)
        ann["domain_l2_note"] = ""

    out_path = (output_dir or json_path.parent) / json_path.name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ann, f, ensure_ascii=False, indent=2)

    return {"key": clip_key, "status": "changed",
            "old": f"{old_l1}/{old_l2}", "new": f"{new_l1}/{new_l2}"}


# ─────────────────────────────────────────────────────────────────────────────
# Visualization: before/after domain distribution
# ─────────────────────────────────────────────────────────────────────────────

DOMAIN_L1_COLORS: dict[str, str] = {
    "knowledge_education": "#4C72B0",
    "film_entertainment": "#C44E52",
    "sports_esports": "#CCB974",
    "lifestyle_vlog": "#8172B2",
    "arts_performance": "#55A868",
    "task_howto": "#64B5CD",
    "other": "#999999",
}


def plot_before_after(
    before_domains: dict[str, str],
    ann_dir: Path,
    output_dir: Path,
    out_path: Path,
) -> None:
    """Generate a side-by-side before/after domain_l2 nested donut chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    # Collect after-state: start from original, overlay with output_dir
    after_domains = dict(before_domains)
    for jp in sorted(output_dir.glob("*.json")):
        try:
            with open(jp, encoding="utf-8") as f:
                a = json.load(f)
            after_domains[jp.stem] = a.get("domain_l2", "other")
        except Exception:
            continue

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for ax, domains, title in [
        (axes[0], before_domains, "Before Reclassification"),
        (axes[1], after_domains, "After Reclassification"),
    ]:
        total = len(domains)
        if total == 0:
            ax.set_title(f"{title} (no data)")
            continue

        l1_counter: Counter = Counter()
        l2_counter: Counter = Counter()
        for d2 in domains.values():
            d1 = resolve_domain_l1(d2)
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

        wedge_kwargs = dict(edgecolor="white", linewidth=1.5)
        ax.pie(inner_sizes, radius=0.65, colors=inner_colors,
               labels=inner_labels, labeldistance=0.35,
               textprops={"fontsize": 7, "fontweight": "bold"},
               wedgeprops=wedge_kwargs, startangle=90)
        ax.pie(outer_sizes, radius=1.0, colors=outer_colors,
               labels=outer_labels, labeldistance=1.12,
               textprops={"fontsize": 6},
               wedgeprops={**wedge_kwargs, "width": 0.3}, startangle=90)
        ax.set_title(f"{title} (N={total})", fontsize=12, pad=15)

    fig.suptitle("Domain Reclassification: Before vs After", fontsize=14, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved to: {out_path}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Re-classify domain_l2 from video_caption for 'other' annotations")
    parser.add_argument("--ann-dir", required=True,
                        help="Directory containing annotation JSONs ({clip_key}.json)")
    parser.add_argument("--api-base", default="https://api.novita.ai/v3/openai")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default="pa/gemini-3.1-pro-preview")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would change without writing files")
    parser.add_argument("--all", action="store_true",
                        help="Reclassify ALL annotations, not just domain_l2=='other'")
    parser.add_argument("--keys", nargs="*", default=None,
                        help="Only reclassify these specific clip keys")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max annotations to process (0=all)")
    parser.add_argument("--output-dir", default=None,
                        help="Output dir for updated JSONs (default: {ann-dir}_reclassified)")
    parser.add_argument("--skip-viz", action="store_true",
                        help="Skip before/after domain distribution chart")
    args = parser.parse_args()

    ann_dir = Path(args.ann_dir)
    if not ann_dir.is_dir():
        print(f"ERROR: {ann_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Collect annotation files
    json_files = sorted(ann_dir.glob("*.json"))
    print(f"Found {len(json_files)} annotation JSONs in {ann_dir}", flush=True)

    # Filter by keys if specified
    if args.keys:
        key_set = set(args.keys)
        json_files = [f for f in json_files if f.stem in key_set]
        print(f"Filtered to {len(json_files)} by --keys", flush=True)

    # Filter to domain_l2=="other" unless --all
    if not args.all and not args.keys:
        targets = []
        for jp in json_files:
            try:
                with open(jp, encoding="utf-8") as f:
                    ann = json.load(f)
                if ann.get("domain_l2", "") == "other":
                    targets.append(jp)
            except Exception:
                continue
        print(f"Filtered to {len(targets)} with domain_l2=='other'", flush=True)
        json_files = targets

    if args.limit > 0:
        json_files = json_files[:args.limit]

    if not json_files:
        print("Nothing to reclassify.", flush=True)
        return

    print(f"\nReclassifying {len(json_files)} annotations "
          f"(model={args.model}, workers={args.workers}, dry_run={args.dry_run})\n",
          flush=True)

    # Resolve output dir
    output_dir = Path(args.output_dir) if args.output_dir else ann_dir.parent / f"{ann_dir.name}_reclassified"
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output dir: {output_dir}", flush=True)

    # Collect before-state for visualization
    before_domains: dict[str, str] = {}  # clip_key → domain_l2
    all_json_for_viz = sorted(ann_dir.glob("*.json"))
    for jp in all_json_for_viz:
        if jp.suffix != ".json":
            continue
        try:
            with open(jp, encoding="utf-8") as f:
                a = json.load(f)
            before_domains[jp.stem] = a.get("domain_l2", "other")
        except Exception:
            continue

    # Run reclassification
    results = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                reclassify_one, jp,
                args.api_base, args.api_key, args.model,
                args.dry_run, output_dir,
            ): jp
            for jp in json_files
        }
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                result = {"key": futures[future].stem, "status": "error", "error": str(e)}
            results.append(result)

            # Live progress
            status = result["status"]
            key = result["key"]
            if status in ("changed", "would_change"):
                print(f"  [{key}] {result['old']} → {result['new']}", flush=True)
            elif status == "error":
                print(f"  [{key}] ERROR: {result.get('error', '?')}", flush=True)

    elapsed = time.time() - t0

    # Summary
    status_counts = Counter(r["status"] for r in results)
    print(f"\n{'=' * 50}", flush=True)
    print(f"  Reclassification Summary ({elapsed:.1f}s)", flush=True)
    print(f"{'=' * 50}", flush=True)
    for s, cnt in status_counts.most_common():
        print(f"  {s:<16} {cnt:>5}", flush=True)

    # Domain transition summary
    changed = [r for r in results if r["status"] in ("changed", "would_change")]
    if changed:
        new_domain_counts = Counter(r["new"].split("/")[1] for r in changed)
        print(f"\n  New domain_l2 distribution (N={len(changed)}):", flush=True)
        for d, cnt in new_domain_counts.most_common():
            print(f"    {d:<22} {cnt:>5}", flush=True)

    # Visualization
    if not args.skip_viz and not args.dry_run and changed:
        fig_path = output_dir / "figures" / "domain_reclassify_before_after.png"
        plot_before_after(before_domains, ann_dir, output_dir, fig_path)


if __name__ == "__main__":
    main()
