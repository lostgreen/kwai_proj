#!/usr/bin/env python3
"""
Rewrite annotation criterion fields into generic training hints.

Reads annotation JSONs containing VLM-generated criterion fields
(global_phase_criterion, event_split_criterion, micro_split_criterion),
calls an LLM to rewrite them into content-agnostic segmentation hints,
and writes the results back as new *_hint fields in the same JSON.

Usage:
    python rewrite_criteria_hints.py \
        --annotation-dir annotations/ \
        --api-base https://api.example.com/v1 \
        --model gpt-4o-mini \
        --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# LLM rewrite prompt
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a text rewriting assistant. Your task is to rewrite video segmentation \
criterion sentences into generic, content-agnostic training hints.

Rules:
1. Remove ALL references to specific video content (object names, actions, \
   materials, tools, body parts, food items, etc.).
2. Keep ONLY the structural/logical reasoning about segmentation granularity.
3. Output must be ONE concise sentence.
4. If the input is empty or meaningless, output an empty string.

Examples:
- Input: "Segmented by the distinct sub-tasks of removing wires, disconnecting \
  hoses/brackets, and unbolting the cover."
  Output: "Segmented by distinct sequential sub-tasks that each complete a \
  specific sub-goal within the phase."

- Input: "This is a repetitive recreational activity with no sequential \
  progression, so no event segmentation is needed."
  Output: "Single repetitive activity with no sequential progression; no \
  sub-event segmentation needed."

- Input: "Broke down by individual state-changing operations where material \
  visibly transforms."
  Output: "Segmented by individual atomic operations where a visible state \
  change occurs."

- Input: ""
  Output: ""
"""

_USER_TEMPLATE = """\
Rewrite the following segmentation criterion into a generic training hint. \
Remove all video-specific content references, keep only the structural \
segmentation logic.

Criterion: {criterion}

Output only the rewritten hint (one sentence), nothing else."""


# ─────────────────────────────────────────────────────────────────────────────
# LLM call (text-only, no images)
# ─────────────────────────────────────────────────────────────────────────────

def _call_llm(
    client,
    model: str,
    criterion: str,
    retries: int = 2,
) -> str:
    """Call LLM to rewrite a single criterion string."""
    if not criterion or not criterion.strip():
        return ""

    user_text = _USER_TEMPLATE.format(criterion=criterion)

    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_text},
                ],
                max_tokens=256,
                temperature=0.0,
            )
            text = resp.choices[0].message.content.strip()
            # Remove surrounding quotes if present
            if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
                text = text[1:-1]
            return text
        except Exception as e:
            if attempt == retries:
                print(f"  WARN: LLM call failed after {retries + 1} attempts: {e}")
                return ""
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Per-annotation processing
# ─────────────────────────────────────────────────────────────────────────────

def _process_annotation(
    ann_path: Path,
    client,
    model: str,
    dry_run: bool = False,
) -> dict:
    """Rewrite criterion fields in one annotation JSON."""
    with open(ann_path) as f:
        ann = json.load(f)

    clip_key = ann.get("clip_key", ann_path.stem)
    changed = False

    # 1. global_phase_criterion → global_phase_hint
    gpc = ann.get("global_phase_criterion", "")
    if gpc and "global_phase_hint" not in ann:
        if dry_run:
            print(f"  [dry] {clip_key}: would rewrite global_phase_criterion")
        else:
            hint = _call_llm(client, model, gpc)
            if hint:
                ann["global_phase_hint"] = hint
                changed = True

    # 2. event_split_criterion → event_split_hint (per phase)
    phases = (ann.get("level1") or {}).get("macro_phases") or []
    for phase in phases:
        esc = phase.get("event_split_criterion", "")
        if esc and "event_split_hint" not in phase:
            if dry_run:
                pid = phase.get("phase_id", "?")
                print(f"  [dry] {clip_key} phase {pid}: would rewrite event_split_criterion")
            else:
                hint = _call_llm(client, model, esc)
                if hint:
                    phase["event_split_hint"] = hint
                    changed = True

    # 3. micro_split_criterion → micro_split_hint (top-level L3)
    l3 = ann.get("level3") or {}
    msc = l3.get("micro_split_criterion", "")
    if msc and "micro_split_hint" not in l3:
        if dry_run:
            print(f"  [dry] {clip_key}: would rewrite micro_split_criterion")
        else:
            hint = _call_llm(client, model, msc)
            if hint:
                l3["micro_split_hint"] = hint
                changed = True

    # 3b. Also rewrite per-segment micro_split_criterion in _segment_calls
    for sc in l3.get("_segment_calls") or []:
        sc_msc = sc.get("micro_split_criterion", "")
        if sc_msc and "micro_split_hint" not in sc:
            if not dry_run:
                hint = _call_llm(client, model, sc_msc)
                if hint:
                    sc["micro_split_hint"] = hint
                    changed = True

    # Write back
    if changed and not dry_run:
        with open(ann_path, "w") as f:
            json.dump(ann, f, indent=2, ensure_ascii=False)
            f.write("\n")

    return {"clip_key": clip_key, "changed": changed, "path": str(ann_path)}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotation-dir", required=True,
                        help="Directory containing annotation JSON files")
    parser.add_argument("--api-base", default="",
                        help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", default="",
                        help="API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="Model name for rewriting (default: gpt-4o-mini)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers (default: 4)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be rewritten without calling LLM")
    args = parser.parse_args()

    ann_dir = Path(args.annotation_dir)
    ann_paths = sorted(ann_dir.glob("*.json"))
    if not ann_paths:
        print(f"No JSON files found in {ann_dir}")
        return

    print(f"Found {len(ann_paths)} annotation files in {ann_dir}")

    client = None
    if not args.dry_run:
        from openai import OpenAI
        key = args.api_key or os.environ.get("OPENAI_API_KEY") or ""
        client = OpenAI(api_key=key, base_url=args.api_base or None)

    done = changed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_process_annotation, p, client, args.model, args.dry_run): p
            for p in ann_paths
        }
        for fut in as_completed(futures):
            result = fut.result()
            done += 1
            if result["changed"]:
                changed += 1
                print(f"  [{done}/{len(ann_paths)}] {result['clip_key']} — hints written")
            elif done % 50 == 0:
                print(f"  [{done}/{len(ann_paths)}] ...")

    print(f"\nDone. {changed}/{done} files updated with training hints.")


if __name__ == "__main__":
    main()
