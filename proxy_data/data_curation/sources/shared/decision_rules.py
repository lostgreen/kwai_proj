"""
Programmatic decision rules for the data curation pipeline.

These rules override or supplement LLM decisions with hard thresholds.
Designed to catch cases where the LLM is inconsistent.

Rule sets:
  - apply_richness_rules(): For Stage A (boundary clarity + phase diversity)
  - apply_group_d_rules(): For Group D / VLM-Curated sources (physical hierarchy score)

Note: ET-Instruct Stage A uses source-based routing with per-group inline rules
      defined in et_instruct_164k/stage_a_coarse_filter.py (not in this file).

Legacy:
  - apply_stage_a_rules(): Old L2-granularity rules (deprecated, kept for reference)

Usage:
    # As a standalone post-processing step
    python decision_rules.py \\
        --input results/stage_a_results.jsonl \\
        --output results/stage_a_ruled.jsonl \\
        --stage A

    # Or import in your pipeline
    from shared.decision_rules import apply_richness_rules
"""

import json
import argparse
import os
from pathlib import Path


# ── Video Richness Rules (current Stage A) ───────────────

def apply_richness_rules(assessment: dict) -> str:
    """Apply programmatic rules to video richness assessment.

    Returns the corrected decision: "keep" | "reject"

    Rules:
    1. Parse error -> reject
    2. boundary_clarity >= 3 AND phase_diversity >= 3 -> keep
    3. Otherwise -> reject
    """
    if assessment.get("_parse_error") or assessment.get("error"):
        return "reject"

    boundary = assessment.get("boundary_clarity_score", 0)
    diversity = assessment.get("phase_diversity_score", 0)

    # Ensure numeric
    for val in [boundary, diversity]:
        if not isinstance(val, (int, float)):
            return "reject"

    # Keep: clear boundaries + diverse phases
    if boundary >= 3 and diversity >= 3:
        return "keep"

    return "reject"


# ── Group D Rules (VLM-Curated sources) ──────────────────

def apply_group_d_rules(assessment: dict) -> str:
    """Apply rules for Group D (VLM-Curated sources like TimeLens).

    Returns the corrected decision: "keep" | "reject"

    Rules:
    1. Parse error or pre-filter -> reject
    2. physical_hierarchy_score >= 3 -> keep
    3. Otherwise -> reject
    """
    if assessment.get("_parse_error") or assessment.get("error"):
        return "reject"
    if assessment.get("_prefilter"):
        return "reject"

    score = assessment.get("physical_hierarchy_score", 0)
    if not isinstance(score, (int, float)):
        return "reject"

    if score >= 3:
        return "keep"
    return "reject"


# ── Stage A Decision Rules (DEPRECATED — kept for reference) ──

def apply_stage_a_rules(assessment: dict) -> str:
    """[DEPRECATED] Old L2-granularity rules. Use apply_richness_rules() instead.

    Returns the corrected decision: "keep" | "maybe" | "reject"

    Rules (in priority order):
    1. Parse error -> reject
    2. mostly_L1_like or mostly_L3_like -> reject
    3. l2_fit_score <= 2 -> reject
    4. mostly_L2_like + l2_fit_score >= 4 + good + low mixed -> keep
    5. mixed or l2_fit_score == 3 -> maybe
    6. Default -> maybe
    """
    if assessment.get("_parse_error") or assessment.get("error"):
        return "reject"

    label = assessment.get("granularity_label", "")
    score = assessment.get("l2_fit_score", 0)
    issue = assessment.get("granularity_issue", "")
    mixed_ratio = assessment.get("mixed_ratio_estimate", "")

    # Hard reject
    if label in ("mostly_L1_like", "mostly_L3_like"):
        return "reject"
    if isinstance(score, (int, float)) and score <= 2:
        return "reject"

    # Hard keep
    if (
        label == "mostly_L2_like"
        and isinstance(score, (int, float)) and score >= 4
        and issue == "good"
        and mixed_ratio == "low"
    ):
        return "keep"

    # Gray zone
    if label == "mixed":
        return "maybe"
    if isinstance(score, (int, float)) and score == 3:
        return "maybe"

    # Soft keep: mostly_L2_like but not all criteria met perfectly
    if label == "mostly_L2_like" and isinstance(score, (int, float)) and score >= 4:
        return "keep"

    # Soft maybe for everything else
    return "maybe"


# ── CLI for post-processing ──────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Apply programmatic decision rules to LLM assessment results",
    )
    parser.add_argument("--input", required=True, help="assessed .jsonl")
    parser.add_argument("--output", required=True, help="ruled .jsonl")
    parser.add_argument("--stage", required=True, choices=["A"],
                        help="Which stage's rules to apply (A=richness/group_d)")
    parser.add_argument("--override", action="store_true",
                        help="Override LLM decision with rule-based decision")
    args = parser.parse_args()

    rule_fn = apply_richness_rules

    results = []
    overrides = 0
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            assessment = sample.get("_assessment", {})

            # For Stage A, auto-detect Group D vs default richness rules
            if args.stage == "A" and sample.get("_group") == "D":
                rule_decision = apply_group_d_rules(assessment)
            else:
                rule_decision = rule_fn(assessment)
            llm_decision = assessment.get("decision", "unknown")

            sample["_rule_decision"] = rule_decision
            if llm_decision != rule_decision:
                sample["_decision_override"] = True
                overrides += 1
                if args.override:
                    assessment["decision"] = rule_decision
                    assessment["_original_decision"] = llm_decision
            results.append(sample)

    # Stats
    print(f"处理 {len(results)} 条样本")
    print(f"规则与 LLM 不一致: {overrides} ({overrides/max(len(results),1)*100:.1f}%)")
    if args.override:
        print(f"已用规则覆盖 LLM decision")

    decisions = {}
    for r in results:
        d = r.get("_assessment", {}).get("decision", "unknown")
        decisions[d] = decisions.get(d, 0) + 1
    print(f"最终 decision 分布:")
    for d, c in sorted(decisions.items(), key=lambda x: -x[1]):
        print(f"  {d}: {c}")

    # Write
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n-> {args.output}")


if __name__ == "__main__":
    main()
