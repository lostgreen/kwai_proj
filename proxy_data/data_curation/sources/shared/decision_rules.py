"""
Programmatic decision rules for the two-stage curation pipeline.

These rules override or supplement LLM decisions with hard thresholds.
Designed to catch cases where the LLM is inconsistent (e.g., gives
l2_fit_score=2 but decision="keep").

Usage:
    # As a standalone post-processing step
    python decision_rules.py \\
        --input results/stage_a_results.jsonl \\
        --output results/stage_a_ruled.jsonl \\
        --stage A

    # Or import in your pipeline
    from shared.decision_rules import apply_stage_a_rules, apply_stage_b_rules
"""

import json
import argparse
import os
from pathlib import Path


# ── Stage A Decision Rules ───────────────────────────────

def apply_stage_a_rules(assessment: dict) -> str:
    """Apply programmatic rules to Stage A assessment.

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


# ── Stage B Decision Rules ───────────────────────────────

def apply_stage_b_rules(assessment: dict) -> str:
    """Apply programmatic rules to Stage B assessment.

    Returns the corrected decision: "keep" | "maybe" | "reject"

    Rules:
    1. Parse error -> reject
    2. overall_score <= 2 -> reject
    3. Any dimension (l1/l3/temporal) <= 1 -> reject
    4. overall_score >= 4 AND all dimensions >= 3 -> keep
    5. overall_score == 3 -> maybe
    6. Default -> maybe
    """
    if assessment.get("_parse_error") or assessment.get("error"):
        return "reject"

    overall = assessment.get("overall_score", 0)
    l1 = assessment.get("l1_potential", 0)
    l3 = assessment.get("l3_potential", 0)
    temporal = assessment.get("temporal_structure", 0)

    # Ensure numeric
    for val in [overall, l1, l3, temporal]:
        if not isinstance(val, (int, float)):
            return "maybe"

    # Hard reject
    if overall <= 2:
        return "reject"
    if any(v <= 1 for v in [l1, l3, temporal]):
        return "reject"

    # Hard keep
    if overall >= 4 and all(v >= 3 for v in [l1, l3, temporal]):
        return "keep"

    # Gray zone
    if overall == 3:
        return "maybe"

    # overall >= 4 but some dimension weak
    if overall >= 4:
        return "maybe"

    return "maybe"


# ── Combined Final Decision ──────────────────────────────

def final_decision(stage_a_assessment: dict, stage_b_assessment: dict) -> str:
    """Combine Stage A and Stage B into a final decision.

    Both stages must agree on "keep" for final keep.
    """
    a_decision = apply_stage_a_rules(stage_a_assessment)
    b_decision = apply_stage_b_rules(stage_b_assessment)

    if a_decision == "reject" or b_decision == "reject":
        return "reject"
    if a_decision == "keep" and b_decision == "keep":
        return "keep"
    return "maybe"


# ── CLI for post-processing ──────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Apply programmatic decision rules to LLM assessment results",
    )
    parser.add_argument("--input", required=True, help="assessed .jsonl")
    parser.add_argument("--output", required=True, help="ruled .jsonl")
    parser.add_argument("--stage", required=True, choices=["A", "B"],
                        help="Which stage's rules to apply")
    parser.add_argument("--override", action="store_true",
                        help="Override LLM decision with rule-based decision")
    args = parser.parse_args()

    rule_fn = apply_stage_a_rules if args.stage == "A" else apply_stage_b_rules

    results = []
    overrides = 0
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            assessment = sample.get("_assessment", {})
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
