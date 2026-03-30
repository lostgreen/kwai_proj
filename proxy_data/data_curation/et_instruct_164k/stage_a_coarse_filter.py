"""
Stage A — 视频内容潜力评估 (ET-Instruct-164K) — Source Routing 版

基于 source 字段将样本分为三组，使用不同的 Prompt 策略：
  - Group A (Dense Manual): coin, activitynet_captions, tacos, didemo, charades_sta
    → 考核边界清晰度 + 篇章多样性
  - Group B (Coarse Manual): activitynet, hacs, thumos14
    → 只考核物理丰富度（粗标签场景）
  - Group C (ASR/Auto): how_to_step, how_to_caption, queryd, ego4d_naq
    → 忽略时间戳，只看文本是否描述多步骤物理操作

未匹配的 source 默认走 Group A。

输入: text_filter.py 产出的 passed.jsonl
输出:
  - stage_a_results.jsonl     — 全部评估结果
  - stage_a_results_keep.jsonl
  - stage_a_results_reject.jsonl
"""

import json
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.llm_client import (
    call_llm,
    load_checkpoint,
    run_concurrent_assessment,
    stratified_sample,
    write_results,
)

# ── Source → Group Mapping ────────────────────────────────

GROUP_A_SOURCES = {"coin", "activitynet_captions", "tacos", "didemo", "charades_sta"}
GROUP_B_SOURCES = {"activitynet", "hacs", "thumos14"}
GROUP_C_SOURCES = {"how_to_step", "how_to_caption", "queryd", "ego4d_naq", "naq"}


def get_source_group(source: str) -> str:
    s = source.lower().strip()
    if s in GROUP_A_SOURCES:
        return "A"
    if s in GROUP_B_SOURCES:
        return "B"
    if s in GROUP_C_SOURCES:
        return "C"
    return "A"  # default


# ── Group A: Dense Manual — 边界清晰度 + 篇章多样性 ──────

SYSTEM_A = """\
You are a data curator selecting videos for a 3-level Hierarchical Temporal \
Segmentation task (Macro-phases L1 -> Local events L2 -> Atomic actions L3).

We are looking for videos that have BOTH clear action boundaries AND \
structural diversity. The video does NOT need to be a strict instructional \
tutorial; sports, daily life, and unstructured events are perfectly fine \
AS LONG AS they can be naturally grouped into distinct "chapters" or \
"macro-phases".

Criteria for a GOOD video (Keep):
1. Clear Boundaries: The annotations describe distinct, observable actions \
or events.
2. Phase Diversity (Crucial): The video progresses through different themes \
or stages (e.g., a sports video with "setup -> gameplay -> celebration", \
or a cooking video with "prep -> cook -> serve").

Criteria for REJECTION:
1. Flat Repetition (Looping): The exact same type of action is repeated \
over and over. These cannot be grouped into meaningful L1 macro-phases.
2. Monolithic Activity: The entire video is essentially one continuous \
state with no clear event boundaries.

## Examples

### Example 1 — diverse phases (keep)
Input: 7 woodworking segments (glue, veneer, trim, sand, lacquer, dry, coat)
Output: {"structural_analysis":"3 chapters: prep(1-2), shaping(3-4), finishing(5-7).","boundary_clarity_score":5,"phase_diversity_score":5,"decision":"keep"}

### Example 2 — flat repetition (reject)
Input: 6 tennis rally segments (serve, return, hit, hit, serve, return)
Output: {"structural_analysis":"Same hitting action repeated. No phase transitions.","boundary_clarity_score":3,"phase_diversity_score":1,"decision":"reject"}

Return ONLY valid JSON."""

USER_A = """\
Video duration: {duration:.1f}s
Domain: {source}
Number of annotated segments: {n_events}

Annotated segments:
{events_text}

Respond with ONLY valid JSON:
{{
  "structural_analysis": "<1-2 sentences>",
  "boundary_clarity_score": <1-5>,
  "phase_diversity_score": <1-5>,
  "decision": "keep | reject"
}}"""


# ── Group B: Coarse Manual — 物理丰富度 ──────────────────

SYSTEM_B = """\
You are a data curator selecting videos for a fine-grained action segmentation \
task. The videos in this batch typically have only 1-3 coarse activity labels \
(e.g., "playing basketball", "washing car").

Your job is to assess whether the UNDERLYING ACTIVITY is physically rich \
enough to support decomposition into many sub-steps and atomic actions.

Criteria for KEEP:
- The activity involves multiple distinct physical tools, objects, or body \
movements (e.g., "cooking" → chop, stir, fry, plate).
- The scene changes or evolves over time (e.g., "car repair" → open hood, \
inspect, replace part, close hood).

Criteria for REJECT:
- The activity is inherently monotonous with no phase changes (e.g., \
"running on a track" — the same motion for the entire duration).
- The activity is static or purely verbal (e.g., "giving a speech", \
"watching TV").

## Examples

### Example 1 — rich physical activity (keep)
Input: 1 segment "washing a car" covering 120s
Output: {"physical_analysis":"Car washing involves multiple sub-phases: rinse, soap, scrub different parts, rinse again, dry. Rich tool/state changes.","physical_richness_score":4,"decision":"keep"}

### Example 2 — monotonous activity (reject)
Input: 1 segment "jogging in the park" covering 180s
Output: {"physical_analysis":"Single repetitive motion throughout. No tool usage, no scene change, no distinct phases.","physical_richness_score":1,"decision":"reject"}

Return ONLY valid JSON."""

USER_B = """\
Video duration: {duration:.1f}s
Domain: {source}
Number of annotated segments: {n_events}

Annotated segments:
{events_text}

Respond with ONLY valid JSON:
{{
  "physical_analysis": "<1-2 sentences>",
  "physical_richness_score": <1-5>,
  "decision": "keep | reject"
}}"""


# ── Group C: ASR/Auto — 多步骤物理操作 ───────────────────

SYSTEM_C = """\
You are a data curator selecting videos for a fine-grained action segmentation \
task. The text annotations in this batch are auto-generated from ASR or \
machine inference — timestamps are unreliable and should be IGNORED.

Focus ONLY on the TEXT CONTENT of the annotations. Determine whether the \
text describes a multi-step PHYSICAL DEMONSTRATION (hands-on procedural \
activity with real objects).

Criteria for KEEP:
- Text describes hands-on procedures (cooking, crafting, repairing, \
assembling, makeup application, etc.).
- Multiple distinct physical steps are mentioned.

Criteria for REJECT:
- Vlog / lifestyle commentary without physical tasks.
- Gaming, software tutorials, or screen recordings.
- Pure theory / lectures / reviews with no physical demonstration.
- Chat, Q&A, or interview formats.

## Examples

### Example 1 — physical demonstration (keep)
Input: Segments describe "apply primer to wall", "use roller for even coat", "tape edges", "paint second layer"
Output: {"text_analysis":"Multi-step painting process with distinct tools and physical actions.","action_density_score":5,"is_physical_demo":true,"decision":"keep"}

### Example 2 — talking / vlog (reject)
Input: Segments describe "talks about best places to visit", "recommends restaurants", "shares travel tips"
Output: {"text_analysis":"Travel vlog / commentary. No physical demonstration or hands-on procedure.","action_density_score":1,"is_physical_demo":false,"decision":"reject"}

Return ONLY valid JSON."""

USER_C = """\
Video duration: {duration:.1f}s
Domain: {source}
Number of annotated segments: {n_events}

Annotated segments (IGNORE timestamps, focus on TEXT ONLY):
{events_text}

Respond with ONLY valid JSON:
{{
  "text_analysis": "<1-2 sentences>",
  "action_density_score": <1-5>,
  "is_physical_demo": true | false,
  "decision": "keep | reject"
}}"""


# ── Prompt Registry ──────────────────────────────────────

PROMPTS = {
    "A": (SYSTEM_A, USER_A),
    "B": (SYSTEM_B, USER_B),
    "C": (SYSTEM_C, USER_C),
}


# ── Decision Rules (per group) ───────────────────────────

def apply_rules(assessment: dict, group: str) -> str:
    """Apply group-specific decision rules."""
    if assessment.get("_parse_error") or assessment.get("error"):
        return "reject"

    if group == "A":
        boundary = assessment.get("boundary_clarity_score", 0)
        diversity = assessment.get("phase_diversity_score", 0)
        if not (isinstance(boundary, (int, float)) and isinstance(diversity, (int, float))):
            return "reject"
        return "keep" if diversity >= 3 else "reject"

    elif group == "B":
        richness = assessment.get("physical_richness_score", 0)
        if not isinstance(richness, (int, float)):
            return "reject"
        return "keep" if richness >= 3 else "reject"

    elif group == "C":
        density = assessment.get("action_density_score", 0)
        if not isinstance(density, (int, float)):
            return "reject"
        return "keep" if density >= 4 else "reject"

    return "reject"


# ── Event Parsing (ET-Instruct specific) ─────────────────

def parse_events(sample: dict) -> list[dict]:
    """Parse ET-Instruct events from tgt timestamps + GPT response text."""
    tgt = sample.get("tgt", [])
    gpt_text = ""
    for turn in sample.get("conversations", []):
        if turn.get("from") == "gpt":
            gpt_text = turn.get("value", "")
            break

    events = []
    n_events = len(tgt) // 2

    pattern = r'([\d.]+)\s*-\s*([\d.]+)\s*seconds?,\s*(.+?)(?=\d+\.?\d*\s*-\s*\d+\.?\d*\s*seconds?|$)'
    matches = re.findall(pattern, gpt_text, re.DOTALL)

    if matches and len(matches) >= n_events:
        for start_s, end_s, desc in matches:
            events.append({
                "start": float(start_s),
                "end": float(end_s),
                "description": desc.strip().rstrip('.').strip(),
            })
    else:
        for i in range(n_events):
            events.append({
                "start": float(tgt[i * 2]),
                "end": float(tgt[i * 2 + 1]),
                "description": f"(event {i+1})",
            })

    return events


def format_events_text(events: list[dict]) -> str:
    lines = []
    for i, ev in enumerate(events, 1):
        lines.append(f"  {i}. [{ev['start']:.1f}s - {ev['end']:.1f}s] {ev['description']}")
    return "\n".join(lines)


# ── Assessment ───────────────────────────────────────────

def assess_sample(
    sample: dict,
    api_base: str,
    api_key: str,
    model: str,
) -> dict:
    """Run Stage A assessment with source-based routing."""
    events = parse_events(sample)
    events_text = format_events_text(events)
    source = sample.get("source", "unknown")
    group = get_source_group(source)
    system_prompt, user_prompt = PROMPTS[group]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt.format(
            duration=sample.get("duration", 0),
            source=source,
            n_events=len(events),
            events_text=events_text,
        )},
    ]

    result = call_llm(messages, api_base, api_key, model)

    # Apply group-specific rules
    llm_decision = result.get("decision", "unknown")
    rule_decision = apply_rules(result, group)
    if llm_decision != rule_decision:
        result["_original_decision"] = llm_decision
        result["decision"] = rule_decision

    assessed = dict(sample)
    assessed["_assessment"] = result
    assessed["_stage"] = "A"
    assessed["_group"] = group
    assessed["_n_events"] = len(events)
    return assessed


def print_stats(results: list[dict]):
    """Print Stage A assessment statistics."""
    assessments = [r for r in results if "_assessment" in r and not r["_assessment"].get("_parse_error")]

    if not assessments:
        print("  无有效评估结果")
        return

    # Decision distribution
    decisions = {}
    for r in assessments:
        d = r["_assessment"].get("decision", "unknown")
        decisions[d] = decisions.get(d, 0) + 1
    print(f"\n  == Decision 分布 ==")
    for d, c in sorted(decisions.items(), key=lambda x: -x[1]):
        print(f"    {d}: {c} ({c/len(assessments)*100:.1f}%)")

    # Per-group stats
    group_stats: dict[str, dict] = {}
    for r in assessments:
        g = r.get("_group", "?")
        d = r["_assessment"].get("decision", "unknown")
        group_stats.setdefault(g, {"total": 0, "keep": 0, "reject": 0})
        group_stats[g]["total"] += 1
        group_stats[g][d] = group_stats[g].get(d, 0) + 1
    print(f"\n  == 各 Group 统计 ==")
    for g in sorted(group_stats):
        gs = group_stats[g]
        total = gs["total"]
        keep = gs.get("keep", 0)
        print(f"    Group {g}: keep={keep}/{total} ({keep/total*100:.1f}%)")

    # Group-specific scores
    for g, score_field in [("A", "phase_diversity_score"), ("B", "physical_richness_score"), ("C", "action_density_score")]:
        group_items = [r["_assessment"] for r in assessments if r.get("_group") == g]
        scores = [a.get(score_field, 0) for a in group_items if isinstance(a.get(score_field), (int, float))]
        if scores:
            print(f"\n  == Group {g}: {score_field} 分布 ==")
            for threshold in [1, 2, 3, 4, 5]:
                count = sum(1 for s in scores if s >= threshold)
                print(f"    >= {threshold}: {count} ({count/len(scores)*100:.1f}%)")
            print(f"    mean={sum(scores)/len(scores):.2f}")

    # Rule override stats
    overrides = [r for r in assessments if "_original_decision" in r["_assessment"]]
    if overrides:
        print(f"\n  == 规则覆盖 ==")
        print(f"    覆盖总数: {len(overrides)}/{len(assessments)} ({len(overrides)/len(assessments)*100:.1f}%)")
        override_details: dict[str, int] = {}
        for r in overrides:
            a = r["_assessment"]
            key = f"[{r.get('_group','?')}] {a['_original_decision']} → {a['decision']}"
            override_details[key] = override_details.get(key, 0) + 1
        for key, c in sorted(override_details.items(), key=lambda x: -x[1]):
            print(f"    {key}: {c}")

    # Per-domain stats
    domain_decisions: dict[str, dict] = {}
    for r in assessments:
        domain = r.get("source", "unknown")
        decision = r["_assessment"].get("decision", "unknown")
        group = r.get("_group", "?")
        key = f"{domain} [{group}]"
        domain_decisions.setdefault(key, {})
        domain_decisions[key][decision] = domain_decisions[key].get(decision, 0) + 1
    if domain_decisions:
        print(f"\n  == 各 Domain Decision 分布 ==")
        for domain in sorted(domain_decisions.keys()):
            dd = domain_decisions[domain]
            total = sum(dd.values())
            parts = ", ".join(f"{k}={v}" for k, v in sorted(dd.items()))
            print(f"    {domain}: {parts} (total={total})")


# ── Main ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Stage A: 视频内容潜力评估 (ET-Instruct-164K, Source Routing)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="text_filter passed.jsonl")
    parser.add_argument("--output", required=True, help="stage_a_results.jsonl")
    parser.add_argument("--sample-n", type=int, default=200, help="抽样数量")
    parser.add_argument("--no-sample", action="store_true", help="全量评估")
    parser.add_argument("--api-base", default="https://api.novita.ai/v3/openai")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default="pa/gmn-2.5-pr")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true", help="断点续评")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("NOVITA_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""

    # Load data
    samples = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f"加载 {len(samples)} 条 passed 样本")
    print(f"API: {args.api_base}  model: {args.model}")

    # Show source group distribution
    group_counts: dict[str, int] = {}
    for s in samples:
        g = get_source_group(s.get("source", "unknown"))
        group_counts[g] = group_counts.get(g, 0) + 1
    print(f"Source Group 分布: {', '.join(f'{g}={c}' for g, c in sorted(group_counts.items()))}")

    # Resume
    existing: list[dict] = []
    if args.resume:
        assessed_ids, existing = load_checkpoint(args.output, id_field="video")
        samples = [s for s in samples if s.get("video", "") not in assessed_ids]
        print(f"已评估 {len(assessed_ids)} 条，剩余 {len(samples)} 条")

    # Sampling
    if not args.no_sample and len(samples) > args.sample_n:
        samples = stratified_sample(samples, args.sample_n)
        print(f"分层抽样 {len(samples)} 条进行评估")

    # Concurrent assessment
    def _assess(s):
        return assess_sample(s, args.api_base, api_key, args.model)

    new_results, failed = run_concurrent_assessment(
        samples, _assess, workers=args.workers,
        stream_output=args.output, strip_fields=["_events_parsed"],
    )

    results = existing + new_results
    print(f"\n== Stage A 评估完成 ==")
    print(f"  成功: {len(new_results) - failed}, 失败: {failed}, 总计: {len(results)}")

    # Stats
    print_stats(results)

    # Split by decision
    keep = [r for r in results if r.get("_assessment", {}).get("decision") == "keep"]
    reject = [r for r in results if r.get("_assessment", {}).get("decision") != "keep"]

    print(f"\n  == 筛选结果 ==")
    print(f"    keep:   {len(keep)}")
    print(f"    reject: {len(reject)}")

    # Write split files
    base = args.output.replace(".jsonl", "")
    if keep:
        write_results(keep, f"{base}_keep.jsonl", strip_fields=["_events_parsed"])
    if reject:
        write_results(reject, f"{base}_reject.jsonl", strip_fields=["_events_parsed"])

    print(f"\nStage A 完成。keep 样本可进入视觉校验阶段。")


if __name__ == "__main__":
    main()
