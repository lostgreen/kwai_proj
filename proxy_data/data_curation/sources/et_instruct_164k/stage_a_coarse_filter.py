"""
Stage A — 视频内容潜力评估 (ET-Instruct-164K)

目的：基于文本标注判断视频是否包含丰富的、适合层次分割的内容。
     不再判断标注本身的粒度，而是评估视频的动作密度、状态变化和时序连贯性。

核心原则：
  - 标注粒度不重要（L3 细粒度标注反而是好信号）
  - 关注视频内容本身的丰富度和层次潜力
  - 只 reject 真正单调/无结构的视频

输入: text_filter.py 产出的 passed.jsonl
输出:
  - stage_a_results.jsonl     — 全部评估结果
  - stage_a_keep.jsonl        — decision=keep 的样本
  - stage_a_maybe.jsonl       — decision=maybe 的边界样本
  - stage_a_reject.jsonl      — decision=reject 的淘汰样本

用法:
    python stage_a_coarse_filter.py \\
        --input results/passed.jsonl \\
        --output results/stage_a_results.jsonl \\
        --sample-n 1000

    # 全量评估（断点续评）
    python stage_a_coarse_filter.py \\
        --input results/passed.jsonl \\
        --output results/stage_a_results.jsonl \\
        --no-sample --resume --workers 16
"""

import json
import argparse
import os
import re
import sys

# Ensure shared module is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.llm_client import (
    call_llm,
    load_checkpoint,
    run_concurrent_assessment,
    stratified_sample,
    write_results,
)
from shared.decision_rules import apply_richness_rules

# ── Stage A Prompts — 视频内容潜力评估 ─────────────────

STAGE_A_SYSTEM = """\
You are evaluating whether a video, described by its temporal text annotations, \
has the potential to support a 3-level hierarchical segmentation \
(macro-phases -> tasks -> atomic actions).

Your goal is to assess the RICHNESS and COMPLEXITY of the underlying video, \
NOT to judge if the current annotations are perfectly formatted. The text is \
merely a proxy to understand what happens in the video.

A highly suitable video for hierarchical analysis should contain:
1. Multi-step processes (e.g., cooking, assembling, repairing) rather than \
a single monotonous activity (e.g., just running or talking).
2. Meaningful state changes or tool usages.
3. A chronological flow of distinct events.

Do NOT reject a sample just because its text annotations are too fine-grained \
(atomic actions) or mixed in granularity. Fine-grained annotations are \
EXCELLENT because they can be easily grouped into higher-level phases later.

ONLY REJECT IF:
- The events are highly repetitive (e.g., "jumping", "jumping again", "jumping").
- The events are extremely sparse or lack any sub-step structure (e.g., a \
single broad event "playing football" covering the whole video).
- The video appears to be a static scene or a non-procedural activity \
without distinct phases.

## Examples

### Example 1 — rich multi-step process (keep)

Input:
Video duration: 180.0s | Domain: coin | Segments: 7
  1. [0.0s - 23.5s] Spread glue evenly on the wood surface
  2. [23.5s - 48.0s] Attach decorative veneer to the glued surface
  3. [48.0s - 72.0s] Trim excess veneer with a utility knife
  4. [72.0s - 98.0s] Sand the edges smooth with fine sandpaper
  5. [98.0s - 125.0s] Apply first coat of lacquer finish
  6. [125.0s - 155.0s] Let dry and sand lightly between coats
  7. [155.0s - 180.0s] Apply final protective coat

Output:
{"richness_analysis":"7 distinct woodworking steps with clear tool usage and state changes (gluing, cutting, sanding, finishing). Strong multi-phase progression.","action_density_score":5,"state_change_score":5,"temporal_flow_score":5,"video_hierarchy_potential":"high","decision":"keep"}

### Example 2 — monotonous talking head (reject)

Input:
Video duration: 150.0s | Domain: queryd | Segments: 5
  1. [10.0s - 35.0s] A person talks about their morning routine
  2. [40.0s - 65.0s] The same person describes their favorite food
  3. [70.0s - 95.0s] A different topic about travel plans
  4. [100.0s - 125.0s] Discussion of weekend hobbies
  5. [130.0s - 150.0s] Summary and closing remarks

Output:
{"richness_analysis":"Talking-head video with no physical actions or state changes. Topics are independent with no procedural flow.","action_density_score":1,"state_change_score":1,"temporal_flow_score":2,"video_hierarchy_potential":"low","decision":"reject"}

Return ONLY valid JSON."""

STAGE_A_USER = """\
Video duration: {duration:.1f}s
Domain: {source}
Number of annotated segments: {n_events}

Annotated segments:
{events_text}

Evaluate whether this video has rich enough content to support hierarchical \
segmentation, based on these text annotations.

Respond with ONLY valid JSON:
{{
  "richness_analysis": "<1-2 sentences analyzing action diversity and hierarchy potential>",
  "action_density_score": <1-5>,
  "state_change_score": <1-5>,
  "temporal_flow_score": <1-5>,
  "video_hierarchy_potential": "high | medium | low",
  "decision": "keep | maybe | reject"
}}"""


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

    # Try extracting "36.0 - 44.0 seconds, description." from GPT text
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
    """Run Stage A assessment on a single ET-Instruct sample."""
    events = parse_events(sample)
    events_text = format_events_text(events)

    messages = [
        {"role": "system", "content": STAGE_A_SYSTEM},
        {"role": "user", "content": STAGE_A_USER.format(
            duration=sample.get("duration", 0),
            source=sample.get("source", "unknown"),
            n_events=len(events),
            events_text=events_text,
        )},
    ]

    result = call_llm(messages, api_base, api_key, model)

    # Apply programmatic rules to override LLM decision
    llm_decision = result.get("decision", "unknown")
    rule_decision = apply_richness_rules(result)
    if llm_decision != rule_decision:
        result["_original_decision"] = llm_decision
        result["decision"] = rule_decision

    assessed = dict(sample)
    assessed["_assessment"] = result
    assessed["_stage"] = "A"
    assessed["_n_events"] = len(events)
    return assessed


def print_stats(results: list[dict]):
    """Print Stage A assessment statistics."""
    assessments = [r["_assessment"] for r in results if "_assessment" in r and not r["_assessment"].get("_parse_error")]

    if not assessments:
        print("  无有效评估结果")
        return

    # Decision distribution
    decisions = {}
    for a in assessments:
        d = a.get("decision", "unknown")
        decisions[d] = decisions.get(d, 0) + 1
    print(f"\n  == Decision 分布 ==")
    for d, c in sorted(decisions.items(), key=lambda x: -x[1]):
        print(f"    {d}: {c} ({c/len(assessments)*100:.1f}%)")

    # Score distributions for each dimension
    for field in ["action_density_score", "state_change_score", "temporal_flow_score"]:
        scores = [a.get(field, 0) for a in assessments if isinstance(a.get(field), (int, float))]
        if scores:
            print(f"\n  == {field} 分布 ==")
            for threshold in [1, 2, 3, 4, 5]:
                count = sum(1 for s in scores if s >= threshold)
                print(f"    >= {threshold}: {count} ({count/len(scores)*100:.1f}%)")
            print(f"    mean={sum(scores)/len(scores):.2f}")

    # Hierarchy potential distribution
    potentials = {}
    for a in assessments:
        p = a.get("video_hierarchy_potential", "unknown")
        potentials[p] = potentials.get(p, 0) + 1
    print(f"\n  == Video Hierarchy Potential 分布 ==")
    for p, c in sorted(potentials.items(), key=lambda x: -x[1]):
        print(f"    {p}: {c} ({c/len(assessments)*100:.1f}%)")

    # Rule override stats
    overrides = sum(1 for a in assessments if "_original_decision" in a)
    if overrides:
        print(f"\n  == 规则覆盖 ==")
        print(f"    覆盖总数: {overrides}/{len(assessments)} ({overrides/len(assessments)*100:.1f}%)")
        override_details: dict[str, int] = {}
        for a in assessments:
            if "_original_decision" in a:
                key = f"{a['_original_decision']} → {a['decision']}"
                override_details[key] = override_details.get(key, 0) + 1
        for key, c in sorted(override_details.items(), key=lambda x: -x[1]):
            print(f"    {key}: {c}")

    # Per-domain stats
    domain_decisions: dict[str, dict] = {}
    for r in results:
        if "_assessment" not in r or r["_assessment"].get("_parse_error"):
            continue
        domain = r.get("source", "unknown")
        decision = r["_assessment"].get("decision", "unknown")
        domain_decisions.setdefault(domain, {})
        domain_decisions[domain][decision] = domain_decisions[domain].get(decision, 0) + 1
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
        description="Stage A: L2 粒度粗筛 (ET-Instruct-164K)",
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

    # Concurrent assessment — stream to output file (crash-safe)
    def _assess(s):
        return assess_sample(s, args.api_base, api_key, args.model)

    new_results, failed = run_concurrent_assessment(
        samples, _assess, workers=args.workers, score_field="l2_fit_score",
        stream_output=args.output, strip_fields=["_events_parsed"],
    )

    results = existing + new_results
    print(f"\n== Stage A 评估完成 ==")
    print(f"  成功: {len(new_results) - failed}, 失败: {failed}, 总计: {len(results)}")

    # Stats
    print_stats(results)

    # Split by decision
    keep = [r for r in results if r.get("_assessment", {}).get("decision") == "keep"]
    maybe = [r for r in results if r.get("_assessment", {}).get("decision") == "maybe"]
    reject = [r for r in results if r.get("_assessment", {}).get("decision") not in ("keep", "maybe")]

    print(f"\n  == 筛选结果 ==")
    print(f"    keep:   {len(keep)}")
    print(f"    maybe:  {len(maybe)}")
    print(f"    reject: {len(reject)}")

    # Write split files (full rewrite — these are derived from the main output)
    base = args.output.replace(".jsonl", "")
    if keep:
        write_results(keep, f"{base}_keep.jsonl", strip_fields=["_events_parsed"])
    if maybe:
        write_results(maybe, f"{base}_maybe.jsonl", strip_fields=["_events_parsed"])
    if reject:
        write_results(reject, f"{base}_reject.jsonl", strip_fields=["_events_parsed"])

    print(f"\nStage A 完成。keep 样本可直接进入 Stage B 精筛。")


if __name__ == "__main__":
    main()
