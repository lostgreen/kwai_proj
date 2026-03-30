"""
Stage A — Route D: VLM-Curated 物理过程审查 (TimeLens-100K)

目的：TimeLens 的事件标注由 Gemini-2.5-Pro 精准生成，时间戳和事件描述高度可靠。
     因此不审查标注质量，而审查：这些精准事件拼在一起，是否构成一个
     "物理层次丰富"的多步骤过程，以剔除"标注得很完美的无聊视频"
     （新闻联播、监控录像、两人对坐纯聊天等）。

路由分组: Group D (VLM-Curated) — 专为 VLM 标注数据源设计

预过滤规则:
  - events < 3 → 直接 reject（跳过 LLM）
    如果 Gemini 也只能标出 1-2 个事件，说明视频极其单调

输入: text_filter.py 产出的 passed_timelens.jsonl
输出:
  - stage_a_results.jsonl         — 全部评估结果
  - stage_a_results_keep.jsonl    — decision=keep
  - stage_a_results_reject.jsonl  — decision=reject

用法:
    python stage_a_coarse_filter.py \\
        --input results/passed_timelens.jsonl \\
        --output results/stage_a_results.jsonl \\
        --sample-n 1000

    # 全量评估（断点续评）
    python stage_a_coarse_filter.py \\
        --input results/passed_timelens.jsonl \\
        --output results/stage_a_results.jsonl \\
        --no-sample --resume --workers 16
"""

import json
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.llm_client import (
    call_llm,
    load_checkpoint,
    run_concurrent_assessment,
    stratified_sample,
    write_results,
)
from shared.decision_rules import apply_group_d_rules

# ── Group D Prompts — VLM-Curated 物理过程审查 ──────────

GROUP_D_SYSTEM = """\
You are evaluating a video based on highly accurate, VLM-generated event \
annotations. Since the timestamps and event descriptions are already accurate, \
your ONLY task is to judge if this video exhibits a "Rich, Multi-step Physical \
Process" suitable for 3-level hierarchical segmentation \
(Macro-phases -> Local tasks -> Atomic actions).

MUST REJECT IF:
1. News/Interviews/Vlogs: The events describe people just talking, sitting, \
or alternating camera angles (e.g., "A man speaks to the camera", \
"The scene changes to a street").
2. Passive Observation: E.g., "Cars driving on a highway", "A cat sleeping".
3. Monolithic/Flat: It's just one continuous action with no distinct phases.

MUST KEEP IF:
- The events clearly outline a progressive physical task, crafting, cooking, \
repairing, or dynamic sports with distinct phases and tool/object interactions.

## Examples

### Example 1 — progressive crafting task (keep)
Input:
Video duration: 195.0s | Events: 8
  1. [0.0s - 22.0s] Measure and mark the wood plank for cutting
  2. [22.0s - 48.0s] Cut the wood using a circular saw
  3. [48.0s - 70.0s] Sand the cut edges smooth
  4. [70.0s - 95.0s] Apply wood glue to join two pieces
  5. [95.0s - 115.0s] Clamp the pieces together and wait
  6. [115.0s - 140.0s] Drill pilot holes for screws
  7. [140.0s - 165.0s] Drive screws to secure the assembly
  8. [165.0s - 195.0s] Apply a coat of varnish finish

Output: {"process_analysis":"Clear 3-phase woodworking: preparation (measure+cut+sand), assembly (glue+clamp+drill+screw), finishing (varnish). Progressive task with distinct tools and state changes.","physical_hierarchy_score":5,"decision":"keep"}

### Example 2 — news/interview (reject)
Input:
Video duration: 180.0s | Events: 6
  1. [0.0s - 30.0s] The anchor introduces the day's top stories
  2. [30.0s - 60.0s] A reporter speaks from a street location
  3. [60.0s - 90.0s] The scene cuts back to the studio
  4. [90.0s - 120.0s] A guest is interviewed about the economy
  5. [120.0s - 150.0s] The anchor reads a weather update
  6. [150.0s - 180.0s] Closing remarks and credits

Output: {"process_analysis":"News broadcast with alternating camera angles and talking heads. No physical activity or progressive task — just people speaking in different settings.","physical_hierarchy_score":1,"decision":"reject"}

Return ONLY valid JSON."""

GROUP_D_USER = """\
Video duration: {duration:.1f}s
Events: {n_events}

{events_text}

Respond with ONLY valid JSON:
{{
  "process_analysis": "<1-2 sentences analyzing if the sequence of events forms \
a progressive physical task or just static/passive scenes>",
  "physical_hierarchy_score": <1-5, 5=excellent multi-step physical task, \
1=news broadcast or pure talking>,
  "decision": "keep | reject"
}}"""


# ── TimeLens-specific event parsing ──────────────────────

def parse_events(sample: dict) -> list[dict]:
    """Parse TimeLens events[i] -> [{start, end, description}, ...]."""
    raw_events = sample.get("events", [])
    parsed = []
    for ev in raw_events:
        desc = ev.get("query", "")
        spans = ev.get("span", [])
        if spans and len(spans[0]) == 2:
            start, end = spans[0][0], spans[0][1]
        else:
            start, end = 0.0, 0.0
        parsed.append({"start": float(start), "end": float(end), "description": desc})
    return parsed


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
    """Run Group D assessment on a single TimeLens sample."""
    events = parse_events(sample)
    assessed = dict(sample)
    assessed["_stage"] = "A"
    assessed["_group"] = "D"
    assessed["_n_events"] = len(events)

    # Pre-filter: events < 3 → direct reject (skip LLM)
    if len(events) < 3:
        assessed["_assessment"] = {
            "decision": "reject",
            "_prefilter": "events<3",
            "process_analysis": f"Only {len(events)} events — too few for hierarchical segmentation.",
            "physical_hierarchy_score": 0,
        }
        return assessed

    events_text = format_events_text(events)

    messages = [
        {"role": "system", "content": GROUP_D_SYSTEM},
        {"role": "user", "content": GROUP_D_USER.format(
            duration=sample.get("duration", 0),
            n_events=len(events),
            events_text=events_text,
        )},
    ]

    result = call_llm(messages, api_base, api_key, model)

    # Apply programmatic rules to override LLM decision
    llm_decision = result.get("decision", "unknown")
    rule_decision = apply_group_d_rules(result)
    if llm_decision != rule_decision:
        result["_original_decision"] = llm_decision
        result["decision"] = rule_decision

    assessed["_assessment"] = result
    return assessed


def print_stats(results: list[dict]):
    """Print Group D assessment statistics."""
    total = len(results)
    prefiltered = sum(1 for r in results if r.get("_assessment", {}).get("_prefilter"))
    assessments = [
        r["_assessment"] for r in results
        if "_assessment" in r
        and not r["_assessment"].get("_parse_error")
        and not r["_assessment"].get("_prefilter")
    ]

    print(f"\n  == 预过滤 ==")
    print(f"    events<3 直接 reject: {prefiltered}/{total} ({prefiltered/max(total,1)*100:.1f}%)")
    print(f"    进入 LLM 评估: {len(assessments)}")

    if not assessments:
        print("  无有效 LLM 评估结果")
        return

    # Decision distribution
    decisions = {}
    for a in assessments:
        d = a.get("decision", "unknown")
        decisions[d] = decisions.get(d, 0) + 1
    print(f"\n  == LLM Decision 分布 ==")
    for d, c in sorted(decisions.items(), key=lambda x: -x[1]):
        print(f"    {d}: {c} ({c/len(assessments)*100:.1f}%)")

    # physical_hierarchy_score distribution
    scores = [
        a.get("physical_hierarchy_score", 0)
        for a in assessments
        if isinstance(a.get("physical_hierarchy_score"), (int, float))
    ]
    if scores:
        print(f"\n  == physical_hierarchy_score 分布 ==")
        for threshold in [1, 2, 3, 4, 5]:
            count = sum(1 for s in scores if s >= threshold)
            print(f"    >= {threshold}: {count} ({count/len(scores)*100:.1f}%)")
        print(f"    mean={sum(scores)/len(scores):.2f}")

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
            total_d = sum(dd.values())
            parts = ", ".join(f"{k}={v}" for k, v in sorted(dd.items()))
            print(f"    {domain}: {parts} (total={total_d})")


# ── Main ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Stage A Route D: VLM-Curated 物理过程审查 (TimeLens-100K)",
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

    # Pre-filter stats preview
    prefilter_count = sum(1 for s in samples if len(s.get("events", [])) < 3)
    print(f"预过滤 (events<3): {prefilter_count} 条将直接 reject")

    # Resume (TimeLens uses video_path as unique ID)
    existing: list[dict] = []
    if args.resume:
        assessed_ids, existing = load_checkpoint(args.output, id_field="video_path")
        samples = [s for s in samples if s.get("video_path", "") not in assessed_ids]
        print(f"已评估 {len(assessed_ids)} 条，剩余 {len(samples)} 条")

    # Sampling
    if not args.no_sample and len(samples) > args.sample_n:
        samples = stratified_sample(samples, args.sample_n)
        print(f"分层抽样 {len(samples)} 条进行评估")

    # Concurrent assessment — stream to output file (crash-safe)
    def _assess(s):
        return assess_sample(s, args.api_base, api_key, args.model)

    new_results, failed = run_concurrent_assessment(
        samples, _assess, workers=args.workers, score_field="physical_hierarchy_score",
        stream_output=args.output,
    )

    results = existing + new_results
    print(f"\n== Stage A (Route D) 评估完成 ==")
    print(f"  成功: {len(new_results) - failed}, 失败: {failed}, 总计: {len(results)}")

    # Stats
    print_stats(results)

    # Split by decision (Group D: keep / reject only, no maybe)
    keep = [r for r in results if r.get("_assessment", {}).get("decision") == "keep"]
    reject = [r for r in results if r.get("_assessment", {}).get("decision") != "keep"]

    print(f"\n  == 筛选结果 ==")
    print(f"    keep:   {len(keep)}")
    print(f"    reject: {len(reject)}")

    # Write split files
    base = args.output.replace(".jsonl", "")
    if keep:
        write_results(keep, f"{base}_keep.jsonl")
    if reject:
        write_results(reject, f"{base}_reject.jsonl")

    print(f"\nStage A (Route D) 完成。keep 样本进入 Vision Filter → Stage B。")


if __name__ == "__main__":
    main()
