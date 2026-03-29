"""
Stage B — 层次潜力精筛 (数据源无关)

目的：对 Stage A 粗筛通过（decision=keep）的样本做深度评估：
  - L1 聚合潜力：这些 L2 事件能否分组为宏观阶段？
  - L3 分解潜力：事件描述是否暗示可分解的原子动作？
  - 时序结构：事件间是否有清晰的先后关系？
  - 综合判断：是否值得进入正式数据池？

Stage B 只处理 Stage A 产出的 _keep.jsonl，样本里已有 _assessment (Stage A 结果)。

输入: stage_a_results_keep.jsonl
输出:
  - stage_b_results.jsonl     — 全部评估结果
  - stage_b_keep.jsonl        — 最终保留样本（进入标注流水线）
  - stage_b_maybe.jsonl       — 边界样本（人工复核）
  - stage_b_reject.jsonl      — 精筛淘汰

用法:
    python stage_b_fine_filter.py \\
        --input results/stage_a_results_keep.jsonl \\
        --output results/stage_b_results.jsonl \\
        --data-source et_instruct \\
        --api-base https://api.novita.ai/v3/openai \\
        --model pa/gmn-2.5-pr

    # 全量评估
    python stage_b_fine_filter.py \\
        --input results/stage_a_results_keep.jsonl \\
        --output results/stage_b_results.jsonl \\
        --data-source timelens \\
        --no-sample --resume --workers 16
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

# ── Stage B Prompts ──────────────────────────────────────

STAGE_B_SYSTEM = """\
You are evaluating whether a set of temporal annotations that are already \
roughly L2-like can support a full 3-level hierarchical segmentation setup.

Hierarchy:
- L1: macro phases grouping multiple events
- L2: provided event-like segments (candidate L2 annotations)
- L3: atomic actions inside each event

Assume the provided segments are candidate L2 annotations, but still judge \
strictly. Focus on whether they support meaningful L1 grouping AND L3 \
decomposition.

Evaluation criteria:
- L1 Potential: Can these events be naturally grouped into 2-4 macro phases \
with distinct themes? (5=clear phases, 1=events are all independent)
- L3 Potential: Do event descriptions suggest decomposable sub-actions? \
(5=rich detail implying multiple steps, 1=events are already near-atomic)
- Temporal Structure: Are events well-ordered with clear temporal flow? \
(5=strong narrative arc, 1=random/overlapping/unclear progression)
- Overall: Combined suitability for a production-quality 3-level annotation

Be strict: only give overall_score >= 4 if ALL three dimensions score >= 3.

Return only valid JSON."""

STAGE_B_USER = """\
Video duration: {duration:.1f}s
Domain: {source}
Number of candidate L2 segments: {n_events}

Candidate L2 segments:
{events_text}

Evaluate this sample for hierarchical segmentation suitability.

Respond with ONLY valid JSON:
{{
  "l1_potential": <1-5>,
  "l3_potential": <1-5>,
  "temporal_structure": <1-5>,
  "overall_score": <1-5>,
  "phase_sketch": ["phase_name: event_indices", ...],
  "decision": "keep | maybe | reject",
  "reasoning": "<1-2 short sentences>"
}}"""


# ── Data-source-specific event parsing ───────────────────

def parse_events_et_instruct(sample: dict) -> list[dict]:
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


def parse_events_timelens(sample: dict) -> list[dict]:
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


PARSERS = {
    "et_instruct": parse_events_et_instruct,
    "timelens": parse_events_timelens,
}


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
    parse_fn,
) -> dict:
    """Run Stage B assessment on a single sample."""
    events = parse_fn(sample)
    events_text = format_events_text(events)

    messages = [
        {"role": "system", "content": STAGE_B_SYSTEM},
        {"role": "user", "content": STAGE_B_USER.format(
            duration=sample.get("duration", 0),
            source=sample.get("source", "unknown"),
            n_events=len(events),
            events_text=events_text,
        )},
    ]

    result = call_llm(messages, api_base, api_key, model)

    assessed = dict(sample)
    # Preserve Stage A assessment, add Stage B
    stage_a = sample.get("_assessment", {})
    assessed["_assessment_a"] = stage_a
    assessed["_assessment"] = result
    assessed["_stage"] = "B"
    assessed["_n_events"] = len(events)
    return assessed


def print_stats(results: list[dict]):
    """Print Stage B assessment statistics."""
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

    # Score distributions
    for field in ["l1_potential", "l3_potential", "temporal_structure", "overall_score"]:
        scores = [a.get(field, 0) for a in assessments if isinstance(a.get(field), (int, float))]
        if scores:
            print(f"\n  == {field} 分布 ==")
            for threshold in [1, 2, 3, 4, 5]:
                count = sum(1 for s in scores if s >= threshold)
                print(f"    >= {threshold}: {count} ({count/len(scores)*100:.1f}%)")
            print(f"    mean={sum(scores)/len(scores):.2f}")

    # Per-domain
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
        description="Stage B: 层次潜力精筛",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="stage_a_keep.jsonl")
    parser.add_argument("--output", required=True, help="stage_b_results.jsonl")
    parser.add_argument("--data-source", required=True, choices=list(PARSERS.keys()),
                        help="数据源类型（决定事件解析方式）")
    parser.add_argument("--sample-n", type=int, default=0, help="抽样数量（0=全量）")
    parser.add_argument("--no-sample", action="store_true", help="全量评估")
    parser.add_argument("--api-base", default="https://api.novita.ai/v3/openai")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default="pa/gmn-2.5-pr")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true", help="断点续评")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("NOVITA_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    parse_fn = PARSERS[args.data_source]

    # Determine unique ID field based on data source
    id_field = "video" if args.data_source == "et_instruct" else "video_path"

    # Load data
    samples = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f"加载 {len(samples)} 条 Stage A keep 样本")
    print(f"API: {args.api_base}  model: {args.model}")
    print(f"数据源: {args.data_source}, ID field: {id_field}")

    # Resume
    existing: list[dict] = []
    if args.resume:
        assessed_ids, existing = load_checkpoint(args.output, id_field=id_field)
        samples = [s for s in samples if s.get(id_field, "") not in assessed_ids]
        print(f"已评估 {len(assessed_ids)} 条，剩余 {len(samples)} 条")

    # Sampling
    if args.sample_n > 0 and not args.no_sample and len(samples) > args.sample_n:
        samples = stratified_sample(samples, args.sample_n)
        print(f"分层抽样 {len(samples)} 条进行评估")

    # Concurrent assessment
    def _assess(s):
        return assess_sample(s, args.api_base, api_key, args.model, parse_fn)

    new_results, failed = run_concurrent_assessment(
        samples, _assess, workers=args.workers, score_field="overall_score",
    )

    results = existing + new_results
    print(f"\n== Stage B 评估完成 ==")
    print(f"  成功: {len(new_results) - failed}, 失败: {failed}, 总计: {len(results)}")

    # Stats
    print_stats(results)

    # Split by decision
    keep = [r for r in results if r.get("_assessment", {}).get("decision") == "keep"]
    maybe = [r for r in results if r.get("_assessment", {}).get("decision") == "maybe"]
    reject = [r for r in results if r.get("_assessment", {}).get("decision") not in ("keep", "maybe")]

    print(f"\n  == 最终筛选结果 ==")
    print(f"    keep:   {len(keep)}")
    print(f"    maybe:  {len(maybe)}")
    print(f"    reject: {len(reject)}")

    # Write outputs
    write_results(results, args.output)

    base = args.output.replace(".jsonl", "")
    if keep:
        write_results(keep, f"{base}_keep.jsonl")
    if maybe:
        write_results(maybe, f"{base}_maybe.jsonl")
    if reject:
        write_results(reject, f"{base}_reject.jsonl")

    print(f"\nStage B 完成。keep 样本可进入层次分割标注流水线。")


if __name__ == "__main__":
    main()
