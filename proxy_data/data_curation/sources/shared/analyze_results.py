"""
Stage A/B 评估结果可视化分析

对 stage_a_results.jsonl 或 stage_b_results.jsonl 的评估结果做统计分析和可视化。
支持终端文本报告 + 可选的 HTML 可视化报告。

用法:
    # 终端报告
    python analyze_results.py --input results/stage_a_results.jsonl --stage A

    # 生成 HTML 报告
    python analyze_results.py --input results/stage_a_results.jsonl --stage A --html results/stage_a_report.html

    # 抽样审查（每类抽 3 条看 reasoning）
    python analyze_results.py --input results/stage_a_results.jsonl --stage A --review 3

    # Stage B 分析
    python analyze_results.py --input results/stage_b_results.jsonl --stage B --html results/stage_b_report.html
"""

import json
import argparse
import os
import random
from collections import defaultdict


# ── Data Loading ─────────────────────────────────────────

def load_results(path: str) -> list[dict]:
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


# ── Stage A Analysis ─────────────────────────────────────

def analyze_stage_a(results: list[dict]) -> dict:
    """Analyze Stage A results and return stats dict."""
    assessments = []
    for r in results:
        a = r.get("_assessment", {})
        if not a.get("_parse_error") and not a.get("error"):
            assessments.append({**a, "_source": r.get("source", "unknown")})

    stats = {"total": len(results), "valid": len(assessments), "parse_errors": len(results) - len(assessments)}

    if not assessments:
        return stats

    # Decision distribution
    stats["decision"] = _count_field(assessments, "decision")

    # Granularity label
    stats["granularity_label"] = _count_field(assessments, "granularity_label")

    # Granularity issue
    stats["granularity_issue"] = _count_field(assessments, "granularity_issue")

    # Mixed ratio estimate
    stats["mixed_ratio_estimate"] = _count_field(assessments, "mixed_ratio_estimate")

    # L2 fit score
    scores = [a["l2_fit_score"] for a in assessments if isinstance(a.get("l2_fit_score"), (int, float))]
    stats["l2_fit_score"] = _score_stats(scores)

    # Domain x Decision cross-tab
    stats["domain_decision"] = _cross_tab(assessments, "_source", "decision")

    # Domain x Granularity cross-tab
    stats["domain_granularity"] = _cross_tab(assessments, "_source", "granularity_label")

    # Domain x Score
    domain_scores: dict[str, list] = defaultdict(list)
    for a in assessments:
        if isinstance(a.get("l2_fit_score"), (int, float)):
            domain_scores[a["_source"]].append(a["l2_fit_score"])
    stats["domain_score"] = {
        d: {"mean": sum(s) / len(s), "count": len(s)}
        for d, s in sorted(domain_scores.items())
    }

    return stats


# ── Stage B Analysis ─────────────────────────────────────

def analyze_stage_b(results: list[dict]) -> dict:
    """Analyze Stage B results and return stats dict."""
    assessments = []
    for r in results:
        a = r.get("_assessment", {})
        if not a.get("_parse_error") and not a.get("error"):
            assessments.append({**a, "_source": r.get("source", "unknown")})

    stats = {"total": len(results), "valid": len(assessments), "parse_errors": len(results) - len(assessments)}

    if not assessments:
        return stats

    # Decision
    stats["decision"] = _count_field(assessments, "decision")

    # Score distributions
    for field in ["l1_potential", "l3_potential", "temporal_structure", "overall_score"]:
        scores = [a[field] for a in assessments if isinstance(a.get(field), (int, float))]
        stats[field] = _score_stats(scores)

    # Domain x Decision
    stats["domain_decision"] = _cross_tab(assessments, "_source", "decision")

    # Domain x Overall Score
    domain_scores: dict[str, list] = defaultdict(list)
    for a in assessments:
        if isinstance(a.get("overall_score"), (int, float)):
            domain_scores[a["_source"]].append(a["overall_score"])
    stats["domain_score"] = {
        d: {"mean": sum(s) / len(s), "count": len(s)}
        for d, s in sorted(domain_scores.items())
    }

    return stats


# ── Helpers ──────────────────────────────────────────────

def _count_field(items: list[dict], field: str) -> dict:
    counts: dict[str, int] = {}
    for item in items:
        v = item.get(field, "unknown")
        counts[v] = counts.get(v, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def _score_stats(scores: list) -> dict:
    if not scores:
        return {}
    return {
        "mean": sum(scores) / len(scores),
        "distribution": {str(i): sum(1 for s in scores if s == i) for i in range(1, 6)},
        "cumulative": {f">={i}": sum(1 for s in scores if s >= i) for i in range(1, 6)},
        "count": len(scores),
    }


def _cross_tab(items: list[dict], row_field: str, col_field: str) -> dict:
    tab: dict[str, dict] = defaultdict(lambda: defaultdict(int))
    for item in items:
        row = item.get(row_field, "unknown")
        col = item.get(col_field, "unknown")
        tab[row][col] += 1
    return {r: dict(cols) for r, cols in sorted(tab.items())}


# ── Print Report ─────────────────────────────────────────

def print_report(stats: dict, stage: str):
    print(f"\n{'=' * 60}")
    print(f" Stage {stage} 评估结果分析")
    print(f"{'=' * 60}")
    print(f"  总样本: {stats['total']}, 有效: {stats['valid']}, 解析失败: {stats['parse_errors']}")

    # Decision
    if "decision" in stats:
        print(f"\n  -- Decision 分布 --")
        total = stats["valid"]
        for k, v in stats["decision"].items():
            print(f"    {k:12s}: {v:5d} ({v/total*100:.1f}%)")

    # Stage A specific
    if stage == "A":
        for field in ["granularity_label", "granularity_issue", "mixed_ratio_estimate"]:
            if field in stats:
                print(f"\n  -- {field} 分布 --")
                total = stats["valid"]
                for k, v in stats[field].items():
                    print(f"    {k:20s}: {v:5d} ({v/total*100:.1f}%)")

        if "l2_fit_score" in stats and stats["l2_fit_score"]:
            s = stats["l2_fit_score"]
            print(f"\n  -- L2 Fit Score --")
            print(f"    mean: {s['mean']:.2f}")
            for k, v in s["distribution"].items():
                print(f"    score={k}: {v}")

    # Stage B specific
    if stage == "B":
        for field in ["l1_potential", "l3_potential", "temporal_structure", "overall_score"]:
            if field in stats and stats[field]:
                s = stats[field]
                print(f"\n  -- {field} --")
                print(f"    mean: {s['mean']:.2f}")
                for k, v in s["distribution"].items():
                    print(f"    score={k}: {v}")

    # Domain x Decision
    if "domain_decision" in stats:
        print(f"\n  -- Domain x Decision --")
        all_decisions = set()
        for dd in stats["domain_decision"].values():
            all_decisions.update(dd.keys())
        decisions_sorted = sorted(all_decisions)

        header = f"    {'Domain':25s}" + "".join(f"{d:>8s}" for d in decisions_sorted) + f"{'total':>8s}"
        print(header)
        for domain, dd in stats["domain_decision"].items():
            row = f"    {domain:25s}"
            total_d = 0
            for d in decisions_sorted:
                v = dd.get(d, 0)
                total_d += v
                row += f"{v:8d}"
            row += f"{total_d:8d}"
            print(row)

    # Domain score
    if "domain_score" in stats:
        print(f"\n  -- Domain 平均分 --")
        for domain, s in stats["domain_score"].items():
            print(f"    {domain:25s}: mean={s['mean']:.2f} (n={s['count']})")


# ── Review Samples ───────────────────────────────────────

def review_samples(results: list[dict], n_per_category: int, stage: str):
    """Print sample reasoning for each decision category."""
    by_decision: dict[str, list] = defaultdict(list)
    for r in results:
        a = r.get("_assessment", {})
        if not a.get("_parse_error"):
            d = a.get("decision", "unknown")
            by_decision[d].append(r)

    for decision in ["keep", "maybe", "reject"]:
        items = by_decision.get(decision, [])
        if not items:
            continue
        n = min(n_per_category, len(items))
        chosen = random.sample(items, n)
        print(f"\n{'=' * 60}")
        print(f" [{decision.upper()}] 抽样 {n} 条 (共 {len(items)} 条)")
        print(f"{'=' * 60}")
        for i, r in enumerate(chosen, 1):
            a = r.get("_assessment", {})
            source = r.get("source", "?")
            duration = r.get("duration", 0)
            n_events = r.get("_n_events", "?")
            print(f"\n  --- [{decision}] #{i} ---")
            print(f"  domain={source}, duration={duration:.1f}s, events={n_events}")
            if stage == "A":
                print(f"  label={a.get('granularity_label')}, score={a.get('l2_fit_score')}, "
                      f"issue={a.get('granularity_issue')}, mixed={a.get('mixed_ratio_estimate')}")
            elif stage == "B":
                print(f"  l1={a.get('l1_potential')}, l3={a.get('l3_potential')}, "
                      f"temporal={a.get('temporal_structure')}, overall={a.get('overall_score')}")
                sketch = a.get("phase_sketch", [])
                if sketch:
                    print(f"  phases: {sketch}")
            print(f"  reasoning: {a.get('reasoning', '?')}")


# ── HTML Report ──────────────────────────────────────────

def generate_html(stats: dict, stage: str, output_path: str):
    """Generate a simple standalone HTML report."""
    html = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'>",
        f"<title>Stage {stage} Report</title>",
        "<style>",
        "body{font-family:system-ui;max-width:900px;margin:0 auto;padding:20px;background:#f9f9f9}",
        "h1,h2{color:#333}",
        "table{border-collapse:collapse;width:100%;margin:10px 0}",
        "th,td{border:1px solid #ddd;padding:6px 10px;text-align:right}",
        "th{background:#f0f0f0}",
        "td:first-child,th:first-child{text-align:left}",
        ".bar{display:inline-block;height:16px;background:#4a90d9;border-radius:2px;vertical-align:middle}",
        ".keep{color:#2e7d32}.maybe{color:#f57f17}.reject{color:#c62828}",
        "</style></head><body>",
        f"<h1>Stage {stage} 评估报告</h1>",
        f"<p>总样本: {stats['total']} | 有效: {stats['valid']} | 解析失败: {stats['parse_errors']}</p>",
    ]

    # Decision distribution
    if "decision" in stats:
        html.append("<h2>Decision 分布</h2><table><tr><th>Decision</th><th>Count</th><th>%</th><th></th></tr>")
        total = stats["valid"]
        for k, v in stats["decision"].items():
            pct = v / total * 100
            bar_w = int(pct * 3)
            css = k if k in ("keep", "maybe", "reject") else ""
            html.append(f"<tr><td class='{css}'><b>{k}</b></td><td>{v}</td>"
                        f"<td>{pct:.1f}%</td><td><span class='bar' style='width:{bar_w}px'></span></td></tr>")
        html.append("</table>")

    # Domain x Decision
    if "domain_decision" in stats:
        all_decisions = set()
        for dd in stats["domain_decision"].values():
            all_decisions.update(dd.keys())
        decisions_sorted = sorted(all_decisions)

        html.append("<h2>Domain x Decision</h2><table><tr><th>Domain</th>")
        for d in decisions_sorted:
            html.append(f"<th class='{d}'>{d}</th>")
        html.append("<th>total</th></tr>")

        for domain, dd in stats["domain_decision"].items():
            html.append(f"<tr><td>{domain}</td>")
            row_total = 0
            for d in decisions_sorted:
                v = dd.get(d, 0)
                row_total += v
                html.append(f"<td>{v}</td>")
            html.append(f"<td><b>{row_total}</b></td></tr>")
        html.append("</table>")

    # Domain mean scores
    if "domain_score" in stats:
        html.append("<h2>Domain 平均分</h2><table><tr><th>Domain</th><th>Mean Score</th><th>Count</th><th></th></tr>")
        for domain, s in stats["domain_score"].items():
            bar_w = int(s["mean"] * 40)
            html.append(f"<tr><td>{domain}</td><td>{s['mean']:.2f}</td>"
                        f"<td>{s['count']}</td><td><span class='bar' style='width:{bar_w}px'></span></td></tr>")
        html.append("</table>")

    # Score distributions
    score_fields = (
        ["l2_fit_score"] if stage == "A"
        else ["l1_potential", "l3_potential", "temporal_structure", "overall_score"]
    )
    for field in score_fields:
        if field in stats and stats[field]:
            s = stats[field]
            html.append(f"<h2>{field} (mean={s['mean']:.2f})</h2>")
            html.append("<table><tr><th>Score</th><th>Count</th><th></th></tr>")
            for score_val, count in s["distribution"].items():
                bar_w = int(count * 3) if stats["valid"] > 0 else 0
                html.append(f"<tr><td>{score_val}</td><td>{count}</td>"
                            f"<td><span class='bar' style='width:{bar_w}px'></span></td></tr>")
            html.append("</table>")

    html.append("</body></html>")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    print(f"\nHTML 报告 -> {output_path}")


# ── Main ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage A/B 评估结果分析")
    parser.add_argument("--input", required=True, help="stage_a_results.jsonl 或 stage_b_results.jsonl")
    parser.add_argument("--stage", required=True, choices=["A", "B"], help="分析的阶段")
    parser.add_argument("--html", default=None, help="生成 HTML 报告到此路径")
    parser.add_argument("--review", type=int, default=0, help="每类抽样审查数量")
    parser.add_argument("--json", default=None, help="输出统计 JSON 到此路径")
    args = parser.parse_args()

    results = load_results(args.input)
    print(f"加载 {len(results)} 条评估结果")

    if args.stage == "A":
        stats = analyze_stage_a(results)
    else:
        stats = analyze_stage_b(results)

    print_report(stats, args.stage)

    if args.review > 0:
        review_samples(results, args.review, args.stage)

    if args.html:
        generate_html(stats, args.stage, args.html)

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"\n统计 JSON -> {args.json}")


if __name__ == "__main__":
    main()
