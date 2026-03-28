"""
LLM 层次标注潜力评估脚本

目标：对 text_filter.py 的 passed 样本，用 LLM (API) 分析其文本标注，
     评估是否适合三层层次分割标注 (L1/L2/L3)。

层次映射关系：
    ET-Instruct 的 tgt 事件  ≈  我们标注体系的 L2 (events)
    L1 (macro_phases)        =  L2 事件能否聚合为 2-3 个宏观阶段
    L3 (atomic_actions)      =  L2 事件内部能否分解为 2-4 个原子动作

评估指标（LLM 输出）：
    - l1_potential (1-5): L2 事件能否被自然分组为不同阶段
    - l3_potential (1-5): 事件描述是否暗示可分解的子步骤
    - temporal_structure (1-5): 事件间时序关系的清晰度
    - overall_score (1-5): 综合适合度
    - phase_sketch: L1 阶段草图（LLM 建议的分组方式）
    - reasoning: 简要说明

用法:
    # 先抽样评估（默认 200 条）
    python assess_hierarchy.py \
        --input passed_5events.jsonl \
        --output assessed.jsonl \
        --sample_n 200 \
        --api-base https://api.novita.ai/v3/openai \
        --model pa/gmn-2.5-pr

    # 全量评估（断点续评）
    python assess_hierarchy.py \
        --input passed_5events.jsonl \
        --output assessed.jsonl \
        --no-sample --resume \
        --api-base https://api.novita.ai/v3/openai \
        --model pa/gmn-2.5-pr \
        --workers 16
"""

import json
import argparse
import os
import re
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ── Prompt 模板 ─────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a video annotation specialist. Your task is to evaluate whether a video's \
text description is suitable for hierarchical temporal segmentation annotation.

Our annotation framework has 3 levels:
- L1 (Macro Phases): 2-4 high-level stages that group related events (e.g., "Preparation → Cooking → Plating")
- L2 (Events): Goal-directed activities with clear start/end times (already provided in the data)
- L3 (Atomic Actions): Fine-grained sub-actions within each event (e.g., "pick up knife → cut onion → push aside")

You will receive:
- Video duration
- A list of events (L2) with timestamps and descriptions

Evaluate and respond in the exact JSON format specified."""

USER_PROMPT_TEMPLATE = """\
Video duration: {duration:.1f}s
Domain: {source}
Number of events: {n_events}

Events (L2):
{events_text}

Evaluate this video for hierarchical annotation suitability. Respond with ONLY valid JSON:
{{
  "l1_potential": <1-5>,       // Can events be grouped into 2-4 macro phases? 5=clear phases, 1=all events are independent
  "l3_potential": <1-5>,       // Do event descriptions suggest decomposable sub-actions? 5=rich detail, 1=too atomic already
  "temporal_structure": <1-5>, // Are events well-ordered with clear temporal flow? 5=strong narrative, 1=random/overlapping
  "overall_score": <1-5>,      // Overall suitability for 3-level annotation
  "phase_sketch": ["phase1_name: event_indices", ...],  // Suggested L1 grouping, e.g. ["Prep: 1-3", "Cook: 4-6", "Serve: 7-8"]
  "reasoning": "<1-2 sentences explaining the score>"
}}"""


def parse_events(sample: dict) -> list[dict]:
    """从样本中解析事件列表 [{start, end, description}, ...]。"""
    tgt = sample.get("tgt", [])
    gpt_text = ""
    for turn in sample.get("conversations", []):
        if turn.get("from") == "gpt":
            gpt_text = turn.get("value", "")
            break

    # 解析 "36.0 - 44.0 seconds, clean the bananas." 格式
    events = []
    # tgt 是成对的 [s1, e1, s2, e2, ...]
    n_events = len(tgt) // 2

    # 尝试从文本中提取描述
    # 格式: "36.0 - 44.0 seconds, description."
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
        # Fallback: 用 tgt 时间戳 + 整段文本
        for i in range(n_events):
            events.append({
                "start": float(tgt[i * 2]),
                "end": float(tgt[i * 2 + 1]),
                "description": f"(event {i+1})",
            })

    return events


def format_events_text(events: list[dict]) -> str:
    """格式化事件列表为可读文本。"""
    lines = []
    for i, ev in enumerate(events, 1):
        lines.append(f"  {i}. [{ev['start']:.1f}s - {ev['end']:.1f}s] {ev['description']}")
    return "\n".join(lines)


def call_llm(
    messages: list[dict],
    api_base: str,
    api_key: str,
    model: str,
    retries: int = 3,
) -> dict:
    """调用 OpenAI-compatible API，返回解析后的 JSON dict。"""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai is required: pip install openai")

    key = api_key or os.environ.get("NOVITA_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    client = OpenAI(api_key=key, base_url=api_base)

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=500,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content.strip()
            return _parse_json(content)
        except Exception as e:
            last_error = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt)

    return {"error": str(last_error), "overall_score": 0}


def _parse_json(text: str) -> dict:
    """Extract JSON from response, handling markdown wrapping."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try stripping markdown code fences
    m = re.search(r"```(?:json)?\s*(\{[\s\S]+?\})\s*```", text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m2 = re.search(r"\{[\s\S]+\}", text)
    if m2:
        try:
            return json.loads(m2.group(0))
        except json.JSONDecodeError:
            pass
    return {"_raw_response": text, "_parse_error": True, "overall_score": 0}


def assess_sample(sample: dict, api_base: str, api_key: str, model: str) -> dict:
    """评估单个样本的层次标注潜力。"""
    events = parse_events(sample)
    events_text = format_events_text(events)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
            duration=sample.get("duration", 0),
            source=sample.get("source", "unknown"),
            n_events=len(events),
            events_text=events_text,
        )},
    ]

    result = call_llm(messages, api_base, api_key, model)

    # 合并评估结果到样本
    assessed = dict(sample)
    assessed["_assessment"] = result
    assessed["_n_events"] = len(events)
    assessed["_events_parsed"] = events
    return assessed


def main():
    parser = argparse.ArgumentParser(
        description="LLM 层次标注潜力评估",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="passed.jsonl 路径")
    parser.add_argument("--output", required=True, help="assessed.jsonl 输出路径")
    parser.add_argument("--sample-n", type=int, default=200, help="抽样评估数量")
    parser.add_argument("--no-sample", action="store_true", help="全量评估")
    parser.add_argument("--api-base", default="https://api.novita.ai/v3/openai",
                        help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", default="",
                        help="API key (fallback: NOVITA_API_KEY / OPENAI_API_KEY env)")
    parser.add_argument("--model", default="pa/gmn-2.5-pr", help="模型名称")
    parser.add_argument("--workers", type=int, default=8, help="并发工作线程数")
    parser.add_argument("--score-threshold", type=int, default=3, help="通过阈值 (overall_score >=)")
    parser.add_argument("--resume", action="store_true", help="断点续评（跳过已评估的）")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("NOVITA_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""

    # 加载数据
    samples = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f"加载 {len(samples)} 条 passed 样本")
    print(f"API: {args.api_base}  model: {args.model}")

    # 断点续评
    assessed_videos = set()
    existing = []
    if args.resume and os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    assessed_videos.add(item.get("video", ""))
                    existing.append(item)
        samples = [s for s in samples if s.get("video", "") not in assessed_videos]
        print(f"已评估 {len(assessed_videos)} 条，剩余 {len(samples)} 条")

    # 抽样
    if not args.no_sample and len(samples) > args.sample_n:
        # 按 domain 分层抽样
        by_domain = {}
        for s in samples:
            d = s.get("source", "unknown")
            by_domain.setdefault(d, []).append(s)

        per_domain = max(1, args.sample_n // len(by_domain))
        chosen = []
        for domain, items in by_domain.items():
            n = min(per_domain, len(items))
            chosen.extend(random.sample(items, n))

        # 补足到 sample_n
        remaining = [s for s in samples if s not in chosen]
        if len(chosen) < args.sample_n and remaining:
            chosen.extend(random.sample(remaining, min(args.sample_n - len(chosen), len(remaining))))

        samples = chosen
        print(f"分层抽样 {len(samples)} 条进行评估")

    # 并发评估
    results = list(existing)
    failed = 0
    print(f"\n开始评估 {len(samples)} 条样本 (workers={args.workers})...")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(assess_sample, s, args.api_base, api_key, args.model): i
            for i, s in enumerate(samples)
        }

        for i, future in enumerate(as_completed(futures)):
            try:
                assessed = future.result()
                results.append(assessed)
                score = assessed.get("_assessment", {}).get("overall_score", 0)
                if (i + 1) % 20 == 0:
                    print(f"  [{i+1}/{len(samples)}] latest score={score}")
            except Exception as e:
                failed += 1
                print(f"  ⚠️ 样本 {futures[future]} 失败: {e}")

    # 统计
    scores = [r.get("_assessment", {}).get("overall_score", 0) for r in results if "_assessment" in r]
    valid_scores = [s for s in scores if s > 0]

    print(f"\n== 评估完成 ==")
    print(f"  成功: {len(valid_scores)}, 失败: {failed}")
    if valid_scores:
        print(f"  分数分布:")
        for threshold in [1, 2, 3, 4, 5]:
            count = sum(1 for s in valid_scores if s >= threshold)
            print(f"    ≥{threshold}: {count} ({count/len(valid_scores)*100:.1f}%)")
        print(f"  mean={sum(valid_scores)/len(valid_scores):.2f}")

    # 按 score 分档
    high = [r for r in results if r.get("_assessment", {}).get("overall_score", 0) >= 4]
    medium = [r for r in results if r.get("_assessment", {}).get("overall_score", 0) == 3]
    low = [r for r in results if r.get("_assessment", {}).get("overall_score", 0) < 3]
    print(f"\n  高潜力 (≥4): {len(high)}")
    print(f"  中等 (=3): {len(medium)}")
    print(f"  低潜力 (<3): {len(low)}")

    # Domain 分布
    domain_scores = {}
    for r in results:
        d = r.get("source", "unknown")
        s = r.get("_assessment", {}).get("overall_score", 0)
        if s > 0:
            domain_scores.setdefault(d, []).append(s)
    print(f"\n  各 Domain 平均分:")
    for d in sorted(domain_scores.keys()):
        scores_d = domain_scores[d]
        print(f"    {d}: mean={sum(scores_d)/len(scores_d):.2f} (n={len(scores_d)})")

    # 写出
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            # 清理不需要序列化的大字段
            r.pop("_events_parsed", None)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n✅ 评估结果 → {args.output}")

    # 按 threshold 输出高潜力样本
    if high:
        high_path = args.output.replace(".jsonl", "_high.jsonl")
        with open(high_path, "w", encoding="utf-8") as f:
            for r in high:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"✅ 高潜力样本 → {high_path} ({len(high)} 条)")


if __name__ == "__main__":
    main()
