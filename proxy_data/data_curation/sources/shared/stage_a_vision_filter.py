"""
Stage A — VLM 视觉校验 (数据源无关)

目的：对 Stage A 文本评估 keep 的样本做视觉安全网检查。
     用 VLM 查看实际视频帧，reject 纯文本看不出的低质量视频：
     - 讲解/新闻类 talking head
     - 静态场景（幻灯片、截屏、桌面录屏）
     - 画面整体无明显动作变化
     - 游戏、动画等非真实物理场景

输入: stage_a 产出的 *_keep.jsonl（已含 _assessment 字段）
输出:
  - vision_results.jsonl         — 全部视觉评估结果
  - vision_results_keep.jsonl    — 视觉通过 → final candidates
  - vision_results_reject.jsonl  — 视觉淘汰

用法:
    python stage_a_vision_filter.py \\
        --input results/stage_a_results_keep.jsonl \\
        --output results/vision_results.jsonl \\
        --video-root /path/to/videos \\
        --workers 4

    # ET-Instruct 模式：video 字段 = "coin/xxx.mp4"
    python stage_a_vision_filter.py \\
        --input results/stage_a_results_keep.jsonl \\
        --output results/vision_results.jsonl \\
        --video-root /data/et_instruct/videos \\
        --video-field video \\
        --workers 4

    # TimeLens 模式：video_path 字段
    python stage_a_vision_filter.py \\
        --input results/stage_a_results_keep.jsonl \\
        --output results/vision_results.jsonl \\
        --video-root /data/timelens/videos \\
        --video-field video_path \\
        --workers 4
"""

import json
import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.video_sampler import sample_frames_base64

# ── VLM Prompt ──────────────────────────────────────────

VISION_SYSTEM = """\
You are a visual quality assessor for a video dataset curation pipeline. \
You are given 6 uniformly sampled frames from a video.

Your job: Determine whether this video depicts REAL-WORLD PHYSICAL ACTIVITIES \
with visible action changes across frames.

KEEP the video if:
- Frames show a person (or people) performing physical tasks with real objects
- There are visible state changes across frames (e.g., object moves, scene \
changes, tool usage, different stages of a process)
- The activity progresses through distinct phases

REJECT the video if:
- Talking head / interview / news anchor: person mostly static, just talking
- Static scene: no meaningful visual change across the 6 frames
- Screen recording / gaming / animation / slides / text-heavy content
- Monolithic activity with no visible progression (all frames look nearly identical)
- Very dark, blurry, or low-quality footage where actions are indiscernible

## Examples

### Example 1 — physical cooking activity (keep)
Frames show: person at kitchen counter → chopping vegetables → stirring pot → \
plating food → different angles of cooking process
Output: {"visual_analysis":"Clear cooking progression with tool usage and \
state changes across 6 frames. Kitchen scene with distinct phases.","visual_quality_score":5,"is_physical_activity":true,"decision":"keep"}

### Example 2 — talking head / lecture (reject)
Frames show: same person at desk → same angle → same background → occasional \
hand gesture → same setting
Output: {"visual_analysis":"Static talking head. No meaningful visual change \
across frames. Person seated at desk throughout.","visual_quality_score":1,"is_physical_activity":false,"decision":"reject"}

Return ONLY valid JSON."""

VISION_USER = """\
I'm showing you 6 uniformly-sampled frames from a video (duration: {duration:.1f}s, \
domain: {source}).

Based on the visual content of these frames, assess whether this video shows \
real-world physical activities with visible progression.

Respond with ONLY valid JSON:
{{
  "visual_analysis": "<1-2 sentences describing what you see>",
  "visual_quality_score": <1-5>,
  "is_physical_activity": true | false,
  "decision": "keep | reject"
}}"""


# ── Decision Rules ──────────────────────────────────────

def apply_vision_rules(assessment: dict) -> str:
    """Apply programmatic rules to vision assessment."""
    if assessment.get("_parse_error") or assessment.get("error"):
        return "reject"

    score = assessment.get("visual_quality_score", 0)
    is_physical = assessment.get("is_physical_activity", False)

    if not isinstance(score, (int, float)):
        return "reject"

    # Hard reject: low visual quality or not physical
    if score <= 2 or not is_physical:
        return "reject"

    return "keep"


# ── VLM Call ────────────────────────────────────────────

def call_vlm(
    messages: list[dict],
    api_base: str,
    api_key: str,
    model: str,
    retries: int = 3,
    temperature: float = 0.1,
    max_tokens: int = 300,
) -> dict:
    """Call VLM API with image content and return parsed JSON dict."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai is required: pip install openai")

    from shared.llm_client import parse_json

    key = (
        api_key
        or os.environ.get("NOVITA_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    )
    client = OpenAI(api_key=key, base_url=api_base)

    last_error = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content.strip()
            return parse_json(content)
        except Exception as e:
            last_error = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt)

    return {"error": str(last_error), "_parse_error": True}


# ── Assessment ──────────────────────────────────────────

def assess_sample(
    sample: dict,
    video_root: str,
    video_field: str,
    api_base: str,
    api_key: str,
    model: str,
    n_frames: int = 6,
) -> dict:
    """Run VLM vision assessment on a single sample."""
    video_rel = sample.get(video_field, "")
    video_path = os.path.join(video_root, video_rel)

    if not os.path.exists(video_path):
        assessed = dict(sample)
        assessed["_vision"] = {
            "error": f"Video not found: {video_path}",
            "_parse_error": True,
            "decision": "reject",
        }
        assessed["_vision_stage"] = "vision"
        return assessed

    # Extract frames
    try:
        frames_b64 = sample_frames_base64(
            video_path, n_frames=n_frames, max_side=512
        )
    except Exception as e:
        assessed = dict(sample)
        assessed["_vision"] = {
            "error": f"Frame extraction failed: {e}",
            "_parse_error": True,
            "decision": "reject",
        }
        assessed["_vision_stage"] = "vision"
        return assessed

    if not frames_b64:
        assessed = dict(sample)
        assessed["_vision"] = {
            "error": "No frames extracted",
            "_parse_error": True,
            "decision": "reject",
        }
        assessed["_vision_stage"] = "vision"
        return assessed

    # Build multimodal message
    user_content = []
    # Text part
    user_content.append({
        "type": "text",
        "text": VISION_USER.format(
            duration=sample.get("duration", 0),
            source=sample.get("source", "unknown"),
        ),
    })
    # Image parts
    for b64 in frames_b64:
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64}",
            },
        })

    messages = [
        {"role": "system", "content": VISION_SYSTEM},
        {"role": "user", "content": user_content},
    ]

    result = call_vlm(messages, api_base, api_key, model)

    # Apply rules
    llm_decision = result.get("decision", "unknown")
    rule_decision = apply_vision_rules(result)
    if llm_decision != rule_decision:
        result["_original_decision"] = llm_decision
        result["decision"] = rule_decision

    assessed = dict(sample)
    assessed["_vision"] = result
    assessed["_vision_stage"] = "vision"
    return assessed


def print_stats(results: list[dict]):
    """Print vision assessment statistics."""
    assessments = [
        r["_vision"]
        for r in results
        if "_vision" in r and not r["_vision"].get("_parse_error")
    ]

    if not assessments:
        print("  无有效视觉评估结果")
        return

    # Decision distribution
    decisions = {}
    for a in assessments:
        d = a.get("decision", "unknown")
        decisions[d] = decisions.get(d, 0) + 1
    print(f"\n  == Vision Decision 分布 ==")
    for d, c in sorted(decisions.items(), key=lambda x: -x[1]):
        print(f"    {d}: {c} ({c/len(assessments)*100:.1f}%)")

    # Score distribution
    scores = [
        a.get("visual_quality_score", 0)
        for a in assessments
        if isinstance(a.get("visual_quality_score"), (int, float))
    ]
    if scores:
        print(f"\n  == visual_quality_score 分布 ==")
        for threshold in [1, 2, 3, 4, 5]:
            count = sum(1 for s in scores if s >= threshold)
            print(f"    >= {threshold}: {count} ({count/len(scores)*100:.1f}%)")
        print(f"    mean={sum(scores)/len(scores):.2f}")

    # Physical activity ratio
    physical = sum(1 for a in assessments if a.get("is_physical_activity"))
    print(f"\n  == is_physical_activity ==")
    print(f"    true:  {physical} ({physical/len(assessments)*100:.1f}%)")
    print(f"    false: {len(assessments)-physical} ({(len(assessments)-physical)/len(assessments)*100:.1f}%)")

    # Rule overrides
    overrides = sum(1 for a in assessments if "_original_decision" in a)
    if overrides:
        print(f"\n  == 规则覆盖 ==")
        print(f"    覆盖总数: {overrides}/{len(assessments)} ({overrides/len(assessments)*100:.1f}%)")

    # Per-domain stats
    domain_decisions: dict[str, dict] = {}
    for r in results:
        if "_vision" not in r or r["_vision"].get("_parse_error"):
            continue
        domain = r.get("source", "unknown")
        decision = r["_vision"].get("decision", "unknown")
        domain_decisions.setdefault(domain, {})
        domain_decisions[domain][decision] = domain_decisions[domain].get(decision, 0) + 1
    if domain_decisions:
        print(f"\n  == 各 Domain Vision Decision ==")
        for domain in sorted(domain_decisions.keys()):
            dd = domain_decisions[domain]
            total = sum(dd.values())
            parts = ", ".join(f"{k}={v}" for k, v in sorted(dd.items()))
            print(f"    {domain}: {parts} (total={total})")

    # Error stats
    errors = [r for r in results if "_vision" in r and r["_vision"].get("_parse_error")]
    if errors:
        print(f"\n  == 错误/解析失败 ==")
        print(f"    总数: {len(errors)}")


# ── Main ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VLM 视觉校验: 对 Stage A keep 样本做视觉安全网筛选",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="stage_a *_keep.jsonl")
    parser.add_argument("--output", required=True, help="vision_results.jsonl")
    parser.add_argument("--video-root", required=True, help="视频文件根目录")
    parser.add_argument("--video-field", default="video",
                        help="样本中视频路径字段名 (et_instruct=video, timelens=video_path)")
    parser.add_argument("--n-frames", type=int, default=6, help="抽取帧数")
    parser.add_argument("--sample-n", type=int, default=0,
                        help="抽样数量（0=全量）")
    parser.add_argument("--no-sample", action="store_true", help="全量评估")
    parser.add_argument("--api-base", default="https://api.novita.ai/v3/openai")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default="pa/gmn-2.5-pr",
                        help="VLM model (须支持 vision)")
    parser.add_argument("--workers", type=int, default=4,
                        help="并发数（视觉任务较重，默认 4）")
    parser.add_argument("--resume", action="store_true", help="断点续评")
    args = parser.parse_args()

    api_key = (
        args.api_key
        or os.environ.get("NOVITA_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    )

    # Determine ID field from video_field
    id_field = args.video_field

    # Load data
    samples = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f"加载 {len(samples)} 条 Stage A keep 样本")
    print(f"API: {args.api_base}  model: {args.model}")
    print(f"Video root: {args.video_root}")
    print(f"Video field: {args.video_field}, N frames: {args.n_frames}")

    # Resume
    existing: list[dict] = []
    if args.resume:
        from shared.llm_client import load_checkpoint
        assessed_ids, existing = load_checkpoint(args.output, id_field=id_field)
        samples = [s for s in samples if s.get(id_field, "") not in assessed_ids]
        print(f"已评估 {len(assessed_ids)} 条，剩余 {len(samples)} 条")

    # Sampling
    if args.sample_n > 0 and not args.no_sample and len(samples) > args.sample_n:
        from shared.llm_client import stratified_sample
        samples = stratified_sample(samples, args.sample_n)
        print(f"分层抽样 {len(samples)} 条进行评估")

    # Concurrent assessment with streaming output
    results: list[dict] = []
    failed = 0
    _write_lock = threading.Lock()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    stream_f = open(args.output, "a", encoding="utf-8")

    print(f"\n开始视觉评估 {len(samples)} 条样本 (workers={args.workers})...")

    def _assess(s):
        return assess_sample(
            s, args.video_root, args.video_field,
            args.api_base, api_key, args.model, args.n_frames,
        )

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_assess, s): i for i, s in enumerate(samples)}
            for i, future in enumerate(as_completed(futures)):
                try:
                    assessed = future.result()
                    results.append(assessed)

                    # Stream write
                    row = dict(assessed)
                    line = json.dumps(row, ensure_ascii=False) + "\n"
                    with _write_lock:
                        stream_f.write(line)
                        stream_f.flush()

                    if (i + 1) % 10 == 0:
                        vision = assessed.get("_vision", {})
                        decision = vision.get("decision", "?")
                        score = vision.get("visual_quality_score", "?")
                        print(f"  [{i+1}/{len(samples)}] score={score} decision={decision}")
                except Exception as e:
                    failed += 1
                    print(f"  样本 {futures[future]} 失败: {e}")
    finally:
        stream_f.close()

    all_results = existing + results
    print(f"\n== VLM 视觉评估完成 ==")
    print(f"  成功: {len(results) - failed}, 失败: {failed}, 总计: {len(all_results)}")

    # Stats
    print_stats(all_results)

    # Split by decision
    keep = [r for r in all_results if r.get("_vision", {}).get("decision") == "keep"]
    reject = [r for r in all_results if r.get("_vision", {}).get("decision") != "keep"]

    print(f"\n  == 最终视觉筛选结果 ==")
    print(f"    keep:   {len(keep)}")
    print(f"    reject: {len(reject)}")

    # Write split files
    from shared.llm_client import write_results as _write
    base = args.output.replace(".jsonl", "")
    if keep:
        _write(keep, f"{base}_keep.jsonl")
    if reject:
        _write(reject, f"{base}_reject.jsonl")

    print(f"\nVLM 视觉校验完成。keep 样本为最终候选（final_candidates）。")


if __name__ == "__main__":
    main()
