"""
ET-Instruct-164K 文本筛选脚本

目标：从 163K 样本中筛选适合进入"层次分割标注"的候选视频。
筛选策略：纯文本 + 元数据过滤，不需要视频文件。

用法:
    python text_filter.py \
        --json_path /m2v_intern/xuboshen/zgw/data/ET-Instruct-164K/et_instruct_164k_txt.json \
        --output_dir results \
        --config ../../configs/et_instruct_164k.yaml

筛选逻辑:
    1. 时长过滤: 60s ≤ duration ≤ 240s
    2. 事件密度: tgt 中至少包含 5 对时间戳 (≥5 个事件，保证 L1/L3 空间)
    3. 任务类型: 不硬过滤，min_events 自然淘汰 tvg 等单事件 task
    4. 文本质量: GPT 回复至少包含 50 字符
    5. 去重: 同一视频只保留事件数最多的样本
    6. 领域上限: 每个 source 最多 5000 条
"""

import json
import argparse
import os
from collections import defaultdict
from pathlib import Path


# ── 筛选规则 ─────────────────────────────────────────────

# ── Task 类型分析（基于 163880 样本实际分布）──
#
# 适合层次分割（多事件、有步骤结构）:
#   slc  (27659) - step localization & captioning → 完美：多步骤 + 时间戳
#   dvc  ( 9830) - dense video captioning → 好：多事件密集描述
#   tal  (25025) - temporal action localization → 可用：多动作段
#   evs  ( 9056) - event summarization → 可用：多事件摘要
#   tvc  ( 2908) - temporal video captioning → 可用
#
# 不太适合（单事件或非时序结构）:
#   tvg  (43960) - temporal video grounding → 通常单事件定位，事件数 = 1
#   rvc  (16907) - reverse video captioning → 单段描述
#   epm  (10546) - episodic memory → 单段查询
#   gvq  (10000) - grounded video QA → 单段 QA
#   vhd  ( 7989) - video highlight detection → 可能多段但结构不同
#
# 策略: 不硬过滤 task，用 min_events >= 5 自然淘汰单事件/少事件任务
PREFERRED_TASKS = None  # None = 不按 task 过滤；min_events 已可自然筛除

DEFAULT_FILTER = {
    "min_duration": 60,       # 最短时长（秒）— 太短无层次
    "max_duration": 240,      # 最长时长（秒）— 240s 内足够 3 层分割
    "min_events": 5,          # 最少事件数 — ≥5 个事件才有 L1 聚合 + L3 分解空间
    "min_text_length": 50,    # GPT 回复最短字符数
    "max_per_domain": 5000,   # 每个 source 上限
    "dedup_by_video": True,   # 同一视频去重，保留事件最多的
}


def load_config(config_path: str | None) -> dict:
    """加载 YAML 配置；如果没有则用默认值。"""
    if config_path and os.path.exists(config_path):
        try:
            import yaml
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            tf = cfg.get("text_filter", {})
            db = cfg.get("domain_balance", {})
            return {
                "min_duration": tf.get("min_duration_sec", DEFAULT_FILTER["min_duration"]),
                "max_duration": tf.get("max_duration_sec", DEFAULT_FILTER["max_duration"]),
                "min_events": tf.get("min_events", DEFAULT_FILTER["min_events"]),
                "min_text_length": tf.get("min_text_length", DEFAULT_FILTER["min_text_length"]),
                "max_per_domain": db.get("max_per_domain", DEFAULT_FILTER["max_per_domain"]),
                "dedup_by_video": DEFAULT_FILTER["dedup_by_video"],
            }
        except ImportError:
            print("⚠️ pyyaml 未安装，使用默认配置")
    return dict(DEFAULT_FILTER)


def count_events(sample: dict) -> int:
    """从 tgt 字段计算事件数量（成对时间戳）。"""
    tgt = sample.get("tgt", [])
    return len(tgt) // 2


def get_gpt_text(sample: dict) -> str:
    """提取 GPT 回复文本。"""
    for turn in sample.get("conversations", []):
        if turn.get("from") == "gpt":
            return turn.get("value", "")
    return ""


def filter_sample(sample: dict, cfg: dict) -> tuple[bool, str]:
    """
    判断单个样本是否通过筛选。
    Returns: (passed: bool, reason: str)
    """
    duration = sample.get("duration", 0)

    # 1. 时长
    if duration < cfg["min_duration"]:
        return False, f"duration_too_short:{duration:.1f}s"
    if duration > cfg["max_duration"]:
        return False, f"duration_too_long:{duration:.1f}s"

    # 2. 事件密度
    n_events = count_events(sample)
    if n_events < cfg["min_events"]:
        return False, f"too_few_events:{n_events}"

    # 3. 任务类型
    if PREFERRED_TASKS is not None:
        task = sample.get("task", "")
        if task not in PREFERRED_TASKS:
            return False, f"task_excluded:{task}"

    # 4. 文本质量
    gpt_text = get_gpt_text(sample)
    if len(gpt_text) < cfg["min_text_length"]:
        return False, f"text_too_short:{len(gpt_text)}chars"

    return True, "ok"


def dedup_by_video(samples: list[dict]) -> list[dict]:
    """同一视频保留事件数最多的样本。"""
    best = {}
    for s in samples:
        vid = s.get("video", "")
        n = count_events(s)
        if vid not in best or n > count_events(best[vid]):
            best[vid] = s
    return list(best.values())


def apply_domain_cap(samples: list[dict], max_per_domain: int) -> tuple[list[dict], dict]:
    """
    每个 domain 最多保留 max_per_domain 条。
    优先保留事件数多的样本。
    Returns: (capped_samples, domain_counts_before_cap)
    """
    by_domain = defaultdict(list)
    for s in samples:
        by_domain[s.get("source", "unknown")].append(s)

    counts_before = {d: len(v) for d, v in by_domain.items()}
    result = []
    for domain, items in by_domain.items():
        items.sort(key=lambda x: count_events(x), reverse=True)
        result.extend(items[:max_per_domain])

    return result, counts_before


def main():
    parser = argparse.ArgumentParser(description="ET-Instruct-164K 文本筛选")
    parser.add_argument("--json_path", required=True, help="et_instruct_164k_txt.json 路径")
    parser.add_argument("--output_dir", default="results", help="输出目录")
    parser.add_argument("--config", default=None, help="YAML 配置文件路径")
    parser.add_argument("--dry_run", action="store_true", help="仅统计，不写文件")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"筛选配置: {json.dumps(cfg, indent=2)}")

    # 加载数据
    print(f"\n加载: {args.json_path}")
    with open(args.json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"总样本数: {len(data)}")

    # ── Step 1: 逐条筛选 ──
    passed, rejected = [], []
    reject_reasons = defaultdict(int)

    for sample in data:
        ok, reason = filter_sample(sample, cfg)
        if ok:
            sample["_n_events"] = count_events(sample)
            passed.append(sample)
        else:
            sample["_reject_reason"] = reason
            rejected.append(sample)
            # 取 reason 的类别部分
            reject_reasons[reason.split(":")[0]] += 1

    print(f"\n== Step 1: 逐条筛选 ==")
    print(f"  通过: {len(passed)}")
    print(f"  拒绝: {len(rejected)}")
    print(f"  拒绝原因分布:")
    for reason, count in sorted(reject_reasons.items(), key=lambda x: -x[1]):
        print(f"    {reason}: {count}")

    # ── Step 2: 去重 ──
    if cfg["dedup_by_video"]:
        before = len(passed)
        passed = dedup_by_video(passed)
        print(f"\n== Step 2: 视频去重 ==")
        print(f"  {before} → {len(passed)} (去除 {before - len(passed)} 条重复)")

    # ── Step 3: 领域均衡 ──
    passed, domain_counts_before = apply_domain_cap(passed, cfg["max_per_domain"])
    print(f"\n== Step 3: 领域均衡 (上限 {cfg['max_per_domain']}/domain) ==")
    print(f"  各 domain 通过数:")
    domain_final = defaultdict(int)
    for s in passed:
        domain_final[s.get("source", "unknown")] += 1
    for domain in sorted(domain_final.keys()):
        before = domain_counts_before.get(domain, 0)
        after = domain_final[domain]
        cap_mark = " (capped)" if after < before else ""
        print(f"    {domain}: {before} → {after}{cap_mark}")

    print(f"\n最终候选: {len(passed)}")

    # ── 事件数分布 ──
    event_counts = [s.get("_n_events", 0) for s in passed]
    if event_counts:
        print(f"\n事件数分布:")
        print(f"  min={min(event_counts)}, max={max(event_counts)}, "
              f"mean={sum(event_counts)/len(event_counts):.1f}")

    # ── Task 分布 ──
    task_dist = defaultdict(int)
    for s in passed:
        task_dist[s.get("task", "unknown")] += 1
    print(f"\n任务类型分布:")
    for task, count in sorted(task_dist.items(), key=lambda x: -x[1]):
        print(f"    {task}: {count}")

    # ── 写出 ──
    if not args.dry_run:
        os.makedirs(args.output_dir, exist_ok=True)

        # 添加溯源元数据
        origin_meta = {
            "dataset": "ET-Instruct-164K",
            "source_file": os.path.abspath(args.json_path),
            "filter_config": os.path.abspath(args.config) if args.config else None,
            "filter_params": cfg,
        }

        passed_path = os.path.join(args.output_dir, "passed.jsonl")
        with open(passed_path, "w", encoding="utf-8") as f:
            for s in passed:
                s.pop("_n_events", None)
                s["_origin"] = origin_meta
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"\n✅ passed → {passed_path} ({len(passed)} 条)")

        rejected_path = os.path.join(args.output_dir, "rejected.jsonl")
        with open(rejected_path, "w", encoding="utf-8") as f:
            for s in rejected:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"✅ rejected → {rejected_path} ({len(rejected)} 条)")

        # 统计 summary
        summary = {
            "total_input": len(data),
            "passed_after_filter": len(passed) + (len(data) - len(passed) - len(rejected)),
            "passed_after_dedup": len(passed),
            "rejected": len(rejected),
            "reject_reasons": dict(reject_reasons),
            "domain_final": dict(domain_final),
            "task_distribution": dict(task_dist),
            "config_used": cfg,
            "origin": origin_meta,
        }
        summary_path = os.path.join(args.output_dir, "filter_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"✅ summary → {summary_path}")
    else:
        print("\n(dry_run 模式，未写文件)")


if __name__ == "__main__":
    main()
