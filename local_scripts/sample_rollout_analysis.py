#!/usr/bin/env python3
"""
Rollout Sampling & Analysis Tool
=================================
分析 rollout JSONL 文件，诊断：
  - 模型是随机答对 还是真正习得能力
  - 奖励曲线平缓 / val 增长不大 的原因
  - 高分 / 中分 / 低分 样本的具体表现

用法:
  python local_scripts/sample_rollout_analysis.py \\
      --rollout_dir /path/to/rollout_dir \\
      --output_dir  /path/to/output \\
      [--task_type aot_v2t]    # 留空则分析全部 task
      [--step_range 10,50]     # 只看指定 step 范围
      [--n_samples 20]         # 每类采样数量
      [--min_attempts 2]       # 至少有 N 次 rollout 才参与分析

输出目录结构:
  output_dir/
    summary.txt               总体统计
    high_consistent.txt       高分且稳定（真学到了）
    high_random.txt           高均分但方差大（可能随机）
    mid_random.txt            中分高方差（最典型随机猜答）
    low_consistent.txt        低分且稳定（真的不会）
    low_random.txt            低均分但偶尔答对（边界）
    per_step_stats.txt        每 step 的奖励统计 + 随机指数
    per_task_stats.txt        各 task_type 统计
    raw_samples/              各类别详细样本 txt
"""

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


# ──────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────

def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))


def parse_reward(raw: Any) -> float:
    """兼容 reward 为 float 或 dict(overall=...) 两种格式"""
    if isinstance(raw, dict):
        return safe_float(raw.get("overall", raw.get("reward", 0.0)))
    return safe_float(raw)


# ──────────────────────────────────────────────────────────────
# 加载 rollout
# ──────────────────────────────────────────────────────────────

def load_rollout_dir(
    rollout_dir: Path,
    task_filter: str | None,
    step_range: tuple[int, int] | None,
) -> dict[str, dict]:
    """
    返回 groups: uid -> {
        uid, problem_type, prompt, ground_truth,
        step, phase,
        attempts: [{reward, response, attempt_index}],
        mean_reward, reward_std, all_1_rate, all_0_rate, random_index
    }
    """
    files = sorted(rollout_dir.glob("step_*.jsonl")) + sorted(rollout_dir.glob("val_step_*.jsonl"))
    if not files:
        files = sorted(rollout_dir.glob("*.jsonl"))
    if not files:
        sys.exit(f"[ERROR] No jsonl files found in {rollout_dir}")

    groups: dict[str, dict] = {}
    total_records = 0

    for f in files:
        # 从文件名解析 step & phase
        fname = f.stem  # e.g. step_10 / val_step_10
        if fname.startswith("val_step_"):
            phase = "val"
            step_num = int(fname.split("_")[-1])
        elif fname.startswith("step_"):
            phase = "train"
            step_num = int(fname.split("_")[-1])
        else:
            phase = "train"
            step_num = 0

        if step_range and not (step_range[0] <= step_num <= step_range[1]):
            continue

        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                task = str(rec.get("problem_type") or rec.get("task_type") or "unknown")
                if task_filter and task_filter not in task:
                    continue

                reward = parse_reward(rec.get("reward"))
                # 跳过 reward 缺失的最后一个未完成 step
                if rec.get("reward") is None:
                    continue

                uid = str(rec.get("uid") or f"{phase}-{step_num}-{total_records}")
                response = str(rec.get("response") or "")
                prompt = str(rec.get("prompt") or "")
                gt = str(rec.get("ground_truth") or "")

                attempt = {
                    "reward": reward,
                    "response": response,
                    "attempt_index": len(groups.get(uid, {}).get("attempts", [])),
                }

                if uid not in groups:
                    groups[uid] = {
                        "uid": uid,
                        "problem_type": task,
                        "prompt": prompt,
                        "ground_truth": gt,
                        "step": step_num,
                        "phase": phase,
                        "attempts": [],
                    }
                groups[uid]["attempts"].append(attempt)
                total_records += 1

    print(f"[load] {len(files)} files, {total_records} records, {len(groups)} unique questions")

    # 计算统计量
    for g in groups.values():
        rewards = [a["reward"] for a in g["attempts"]]
        g["mean_reward"] = sum(rewards) / len(rewards)
        g["reward_std"] = std(rewards)
        g["all_1_rate"] = sum(1 for r in rewards if r >= 0.99) / len(rewards)
        g["all_0_rate"] = sum(1 for r in rewards if r <= 0.01) / len(rewards)
        # 随机指数：标准差越高、均值越接近0.5，越随机
        # 取值 [0,1]，越高说明越像随机猜测
        mean_r = g["mean_reward"]
        g["random_index"] = g["reward_std"] * (1 - abs(mean_r - 0.5) * 2)

    return groups


# ──────────────────────────────────────────────────────────────
# 分类
# ──────────────────────────────────────────────────────────────

CATEGORIES = {
    "high_consistent": "高分稳定（mean≥0.8, std<0.2）→ 真正学会",
    "high_random":     "高均分但方差大（mean≥0.5, std≥0.3）→ 疑似运气",
    "mid_random":      "中分高方差（0.2<mean<0.8, std≥0.3）→ 最典型随机猜答",
    "low_random":      "低均分偶尔对（mean<0.5, std≥0.3）→ 偶然答对",
    "low_consistent":  "低分稳定（mean<0.2, std<0.2）→ 真的不会",
}


def categorize(groups: dict, min_attempts: int) -> dict[str, list[str]]:
    cats: dict[str, list[str]] = {k: [] for k in CATEGORIES}

    for uid, g in groups.items():
        if len(g["attempts"]) < min_attempts:
            continue
        m = g["mean_reward"]
        s = g["reward_std"]

        if m >= 0.8 and s < 0.2:
            cats["high_consistent"].append(uid)
        elif m >= 0.5 and s >= 0.3:
            cats["high_random"].append(uid)
        elif 0.2 < m < 0.8 and s >= 0.3:
            cats["mid_random"].append(uid)
        elif m < 0.5 and s >= 0.3:
            cats["low_random"].append(uid)
        elif m < 0.2 and s < 0.2:
            cats["low_consistent"].append(uid)

    return cats


# ──────────────────────────────────────────────────────────────
# 格式化输出
# ──────────────────────────────────────────────────────────────

def fmt_group(g: dict, idx: int) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append(f"[{idx+1}] uid={g['uid']}  task={g['problem_type']}  step={g['step']}({g['phase']})")
    lines.append(f"    mean_reward={g['mean_reward']:.3f}  std={g['reward_std']:.3f}"
                 f"  all1_rate={g['all_1_rate']:.2f}  all0_rate={g['all_0_rate']:.2f}"
                 f"  random_idx={g['random_index']:.3f}")
    lines.append("")
    # 截断 prompt，保留核心部分
    prompt_short = g["prompt"][-600:] if len(g["prompt"]) > 600 else g["prompt"]
    lines.append(f"PROMPT (tail 600 chars):\n{prompt_short}")
    lines.append(f"\nGROUND_TRUTH: {g['ground_truth']}")
    lines.append("")
    for i, att in enumerate(g["attempts"]):
        resp = att["response"][:400] if len(att["response"]) > 400 else att["response"]
        lines.append(f"  -- attempt {i}  reward={att['reward']:.3f} --")
        lines.append(f"  {resp}")
        lines.append("")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# 统计
# ──────────────────────────────────────────────────────────────

def build_summary(groups: dict, cats: dict) -> str:
    total = len(groups)
    mean_all = sum(g["mean_reward"] for g in groups.values()) / max(total, 1)
    std_all = std([g["mean_reward"] for g in groups.values()])
    high_ri = [g for g in groups.values() if g["random_index"] > 0.2]
    random_ratio = len(high_ri) / max(total, 1)

    lines = ["=" * 80, "ROLLOUT ANALYSIS SUMMARY", "=" * 80, ""]
    lines.append(f"Total unique questions : {total}")
    lines.append(f"Overall mean reward    : {mean_all:.4f}")
    lines.append(f"Std of mean rewards    : {std_all:.4f}")
    lines.append(f"High-random-index (>0.2) questions: {len(high_ri)} ({random_ratio:.1%})")
    lines.append("")

    lines.append("── Category breakdown ──")
    for k, desc in CATEGORIES.items():
        n = len(cats[k])
        lines.append(f"  {k:20s}: {n:5d} ({n/max(total,1):.1%})  # {desc}")
    lines.append("")

    # 诊断建议
    lines.append("── 诊断建议 ──")
    if random_ratio > 0.3:
        lines.append("  ⚠  超过30%的问题表现出高随机性（random_index>0.2）")
        lines.append("     → 奖励平缓可能由随机猜答贡献，模型未真正学习该能力")
    mc = len(cats["mid_random"])
    lc = len(cats["low_consistent"])
    if mc > total * 0.2:
        lines.append(f"  ⚠  mid_random占比{mc/total:.1%}，中分高方差问题多")
        lines.append("     → 模型在这些样本上随机猜，val不增是因为没有稳定学会")
    if lc > total * 0.3:
        lines.append(f"  ⚠  low_consistent占比{lc/total:.1%}，有大量模型完全不会的样本")
        lines.append("     → 这些样本提供的梯度信号几乎为0，考虑课程学习/难度过滤")
    hc = len(cats["high_consistent"])
    if hc > total * 0.4:
        lines.append(f"  ℹ  high_consistent占比{hc/total:.1%}，超过40%已学会")
        lines.append("     → 这些样本已'饱和'，可以考虑减少其采样权重")
    lines.append("")
    return "\n".join(lines)


def build_per_step_stats(groups: dict) -> str:
    step_data: dict[str, list] = defaultdict(list)
    for g in groups.values():
        key = f"{g['phase']}:{g['step']:05d}"
        step_data[key].append(g)

    lines = ["step              | n_q  | mean_rwd | rwd_std | random_idx_mean | pct_random"]
    lines.append("-" * 85)
    for key in sorted(step_data.keys()):
        gs = step_data[key]
        n = len(gs)
        mr = sum(g["mean_reward"] for g in gs) / n
        rs = std([g["mean_reward"] for g in gs])
        ri = sum(g["random_index"] for g in gs) / n
        pct_r = sum(1 for g in gs if g["random_index"] > 0.2) / n
        lines.append(f"{key:18s} | {n:4d} | {mr:8.4f} | {rs:7.4f} | {ri:15.4f} | {pct_r:9.1%}")
    return "\n".join(lines)


def build_per_task_stats(groups: dict) -> str:
    task_data: dict[str, list] = defaultdict(list)
    for g in groups.values():
        task_data[g["problem_type"]].append(g)

    lines = ["task_type            | n_q  | mean_rwd | rwd_std | random_idx | pct_consistent | pct_random"]
    lines.append("-" * 95)
    for task in sorted(task_data.keys()):
        gs = task_data[task]
        n = len(gs)
        mr = sum(g["mean_reward"] for g in gs) / n
        rs = std([g["mean_reward"] for g in gs])
        ri = sum(g["random_index"] for g in gs) / n
        pct_cons = sum(1 for g in gs if g["all_1_rate"] > 0.8) / n
        pct_rand = sum(1 for g in gs if g["random_index"] > 0.2) / n
        lines.append(f"{task:21s} | {n:4d} | {mr:8.4f} | {rs:7.4f} | {ri:10.4f} | {pct_cons:14.1%} | {pct_rand:9.1%}")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Rollout sampling & analysis")
    parser.add_argument("--rollout_dir", required=True, help="rollout JSONL 目录")
    parser.add_argument("--output_dir",  required=True, help="输出目录")
    parser.add_argument("--task_type",   default=None,  help="只分析指定 task，如 aot_v2t")
    parser.add_argument("--step_range",  default=None,  help="step 范围，如 0,60")
    parser.add_argument("--n_samples",   type=int, default=20, help="每类采样数量")
    parser.add_argument("--min_attempts",type=int, default=2,  help="最少 rollout 次数")
    parser.add_argument("--sort_by",     default="random_index",
                        choices=["random_index", "mean_reward", "reward_std"],
                        help="类内排序字段")
    args = parser.parse_args()

    rollout_dir = Path(args.rollout_dir)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "raw_samples").mkdir(exist_ok=True)

    step_range = None
    if args.step_range:
        lo, hi = args.step_range.split(",")
        step_range = (int(lo), int(hi))

    # ── 加载 ──
    groups = load_rollout_dir(rollout_dir, args.task_type, step_range)
    if not groups:
        sys.exit("[ERROR] No data loaded, check filters.")

    # ── 分类 ──
    cats = categorize(groups, args.min_attempts)

    # ── 写 summary ──
    summary = build_summary(groups, cats)
    (output_dir / "summary.txt").write_text(summary)
    print(summary)

    # ── 写 per_step_stats ──
    (output_dir / "per_step_stats.txt").write_text(build_per_step_stats(groups))
    print("[wrote] per_step_stats.txt")

    # ── 写 per_task_stats ──
    (output_dir / "per_task_stats.txt").write_text(build_per_task_stats(groups))
    print("[wrote] per_task_stats.txt")

    # ── 各类别采样并写文件 ──
    import random
    random.seed(42)

    for cat_name, desc in CATEGORIES.items():
        uid_list = cats[cat_name]
        # 按指定字段排序（取 top-N 最具代表性）
        uid_list_sorted = sorted(
            uid_list,
            key=lambda u: groups[u][args.sort_by],
            reverse=(args.sort_by != "mean_reward" if cat_name.startswith("low") else True),
        )
        sampled = uid_list_sorted[:args.n_samples]

        lines = [f"CATEGORY: {cat_name}", f"DESC    : {desc}",
                 f"Total   : {len(uid_list)}  Shown: {len(sampled)}", ""]

        for i, uid in enumerate(sampled):
            lines.append(fmt_group(groups[uid], i))

        out_path = output_dir / f"{cat_name}.txt"
        out_path.write_text("\n".join(lines), encoding="utf-8")

        # 也写到 raw_samples/
        (output_dir / "raw_samples" / f"{cat_name}_full.txt").write_text(
            "\n".join(fmt_group(groups[u], i) for i, u in enumerate(uid_list_sorted)),
            encoding="utf-8",
        )
        print(f"[wrote] {cat_name}.txt  ({len(uid_list)} total, {len(sampled)} sampled)")

    print(f"\n[done] All outputs -> {output_dir}")


if __name__ == "__main__":
    main()
