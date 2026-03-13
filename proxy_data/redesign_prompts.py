#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新设计选择题/排序题的 Prompt，添加 Chain-of-Thought 指令。

问题背景:
    当前选择题 (add/delete/replace) 的 prompt 末尾仅要求
    "Output your answer as a single letter (e.g., A, B, C, D)."
    模型输出极短 (response_length=2)，导致 rollout.n=8 次采样完全一致，
    GRPO 组内方差为零 → advantage=0 → 无有效梯度 → KL 恒为 0。

    排序题 (sort) 也只输出 5 个数字，偶尔才有采样差异。

修复方案:
    1. 将所有选择题/排序题的 prompt 末尾替换为 CoT 格式指令:
       要求模型先在 <think> 标签内推理，再在 <answer> 标签内给出答案。
    2. 每种任务有定制化的 CoT 引导语，鼓励模型分析视频内容。
    3. temporal_seg 保持不变（其回复本身就够长）。

用法:
    python proxy_data/redesign_prompts.py \
        --input  proxy_data/mixed_train_clean.jsonl \
        --output proxy_data/mixed_train_cot.jsonl \
        [--dry-run]  # 只统计不写入
"""

import argparse
import json
import re
import sys
from pathlib import Path
from collections import Counter

# ============================================================
# 各任务新的 CoT 尾部指令 (替换原始 prompt 末尾的输出要求)
# ============================================================

# --- 选择题通用 CoT 模板 ---
# 选择题 (add/delete/replace) 共用格式，但引导语略有不同

COT_SUFFIX_ADD = """First, carefully observe the actions and visual content in each Context Video to understand the cooking progression. Then, watch each Option Video and reason about which one logically continues the sequence.

Think step by step inside <think> </think> tags, then provide your final answer (a single letter A, B, C, or D) inside <answer> </answer> tags."""

COT_SUFFIX_DELETE = """First, carefully observe the visual content and cooking actions in each video clip. Identify the overall cooking flow and reason about which clip does NOT fit the logical sequence.

Think step by step inside <think> </think> tags, then provide your final answer (a single letter A, B, C, or D) inside <answer> </answer> tags."""

COT_SUFFIX_REPLACE = """First, carefully observe the Context Sequence to understand the cooking flow before and after the [MISSING] step. Then, watch each Option Video and reason about which one best fills the gap.

Think step by step inside <think> </think> tags, then provide your final answer (a single letter A, B, C, or D) inside <answer> </answer> tags."""

COT_SUFFIX_SORT = """First, carefully observe the cooking actions in each video clip. Reason about the logical chronological order of the cooking process.

Think step by step inside <think> </think> tags, then provide your final answer (the correct order as a continuous sequence of video indices) inside <answer> </answer> tags."""


# ============================================================
# 旧 prompt 尾部模式匹配 (用于定位替换位置)
# ============================================================

# 选择题：匹配 "Output your answer as a single letter ..."
_OLD_CHOICE_TAIL = re.compile(
    r"Output your answer as a single letter.*$",
    re.IGNORECASE | re.DOTALL
)

# 排序题：匹配 "Output only the order as a continuous sequence ..."
_OLD_SORT_TAIL = re.compile(
    r"Output only the order.*$",
    re.IGNORECASE | re.DOTALL
)


# 任务 → (旧尾部 pattern, 新 CoT 后缀)
TASK_CONFIG = {
    "add":     (_OLD_CHOICE_TAIL, COT_SUFFIX_ADD),
    "delete":  (_OLD_CHOICE_TAIL, COT_SUFFIX_DELETE),
    "replace": (_OLD_CHOICE_TAIL, COT_SUFFIX_REPLACE),
    "sort":    (_OLD_SORT_TAIL,   COT_SUFFIX_SORT),
}


def rewrite_prompt(prompt_text: str, problem_type: str) -> str:
    """
    将 prompt 末尾的简单输出指令替换为 CoT 格式指令。
    
    Args:
        prompt_text: 原始 prompt 文本
        problem_type: 任务类型
    
    Returns:
        修改后的 prompt 文本
    """
    config = TASK_CONFIG.get(problem_type)
    if config is None:
        # temporal_seg 或未知类型，保持不变
        return prompt_text
    
    old_pattern, new_suffix = config
    
    # 尝试正则替换
    new_prompt, n_subs = old_pattern.subn(new_suffix, prompt_text)
    
    if n_subs == 0:
        # 没有匹配到旧模式 → 直接追加 CoT 指令
        new_prompt = prompt_text.rstrip() + "\n\n" + new_suffix
    
    return new_prompt


def rewrite_messages(messages: list, problem_type: str) -> list:
    """
    重写 messages 中 user 消息的 prompt 文本。
    支持 str 和 list[dict] 两种 content 格式。
    """
    new_messages = []
    for msg in messages:
        if msg["role"] != "user":
            new_messages.append(msg)
            continue
        
        content = msg["content"]
        
        if isinstance(content, str):
            new_content = rewrite_prompt(content, problem_type)
            new_messages.append({"role": "user", "content": new_content})
            
        elif isinstance(content, list):
            # list of {"type": "text/video/image", ...}
            new_content_list = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    new_text = rewrite_prompt(item["text"], problem_type)
                    new_content_list.append({"type": "text", "text": new_text})
                else:
                    new_content_list.append(item)
            new_messages.append({"role": "user", "content": new_content_list})
        else:
            new_messages.append(msg)
    
    return new_messages


def process_file(input_path: str, output_path: str, dry_run: bool = False):
    """处理整个 JSONL 文件。"""
    input_path = Path(input_path)
    
    stats = Counter()
    samples = []
    
    with open(input_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Line {line_no}: JSON parse error: {e}", file=sys.stderr)
                stats["error"] += 1
                continue
            
            problem_type = sample.get("problem_type", "")
            stats[f"total_{problem_type}"] += 1
            
            if problem_type in TASK_CONFIG:
                # 重写 messages
                if "messages" in sample:
                    sample["messages"] = rewrite_messages(
                        sample["messages"], problem_type
                    )
                
                # 重写 prompt (如果有单独的 prompt 字段)
                if "prompt" in sample and isinstance(sample["prompt"], str):
                    sample["prompt"] = rewrite_prompt(
                        sample["prompt"], problem_type
                    )
                
                stats[f"rewritten_{problem_type}"] += 1
            else:
                stats[f"kept_{problem_type}"] += 1
            
            samples.append(sample)
    
    # 打印统计
    print(f"\n{'='*60}")
    print(f"Prompt Redesign Statistics")
    print(f"{'='*60}")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Total samples: {len(samples)}")
    print()
    
    for key in sorted(stats):
        print(f"  {key}: {stats[key]}")
    print()
    
    if dry_run:
        print("[DRY RUN] No output file written.")
        print("\nSample rewrites:")
        for pt in ["add", "delete", "replace", "sort"]:
            for s in samples:
                if s.get("problem_type") == pt:
                    # 提取文本展示
                    msgs = s.get("messages", [])
                    for m in msgs:
                        if m["role"] == "user":
                            c = m["content"]
                            if isinstance(c, list):
                                text = " ".join(
                                    item["text"] for item in c
                                    if isinstance(item, dict) and item.get("type") == "text"
                                )
                            else:
                                text = c
                    # 只显示最后 500 字符
                    print(f"\n--- {pt} (last 500 chars) ---")
                    print(text[-500:])
                    break
        return
    
    # 写出
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    print(f"Written {len(samples)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="重新设计选择题/排序题的 Prompt，添加 CoT 推理指令"
    )
    parser.add_argument(
        "--input", "-i",
        default="proxy_data/mixed_train_clean.jsonl",
        help="输入的 JSONL 文件路径"
    )
    parser.add_argument(
        "--output", "-o",
        default="proxy_data/mixed_train_cot.jsonl",
        help="输出的 JSONL 文件路径"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="只统计和展示，不写入输出文件"
    )
    
    args = parser.parse_args()
    process_file(args.input, args.output, args.dry_run)


if __name__ == "__main__":
    main()
