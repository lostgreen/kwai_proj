#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EasyR1 Qwen3-VL 指令遵循诊断脚本
====================================
完全复用框架中 RLHFDataset + processor 的流水线,
从 JSONL 加载数据 → 构建 messages → apply_chat_template → 处理视频 → 模型推理,
逐步打印中间结果, 定位模型不遵循指令 / 输出格式错误的原因。

典型用法:
    # (1) 完整推理诊断 (需要 GPU)
    python test_diagnostic.py \
        --model_path /path/to/Qwen3-VL-4B-Instruct \
        --jsonl proxy_data/youcook2_train_easyr1.jsonl \
        --index 0 \
        --num_samples 8 \
        --max_new_tokens 256

    # (2) 对比: 训练参数 vs 低温 vs 加system prompt
    python test_diagnostic.py \
        --model_path /path/to/Qwen3-VL-4B-Instruct \
        --jsonl proxy_data/youcook2_train_easyr1.jsonl \
        --index 0 \
        --ablation

    # (3) 仅做 tokenize 分析, 不加载模型 (无需 GPU)
    python test_diagnostic.py \
        --model_path /path/to/Qwen3-VL-4B-Instruct \
        --jsonl proxy_data/youcook2_train_easyr1.jsonl \
        --cpu

    # (4) 离线分析已有的模型输出 (无需 GPU 和模型)
    python test_diagnostic.py --offline_analyze
"""

import sys
import os
import json
import argparse
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

# ====================================================
# 0. 直接引用框架中的数据处理函数
# ====================================================
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from verl.utils.dataset import (
    RLHFDataset,
    process_video,
    process_image,
    collate_fn,
    QUESTION_TEMPLATE,
    TYPE_TEMPLATE,
)
from verl.reward_function.youcook2_temporal_seg_reward import (
    compute_score as youcook2_reward,
    parse_segments,
    has_events_tag,
    SEGMENT_PATTERN,
    EVENTS_PATTERN,
)


def sep(title: str = "", char: str = "=", width: int = 80):
    if title:
        print(f"\n{char * 3} {title} {char * (width - len(title) - 5)}")
    else:
        print(char * width)


# ====================================================
# 1. 从 JSONL 读取原始数据
# ====================================================
def load_sample(jsonl_path: str, index: int) -> dict:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert index < len(lines), f"index {index} out of range [0, {len(lines)-1}]"
    return json.loads(lines[index])


# ====================================================
# 2. 复用 RLHFDataset._build_messages 逻辑
# ====================================================
def build_messages_like_framework(example: dict, video_key="videos", image_key="images") -> list:
    """完全复现 verl/utils/dataset.py → RLHFDataset._build_messages"""
    if "messages" in example and isinstance(example["messages"], list) and len(example["messages"]) > 0:
        prompt_str = ""
        for msg in example["messages"]:
            if msg.get("role") == "user":
                prompt_str = msg.get("content", "")

        if video_key in example and isinstance(example.get(video_key), list) and len(example.get(video_key)) > 0:
            content_list = []
            for i, content in enumerate(prompt_str.split("<video>")):
                if i != 0:
                    content_list.append({"type": "video"})
                if content:
                    content_list.append({"type": "text", "text": content})
            return [{"role": "user", "content": content_list}]
        elif image_key in example and isinstance(example.get(image_key), list) and len(example.get(image_key)) > 0:
            content_list = []
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image"})
                if content:
                    content_list.append({"type": "text", "text": content})
            return [{"role": "user", "content": content_list}]
        else:
            return [{"role": "user", "content": prompt_str}]
    else:
        raise ValueError("数据中没有 'messages' 字段, 请检查 JSONL 格式")


# ====================================================
# 3. 诊断 chat_template 的最终文本
# ====================================================
def diagnose_chat_template(processor, messages: list, label: str = ""):
    prompt_text = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    sep(f"apply_chat_template 输出 {label}")

    # 只显示视觉 token 外的文本(视觉 token 太多)
    # 把连续的 <|video_pad|> 折叠显示
    display = prompt_text
    display = re.sub(r'(<\|video_pad\|>)+', lambda m: f'<|video_pad|>×{m.group(0).count("<|video_pad|>")}', display)
    display = re.sub(r'(<\|image_pad\|>)+', lambda m: f'<|image_pad|>×{m.group(0).count("<|image_pad|>")}', display)
    print(display[:3000])
    if len(display) > 3000:
        print(f"... (总长 {len(prompt_text)} 字符)")
    sep()

    issues = []
    if "<|im_start|>system" not in prompt_text:
        issues.append("🔴 没有 system prompt! Qwen3 chat template 没有自动注入 system message。")
        issues.append("    → 训练时模型没有 'You are a helpful assistant.' 之类的角色定义。")
    else:
        # 检查 system prompt 内容
        sys_match = re.search(r'<\|im_start\|>system\n(.*?)<\|im_end\|>', prompt_text, re.DOTALL)
        if sys_match:
            issues.append(f"ℹ️  System prompt 内容: '{sys_match.group(1).strip()}'")

    if "<|im_start|>assistant" not in prompt_text:
        issues.append("🔴 没有 assistant 生成提示头部! 模型不知道从哪里开始回答。")
    else:
        # 检查 assistant 之后有什么
        after_assistant = prompt_text.split("<|im_start|>assistant")[-1]
        if after_assistant.strip() == "\n":
            issues.append("✅ assistant 头部正常, 后面是换行符(等待模型生成)")
        elif "<think>" in after_assistant and "</think>" not in after_assistant:
            issues.append("ℹ️  检测到 assistant 后有 <think> 但没 </think> — 可能是 forced thinking prefix")
        else:
            issues.append(f"ℹ️  assistant 后内容: '{after_assistant[:100]}'")

    if prompt_text.strip().endswith("<|im_end|>"):
        issues.append("🔴 prompt 以 <|im_end|> 结尾, 模型不会继续生成!")

    if "<|endoftext|>" in prompt_text:
        issues.append("🔴 prompt 中出现了 <|endoftext|>, 模型会提前停止!")

    # 检查视觉 token 数量
    if "<|vision_start|>" in prompt_text:
        vision_count = prompt_text.count("<|vision_start|>")
        video_pad_count = prompt_text.count("<|video_pad|>")
        issues.append(f"ℹ️  视频帧 block 数: {vision_count}, video_pad token 总数: {video_pad_count}")

    for issue in issues:
        print(issue)

    return prompt_text


# ====================================================
# 4. tokenize 分析
# ====================================================
def diagnose_tokenization(processor, tokenizer, prompt_text: str, processed_video, video_metadata):
    sep("Tokenization 分析")

    if processed_video is not None:
        model_inputs = processor(
            text=[prompt_text],
            videos=[processed_video],
            video_metadata=[video_metadata] if video_metadata else None,
            add_special_tokens=False,
            return_tensors="pt",
            do_resize=False,
            do_sample_frames=False,
        )
    else:
        model_inputs = tokenizer([prompt_text], add_special_tokens=False, return_tensors="pt")

    input_ids = model_inputs["input_ids"][0]
    print(f"总 input_ids 长度: {len(input_ids)}")

    # 检查特殊 token 分布
    vocab = tokenizer.get_vocab()
    special_tokens_to_check = ["<|im_start|>", "<|im_end|>", "<|endoftext|>",
                                "<|vision_start|>", "<|vision_end|>",
                                "<|video_pad|>", "<|image_pad|>"]
    for name in special_tokens_to_check:
        if name in vocab:
            tid = vocab[name]
            count = (input_ids == tid).sum().item()
            if count > 0:
                print(f"  {name} (id={tid}): {count} 次")

    # prompt 结尾
    tail_ids = input_ids[-30:]
    tail_text = tokenizer.decode(tail_ids, skip_special_tokens=False)
    print(f"\n  Prompt 最后 30 token:\n  {tail_text}")

    # video_grid_thw
    if "video_grid_thw" in model_inputs:
        vgt = model_inputs["video_grid_thw"]
        print(f"\n  video_grid_thw shape: {vgt.shape}")
        total_visual = 0
        for i in range(vgt.shape[0]):
            t, h, w = vgt[i].tolist()
            n = t * h * w
            total_visual += n
            if i < 5 or i == vgt.shape[0] - 1:
                print(f"    [{i}] t={t}, h={h}, w={w} → {n} tokens")
            elif i == 5:
                print(f"    ... (共 {vgt.shape[0]} 个视频块)")
        print(f"  总视觉 token: {total_visual}")
        text_tokens = len(input_ids) - total_visual
        print(f"  文本 token: {text_tokens}")
        print(f"  视觉占比: {total_visual/len(input_ids)*100:.1f}%")

    return model_inputs


# ====================================================
# 5. 完整推理并分析输出
# ====================================================
def run_inference(model, processor, tokenizer, model_inputs,
                  max_new_tokens=256, n_samples=5, temperature=1.0,
                  label="", quiet=False):
    import torch
    if not quiet:
        sep(f"推理 {label} (T={temperature}, max_tokens={max_new_tokens}, n={n_samples})")

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in model_inputs.items() if hasattr(v, 'to')}

    all_outputs = []
    all_outputs_raw = []
    all_gen_lens = []

    for i in range(n_samples):
        with torch.no_grad():
            gen_kwargs = dict(
                **inputs,
                max_new_tokens=max_new_tokens,
            )
            if temperature > 0:
                gen_kwargs.update(temperature=temperature, do_sample=True, top_p=1.0, top_k=-1)
            else:
                gen_kwargs.update(do_sample=False)

            generated_ids = model.generate(**gen_kwargs)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        output_text_raw = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]

        gen_len = len(generated_ids_trimmed[0])
        all_outputs.append(output_text)
        all_outputs_raw.append(output_text_raw)
        all_gen_lens.append(gen_len)

        if not quiet:
            print(f"\n{'─'*60}")
            print(f"生成 #{i+1} | {gen_len} tokens {'⚠️ 达到上限!' if gen_len >= max_new_tokens else ''}")
            print(f"{'─'*60}")
            print(output_text[:600])
            if len(output_text) > 600:
                print(f"... (总长 {len(output_text)})")

    return all_outputs, all_outputs_raw, all_gen_lens


# ====================================================
# 6. 输出深度分析
# ====================================================
def deep_analyze_outputs(outputs: list, outputs_raw: list, gen_lens: list,
                         ground_truth: str, max_new_tokens: int = 256):
    sep("输出深度分析")
    gt_segs = parse_segments(ground_truth)
    print(f"Ground Truth 时间段: {gt_segs}")
    if gt_segs:
        max_time = max(s[1] for s in gt_segs)
        print(f"GT 最大时间: {max_time}")

    # ---- 逐条分析 ----
    reward_inputs = []
    for i, (out, out_raw, gl) in enumerate(zip(outputs, outputs_raw, gen_lens)):
        print(f"\n{'─'*40} 输出 #{i+1} {'─'*40}")

        # (a) thinking mode 分析
        has_think = "<think>" in out
        has_think_close = "</think>" in out
        if has_think and has_think_close:
            think_match = re.search(r"<think>(.*?)</think>", out, re.DOTALL)
            think_len = len(think_match.group(1)) if think_match else 0
            after_think = out.split("</think>")[-1].strip()
            print(f"  [Think] ✅ 完整 think 块 ({think_len} 字符)")
            print(f"  [Think] 后续回答: '{after_think[:200]}'")
            analysis_text = after_think  # 只分析 think 之后的内容
        elif has_think and not has_think_close:
            print(f"  [Think] 🔴 未闭合! 模型一直在 thinking 直到 max_tokens 截断!")
            print(f"         这是不输出 <events> 的直接原因之一!")
            analysis_text = out
        else:
            print(f"  [Think] 无 thinking")
            analysis_text = out

        # (b) 格式分析
        has_events = has_events_tag(analysis_text)
        pred_segs = parse_segments(analysis_text)
        print(f"  [Format] 有 <events> 标签: {has_events}")
        print(f"  [Format] 正则解析出的合法段: {pred_segs}")

        # (c) 检查是否有"接近正确但格式错误"的段
        # 例如 [0-39] 用了减号而非逗号
        dash_pattern = re.compile(r'\[\s*(\d+)\s*-\s*(\d+)\s*\]')
        dash_matches = dash_pattern.findall(analysis_text)
        if dash_matches:
            print(f"  [Format] 🔴 发现用减号分隔的段(不被 reward regex 识别): {dash_matches}")
            print(f"           模型输出 [start-end] 而不是 [start, end]!")
            print(f"           → reward 解析不到任何合法段 → accuracy=0")

        # 检查超范围的值
        all_numbers = re.findall(r'\b(\d+)\b', analysis_text)
        large_nums = [int(n) for n in all_numbers if int(n) > 200]
        if large_nums:
            print(f"  [Range] 🔴 出现超大数值: {large_nums[:10]} (视频只有几十到一百多秒)")

        # (d) 退化重复检测
        repeated_tags = analysis_text.count("</events>")
        if repeated_tags > 2:
            print(f"  [Degenerate] 🔴 </events> 重复了 {repeated_tags} 次! 模型进入了重复循环!")

        unexpected_tags = re.findall(r'</(?!events|think)[a-z]+>', analysis_text)
        if unexpected_tags:
            tag_counts = Counter(unexpected_tags)
            print(f"  [Degenerate] 🔴 出现无关闭合标签: {dict(tag_counts)}")

        # (e) 截断检测
        if gl >= max_new_tokens:
            print(f"  [Truncate] 🔴 达到 max_new_tokens={max_new_tokens}, 输出被截断!")

        # (f) 空输出
        if not analysis_text.strip():
            print(f"  [Empty] 🔴 有效输出为空!")

        reward_inputs.append({
            "response": out,
            "ground_truth": ground_truth,
            "data_type": "video",
            "problem_type": "",
        })

    # ---- Reward 计算 ----
    sep("Reward 分析")
    scores = youcook2_reward(reward_inputs)
    all_overall = []
    all_accuracy = []
    all_format = []

    for i, (out, score) in enumerate(zip(outputs, scores)):
        all_overall.append(score["overall"])
        all_accuracy.append(score["accuracy"])
        all_format.append(score["format"])
        print(f"  #{i+1}: overall={score['overall']:.4f}  format={score['format']:.4f}  accuracy={score['accuracy']:.4f}")

    print(f"\n  === GRPO 优势估计分析 (模拟 n={len(outputs)} 候选) ===")
    import numpy as np
    scores_arr = np.array(all_overall)
    mean_r = scores_arr.mean()
    std_r = scores_arr.std()
    print(f"  mean(reward) = {mean_r:.4f}")
    print(f"  std(reward)  = {std_r:.6f}")

    if std_r < 1e-6:
        print(f"  🔴 所有候选的 reward 完全相同! GRPO advantage = 0, 没有任何梯度信号!")
        print(f"     这是训练无法收敛的根本原因: reward 没有方差 → 无法区分好坏 → 无法学习")
    elif std_r < 0.01:
        print(f"  ⚠️  reward 方差极小, GRPO 的优势估计信号很弱!")

    fmt_rate = sum(1 for s in all_format if s > 0) / len(all_format)
    acc_rate = sum(1 for s in all_accuracy if s > 0) / len(all_accuracy)
    print(f"\n  格式正确率: {fmt_rate:.1%}")
    print(f"  有效 accuracy > 0 比率: {acc_rate:.1%}")

    return scores


# ====================================================
# 7. 离线分析 (不需要模型/GPU)
# ====================================================
def offline_analyze():
    """直接分析已有的模型输出, 诊断问题"""
    sep("离线输出分析")
    print("粘贴模型输出 (输入空行结束):")
    lines = []
    while True:
        try:
            line = input()
            if line == "":
                break
            lines.append(line)
        except EOFError:
            break

    output = "\n".join(lines)
    if not output.strip():
        # 使用用户给的示例
        output = """<events>
[0-39]
[56-199]
[135-259]
</events>
[< 0-19]
[< 255-359]
</events>
</events>
</events>
</events>
</author>
</events>"""

    print(f"\n分析输出 (长度={len(output)}):")
    print(output[:500])

    print(f"\n{'─'*60}")
    # 用 reward 正则分析
    has_fmt = has_events_tag(output)
    pred_segs = parse_segments(output)
    print(f"有 <events> 标签: {has_fmt}")
    print(f"正则解析出的合法段 [start, end]: {pred_segs}")

    # 检查减号分隔
    dash_pattern = re.compile(r'\[\s*(\d+)\s*-\s*(\d+)\s*\]')
    dash_matches = dash_pattern.findall(output)
    if dash_matches:
        print(f"\n🔴 发现减号分隔的段 (reward 不识别): {dash_matches}")
        print(f"   模型输出 [0-39] 但 reward 要求 [0, 39] (逗号分隔)")
        # 如果这些格式能被识别, reward 会是多少?
        print(f"\n   假设这些段是合法的, 它们的值为:")
        for s, e in dash_matches:
            print(f"     [{s}, {e}]")

    # 检查重复
    repeated = output.count("</events>")
    if repeated > 1:
        print(f"\n🔴 </events> 出现 {repeated} 次 — 退化重复!")

    unexpected_tags = re.findall(r'</(?!events|think)[a-z]+>', output)
    if unexpected_tags:
        print(f"🔴 出现无关标签: {Counter(unexpected_tags)}")

    print()


# ====================================================
# 8. 消融实验: 对比不同配置
# ====================================================
def run_ablation(model, processor, tokenizer, model_inputs,
                 messages, processed_video, video_metadata,
                 ground_truth, max_new_tokens, num_samples):
    sep("消融实验: 对比不同配置对输出质量的影响")

    configs = [
        ("A: 训练参数 (T=1.0, 无system)", {"temperature": 1.0, "messages": messages}),
        ("B: 低温 (T=0, greedy, 无system)", {"temperature": 0, "messages": messages}),
        ("C: 中温 (T=0.6, 无system)", {"temperature": 0.6, "messages": messages}),
        ("D: 训练参数+system prompt (T=1.0)", {
            "temperature": 1.0,
            "messages": [{"role": "system", "content": "You are a helpful assistant. Strictly follow the user's output format requirements. Do not use <think> tags. Respond directly."}] + messages,
        }),
        ("E: 低温+system prompt (T=0)", {
            "temperature": 0,
            "messages": [{"role": "system", "content": "You are a helpful assistant. Strictly follow the user's output format requirements. Do not use <think> tags. Respond directly."}] + messages,
        }),
        ("F: system+强制格式 (T=0.6)", {
            "temperature": 0.6,
            "messages": [{"role": "system", "content": "You are a video analysis assistant. You MUST output temporal segments in this EXACT format:\n<events>\n[start, end]\n</events>\nUse comma to separate start and end. Do NOT use dashes. Do NOT repeat tags."}] + messages,
        }),
    ]

    results = {}
    for name, cfg in configs:
        sep(f"配置: {name}")
        msgs = cfg["messages"]
        temp = cfg["temperature"]

        prompt = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

        if processed_video is not None:
            inputs = processor(
                text=[prompt], videos=[processed_video],
                video_metadata=[video_metadata] if video_metadata else None,
                add_special_tokens=False, return_tensors="pt",
                do_resize=False, do_sample_frames=False,
            )
        else:
            inputs = tokenizer([prompt], add_special_tokens=False, return_tensors="pt")

        n = 1 if temp == 0 else num_samples
        outs, outs_raw, lens = run_inference(
            model, processor, tokenizer, inputs,
            max_new_tokens=max_new_tokens, n_samples=n,
            temperature=temp, label=name, quiet=False,
        )

        scores = deep_analyze_outputs(outs, outs_raw, lens, ground_truth, max_new_tokens)
        results[name] = {
            "outputs": outs,
            "scores": scores,
            "avg_overall": sum(s["overall"] for s in scores) / len(scores),
            "avg_accuracy": sum(s["accuracy"] for s in scores) / len(scores),
            "fmt_rate": sum(1 for s in scores if s["format"] > 0) / len(scores),
        }

    # ---- 汇总对比 ----
    sep("消融实验汇总")
    print(f"{'配置':<45} {'avg_overall':>12} {'avg_accuracy':>14} {'格式正确率':>10}")
    print("─" * 85)
    for name, r in results.items():
        print(f"{name:<45} {r['avg_overall']:>12.4f} {r['avg_accuracy']:>14.4f} {r['fmt_rate']:>10.1%}")


# ====================================================
# 主流程
# ====================================================
def main():
    parser = argparse.ArgumentParser(description="EasyR1 Qwen3-VL 指令遵循诊断")
    parser.add_argument("--model_path", type=str, default="", help="Qwen3-VL 模型路径")
    parser.add_argument("--jsonl", type=str, default="proxy_data/youcook2_train_easyr1.jsonl", help="JSONL 数据路径")
    parser.add_argument("--index", type=int, default=0, help="测试第几条数据")
    parser.add_argument("--num_samples", type=int, default=8, help="每条数据生成几个回复(模拟 GRPO n=8)")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="最大生成 token 数(训练=256)")
    parser.add_argument("--temperature", type=float, default=1.0, help="采样温度(训练时=1.0)")
    parser.add_argument("--video_fps", type=float, default=2.0, help="视频采样帧率")
    parser.add_argument("--max_pixels", type=int, default=12288, help="每帧最大像素数")
    parser.add_argument("--min_pixels", type=int, default=3136, help="每帧最小像素数")
    parser.add_argument("--max_frames", type=int, default=256, help="最大抽取帧数")
    parser.add_argument("--cpu", action="store_true", help="仅做 tokenize 分析, 不推理")
    parser.add_argument("--ablation", action="store_true", help="运行消融实验(对比不同配置)")
    parser.add_argument("--offline_analyze", action="store_true", help="离线分析已有输出")
    args = parser.parse_args()

    # ---- 离线分析模式 ----
    if args.offline_analyze:
        offline_analyze()
        return

    assert args.model_path, "请指定 --model_path"

    # ==================================================================
    # 先打印配置对比警告
    # ==================================================================
    sep("⚠️  配置一致性检查")
    config_path = os.path.join(os.path.dirname(__file__),
                               "checkpoints/qwen3_vl_youcook2_temporal_seg/experiment_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            exp_cfg = json.load(f)
        stored_max_pixels = exp_cfg.get("data", {}).get("max_pixels")
        stored_min_pixels = exp_cfg.get("data", {}).get("min_pixels")
        print(f"  experiment_config.json 中 max_pixels = {stored_max_pixels}")
        print(f"  experiment_config.json 中 min_pixels = {stored_min_pixels}")
        print(f"  训练脚本中 MAX_PIXELS = 12288")
        print(f"  YAML 基础配置 max_pixels = 1048576")
        if stored_max_pixels and stored_max_pixels != 12288:
            print(f"  🔴 实际训练用了 max_pixels={stored_max_pixels}, 不是脚本里写的 12288!")
            print(f"     YAML 的 max_pixels=1048576 覆盖了命令行的 12288!")
            print(f"     max_pixels=1048576 意味着每帧可能有几千个视觉 token, 极大地稀释了文本指令!")
            args.max_pixels = stored_max_pixels  # 用实际值来测试

    # ---- 加载 processor & tokenizer ----
    sep("加载 Processor & Tokenizer")
    from transformers import AutoProcessor, AutoTokenizer
    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    print(f"Processor: {processor.__class__.__name__}")
    print(f"Tokenizer: {tokenizer.__class__.__name__}")
    print(f"EOS: '{tokenizer.eos_token}' (id={tokenizer.eos_token_id})")
    print(f"PAD: '{tokenizer.pad_token}' (id={tokenizer.pad_token_id})")

    # 打印 chat template
    sep("Chat Template (Jinja2)")
    tmpl = getattr(processor, 'chat_template', None) or getattr(tokenizer, 'chat_template', None)
    if tmpl:
        print(tmpl[:2000])
        # 检查模板是否会自动注入 system
        if "system" in tmpl.lower():
            print("\nℹ️  模板中包含 'system' 相关逻辑")
        if "think" in tmpl.lower():
            print("ℹ️  模板中包含 'think' 相关逻辑 (Qwen3 thinking mode)")
    else:
        print("⚠️  没有找到 chat_template")

    # ---- 读取数据 ----
    sep(f"读取数据: {args.jsonl}, index={args.index}")
    sample = load_sample(args.jsonl, args.index)
    print(f"字段: {list(sample.keys())}")
    ground_truth = sample.get("answer", "")
    print(f"Ground Truth: {ground_truth}")
    gt_segs = parse_segments(ground_truth)
    print(f"GT 时间段: {gt_segs}")

    clip_duration = sample.get("metadata", {}).get("clip_duration", "?")
    print(f"视频时长: {clip_duration}s")

    # ---- Step 1: _build_messages ----
    sep("Step 1: _build_messages")
    messages = build_messages_like_framework(sample)
    roles = [m["role"] for m in messages]
    print(f"消息角色列表: {roles}")
    if "system" not in roles:
        print("🔴 没有 system 角色! 框架的 _build_messages 只创建了 user message。")

    # 显示纯文本部分
    for msg in messages:
        if msg["role"] == "user":
            if isinstance(msg["content"], list):
                for part in msg["content"]:
                    if part["type"] == "text":
                        print(f"\n  [text] {part['text'][:300]}...")
                    else:
                        print(f"  [{part['type']}]")
            else:
                print(f"\n  {msg['content'][:300]}...")

    # ---- Step 2: chat template ----
    sep("Step 2: apply_chat_template")
    prompt_text = diagnose_chat_template(processor, messages)

    # ---- Step 3: 处理视频 ----
    processed_video = None
    video_metadata = None
    videos = sample.get("videos", [])
    if videos:
        sep("Step 3: 视频处理")
        video_path = videos[0]
        print(f"路径: {video_path}")
        if not os.path.exists(video_path):
            print(f"⚠️  文件不存在, 跳过视频处理")
        else:
            try:
                result = process_video(
                    video_path, min_pixels=args.min_pixels, max_pixels=args.max_pixels,
                    max_frames=args.max_frames, video_fps=args.video_fps, return_fps=True,
                )
                processed_video_raw, video_fps = result
                if isinstance(processed_video_raw, tuple):
                    processed_video, video_metadata = processed_video_raw
                else:
                    processed_video = processed_video_raw

                print(f"帧 tensor shape: {processed_video.shape}")
                print(f"采样帧率: {video_fps}")
                if video_metadata:
                    print(f"metadata: {video_metadata}")
                n_frames = processed_video.shape[0]
                print(f"帧数: {n_frames}")
                print(f"max_pixels={args.max_pixels} → 每帧约 {args.max_pixels // (16*16)} 个视觉 token")
                print(f"预估总视觉 token: ~{n_frames * args.max_pixels // (16*16)}")
            except Exception as e:
                print(f"视频处理失败: {e}")
                import traceback; traceback.print_exc()

    # ---- Step 4: tokenize ----
    model_inputs = diagnose_tokenization(processor, tokenizer, prompt_text, processed_video, video_metadata)

    # ---- Step 5: 模型推理 ----
    if not args.cpu:
        import torch
        sep("Step 5: 加载模型")
        print(f"路径: {args.model_path}")
        try:
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(
                args.model_path, torch_dtype=torch.bfloat16,
                device_map="auto", attn_implementation="flash_attention_2",
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"flash_attention_2 失败, 用 eager: {e}")
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(
                args.model_path, torch_dtype=torch.bfloat16,
                device_map="auto", trust_remote_code=True,
            )
        model.eval()
        print(f"模型: {model.__class__.__name__}")
        n_params = sum(p.numel() for p in model.parameters()) / 1e9
        print(f"参数: {n_params:.2f}B")

        if args.ablation:
            # ---- 消融实验 ----
            run_ablation(
                model, processor, tokenizer, model_inputs,
                messages, processed_video, video_metadata,
                ground_truth, args.max_new_tokens, args.num_samples,
            )
        else:
            # ---- 单配置推理 ----
            outputs, outputs_raw, gen_lens = run_inference(
                model, processor, tokenizer, model_inputs,
                max_new_tokens=args.max_new_tokens, n_samples=args.num_samples,
                temperature=args.temperature, label="训练参数",
            )
            deep_analyze_outputs(outputs, outputs_raw, gen_lens, ground_truth, args.max_new_tokens)
    else:
        print("\n(--cpu 模式: 跳过推理)")

    # ====================================================
    sep("诊断总结")
    print("""
根据框架代码和训练配置分析, 指令遵循失败的可能原因 (按优先级):

🔴 1. [Reward沙漠] 基座模型不认识 <events> 格式
   - Qwen3-VL-4B-Instruct 从未见过 <events>[start, end]</events> 格式
   - 模型猜测出 [0-39] (减号) 而非 [0, 39] (逗号)
   - reward regex 要求逗号分隔 → 无法解析 → accuracy=0
   - GRPO 的 n=8 候选全部 reward=0.05 → advantage=0 → 无梯度 → 无法学习
   ✅ 修复: 在 reward 中增加宽容解析, 或用 SFT 冷启动让模型先学会格式

🔴 2. [max_pixels 配置没生效] YAML覆盖了命令行
   - 脚本写了 MAX_PIXELS=12288, 但 YAML 基础配置 max_pixels=1048576
   - experiment_config.json 记录的也是 1048576, 说明实际用了这个值
   - 1048576 像素/帧 → 每帧可能有上千视觉 token → 文本指令被稀释

🔴 3. [退化重复] 模型陷入 </events> 无限循环
   - 输出 </events> 后不知道何时停止
   - max_new_tokens=256 全部浪费在重复标签上
   - 根因: 模型对 <events>...</events> 结构没有先验知识

⚠️  4. [无 System Prompt] _build_messages 不添加 system 角色
   - 模型没有明确的行为指引, 更容易输出不稳定

⚠️  5. [温度过高] temperature=1.0 对格式任务不利
   - 高温让采样更随机, 格式遵循更难

推荐修复方案 (优先级从高到低):

  (1) SFT 冷启动: 用少量正确格式的样本 (100-500条) 对基座做 1-2 epoch SFT
      让模型先学会 <events>[start, end]</events> 格式, 再进 GRPO
      
  (2) 宽容 Reward: 修改 youcook2_temporal_seg_reward.py 的 SEGMENT_PATTERN
      让它也能识别 [start-end] / [start;end] 等格式, 给部分 reward

  (3) 修复 max_pixels: 确保 data.max_pixels 正确生效
      检查 YAML config 的优先级, 或直接在 YAML 中修改

  (4) 添加 System Prompt: 在 _build_messages 中为 Qwen3 添加 system message

  (5) 降温 / 增加 max_response_length: T=0.6, max_tokens=512
""")


if __name__ == "__main__":
    main()
