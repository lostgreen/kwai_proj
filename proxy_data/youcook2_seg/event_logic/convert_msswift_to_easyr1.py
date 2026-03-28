#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ms-swift YouCook2 训练数据 → EasyR1 格式转换脚本

ms-swift 格式 (youcook2_train.jsonl):
{
    "videos": ["/path/to/video.mp4"],
    "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "...\\n<video>\\n..."}
    ],
    "solution": "<events>\\n[0, 12]\\n[24, 37]\\n</events>",
    "metadata": { ... }
}

EasyR1 格式 (输出 JSONL):
{
    "messages": [
        {"role": "user", "content": "...\\n<video>\\n..."}
    ],
    "prompt": "...(user content 的副本，fallback 用)...",
    "answer": "<events>\\n[0, 12]\\n[24, 37]\\n</events>",
    "videos": ["/path/to/video.mp4"],
    "data_type": "video",
    "problem_type": ""
}

说明:
    - 保留 ms-swift 的 messages 字段（仅 user 消息），触发 dataset.py 的直接使用路径
    - 跳过 EasyR1 的 QUESTION_TEMPLATE 包装，不追加 <think>/<answer> 提示词
    - solution → answer

python /home/xuboshen/zgw/OneThinker/EasyR1/proxy_data/convert_msswift_to_easyr1.py \
    -i /home/xuboshen/zgw/OneThinker/EasyR1/proxy_data/youcook2_train_clean.jsonl \
    -o /home/xuboshen/zgw/OneThinker/EasyR1/proxy_data/youcook2_train_easyr1.jsonl \
    --stats
"""

import json
import argparse
import os


def extract_user_message(messages: list) -> dict:
    """提取最后一个 user 消息"""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg
    return {"role": "user", "content": ""}


def convert_sample(sample: dict, video_dir_replace: str = None) -> dict:
    """将单个 ms-swift 样本转换为 EasyR1 格式"""
    # 视频路径
    videos = sample.get("videos", [])
    if video_dir_replace and videos:
        old_prefix, new_prefix = video_dir_replace.split(":", 1)
        videos = [v.replace(old_prefix, new_prefix) for v in videos]

    # 提取 user 消息（去掉 system message）
    user_msg = extract_user_message(sample.get("messages", []))
    user_content = user_msg.get("content", "")

    # 答案
    answer = sample.get("solution", "")

    # metadata
    metadata = sample.get("metadata", {})

    result = {
        # messages 字段触发 dataset.py 直接使用路径（跳过模板包装）
        "messages": [{"role": "user", "content": user_content}],
        # prompt 作为 fallback（answer_key 仍需要）
        "prompt": user_content,
        "answer": answer,
        "videos": videos,
        "data_type": "video",
        "problem_type": "",
    }

    if metadata:
        result["metadata"] = metadata

    return result


def main():
    parser = argparse.ArgumentParser(
        description="将 ms-swift YouCook2 训练数据转换为 EasyR1 格式"
    )
    parser.add_argument("--input", "-i", required=True, help="输入 JSONL 文件")
    parser.add_argument("--output", "-o", required=True, help="输出 JSONL 文件")
    parser.add_argument(
        "--video_dir_replace", default=None,
        help="替换视频路径前缀, 格式: 'old_prefix:new_prefix'"
    )
    parser.add_argument("--max_samples", type=int, default=None, help="最多转换样本数")
    parser.add_argument("--stats", action="store_true", help="打印统计信息")
    args = parser.parse_args()

    total = 0
    num_events_dist = {}
    clip_durations = []
    video_ids = set()

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:

        for line_no, line in enumerate(fin, 1):
            if args.max_samples and total >= args.max_samples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] 第 {line_no} 行 JSON 解析失败: {e}")
                continue

            converted = convert_sample(sample, args.video_dir_replace)
            fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
            total += 1

            if args.stats:
                meta = sample.get("metadata", {})
                ne = meta.get("num_events", 0)
                num_events_dist[ne] = num_events_dist.get(ne, 0) + 1
                if "clip_duration" in meta:
                    clip_durations.append(meta["clip_duration"])
                if "video_id" in meta:
                    video_ids.add(meta["video_id"])

    print(f"✅ 转换完成: {total} 个样本 → {args.output}")

    if args.stats and total > 0:
        print(f"\n📊 数据统计:")
        print(f"  总样本数: {total}")
        print(f"  独立视频数: {len(video_ids)}")
        if clip_durations:
            print(f"  片段时长: min={min(clip_durations):.0f}s, "
                  f"max={max(clip_durations):.0f}s, "
                  f"avg={sum(clip_durations)/len(clip_durations):.1f}s")
        print(f"  事件数分布:")
        for k in sorted(num_events_dist.keys()):
            print(f"    {k} 个事件: {num_events_dist[k]} 样本")


if __name__ == "__main__":
    main()
