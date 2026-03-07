#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证数据集中所有视频文件是否可读。
输出不可读/损坏的视频列表，并可选过滤掉这些样本生成干净数据集。

用法:
    # 仅检查
    python youcook_proxy/validate_videos.py -i youcook_proxy/youcook2_train.jsonl

    # 检查并输出干净数据集（过滤掉不可读的样本）
    python /home/xuboshen/zgw/OneThinker/EasyR1/proxy_data/validate_videos.py \
        -i /home/xuboshen/zgw/OneThinker/EasyR1/proxy_data/youcook2_train.jsonl \
        -o /home/xuboshen/zgw/OneThinker/EasyR1/proxy_data/youcook2_train_clean.jsonl
"""

import json
import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict


def check_video_readable(video_path: str, timeout: int = 10) -> dict:
    """
    用 ffprobe 检查视频是否可读。
    返回 {"path": ..., "ok": bool, "error": str}
    """
    result = {"path": video_path, "ok": False, "error": ""}

    if not os.path.exists(video_path):
        result["error"] = "文件不存在"
        return result

    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_type,duration,nb_frames",
                "-of", "json",
                video_path
            ],
            capture_output=True, text=True, timeout=timeout
        )

        if proc.returncode != 0:
            result["error"] = proc.stderr.strip()[:200]
            return result

        # 检查是否有视频流
        info = json.loads(proc.stdout)
        streams = info.get("streams", [])
        if not streams:
            result["error"] = "无视频流"
            return result

        result["ok"] = True
        return result

    except subprocess.TimeoutExpired:
        result["error"] = f"超时 ({timeout}s)"
        return result
    except FileNotFoundError:
        result["error"] = "ffprobe 未安装"
        return result
    except Exception as e:
        result["error"] = str(e)[:200]
        return result


def main():
    parser = argparse.ArgumentParser(description="验证数据集中视频文件可读性")
    parser.add_argument("--input", "-i", required=True, help="输入 JSONL 文件")
    parser.add_argument("--output", "-o", default=None, help="输出干净 JSONL（过滤不可读视频）")
    parser.add_argument("--video_key", default="videos", help="视频路径字段名")
    parser.add_argument("--workers", "-w", type=int, default=16, help="并行检查线程数")
    parser.add_argument("--timeout", type=int, default=10, help="ffprobe 超时秒数")
    parser.add_argument("--bad_list", default=None, help="输出不可读视频路径列表文件")
    args = parser.parse_args()

    # 1. 收集所有唯一视频路径 + 记录每行的视频
    print("📂 读取数据集...")
    lines = []
    all_videos = set()
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                videos = sample.get(args.video_key, [])
                if isinstance(videos, list):
                    all_videos.update(videos)
                lines.append((sample, line))
            except json.JSONDecodeError:
                lines.append((None, line))

    print(f"  总样本数: {len(lines)}")
    print(f"  唯一视频数: {len(all_videos)}")

    # 2. 并行检查视频
    print(f"\n🔍 验证视频可读性 (workers={args.workers})...")
    bad_videos = set()
    checked = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(check_video_readable, v, args.timeout): v
            for v in all_videos
        }
        for future in as_completed(futures):
            result = future.result()
            checked += 1
            if not result["ok"]:
                bad_videos.add(result["path"])
                print(f"  ❌ [{checked}/{len(all_videos)}] {result['path']}")
                print(f"     原因: {result['error']}")
            else:
                if checked % 100 == 0 or checked == len(all_videos):
                    print(f"  ✅ [{checked}/{len(all_videos)}] 已检查...")

    # 3. 汇总
    good_count = len(all_videos) - len(bad_videos)
    print(f"\n📊 结果汇总:")
    print(f"  可读视频: {good_count}/{len(all_videos)}")
    print(f"  不可读视频: {len(bad_videos)}/{len(all_videos)}")

    # 4. 输出不可读列表
    if args.bad_list and bad_videos:
        with open(args.bad_list, "w") as f:
            for v in sorted(bad_videos):
                f.write(v + "\n")
        print(f"  不可读列表已保存: {args.bad_list}")

    # 5. 输出干净数据集
    if args.output and bad_videos:
        kept = 0
        removed = 0
        with open(args.output, "w", encoding="utf-8") as fout:
            for sample, raw_line in lines:
                if sample is None:
                    continue
                videos = sample.get(args.video_key, [])
                if isinstance(videos, list) and any(v in bad_videos for v in videos):
                    removed += 1
                    continue
                fout.write(raw_line + "\n")
                kept += 1
        print(f"\n✅ 干净数据集: {kept} 样本 → {args.output} (移除 {removed} 样本)")
    elif args.output and not bad_videos:
        # 所有视频都可读，直接复制
        import shutil
        shutil.copy2(args.input, args.output)
        print(f"\n✅ 所有视频可读，数据集已复制到: {args.output}")

    if bad_videos:
        print(f"\n⚠️  建议: 使用 --output 参数生成过滤后的干净数据集再训练")


if __name__ == "__main__":
    main()
