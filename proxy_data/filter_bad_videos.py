#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用 decord 直接验证并过滤数据集中无法读取的视频样本。

之所以用 decord 而不是 ffprobe：
  训练时 qwen_vl_utils 优先调用 decord.VideoReader，
  decord 失败的视频会直接导致 DataLoader worker 崩溃（KeyError: 'video_fps'）。
  本脚本与训练代码行为完全对齐，只保留 decord 能成功打开的视频。

用法:
    # 检查 + 过滤，生成干净数据集
    python proxy_data/filter_bad_videos.py \
        -i proxy_data/mixed_train.jsonl \
        -o proxy_data/mixed_train_clean.jsonl

    # 只检查，不输出
    python proxy_data/filter_bad_videos.py -i proxy_data/mixed_train.jsonl

    # 指定并行线程数
    python proxy_data/filter_bad_videos.py \
        -i proxy_data/mixed_train.jsonl \
        -o proxy_data/mixed_train_clean.jsonl \
        --workers 32

    # 将所有坏视频路径写入文件
    python proxy_data/filter_bad_videos.py \
        -i proxy_data/mixed_train.jsonl \
        -o proxy_data/mixed_train_clean.jsonl \
        --bad_list proxy_data/bad_videos.txt
"""

import json
import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional


# ─────────────────────────────────────────────
# 单个视频可读性检查
# ─────────────────────────────────────────────

def check_video_decord(video_path: str) -> dict:
    """
    用 decord.VideoReader 尝试打开视频，和训练时行为完全一致。
    返回 {"path": str, "ok": bool, "error": str}
    """
    result = {"path": video_path, "ok": False, "error": ""}

    if not os.path.exists(video_path):
        result["error"] = "文件不存在"
        return result

    if os.path.getsize(video_path) == 0:
        result["error"] = "文件为空 (0 bytes)"
        return result

    try:
        import decord
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        if len(vr) == 0:
            result["error"] = "视频帧数为 0"
            return result
        # 额外验证：尝试读第 0 帧
        _ = vr[0]
        result["ok"] = True
    except ImportError:
        # decord 未安装时，降级用 ffprobe
        result = _check_video_ffprobe(video_path)
    except Exception as e:
        result["error"] = str(e)[:300]

    return result


def _check_video_ffprobe(video_path: str, timeout: int = 10) -> dict:
    """ffprobe 降级检查（当 decord 未安装时使用）。"""
    import subprocess
    result = {"path": video_path, "ok": False, "error": ""}
    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_type",
                "-of", "json", video_path,
            ],
            capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode != 0:
            result["error"] = proc.stderr.strip()[:200]
            return result
        info = json.loads(proc.stdout)
        if not info.get("streams"):
            result["error"] = "无视频流"
            return result
        result["ok"] = True
    except subprocess.TimeoutExpired:
        result["error"] = "ffprobe 超时"
    except FileNotFoundError:
        result["error"] = "ffprobe 未安装，且 decord 也未安装"
    except Exception as e:
        result["error"] = str(e)[:200]
    return result


# ─────────────────────────────────────────────
# 主逻辑
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="用 decord 过滤数据集中不可读的视频样本"
    )
    parser.add_argument("--input",  "-i", required=True, help="输入 JSONL 文件")
    parser.add_argument("--output", "-o", default=None,  help="输出干净 JSONL（省略则只检查不写出）")
    parser.add_argument("--video_key", default="videos",  help="视频路径字段名（默认 videos）")
    parser.add_argument("--workers", "-w", type=int, default=16, help="并行线程数（默认 16）")
    parser.add_argument("--bad_list", default=None, help="将不可读视频路径写入此文件")
    args = parser.parse_args()

    # ── 1. 读取数据集，收集所有唯一视频路径 ──
    print(f"📂 读取数据集: {args.input}")
    samples = []       # [(dict, raw_line_str)]
    all_videos: set = set()

    with open(args.input, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [WARN] 第 {lineno} 行 JSON 解析失败，跳过: {e}")
                continue

            videos = sample.get(args.video_key, [])
            if isinstance(videos, str):
                videos = [videos]
            if isinstance(videos, list):
                all_videos.update(v for v in videos if isinstance(v, str))

            samples.append((sample, line))

    print(f"  总样本数   : {len(samples)}")
    print(f"  唯一视频数  : {len(all_videos)}")

    if not all_videos:
        print("⚠️  未找到任何视频路径（检查 --video_key 是否正确）")
        return

    # ── 2. 并行验证视频 ──
    print(f"\n🔍 验证视频可读性（workers={args.workers}）...")
    bad_videos: set = set()
    checked = 0
    total = len(all_videos)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(check_video_decord, v): v
            for v in all_videos
        }
        for future in as_completed(futures):
            res = future.result()
            checked += 1
            if not res["ok"]:
                bad_videos.add(res["path"])
                print(f"  ❌ [{checked:>6}/{total}] {res['path']}")
                print(f"             原因: {res['error']}")
            elif checked % 500 == 0 or checked == total:
                print(f"  ✅ [{checked:>6}/{total}] 进度...")

    # ── 3. 汇总统计 ──
    good_count = total - len(bad_videos)
    print(f"\n📊 结果汇总:")
    print(f"  可读视频   : {good_count} / {total}")
    print(f"  不可读视频  : {len(bad_videos)} / {total}")

    # 按任务类型统计丢弃情况
    if bad_videos:
        task_counts: dict = {}
        for sample, _ in samples:
            videos = sample.get(args.video_key, [])
            if isinstance(videos, str):
                videos = [videos]
            if isinstance(videos, list) and any(v in bad_videos for v in videos):
                pt = sample.get("problem_type", sample.get("data_type", "unknown"))
                task_counts[pt] = task_counts.get(pt, 0) + 1
        if task_counts:
            print(f"\n  按任务类型统计受影响样本数:")
            for t, c in sorted(task_counts.items()):
                print(f"    {t:20s}: {c}")

    # ── 4. 输出不可读视频列表 ──
    if args.bad_list and bad_videos:
        os.makedirs(os.path.dirname(os.path.abspath(args.bad_list)), exist_ok=True)
        with open(args.bad_list, "w", encoding="utf-8") as f:
            for v in sorted(bad_videos):
                f.write(v + "\n")
        print(f"\n  不可读视频列表 → {args.bad_list}")

    # ── 5. 输出干净数据集 ──
    if args.output is None:
        print("\n（未指定 --output，跳过写出）")
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    if not bad_videos:
        import shutil
        shutil.copy2(args.input, args.output)
        print(f"\n✅ 所有视频均可读，数据集已复制 → {args.output}")
        return

    kept = removed = 0
    with open(args.output, "w", encoding="utf-8") as fout:
        for sample, raw_line in samples:
            videos = sample.get(args.video_key, [])
            if isinstance(videos, str):
                videos = [videos]
            # 只要有一个视频不可读，整个样本丢弃
            if isinstance(videos, list) and any(v in bad_videos for v in videos):
                removed += 1
                continue
            fout.write(raw_line + "\n")
            kept += 1

    print(f"\n✅ 干净数据集: {kept} 样本 → {args.output}")
    print(f"   移除样本数: {removed}（含 ≥1 个不可读视频）")


if __name__ == "__main__":
    main()
