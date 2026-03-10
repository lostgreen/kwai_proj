#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用 decord 直接验证并过滤数据集中无法读取 / 帧数不足的视频样本。

之所以用 decord 而不是 ffprobe：
  训练时 qwen_vl_utils 优先调用 decord.VideoReader，
  decord 失败的视频会直接导致 DataLoader worker 崩溃（KeyError: 'video_fps'）。
  本脚本与训练代码行为完全对齐，只保留 decord 能成功打开的视频。

  此外，Qwen3-VL 要求 temporal_patch_size=2，即视频至少需要 2 帧。
  --min-frames 参数（默认 4）过滤掉帧数过少的视频，避免
  "nframes should in interval [FRAME_FACTOR, total_frames]" 崩溃。

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
import re
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple


# ─────────────────────────────────────────────
# 时间戳解析
# ─────────────────────────────────────────────

def parse_timestamp_from_filename(video_path: str) -> Optional[Tuple[float, float]]:
    """
    从文件名中解析时间戳（秒级）。
    
    YouCook2 格式示例:
        Fe4tO5vW9_E_event00_42_51.mp4  → (42, 51)  # 表示 42-51 秒的片段
    
    返回 (start_sec, end_sec) 或 None（如果格式不匹配）
    """
    basename = os.path.basename(video_path)
    # 匹配 _event\d+_\d+_\d+ 格式
    match = re.search(r'_event\d+_(\d+)_(\d+)', basename)
    if match:
        try:
            start = float(match.group(1))
            end = float(match.group(2))
            return (start, end)
        except (ValueError, IndexError):
            return None
    return None


# ─────────────────────────────────────────────
# 单个视频可读性检查
# ─────────────────────────────────────────────

def check_video_decord(video_path: str, min_frames: int = 4) -> dict:
    """
    用 decord.VideoReader 尝试打开视频，和训练时行为完全一致。
    
    Args:
        video_path: 视频文件路径
        min_frames: 最小帧数要求。Qwen3-VL 的 temporal_patch_size=2
                    要求至少 2 帧；多视频场景 max_frames_per_video 可能
                    很小，默认 4 留足余量。
    
    返回 {
        "path": str, 
        "ok": bool, 
        "error": str, 
        "num_frames": int,
        "duration": float,          # 实际视频时长（秒）
        "timestamp_duration": float, # 文件名中标注的时长，None 则无标注
        "duration_mismatch": bool    # 标注时长 vs 实际时长差异过大
    }
    """
    result = {
        "path": video_path, 
        "ok": False, 
        "error": "", 
        "num_frames": 0,
        "duration": 0.0,
        "timestamp_duration": None,
        "duration_mismatch": False
    }

    if not os.path.exists(video_path):
        result["error"] = "文件不存在"
        return result

    if os.path.getsize(video_path) == 0:
        result["error"] = "文件为空 (0 bytes)"
        return result

    try:
        import decord
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        n_frames = len(vr)
        result["num_frames"] = n_frames
        
        # 计算实际时长
        fps = vr.get_avg_fps() or 24.0  # 默认 24fps
        actual_duration = n_frames / fps
        result["duration"] = actual_duration
        
        if n_frames == 0:
            result["error"] = "视频帧数为 0"
            return result
        if n_frames < min_frames:
            result["error"] = f"视频帧数不足: {n_frames} < {min_frames} (min_frames)"
            return result
        
        # 检查文件名中的时间戳
        ts = parse_timestamp_from_filename(video_path)
        if ts is not None:
            start_sec, end_sec = ts
            ts_duration = end_sec - start_sec
            result["timestamp_duration"] = ts_duration
            # 允许 ±10% 误差（帧率/采样问题）
            if abs(actual_duration - ts_duration) > max(ts_duration * 0.1, 0.5):
                result["duration_mismatch"] = True
        
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
    parser.add_argument("--min-frames", type=int, default=4,
                        help="视频最少帧数（默认 4，Qwen3-VL temporal_patch_size=2 至少需要 2 帧）")
    parser.add_argument("--filter-duration-mismatch", action="store_true", default=False,
                        help="丢弃时长不匹配样本（标注时长 vs 实际时长差异 >10%%）")
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
    duration_mismatch_videos: list = []  # [(path, actual_duration, ts_duration), ...]
    checked = 0
    total = len(all_videos)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(check_video_decord, v, args.min_frames): v
            for v in all_videos
        }
        for future in as_completed(futures):
            res = future.result()
            checked += 1
            if not res["ok"]:
                bad_videos.add(res["path"])
                print(f"  ❌ [{checked:>6}/{total}] {res['path']}")
                print(f"             原因: {res['error']}")
            elif res.get("duration_mismatch", False):
                # 时长不匹配但视频本身可读
                duration_mismatch_videos.append((
                    res["path"],
                    res["duration"],
                    res["timestamp_duration"]
                ))
                if args.filter_duration_mismatch:
                    bad_videos.add(res["path"])
                    print(f"  ⚠️  [{checked:>6}/{total}] {res['path']}")
                    print(f"             原因: 时长不匹配（标注 {res['timestamp_duration']:.1f}s vs 实际 {res['duration']:.1f}s）")
            elif checked % 500 == 0 or checked == total:
                print(f"  ✅ [{checked:>6}/{total}] 进度...")

    # ── 3. 汇总统计 ──
    good_count = total - len(bad_videos)
    print(f"\n📊 结果汇总:")
    print(f"  可读视频   : {good_count} / {total}")
    print(f"  不可读视频  : {len(bad_videos)} / {total}")
    if duration_mismatch_videos:
        mismatch_status = "已丢弃" if args.filter_duration_mismatch else "仅警告，未丢弃（加 --filter-duration-mismatch 可丢弃）"
        print(f"  时长不匹配  : {len(duration_mismatch_videos)} 个（{mismatch_status}）")

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

    # ── 4. 输出时长不匹配诊断 ──
    if duration_mismatch_videos:
        print(f"\n⚠️  时长不匹配诊断（标注vs实际，允许±10%误差）:")
        for video_path, actual_dur, ts_dur in sorted(duration_mismatch_videos)[:20]:  # 显示前 20 个
            print(f"  {os.path.basename(video_path)}")
            print(f"    标注时长: {ts_dur:.2f}s  →  实际时长: {actual_dur:.2f}s  (差 {abs(actual_dur - ts_dur):.2f}s, {abs(actual_dur - ts_dur) / ts_dur * 100:.1f}%)")
        if len(duration_mismatch_videos) > 20:
            print(f"  ... 及其他 {len(duration_mismatch_videos) - 20} 个视频")
        print(f"\n  可能原因:")
        print(f"    1. 视频文件损坏或转码导致实际时长不同")
        print(f"    2. 帧率 (fps) 变化导致采样帧数异常")
        print(f"    3. 时间戳标注错误（罕见）")
        print(f"  建议: 检查这些视频是否确实能被 qwen_vl_utils 正确处理")

    # ── 5. 输出不可读视频列表 ──
    if args.bad_list and bad_videos:
        os.makedirs(os.path.dirname(os.path.abspath(args.bad_list)), exist_ok=True)
        with open(args.bad_list, "w", encoding="utf-8") as f:
            for v in sorted(bad_videos):
                f.write(v + "\n")
        print(f"\n  不可读视频列表 → {args.bad_list}")

    # ── 6. 输出干净数据集 ──
    if args.output is None:
        print("\n（未指定 --output，跳过写出）")
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    if not bad_videos:
        import shutil
        shutil.copy2(args.input, args.output)
        print(f"\n✅ 无需过滤，数据集已复制 → {args.output}")
        return

    kept = removed = 0
    with open(args.output, "w", encoding="utf-8") as fout:
        for sample, raw_line in samples:
            videos = sample.get(args.video_key, [])
            if isinstance(videos, str):
                videos = [videos]
            # 只要有一个视频在 bad_videos 中（不可读 或 时长不匹配且开启了丢弃开关），整个样本丢弃
            if isinstance(videos, list) and any(v in bad_videos for v in videos):
                removed += 1
                continue
            fout.write(raw_line + "\n")
            kept += 1

    mismatch_removed = sum(
        1 for sample, _ in samples
        if any(v in bad_videos and v in {p for p, _, _ in duration_mismatch_videos}
               for v in (sample.get(args.video_key, []) if isinstance(sample.get(args.video_key, []), list) else [sample.get(args.video_key, [])]))
    ) if args.filter_duration_mismatch else 0

    print(f"\n✅ 干净数据集: {kept} 样本 → {args.output}")
    print(f"   移除样本数: {removed}（不可读: {removed - mismatch_removed}，时长不匹配: {mismatch_removed}）")


if __name__ == "__main__":
    main()
