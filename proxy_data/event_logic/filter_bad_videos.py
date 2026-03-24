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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
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

def check_video_decord(
    video_path: str,
    min_frames: int = 4,
    full_decode: bool = False,
    decode_timeout: int = 120,
    video_fps: float = 2.0,
    max_frames: int = 256,
    use_fetch_video: bool = False,
    min_pixels: int = 3136,
    max_pixels: int = 49152,
) -> dict:
    """
    用 decord.VideoReader 尝试打开视频，和训练时行为完全一致。

    Args:
        video_path: 视频文件路径
        min_frames: 最小帧数要求。Qwen3-VL 的 temporal_patch_size=2
                    要求至少 2 帧；多视频场景 max_frames_per_video 可能
                    很小，默认 4 留足余量。
        full_decode: 是否模拟训练时的跳帧采样来解码（比顺序读取更能检测
                     seek 时的损坏，如 h264 mmco 导致的 skip frames 卡死）。
        decode_timeout: full_decode 模式下单个视频的超时秒数（默认 120s）。
        video_fps: 模拟训练时的目标采样帧率（默认 2.0，与 common.sh 对齐）。
        max_frames: 模拟训练时的最大帧数（默认 256，与 common.sh 对齐）。
        use_fetch_video: 直接调用 qwen_vl_utils.fetch_video()（与训练代码完全一致的
                         解码路径），比 full_decode 更准确地检测 seek 失败的视频。
        min_pixels: use_fetch_video 模式下的最小像素数（与 common.sh 对齐）。
        max_pixels: use_fetch_video 模式下的最大像素数（与 common.sh 对齐）。

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

    # ---- use_fetch_video 模式：直接走训练时的完整解码路径 ----
    if use_fetch_video:
        import signal

        def _timeout_handler(signum, frame):
            raise TimeoutError(f"fetch_video 超时 ({decode_timeout}s)")

        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(decode_timeout)
        try:
            from qwen_vl_utils.vision_process import fetch_video as _fetch_video
            vision_info = {
                "video": video_path,
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
                "max_frames": max_frames,
                "fps": video_fps,
            }
            _fetch_video(vision_info, image_patch_size=16,
                         return_video_sample_fps=True,
                         return_video_metadata=True)
            result["ok"] = True
        except TimeoutError as e:
            result["error"] = str(e)[:300]
        except Exception as e:
            result["error"] = str(e)[:300]
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        return result

    # ---- 原有 decord 模式 ----

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

        if full_decode:
            # 模拟训练时的跳帧采样（与 qwen_vl_utils.fetch_video 逻辑对齐）：
            #   step = max(1, round(native_fps / target_fps))
            #   sampled = [0, step, 2*step, ...][:max_frames]
            # 这会触发 decord 的 seek 路径（非顺序读取），能复现训练崩溃的场景。
            import signal

            def _timeout_handler(signum, frame):
                raise TimeoutError(f"视频解码超时 ({decode_timeout}s)")

            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(decode_timeout)
            try:
                step = max(1, round(fps / video_fps))
                sampled_indices = list(range(0, n_frames, step))[:max_frames]
                if not sampled_indices:
                    sampled_indices = [0]
                _ = vr.get_batch(sampled_indices)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        else:
            # 仅验证第 0 帧（快速模式）
            _ = vr[0]

        result["ok"] = True
    except ImportError:
        # decord 未安装时，降级用 ffprobe
        result = _check_video_ffprobe(video_path)
    except TimeoutError as e:
        result["error"] = str(e)[:300]
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
    parser.add_argument("--bad-list-input", default=None,
                        help="读取已有的坏视频列表（每行一个路径），跳过视频解码直接过滤")
    parser.add_argument("--min-frames", type=int, default=4,
                        help="视频最少帧数（默认 4，Qwen3-VL temporal_patch_size=2 至少需要 2 帧）")
    parser.add_argument("--full-decode", action="store_true", default=False,
                        help="模拟训练时跳帧采样解码（比顺序读取更能检测 seek 损坏，如 h264 mmco 错误）")
    parser.add_argument("--decode-timeout", type=int, default=120,
                        help="full-decode 模式下单个视频的超时秒数（默认 120）")
    parser.add_argument("--video-fps", type=float, default=2.0,
                        help="模拟训练时的目标采样帧率（默认 2.0，与 common.sh 对齐）")
    parser.add_argument("--max-frames", type=int, default=256,
                        help="模拟训练时的最大帧数（默认 256，与 common.sh 对齐）")
    parser.add_argument("--filter-duration-mismatch", action="store_true", default=False,
                        help="丢弃时长不匹配样本（标注时长 vs 实际时长差异 >10%%）")
    parser.add_argument("--use-fetch-video", action="store_true", default=False,
                        help="直接调用 qwen_vl_utils.fetch_video()（与训练完全一致的解码路径），"
                             "比 --full-decode 更准确。需要安装 qwen_vl_utils。"
                             "隐含 --full-decode 行为，自动使用多进程 + 超时保护。")
    parser.add_argument("--min-pixels", type=int, default=3136,
                        help="use-fetch-video 模式下的最小像素数（默认 3136，与 common.sh 对齐）")
    parser.add_argument("--max-pixels", type=int, default=49152,
                        help="use-fetch-video 模式下的最大像素数（默认 49152，与 common.sh 对齐）")
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

    # ── 2. 验证视频 ──
    bad_videos: set = set()
    duration_mismatch_videos: list = []  # [(path, actual_duration, ts_duration), ...]

    if args.bad_list_input:
        # 直接从已有列表加载，跳过解码
        if not os.path.exists(args.bad_list_input):
            print(f"\n📋 坏视频列表 {args.bad_list_input} 不存在，视为无坏视频")
        else:
            with open(args.bad_list_input, "r", encoding="utf-8") as f:
                for line in f:
                    v = line.strip()
                    if v:
                        bad_videos.add(v)
            # 只保留当前数据集中实际存在的坏视频
            bad_videos &= all_videos
            print(f"\n📋 从 {args.bad_list_input} 加载坏视频列表: {len(bad_videos)} 个命中当前数据集")
    else:
        # 并行验证视频
        use_fv = args.use_fetch_video
        mode_desc = "fetch_video" if use_fv else ("完整解码" if args.full_decode else "快速检查")
        print(f"\n🔍 验证视频可读性（workers={args.workers}, 模式={mode_desc}）...")
        checked = 0
        total = len(all_videos)

        # use_fetch_video / full_decode 模式用多进程（SIGALRM 超时要求在主线程）
        # 快速模式用多线程（更轻量）
        PoolClass = ProcessPoolExecutor if (args.full_decode or use_fv) else ThreadPoolExecutor

        with PoolClass(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    check_video_decord, v, args.min_frames,
                    full_decode=args.full_decode,
                    decode_timeout=args.decode_timeout,
                    video_fps=args.video_fps,
                    max_frames=args.max_frames,
                    use_fetch_video=use_fv,
                    min_pixels=args.min_pixels,
                    max_pixels=args.max_pixels,
                ): v
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
    total = len(all_videos)
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
    if args.bad_list:
        os.makedirs(os.path.dirname(os.path.abspath(args.bad_list)), exist_ok=True)
        with open(args.bad_list, "w", encoding="utf-8") as f:
            for v in sorted(bad_videos):
                f.write(v + "\n")
        print(f"\n  不可读视频列表 ({len(bad_videos)} 个) → {args.bad_list}")

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
