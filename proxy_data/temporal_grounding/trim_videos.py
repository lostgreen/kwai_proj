#!/usr/bin/env python3
"""
裁切 TimeR1 训练视频中需要按 video_start/video_end 截取的片段。

对 train_2k5.json 中 video_start/video_end 不为 null 的 665 条样本,
用 ffmpeg stream-copy 裁切出对应片段，保存为 *_clipped.mp4，与原视频同目录。

裁切完成后，build_dataset.py 中的 clip_start 过滤逻辑可以移除，
训练数据从 ~1835 条恢复到 ~2500 条。

用法 (在服务器上运行):
    python trim_videos.py \
        --annotation /path/to/train_2k5.json \
        --video-root /m2v_intern/xuboshen/zgw/data/VideoProxyMixed/TimeR1-Dataset \
        [--workers 8] [--overwrite]

    # 也支持 tvgbench.json
    python trim_videos.py \
        --annotation /path/to/train_2k5.json /path/to/tvgbench.json \
        --video-root /m2v_intern/xuboshen/zgw/data/VideoProxyMixed/TimeR1-Dataset \
        [--workers 8] [--overwrite]

裁切后的视频命名: {video_id}_clipped.mp4 (同目录)
"""

import argparse
import json
import os
import subprocess
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


# ── 服务器默认路径 ───────────────────────────────────────
DEFAULT_VIDEO_ROOT = "/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/TimeR1-Dataset"


def clipped_path(video_path: str) -> str:
    """foo/bar.mp4 -> foo/bar_clipped.mp4"""
    base, ext = os.path.splitext(video_path)
    return f"{base}_clipped{ext}"


def parse_train_2k5(items: list, video_root: str):
    """从 train_2k5.json 提取需要裁切的任务列表。"""
    tasks = {}
    skip = 0
    for item in items:
        video_start = item.get("video_start")
        video_end = item.get("video_end")
        if video_start is None or video_end is None:
            skip += 1
            continue

        video_filename = os.path.basename(item["video"])
        src = os.path.join(video_root, "timerft_data", video_filename)

        if src in tasks:
            continue

        tasks[src] = {
            "src": src,
            "dst": clipped_path(src),
            "start": float(video_start),
            "end": float(video_end),
            "duration": float(item["duration"]),
        }
    return tasks, skip


def parse_tvgbench(items: list, video_root: str):
    """从 tvgbench.json 提取需要裁切的任务列表。"""
    tasks = {}
    skip = 0
    for item in items:
        start = item.get("start")
        end = item.get("end")
        if start is None or end is None:
            skip += 1
            continue

        video_filename = os.path.basename(item["path"])
        src = os.path.join(video_root, "tvgbench_data", video_filename)

        if src in tasks:
            continue

        tasks[src] = {
            "src": src,
            "dst": clipped_path(src),
            "start": float(start),
            "end": float(end),
            "duration": float(item["duration"]),
        }
    return tasks, skip


def trim_one(args_tuple) -> str:
    """ffmpeg 裁切单个视频，返回状态字符串。"""
    src, dst, start, end, overwrite = args_tuple

    if not os.path.isfile(src):
        return f"MISSING {src}"

    if os.path.isfile(dst) and not overwrite:
        return f"SKIP {dst}"

    clip_duration = end - start
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-i", src,
        "-t", f"{clip_duration:.3f}",
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        dst,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return f"FAIL {src} -> {dst}: {result.stderr[-300:]}"
    return f"OK {dst}"


def main():
    parser = argparse.ArgumentParser(
        description="裁切 TimeR1 视频 (video_start/video_end)"
    )
    parser.add_argument(
        "--annotation", nargs="+", required=True,
        help="标注 JSON 文件 (train_2k5.json, tvgbench.json)",
    )
    parser.add_argument(
        "--video-root", default=DEFAULT_VIDEO_ROOT,
        help=f"服务器视频根目录 (default: {DEFAULT_VIDEO_ROOT})",
    )
    parser.add_argument("--workers", type=int, default=8, help="并行 ffmpeg 进程数")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的裁切文件")
    parser.add_argument("--dry-run", action="store_true", help="仅列出任务，不执行裁切")
    args = parser.parse_args()

    all_tasks = {}

    for ann_path in args.annotation:
        with open(ann_path) as f:
            data = json.load(f)

        basename = os.path.basename(ann_path)

        # 自动识别 JSON 格式
        if data and "video" in data[0]:
            tasks, skip = parse_train_2k5(data, args.video_root)
        elif data and "path" in data[0]:
            tasks, skip = parse_tvgbench(data, args.video_root)
        else:
            print(f"[WARN] 无法识别 {ann_path} 的格式，跳过")
            continue

        print(f"\n{basename}: {len(data)} 条, 需裁切 {len(tasks)} 个视频, "
              f"无需裁切 {skip} 条")
        all_tasks.update(tasks)

    print(f"\n总共需裁切: {len(all_tasks)} 个唯一视频")

    if args.dry_run:
        print("\n[dry-run] 任务列表:")
        for src, t in sorted(all_tasks.items()):
            print(f"  {os.path.basename(src)}: {t['start']:.1f}s ~ {t['end']:.1f}s "
                  f"(clip={t['end']-t['start']:.1f}s)")
        return

    # 执行裁切
    work_items = [
        (t["src"], t["dst"], t["start"], t["end"], args.overwrite)
        for t in all_tasks.values()
    ]

    stats = Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(trim_one, w): w[0] for w in work_items}
        with tqdm(total=len(futures), desc="Trimming") as pbar:
            for future in as_completed(futures):
                result = future.result()
                status = result.split()[0]
                stats[status] += 1
                if status in ("MISSING", "FAIL"):
                    tqdm.write(result)
                pbar.update(1)

    print(f"\n{'='*40}")
    print(f"  裁切完成")
    print(f"{'='*40}")
    for k, v in stats.most_common():
        print(f"  {k}: {v}")

    if stats.get("OK", 0) > 0:
        print(f"\n裁切后的文件: *_clipped.mp4 (与原视频同目录)")
        print(f"下一步: 修改 build_dataset.py 使裁切样本使用 _clipped 路径")


if __name__ == "__main__":
    main()
