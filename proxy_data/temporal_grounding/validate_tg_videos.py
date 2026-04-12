"""
TG 训练数据视频验证 & 过滤脚本。

检查项:
  1. 视频文件是否存在且可读 (decord / opencv)
  2. 实际视频时长与 metadata.duration 是否一致 (容差 ±5s)
  3. 时间戳是否在视频时长范围内

用法:
    # 验证 + 生成过滤报告（不修改原文件）
    python validate_tg_videos.py --input data/tg_train_max256s_cot.jsonl --report

    # 验证 + 输出过滤后的 JSONL
    python validate_tg_videos.py \
        --input data/tg_train_max256s_cot.jsonl \
        --output data/tg_train_validated.jsonl

    # 批量验证多个文件
    python validate_tg_videos.py \
        --input data/tg_train_max256s_cot.jsonl data/tg_val_max256s_cot.jsonl \
        --output data/tg_train_validated.jsonl data/tg_val_validated.jsonl
"""

import argparse
import json
import os
import sys
from collections import Counter


def get_video_duration(video_path: str) -> float | None:
    """用 decord (优先) 或 opencv 读取视频实际时长(秒)。返回 None 表示无法读取。"""
    # 尝试 decord
    try:
        from decord import VideoReader
        vr = VideoReader(video_path)
        n_frames = len(vr)
        fps = vr.get_avg_fps()
        if fps > 0 and n_frames > 0:
            return n_frames / fps
    except Exception:
        pass

    # 回退 opencv
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if fps > 0 and n_frames > 0:
                return n_frames / fps
            cap.release()
    except Exception:
        pass

    return None


def validate_record(
    record: dict,
    duration_tolerance: float = 5.0,
    check_video: bool = True,
) -> tuple[bool, str]:
    """
    验证单条记录。

    Returns:
        (is_valid, reason)  reason 为空字符串表示通过
    """
    # 基础字段检查
    videos = record.get("videos", [])
    if not videos:
        return False, "no_video_path"

    video_path = videos[0]
    metadata = record.get("metadata") or {}
    claimed_duration = metadata.get("duration", 0.0)
    timestamp = metadata.get("timestamp", [])

    # 检查时间戳合法性
    if len(timestamp) != 2:
        return False, "bad_timestamp"
    gt_start, gt_end = float(timestamp[0]), float(timestamp[1])
    if gt_start < 0 or gt_end <= gt_start:
        return False, "invalid_timestamp_range"
    if claimed_duration > 0 and gt_end > claimed_duration + 1.0:
        return False, f"timestamp_exceeds_duration({gt_end:.1f}>{claimed_duration:.1f})"

    if not check_video:
        return True, ""

    # 文件存在性
    if not os.path.isfile(video_path):
        return False, "file_not_found"

    # 读取实际时长
    actual_duration = get_video_duration(video_path)
    if actual_duration is None:
        return False, "unreadable_video"

    # 实际时长 vs 标注时长
    if claimed_duration > 0:
        diff = abs(actual_duration - claimed_duration)
        if diff > duration_tolerance:
            return False, f"duration_mismatch(claimed={claimed_duration:.1f},actual={actual_duration:.1f},diff={diff:.1f})"

    # 时间戳超出实际视频时长
    if gt_end > actual_duration + 2.0:
        return False, f"timestamp_exceeds_actual({gt_end:.1f}>{actual_duration:.1f})"

    return True, ""


def process_file(
    input_path: str,
    output_path: str | None,
    duration_tolerance: float,
    check_video: bool,
    report: bool,
) -> dict:
    """处理单个 JSONL 文件，返回统计信息。"""
    print(f"\n{'=' * 60}")
    print(f"  Validating: {input_path}")
    print(f"  check_video={check_video}, tolerance={duration_tolerance}s")
    print(f"{'=' * 60}")

    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    valid_records = []
    invalid_reasons = Counter()
    invalid_examples = []

    for i, rec in enumerate(records):
        is_valid, reason = validate_record(rec, duration_tolerance, check_video)
        if is_valid:
            valid_records.append(rec)
        else:
            invalid_reasons[reason.split("(")[0]] += 1
            if len(invalid_examples) < 10:
                vid = (rec.get("metadata") or {}).get("video_id", "?")
                invalid_examples.append(f"  [{i}] {vid}: {reason}")

    # 统计
    stats = {
        "total": len(records),
        "valid": len(valid_records),
        "invalid": len(records) - len(valid_records),
        "reasons": dict(invalid_reasons),
    }

    print(f"\n  Total:   {stats['total']}")
    print(f"  Valid:   {stats['valid']}")
    print(f"  Invalid: {stats['invalid']}")
    if invalid_reasons:
        print(f"\n  Rejection reasons:")
        for reason, cnt in invalid_reasons.most_common():
            print(f"    {reason}: {cnt}")
    if invalid_examples:
        print(f"\n  Sample rejections:")
        for ex in invalid_examples:
            print(ex)

    # 写输出
    if output_path and valid_records:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for r in valid_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n  Output: {len(valid_records)} records → {output_path}")

    if report and not output_path:
        print(f"\n  [dry-run] Would keep {stats['valid']}/{stats['total']} records")

    return stats


def main():
    parser = argparse.ArgumentParser(description="验证 TG 训练数据视频质量")
    parser.add_argument(
        "--input", nargs="+", required=True,
        help="输入 JSONL 文件路径（支持多个）",
    )
    parser.add_argument(
        "--output", nargs="*", default=None,
        help="输出 JSONL 路径（与 --input 一一对应）。不指定则仅报告。",
    )
    parser.add_argument(
        "--tolerance", type=float, default=5.0,
        help="metadata.duration 与实际时长的最大允许差异(秒)，默认 5.0",
    )
    parser.add_argument(
        "--no-video-check", action="store_true",
        help="跳过视频文件读取检查（仅做元数据层面验证）",
    )
    parser.add_argument(
        "--report", action="store_true",
        help="仅输出报告，不生成过滤文件（dry-run）",
    )
    args = parser.parse_args()

    outputs = args.output or [None] * len(args.input)
    if len(outputs) != len(args.input):
        parser.error(f"--output 数量 ({len(outputs)}) 与 --input ({len(args.input)}) 不匹配")

    all_stats = []
    for inp, out in zip(args.input, outputs):
        if args.report:
            out = None
        s = process_file(inp, out, args.tolerance, not args.no_video_check, args.report)
        all_stats.append(s)

    # 汇总
    if len(args.input) > 1:
        total = sum(s["total"] for s in all_stats)
        valid = sum(s["valid"] for s in all_stats)
        print(f"\n{'=' * 60}")
        print(f"  Overall: {valid}/{total} valid ({100*valid/total:.1f}%)")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
