"""
将 Time-R1 的 TimeRFT 训练数据 / TVGBench 评估数据转换为 EasyR1 JSONL 格式。

用法:
    # 生成 ≤256s 无 CoT 版本（默认）
    python build_dataset.py \
        --timerft_json /path/to/train_2k5.json \
        --tvgbench_json /path/to/tvgbench.json \
        --video_base /m2v_intern/xuboshen/zgw/data/VideoProxyMixed/TimeR1-Dataset \
        --output_dir ./data --max_duration 256

    # 生成 ≤256s CoT 版本（已弃用，CoT 容易陷入死循环）
    python build_dataset.py \
        --timerft_json /path/to/train_2k5.json \
        --video_base /m2v_intern/xuboshen/zgw/data/VideoProxyMixed/TimeR1-Dataset \
        --output_dir ./data --max_duration 256 --mode cot
"""

import argparse
import json
import os
from collections import Counter

# ── 服务器数据根目录 ──────────────────────────────────────────────
DEFAULT_VIDEO_BASE = "/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/TimeR1-Dataset"

# ── Prompt 模板（无 CoT）——————————————————————————
PROMPT_TEMPLATE_NO_COT = (
    'Watch the following video carefully:\n'
    '<video>\n\n'
    'This video is {duration:.1f} seconds long.\n\n'
    'To accurately pinpoint the event "{sentence}" in the video, '
    'determine the precise time period of the event.\n\n'
    'Provide the start and end times in **seconds** (as decimal numbers, e.g. 12.54), '
    'NOT in mm:ss or hh:mm:ss format.\n'
    'Use the format "X.XX to Y.YY" within the <answer> </answer> tags.\n'
    'Example: <answer>12.54 to 17.83</answer>'
)

# ── Prompt 模板（CoT：鼓励模型在 <think> 中分析时间线）——————
PROMPT_TEMPLATE_COT = (
    'Watch the following video carefully:\n'
    '<video>\n\n'
    'This video is {duration:.1f} seconds long.\n\n'
    'To accurately pinpoint the event "{sentence}" in the video, '
    'determine the precise time period of the event.\n\n'
    'Output your thought process within the <think> </think> tags, '
    'including analysis with either specific time ranges (xx.xx to xx.xx) '
    'in <timestep> </timestep> tags.\n\n'
    'Then, provide the start and end times (in seconds, precise to two decimal places) '
    'in the format "start time to end time" within the <answer> </answer> tags. '
    'For example: "12.54 to 17.83".'
)


def parse_source_from_qid(qid: str) -> str:
    """从 qid 中提取数据来源，如 'cosmo', 'yt-temporal' 等。"""
    parts = qid.split("|")
    return parts[1] if len(parts) >= 2 else "unknown"


def round2(val):
    """保留两位小数。"""
    return round(val, 2)


def convert_timerft(items: list, video_base: str, max_duration: float = None, mode: str = "no_cot") -> list:
    """将 TimeRFT train_2k5.json 转换为 EasyR1 格式。"""
    results = []
    stats = {"total": 0, "skipped_no_ts": 0, "skipped_duration": 0, "durations": [], "event_lens": []}
    prompt_tpl = PROMPT_TEMPLATE_COT if mode == "cot" else PROMPT_TEMPLATE_NO_COT

    for item in items:
        stats["total"] += 1

        ts = item.get("timestamp")
        if ts is None or len(ts) != 2:
            stats["skipped_no_ts"] += 1
            continue

        gt_start, gt_end = float(ts[0]), float(ts[1])
        duration = float(item["duration"])
        sentence = item["sentence"]

        # 时长过滤
        if max_duration is not None and duration > max_duration:
            stats["skipped_duration"] += 1
            continue

        video_filename = os.path.basename(item["video"])
        video_path = os.path.join(video_base, "timerft_data", video_filename)

        # 处理 video_start / video_end（部分样本需要裁切）
        video_start = item.get("video_start")
        video_end = item.get("video_end")
        if video_start is not None and video_end is not None:
            # 使用预裁切的 _clipped 视频文件
            base, ext = os.path.splitext(video_path)
            video_path = f"{base}_clipped{ext}"

        # 构建 prompt
        prompt = prompt_tpl.format(duration=duration, sentence=sentence)

        # 构建 answer
        answer = f"<answer>{round2(gt_start):.2f} to {round2(gt_end):.2f}</answer>"

        # 构建 metadata
        qid = item.get("qid", "")
        source = parse_source_from_qid(qid)
        metadata = {
            "video_id": video_filename.replace(".mp4", ""),
            "duration": duration,
            "timestamp": [round2(gt_start), round2(gt_end)],
            "sentence": sentence,
            "source": source,
            "difficulty": item.get("difficulty"),
            "qid": qid,
        }

        record = {
            "messages": [{"role": "user", "content": prompt}],
            "prompt": prompt,
            "answer": answer,
            "videos": [video_path],
            "data_type": "video",
            "problem_type": "temporal_grounding",
            "metadata": metadata,
        }
        results.append(record)
        stats["durations"].append(duration)
        stats["event_lens"].append(gt_end - gt_start)

    return results, stats


def convert_tvgbench(items: list, video_base: str, max_duration: float = None, mode: str = "no_cot") -> list:
    """将 TVGBench tvgbench.json 转换为 EasyR1 格式。"""
    results = []
    stats = {"total": 0, "skipped_bad_answer": 0, "skipped_duration": 0, "durations": [], "datasets": []}
    prompt_tpl = PROMPT_TEMPLATE_COT if mode == "cot" else PROMPT_TEMPLATE_NO_COT

    for item in items:
        stats["total"] += 1

        # 解析 answer "13.4-28.1" → [13.4, 28.1]
        answer_str = item.get("answer", "")
        parts = answer_str.split("-")
        if len(parts) != 2:
            stats["skipped_bad_answer"] += 1
            continue
        try:
            gt_start, gt_end = float(parts[0]), float(parts[1])
        except ValueError:
            stats["skipped_bad_answer"] += 1
            continue

        duration = float(item["duration"])
        sentence = item["question"]

        # 时长过滤
        if max_duration is not None and duration > max_duration:
            stats["skipped_duration"] += 1
            continue

        video_filename = os.path.basename(item["path"])
        video_path = os.path.join(video_base, "tvgbench_data", video_filename)

        # 处理 start / end（部分样本需要裁切）
        clip_start = item.get("start")
        clip_end = item.get("end")
        if clip_start is not None and clip_end is not None:
            base, ext = os.path.splitext(video_path)
            video_path = f"{base}_clipped{ext}"

        prompt = prompt_tpl.format(duration=duration, sentence=sentence)
        answer = f"<answer>{round2(gt_start):.2f} to {round2(gt_end):.2f}</answer>"

        metadata = {
            "video_id": video_filename.replace(".mp4", ""),
            "duration": duration,
            "timestamp": [round2(gt_start), round2(gt_end)],
            "sentence": sentence,
            "dataset_name": item.get("dataset_name", ""),
            "qsemtype": item.get("qsemtype", ""),
        }

        record = {
            "messages": [{"role": "user", "content": prompt}],
            "prompt": prompt,
            "answer": answer,
            "videos": [video_path],
            "data_type": "video",
            "problem_type": "temporal_grounding",
            "metadata": metadata,
        }
        results.append(record)
        stats["durations"].append(duration)
        stats["datasets"].append(item.get("dataset_name", "unknown"))

    return results, stats


def print_stats(name: str, stats: dict):
    """打印数据统计信息。"""
    print(f"\n{'='*60}")
    print(f"  {name} 统计")
    print(f"{'='*60}")
    print(f"  总条数:       {stats['total']}")

    if "skipped_no_ts" in stats:
        print(f"  跳过(无时间戳): {stats['skipped_no_ts']}")
    if "skipped_bad_answer" in stats:
        print(f"  跳过(格式错误): {stats['skipped_bad_answer']}")
    if "skipped_duration" in stats:
        print(f"  跳过(超时长):   {stats['skipped_duration']}")

    durs = stats["durations"]
    if durs:
        print(f"  有效条数:     {len(durs)}")
        print(f"  时长分布:")
        print(f"    min:  {min(durs):.1f}s")
        print(f"    max:  {max(durs):.1f}s")
        print(f"    mean: {sum(durs)/len(durs):.1f}s")
        # 分段统计
        bins = [0, 30, 60, 128, 256, 600, float("inf")]
        labels = ["≤30s", "30-60s", "60-128s", "128-256s", "256-600s", ">600s"]
        for i in range(len(labels)):
            count = sum(1 for d in durs if bins[i] <= d < bins[i + 1])
            pct = 100 * count / len(durs)
            marker = " ⚠️ 超过128s" if bins[i] >= 128 and count > 0 else ""
            print(f"    {labels[i]:>10}: {count:4d} ({pct:5.1f}%){marker}")

    if "event_lens" in stats and stats["event_lens"]:
        elens = stats["event_lens"]
        print(f"  事件长度分布:")
        print(f"    min:  {min(elens):.1f}s")
        print(f"    max:  {max(elens):.1f}s")
        print(f"    mean: {sum(elens)/len(elens):.1f}s")

    if "datasets" in stats and stats["datasets"]:
        print(f"  数据集分布:")
        for ds, cnt in Counter(stats["datasets"]).most_common():
            print(f"    {ds}: {cnt}")


def write_jsonl(records: list, output_path: str):
    """写入 JSONL 文件。"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  已写入 {len(records)} 条 → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Time-R1 data → EasyR1 JSONL 转换")
    parser.add_argument(
        "--timerft_json",
        type=str,
        default=None,
        help="TimeRFT train_2k5.json 路径",
    )
    parser.add_argument(
        "--tvgbench_json",
        type=str,
        default=None,
        help="TVGBench tvgbench.json 路径",
    )
    parser.add_argument(
        "--video_base",
        type=str,
        default=DEFAULT_VIDEO_BASE,
        help="服务器上视频根目录",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="输出 JSONL 存放目录",
    )
    parser.add_argument(
        "--max_duration",
        type=float,
        default=None,
        help="最大视频时长(秒)，超过的样本将被过滤。如 256 表示只保留 ≤256s 的视频",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["cot", "no_cot"],
        default="no_cot",
        help="Prompt 模式: cot (带 CoT 思考，已弃用) 或 no_cot (直接输出)",
    )
    parser.add_argument(
        "--n_val",
        type=int,
        default=100,
        help="验证集条数 (从合并数据中随机抽取)",
    )
    args = parser.parse_args()

    suffix = f"_{args.mode}" if args.mode != "no_cot" else ""
    dur_tag = f"_max{int(args.max_duration)}s" if args.max_duration else ""

    all_records = []

    if args.timerft_json:
        print(f"读取 TimeRFT: {args.timerft_json}")
        print(f"模式: {args.mode}, 最大时长: {args.max_duration or '不限'}s")
        with open(args.timerft_json, "r", encoding="utf-8") as f:
            timerft_items = json.load(f)
        records, stats = convert_timerft(timerft_items, args.video_base, args.max_duration, args.mode)
        print_stats("TimeRFT (train_2k5)", stats)

        all_records.extend(records)

    if args.tvgbench_json:
        print(f"\n读取 TVGBench: {args.tvgbench_json}")
        with open(args.tvgbench_json, "r", encoding="utf-8") as f:
            tvgbench_items = json.load(f)
        records, stats = convert_tvgbench(tvgbench_items, args.video_base, args.max_duration, args.mode)
        print_stats("TVGBench (eval)", stats)
        all_records.extend(records)

    if not all_records:
        print("请指定 --timerft_json 或 --tvgbench_json（至少一个）")
        return

    # Shuffle + train/val split
    import random
    rng = random.Random(42)
    rng.shuffle(all_records)

    n_val = min(args.n_val, len(all_records) // 5)
    val_records = all_records[:n_val]
    train_records = all_records[n_val:]

    print(f"\n合并后: {len(all_records)} 条 → train {len(train_records)} + val {n_val}")

    train_path = os.path.join(args.output_dir, f"tg_train{dur_tag}{suffix}.jsonl")
    val_path = os.path.join(args.output_dir, f"tg_val{dur_tag}{suffix}.jsonl")
    write_jsonl(train_records, train_path)
    write_jsonl(val_records, val_path)


if __name__ == "__main__":
    main()
