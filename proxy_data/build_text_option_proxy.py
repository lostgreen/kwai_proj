#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 YouCookII 原始标注重新生成 proxy 训练数据（add / replace 文本选项），
并将“选项”从视频改为 trainval 标注文本（sentence）。

输出为 EasyR1 JSONL 格式：
{
  "messages": [{"role": "user", "content": "..."}],
  "prompt": "...",
  "answer": "A|B|C|D",
  "videos": ["...mp4", ...],
  "data_type": "video",
    "problem_type": "add|replace",
  "metadata": {...}
}

示例:
python proxy_data/build_text_option_proxy.py \
  -a proxy_data/youcookii_annotations_trainval.json \
  -o proxy_data/proxy_train_text_options.jsonl \
  --event-clips-root /m2v_intern/xuboshen/zgw/data/youcook2_event_clips \
  --add-per-video 1 \
    --replace-per-video 1 \
  --seed 42 \
  --max-samples 1000 \
  --shuffle \
  --min-context 3 \
  --max-context 4 \
  --replace-context-len 5 \
  --validate-clips \
  --duration-tol 1.5
"""

import argparse
import json
import os
import random
import re
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed


LETTERS = [chr(ord("A") + i) for i in range(26)]
EVENT_FILE_RE = re.compile(r"_event\d+_(\d+)_(\d+)\.mp4$")


def get_option_labels(n):
    """返回 n 个选项标签（A, B, C...），最多支持 26 个。"""
    if n < 2:
        raise ValueError(f"选项数必须 >=2，当前: {n}")
    if n > len(LETTERS):
        raise ValueError(f"选项数超过上限 {len(LETTERS)}，当前: {n}")
    return LETTERS[:n]


def normalize_segment(seg):
    """segment [start, end] -> (int_start, int_end)"""
    if not isinstance(seg, (list, tuple)) or len(seg) != 2:
        return None
    try:
        s = int(round(float(seg[0])))
        e = int(round(float(seg[1])))
    except (TypeError, ValueError):
        return None
    if s >= e:
        return None
    return s, e


def build_event_path(root, subset, recipe_type, video_id, event_id, start, end):
    fname = f"{video_id}_event{event_id:02d}_{start}_{end}.mp4"
    return os.path.join(root, subset, str(recipe_type), video_id, fname)


def parse_expected_duration_from_filename(path):
    """从 ..._eventXX_start_end.mp4 解析期望时长 (end-start)"""
    m = EVENT_FILE_RE.search(path)
    if not m:
        return None
    try:
        start = int(m.group(1))
        end = int(m.group(2))
    except ValueError:
        return None
    if end <= start:
        return None
    return float(end - start)


def probe_video(video_path, timeout=10):
    """ffprobe 检查可读性并读取时长。"""
    result = {"path": video_path, "ok": False, "duration": None, "error": ""}

    if not os.path.exists(video_path):
        result["error"] = "文件不存在"
        return result

    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=duration",
                "-of", "json",
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            result["error"] = (proc.stderr or "ffprobe failed").strip()[:200]
            return result

        info = json.loads(proc.stdout or "{}")
        streams = info.get("streams", [])
        if not streams:
            result["error"] = "无视频流"
            return result

        duration = streams[0].get("duration", None)
        if duration is None:
            result["error"] = "无duration"
            return result

        duration = float(duration)
        result["duration"] = duration
        result["ok"] = True
        return result

    except subprocess.TimeoutExpired:
        result["error"] = f"ffprobe 超时({timeout}s)"
        return result
    except FileNotFoundError:
        result["error"] = "ffprobe 未安装"
        return result
    except Exception as e:  # noqa: BLE001
        result["error"] = str(e)[:200]
        return result


def validate_event_clips(videos, timeout=10, workers=16, duration_tol=1.5, check_duration=True):
    """
    校验事件 clip：
    1) 可读性
    2) 时长与文件名中的 (end-start) 是否接近

    返回:
      - path_status: {path: {ok, duration, expected, reason}}
      - summary: Counter
    """
    all_paths = set()
    for v in videos:
        for ev in v["events"]:
            all_paths.add(ev["path"])

    path_status = {}
    summary = Counter()
    if not all_paths:
        return path_status, summary

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(probe_video, p, timeout): p for p in all_paths}
        for fut in as_completed(futs):
            p = futs[fut]
            r = fut.result()
            if not r["ok"]:
                path_status[p] = {
                    "ok": False,
                    "duration": None,
                    "expected": parse_expected_duration_from_filename(p),
                    "reason": r["error"] or "不可读",
                }
                summary["unreadable"] += 1
                continue

            expected = parse_expected_duration_from_filename(p)
            duration = r["duration"]
            if check_duration and expected is not None and abs(duration - expected) > duration_tol:
                path_status[p] = {
                    "ok": False,
                    "duration": duration,
                    "expected": expected,
                    "reason": f"duration_mismatch(|{duration:.2f}-{expected:.2f}|>{duration_tol:.2f})",
                }
                summary["duration_mismatch"] += 1
                continue

            path_status[p] = {
                "ok": True,
                "duration": duration,
                "expected": expected,
                "reason": "",
            }
            summary["valid"] += 1

    summary["total_checked"] = len(all_paths)
    return path_status, summary


def filter_videos_by_path_status(videos, path_status, min_events):
    """基于 path_status 过滤事件，保留事件数 >= min_events 的视频。"""
    filtered = []
    removed_events = 0
    removed_videos = 0
    for v in videos:
        new_events = []
        for ev in v["events"]:
            st = path_status.get(ev["path"])
            if st is not None and not st["ok"]:
                removed_events += 1
                continue
            new_events.append(ev)
        if len(new_events) < min_events:
            removed_videos += 1
            continue
        vv = dict(v)
        vv["events"] = new_events
        filtered.append(vv)
    return filtered, removed_events, removed_videos


def rebuild_sentence_pool_by_recipe(videos):
    pool = defaultdict(list)
    for v in videos:
        for ev in v["events"]:
            pool[v["recipe_type"]].append((v["video_id"], ev["id"], ev["sentence"]))
    return pool


def load_videos(anno_path, event_clips_root, min_events=4):
    with open(anno_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    db = raw.get("database", {})
    videos = []
    sentence_pool_by_recipe = defaultdict(list)

    for video_id, item in db.items():
        subset = item.get("subset", "")
        recipe_type = str(item.get("recipe_type", ""))
        ann = item.get("annotations", [])
        if not subset or not recipe_type or not isinstance(ann, list):
            continue

        events = []
        for ev in ann:
            seg = normalize_segment(ev.get("segment"))
            if seg is None:
                continue
            sentence = (ev.get("sentence") or "").strip()
            if not sentence:
                continue
            event_id = int(ev.get("id", len(events)))
            s, e = seg
            path = build_event_path(event_clips_root, subset, recipe_type, video_id, event_id, s, e)
            events.append({
                "id": event_id,
                "start": s,
                "end": e,
                "sentence": sentence,
                "path": path,
            })
            sentence_pool_by_recipe[recipe_type].append((video_id, event_id, sentence))

        events.sort(key=lambda x: x["id"])
        if len(events) < min_events:
            continue

        videos.append({
            "video_id": video_id,
            "subset": subset,
            "recipe_type": recipe_type,
            "events": events,
        })

    return videos, sentence_pool_by_recipe


def sample_negative_sentences(sentence_pool_by_recipe, anchor_video_id,
                              anchor_recipe_type, gt_sentence,
                              same_video_other_sentences, k=3):
    """
    采样 k 个负例文本，混合两类干扰项以保证可区分性：
      1. 从别的菜谱（不同 recipe_type）采样 1-2 个句子（容易区分的干扰项）
      2. 从相同视频的其他事件采样剩余句子（较难但可区分的干扰项）
    如果同视频其他事件不够，从同菜谱不同视频补齐。
    """
    n_cross = random.randint(1, min(2, k))
    n_same = k - n_cross

    used = {gt_sentence}
    result = []

    # ---- 1. 从不同菜谱采样 n_cross 个 ----
    cross_cands = set()
    for rt, pool in sentence_pool_by_recipe.items():
        if rt == anchor_recipe_type:
            continue
        for _, _, sent in pool:
            if sent not in used:
                cross_cands.add(sent)

    if len(cross_cands) < n_cross:
        return None

    sampled_cross = random.sample(sorted(cross_cands), n_cross)
    result.extend(sampled_cross)
    used.update(sampled_cross)

    # ---- 2. 从同视频其他事件采样 n_same 个（严格，不再回退到同菜谱其他视频） ----
    same_vid_cands = []
    seen_same_vid = set()
    for s in same_video_other_sentences:
        if s in used or s in seen_same_vid:
            continue
        seen_same_vid.add(s)
        same_vid_cands.append(s)

    if len(same_vid_cands) < n_same:
        return None

    sampled_same = random.sample(same_vid_cands, n_same)
    result.extend(sampled_same)

    return result


def format_add_prompt(num_ctx, options, cot=True):
    labels = get_option_labels(len(options))
    lines = ["Context Video Sequence:"]
    for i in range(num_ctx):
        lines.append(f"{i + 1}. <video>")

    lines.extend([
        "",
        "Based on the continuous actions shown in the Context Video Sequence above, which of the following textual options shows the most logical continuous next cooking step?",
        "Options:",
    ])

    for i, opt in enumerate(options):
        lines.append(f"{labels[i]}. {opt}")

    if cot:
        lines.extend([
            "",
            "First, carefully observe the actions and visual content in each Context Video to understand the cooking progression. Then, reason about which text option best continues the sequence.",
            "",
            f"Think step by step inside <think> </think> tags, then provide your final answer (a single letter from {', '.join(labels)}) inside <answer> </answer> tags.",
        ])
    else:
        lines.extend([
            "",
            f"Output your answer as a single letter (e.g., {', '.join(labels)}).",
        ])

    return "\n".join(lines)


def format_replace_prompt(total_steps, missing_pos, options, cot=True):
    labels = get_option_labels(len(options))
    lines = ["Watch the following cooking process carefully. The sequence has a [MISSING] step in the middle."]
    lines.append("Context Sequence:")
    for i in range(total_steps):
        if i == missing_pos:
            lines.append(f"Step {i + 1}: [MISSING]")
        else:
            lines.append(f"Step {i + 1}: <video>")

    lines.extend([
        "",
        "Based on the chronological visual content of the sequence, pick the correct textual option to fill in the [MISSING] step.",
        "Options:",
    ])

    for i, opt in enumerate(options):
        lines.append(f"{labels[i]}. {opt}")

    if cot:
        lines.extend([
            "",
            "First, carefully observe the Context Sequence to understand the cooking flow before and after the [MISSING] step. Then, reason about which text option best fills the gap.",
            "",
            f"Think step by step inside <think> </think> tags, then provide your final answer (a single letter from {', '.join(labels)}) inside <answer> </answer> tags.",
        ])
    else:
        lines.extend([
            "",
            f"Output your answer as a single letter (e.g., {', '.join(labels)}).",
        ])

    return "\n".join(lines)


def build_add_sample(v, sentence_pool_by_recipe, min_ctx=2, max_ctx=4, cot=True):
    events = v["events"]
    if len(events) < min_ctx + 1:
        return None

    ctx_len = random.randint(min_ctx, min(max_ctx, len(events) - 1))
    max_start = len(events) - (ctx_len + 1)
    start = random.randint(0, max_start)

    ctx_events = events[start:start + ctx_len]
    gt_event = events[start + ctx_len]

    # 同视频其他事件的句子（排除上下文和 gt）
    used_ids = {ev["id"] for ev in ctx_events} | {gt_event["id"]}
    same_video_other = [ev["sentence"] for ev in events if ev["id"] not in used_ids]

    negs = sample_negative_sentences(
        sentence_pool_by_recipe=sentence_pool_by_recipe,
        anchor_video_id=v["video_id"],
        anchor_recipe_type=v["recipe_type"],
        gt_sentence=gt_event["sentence"],
        same_video_other_sentences=same_video_other,
        k=3,
    )
    if negs is None:
        return None

    options = negs + [gt_event["sentence"]]
    random.shuffle(options)
    labels = get_option_labels(len(options))
    answer = labels[options.index(gt_event["sentence"])]

    prompt = format_add_prompt(len(ctx_events), options, cot=cot)
    videos = [x["path"] for x in ctx_events]

    return {
        "messages": [{"role": "user", "content": prompt}],
        "prompt": prompt,
        "answer": answer,
        "videos": videos,
        "data_type": "video",
        "problem_type": "add",
        "metadata": {
            "task_type": "add",
            "video_id": v["video_id"],
            "recipe_type": v["recipe_type"],
            "context_start_event": ctx_events[0]["id"],
            "context_end_event": ctx_events[-1]["id"],
            "target_event": gt_event["id"],
            "target_sentence": gt_event["sentence"],
            "option_type": "text",
        },
    }


def build_replace_sample(v, sentence_pool_by_recipe, seq_len=5, cot=True):
    events = v["events"]
    if len(events) < seq_len:
        return None

    start = random.randint(0, len(events) - seq_len)
    seq_events = events[start:start + seq_len]
    if seq_len < 3:
        return None

    # 缺失位置放在中间，避免退化为开头/结尾猜测
    missing_pos = random.randint(1, seq_len - 2)
    gt_event = seq_events[missing_pos]

    # 同视频其他事件的句子（排除序列中的所有事件）
    seq_event_ids = {ev["id"] for ev in seq_events}
    same_video_other = [ev["sentence"] for ev in events if ev["id"] not in seq_event_ids]

    negs = sample_negative_sentences(
        sentence_pool_by_recipe=sentence_pool_by_recipe,
        anchor_video_id=v["video_id"],
        anchor_recipe_type=v["recipe_type"],
        gt_sentence=gt_event["sentence"],
        same_video_other_sentences=same_video_other,
        k=3,
    )
    if not negs:
        return None
    options = negs + [gt_event["sentence"]]
    random.shuffle(options)
    labels = get_option_labels(len(options))
    gt_letter = labels[options.index(gt_event["sentence"])]

    context_events = [ev for i, ev in enumerate(seq_events) if i != missing_pos]

    prompt = format_replace_prompt(
        total_steps=len(seq_events),
        missing_pos=missing_pos,
        options=options,
        cot=cot,
    )
    videos = [x["path"] for x in context_events]

    return {
        "messages": [{"role": "user", "content": prompt}],
        "prompt": prompt,
        "answer": gt_letter,
        "videos": videos,
        "data_type": "video",
        "problem_type": "replace",
        "metadata": {
            "task_type": "replace",
            "video_id": v["video_id"],
            "recipe_type": v["recipe_type"],
            "context_start_event": seq_events[0]["id"],
            "context_end_event": seq_events[-1]["id"],
            "missing_pos": missing_pos,
            "target_event": gt_event["id"],
            "target_sentence": gt_event["sentence"],
            "option_type": "text",
        },
    }


def main():
    parser = argparse.ArgumentParser(description="从原始标注生成文本选项版 proxy(add/replace) 训练数据")
    parser.add_argument("--annotations", "-a", required=True, help="youcookii_annotations_trainval.json 路径")
    parser.add_argument("--output", "-o", required=True, help="输出 JSONL 文件")
    parser.add_argument("--event-clips-root", required=True, help="事件切片根目录，例如 /.../youcook2_event_clips")

    parser.add_argument("--add-per-video", type=int, default=1, help="每个视频生成 add 样本数")
    parser.add_argument("--replace-per-video", type=int, default=1, help="每个视频生成 replace 样本数")
    parser.add_argument("--min-events", type=int, default=4, help="最少事件数")
    parser.add_argument("--min-context", type=int, default=2, help="add 任务最小上下文长度")
    parser.add_argument("--max-context", type=int, default=4, help="add 任务最大上下文长度")
    parser.add_argument("--replace-context-len", type=int, default=5, help="replace 任务总步骤数（输入视频数=总步骤数-1）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--shuffle", action="store_true", help="写出前打乱样本")
    parser.add_argument("--max-samples", type=int, default=None, help="最多写出样本数")
    parser.add_argument("--no-cot", action="store_true", help="不使用 <think>/<answer> 指令")
    parser.add_argument("--validate-clips", action="store_true", help="生成前校验 clip 可读性与时长")
    parser.add_argument("--validate-workers", type=int, default=16, help="校验 clip 并行线程数")
    parser.add_argument("--validate-timeout", type=int, default=10, help="单个 ffprobe 超时时间(秒)")
    parser.add_argument("--duration-tol", type=float, default=1.5, help="时长容差(秒)，用于比对文件名中的 end-start")
    parser.add_argument("--no-duration-check", action="store_true", help="只检查可读性，不检查时长一致性")
    parser.add_argument("--validation-report", default=None, help="输出校验失败明细 JSONL")

    args = parser.parse_args()
    random.seed(args.seed)

    if args.replace_context_len < 3:
        raise ValueError("--replace-context-len 必须 >= 3")

    videos, sentence_pool_by_recipe = load_videos(
        anno_path=args.annotations,
        event_clips_root=args.event_clips_root,
        min_events=args.min_events,
    )
    if not videos:
        raise RuntimeError("没有可用视频，请检查标注文件或 --min-events 参数")

    if args.validate_clips:
        print("🔍 开始校验事件 clips（可读性/时长）...")
        path_status, val_summary = validate_event_clips(
            videos,
            timeout=args.validate_timeout,
            workers=args.validate_workers,
            duration_tol=args.duration_tol,
            check_duration=not args.no_duration_check,
        )
        videos, removed_events, removed_videos = filter_videos_by_path_status(
            videos,
            path_status=path_status,
            min_events=args.min_events,
        )
        sentence_pool_by_recipe = rebuild_sentence_pool_by_recipe(videos)
        print(
            f"  校验完成: total={val_summary['total_checked']}, "
            f"valid={val_summary['valid']}, unreadable={val_summary['unreadable']}, "
            f"duration_mismatch={val_summary['duration_mismatch']}"
        )
        print(f"  过滤事件数: {removed_events}, 过滤视频数: {removed_videos}, 剩余视频数: {len(videos)}")

        if args.validation_report:
            out_dir = os.path.dirname(args.validation_report)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(args.validation_report, "w", encoding="utf-8") as rf:
                for p, st in path_status.items():
                    if st["ok"]:
                        continue
                    rf.write(json.dumps({"path": p, **st}, ensure_ascii=False) + "\n")
            print(f"  校验失败报告: {args.validation_report}")

    if not videos:
        raise RuntimeError("校验后无可用视频，请放宽过滤条件或检查 clip 根目录")

    samples = []
    stats = Counter()
    ctx_stats = Counter()

    for v in videos:
        for _ in range(args.add_per_video):
            s = build_add_sample(
                v,
                sentence_pool_by_recipe=sentence_pool_by_recipe,
                min_ctx=args.min_context,
                max_ctx=args.max_context,
                cot=not args.no_cot,
            )
            if s is not None:
                samples.append(s)
                stats["add"] += 1
                ctx_stats[f"add_ctx_{len(s['videos'])}"] += 1

        for _ in range(args.replace_per_video):
            s = build_replace_sample(
                v,
                sentence_pool_by_recipe=sentence_pool_by_recipe,
                seq_len=args.replace_context_len,
                cot=not args.no_cot,
            )
            if s is not None:
                samples.append(s)
                stats["replace"] += 1
                ctx_stats[f"replace_ctx_{len(s['videos'])}"] += 1

    if args.shuffle:
        random.shuffle(samples)

    if args.max_samples is not None and args.max_samples > 0:
        samples = samples[:args.max_samples]

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"✅ 写出完成: {len(samples)} samples -> {args.output}")
    print("📊 统计:")
    print(f"  可用视频数: {len(videos)}")
    print(f"  add: {stats['add']}")
    print(f"  replace: {stats['replace']}")
    print("  上下文视频数分布:")
    for k in sorted(ctx_stats.keys()):
        print(f"    {k}: {ctx_stats[k]}")


if __name__ == "__main__":
    main()
