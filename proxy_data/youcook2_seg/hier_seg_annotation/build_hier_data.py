#!/usr/bin/env python3
"""
build_hier_data.py — 直接从原始标注 JSON 一步构建 hier-seg 训练数据。

替代原先的 5 步流水线 (extract_frames → annotate → build_dataset → prepare_clips → prepare_data)。

支持 L1 (macro phase) / L2 (event detection) / L3_seg (free segmentation)。
所有时间戳在构建时直接转为 0-based 窗口/clip 相对坐标。

用法:
    # 三层全部构建
    python build_hier_data.py \
        --annotation-dir /path/to/annotations \
        --clip-dir-l2 /path/to/clips/L2 \
        --clip-dir-l3 /path/to/clips/L3 \
        --output-dir ./data/hier_seg \
        --levels L1 L2 L3_seg

    # 只构建 L2
    python build_hier_data.py \
        --annotation-dir /path/to/annotations \
        --clip-dir-l2 /path/to/clips/L2 \
        --output-dir ./data/hier_L2_only \
        --levels L2
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

# 添加 repo root 到 sys.path 以便 import shared
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# SCRIPT_DIR = proxy_data/youcook2_seg/hier_seg_annotation/
# → repo root 在上三级
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
PROMPTS_DIR = SCRIPT_DIR  # archetypes.py 就在同目录
PROXY_DATA_DIR = os.path.join(REPO_ROOT, "proxy_data")
ABLATION_PROMPTS_DIR = os.path.join(
    REPO_ROOT, "local_scripts", "hier_seg_ablations", "prompt_ablation",
)
for _p in (PROMPTS_DIR, PROXY_DATA_DIR, ABLATION_PROMPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from archetypes import (
    ARCHETYPE_IDS,
    TOPOLOGY_TO_DEFAULT_ARCHETYPE,
)
from prompt_variants_v4 import PROMPT_VARIANTS_V4
from shared.seg_source import (
    load_annotations,
    compute_l3_clip as _compute_l3_clip,
    get_l1_clip_path,
    get_l2_phase_clip_path,
    get_l3_clip_path,
)

PROBLEM_TYPES = {
    "L1": "temporal_seg_hier_L1",
    "L2": "temporal_seg_hier_L2",
    "L3_seg": "temporal_seg_hier_L3_seg",
}


def _make_prompt(level: str, duration: int) -> str:
    """用 V4 shot-first prompt 模板生成训练 prompt (领域无关)。"""
    template = PROMPT_VARIANTS_V4[level]["V1"]
    body = template.format(duration=duration)
    return f"Watch the following video clip carefully:\n<video>\n\n{body}"


def _resolve_archetype(ann: dict) -> str:
    """Resolve archetype from annotation, with backward compat for topology-only JSONs."""
    archetype = ann.get("archetype")
    if archetype and archetype in ARCHETYPE_IDS:
        return archetype
    topology = ann.get("topology_type", "")
    return TOPOLOGY_TO_DEFAULT_ARCHETYPE.get(topology, "tutorial")


# =====================================================================
# JSONL 写入
# =====================================================================
def write_jsonl(records: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# =====================================================================
# L1: Macro Phase Segmentation (全视频 fps 重采样，时间戳模式)
# =====================================================================
def build_l1_records(
    ann: dict,
    clip_dir_l1: str = "",
    l1_fps: int = 1,
) -> list[dict]:
    """从标注 JSON 构建 L1 训练记录 (基于真实时间戳)。

    视频输入: 原始视频以 l1_fps 帧率重采样后的 MP4 (由 prepare_clips.py 生成)。
    当 clip_dir_l1 为空时，videos 指向原始源视频 (可后续由 prepare_clips.py 处理)。
    """
    l1 = ann.get("level1")
    if not l1 or l1.get("_parse_error"):
        return []

    clip_duration = float(ann.get("clip_duration_sec") or 0)
    if clip_duration <= 0:
        return []

    phases = l1.get("macro_phases", [])
    if not phases:
        return []

    source_video = ann.get("source_video_path") or ann.get("video_path", "")
    clip_key = ann.get("clip_key", "")
    duration = int(clip_duration)
    archetype = _resolve_archetype(ann)

    # 提取 phase 边界 (真实秒数)
    spans = []
    for p in phases:
        st = p.get("start_time")
        et = p.get("end_time")
        if not isinstance(st, (int, float)) or not isinstance(et, (int, float)):
            continue
        st, et = int(st), int(et)
        if st >= et or st < 0:
            continue
        st = max(0, st)
        et = min(duration, et)
        if st < et:
            spans.append([st, et])

    if not spans:
        return []

    # 确定视频路径: 优先用 fps 重采样 clip，否则用源视频
    if clip_dir_l1:
        vp = get_l1_clip_path(clip_key, clip_dir_l1, fps=l1_fps)
    else:
        vp = source_video

    prompt = _make_prompt("L1", duration)
    answer = f"<events>{json.dumps(spans)}</events>"

    return [{
        "messages": [{"role": "user", "content": prompt}],
        "prompt": prompt,
        "answer": answer,
        "videos": [vp] if vp else [],
        "data_type": "video",
        "problem_type": PROBLEM_TYPES["L1"],
        "metadata": {
            "clip_key": clip_key,
            "clip_duration_sec": clip_duration,
            "level": 1,
            "l1_fps": l1_fps,
            "source_video_path": source_video,
            "n_phases": len(spans),
            "archetype": archetype,
            "domain_l1": ann.get("domain_l1", "other"),
            "domain_l2": ann.get("domain_l2", "other"),
            "topology": ann.get("topology_type", ""),
            "output_count": len(spans),
            "source": "annotation_json",
        },
    }]


# =====================================================================
# L2: Event Detection (per-phase 模式, 每个 L1 phase 作为输入)
# =====================================================================
def build_l2_phase_records(
    ann: dict,
    clip_dir_l2: str = "",
    min_events: int = 2,
) -> list[dict]:
    """从标注 JSON 构建 L2 训练记录 (per-phase 模式).

    每个 L1 macro_phase 生成一条记录, phase 内的 events 作为输出.
    时间戳归零到 phase 起始点.
    """
    l1 = ann.get("level1")
    l2 = ann.get("level2")
    if not l1 or not l2 or l1.get("_parse_error") or l2.get("_parse_error"):
        return []

    clip_duration = float(ann.get("clip_duration_sec") or 0)
    if clip_duration <= 0:
        return []

    phases = l1.get("macro_phases", [])
    events = l2.get("events", [])
    clip_key = ann.get("clip_key", "")
    video_path = ann.get("source_video_path") or ann.get("video_path", "")
    archetype = _resolve_archetype(ann)

    # Build phase → events mapping
    from collections import defaultdict
    phase_event_map: dict[int, list[dict]] = defaultdict(list)
    for ev in events:
        if not isinstance(ev, dict):
            continue
        pid = ev.get("parent_phase_id")
        if pid is not None:
            phase_event_map[pid].append(ev)

    records = []
    for phase in phases:
        if not isinstance(phase, dict):
            continue
        phase_id = phase.get("phase_id")
        ph_start = phase.get("start_time")
        ph_end = phase.get("end_time")

        if not isinstance(ph_start, (int, float)) or not isinstance(ph_end, (int, float)):
            continue
        ph_start, ph_end = int(ph_start), int(ph_end)
        if ph_start >= ph_end:
            continue

        duration = ph_end - ph_start

        # Collect events within this phase, clip and zero-base
        children = phase_event_map.get(phase_id, [])
        matched = []
        for ev in children:
            ev_start = ev.get("start_time")
            ev_end = ev.get("end_time")
            if not isinstance(ev_start, (int, float)) or not isinstance(ev_end, (int, float)):
                continue
            ev_start, ev_end = int(ev_start), int(ev_end)
            # Clip to phase boundary and zero-base
            s = max(ev_start, ph_start) - ph_start
            e = min(ev_end, ph_end) - ph_start
            if s < e:
                matched.append([s, e])

        if len(matched) < min_events:
            continue

        # Determine video path
        if clip_dir_l2:
            vp = get_l2_phase_clip_path(clip_key, phase_id, ph_start, ph_end, clip_dir_l2)
        else:
            vp = video_path

        prompt = _make_prompt("L2", duration)
        answer = f"<events>{json.dumps(matched)}</events>"

        records.append({
            "messages": [{"role": "user", "content": prompt}],
            "prompt": prompt,
            "answer": answer,
            "videos": [vp],
            "data_type": "video",
            "problem_type": PROBLEM_TYPES["L2"],
            "metadata": {
                "clip_key": clip_key,
                "clip_duration_sec": clip_duration,
                "level": 2,
                "phase_id": phase_id,
                "phase_start_sec": ph_start,
                "phase_end_sec": ph_end,
                "source_video_path": video_path,
                "n_events_in_phase": len(matched),
                "archetype": archetype,
                "domain_l1": ann.get("domain_l1", "other"),
                "domain_l2": ann.get("domain_l2", "other"),
                "topology": ann.get("topology_type", ""),
                "output_count": len(matched),
                "source": "annotation_json",
            },
        })

    return records


# =====================================================================
# L2 Full-Video: Event Detection (全视频模式, 每 annotation 一条记录)
# =====================================================================
def build_l2_fullvideo_records(
    ann: dict,
    clip_dir_l1: str = "",
    l1_fps: int = 1,
    min_events: int = 3,
) -> list[dict]:
    """从标注 JSON 构建 L2 训练记录 (全视频模式).

    全视频作为输入, 输出全部 events 的绝对时间戳 (0-based from video start).
    与 L1 使用相同的视频文件 (1fps 重采样 clip).
    每个 annotation 产生一条记录, 而非 per-phase N 条.
    """
    l2 = ann.get("level2")
    if not l2 or l2.get("_parse_error"):
        return []

    clip_duration = float(ann.get("clip_duration_sec") or 0)
    if clip_duration <= 0:
        return []

    events = l2.get("events", [])
    clip_key = ann.get("clip_key", "")
    source_video = ann.get("source_video_path") or ann.get("video_path", "")
    duration = int(clip_duration)
    archetype = _resolve_archetype(ann)

    # 收集所有 events 的绝对时间戳, 过滤无效项
    spans = []
    for ev in sorted(events, key=lambda e: e.get("start_time", 0)):
        if not isinstance(ev, dict):
            continue
        st = ev.get("start_time")
        et = ev.get("end_time")
        if not isinstance(st, (int, float)) or not isinstance(et, (int, float)):
            continue
        st, et = int(st), int(et)
        st = max(0, st)
        et = min(duration, et)
        if st < et:
            spans.append([st, et])

    if len(spans) < min_events:
        return []

    # 视频路径: 复用 L1 clip (1fps full-video resample)
    if clip_dir_l1:
        vp = get_l1_clip_path(clip_key, clip_dir_l1, fps=l1_fps)
    else:
        vp = source_video

    prompt = _make_prompt("L2", duration)
    answer = f"<events>{json.dumps(spans)}</events>"

    return [{
        "messages": [{"role": "user", "content": prompt}],
        "prompt": prompt,
        "answer": answer,
        "videos": [vp] if vp else [],
        "data_type": "video",
        "problem_type": PROBLEM_TYPES["L2"],
        "metadata": {
            "clip_key": clip_key,
            "clip_duration_sec": clip_duration,
            "level": 2,
            "l2_mode": "full",
            "l2_fps": l1_fps,
            "source_video_path": source_video,
            "n_events": len(spans),
            "archetype": archetype,
            "domain_l1": ann.get("domain_l1", "other"),
            "domain_l2": ann.get("domain_l2", "other"),
            "topology": ann.get("topology_type", ""),
            "output_count": len(spans),
            "source": "annotation_json",
        },
    }]


# =====================================================================
# L3 Seg: Free Segmentation (无 query, shot-first state-change)
# =====================================================================
def build_l3_seg_records(
    ann: dict,
    clip_dir_l3: str = "",
    min_actions: int = 2,
) -> list[dict]:
    """从标注 JSON 构建 L3 segmentation (无 query) 训练记录。

    标注格式 (nested):
      level3.grounding_results[] = {
          event_id, parent_phase_id, event_start, event_end,
          sub_actions: [{action_id, start_time, end_time, sub_action}, ...]
      }
    每个 grounding_result 对应一个 L2 event, sub_actions 是该 event 内的细粒度切分。
    """
    l3 = ann.get("level3")
    if not l3 or l3.get("_parse_error"):
        return []

    clip_duration = float(ann.get("clip_duration_sec") or 0)
    if clip_duration <= 0:
        return []

    video_path = ann.get("source_video_path") or ann.get("video_path", "")
    clip_key = ann.get("clip_key", "")
    grounding_results = l3.get("grounding_results", [])
    archetype = _resolve_archetype(ann)

    records = []
    for gr in grounding_results:
        if not isinstance(gr, dict):
            continue

        event_id = gr.get("event_id")
        ev_start = gr.get("event_start")
        ev_end = gr.get("event_end")
        sub_actions = gr.get("sub_actions", [])

        if event_id is None or ev_start is None or ev_end is None:
            continue
        ev_start, ev_end = int(ev_start), int(ev_end)
        if ev_start >= ev_end:
            continue

        # 过滤无效 sub_actions 并排序
        valid_subs = sorted(
            [sa for sa in sub_actions if isinstance(sa, dict)
             and sa.get("start_time") is not None
             and sa.get("end_time") is not None],
            key=lambda sa: sa.get("start_time", 0),
        )
        if len(valid_subs) < min_actions:
            continue

        clip_start, clip_end, duration = _compute_l3_clip(
            ev_start, ev_end, int(clip_duration),
        )

        # 构建时间段 (0-based relative to clip_start, chronological)
        spans = []
        for sa in valid_subs:
            st = max(0, int(sa["start_time"]) - clip_start)
            et = min(duration, int(sa["end_time"]) - clip_start)
            if st < et:
                spans.append([st, et])

        if len(spans) < min_actions:
            continue

        if clip_dir_l3:
            vp = get_l3_clip_path(clip_key, event_id, clip_start, clip_end, clip_dir_l3)
        else:
            vp = video_path

        prompt = _make_prompt("L3", duration)
        answer = f"<events>{json.dumps(spans)}</events>"

        records.append({
            "messages": [{"role": "user", "content": prompt}],
            "prompt": prompt,
            "answer": answer,
            "videos": [vp],
            "data_type": "video",
            "problem_type": PROBLEM_TYPES["L3_seg"],
            "metadata": {
                "clip_key": clip_key,
                "clip_duration_sec": clip_duration,
                "level": 3,
                "parent_event_id": event_id,
                "event_start_sec": ev_start,
                "event_end_sec": ev_end,
                "clip_start_sec": clip_start,
                "clip_end_sec": clip_end,
                "source_video_path": video_path,
                "n_actions": len(spans),
                "archetype": archetype,
                "domain_l1": ann.get("domain_l1", "other"),
                "domain_l2": ann.get("domain_l2", "other"),
                "topology": ann.get("topology_type", ""),
                "output_count": len(spans),
                "source": "annotation_json",
            },
        })

    return records


# =====================================================================
# 领域均衡采样
# =====================================================================
def _quota_sample(records: list[dict], target: int, key: str) -> list[dict]:
    """按 metadata[key] 等额分组，超额的组内按 output_count 降序保留。

    两轮分配:
      Round 1 — 不足 quota 的域全量保留，剩余名额累计给大域
      Round 2 — 大域按名额截取 (output_count 降序, 优先保留标注多的记录)
    """
    from collections import defaultdict

    if len(records) <= target:
        return records

    by_key: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        k = rec.get("metadata", {}).get(key, "other")
        by_key[k].append(rec)

    n_groups = len(by_key)
    base_quota = target // n_groups
    remaining = target

    small, large = {}, {}
    for k, recs in by_key.items():
        if len(recs) <= base_quota:
            small[k] = recs
            remaining -= len(recs)
        else:
            large[k] = recs

    sampled: list[dict] = list(r for recs in small.values() for r in recs)

    if large:
        q_large = remaining // len(large)
        extra = remaining - q_large * len(large)
        for idx, (k, recs) in enumerate(sorted(large.items(), key=lambda x: -len(x[1]))):
            q = q_large + (1 if idx < extra else 0)
            recs_sorted = sorted(recs, key=lambda r: -r.get("metadata", {}).get("output_count", 0))
            sampled.extend(recs_sorted[:q])

    return sampled


def balanced_sample(
    level_records: dict[str, list[dict]],
    target_per_level: int,
) -> dict[str, list[dict]]:
    """两级领域均衡采样 (domain_l1 → domain_l2)。

    策略 (每层级独立):
      Step 1 — 按 domain_l1 等额分组, 各 L1 域获得相同 budget
      Step 2 — 在每个 L1 域内, 按 domain_l2 等额细分 (同 _quota_sample 逻辑)
      优先保留 output_count 大 (标注数多) 的记录
    """
    from collections import defaultdict

    result: dict[str, list[dict]] = {}

    for lv, records in level_records.items():
        if not records or target_per_level <= 0:
            result[lv] = records
            continue

        if len(records) <= target_per_level:
            result[lv] = records
            print(f"  {lv}: {len(records)} records (< target, keep all)")
            continue

        # Step 1: 按 domain_l1 分组并等额分配
        by_l1: dict[str, list[dict]] = defaultdict(list)
        for rec in records:
            l1 = rec.get("metadata", {}).get("domain_l1", "other")
            by_l1[l1].append(rec)

        n_l1 = len(by_l1)
        base_l1 = target_per_level // n_l1
        remaining_l1 = target_per_level

        small_l1, large_l1 = {}, {}
        for l1, recs in by_l1.items():
            if len(recs) <= base_l1:
                small_l1[l1] = recs
                remaining_l1 -= len(recs)
            else:
                large_l1[l1] = recs

        sampled: list[dict] = list(r for recs in small_l1.values() for r in recs)

        if large_l1:
            q_l1 = remaining_l1 // len(large_l1)
            extra_l1 = remaining_l1 - q_l1 * len(large_l1)

            for idx, (l1, recs) in enumerate(sorted(large_l1.items(), key=lambda x: -len(x[1]))):
                budget = q_l1 + (1 if idx < extra_l1 else 0)
                # Step 2: 在该 L1 域内按 domain_l2 二级均衡
                l2_sampled = _quota_sample(recs, budget, "domain_l2")
                sampled.extend(l2_sampled)

        # 统计并打印分布
        l1_dist = defaultdict(int)
        for rec in sampled:
            l1_dist[rec.get("metadata", {}).get("domain_l1", "other")] += 1
        dist_str = " | ".join(f"{k}:{v}" for k, v in sorted(l1_dist.items()))
        print(f"  {lv}: balanced {len(records)} → {len(sampled)}  [{dist_str}]")

        result[lv] = sampled

    return result


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="从原始标注 JSON 直接构建 hier-seg 训练数据 (L1/L2/L3_seg)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--annotation-dir", required=True,
                        help="标注 JSON 目录 (annotations/*.json)")
    parser.add_argument("--clip-dir-l1", default="",
                        help="L1 fps-重采样 clips 目录 (clips/L1/), 留空则用原始视频路径")
    parser.add_argument("--l1-fps", type=int, default=1,
                        help="L1 视频重采样帧率 (默认: 1fps)")
    parser.add_argument("--clip-dir-l2", default="",
                        help="L2 clips 目录 (clips/L2/), 留空则用原始视频路径")
    parser.add_argument("--clip-dir-l3", default="",
                        help="L3 clips 目录 (clips/L3/), 留空则用原始视频路径")
    parser.add_argument("--output-dir", required=True,
                        help="输出目录")
    parser.add_argument("--levels", nargs="+", required=True,
                        choices=["L1", "L2", "L3_seg"],
                        help="要构建的层级列表")
    parser.add_argument("--total-val", type=int, default=200,
                        help="总验证集样本数")
    parser.add_argument("--train-per-level", type=int, default=-1,
                        help="每层最多 train 条数 (-1 = 无限制)")
    # ---- per-level 筛选 (与 visualize_annotations.py FilterConfig 对齐) ----
    parser.add_argument("--l1-min-phases", type=int, default=0,
                        help="L1 最少 phase 数 (default=0 不限制)")
    parser.add_argument("--l1-max-phases", type=int, default=999,
                        help="L1 最多 phase 数 (default=999 不限制)")
    parser.add_argument("--l2-min-events", type=int, default=2,
                        help="L2 每 phase 最少事件数")
    parser.add_argument("--l2-max-events", type=int, default=999,
                        help="L2 每 phase 最多事件数 (default=999 不限制)")
    parser.add_argument("--l3-min-actions", type=int, default=3,
                        help="L3 每事件最少 grounding results 数")
    parser.add_argument("--l3-max-actions", type=int, default=999,
                        help="L3 每事件最多 grounding results 数 (default=999 不限制)")
    parser.add_argument("--complete-only", action="store_true",
                        help="仅处理 L1+L2+L3 均完整的 clip")
    parser.add_argument("--balance-per-level", type=int, default=-1,
                        help="每层级领域均衡采样目标数 (-1 = 不做均衡采样)")
    parser.add_argument("--l2-mode", choices=["phase", "full"], default="phase",
                        help="L2 输入模式: phase=每 phase 一条记录(子clip), full=全视频一条记录")
    parser.add_argument("--use-hint", action="store_true",
                        help="(保留参数, 暂未实现) 在 prompt 中加入上一层标注作为 hint")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    ann_dir = Path(args.annotation_dir)
    if not ann_dir.exists():
        print(f"ERROR: annotation-dir not found: {ann_dir}")
        return

    # 使用 shared 统一加载接口
    ann_list = load_annotations(ann_dir, complete_only=args.complete_only)
    print(f"Found {len(ann_list)} annotation files"
          + (" (complete-only filtered)" if args.complete_only else ""))

    # 按层级收集记录
    level_records: dict[str, list[dict]] = {lv: [] for lv in args.levels}
    stats = {lv: {"clips": 0, "records": 0} for lv in args.levels}

    for ann in ann_list:
        for lv in args.levels:
            if lv == "L1":
                recs = build_l1_records(ann, args.clip_dir_l1, args.l1_fps)
            elif lv == "L2":
                if args.l2_mode == "full":
                    recs = build_l2_fullvideo_records(
                        ann, args.clip_dir_l1, args.l1_fps, args.l2_min_events,
                    )
                else:
                    recs = build_l2_phase_records(
                        ann, args.clip_dir_l2, args.l2_min_events,
                    )
            elif lv == "L3_seg":
                recs = build_l3_seg_records(ann, args.clip_dir_l3, args.l3_min_actions)
            else:
                continue

            if recs:
                stats[lv]["clips"] += 1
                stats[lv]["records"] += len(recs)
                level_records[lv].extend(recs)

    # 打印提取统计
    print(f"\n=== Extraction Stats (raw) ===")
    for lv in args.levels:
        print(f"  {lv}: {stats[lv]['clips']} clips, {stats[lv]['records']} records")

    # ---- per-level min/max 筛选 (与 visualize FilterConfig 对齐) ----
    filter_map = {
        "L1":     (args.l1_min_phases, args.l1_max_phases),
        "L2":     (args.l2_min_events, args.l2_max_events),
        "L3_seg": (args.l3_min_actions, args.l3_max_actions),
    }
    any_filtered = False
    for lv in args.levels:
        lo, hi = filter_map.get(lv, (0, 999))
        if lo > 0 or hi < 999:
            before = len(level_records[lv])
            level_records[lv] = [
                r for r in level_records[lv]
                if lo <= r.get("metadata", {}).get("output_count", 0) <= hi
            ]
            after = len(level_records[lv])
            if before != after:
                print(f"  {lv}: filtered {before} → {after} (output_count ∈ [{lo}, {hi}])")
                any_filtered = True
    if not any_filtered:
        print("  (no additional filtering applied)")

    # 领域均衡采样 (在 train/val 分割之前)
    if args.balance_per_level > 0:
        print(f"\n=== Balanced Sampling (target={args.balance_per_level}/level) ===")
        level_records = balanced_sample(level_records, args.balance_per_level)

    # 合并、分割 train/val
    train_per_level = None if args.train_per_level < 0 else args.train_per_level
    n_levels = len(args.levels)
    val_per_level = max(1, args.total_val // n_levels)

    all_train = []
    all_val = []

    for lv in args.levels:
        records = level_records[lv]
        rng.shuffle(records)

        n_val = min(val_per_level, len(records) // 5)
        val = records[:n_val]
        train_pool = records[n_val:]

        if train_per_level is not None:
            train = train_pool[:train_per_level]
        else:
            train = train_pool

        all_val.extend(val)
        all_train.extend(train)

        print(f"  {lv}: {len(train)} train + {len(val)} val")

    rng.shuffle(all_train)
    rng.shuffle(all_val)

    # 输出
    train_path = os.path.join(args.output_dir, "train.jsonl")
    val_path = os.path.join(args.output_dir, "val.jsonl")
    write_jsonl(all_train, train_path)
    write_jsonl(all_val, val_path)

    print(f"\n=== Output ===")
    print(f"  Train: {len(all_train)}")
    print(f"  Val:   {len(all_val)}")
    print(f"  Dir:   {args.output_dir}")

    if all_train:
        ex = all_train[0]
        print(f"\n  --- Example (first record) ---")
        print(f"  problem_type: {ex['problem_type']}")
        print(f"  video: {ex['videos'][0] if ex['videos'] else 'N/A'}")
        print(f"  answer: {ex['answer'][:200]}")


if __name__ == "__main__":
    main()
