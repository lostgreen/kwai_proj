#!/usr/bin/env python3
"""
build_aot_from_seg.py — 从三层分割标注 JSON 构建 AOT (Action Ordering Task) MCQ 数据。

使用 prepare_all_clips.py 产出的原子 clips，按不同顺序拼接后作为视频输入，
消除时长差异和未标注间隙导致的 shortcut。

六种任务类型（3 粒度 × 2 方向）:

  L1 (phase) — 原子 clips: 各 phase clip
  - seg_aot_phase_v2t:  给 forward-concat phase 视频，判断哪个阶段列表顺序正确   (A/B/C)
  - seg_aot_phase_t2v:  给 fwd/shuf/rev 阶段列表，从 3 个拼接视频中选出匹配的   (A/B/C, round-robin)

  L2 (event) — 原子 clips: 各 event clip (按 parent_phase_id 分组)
  - seg_aot_event_v2t:  给 forward-concat event 视频，判断哪个事件列表顺序正确   (A/B/C)
  - seg_aot_event_t2v:  给 fwd/shuf/rev 事件列表，从 3 个拼接视频中选出匹配的   (A/B/C, round-robin)

  L3 (action) — 原子 clips: 各 action clip (按 parent_event_id 分组)
  - seg_aot_action_v2t: 给 forward-concat action 视频，判断哪个动作列表顺序正确  (A/B/C)
  - seg_aot_action_t2v: 给 fwd/shuf/rev 动作列表，从 3 个拼接视频中选出匹配的   (A/B/C, round-robin)

流程:
    1. 加载标注 → 收集 concat jobs + group metadata
    2. 并行 ffmpeg concat (demuxer, no re-encode)
    3. 从 group metadata 构建 JSONL records

用法:
    python proxy_data/youcook2_seg/temporal_aot/build_aot_from_seg.py \\
        --annotation-dir /path/to/annotations \\
        --clip-dir /path/to/clips \\
        --output-dir /path/to/output \\
        --tasks phase_v2t phase_t2v action_v2t action_t2v event_v2t event_t2v \\
        --complete-only
"""

import argparse
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# 添加 proxy_data 父目录到 sys.path 以便 import shared
_PROXY_DATA_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROXY_DATA_DIR not in sys.path:
    sys.path.insert(0, _PROXY_DATA_DIR)

from shared.seg_source import (  # noqa: E402
    load_annotations,
    get_l1_phase_atomic_path,
    get_l2_event_atomic_path,
    get_l3_action_atomic_path,
)


def _balanced_sample_by_domain(
    records: list[dict],
    budget: int,
    rng: random.Random,
) -> list[dict]:
    """按 domain_l1 → domain_l2 两级嵌套均衡采样。

    策略:
      1. 按 domain_l1 分组，均分 budget 给各 L1 域
      2. 各 L1 域内再按 domain_l2 均衡采样
      3. 不足 quota 的域全部保留，多余名额重分配
    """
    if len(records) <= budget:
        rng.shuffle(records)
        return records

    # ---- L1 层均衡 ----
    by_l1: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        meta = rec.get("metadata", {})
        by_l1[meta.get("domain_l1", "other")].append(rec)

    l1_quotas = _distribute_quota(budget, {d: len(recs) for d, recs in by_l1.items()})

    sampled: list[dict] = []
    for d1, l1_recs in by_l1.items():
        q1 = l1_quotas.get(d1, 0)
        if q1 <= 0:
            continue
        if len(l1_recs) <= q1:
            rng.shuffle(l1_recs)
            sampled.extend(l1_recs)
            continue

        # ---- L2 层均衡 (within this L1 domain) ----
        by_l2: dict[str, list[dict]] = defaultdict(list)
        for rec in l1_recs:
            meta = rec.get("metadata", {})
            by_l2[meta.get("domain_l2", "other")].append(rec)

        l2_quotas = _distribute_quota(q1, {d: len(recs) for d, recs in by_l2.items()})
        for d2, l2_recs in by_l2.items():
            q2 = l2_quotas.get(d2, 0)
            rng.shuffle(l2_recs)
            sampled.extend(l2_recs[:q2])

    return sampled


def _distribute_quota(
    total: int,
    group_sizes: dict[str, int],
) -> dict[str, int]:
    """均分 total 到各 group，不足均额的 group 全部保留，多余名额重分配。"""
    if not group_sizes:
        return {}

    n = len(group_sizes)
    base = total // n
    remaining = total

    small: dict[str, int] = {}
    large: list[str] = []
    for g, sz in group_sizes.items():
        if sz <= base:
            small[g] = sz
            remaining -= sz
        else:
            large.append(g)

    result = dict(small)
    if large:
        q = remaining // len(large)
        extra = remaining - q * len(large)
        for idx, g in enumerate(sorted(large)):
            result[g] = q + (1 if idx < extra else 0)

    return result

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ANSWER_LETTERS = ["A", "B", "C"]


# =====================================================================
# ffmpeg concat utility
# =====================================================================
def ffmpeg_concat_clips(
    clip_paths: list[str],
    output_path: str,
    overwrite: bool = False,
) -> bool:
    """Concatenate atomic clips using ffmpeg concat demuxer (stream copy, no re-encode).

    All clips must share codec, resolution, and frame rate (guaranteed when
    produced by prepare_all_clips.py from the same source video).
    """
    if not clip_paths:
        return False
    if os.path.exists(output_path) and not overwrite:
        return True

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if len(clip_paths) == 1:
        shutil.copy2(clip_paths[0], output_path)
        return True

    list_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False,
            dir=os.path.dirname(output_path),
        ) as f:
            for p in clip_paths:
                escaped = p.replace("'", "'\\''")
                f.write(f"file '{escaped}'\n")
            list_path = f.name

        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", list_path, "-c", "copy", "-an", output_path],
            check=True, capture_output=True, timeout=120,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        log.warning("Concat failed for %s: %s", output_path, exc)
        if os.path.exists(output_path):
            os.unlink(output_path)
        return False
    finally:
        if list_path and os.path.exists(list_path):
            os.unlink(list_path)


# =====================================================================
# Concat path naming
# =====================================================================
def _phase_concat_path(clip_key: str, order: str, concat_dir: str) -> str:
    return os.path.join(concat_dir, "phase", f"{clip_key}_{order}.mp4")


def _event_concat_path(clip_key: str, phase_id: int, order: str, concat_dir: str) -> str:
    return os.path.join(concat_dir, "event", f"{clip_key}_ph{phase_id}_{order}.mp4")


def _action_concat_path(clip_key: str, event_id: int, order: str, concat_dir: str) -> str:
    return os.path.join(concat_dir, "action", f"{clip_key}_ev{event_id}_{order}.mp4")


# =====================================================================
# Prompt templates
# =====================================================================
_PHASE_V2T_PROMPT = """\
Watch the video carefully.

Which numbered list correctly describes the temporal order of high-level phases in this video?

A.
{option_a}

B.
{option_b}

C.
{option_c}

Think step by step inside <think></think> tags, then provide your final answer \
(A, B, or C) inside <answer></answer> tags."""

_PHASE_T2V_PROMPT = """\
Here are three video clips (Clip A, Clip B, and Clip C).

The high-level phases below occurred in this exact order:
{forward_list}

Which clip (A, B, or C) shows these phases in the listed order?

Think step by step inside <think></think> tags, then provide your final answer \
(A, B, or C) inside <answer></answer> tags."""

_EVENT_V2T_PROMPT = """\
Watch the video carefully.

Which numbered list correctly describes the temporal order of events visible in this video?

A.
{option_a}

B.
{option_b}

C.
{option_c}

Think step by step inside <think></think> tags, then provide your final answer \
(A, B, or C) inside <answer></answer> tags."""

_EVENT_T2V_PROMPT = """\
Here are three video clips (Clip A, Clip B, and Clip C).

The events below occurred in this exact order:
{forward_list}

Which clip (A, B, or C) shows these events in the listed order?

Think step by step inside <think></think> tags, then provide your final answer \
(A, B, or C) inside <answer></answer> tags."""

_ACTION_V2T_PROMPT = """\
Watch the video clip carefully.

Which numbered list correctly describes the temporal order of atomic actions in this video?

A.
{option_a}

B.
{option_b}

C.
{option_c}

Think step by step inside <think></think> tags, then provide your final answer \
(A, B, or C) inside <answer></answer> tags."""

_ACTION_T2V_PROMPT = """\
Here are three video clips (Clip A, Clip B, and Clip C).

The atomic actions below were performed in this exact order:
{forward_list}

Which clip (A, B, or C) shows these actions in the listed order?

Think step by step inside <think></think> tags, then provide your final answer \
(A, B, or C) inside <answer></answer> tags."""


def _format_list(items: list[str]) -> str:
    return "\n".join(f"   {i + 1}. {item}" for i, item in enumerate(items))


# =====================================================================
# JSONL IO
# =====================================================================
def write_jsonl(records: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# =====================================================================
# Phase 1: Collect concat jobs + group metadata
# =====================================================================
def _collect_phase_groups(
    ann: dict,
    clip_dir: str,
    concat_dir: str,
    need_v2t: bool,
    need_t2v: bool,
    min_phases: int,
    max_phases: int,
    complete_only: bool,
    rng: random.Random,
) -> tuple[dict | None, list[tuple[list[str], str]]]:
    """Collect phase-level concat jobs.

    Returns (group_info, [(clip_paths, output_path), ...]) or (None, []).
    """
    l1 = ann.get("level1")
    if not l1 or l1.get("_parse_error"):
        return None, []

    phases = sorted(
        [p for p in l1.get("macro_phases", [])
         if isinstance(p, dict) and isinstance(p.get("start_time"), (int, float))
         and p.get("phase_name", "").strip()],
        key=lambda p: p["start_time"],
    )
    if len(phases) < min_phases or len(phases) > max_phases:
        return None, []

    clip_key = ann.get("clip_key", "")
    fwd_paths = []
    for ph in phases:
        path = get_l1_phase_atomic_path(
            clip_key, ph["phase_id"],
            int(ph["start_time"]), int(ph["end_time"]), clip_dir,
        )
        if complete_only and not os.path.exists(path):
            return None, []
        fwd_paths.append(path)

    phase_names = [p["phase_name"].strip() for p in phases]
    total_dur = sum(int(p["end_time"]) - int(p["start_time"]) for p in phases)

    jobs: list[tuple[list[str], str]] = []
    fwd_out = _phase_concat_path(clip_key, "fwd", concat_dir)
    if need_v2t or need_t2v:
        jobs.append((fwd_paths, fwd_out))

    shuf_out = rev_out = None
    shuf_names = None
    if need_t2v:
        # Shuffled order
        indices = list(range(len(phases)))
        shuf_idx = indices[:]
        for _ in range(20):
            rng.shuffle(shuf_idx)
            if shuf_idx != indices:
                break
        if shuf_idx == indices:
            return None, []  # can't produce distinct shuffle

        shuf_out = _phase_concat_path(clip_key, "shuf", concat_dir)
        jobs.append(([fwd_paths[i] for i in shuf_idx], shuf_out))
        shuf_names = [phase_names[i] for i in shuf_idx]

        rev_out = _phase_concat_path(clip_key, "rev", concat_dir)
        jobs.append((list(reversed(fwd_paths)), rev_out))

    info = {
        "clip_key": clip_key,
        "phase_names": phase_names,
        "n_phases": len(phases),
        "total_duration": total_dur,
        "fwd_video": fwd_out,
        "shuf_video": shuf_out,
        "rev_video": rev_out,
        "shuf_names": shuf_names,
        "domain_l1": ann.get("domain_l1", "other"),
        "domain_l2": ann.get("domain_l2", "other"),
    }
    return info, jobs


def _collect_event_groups(
    ann: dict,
    clip_dir: str,
    concat_dir: str,
    need_v2t: bool,
    need_t2v: bool,
    min_events: int,
    max_events: int,
    complete_only: bool,
    rng: random.Random,
) -> tuple[list[dict], list[tuple[list[str], str]]]:
    """Collect event-level concat jobs (one group per phase)."""
    l1 = ann.get("level1")
    l2 = ann.get("level2")
    if not l1 or not l2 or l2.get("_parse_error"):
        return [], []

    clip_key = ann.get("clip_key", "")
    phases = l1.get("macro_phases", [])
    events = l2.get("events", [])

    groups: list[dict] = []
    jobs: list[tuple[list[str], str]] = []

    for phase in phases:
        if not isinstance(phase, dict):
            continue
        ph_id = phase.get("phase_id")

        # Gather child events sorted by start_time
        child_events = sorted(
            [ev for ev in events
             if isinstance(ev, dict) and ev.get("parent_phase_id") == ph_id
             and ev.get("instruction", "").strip()
             and isinstance(ev.get("start_time"), (int, float))],
            key=lambda ev: ev["start_time"],
        )
        if len(child_events) < min_events or len(child_events) > max_events:
            continue

        fwd_paths = []
        for ev in child_events:
            path = get_l2_event_atomic_path(
                clip_key, ev["event_id"],
                int(ev["start_time"]), int(ev["end_time"]), clip_dir,
            )
            if complete_only and not os.path.exists(path):
                fwd_paths = None
                break
            fwd_paths.append(path)
        if fwd_paths is None:
            continue

        instructions = [ev["instruction"].strip() for ev in child_events]
        total_dur = sum(int(ev["end_time"]) - int(ev["start_time"]) for ev in child_events)

        fwd_out = _event_concat_path(clip_key, ph_id, "fwd", concat_dir)
        if need_v2t or need_t2v:
            jobs.append((fwd_paths, fwd_out))

        shuf_out = rev_out = None
        shuf_instructions = None
        if need_t2v:
            indices = list(range(len(child_events)))
            shuf_idx = indices[:]
            for _ in range(20):
                rng.shuffle(shuf_idx)
                if shuf_idx != indices:
                    break
            if shuf_idx == indices:
                continue

            shuf_out = _event_concat_path(clip_key, ph_id, "shuf", concat_dir)
            jobs.append(([fwd_paths[i] for i in shuf_idx], shuf_out))
            shuf_instructions = [instructions[i] for i in shuf_idx]

            rev_out = _event_concat_path(clip_key, ph_id, "rev", concat_dir)
            jobs.append((list(reversed(fwd_paths)), rev_out))

        groups.append({
            "clip_key": clip_key,
            "phase_id": ph_id,
            "instructions": instructions,
            "n_events": len(child_events),
            "total_duration": total_dur,
            "fwd_video": fwd_out,
            "shuf_video": shuf_out,
            "rev_video": rev_out,
            "shuf_instructions": shuf_instructions,
            "domain_l1": ann.get("domain_l1", "other"),
            "domain_l2": ann.get("domain_l2", "other"),
        })

    return groups, jobs


def _collect_action_groups(
    ann: dict,
    clip_dir: str,
    concat_dir: str,
    need_v2t: bool,
    need_t2v: bool,
    min_actions: int,
    max_actions: int,
    complete_only: bool,
    rng: random.Random,
) -> tuple[list[dict], list[tuple[list[str], str]]]:
    """Collect action-level concat jobs (one group per event)."""
    l2 = ann.get("level2")
    l3 = ann.get("level3")
    if not l2 or not l3 or l3.get("_parse_error"):
        return [], []

    clip_key = ann.get("clip_key", "")
    events = l2.get("events", [])
    all_results = l3.get("grounding_results", [])

    groups: list[dict] = []
    jobs: list[tuple[list[str], str]] = []

    for event in events:
        if not isinstance(event, dict):
            continue
        ev_id = event.get("event_id")

        child_actions = sorted(
            [r for r in all_results
             if isinstance(r, dict) and r.get("parent_event_id") == ev_id
             and r.get("sub_action", "").strip()
             and isinstance(r.get("start_time"), (int, float))],
            key=lambda r: r["start_time"],
        )
        if len(child_actions) < min_actions or len(child_actions) > max_actions:
            continue

        fwd_paths = []
        for act in child_actions:
            path = get_l3_action_atomic_path(
                clip_key, act["action_id"], ev_id,
                int(act["start_time"]), int(act["end_time"]), clip_dir,
            )
            if complete_only and not os.path.exists(path):
                fwd_paths = None
                break
            fwd_paths.append(path)
        if fwd_paths is None:
            continue

        sub_actions = [r["sub_action"].strip() for r in child_actions]
        total_dur = sum(int(r["end_time"]) - int(r["start_time"]) for r in child_actions)

        fwd_out = _action_concat_path(clip_key, ev_id, "fwd", concat_dir)
        if need_v2t or need_t2v:
            jobs.append((fwd_paths, fwd_out))

        shuf_out = rev_out = None
        shuf_actions = None
        if need_v2t or need_t2v:
            # Shuffled order
            indices = list(range(len(child_actions)))
            shuf_idx = indices[:]
            for _ in range(20):
                rng.shuffle(shuf_idx)
                if shuf_idx != indices:
                    break
            if shuf_idx == indices:
                continue  # can't produce distinct shuffle

            shuf_out = _action_concat_path(clip_key, ev_id, "shuf", concat_dir)
            jobs.append(([fwd_paths[i] for i in shuf_idx], shuf_out))
            shuf_actions = [sub_actions[i] for i in shuf_idx]

            rev_out = _action_concat_path(clip_key, ev_id, "rev", concat_dir)
            jobs.append((list(reversed(fwd_paths)), rev_out))

        groups.append({
            "clip_key": clip_key,
            "event_id": ev_id,
            "sub_actions": sub_actions,
            "n_actions": len(child_actions),
            "total_duration": total_dur,
            "fwd_video": fwd_out,
            "shuf_video": shuf_out,
            "rev_video": rev_out,
            "shuf_actions": shuf_actions,
            "domain_l1": ann.get("domain_l1", "other"),
            "domain_l2": ann.get("domain_l2", "other"),
        })

    return groups, jobs


# =====================================================================
# Phase 3: Build records from group metadata
# =====================================================================
def _build_phase_v2t(info: dict, rng: random.Random) -> dict | None:
    names = info["phase_names"]
    shuffled = names[:]
    for _ in range(20):
        rng.shuffle(shuffled)
        if shuffled != names:
            break
    if shuffled == names:
        return None

    reversed_names = list(reversed(names))
    options = [("forward", names), ("shuffled", shuffled), ("reversed", reversed_names)]
    rng.shuffle(options)
    correct = ANSWER_LETTERS[[o[0] for o in options].index("forward")]

    body = _PHASE_V2T_PROMPT.format(
        option_a=_format_list(options[0][1]),
        option_b=_format_list(options[1][1]),
        option_c=_format_list(options[2][1]),
    )
    prompt = f"<video>\n\n{body}"

    return {
        "messages": [{"role": "user", "content": prompt}],
        "prompt": prompt,
        "answer": correct,
        "videos": [info["fwd_video"]],
        "data_type": "video",
        "problem_type": "seg_aot_phase_v2t",
        "metadata": {
            "clip_key": info["clip_key"],
            "total_duration_sec": info["total_duration"],
            "n_phases": info["n_phases"],
            "forward_descriptions": names,
            "option_a_type": options[0][0],
            "option_b_type": options[1][0],
            "option_c_type": options[2][0],
            "domain_l1": info["domain_l1"],
            "domain_l2": info["domain_l2"],
            "source": "seg_annotation",
            "level": 1,
            "l1_fps": 1,
        },
    }


_T2V_TEXT_ORDERS = ["forward", "shuffled", "reversed"]


def _build_phase_t2v(info: dict, rng: random.Random, text_order: str = "forward") -> dict | None:
    """Build one T2V record using *text_order* as the ground-truth text ordering."""
    if not info.get("shuf_video") or not info.get("rev_video"):
        return None

    names = info["phase_names"]
    text_map = {
        "forward": names,
        "shuffled": info.get("shuf_names") or names,
        "reversed": list(reversed(names)),
    }
    video_map = {
        "forward": info["fwd_video"],
        "shuffled": info["shuf_video"],
        "reversed": info["rev_video"],
    }

    text_list = text_map[text_order]
    fwd_list = _format_list(text_list)

    video_opts = [
        ("forward", video_map["forward"]),
        ("shuffled", video_map["shuffled"]),
        ("reversed", video_map["reversed"]),
    ]
    rng.shuffle(video_opts)
    correct = ANSWER_LETTERS[[o[0] for o in video_opts].index(text_order)]
    videos = [v for _, v in video_opts]

    tags = "".join("<video>" for _ in videos)
    body = _PHASE_T2V_PROMPT.format(forward_list=fwd_list)
    prompt = f"{tags}\n\n{body}"

    return {
        "messages": [{"role": "user", "content": prompt}],
        "prompt": prompt,
        "answer": correct,
        "videos": videos,
        "data_type": "video",
        "problem_type": "seg_aot_phase_t2v",
        "metadata": {
            "clip_key": info["clip_key"],
            "total_duration_sec": info["total_duration"],
            "n_phases": info["n_phases"],
            "forward_descriptions": names,
            "text_order": text_order,
            "video_a_type": video_opts[0][0],
            "video_b_type": video_opts[1][0],
            "video_c_type": video_opts[2][0],
            "correct_position": correct,
            "domain_l1": info["domain_l1"],
            "domain_l2": info["domain_l2"],
            "source": "seg_annotation",
            "level": 1,
            "l1_fps": 1,
        },
    }


def _build_event_v2t(info: dict, rng: random.Random) -> dict | None:
    instr = info["instructions"]
    shuffled = instr[:]
    for _ in range(20):
        rng.shuffle(shuffled)
        if shuffled != instr:
            break
    if shuffled == instr:
        return None

    reversed_instr = list(reversed(instr))
    options = [("forward", instr), ("shuffled", shuffled), ("reversed", reversed_instr)]
    rng.shuffle(options)
    correct = ANSWER_LETTERS[[o[0] for o in options].index("forward")]

    body = _EVENT_V2T_PROMPT.format(
        option_a=_format_list(options[0][1]),
        option_b=_format_list(options[1][1]),
        option_c=_format_list(options[2][1]),
    )
    prompt = f"<video>\n\n{body}"

    return {
        "messages": [{"role": "user", "content": prompt}],
        "prompt": prompt,
        "answer": correct,
        "videos": [info["fwd_video"]],
        "data_type": "video",
        "problem_type": "seg_aot_event_v2t",
        "metadata": {
            "clip_key": info["clip_key"],
            "phase_id": info["phase_id"],
            "total_duration_sec": info["total_duration"],
            "n_events": info["n_events"],
            "forward_descriptions": instr,
            "option_a_type": options[0][0],
            "option_b_type": options[1][0],
            "option_c_type": options[2][0],
            "domain_l1": info["domain_l1"],
            "domain_l2": info["domain_l2"],
            "source": "seg_annotation",
        },
    }


def _build_event_t2v(info: dict, rng: random.Random, text_order: str = "forward") -> dict | None:
    """Build one T2V record using *text_order* as the ground-truth text ordering."""
    if not info.get("shuf_video") or not info.get("rev_video"):
        return None

    instr = info["instructions"]
    text_map = {
        "forward": instr,
        "shuffled": info.get("shuf_instructions") or instr,
        "reversed": list(reversed(instr)),
    }
    video_map = {
        "forward": info["fwd_video"],
        "shuffled": info["shuf_video"],
        "reversed": info["rev_video"],
    }

    text_list = text_map[text_order]
    fwd_list = _format_list(text_list)

    video_opts = [
        ("forward", video_map["forward"]),
        ("shuffled", video_map["shuffled"]),
        ("reversed", video_map["reversed"]),
    ]
    rng.shuffle(video_opts)
    correct = ANSWER_LETTERS[[o[0] for o in video_opts].index(text_order)]
    videos = [v for _, v in video_opts]

    tags = "".join("<video>" for _ in videos)
    body = _EVENT_T2V_PROMPT.format(forward_list=fwd_list)
    prompt = f"{tags}\n\n{body}"

    return {
        "messages": [{"role": "user", "content": prompt}],
        "prompt": prompt,
        "answer": correct,
        "videos": videos,
        "data_type": "video",
        "problem_type": "seg_aot_event_t2v",
        "metadata": {
            "clip_key": info["clip_key"],
            "phase_id": info["phase_id"],
            "total_duration_sec": info["total_duration"],
            "n_events": info["n_events"],
            "forward_descriptions": instr,
            "text_order": text_order,
            "video_a_type": video_opts[0][0],
            "video_b_type": video_opts[1][0],
            "video_c_type": video_opts[2][0],
            "correct_position": correct,
            "domain_l1": info["domain_l1"],
            "domain_l2": info["domain_l2"],
            "source": "seg_annotation",
        },
    }


def _build_action_v2t(info: dict, rng: random.Random) -> dict | None:
    acts = info["sub_actions"]
    shuffled = acts[:]
    for _ in range(20):
        rng.shuffle(shuffled)
        if shuffled != acts:
            break
    if shuffled == acts:
        return None

    reversed_acts = list(reversed(acts))
    options = [("forward", acts), ("shuffled", shuffled), ("reversed", reversed_acts)]
    rng.shuffle(options)
    correct = ANSWER_LETTERS[[o[0] for o in options].index("forward")]

    body = _ACTION_V2T_PROMPT.format(
        option_a=_format_list(options[0][1]),
        option_b=_format_list(options[1][1]),
        option_c=_format_list(options[2][1]),
    )
    prompt = f"<video>\n\n{body}"

    return {
        "messages": [{"role": "user", "content": prompt}],
        "prompt": prompt,
        "answer": correct,
        "videos": [info["fwd_video"]],
        "data_type": "video",
        "problem_type": "seg_aot_action_v2t",
        "metadata": {
            "clip_key": info["clip_key"],
            "event_id": info["event_id"],
            "total_duration_sec": info["total_duration"],
            "n_actions": info["n_actions"],
            "forward_descriptions": acts,
            "option_a_type": options[0][0],
            "option_b_type": options[1][0],
            "option_c_type": options[2][0],
            "domain_l1": info["domain_l1"],
            "domain_l2": info["domain_l2"],
            "source": "seg_annotation",
        },
    }


def _build_action_t2v(info: dict, rng: random.Random, text_order: str = "forward") -> dict | None:
    """Build one T2V record using *text_order* as the ground-truth text ordering."""
    if not info.get("shuf_video") or not info.get("rev_video"):
        return None

    acts = info["sub_actions"]
    text_map = {
        "forward": acts,
        "shuffled": info.get("shuf_actions") or acts,
        "reversed": list(reversed(acts)),
    }
    video_map = {
        "forward": info["fwd_video"],
        "shuffled": info["shuf_video"],
        "reversed": info["rev_video"],
    }

    text_list = text_map[text_order]
    fwd_list = _format_list(text_list)

    video_opts = [
        ("forward", video_map["forward"]),
        ("shuffled", video_map["shuffled"]),
        ("reversed", video_map["reversed"]),
    ]
    rng.shuffle(video_opts)
    correct = ANSWER_LETTERS[[o[0] for o in video_opts].index(text_order)]
    videos = [v for _, v in video_opts]

    tags = "".join("<video>" for _ in videos)
    body = _ACTION_T2V_PROMPT.format(forward_list=fwd_list)
    prompt = f"{tags}\n\n{body}"

    return {
        "messages": [{"role": "user", "content": prompt}],
        "prompt": prompt,
        "answer": correct,
        "videos": videos,
        "data_type": "video",
        "problem_type": "seg_aot_action_t2v",
        "metadata": {
            "clip_key": info["clip_key"],
            "event_id": info["event_id"],
            "total_duration_sec": info["total_duration"],
            "n_actions": info["n_actions"],
            "forward_descriptions": acts,
            "text_order": text_order,
            "video_a_type": video_opts[0][0],
            "video_b_type": video_opts[1][0],
            "video_c_type": video_opts[2][0],
            "correct_position": correct,
            "domain_l1": info["domain_l1"],
            "domain_l2": info["domain_l2"],
            "source": "seg_annotation",
        },
    }


# =====================================================================
# Main
# =====================================================================
def main():
    ALL_TASKS = [
        "phase_v2t", "phase_t2v",
        "event_v2t", "event_t2v",
        "action_v2t", "action_t2v",
    ]

    parser = argparse.ArgumentParser(
        description="从三层分割标注构建 AOT MCQ 数据（原子 clip 拼接版）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--annotation-dir", required=True)
    parser.add_argument("--clip-dir", required=True,
                        help="原子 clips 根目录（含 L1/ L2/ L3/），由 prepare_all_clips.py 产出")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--concat-dir", default="",
                        help="拼接视频输出目录（默认: {output-dir}/concat_videos）")
    parser.add_argument("--concat-workers", type=int, default=8,
                        help="并发 ffmpeg 拼接线程数")
    parser.add_argument("--overwrite-concat", action="store_true",
                        help="强制重新生成已存在的拼接视频")
    parser.add_argument("--tasks", nargs="+", choices=ALL_TASKS, default=ALL_TASKS)
    parser.add_argument("--min-phases", type=int, default=3)
    parser.add_argument("--max-phases", type=int, default=999)
    parser.add_argument("--min-events", type=int, default=3)
    parser.add_argument("--max-events", type=int, default=999)
    parser.add_argument("--min-actions", type=int, default=3)
    parser.add_argument("--max-actions", type=int, default=999)
    parser.add_argument("--max-t2v-duration", type=int, default=85,
                        help="T2V 最大单视频时长(s)，超过则跳过 (3视频×85帧=255≤MAX_FRAMES)")
    parser.add_argument("--total-val", type=int, default=200)
    parser.add_argument("--train-total", type=int, default=-1,
                        help="总训练样本数上限 (-1=不限制)")
    parser.add_argument("--level-ratio", type=str, default="1:2:2",
                        help="L1:L2:L3 采样比例 (默认 1:2:2)")
    parser.add_argument("--complete-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    concat_dir = args.concat_dir or os.path.join(args.output_dir, "concat_videos")
    tasks = set(args.tasks)
    max_t2v_dur = args.max_t2v_duration

    need_phase_v2t = "phase_v2t" in tasks
    need_phase_t2v = "phase_t2v" in tasks
    need_event_v2t = "event_v2t" in tasks
    need_event_t2v = "event_t2v" in tasks
    need_action_v2t = "action_v2t" in tasks
    need_action_t2v = "action_t2v" in tasks

    # ---- 1. Load annotations ----
    ann_list = load_annotations(args.annotation_dir, complete_only=False)
    log.info("Loaded %d annotations", len(ann_list))

    # ---- 2. Collect concat jobs + group metadata ----
    all_jobs: dict[str, tuple[list[str], str]] = {}  # keyed by output_path (dedup)
    phase_groups: list[dict] = []
    event_groups: list[dict] = []
    action_groups: list[dict] = []

    for ann in ann_list:
        ann_rng = random.Random(rng.random())

        if need_phase_v2t or need_phase_t2v:
            info, jobs = _collect_phase_groups(
                ann, args.clip_dir, concat_dir,
                need_phase_v2t, need_phase_t2v,
                args.min_phases, args.max_phases,
                args.complete_only, ann_rng,
            )
            if info:
                phase_groups.append(info)
                for j in jobs:
                    all_jobs.setdefault(j[1], j)

        if need_event_v2t or need_event_t2v:
            infos, jobs = _collect_event_groups(
                ann, args.clip_dir, concat_dir,
                need_event_v2t, need_event_t2v,
                args.min_events, args.max_events,
                args.complete_only, ann_rng,
            )
            event_groups.extend(infos)
            for j in jobs:
                all_jobs.setdefault(j[1], j)

        if need_action_v2t or need_action_t2v:
            infos, jobs = _collect_action_groups(
                ann, args.clip_dir, concat_dir,
                need_action_v2t, need_action_t2v,
                args.min_actions, args.max_actions,
                args.complete_only, ann_rng,
            )
            action_groups.extend(infos)
            for j in jobs:
                all_jobs.setdefault(j[1], j)

    log.info(
        "Collected %d concat jobs (phase=%d, event=%d, action=%d groups)",
        len(all_jobs), len(phase_groups), len(event_groups), len(action_groups),
    )

    # ---- 3. Execute concat jobs in parallel ----
    success = 0
    failed_paths: set[str] = set()

    with ThreadPoolExecutor(max_workers=args.concat_workers) as pool:
        futures = {
            pool.submit(
                ffmpeg_concat_clips, clip_paths, out_path,
                overwrite=args.overwrite_concat,
            ): out_path
            for clip_paths, out_path in all_jobs.values()
        }
        for fut in as_completed(futures):
            out = futures[fut]
            try:
                if fut.result():
                    success += 1
                else:
                    failed_paths.add(out)
            except Exception:
                failed_paths.add(out)

    log.info("Concat done: %d success, %d failed", success, len(failed_paths))

    # ---- 4. Build records ----
    # T2V uses round-robin text_order (fwd/shuf/rev) for balanced distribution
    task_records: dict[str, list[dict]] = {t: [] for t in args.tasks}
    t2v_order_idx = 0  # round-robin counter across all T2V groups

    for info in phase_groups:
        if info["fwd_video"] in failed_paths:
            continue
        if need_phase_v2t:
            rec = _build_phase_v2t(info, random.Random(rng.random()))
            if rec:
                task_records["phase_v2t"].append(rec)
        if need_phase_t2v:
            if info.get("shuf_video") not in failed_paths and info.get("rev_video") not in failed_paths:
                text_order = _T2V_TEXT_ORDERS[t2v_order_idx % 3]
                rec = _build_phase_t2v(info, random.Random(rng.random()), text_order)
                if rec:
                    task_records["phase_t2v"].append(rec)
                    t2v_order_idx += 1

    for info in event_groups:
        if info["fwd_video"] in failed_paths:
            continue
        if need_event_v2t:
            rec = _build_event_v2t(info, random.Random(rng.random()))
            if rec:
                task_records["event_v2t"].append(rec)
        if need_event_t2v and info["total_duration"] <= max_t2v_dur:
            if info.get("shuf_video") not in failed_paths and info.get("rev_video") not in failed_paths:
                text_order = _T2V_TEXT_ORDERS[t2v_order_idx % 3]
                rec = _build_event_t2v(info, random.Random(rng.random()), text_order)
                if rec:
                    task_records["event_t2v"].append(rec)
                    t2v_order_idx += 1

    for info in action_groups:
        if info["fwd_video"] in failed_paths:
            continue
        if need_action_v2t:
            rec = _build_action_v2t(info, random.Random(rng.random()))
            if rec:
                task_records["action_v2t"].append(rec)
        if need_action_t2v and info["total_duration"] <= max_t2v_dur:
            if info.get("shuf_video") not in failed_paths and info.get("rev_video") not in failed_paths:
                text_order = _T2V_TEXT_ORDERS[t2v_order_idx % 3]
                rec = _build_action_t2v(info, random.Random(rng.random()), text_order)
                if rec:
                    task_records["action_t2v"].append(rec)
                    t2v_order_idx += 1

    # ---- Stats ----
    print("\n=== Extraction Stats ===")
    for task in args.tasks:
        print(f"  {task}: {len(task_records.get(task, []))} records")

    # ---- 5. Train/val split with level-balanced sampling ----
    # Parse level ratio
    ratio_parts = [float(x) for x in args.level_ratio.split(":")]
    if len(ratio_parts) != 3:
        raise ValueError(f"--level-ratio must be L1:L2:L3, got: {args.level_ratio}")
    ratio_sum = sum(ratio_parts)
    level_weights = {
        "phase": ratio_parts[0] / ratio_sum,
        "event": ratio_parts[1] / ratio_sum,
        "action": ratio_parts[2] / ratio_sum,
    }

    # Group tasks by level
    level_tasks = {"phase": [], "event": [], "action": []}
    for task in args.tasks:
        level = task.split("_")[0]  # phase_v2t → phase
        level_tasks[level].append(task)

    # Compute per-level train budget
    train_total = args.train_total
    n_tasks = len(args.tasks)
    val_per_task = max(1, args.total_val // max(n_tasks, 1))

    all_train: list[dict] = []
    all_val: list[dict] = []

    # ---- Collect per-task pools (separate val first) ----
    task_pools: dict[str, tuple[list[dict], list[dict]]] = {}  # task → (train_pool, val)
    task_to_level: dict[str, str] = {}
    for level, tasks_in_level in level_tasks.items():
        for task in tasks_in_level:
            records = task_records.get(task, [])
            rng.shuffle(records)
            n_val = min(val_per_task, max(1, len(records) // 5))
            task_pools[task] = (records[n_val:], records[:n_val])
            task_to_level[task] = level

    # ---- Weighted quota with overflow redistribution ----
    if train_total > 0:
        # Compute available pool sizes per level
        level_pool_sizes: dict[str, int] = {}
        for level, tasks_in_level in level_tasks.items():
            if not tasks_in_level:
                continue
            level_pool_sizes[level] = sum(len(task_pools[t][0]) for t in tasks_in_level)

        # Iterative redistribution: levels below their quota release excess
        remaining_budget = train_total
        level_budgets: dict[str, int] = {}
        active_levels = {lv: w for lv, w in level_weights.items() if lv in level_pool_sizes}
        while active_levels and remaining_budget > 0:
            w_sum = sum(active_levels.values())
            if w_sum <= 0:
                break
            done_this_round = True
            next_active = {}
            for lv, w in active_levels.items():
                quota = int(remaining_budget * w / w_sum)
                pool_sz = level_pool_sizes[lv] - level_budgets.get(lv, 0)
                if pool_sz <= quota:
                    # This level can't fill its quota → take all, redistribute remainder
                    level_budgets[lv] = level_pool_sizes[lv]
                    remaining_budget -= pool_sz
                    done_this_round = False
                else:
                    next_active[lv] = w
            if done_this_round:
                # All active levels can fill their quota
                w_sum = sum(next_active.values())
                for lv, w in next_active.items():
                    level_budgets[lv] = level_budgets.get(lv, 0) + int(remaining_budget * w / w_sum)
                remaining_budget = 0
                break
            active_levels = next_active
    else:
        level_budgets = {}

    # ---- Sample per task ----
    for level, tasks_in_level in level_tasks.items():
        if not tasks_in_level:
            continue
        lb = level_budgets.get(level)
        per_task_budget = max(1, lb // len(tasks_in_level)) if lb else None

        for task in tasks_in_level:
            train_pool, val = task_pools[task]
            if per_task_budget is not None and len(train_pool) > per_task_budget:
                train = _balanced_sample_by_domain(train_pool, per_task_budget, rng)
            else:
                train = train_pool
            all_val.extend(val)
            all_train.extend(train)
            l1_dist = Counter(r.get("metadata", {}).get("domain_l1", "other") for r in train)
            l1_str = ", ".join(f"{d}={c}" for d, c in sorted(l1_dist.items()))
            print(f"  {task}: {len(train)} train + {len(val)} val"
                  f"  (level={level}, budget={per_task_budget or 'unlimited'},"
                  f" L1: {l1_str})")

    rng.shuffle(all_train)
    rng.shuffle(all_val)

    # ---- Write ----
    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.jsonl")
    val_path = os.path.join(args.output_dir, "val.jsonl")
    write_jsonl(all_train, train_path)
    write_jsonl(all_val, val_path)

    stats = {
        "total_annotations": len(ann_list),
        "tasks": args.tasks,
        "level_ratio": args.level_ratio,
        "train_total_budget": train_total,
        "concat_dir": concat_dir,
        "concat_total": len(all_jobs),
        "concat_failed": len(failed_paths),
        "train_total": len(all_train),
        "val_total": len(all_val),
        "train_by_type": dict(Counter(r["problem_type"] for r in all_train)),
        "val_by_type": dict(Counter(r["problem_type"] for r in all_val)),
        "train_by_domain_l1": dict(Counter(r.get("metadata", {}).get("domain_l1", "other") for r in all_train)),
        "train_by_domain_l2": dict(Counter(r.get("metadata", {}).get("domain_l2", "other") for r in all_train)),
        "val_by_domain_l1": dict(Counter(r.get("metadata", {}).get("domain_l1", "other") for r in all_val)),
        "val_by_domain_l2": dict(Counter(r.get("metadata", {}).get("domain_l2", "other") for r in all_val)),
    }
    stats_path = os.path.join(args.output_dir, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n=== Output ===")
    print(f"  Train: {len(all_train)}  →  {train_path}")
    print(f"  Val:   {len(all_val)}  →  {val_path}")
    print(f"  Concat: {success}/{len(all_jobs)} success  →  {concat_dir}")
    print(f"  Stats: {stats_path}")

    if all_train:
        ex = all_train[0]
        print(f"\n=== Example record ===")
        print(f"  problem_type: {ex['problem_type']}")
        print(f"  answer: {ex['answer']}")
        print(f"  videos ({len(ex['videos'])}): {ex['videos'][0]}")
        print(f"  prompt (first 300 chars):\n  {ex['prompt'][:300]}")


if __name__ == "__main__":
    main()
