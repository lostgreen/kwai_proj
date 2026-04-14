#!/usr/bin/env python3
"""
prepare_all_clips.py — 从三层分割标注切出原子视频片段。

读取 annotation JSON，按标注的时间戳直接切分：
  L1 phase:  每个 macro_phase → 独立 clip (默认 1fps)
  L2 event:  每个 event → 独立 clip (默认 2fps)
  L3 action: 每个 sub_action → 独立 clip (默认 2fps)

产出的原子 clips 被 temporal_aot、event_logic、hier_seg 三条管线共用。
下游任务若需要更长片段，直接拼接原子 clips 即可。

用法:
    python proxy_data/youcook2_seg/prepare_all_clips.py \\
        --annotation-dir /path/to/annotations \\
        --source-video-dir /path/to/source_videos \\
        --output-dir /path/to/clips \\
        --levels L1 L2 L3 \\
        --l1-fps 1 --l2l3-fps 2 \\
        --workers 8

输出结构:
    {output-dir}/
    ├── L1/  {key}_L1_ph{id}_{start}_{end}.mp4
    ├── L2/  {key}_L2_ev{id}_{start}_{end}.mp4
    └── L3/  {key}_L3_act{id}_{start}_{end}.mp4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ── 添加 proxy_data/ 到 sys.path 以便 import shared ──
_PROXY_DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROXY_DATA_DIR not in sys.path:
    sys.path.insert(0, _PROXY_DATA_DIR)

from shared.seg_source import load_annotations  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ── ffmpeg helpers ──────────────────────────────────────────────────────────

def _ffmpeg_cut(
    src: str | Path,
    dst: str | Path,
    start: int,
    end: int,
    fps: int | None = None,
    overwrite: bool = False,
) -> bool:
    """用 ffmpeg 从 src 切出 [start, end) 片段，可选 fps 重采样。

    Returns True on success, False on failure.
    """
    dst = Path(dst)

    if dst.exists() and not overwrite:
        return True  # 已存在，跳过

    dst.parent.mkdir(parents=True, exist_ok=True)

    duration = end - start
    if duration <= 0:
        log.warning("Skipping %s: duration=%d <= 0", dst.name, duration)
        return False

    vf_parts: list[str] = []
    if fps is not None:
        vf_parts.append(f"fps={fps}")

    cmd: list[str] = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", str(src),
        "-t", str(duration),
    ]
    if vf_parts:
        cmd += ["-vf", ",".join(vf_parts)]
    cmd += [
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-an",  # 去音频
        str(dst),
    ]

    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            timeout=120,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        log.error("ffmpeg failed for %s: %s", dst.name, exc)
        if dst.exists():
            dst.unlink()
        return False


# ── Job 生成 ────────────────────────────────────────────────────────────────

def _resolve_source_video(ann: dict, source_video_dir: str | None) -> str | None:
    """从标注或 source_video_dir 解析出源视频路径。"""
    # 1) 标注内的 source_video_path
    vp = ann.get("source_video_path") or ann.get("video_path")
    if vp and os.path.isfile(vp):
        return vp

    # 2) source_video_dir/{clip_key}.mp4
    if source_video_dir:
        clip_key = ann["clip_key"]
        for ext in (".mp4", ".mkv", ".webm", ".avi"):
            candidate = os.path.join(source_video_dir, f"{clip_key}{ext}")
            if os.path.isfile(candidate):
                return candidate

    return None


def _generate_l1_jobs(
    ann: dict,
    src_video: str,
    output_dir: Path,
    fps: int,
) -> list[dict]:
    """为每个 L1 macro_phase 生成切分 job。"""
    jobs: list[dict] = []
    phases = ann.get("level1", {}).get("macro_phases", [])
    clip_key = ann["clip_key"]

    for ph in phases:
        ph_id = ph.get("phase_id", 0)
        start = int(ph.get("start_time", 0))
        end = int(ph.get("end_time", 0))
        if end <= start:
            continue
        dst = output_dir / "L1" / f"{clip_key}_L1_ph{ph_id}_{start}_{end}.mp4"
        jobs.append({
            "src": src_video, "dst": str(dst),
            "start": start, "end": end, "fps": fps,
            "level": "L1", "clip_key": clip_key,
        })
    return jobs


def _generate_l2_jobs(
    ann: dict,
    src_video: str,
    output_dir: Path,
    fps: int,
) -> list[dict]:
    """为每个 L2 event 生成切分 job。"""
    jobs: list[dict] = []
    events = ann.get("level2", {}).get("events", [])
    clip_key = ann["clip_key"]

    for ev in events:
        ev_id = ev.get("event_id", 0)
        start = int(ev.get("start_time", 0))
        end = int(ev.get("end_time", 0))
        if end <= start:
            continue
        dst = output_dir / "L2" / f"{clip_key}_L2_ev{ev_id}_{start}_{end}.mp4"
        jobs.append({
            "src": src_video, "dst": str(dst),
            "start": start, "end": end, "fps": fps,
            "level": "L2", "clip_key": clip_key,
        })
    return jobs


def _generate_l3_jobs(
    ann: dict,
    src_video: str,
    output_dir: Path,
    fps: int,
) -> list[dict]:
    """为每个 L3 sub_action 生成切分 job。

    当前 schema（annotate.py 产生的嵌套格式）:
        level3.grounding_results[i] = {
            "event_id": <int>,
            "sub_actions": [{"action_id": <int>, "start_time": ..., "end_time": ..., ...}]
        }
    旧格式（平铺）已废弃，不再支持。
    """
    jobs: list[dict] = []
    grounding_results = ann.get("level3", {}).get("grounding_results", [])
    clip_key = ann["clip_key"]

    for gr in grounding_results:
        if not isinstance(gr, dict):
            continue
        parent_ev = gr.get("event_id", 0)
        for sa in gr.get("sub_actions", []):
            if not isinstance(sa, dict):
                continue
            act_id = sa.get("action_id", 0)
            start = int(sa.get("start_time", 0))
            end = int(sa.get("end_time", 0))
            if end <= start:
                continue
            dst = output_dir / "L3" / f"{clip_key}_L3_act{act_id}_ev{parent_ev}_{start}_{end}.mp4"
            jobs.append({
                "src": src_video, "dst": str(dst),
                "start": start, "end": end, "fps": fps,
                "level": "L3", "clip_key": clip_key,
            })
    return jobs


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="从三层分割标注切出 L1/L2/L3 原子视频片段",
    )
    parser.add_argument("--annotation-dir", required=True,
                        help="标注 JSON 所在目录")
    parser.add_argument("--source-video-dir", default=None,
                        help="源视频目录（回退路径，优先用标注内 source_video_path）")
    parser.add_argument("--output-dir", required=True,
                        help="输出根目录，下设 L1/ L2/ L3/ 子文件夹")
    parser.add_argument("--levels", nargs="+", default=["L1", "L2", "L3"],
                        choices=["L1", "L2", "L3"],
                        help="要切分的层级 (default: L1 L2 L3)")
    parser.add_argument("--l1-fps", type=int, default=1,
                        help="L1 phase clips 的 fps (default: 1)")
    parser.add_argument("--l2l3-fps", type=int, default=2,
                        help="L2/L3 clips 的 fps (default: 2)")
    parser.add_argument("--workers", type=int, default=8,
                        help="并发 ffmpeg 进程数 (default: 8)")
    parser.add_argument("--overwrite", action="store_true",
                        help="覆盖已存在的 clips")
    parser.add_argument("--dry-run", action="store_true",
                        help="只列出要切分的 jobs，不实际执行")
    parser.add_argument("--complete-only", action="store_true",
                        help="只处理标注完整的视频")
    parser.add_argument("--min-phases", type=int, default=0,
                        help="只切 L1 phases ≥ 此数量的标注 (0=不过滤)")
    parser.add_argument("--max-phases", type=int, default=999,
                        help="只切 L1 phases ≤ 此数量的标注")
    parser.add_argument("--min-events", type=int, default=0,
                        help="只切含 ≥ 此数量 events 的 phase 组 (0=不过滤)")
    parser.add_argument("--max-events", type=int, default=999,
                        help="只切含 ≤ 此数量 events 的 phase 组")
    parser.add_argument("--min-actions", type=int, default=0,
                        help="只切含 ≥ 此数量 actions 的 event 组 (0=不过滤)")
    parser.add_argument("--max-actions", type=int, default=999,
                        help="只切含 ≤ 此数量 actions 的 event 组")
    parser.add_argument("--clip-list", default=None,
                        help="只切此文件中列出的 clip 路径（每行一个绝对路径）。"
                             "配合 build_event_logic_vlm.py --collect-clips-only 使用，"
                             "跳过 LLM 判定为不需要的 clips。")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    levels = set(args.levels)

    # ---- 加载标注 ----
    anns = load_annotations(args.annotation_dir, complete_only=args.complete_only)
    log.info("Loaded %d annotations from %s", len(anns), args.annotation_dir)

    # ---- 生成 jobs ----
    all_jobs: list[dict] = []
    skipped_videos = 0

    for ann in anns:
        src_video = _resolve_source_video(ann, args.source_video_dir)
        if not src_video:
            skipped_videos += 1
            continue

        # ---- 过滤: 只切满足最低数量要求的标注 ----
        l1 = ann.get("level1", {})
        l2 = ann.get("level2", {})
        l3 = ann.get("level3", {})
        phases = [p for p in l1.get("macro_phases", [])
                  if isinstance(p, dict) and isinstance(p.get("start_time"), (int, float))
                  and p.get("phase_name", "").strip()]
        events = [e for e in l2.get("events", [])
                  if isinstance(e, dict) and isinstance(e.get("start_time"), (int, float))
                  and e.get("instruction", "").strip()]

        cut_l1 = "L1" in levels and args.min_phases <= len(phases) <= args.max_phases
        # L2: 至少有一个 phase 拥有 [min_events, max_events] 个 child events
        events_by_phase = {}
        for e in events:
            events_by_phase.setdefault(e.get("parent_phase_id"), []).append(e)
        cut_l2 = "L2" in levels and any(
            args.min_events <= len(evs) <= args.max_events
            for evs in events_by_phase.values()
        )
        # L3: 至少有一个 event 拥有 [min_actions, max_actions] 个 child actions
        # 按 event_id 统计 sub_actions 数量（嵌套格式直接从 grounding_results 读）
        actions_by_event = {
            gr["event_id"]: gr.get("sub_actions", [])
            for gr in l3.get("grounding_results", [])
            if isinstance(gr, dict) and "event_id" in gr
        }
        cut_l3 = "L3" in levels and any(
            args.min_actions <= len(acts) <= args.max_actions
            for acts in actions_by_event.values()
        )

        if cut_l1:
            all_jobs.extend(_generate_l1_jobs(ann, src_video, output_dir, args.l1_fps))
        if cut_l2:
            all_jobs.extend(_generate_l2_jobs(ann, src_video, output_dir, args.l2l3_fps))
        if cut_l3:
            all_jobs.extend(_generate_l3_jobs(ann, src_video, output_dir, args.l2l3_fps))

    log.info(
        "Generated %d clip jobs (L1=%d, L2=%d, L3=%d), skipped %d videos (no source found)",
        len(all_jobs),
        sum(1 for j in all_jobs if j["level"] == "L1"),
        sum(1 for j in all_jobs if j["level"] == "L2"),
        sum(1 for j in all_jobs if j["level"] == "L3"),
        skipped_videos,
    )

    # ---- 按需过滤: 只切 --clip-list 中指定的 clips ----
    if args.clip_list:
        if not os.path.isfile(args.clip_list):
            log.error("--clip-list file not found: %s", args.clip_list)
            raise SystemExit(1)
        with open(args.clip_list, "r", encoding="utf-8") as f:
            needed = {line.strip() for line in f if line.strip()}
        before = len(all_jobs)
        all_jobs = [j for j in all_jobs if j["dst"] in needed]
        log.info(
            "clip-list filter: %d → %d jobs (%d clips in list, %d not matched by any job)",
            before, len(all_jobs), len(needed), len(needed) - len(all_jobs),
        )

    if args.dry_run:
        for j in all_jobs:
            print(f"[{j['level']}] {j['clip_key']}  {j['start']}-{j['end']}s  → {j['dst']}")
        return

    # ---- 执行切分 ----
    success = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _ffmpeg_cut, j["src"], j["dst"], j["start"], j["end"],
                fps=j["fps"], overwrite=args.overwrite,
            ): j
            for j in all_jobs
        }

        for fut in as_completed(futures):
            job = futures[fut]
            try:
                if fut.result():
                    success += 1
                else:
                    failed += 1
            except Exception as exc:
                log.error("Unexpected error for %s: %s", job["dst"], exc)
                failed += 1

    log.info("Done: %d success, %d failed, %d total", success, failed, len(all_jobs))


if __name__ == "__main__":
    main()
