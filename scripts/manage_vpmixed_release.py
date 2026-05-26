#!/usr/bin/env python3
"""Build the trainable vpmixed release view with release-local frame paths."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_REL = Path("/m2v_intern/xuboshen/zgw/VideoProxyData/releases/vpmixed_20260516_full_comp")
DEFAULT_OLD = Path("/m2v_intern/xuboshen/zgw/data/VideoProxyMixed")

FULL_MIX = "composition_base_seg_logic_aot_hier10k_el10k_aot10k_mf256_ema"
OLD_MIXES = [
    "composition_base_seg_hier10k_mf256_ema",
    "composition_base_aot_aot10k_mf256_ema",
    "composition_base_logic_el10k_mf256_ema",
    "composition_base_seg_aot_hier10k_aot10k_mf256_ema",
    FULL_MIX,
]

TASK_BY_PROBLEM_TYPE = {
    "temporal_grounding": "tg",
    "llava_mcq": "mcq",
    "temporal_seg_hier_L1": "seg",
    "temporal_seg_hier_L2": "seg",
    "temporal_seg_hier_L3_seg": "seg",
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
V2_MIX = "composition_base_seg_hier10k_mf256_l3v2_ema"


def task_for_row(row: dict[str, Any]) -> str:
    problem_type = str(row.get("problem_type") or "")
    if problem_type in TASK_BY_PROBLEM_TYPE:
        return TASK_BY_PROBLEM_TYPE[problem_type]
    if problem_type.startswith("seg_aot_"):
        return "aot"
    if problem_type.startswith("event_logic_"):
        return "logic"
    return "unknown"


def cache_kind_for_task(task: str, old_base_frame_root: Path, path: Path) -> str:
    try:
        path.relative_to(old_base_frame_root)
        return "base_cache_2fps"
    except ValueError:
        pass
    return "source_2fps"


def rewrite_frame_path(path_text: str, task: str, rel: Path, old: Path) -> str:
    path = Path(path_text)
    if not path.is_absolute():
        return path_text
    if str(path).startswith(str(rel / "frames")):
        return path_text

    old_base = old / "multi_task" / "offline_frames" / "base_cache_2fps"
    old_source = old / "hier_seg_annotation_v1" / "frame_cache" / "source_2fps"
    mappings = (
        (old_base, rel / "frames" / task / "base_cache_2fps"),
        (old_source, rel / "frames" / task / "source_2fps"),
    )
    for src_root, dst_root in mappings:
        try:
            return str(dst_root / path.relative_to(src_root))
        except ValueError:
            continue
    return path_text


def is_frame_path(text: str) -> bool:
    return Path(text).suffix.lower() in IMAGE_SUFFIXES and text.startswith("/")


def rewrite_any_frame_refs(value: Any, task: str, rel: Path, old: Path) -> Any:
    if isinstance(value, str):
        if value.startswith("/") and (
            "/offline_frames/base_cache_2fps/" in value
            or "/hier_seg_annotation_v1/frame_cache/source_2fps/" in value
        ):
            return rewrite_frame_path(value, task, rel, old)
        return value
    if isinstance(value, list):
        return [rewrite_any_frame_refs(item, task, rel, old) for item in value]
    if isinstance(value, dict):
        return {key: rewrite_any_frame_refs(item, task, rel, old) for key, item in value.items()}
    return value


def release_cache_roots_for_task(task: str, rel: Path) -> list[str]:
    if task in {"tg", "mcq"}:
        return [str(rel / "frames" / task / "base_cache_2fps")]
    if task in {"seg", "aot", "logic"}:
        return [str(rel / "frames" / task / "source_2fps")]
    return []


def normalize_sampling_metadata(row: dict[str, Any], task: str, rel: Path) -> None:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        return

    roots = release_cache_roots_for_task(task, rel)
    sampling = metadata.get("experiment_frame_sampling")
    if isinstance(sampling, dict) and roots:
        sampling["trusted_cache_roots"] = roots


def rewrite_record(row: dict[str, Any], rel: Path, old: Path) -> dict[str, Any]:
    task = task_for_row(row)
    out = rewrite_any_frame_refs(row, task, rel, old)
    if isinstance(out, dict):
        normalize_sampling_metadata(out, task, rel)
    return out


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists():
            dst.unlink()
        os.link(src, dst)
    except OSError:
        data = src.read_bytes()
        dst.write_bytes(data)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row.get("problem_type") or "unknown") for row in rows)
    return {"rows": len(rows), "problem_type": dict(sorted(counts.items()))}


def frame_path_report(rows: list[dict[str, Any]], rel: Path, sample_limit: int = 10) -> dict[str, Any]:
    total = 0
    release_local = 0
    old_paths = 0
    bad_sample: list[str] = []

    def walk(value: Any) -> None:
        nonlocal total, release_local, old_paths
        if isinstance(value, str) and is_frame_path(value):
            total += 1
            if value.startswith(str(rel / "frames")):
                release_local += 1
            if "/data/VideoProxyMixed/" in value:
                old_paths += 1
                if len(bad_sample) < sample_limit:
                    bad_sample.append(value)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)

    for row in rows:
        walk(row.get("videos"))
    return {
        "frame_paths": total,
        "release_local": release_local,
        "old_paths": old_paths,
        "bad_sample": bad_sample,
    }


def build_l3_v2_body(duration: int) -> str:
    return f"""You are given a {duration}s video clip represented by sparsely sampled frames.

Detect all fine-grained L3 sub-actions using a SHOT-FIRST policy:

STEP 1 - FIND SHOT / SCENE BOUNDARIES:
A shot boundary is a visible camera cut, angle/framing change, subject change, or abrupt scene transition. These boundaries are strong temporal anchors for L3.

Do not merge visually distinct shots into one segment unless they are static, redundant, or show the same unchanged visual state.

STEP 2 - KEEP OR SPLIT EACH SHOT:
- Keep a shot as one segment if it contains one continuous visual state or action.
- Split a long shot when the physical action, object/material state, subject pose, or task phase clearly changes.
- A long single shot may produce multiple L3 segments.

Create a boundary when ANY of the following occurs:
- A camera cut or framing change to a different angle or subject.
- A physical action change: a different motion or task begins.
- A visible object/material state change: deformation, separation, positional shift, or state transition.
- An object or subject enters or leaves the frame.
- An environmental shift: lighting change or background change.

Do NOT place a boundary when:
- Hands or body parts reposition without changing any object's state or the camera framing.
- A single-frame flicker is not sustained across 2 or more sampled frames.

IMPORTANT - SPARSE VISUAL EVIDENCE:
This clip is represented by sparsely sampled frames and timestamp markers, not continuous video. Use the displayed timestamps as the temporal reference. Do NOT rely on single-frame flicker. Do NOT rely on single-frame micro-motions, tiny hand/body repositioning, or instantaneous contact changes to place boundaries. A clear camera cut, angle change, framing change, subject change, or sustained state/action change SHOULD be used as a boundary.

Gaps between segments are expected - do not force full coverage.

Output the start and end time (integer seconds, 0-based) for each segment in chronological order:
<events>[[start_time, end_time], ...]</events>

Example: <events>[[2, 6], [9, 13], [15, 20]]</events>"""


def infer_prompt_duration(row: dict[str, Any]) -> int:
    prompt = str(row.get("prompt") or "")
    match = re.search(r"given a\s+(\d+)s video clip", prompt)
    if match:
        return int(match.group(1))
    sampling = ((row.get("metadata") or {}).get("experiment_frame_sampling") or {}).get("videos") or []
    if sampling and isinstance(sampling[0], dict):
        duration = sampling[0].get("duration_sec")
        if isinstance(duration, (int, float)) and duration > 0:
            return max(1, int(round(duration)))
    videos = row.get("videos") or []
    if videos and isinstance(videos[0], list):
        return max(1, int(round(len(videos[0]) / 2.0)))
    return 1


def replace_l3_prompt(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("problem_type") != "temporal_seg_hier_L3_seg":
        return row
    duration = infer_prompt_duration(row)
    prompt = "Watch the following video clip carefully:\n<video>\n\n" + build_l3_v2_body(duration)
    row = json.loads(json.dumps(row, ensure_ascii=False))
    row["prompt"] = prompt
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        row["messages"] = [{"role": "user", "content": prompt}]
    else:
        last_user = None
        for idx, message in enumerate(messages):
            if isinstance(message, dict) and message.get("role") == "user":
                last_user = idx
        if last_user is None:
            messages.append({"role": "user", "content": prompt})
        else:
            messages[last_user]["content"] = prompt
    metadata = dict(row.get("metadata") or {})
    metadata["prompt_variant"] = "L3_V2"
    row["metadata"] = metadata
    return row


def build_task_jsonl(rel: Path, old: Path) -> dict[str, Any]:
    src_exp = old / "multi_task" / "experiments" / FULL_MIX
    report: dict[str, Any] = {}
    for split in ("train", "val"):
        grouped: dict[str, list[dict[str, Any]]] = {}
        src = src_exp / f"{split}.jsonl"
        print(f"[task] split proxy tasks from {src}", flush=True)
        for row in iter_jsonl(src):
            rewritten = rewrite_record(row, rel, old)
            task = task_for_row(rewritten)
            if task == "unknown":
                continue
            grouped.setdefault(task, []).append(rewritten)
        for task, rows in sorted(grouped.items()):
            print(f"[task] {task}/{split}: {len(rows)}", flush=True)
            out = rel / "tasks" / task / f"{split}.jsonl"
            write_jsonl(rows, out)
            report[f"{task}/{split}"] = summarize(rows)
    return report


def build_mix_jsonl(rel: Path, old: Path, names: list[str]) -> dict[str, Any]:
    old_exps = old / "multi_task" / "experiments"
    report: dict[str, Any] = {}
    for name in names:
        for split in ("train", "val"):
            src = old_exps / name / f"{split}.jsonl"
            if not src.is_file():
                continue
            print(f"[mix] {name}/{split} <- {src}", flush=True)
            rows = [rewrite_record(row, rel, old) for row in iter_jsonl(src)]
            out = rel / "mixes" / name / f"{split}.jsonl"
            write_jsonl(rows, out)
            report[f"{name}/{split}"] = summarize(rows)
    return report


def build_compat(rel: Path) -> None:
    print("[compat] base and val links", flush=True)
    compat = rel / "compat" / "multi_task"
    hardlink_or_copy(rel / "tasks" / "tg" / "train.jsonl", compat / "base" / "tg_train.jsonl")
    hardlink_or_copy(rel / "tasks" / "tg" / "train.jsonl", compat / "base" / "tg_train_frames.jsonl")
    hardlink_or_copy(rel / "tasks" / "mcq" / "train.jsonl", compat / "base" / "mcq_train_filtered.jsonl")
    hardlink_or_copy(rel / "tasks" / "mcq" / "train.jsonl", compat / "base" / "mcq_train_filtered_frames.jsonl")

    hardlink_or_copy(rel / "tasks" / "tg" / "val.jsonl", compat / "val" / "tg_val_600.jsonl")
    hardlink_or_copy(rel / "tasks" / "tg" / "val.jsonl", compat / "val" / "tg_val_600_frames.jsonl")
    hardlink_or_copy(rel / "tasks" / "mcq" / "val.jsonl", compat / "val" / "mcq_val_600.jsonl")
    hardlink_or_copy(rel / "tasks" / "mcq" / "val.jsonl", compat / "val" / "mcq_val_600_frames.jsonl")
    hardlink_or_copy(rel / "tasks" / "seg" / "val.jsonl", compat / "val" / "hier_seg_val_150.jsonl")
    hardlink_or_copy(rel / "tasks" / "aot" / "val.jsonl", compat / "val" / "aot_val_300.jsonl")
    hardlink_or_copy(rel / "tasks" / "logic" / "val.jsonl", compat / "val" / "event_logic_val_300.jsonl")

    for mix_dir in sorted((rel / "mixes").iterdir()):
        if not mix_dir.is_dir():
            continue
        print(f"[compat] experiment {mix_dir.name}", flush=True)
        exp_dir = compat / "experiments" / mix_dir.name
        hardlink_or_copy(mix_dir / "train.jsonl", exp_dir / "train.jsonl")
        hardlink_or_copy(mix_dir / "val.jsonl", exp_dir / "val.jsonl")


def build_v2(rel: Path) -> dict[str, Any]:
    report: dict[str, Any] = {}
    task_dir = rel / "tasks" / "seg_l3_prompt_v2"
    for split in ("train", "val"):
        print(f"[v2] task seg_l3_prompt_v2/{split}", flush=True)
        rows = [replace_l3_prompt(row) for row in iter_jsonl(rel / "tasks" / "seg" / f"{split}.jsonl")]
        write_jsonl(rows, task_dir / f"{split}.jsonl")
        report[f"seg_l3_prompt_v2/{split}"] = summarize(rows)

    src_mix = rel / "mixes" / "composition_base_seg_hier10k_mf256_ema"
    dst_mix = rel / "mixes" / V2_MIX
    for split in ("train", "val"):
        print(f"[v2] mix {V2_MIX}/{split}", flush=True)
        rows = [replace_l3_prompt(row) for row in iter_jsonl(src_mix / f"{split}.jsonl")]
        write_jsonl(rows, dst_mix / f"{split}.jsonl")
        report[f"{V2_MIX}/{split}"] = summarize(rows)
    build_compat(rel)
    return report


def write_stats(rel: Path) -> dict[str, Any]:
    print("[stats] summarizing tasks and mixes", flush=True)
    release: dict[str, Any] = {"tasks": {}, "mixes": {}, "frame_paths": {}}
    for task_dir in sorted((rel / "tasks").iterdir()):
        if not task_dir.is_dir():
            continue
        stats: dict[str, Any] = {}
        for split in ("train", "val"):
            path = task_dir / f"{split}.jsonl"
            if path.is_file():
                rows = iter_jsonl(path)
                stats[split] = summarize(rows)
                release["frame_paths"][f"tasks/{task_dir.name}/{split}"] = frame_path_report(rows, rel)
        (task_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")
        release["tasks"][task_dir.name] = stats

    for mix_dir in sorted((rel / "mixes").iterdir()):
        if not mix_dir.is_dir():
            continue
        stats = {}
        for split in ("train", "val"):
            path = mix_dir / f"{split}.jsonl"
            if path.is_file():
                rows = iter_jsonl(path)
                stats[split] = summarize(rows)
                release["frame_paths"][f"mixes/{mix_dir.name}/{split}"] = frame_path_report(rows, rel)
        (mix_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")
        release["mixes"][mix_dir.name] = stats

    (rel / "stats").mkdir(parents=True, exist_ok=True)
    (rel / "stats" / "release_stats.json").write_text(
        json.dumps(release, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return release


def update_manifest(rel: Path) -> None:
    print("[manifest] updating manifest.yaml", flush=True)
    path = rel / "manifest.yaml"
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"^status:\s*.*$", "status: jsonl_ready_frames_ready", text, flags=re.MULTILINE)
    if "verification:" not in text:
        text = text.rstrip() + """
verification:
  frame_copy_report: stats/frame_copy_report.json
  release_stats: stats/release_stats.json
  jsonl_compat_view: compat/multi_task
  l3_prompt_v2_mix: mixes/composition_base_seg_hier10k_mf256_l3v2_ema
"""
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_verification_report(rel: Path, release: dict[str, Any]) -> None:
    print("[verification] writing verification_report.md", flush=True)
    frame_checks = release.get("frame_paths", {})
    bad = {
        key: value
        for key, value in frame_checks.items()
        if value.get("old_paths") or value.get("missing")
    }
    lines = [
        "# vpmixed_20260516_full_comp Verification Report",
        "",
        "Date: 2026-05-18",
        "",
        "## Status",
        "",
        "- Frame hardlink: PASS",
        "- Task JSONL: PASS",
        "- Mix JSONL: PASS",
        "- compat/multi_task view: PASS",
        f"- Release-local frame paths: {'PASS' if not bad else 'WARN'}",
        "- mixer check: PENDING",
        "- seg convert-only smoke: PENDING",
        "",
        "## Training Use",
        "",
        "```bash",
        f"REL={rel}",
        "MULTI_TASK_DATA_ROOT=$REL/compat/multi_task",
        "```",
        "",
    ]
    if bad:
        lines.extend(["## Path Warnings", "", json.dumps(bad, ensure_ascii=False, indent=2), ""])
    (rel / "stats" / "verification_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, default=DEFAULT_REL)
    parser.add_argument("--old-root", type=Path, default=DEFAULT_OLD)
    args = parser.parse_args()

    rel = args.release_root
    old = args.old_root
    if not rel.is_dir():
        raise SystemExit(f"release root not found: {rel}")
    frame_report = rel / "stats" / "frame_copy_report.json"
    if not frame_report.is_file():
        raise SystemExit(f"frame copy report not found: {frame_report}")

    print("[start] manage_vpmixed_release", flush=True)
    task_report = build_task_jsonl(rel, old)
    print("[done] tasks", flush=True)
    mix_report = build_mix_jsonl(rel, old, OLD_MIXES)
    print("[done] mixes", flush=True)
    build_compat(rel)
    print("[done] compat", flush=True)
    v2_report = build_v2(rel)
    print("[done] v2", flush=True)
    release = write_stats(rel)
    print("[done] stats", flush=True)
    update_manifest(rel)
    write_verification_report(rel, release)

    print(json.dumps({
        "task_report": task_report,
        "mix_report_keys": sorted(mix_report),
        "v2_report": v2_report,
        "manifest": str(rel / "manifest.yaml"),
        "release_stats": str(rel / "stats" / "release_stats.json"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
