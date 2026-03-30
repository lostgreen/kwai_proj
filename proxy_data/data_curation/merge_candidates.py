"""
合并多数据源筛选结果 → results/merged/candidates.jsonl

将 ET-Instruct 和 TimeLens 的筛选产出统一为下游 hier_seg_annotation 所需格式。

用法:
    python merge_candidates.py \
        --inputs results/et_instruct_164k/vision_results_keep.jsonl \
                 results/timelens_100k/stage_a_results_keep.jsonl \
        --outdir results/merged

输出格式 (每行一条 JSON):
    {
      "videos": ["source/video.mp4"],
      "duration": 120.5,
      "source": "how_to_step",
      "dataset": "ET-Instruct-164K",
      "events": [ {"description": "...", "start": 10.0, "end": 20.0}, ... ],
      "n_events": 5,
      "metadata": {
        "clip_key": "video_stem",
        "video_id": "video_stem",
        "clip_start": 0,
        "clip_end": 120.5,
        "original_duration": 120.5
      },
      "_curation": { ... }  // 保留筛选阶段的追溯信息
    }
"""

import json
import argparse
import os
import re
from pathlib import Path
from collections import defaultdict


# ── Parsers: 各数据源 → 统一格式 ────────────────────────


def parse_et_instruct(record: dict) -> dict:
    """ET-Instruct-164K: video, tgt, conversations → unified."""
    video_path = record.get("video", "")
    duration = record.get("duration", 0)

    # 解析 events: tgt 是 [s1,e1,s2,e2,...] 的扁平列表
    tgt = record.get("tgt", [])
    events = []
    # 从 conversations 中提取事件描述
    descriptions = []
    for conv in record.get("conversations", []):
        if conv.get("from") == "gpt":
            text = conv.get("value", "")
            # 按 "XX.X - YY.Y seconds, description" 的模式拆分
            parts = re.split(r'(?<=\.)\s+(?=\d+\.\d+\s*-\s*\d+\.\d+\s+seconds)', text)
            for part in parts:
                m = re.match(r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\s+seconds?,\s*(.*)', part.strip())
                if m:
                    descriptions.append({
                        "start": float(m.group(1)),
                        "end": float(m.group(2)),
                        "description": m.group(3).strip().rstrip('.'),
                    })

    # 优先使用从文本解析的 events（包含描述），否则用 tgt
    if descriptions:
        events = descriptions
    elif tgt and len(tgt) >= 2:
        for i in range(0, len(tgt) - 1, 2):
            events.append({"start": float(tgt[i]), "end": float(tgt[i + 1]), "description": ""})

    video_stem = Path(video_path).stem

    return {
        "videos": [video_path],
        "duration": duration,
        "source": record.get("source", "unknown"),
        "dataset": record.get("_origin", {}).get("dataset", "ET-Instruct-164K"),
        "events": events,
        "n_events": len(events),
        "metadata": {
            "clip_key": video_stem,
            "video_id": video_stem,
            "clip_start": 0,
            "clip_end": duration,
            "original_duration": duration,
        },
        "_curation": {
            "stage": record.get("_stage", ""),
            "group": record.get("_group", ""),
            "assessment": record.get("_assessment"),
            "vision": record.get("_vision"),
            "origin": record.get("_origin"),
        },
    }


def parse_timelens(record: dict) -> dict:
    """TimeLens-100K: video_path, events → unified."""
    video_path = record.get("video_path", "")
    duration = record.get("duration", 0)

    # 解析 events: [{query, span: [[s,e]]}]
    raw_events = record.get("events", [])
    events = []
    for ev in raw_events:
        spans = ev.get("span", [])
        for span in spans:
            if len(span) >= 2:
                events.append({
                    "start": float(span[0]),
                    "end": float(span[1]),
                    "description": ev.get("query", ""),
                })

    video_stem = Path(video_path).stem

    return {
        "videos": [video_path],
        "duration": duration,
        "source": record.get("source", "unknown"),
        "dataset": record.get("_origin", {}).get("dataset", "TimeLens-100K"),
        "events": events,
        "n_events": len(events),
        "metadata": {
            "clip_key": video_stem,
            "video_id": video_stem,
            "clip_start": 0,
            "clip_end": duration,
            "original_duration": duration,
        },
        "_curation": {
            "stage": record.get("_stage", ""),
            "group": record.get("_group", ""),
            "assessment": record.get("_assessment"),
            "origin": record.get("_origin"),
        },
    }


# ── Dataset detection ────────────────────────────────────


def detect_dataset(record: dict) -> str:
    """根据字段特征判断数据来源。"""
    if "video" in record and "conversations" in record:
        return "et_instruct"
    if "video_path" in record and "events" in record:
        return "timelens"
    # fallback: 看 _origin
    dataset = record.get("_origin", {}).get("dataset", "")
    if "ET-Instruct" in dataset:
        return "et_instruct"
    if "TimeLens" in dataset:
        return "timelens"
    return "unknown"


PARSERS = {
    "et_instruct": parse_et_instruct,
    "timelens": parse_timelens,
}


# ── Main ─────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="合并多数据源筛选结果为统一候选列表",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="各数据源的筛选结果 JSONL 文件")
    parser.add_argument("--outdir", default="results/merged",
                        help="输出目录")
    parser.add_argument("--dedup", action="store_true", default=True,
                        help="按 video path 去重")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load all records
    all_records = []
    stats = defaultdict(int)
    for path in args.inputs:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                ds = detect_dataset(record)
                parser_fn = PARSERS.get(ds)
                if parser_fn is None:
                    print(f"  WARN: unknown dataset for record, skipping: {record.get('_origin', {})}")
                    stats["skipped_unknown"] += 1
                    continue
                unified = parser_fn(record)
                all_records.append(unified)
                stats[f"loaded_{ds}"] += 1

    print(f"加载总计: {len(all_records)} 条")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")

    # Dedup by video path
    if args.dedup:
        seen = set()
        deduped = []
        for rec in all_records:
            key = rec["videos"][0]
            if key not in seen:
                seen.add(key)
                deduped.append(rec)
        n_dup = len(all_records) - len(deduped)
        if n_dup > 0:
            print(f"去重: 移除 {n_dup} 条重复 (by video path)")
        all_records = deduped

    # Print source distribution summary
    source_counts = defaultdict(int)
    dataset_counts = defaultdict(int)
    for rec in all_records:
        source_counts[rec["source"]] += 1
        dataset_counts[rec["dataset"]] += 1

    print(f"\n最终候选: {len(all_records)} 条")
    print(f"\n按 dataset 分布:")
    for ds, cnt in sorted(dataset_counts.items(), key=lambda x: -x[1]):
        print(f"  {ds:25s}: {cnt:5d} ({cnt/len(all_records)*100:.1f}%)")
    print(f"\n按 source 分布:")
    for src, cnt in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"  {src:25s}: {cnt:5d} ({cnt/len(all_records)*100:.1f}%)")

    # Write output
    out_path = os.path.join(args.outdir, "candidates.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n输出: {out_path} ({len(all_records)} 条)")


if __name__ == "__main__":
    main()
