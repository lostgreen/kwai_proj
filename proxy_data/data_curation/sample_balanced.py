#!/usr/bin/env python3
"""
sample_balanced.py — 从 candidates.jsonl 中按 source 域均衡采样。

用法:
    python sample_balanced.py \
        --input results/merged/candidates.jsonl \
        --output-dir results/merged/sampled \
        --total 1000 --dev-n 100 --min-per-source 10 \
        --seed 42

输出:
    sampled/sampled_{total}.jsonl   — 总样本（视频路径已解析为绝对路径）
    sampled/dev_{dev_n}.jsonl       — 开发子集（从总样本中均衡抽取）
    sampled/distribution.txt        — source × dataset 分布统计

视频路径解析:
    根据 dataset 字段自动拼接 video_root:
      ET-Instruct-164K → VIDEO_ROOT_ET (环境变量或默认值)
      TimeLens-100K    → VIDEO_ROOT_TL (环境变量或默认值)
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path


# ── 视频根目录 ──
DEFAULT_VIDEO_ROOTS = {
    "ET-Instruct-164K": "/m2v_intern/xuboshen/zgw/data/ET-Instruct-164K/videos",
    "TimeLens-100K": "/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/TimeLens-100K/video_shards",
}


def resolve_video_path(record: dict, video_roots: dict[str, str]) -> str:
    """将 videos[0] 的相对路径解析为绝对路径。"""
    rel = record["videos"][0]
    dataset = record.get("dataset", "")
    root = video_roots.get(dataset, "")
    if root:
        return str(Path(root) / rel)
    return rel


def load_candidates(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def print_distribution(records: list[dict], label: str = "") -> None:
    by_source: dict[str, int] = defaultdict(int)
    by_dataset: dict[str, int] = defaultdict(int)
    by_source_dataset: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for r in records:
        src = r.get("source", "unknown")
        ds = r.get("dataset", "unknown")
        by_source[src] += 1
        by_dataset[ds] += 1
        by_source_dataset[src][ds] += 1

    title = f"Distribution: {label} (N={len(records)})" if label else f"Distribution (N={len(records)})"
    print(f"\n{'=' * 60}")
    print(title)
    print(f"{'=' * 60}")
    print(f"\n{'Dataset':<25} {'Count':>6}")
    print("-" * 35)
    for ds, cnt in sorted(by_dataset.items(), key=lambda x: -x[1]):
        print(f"  {ds:<23} {cnt:>6}")

    print(f"\n{'Source':<25} {'Count':>6}  {'Dataset':<25}")
    print("-" * 60)
    for src, cnt in sorted(by_source.items(), key=lambda x: -x[1]):
        datasets = ", ".join(f"{d}({n})" for d, n in by_source_dataset[src].items())
        print(f"  {src:<23} {cnt:>6}  {datasets}")
    print()


def balanced_sample(
    records: list[dict],
    total: int,
    min_per_source: int,
    rng: random.Random,
) -> list[dict]:
    """按 source 域均衡采样。

    1. 每个 source 至少选 min_per_source 条（不足则全选）
    2. 剩余名额按各 source 的原始比例分配
    3. 最终打乱顺序
    """
    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_source[r.get("source", "unknown")].append(r)

    # 每个 source 内部打乱
    for src in by_source:
        rng.shuffle(by_source[src])

    selected: list[dict] = []
    remaining_pool: dict[str, list[dict]] = {}

    # Phase 1: 保证最低代表
    guaranteed_total = 0
    for src, pool in by_source.items():
        n_take = min(min_per_source, len(pool))
        selected.extend(pool[:n_take])
        leftover = pool[n_take:]
        if leftover:
            remaining_pool[src] = leftover
        guaranteed_total += n_take

    # Phase 2: 按比例分配剩余名额
    slots_left = total - guaranteed_total
    if slots_left > 0 and remaining_pool:
        total_remaining = sum(len(v) for v in remaining_pool.values())
        for src, pool in sorted(remaining_pool.items()):
            n_alloc = min(len(pool), round(slots_left * len(pool) / max(total_remaining, 1)))
            selected.extend(pool[:n_alloc])
            pool[:] = pool[n_alloc:]

        # 补齐（浮点截断可能有差额）
        still_left = total - len(selected)
        if still_left > 0:
            flat_remaining = [r for pool in remaining_pool.values() for r in pool]
            rng.shuffle(flat_remaining)
            selected.extend(flat_remaining[:still_left])

    # 截断至目标总数
    selected = selected[:total]
    rng.shuffle(selected)
    return selected


def sample_dev_subset(
    records: list[dict],
    dev_n: int,
    rng: random.Random,
) -> list[dict]:
    """从总样本中均衡抽取 dev 子集。"""
    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_source[r.get("source", "unknown")].append(r)

    n_sources = len(by_source)
    base_per = max(1, dev_n // n_sources)

    dev: list[dict] = []
    for src, pool in by_source.items():
        rng.shuffle(pool)
        dev.extend(pool[:min(base_per, len(pool))])

    # 补齐
    if len(dev) < dev_n:
        used = set(id(r) for r in dev)
        remaining = [r for r in records if id(r) not in used]
        rng.shuffle(remaining)
        dev.extend(remaining[:dev_n - len(dev)])

    dev = dev[:dev_n]
    rng.shuffle(dev)
    return dev


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 candidates.jsonl 中按 source 域均衡采样",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True,
                        help="输入 candidates.jsonl 路径")
    parser.add_argument("--exclude", default="",
                        help="排除 JSONL 路径（已采样过的记录，按 videos[0] 去重）")
    parser.add_argument("--output-dir", required=True,
                        help="输出目录")
    parser.add_argument("--total", type=int, default=1000,
                        help="总采样数")
    parser.add_argument("--dev-n", type=int, default=100,
                        help="开发子集大小（从总样本中抽取）")
    parser.add_argument("--min-per-source", type=int, default=10,
                        help="每个 source 域最低保证样本数")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--video-root-et", default=None,
                        help="ET-Instruct-164K 视频根目录 (默认: $VIDEO_ROOT_ET 或内置路径)")
    parser.add_argument("--video-root-tl", default=None,
                        help="TimeLens-100K 视频根目录 (默认: $VIDEO_ROOT_TL 或内置路径)")
    parser.add_argument("--resolve-paths", action="store_true", default=True,
                        help="将 videos[0] 解析为绝对路径")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # 视频根目录
    video_roots = dict(DEFAULT_VIDEO_ROOTS)
    if args.video_root_et or os.environ.get("VIDEO_ROOT_ET"):
        video_roots["ET-Instruct-164K"] = args.video_root_et or os.environ["VIDEO_ROOT_ET"]
    if args.video_root_tl or os.environ.get("VIDEO_ROOT_TL"):
        video_roots["TimeLens-100K"] = args.video_root_tl or os.environ["VIDEO_ROOT_TL"]

    # 加载
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: {input_path} not found", file=sys.stderr)
        sys.exit(1)
    records = load_candidates(input_path)
    print(f"Loaded {len(records)} candidates from {input_path}")

    # 排除已采样的记录
    if args.exclude:
        exclude_path = Path(args.exclude)
        if exclude_path.exists():
            excluded = load_candidates(exclude_path)
            exclude_videos = {r["videos"][0] for r in excluded if r.get("videos")}
            before = len(records)
            records = [r for r in records if r.get("videos", [""])[0] not in exclude_videos]
            print(f"Excluded {before - len(records)} already-sampled records ({len(records)} remaining)")
        else:
            print(f"WARNING: exclude file not found: {exclude_path}, skipping exclusion", file=sys.stderr)

    print_distribution(records, "All Candidates")

    # 采样
    sampled = balanced_sample(records, args.total, args.min_per_source, rng)
    print_distribution(sampled, f"Sampled {len(sampled)}")

    dev = sample_dev_subset(sampled, args.dev_n, rng)
    print_distribution(dev, f"Dev Subset {len(dev)}")

    # 解析视频路径
    if args.resolve_paths:
        for r in sampled:
            r["videos"] = [resolve_video_path(r, video_roots)]

    # 输出
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sampled_path = out_dir / f"sampled_{len(sampled)}.jsonl"
    dev_path = out_dir / f"dev_{len(dev)}.jsonl"
    dist_path = out_dir / "distribution.txt"

    for path, data in [(sampled_path, sampled), (dev_path, dev)]:
        with open(path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Written: {path} ({len(data)} records)")

    # 分布统计写入文件
    import io
    buf = io.StringIO()
    _orig_stdout = sys.stdout
    sys.stdout = buf
    print_distribution(records, "All Candidates")
    print_distribution(sampled, f"Sampled {len(sampled)}")
    print_distribution(dev, f"Dev Subset {len(dev)}")
    sys.stdout = _orig_stdout
    dist_path.write_text(buf.getvalue(), encoding="utf-8")
    print(f"Written: {dist_path}")


if __name__ == "__main__":
    main()
