from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def build_local_screen_command(args: argparse.Namespace) -> list[str]:
    script = Path(__file__).resolve().parents[1] / "shared" / "local_screen.py"
    command = [
        sys.executable,
        str(script),
        "--input_jsonl",
        str(args.input_jsonl),
        "--output_jsonl",
        str(args.output_jsonl),
        "--keep_jsonl",
        str(args.keep_jsonl),
        "--reject_jsonl",
        str(args.reject_jsonl),
        "--model_path",
        str(args.model_path),
        "--gpu_memory_utilization",
        str(args.gpu_memory_utilization),
        "--batch_size",
        str(args.batch_size),
        "--max_num_batched_tokens",
        str(args.max_num_batched_tokens),
        "--unified",
    ]
    if args.tensor_parallel_size > 1:
        command.extend(["--tensor_parallel_size", str(args.tensor_parallel_size)])
    if args.shard_id >= 0 and args.num_shards > 1:
        command.extend(["--shard_id", str(args.shard_id), "--num_shards", str(args.num_shards)])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="Run optional local model scoring for curated videos")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--keep-jsonl", required=True)
    parser.add_argument("--reject-jsonl", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=-1)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    args = parser.parse_args()

    subprocess.run(build_local_screen_command(args), check=True)


if __name__ == "__main__":
    main()
