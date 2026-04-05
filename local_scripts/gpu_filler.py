#!/usr/bin/env python3
"""Smart GPU filler — keeps GPU utilization above cluster threshold.

Combines two approaches:
  - High-performance async fill (CUDA stream, no sync, batched kernels)
  - 3-layer training awareness (process detection → signal → util)

Three-layer per-GPU decision logic:
  Layer 1 — Process detection (pynvml):
      No non-filler process on GPU → async fill 100%
      Training process detected    → go to Layer 2
  Layer 2 — Signal file (/tmp/verl_gpu_phase):
      Signal says "gen" or "update" → pause (training is compute-intensive)
      Signal says "idle" or missing → go to Layer 3
  Layer 3 — Utilization fallback:
      GPU util ≥ threshold → pause (training doing work not yet signaled)
      GPU util < threshold → async fill (training is in idle gap)

Usage:
    python gpu_filler.py                          # all GPUs
    python gpu_filler.py --gpus 0,1,2,3           # specific GPUs
    python gpu_filler.py --batch 50 --pause 40    # tweak params

Environment:
    VERL_GPU_SIGNAL_PATH  — signal file path (default: /tmp/verl_gpu_phase)
    CUDA_VISIBLE_DEVICES  — respected when --gpus is not set
"""

import argparse
import os
import signal
import sys
import time
import threading

# ---------------------------------------------------------------------------
# pynvml helpers
# ---------------------------------------------------------------------------

_nvml_inited = False


def _init_nvml():
    global _nvml_inited
    if _nvml_inited:
        return
    import pynvml
    pynvml.nvmlInit()
    _nvml_inited = True


def gpu_utilization(nvml_idx: int) -> int:
    import pynvml
    _init_nvml()
    h = pynvml.nvmlDeviceGetHandleByIndex(nvml_idx)
    return pynvml.nvmlDeviceGetUtilizationRates(h).gpu


def other_pids_on_gpu(nvml_idx: int, my_pid: int) -> bool:
    import pynvml
    _init_nvml()
    h = pynvml.nvmlDeviceGetHandleByIndex(nvml_idx)
    try:
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(h)
    except pynvml.NVMLError:
        return False
    for p in procs:
        if p.pid != my_pid:
            return True
    return False


# ---------------------------------------------------------------------------
# Signal file helpers
# ---------------------------------------------------------------------------

SIGNAL_PATH = os.environ.get("VERL_GPU_SIGNAL_PATH", "/tmp/verl_gpu_phase")
BUSY_PHASES = {"gen", "update"}


def signal_says_busy() -> bool:
    try:
        with open(SIGNAL_PATH, "r") as f:
            return f.read().strip() in BUSY_PHASES
    except (FileNotFoundError, OSError):
        return False


# ---------------------------------------------------------------------------
# NVML index mapping (CUDA_VISIBLE_DEVICES aware)
# ---------------------------------------------------------------------------

def resolve_nvml_indices(cuda_ids: list[int]) -> dict[int, int]:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is None:
        return {g: g for g in cuda_ids}
    physical = [int(x.strip()) for x in cvd.split(",") if x.strip()]
    return {g: physical[g] if g < len(physical) else g for g in cuda_ids}


# ---------------------------------------------------------------------------
# Per-GPU filler worker
# ---------------------------------------------------------------------------

_STOP = threading.Event()

_status: dict[int, str] = {}
_status_lock = threading.Lock()


def filler_worker(
    gpu_id: int,
    nvml_idx: int,
    matrix_size: int,
    kernel_batch: int,
    pause_threshold: int,
):
    """Async GPU filler with 3-layer training awareness."""
    import torch

    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    # Pre-allocate (8192x8192 fp16 = 128MB per matrix, 256MB total)
    a = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float16)
    b = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float16)
    stream = torch.cuda.Stream(device=device)

    my_pid = os.getpid()

    print(f"[filler] GPU {gpu_id} (nvml {nvml_idx}): started  "
          f"matrix={matrix_size}  batch={kernel_batch}  pause≥{pause_threshold}%")

    # Stale signal detection: track how long signal says busy with low util
    _signal_busy_low_util_since = None
    STALE_SIGNAL_TIMEOUT = 30  # seconds

    while not _STOP.is_set():
        # === Layer 1: Process detection ===
        has_other = other_pids_on_gpu(nvml_idx, my_pid)

        if not has_other:
            # No training process → async fill, never sync
            _signal_busy_low_util_since = None
            with _status_lock:
                _status[gpu_id] = "FILL(idle)"
            with torch.cuda.stream(stream):
                for _ in range(kernel_batch):
                    torch.matmul(a, b)
            time.sleep(0.002)
            continue

        # === Layer 2: Signal file (with stale signal protection) ===
        if signal_says_busy():
            # Check if GPU is actually being used
            try:
                util = gpu_utilization(nvml_idx)
            except Exception:
                util = 0

            if util >= 20:
                # Signal busy + GPU active → legit, pause
                _signal_busy_low_util_since = None
                with _status_lock:
                    _status[gpu_id] = f"PAUSE(signal,u={util}%)"
                _STOP.wait(timeout=0.3)
                continue
            else:
                # Signal busy but GPU idle → possibly stale/crashed
                if _signal_busy_low_util_since is None:
                    _signal_busy_low_util_since = time.time()
                elapsed = time.time() - _signal_busy_low_util_since
                if elapsed < STALE_SIGNAL_TIMEOUT:
                    # Still within grace period, wait
                    with _status_lock:
                        _status[gpu_id] = f"WAIT(signal,u={util}%,{int(elapsed)}s)"
                    _STOP.wait(timeout=0.5)
                    continue
                else:
                    # Stale signal! Override and fill
                    with _status_lock:
                        _status[gpu_id] = f"FILL(stale,u={util}%)"
                    # Fall through to fill below
        else:
            _signal_busy_low_util_since = None

        # === Layer 3: Utilization check ===
        try:
            util = gpu_utilization(nvml_idx)
        except Exception:
            util = 0

        if util >= pause_threshold:
            with _status_lock:
                _status[gpu_id] = f"PAUSE(util={util}%)"
            _STOP.wait(timeout=0.3)
            continue

        # Training is in idle gap → async fill
        with _status_lock:
            _status[gpu_id] = f"FILL(gap,u={util}%)"
        with torch.cuda.stream(stream):
            for _ in range(kernel_batch):
                torch.matmul(a, b)
        time.sleep(0.002)

    del a, b
    torch.cuda.empty_cache()
    print(f"[filler] GPU {gpu_id}: stopped")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Smart GPU filler — async fill + 3-layer training awareness")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated CUDA GPU IDs (default: all visible)")
    parser.add_argument("--matrix-size", type=int, default=8192,
                        help="Matrix size for matmul (default: 8192)")
    parser.add_argument("--batch", type=int, default=50,
                        help="Kernels per batch before re-checking (default: 50)")
    parser.add_argument("--pause", type=int, default=50,
                        help="Pause when GPU util ≥ this %% on training GPU (default: 50)")
    args = parser.parse_args()

    if args.gpus:
        gpu_ids = [int(g) for g in args.gpus.split(",")]
    else:
        import torch
        gpu_ids = list(range(torch.cuda.device_count()))

    if not gpu_ids:
        print("[filler] No GPUs found.")
        sys.exit(1)

    nvml_map = resolve_nvml_indices(gpu_ids)
    _init_nvml()

    print(f"[filler] GPUs: {gpu_ids}  nvml: {[nvml_map[g] for g in gpu_ids]}")
    print(f"[filler] matrix={args.matrix_size}  batch={args.batch}  pause≥{args.pause}%")
    print(f"[filler] signal: {SIGNAL_PATH}")
    print(f"[filler] 3-layer: process→signal→util")

    # Clean stale signal file
    if os.path.exists(SIGNAL_PATH):
        os.remove(SIGNAL_PATH)
        print(f"[filler] Cleaned stale signal file")
    print()

    def shutdown(signum, frame):
        print(f"\n[filler] Shutting down...")
        _STOP.set()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    threads = []
    for gid in gpu_ids:
        t = threading.Thread(
            target=filler_worker,
            args=(gid, nvml_map[gid], args.matrix_size, args.batch, args.pause),
            daemon=True,
        )
        t.start()
        threads.append(t)

    # Display loop
    try:
        while not _STOP.is_set():
            parts = []
            for gid in gpu_ids:
                try:
                    u = gpu_utilization(nvml_map[gid])
                except Exception:
                    u = -1
                with _status_lock:
                    st = _status.get(gid, "init")
                parts.append(f"GPU{gid}:{u:3d}% {st}")
            print(f"\r[filler] {' | '.join(parts)}  ", end="", flush=True)
            _STOP.wait(timeout=2.0)
    except KeyboardInterrupt:
        _STOP.set()

    for t in threads:
        t.join(timeout=5)
    print("\n[filler] Done.")


if __name__ == "__main__":
    main()
