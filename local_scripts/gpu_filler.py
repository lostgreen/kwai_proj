#!/usr/bin/env python3
"""Smart GPU filler — keeps GPU utilization above cluster threshold.

Combines two approaches:
  - High-performance async fill (CUDA stream, no sync, batched kernels)
  - 3-layer training awareness (process detection → signal → util)

Three-layer per-GPU decision logic:
  Layer 1 — Process detection (pynvml):
      No non-filler process on GPU → big matmul fill 100%
      Training process detected    → go to Layer 2
  Layer 2 — Util-based with escalation:
      High util (≥threshold)  → light fill (small matrix)
      Low util, brief (<2s)   → small matrix gap fill
      Low util, sustained (≥2s) → big matrix (process holds context but idle)

  Three-tier fill intensity:
    - FULL:   big matrix (8192) × full batch  — idle GPU or sustained idle process
    - MEDIUM: small matrix (1024) × high batch — brief training gap
    - LIGHT:  small matrix (1024) × low batch  — high-util training (+6% util)

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
    """Check if any OTHER live process has a CUDA context on this GPU."""
    import pynvml
    _init_nvml()
    h = pynvml.nvmlDeviceGetHandleByIndex(nvml_idx)
    try:
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(h)
    except pynvml.NVMLError:
        return False
    for p in procs:
        if p.pid != my_pid:
            # Verify the process actually exists (guard against zombie CUDA contexts
            # left behind by killed processes — pynvml still reports them)
            try:
                os.kill(p.pid, 0)  # signal 0 = existence check only
                return True
            except (ProcessLookupError, PermissionError):
                pass  # Dead process, zombie CUDA context — ignore
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
    train_matrix: int,
    train_batch: int,
    train_sleep: float,
    gap_matrix: int,
):
    """Async GPU filler with 3-layer training awareness.

    Four-tier fill strategy:
      - Idle GPU (no training):    big matrix (8192) full blast → 95%+ util
      - Training + sustained idle: big matrix half-batch → fill the gap aggressively
      - Training + low util gap:   medium matrix (4096) → 60-70% util, safe
      - Training + high util:      small matrix (1024) light fill → +6% util
    """
    import torch

    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    # Big matrices for idle fill and low-util gap fill
    a_big = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float16)
    b_big = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float16)
    # Medium matrices for gap/mid fill during training transitions
    a_med = torch.randn(gap_matrix, gap_matrix, device=device, dtype=torch.float16)
    b_med = torch.randn(gap_matrix, gap_matrix, device=device, dtype=torch.float16)
    # Small matrices for light fill during high-util training
    a_sm = torch.randn(train_matrix, train_matrix, device=device, dtype=torch.float16)
    b_sm = torch.randn(train_matrix, train_matrix, device=device, dtype=torch.float16)
    stream = torch.cuda.Stream(device=device)

    my_pid = os.getpid()

    # Diagnostic: show what PIDs are on this GPU at startup
    import pynvml as _pynvml
    _h = _pynvml.nvmlDeviceGetHandleByIndex(nvml_idx)
    try:
        _procs = _pynvml.nvmlDeviceGetComputeRunningProcesses(_h)
        _pids = [p.pid for p in _procs]
    except _pynvml.NVMLError:
        _pids = []
    print(f"[filler] GPU {gpu_id} (nvml {nvml_idx}): started  "
          f"idle={matrix_size}x{kernel_batch}  "
          f"gap={gap_matrix}  "
          f"train_light={train_matrix}x{train_batch}+{train_sleep*1000:.0f}ms  "
          f"pause≥{pause_threshold}%  "
          f"my_pid={my_pid} gpu_pids={_pids}")

    # Unified low-util timer: tracks how long GPU stays at low util
    # while training process holds CUDA context but isn't computing.
    # After IDLE_ESCALATION_TIMEOUT, switch to big matrix (safe — GPU is idle).
    _low_util_since = None
    IDLE_ESCALATION_TIMEOUT = 2  # seconds before escalating to big matrix
    STALE_SIGNAL_TIMEOUT = 30

    while not _STOP.is_set():
        # === Layer 1: Process detection ===
        has_other = other_pids_on_gpu(nvml_idx, my_pid)

        if not has_other:
            # No training process → big matmul, full blast
            _low_util_since = None
            with _status_lock:
                _status[gpu_id] = "FILL(idle)"
            with torch.cuda.stream(stream):
                for _ in range(kernel_batch):
                    torch.matmul(a_big, b_big)
            time.sleep(0.002)
            continue

        # === Shared util check ===
        is_busy_signal = signal_says_busy()

        try:
            util = gpu_utilization(nvml_idx)
        except Exception:
            util = 0

        # --- High util: any process actively computing → light fill ---
        if util >= pause_threshold:
            _low_util_since = None
            sig_tag = "busy" if is_busy_signal else "nosig"
            with _status_lock:
                _status[gpu_id] = f"LFILL({sig_tag},u={util}%)"
            with torch.cuda.stream(stream):
                for _ in range(train_batch):
                    torch.matmul(a_sm, b_sm)
            time.sleep(train_sleep)
            continue

        # --- Low util (<pause_threshold): process holds context but may be idle ---
        if _low_util_since is None:
            _low_util_since = time.time()
        elapsed = time.time() - _low_util_since

        if elapsed >= IDLE_ESCALATION_TIMEOUT:
            # Process has been idle for 2+s → GPU is effectively free.
            # Big matrix is safe here — no active compute to compete with.
            # Use fewer kernels than idle path (15 vs 50) so if training
            # resumes mid-batch, max queue interference is ~50ms not ~175ms.
            if elapsed >= STALE_SIGNAL_TIMEOUT:
                tag = f"FILL(stale,u={util}%)"
            else:
                sig_tag = "sig" if is_busy_signal else "nosig"
                tag = f"FILL(idle-esc,{sig_tag},u={util}%,{int(elapsed)}s)"
            with _status_lock:
                _status[gpu_id] = tag
            esc_batch = min(kernel_batch, 15)
            with torch.cuda.stream(stream):
                for _ in range(esc_batch):
                    torch.matmul(a_big, b_big)
            time.sleep(0.002)
        elif util < 20:
            # Brief gap (<2s), very low util → medium matrix for meaningful SM occupancy
            sig_tag = "gap" if is_busy_signal else "nosig"
            with _status_lock:
                _status[gpu_id] = f"GFILL({sig_tag},u={util}%,{int(elapsed)}s)"
            with torch.cuda.stream(stream):
                for _ in range(10):
                    torch.matmul(a_med, b_med)
            time.sleep(0.001)
        else:
            # Mid util (20-pause_threshold), brief → medium matrix moderate batch
            sig_tag = "mid" if is_busy_signal else "nosig-mid"
            with _status_lock:
                _status[gpu_id] = f"GFILL({sig_tag},u={util}%)"
            with torch.cuda.stream(stream):
                for _ in range(5):
                    torch.matmul(a_med, b_med)
            time.sleep(0.002)

    del a_big, b_big, a_med, b_med, a_sm, b_sm
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
                        help="Matrix size for idle fill (default: 8192)")
    parser.add_argument("--batch", type=int, default=50,
                        help="Kernels per batch for idle fill (default: 50)")
    parser.add_argument("--train-matrix", type=int, default=1024,
                        help="Matrix size during training (default: 1024)")
    parser.add_argument("--train-batch", type=int, default=600,
                        help="Kernels per batch during training (default: 600)")
    parser.add_argument("--train-sleep", type=float, default=0.003,
                        help="Sleep between batches during training in sec (default: 0.003)")
    parser.add_argument("--pause", type=int, default=50,
                        help="Util threshold for reduced fill (default: 50)")
    parser.add_argument("--gap-matrix", type=int, default=4096,
                        help="Matrix size for gap/mid fill during training transitions (default: 4096)")
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
    print(f"[filler] idle: matrix={args.matrix_size} batch={args.batch}")
    print(f"[filler] gap: matrix={args.gap_matrix}")
    print(f"[filler] train: matrix={args.train_matrix} batch={args.train_batch} sleep={args.train_sleep*1000:.0f}ms")
    print(f"[filler] pause≥{args.pause}%  signal: {SIGNAL_PATH}")
    print(f"[filler] 3-layer: process→signal→util")

    # Clean stale signal file
    if os.path.exists(SIGNAL_PATH):
        os.remove(SIGNAL_PATH)
        print(f"[filler] Cleaned stale signal file")

    # Kill old filler instances (prevent them from being detected as "training")
    import subprocess
    my_pid = os.getpid()
    killed_any = False
    result = subprocess.run(
        ["pgrep", "-f", "gpu_filler.py"],
        capture_output=True, text=True
    )
    for line in result.stdout.strip().split("\n"):
        pid_str = line.strip()
        if pid_str and int(pid_str) != my_pid:
            try:
                os.kill(int(pid_str), signal.SIGTERM)
                print(f"[filler] Killed old filler PID {pid_str}")
                killed_any = True
            except ProcessLookupError:
                pass
    if killed_any:
        print("[filler] Waiting 3s for old filler to release GPU resources...")
        time.sleep(3)
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
            args=(gid, nvml_map[gid], args.matrix_size, args.batch,
                  args.pause, args.train_matrix, args.train_batch,
                  args.train_sleep, args.gap_matrix),
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
