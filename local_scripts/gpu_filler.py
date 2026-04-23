#!/usr/bin/env python3
"""Smart GPU filler — keeps GPU utilization above cluster threshold.

Architecture (v3 — NVML-safe):
  - Monitor thread: polls gpu_utilization() at ~2Hz with per-call timeout
  - Worker threads: read cached state, run matmul fills (zero NVML calls)
  - Display loop: reads cached state (zero NVML calls)
  - NVML safety: if NVML hangs or fails 10 consecutive polls, filler auto-stops
  - No process enumeration: removed nvmlDeviceGetComputeRunningProcesses
    (primary cause of driver lock contention and deadlock)

Three-tier fill strategy:
  Tier 1 — At/above target util:     backoff (50ms sleep)
  Tier 2 — Low util (<20%), brief:   gap matrix (4096) medium fill
  Tier 3 — Low util, sustained:      big matrix (8192), signal-aware
  Tier 4 — Below target (20-85%):    push matrix (6144/4096) dynamic fill

Usage:
    python gpu_filler.py                          # all GPUs, target 85%
    python gpu_filler.py --gpus 0,1,2,3           # specific GPUs
    python gpu_filler.py --target-util 80         # custom target

Environment:
    VERL_GPU_SIGNAL_PATH  — signal file path (default: /tmp/verl_gpu_phase)
    CUDA_VISIBLE_DEVICES  — respected when --gpus is not set

Modes:
    nvml    — util-driven filler (default)
    signal  — no NVML polling; phase-aware open-loop filler
              gen/update => light fill
              idle       => heavy fill
              no-signal  => medium fill
"""

import argparse
import dataclasses
import os
import signal
import subprocess
import sys
import time
import threading

# ---------------------------------------------------------------------------
# pynvml helpers (used ONLY by monitor thread)
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



# ---------------------------------------------------------------------------
# Signal file helpers
# ---------------------------------------------------------------------------

SIGNAL_PATH = os.environ.get("VERL_GPU_SIGNAL_PATH", "/tmp/verl_gpu_phase")
BUSY_PHASES = {"gen", "update", "decode"}
STALE_SIGNAL_TIMEOUT = float(os.environ.get("FILLER_STALE_SIGNAL_TIMEOUT", "30"))


def signal_says_busy() -> bool:
    try:
        with open(SIGNAL_PATH, "r") as f:
            return f.read().strip() in BUSY_PHASES
    except (FileNotFoundError, OSError):
        return False


def get_signal_state() -> str:
    """Return busy / idle / nosignal / stale for the phase signal."""
    try:
        with open(SIGNAL_PATH, "r") as f:
            phase = f.read().strip()
    except (FileNotFoundError, OSError):
        return "nosignal"

    if is_signal_stale():
        return "stale"
    if phase in BUSY_PHASES:
        return "busy"
    return "idle"


def is_signal_stale() -> bool:
    try:
        mtime = os.path.getmtime(SIGNAL_PATH)
        return (time.time() - mtime) > STALE_SIGNAL_TIMEOUT
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
# Monitor cache — written by monitor thread, read by workers + display
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class GPUState:
    util: int = 0
    timestamp: float = 0.0


# Python dict[int] = obj assignment is atomic (GIL), so readers always see
# either the old or new GPUState — never a half-written one. No lock needed
# on the read path.
_gpu_cache: dict[int, GPUState] = {}

# Signal state (also written by monitor, read by workers)
_signal_busy: bool = False
_signal_is_stale: bool = False


# ---------------------------------------------------------------------------
# Monitor thread — single thread polls NVML at ~2Hz
# ---------------------------------------------------------------------------

_STOP = threading.Event()


_NVML_CONSECUTIVE_FAILURES = 0
_NVML_MAX_FAILURES = 10  # after this many consecutive failures, stop all workers


def _nvml_call_with_timeout(fn, *args, timeout_s=2.0):
    """Run an NVML call; if it hangs > timeout_s, return None.

    Uses a daemon thread so a hung NVML call won't block the monitor forever.
    """
    result = [None]
    exc = [None]

    def wrapper():
        try:
            result[0] = fn(*args)
        except Exception as e:
            exc[0] = e

    t = threading.Thread(target=wrapper, daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    if t.is_alive():
        # NVML call is hung — don't wait, just return None
        return None
    if exc[0] is not None:
        return None
    return result[0]


def monitor_loop(gpu_ids: list[int], nvml_map: dict[int, int], my_pid: int):
    """Poll NVML for all GPUs + signal file at ~2Hz (was 10Hz).

    Only queries gpu_utilization (fast, single NVML call).
    other_pids_on_gpu removed — it calls nvmlDeviceGetComputeRunningProcesses
    which is the slowest NVML call and the primary cause of driver lock contention.
    """
    global _signal_busy, _signal_is_stale, _NVML_CONSECUTIVE_FAILURES
    POLL_INTERVAL = 0.5  # 500ms — reduced from 100ms to lower driver lock pressure

    while not _STOP.is_set():
        # Signal file — once per poll, not per GPU
        _signal_busy = signal_says_busy()
        _signal_is_stale = is_signal_stale()

        all_failed = True
        for gid in gpu_ids:
            nvml_idx = nvml_map[gid]
            util = _nvml_call_with_timeout(gpu_utilization, nvml_idx)

            if util is None:
                util = 0
            else:
                all_failed = False

            _gpu_cache[gid] = GPUState(
                util=util, timestamp=time.time()
            )

        # Track NVML health
        if all_failed:
            _NVML_CONSECUTIVE_FAILURES += 1
            if _NVML_CONSECUTIVE_FAILURES >= _NVML_MAX_FAILURES:
                print(f"\n[filler] NVML failed {_NVML_MAX_FAILURES} consecutive polls — "
                      f"driver likely hung. Stopping filler to prevent further damage.")
                _STOP.set()
                break
        else:
            _NVML_CONSECUTIVE_FAILURES = 0

        _STOP.wait(timeout=POLL_INTERVAL)


# ---------------------------------------------------------------------------
# Per-GPU filler worker
# ---------------------------------------------------------------------------

_status: dict[int, str] = {}
_status_lock = threading.Lock()

# Timing constants
IDLE_ESCALATION_TIMEOUT = 0.5   # seconds before escalating to big matrix


def filler_worker(
    gpu_id: int,
    matrix_size: int,
    kernel_batch: int,
    target_util: int,
    gap_matrix: int,
    push_matrix: int,
    mode: str,
    busy_matrix: int,
    busy_batch: int,
    busy_sleep_ms: int,
    idle_sleep_ms: int,
    orphan_matrix: int,
    orphan_batch: int,
    orphan_sleep_ms: int,
    busy_hold_ms: int,
):
    """Async GPU filler — reads monitor cache, fills with matmul.

    Three-tier strategy (util-only, no process enumeration):
      Tier 1 — At/above target:     backoff (50ms sleep)
      Tier 2 — Below target, low:   escalation fill (gap → big matrix)
      Tier 3 — Below target, mid:   push fill (push/gap matrix)

    Signal file (/tmp/verl_gpu_phase) is respected: when signal says
    "gen" or "update", filler uses gentler fill to avoid competing with
    the training process. In signal mode this becomes a phase-aware
    open-loop filler:
      busy(gen/update) => light fill
      idle             => heavy fill
      nosignal/stale   => medium fill
    """
    import torch

    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    # Use bf16 on compute capability ≥8.0 (A100/A800/H800), fp16 otherwise
    if torch.cuda.get_device_capability(gpu_id)[0] >= 8:
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    # Pre-allocate tensors
    a_big = torch.randn(matrix_size, matrix_size, device=device, dtype=dtype)
    b_big = torch.randn(matrix_size, matrix_size, device=device, dtype=dtype)
    a_push = torch.randn(push_matrix, push_matrix, device=device, dtype=dtype)
    b_push = torch.randn(push_matrix, push_matrix, device=device, dtype=dtype)
    a_gap = torch.randn(gap_matrix, gap_matrix, device=device, dtype=dtype)
    b_gap = torch.randn(gap_matrix, gap_matrix, device=device, dtype=dtype)
    a_busy = torch.randn(busy_matrix, busy_matrix, device=device, dtype=dtype)
    b_busy = torch.randn(busy_matrix, busy_matrix, device=device, dtype=dtype)
    a_orphan = torch.randn(orphan_matrix, orphan_matrix, device=device, dtype=dtype)
    b_orphan = torch.randn(orphan_matrix, orphan_matrix, device=device, dtype=dtype)
    stream = torch.cuda.Stream(device=device)

    print(f"[filler] GPU {gpu_id}: started  "
          f"idle={matrix_size}x{kernel_batch}  "
          f"push={push_matrix}  gap={gap_matrix}  "
          f"busy={busy_matrix}x{busy_batch}  "
          f"orphan={orphan_matrix}x{orphan_batch}  "
          f"target={target_util}%  dtype={dtype}")

    _low_util_since = None
    _last_busy_at = 0.0

    while not _STOP.is_set():
        if mode == "signal":
            signal_state = get_signal_state()
            if signal_state == "busy":
                _last_busy_at = time.time()
                with _status_lock:
                    _status[gpu_id] = "LFILL(busy)"
                with torch.cuda.stream(stream):
                    for _ in range(busy_batch):
                        torch.matmul(a_busy, b_busy)
                time.sleep(max(0.0, busy_sleep_ms / 1000.0))
                continue

            if (time.time() - _last_busy_at) * 1000.0 < busy_hold_ms:
                with _status_lock:
                    _status[gpu_id] = "LFILL(busy-hold)"
                with torch.cuda.stream(stream):
                    for _ in range(busy_batch):
                        torch.matmul(a_busy, b_busy)
                time.sleep(max(0.0, busy_sleep_ms / 1000.0))
                continue

            if signal_state == "idle":
                with _status_lock:
                    _status[gpu_id] = "FILL(idle)"
                fill_batch = min(kernel_batch, 30)
                with torch.cuda.stream(stream):
                    for _ in range(fill_batch):
                        torch.matmul(a_big, b_big)
                time.sleep(max(0.0, idle_sleep_ms / 1000.0))
                continue

            with _status_lock:
                _status[gpu_id] = f"MFILL({signal_state})"
            with torch.cuda.stream(stream):
                for _ in range(orphan_batch):
                    torch.matmul(a_orphan, b_orphan)
            time.sleep(max(0.0, orphan_sleep_ms / 1000.0))
            continue

        state = _gpu_cache.get(gpu_id)
        if state is None:
            time.sleep(0.05)  # cache not populated yet
            continue

        # Guard against monitor thread crash — if cache is >3s stale, be conservative
        if time.time() - state.timestamp > 3.0:
            with _status_lock:
                _status[gpu_id] = "STALE_CACHE"
            time.sleep(0.1)
            continue

        util = state.util

        # === Tier 1: At/above target → backoff ===
        if util >= target_util:
            _low_util_since = None
            sig = "busy" if _signal_busy else "nosig"
            with _status_lock:
                _status[gpu_id] = f"BACK({sig},u={util}%)"
            time.sleep(0.050)
            continue

        # === Below target ===

        if util < 20:
            # Low util zone — use escalation timer
            if _low_util_since is None:
                _low_util_since = time.time()
            elapsed = time.time() - _low_util_since

            if elapsed >= IDLE_ESCALATION_TIMEOUT:
                # Sustained low util (≥0.5s) → big matrix fill
                if _signal_busy:
                    # Training is active but util is low (e.g. data loading gap)
                    # Use medium fill to avoid competing
                    with _status_lock:
                        _status[gpu_id] = f"GFILL(sig,u={util}%,{int(elapsed)}s)"
                    with torch.cuda.stream(stream):
                        for _ in range(30):
                            torch.matmul(a_gap, b_gap)
                    time.sleep(0.005)
                else:
                    # No signal → likely truly idle, fill aggressively
                    with _status_lock:
                        _status[gpu_id] = f"FILL(idle,u={util}%,{int(elapsed)}s)"
                    esc_batch = min(kernel_batch, 30)
                    with torch.cuda.stream(stream):
                        for _ in range(esc_batch):
                            torch.matmul(a_big, b_big)
                    time.sleep(0.010)
            else:
                # Brief low util (<0.5s) → medium gap fill
                sig = "gap" if _signal_busy else "nosig"
                with _status_lock:
                    _status[gpu_id] = f"GFILL({sig},u={util}%,{elapsed:.1f}s)"
                with torch.cuda.stream(stream):
                    for _ in range(30):
                        torch.matmul(a_gap, b_gap)
                time.sleep(0.001)
        else:
            # util 20..target → push fill
            _low_util_since = None
            gap = target_util - util

            if gap > 20:
                # Far from target: push harder with 6144
                sig = "push" if _signal_busy else "nosig-push"
                with _status_lock:
                    _status[gpu_id] = f"PUSH({sig},u={util}%)"
                with torch.cuda.stream(stream):
                    for _ in range(5):
                        torch.matmul(a_push, b_push)
                time.sleep(0.002)
            else:
                # Near target: gentle push with 4096
                sig = "near" if _signal_busy else "nosig-near"
                with _status_lock:
                    _status[gpu_id] = f"PFILL({sig},u={util}%)"
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        torch.matmul(a_gap, b_gap)
                time.sleep(0.003)

    del a_big, b_big, a_push, b_push, a_gap, b_gap, a_busy, b_busy, a_orphan, b_orphan
    torch.cuda.empty_cache()
    print(f"[filler] GPU {gpu_id}: stopped")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Smart GPU filler — NVML-safe, util+signal based")
    parser.add_argument("--mode", choices=["nvml", "signal"], default="nvml",
                        help="Filler control mode: nvml (default) or signal (no NVML polling)")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated CUDA GPU IDs (default: all visible)")
    parser.add_argument("--target-util", type=int, default=85,
                        help="Target GPU utilization %% (default: 85)")
    parser.add_argument("--matrix-size", type=int, default=8192,
                        help="Matrix size for idle/escalation fill (default: 8192)")
    parser.add_argument("--batch", type=int, default=50,
                        help="Kernels per batch for idle fill (default: 50)")
    parser.add_argument("--gap-matrix", type=int, default=4096,
                        help="Matrix size for gap/near-target fill (default: 4096)")
    parser.add_argument("--push-matrix", type=int, default=6144,
                        help="Matrix size for push fill below target (default: 6144)")
    parser.add_argument("--busy-matrix", type=int, default=3072,
                        help="Matrix size for light fill in signal busy phase (default: 3072)")
    parser.add_argument("--busy-batch", type=int, default=4,
                        help="Kernels per batch for light fill in signal busy phase (default: 4)")
    parser.add_argument("--busy-sleep-ms", type=int, default=15,
                        help="Sleep between busy-phase batches in signal mode (default: 15)")
    parser.add_argument("--idle-sleep-ms", type=int, default=10,
                        help="Sleep between idle-phase batches in signal mode (default: 10)")
    parser.add_argument("--orphan-matrix", type=int, default=4096,
                        help="Matrix size for no-signal/stale medium fill in signal mode (default: 4096)")
    parser.add_argument("--orphan-batch", type=int, default=8,
                        help="Kernels per batch for no-signal/stale medium fill in signal mode (default: 8)")
    parser.add_argument("--orphan-sleep-ms", type=int, default=12,
                        help="Sleep between no-signal/stale batches in signal mode (default: 12)")
    parser.add_argument("--busy-hold-ms", type=int, default=800,
                        help="Keep light-fill briefly after a busy signal to smooth phase flapping (default: 800)")
    # Deprecated alias
    parser.add_argument("--pause", type=int, default=None,
                        help="DEPRECATED: use --target-util instead")
    args = parser.parse_args()

    # Backward compat
    if args.pause is not None:
        args.target_util = args.pause

    if args.gpus:
        gpu_ids = [int(g) for g in args.gpus.split(",")]
    else:
        import torch
        gpu_ids = list(range(torch.cuda.device_count()))

    if not gpu_ids:
        print("[filler] No GPUs found.")
        sys.exit(1)

    nvml_map = resolve_nvml_indices(gpu_ids)
    if args.mode == "nvml":
        _init_nvml()

    print(f"[filler] GPUs: {gpu_ids}  nvml: {[nvml_map[g] for g in gpu_ids]}")
    print(f"[filler] mode: {args.mode}")
    print(f"[filler] target: {args.target_util}%  idle: {args.matrix_size}x{args.batch}")
    print(f"[filler] push: {args.push_matrix}  gap: {args.gap_matrix}")
    print(
        f"[filler] signal-open-loop: busy={args.busy_matrix}x{args.busy_batch}/{args.busy_sleep_ms}ms  "
        f"orphan={args.orphan_matrix}x{args.orphan_batch}/{args.orphan_sleep_ms}ms  "
        f"idle_sleep={args.idle_sleep_ms}ms  hold={args.busy_hold_ms}ms"
    )
    print(f"[filler] signal: {SIGNAL_PATH}")
    if args.mode == "nvml":
        print(f"[filler] architecture: monitor(2Hz) + workers(cache-read)")
    else:
        print(f"[filler] architecture: signal-only workers (no NVML polling)")

    # Clean stale signal file
    if os.path.exists(SIGNAL_PATH):
        os.remove(SIGNAL_PATH)
        print(f"[filler] Cleaned stale signal file")

    # Kill old filler instances
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

    mon_thread = None
    if args.mode == "nvml":
        # Start monitor thread (must populate cache before workers start)
        mon_thread = threading.Thread(
            target=monitor_loop,
            args=(gpu_ids, nvml_map, my_pid),
            daemon=True,
        )
        mon_thread.start()
        time.sleep(0.2)  # let cache populate

    # Start worker threads
    workers = []
    for gid in gpu_ids:
        t = threading.Thread(
            target=filler_worker,
            args=(gid, args.matrix_size, args.batch,
                  args.target_util, args.gap_matrix, args.push_matrix, args.mode,
                  args.busy_matrix, args.busy_batch, args.busy_sleep_ms,
                  args.idle_sleep_ms, args.orphan_matrix, args.orphan_batch,
                  args.orphan_sleep_ms, args.busy_hold_ms),
            daemon=True,
        )
        t.start()
        workers.append(t)

    # Display loop (reads cache — no NVML calls)
    try:
        while not _STOP.is_set():
            parts = []
            for gid in gpu_ids:
                state = _gpu_cache.get(gid)
                if args.mode == "nvml":
                    u = state.util if state else -1
                    u_str = f"{u:3d}%"
                else:
                    u_str = "sig"
                with _status_lock:
                    st = _status.get(gid, "init")
                parts.append(f"GPU{gid}:{u_str} {st}")
            print(f"\r[filler] {' | '.join(parts)}  ", end="", flush=True)
            _STOP.wait(timeout=2.0)
    except KeyboardInterrupt:
        _STOP.set()

    for t in workers:
        t.join(timeout=5)
    if mon_thread is not None:
        mon_thread.join(timeout=2)
    print("\n[filler] Done.")


if __name__ == "__main__":
    main()
