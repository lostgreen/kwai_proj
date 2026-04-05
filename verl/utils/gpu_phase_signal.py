"""GPU phase signal for coordinating with external GPU filler scripts.

The training process writes its current phase to a signal file so that a GPU
filler script can pause during compute-intensive phases (gen, update) and only
fill GPU utilization during idle gaps (data loading, reward computation, etc.).

Signal file: /tmp/verl_gpu_phase  (configurable via VERL_GPU_SIGNAL_PATH)

Phases:
    "gen"      — vLLM generation (GPU busy, filler must pause)
    "update"   — FSDP forward/backward/optimizer (GPU busy, filler must pause)
    "idle"     — between phases, data loading, etc. (filler can run)
    file removed — training finished (filler can run freely)
"""

import atexit
import os

_SIGNAL_PATH = os.environ.get("VERL_GPU_SIGNAL_PATH", "/tmp/verl_gpu_phase")


def set_gpu_phase(phase: str) -> None:
    """Write the current training phase to the signal file."""
    try:
        with open(_SIGNAL_PATH, "w") as f:
            f.write(phase)
    except OSError:
        pass  # non-critical, don't break training


def clear_gpu_phase() -> None:
    """Remove the signal file (training finished)."""
    try:
        os.remove(_SIGNAL_PATH)
    except FileNotFoundError:
        pass


def register_cleanup() -> None:
    """Register atexit handler to clean up signal file on exit."""
    atexit.register(clear_gpu_phase)


def get_gpu_phase() -> str | None:
    """Read the current phase from the signal file. Returns None if no file."""
    try:
        with open(_SIGNAL_PATH, "r") as f:
            return f.read().strip()
    except (FileNotFoundError, OSError):
        return None
