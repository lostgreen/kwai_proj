"""Lightweight timing logger.

Usage:
    # Driver (ray_trainer.py): init once, then log per step when enabled
    from verl.utils.timing_logger import init_timing_log, log_timing

    init_timing_log("/path/to/checkpoint_dir")
    log_timing(step=1, timing_raw={"gen": 328.0, "gen/prepare": 12.5, ...})

    # Workers (fsdp_workers.py, fsdp_vllm.py, vllm_rollout_spmd.py):
    from verl.utils.timing_logger import tlog

    tlog(f"[gen][rank=0 dp=0 tp=0] vllm generate: 45.23s")
"""

import logging
import os
import time

_timing_logger: logging.Logger | None = None
_log_path: str | None = None


def _file_logging_enabled() -> bool:
    return os.environ.get("VERL_TIMING_LOG_TO_FILE", "").lower() in {"1", "true", "yes"}


def init_timing_log(checkpoint_dir: str, filename: str = "timing.log") -> None:
    """Initialise the optional file-backed timing logger."""
    global _timing_logger, _log_path
    if not _file_logging_enabled():
        return

    os.makedirs(checkpoint_dir, exist_ok=True)
    _log_path = os.path.join(checkpoint_dir, filename)

    _timing_logger = logging.getLogger("verl.timing")
    _timing_logger.setLevel(logging.DEBUG)
    _timing_logger.propagate = False
    # Remove previous handlers (e.g. if init is called twice)
    _timing_logger.handlers.clear()

    fh = logging.FileHandler(_log_path, mode="a")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
    _timing_logger.addHandler(fh)

    _timing_logger.info("=== timing log initialised ===")

    # Also set env var so Ray workers can discover the path
    os.environ["VERL_TIMING_LOG_DIR"] = checkpoint_dir


def _ensure_logger() -> logging.Logger | None:
    """Lazily create a worker-side logger from the env-var path."""
    global _timing_logger, _log_path
    if not _file_logging_enabled():
        return None

    if _timing_logger is not None:
        return _timing_logger

    log_dir = os.environ.get("VERL_TIMING_LOG_DIR")
    if log_dir is None:
        return None

    os.makedirs(log_dir, exist_ok=True)
    _log_path = os.path.join(log_dir, "timing.log")

    _timing_logger = logging.getLogger("verl.timing")
    _timing_logger.setLevel(logging.DEBUG)
    _timing_logger.propagate = False
    if not _timing_logger.handlers:
        fh = logging.FileHandler(_log_path, mode="a")
        fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
        _timing_logger.addHandler(fh)

    return _timing_logger


def tlog(msg: str) -> None:
    """Log a single timing message (also prints to stdout)."""
    logger = _ensure_logger()
    if logger is not None:
        logger.info(msg)
    print(msg)


def log_timing(step: int, timing_raw: dict[str, float]) -> None:
    """Log all timing_raw entries for a given step."""
    logger = _ensure_logger()
    if logger is None:
        return
    logger.info(f"--- step {step} ---")
    for k, v in sorted(timing_raw.items()):
        logger.info(f"  {k}: {v:.3f}s")
