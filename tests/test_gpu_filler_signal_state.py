from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_gpu_filler(signal_path: Path):
    old_signal_path = os.environ.get("VERL_GPU_SIGNAL_PATH")
    os.environ["VERL_GPU_SIGNAL_PATH"] = str(signal_path)
    try:
        spec = importlib.util.spec_from_file_location(
            "gpu_filler_under_test",
            REPO_ROOT / "local_scripts" / "gpu_filler.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules.pop("gpu_filler_under_test", None)
        spec.loader.exec_module(module)
        return module
    finally:
        if old_signal_path is None:
            os.environ.pop("VERL_GPU_SIGNAL_PATH", None)
        else:
            os.environ["VERL_GPU_SIGNAL_PATH"] = old_signal_path


def test_missing_training_signal_is_treated_as_idle(tmp_path):
    filler = _load_gpu_filler(tmp_path / "missing_signal")

    assert filler.get_signal_state() == "idle"


def test_val_decode_is_treated_as_busy(tmp_path):
    signal_path = tmp_path / "phase"
    signal_path.write_text("val_decode")
    filler = _load_gpu_filler(signal_path)

    assert filler.get_signal_state() == "busy"
