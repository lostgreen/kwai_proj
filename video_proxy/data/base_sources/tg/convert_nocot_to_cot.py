#!/usr/bin/env python3
"""Compatibility wrapper for the shared JSONL CoT converter.

Prefer ``video_proxy/data/scripts/convert_jsonl_to_cot.py`` for new usage.
This entrypoint remains for older TG docs and scripts.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from video_proxy.data.scripts.convert_jsonl_to_cot import main  # noqa: E402


if __name__ == "__main__":
    main()
