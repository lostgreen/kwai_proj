#!/usr/bin/env bash
# Shared Event Logic harder-data defaults for task-composition runs.

EL_HARDER_DATA="${EL_HARDER_DATA:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/hier_seg_annotation_v1/event_logic_harder}"
EL_SORT_DATA="${EL_SORT_DATA:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/hier_seg_annotation_v1/event_logic_harder_sort}"

export EL_TRAIN="${EL_TRAIN:-${EL_HARDER_DATA}/train_10k.jsonl}"
export EL_TARGET="${EL_TARGET:-10000}"
export VAL_EL_N="${VAL_EL_N:-300}"

_DEFAULT_EL_VAL="${EL_HARDER_DATA}/val_logic.jsonl"
if [[ -z "${EL_VAL_SOURCE:-}" ]]; then
    if [[ ! -s "${_DEFAULT_EL_VAL}" ]]; then
        python3 - "${_DEFAULT_EL_VAL}" "${EL_HARDER_DATA}" "${EL_SORT_DATA}" <<'PY'
import json
import sys
from pathlib import Path

out_path = Path(sys.argv[1])
harder_dir = Path(sys.argv[2])
sort_dir = Path(sys.argv[3])

sources = []
combined_pn_fb = harder_dir / "val.jsonl"
if combined_pn_fb.is_file():
    sources.append(combined_pn_fb)
else:
    sources.extend([
        harder_dir / "val_predict_next.jsonl",
        harder_dir / "val_fill_blank.jsonl",
    ])

sources.append(sort_dir / "val_sort.jsonl")

rows = []
seen = set()
used = []
for path in sources:
    if not path.is_file():
        continue
    used.append(str(path))
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = (
                row.get("problem_type", ""),
                row.get("prompt", ""),
                row.get("answer", ""),
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)

if not rows:
    raise SystemExit(
        "No Event Logic val records found. Checked: "
        + ", ".join(str(path) for path in sources)
    )

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"[event-logic-common] wrote {out_path}: {len(rows)} records")
for path in used:
    print(f"[event-logic-common]   source: {path}")
PY
    fi
    export EL_VAL_SOURCE="${_DEFAULT_EL_VAL}"
fi

export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"

echo "[event-logic-common] EL_TRAIN=${EL_TRAIN}"
echo "[event-logic-common] EL_VAL_SOURCE=${EL_VAL_SOURCE}"
echo "[event-logic-common] EL_TARGET=${EL_TARGET} VAL_EL_N=${VAL_EL_N}"
