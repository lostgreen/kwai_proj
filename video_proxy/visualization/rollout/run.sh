set -euo pipefail
   
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

VIZHOST="${VIZHOST:-0.0.0.0}"
PORT="${PORT:-8890}"
ROLLOUT_DIR="${ROLLOUT_DIR:-}"
LOG_FILE="${LOG_FILE:-}"
PRELOAD_TRAIN_STEP_INTERVAL="${PRELOAD_TRAIN_STEP_INTERVAL:-1}"

echo "[rollout-viz] host=${VIZHOST} port=${PORT}"
echo "[rollout-viz] rollout_dir=${ROLLOUT_DIR}"
echo "[rollout-viz] log_file=${LOG_FILE}"
echo "[rollout-viz] preload_train_step_interval=${PRELOAD_TRAIN_STEP_INTERVAL}"
echo "[rollout-viz] open: http://localhost:${PORT}/"

cd "${REPO_ROOT}"
CMD=(python video_proxy/visualization/rollout/server.py
  --host "${VIZHOST}"
  --port "${PORT}"
  --static-dir video_proxy/visualization/rollout
  --preload-train-step-interval "${PRELOAD_TRAIN_STEP_INTERVAL}")

if [[ -n "${ROLLOUT_DIR}" ]]; then
  CMD+=(--rollout-dir "${ROLLOUT_DIR}")
fi
if [[ -n "${LOG_FILE}" ]]; then
  CMD+=(--log-file "${LOG_FILE}")
fi

"${CMD[@]}"
