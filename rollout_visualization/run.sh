set -euo pipefail
   
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

VIZHOST="${VIZHOST:-0.0.0.0}"
PORT="${PORT:-8890}"
ROLLOUT_DIR="${ROLLOUT_DIR:-//m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/multi_task/reward_ablation_R1_f1iou/rollouts}"
LOG_FILE="${LOG_FILE:-/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/multi_task/reward_ablation_R1_f1iou/experiment_log.jsonl}"
PRELOAD_TRAIN_STEP_INTERVAL="${PRELOAD_TRAIN_STEP_INTERVAL:-1}"

echo "[rollout-viz] host=${VIZHOST} port=${PORT}"
echo "[rollout-viz] rollout_dir=${ROLLOUT_DIR}"
echo "[rollout-viz] log_file=${LOG_FILE}"
echo "[rollout-viz] preload_train_step_interval=${PRELOAD_TRAIN_STEP_INTERVAL}"
echo "[rollout-viz] open: http://localhost:${PORT}/"

cd "${REPO_ROOT}"
python rollout_visualization/server.py \
  --host "${VIZHOST}" \
  --port "${PORT}" \
  --static-dir rollout_visualization \
  --rollout-dir "${ROLLOUT_DIR}" \
  --log-file "${LOG_FILE}" \
  --preload-train-step-interval "${PRELOAD_TRAIN_STEP_INTERVAL}"
