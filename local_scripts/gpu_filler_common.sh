#!/usr/bin/env bash

# Shared helpers for running gpu_filler.py alongside training.

GPU_FILLER_SCRIPT="${GPU_FILLER_SCRIPT:-${REPO_ROOT}/local_scripts/gpu_filler.py}"
GPU_FILLER_SIGNAL_PATH="${VERL_GPU_SIGNAL_PATH:-/tmp/verl_gpu_phase}"
GPU_FILLER_LOG_PATH="${FILLER_LOG_PATH:-/tmp/filler.log}"

gpu_filler_enabled() {
  case "${ENABLE_GPU_FILLER:-true}" in
    true|TRUE|1|yes|YES) ;;
    *) return 1 ;;
  esac
  [[ -f "${GPU_FILLER_SCRIPT}" ]]
}

gpu_filler_stop_existing() {
  if pgrep -f "gpu_filler.py" > /dev/null 2>&1; then
    pkill -f "gpu_filler.py" 2>/dev/null || true
    sleep 2
  fi
}

gpu_filler_stop_on_exit_enabled() {
  case "${STOP_GPU_FILLER_ON_EXIT:-false}" in
    true|TRUE|1|yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

gpu_filler_start() {
  local prefix="${1:-[gpu-filler]}"
  local filler_args=()
  local start_delay="${FILLER_START_DELAY:-0}"

  if ! gpu_filler_enabled; then
    return 0
  fi

  gpu_filler_stop_existing

  mkdir -p "$(dirname "${GPU_FILLER_LOG_PATH}")"

  filler_args=(
    --target-util "${FILLER_TARGET_UTIL:-80}"
    --batch "${FILLER_BATCH:-50}"
    --matrix-size "${FILLER_MATRIX:-8192}"
    --gap-matrix "${FILLER_GAP_MATRIX:-4096}"
    --push-matrix "${FILLER_PUSH_MATRIX:-6144}"
  )
  if [[ -n "${FILLER_GPUS:-}" ]]; then
    filler_args+=(--gpus "${FILLER_GPUS}")
  fi

  echo "${prefix} Starting GPU filler for training (target=${FILLER_TARGET_UTIL:-80}%)"
  if [[ -n "${FILLER_GPUS:-}" ]]; then
    echo "${prefix} GPU filler local GPU ids: ${FILLER_GPUS}"
  fi
  if [[ "${start_delay}" != "0" ]]; then
    echo "${prefix} GPU filler warmup delay: ${start_delay}s"
  fi
  (
    sleep "${start_delay}"
    exec python3 "${GPU_FILLER_SCRIPT}" "${filler_args[@]}"
  ) > "${GPU_FILLER_LOG_PATH}" 2>&1 &
  echo "${prefix} GPU filler started (PID $!), log: ${GPU_FILLER_LOG_PATH}"
}

gpu_filler_clear_signal() {
  rm -f "${GPU_FILLER_SIGNAL_PATH}"
}

gpu_filler_cleanup() {
  gpu_filler_clear_signal
  if gpu_filler_stop_on_exit_enabled; then
    gpu_filler_stop_existing
  fi
}
