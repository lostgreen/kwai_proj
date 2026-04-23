#!/usr/bin/env bash

# Shared helpers for running gpu_filler.py alongside training.

GPU_FILLER_SCRIPT="${GPU_FILLER_SCRIPT:-${REPO_ROOT}/local_scripts/gpu_filler.py}"
GPU_FILLER_SIGNAL_PATH="${VERL_GPU_SIGNAL_PATH:-/tmp/verl_gpu_phase}"
GPU_FILLER_LOG_PATH="${FILLER_LOG_PATH:-/tmp/filler.log}"

gpu_filler_per_gpu_enabled() {
  case "${FILLER_PER_GPU:-false}" in
    true|TRUE|1|yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

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
  local filler_mode="${FILLER_MODE:-nvml}"
  local signal_prefix="${FILLER_SIGNAL_PREFIX:-/tmp/verl_gpu_phase_gpu}"

  if ! gpu_filler_enabled; then
    return 0
  fi

  gpu_filler_stop_existing

  mkdir -p "$(dirname "${GPU_FILLER_LOG_PATH}")"

  filler_args=(
    --mode "${filler_mode}"
    --target-util "${FILLER_TARGET_UTIL:-80}"
    --batch "${FILLER_BATCH:-50}"
    --matrix-size "${FILLER_MATRIX:-8192}"
    --gap-matrix "${FILLER_GAP_MATRIX:-4096}"
    --push-matrix "${FILLER_PUSH_MATRIX:-6144}"
  )
  if [[ -n "${FILLER_BUSY_MATRIX:-}" ]]; then
    filler_args+=(--busy-matrix "${FILLER_BUSY_MATRIX}")
  fi
  if [[ -n "${FILLER_BUSY_BATCH:-}" ]]; then
    filler_args+=(--busy-batch "${FILLER_BUSY_BATCH}")
  fi
  if [[ -n "${FILLER_BUSY_SLEEP_MS:-}" ]]; then
    filler_args+=(--busy-sleep-ms "${FILLER_BUSY_SLEEP_MS}")
  fi
  if [[ -n "${FILLER_IDLE_SLEEP_MS:-}" ]]; then
    filler_args+=(--idle-sleep-ms "${FILLER_IDLE_SLEEP_MS}")
  fi
  if [[ -n "${FILLER_ORPHAN_MATRIX:-}" ]]; then
    filler_args+=(--orphan-matrix "${FILLER_ORPHAN_MATRIX}")
  fi
  if [[ -n "${FILLER_ORPHAN_BATCH:-}" ]]; then
    filler_args+=(--orphan-batch "${FILLER_ORPHAN_BATCH}")
  fi
  if [[ -n "${FILLER_ORPHAN_SLEEP_MS:-}" ]]; then
    filler_args+=(--orphan-sleep-ms "${FILLER_ORPHAN_SLEEP_MS}")
  fi
  if [[ -n "${FILLER_BUSY_HOLD_MS:-}" ]]; then
    filler_args+=(--busy-hold-ms "${FILLER_BUSY_HOLD_MS}")
  fi
  if [[ -n "${FILLER_GPUS:-}" ]] && ! gpu_filler_per_gpu_enabled; then
    filler_args+=(--gpus "${FILLER_GPUS}")
  fi

  echo "${prefix} Starting GPU filler for training (target=${FILLER_TARGET_UTIL:-80}%)"
  if [[ -n "${FILLER_GPUS:-}" ]]; then
    echo "${prefix} GPU filler local GPU ids: ${FILLER_GPUS}"
  fi
  if [[ "${start_delay}" != "0" ]]; then
    echo "${prefix} GPU filler warmup delay: ${start_delay}s"
  fi
  if gpu_filler_per_gpu_enabled; then
    local gpu_list=()
    IFS=',' read -r -a gpu_list <<< "${FILLER_GPUS:-}"
    for gpu in "${gpu_list[@]}"; do
      local gpu_trimmed="${gpu//[[:space:]]/}"
      [[ -z "${gpu_trimmed}" ]] && continue
      local signal_path="${signal_prefix}${gpu_trimmed}"
      local log_path="${GPU_FILLER_LOG_PATH}"
      if [[ "${GPU_FILLER_LOG_PATH}" == *.log ]]; then
        log_path="${GPU_FILLER_LOG_PATH%.log}_gpu${gpu_trimmed}.log"
      else
        log_path="${GPU_FILLER_LOG_PATH}_gpu${gpu_trimmed}"
      fi
      (
        sleep "${start_delay}"
        exec env CUDA_VISIBLE_DEVICES="${gpu_trimmed}" VERL_GPU_SIGNAL_PATH="${signal_path}" \
          python3 "${GPU_FILLER_SCRIPT}" "${filler_args[@]}" --gpus 0
      ) > "${log_path}" 2>&1 &
      echo "${prefix} GPU filler started for GPU ${gpu_trimmed} (PID $!), log: ${log_path}, signal: ${signal_path}"
    done
  else
    (
      sleep "${start_delay}"
      exec python3 "${GPU_FILLER_SCRIPT}" "${filler_args[@]}"
    ) > "${GPU_FILLER_LOG_PATH}" 2>&1 &
    echo "${prefix} GPU filler started (PID $!), log: ${GPU_FILLER_LOG_PATH}"
  fi
}

gpu_filler_clear_signal() {
  if gpu_filler_per_gpu_enabled; then
    local signal_prefix="${FILLER_SIGNAL_PREFIX:-/tmp/verl_gpu_phase_gpu}"
    local gpu_list=()
    IFS=',' read -r -a gpu_list <<< "${FILLER_GPUS:-}"
    for gpu in "${gpu_list[@]}"; do
      local gpu_trimmed="${gpu//[[:space:]]/}"
      [[ -z "${gpu_trimmed}" ]] && continue
      rm -f "${signal_prefix}${gpu_trimmed}"
    done
  else
    rm -f "${GPU_FILLER_SIGNAL_PATH}"
  fi
}

gpu_filler_cleanup() {
  gpu_filler_clear_signal
  if gpu_filler_stop_on_exit_enabled; then
    gpu_filler_stop_existing
  fi
}
