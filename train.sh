#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

GPU_IDS="${1:-${CUDA_VISIBLE_DEVICES:-}}"
if [ -z "$GPU_IDS" ]; then
  echo "Usage: $0 <gpu_ids>"
  echo "Example: $0 0 or $0 0,1,2,3"
  exit 2
fi

run_gpuon() {
  local exit_code=$?
  set +e
  echo "main.py exited with code ${exit_code}; starting GPU keep-alive..."

  local keep_alive_script="/vepfs-mlp2/c20250516/252703012/keep_run.py"
  local python_bin="${PYTHON:-python}"
  local keep_alive_gpu_ids="${KEEP_ALIVE_GPU_IDS:-$GPU_IDS}"

  if [ ! -f "$keep_alive_script" ]; then
    echo "GPU keep-alive cannot start; missing script: $keep_alive_script" >&2
  else
    IFS=',' read -ra gpu_list <<< "$keep_alive_gpu_ids"
    for gpu_id in "${gpu_list[@]}"; do
      local pid_file="/tmp/gpu_keep_alive_${gpu_id}.pid"
      local log_file="/tmp/gpu_keep_alive_${gpu_id}.log"

      if [ -f "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
        echo "GPU keep-alive is already running for GPU ${gpu_id} (PID: $(cat "$pid_file"))."
      else
        nohup env CUDA_VISIBLE_DEVICES="$gpu_id" "$python_bin" "$keep_alive_script" 0 > "$log_file" 2>&1 &
        echo $! > "$pid_file"
        echo "GPU keep-alive started (PID: $!, GPU: $gpu_id, log: $log_file)."
      fi
    done
  fi

  exit "$exit_code"
}

trap run_gpuon EXIT

export CUDA_VISIBLE_DEVICES="$GPU_IDS"

"${PYTHON:-python}" main.py
