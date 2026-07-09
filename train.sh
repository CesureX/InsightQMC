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

  local keep_alive_script="/vepfs-mlp2/c20250516/250504030/jing/keep_gpu_alive.py"
  local keep_alive_python="/vepfs-mlp2/c20250516/250504030/env/flux/bin/python"
  local pid_file="/tmp/gpu_keep_alive.pid"
  local log_file="/tmp/gpu_keep_alive.log"
  local keep_alive_gpu_ids="all"

  if [ -f "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "GPU keep-alive is already running (PID: $(cat "$pid_file"))."
  elif [ -x "$keep_alive_python" ] && [ -f "$keep_alive_script" ]; then
    nohup env -u CUDA_VISIBLE_DEVICES "$keep_alive_python" "$keep_alive_script" "$keep_alive_gpu_ids" > "$log_file" 2>&1 &
    echo $! > "$pid_file"
    echo "GPU keep-alive started (PID: $!, GPU: $keep_alive_gpu_ids, log: $log_file)."
  else
    echo "GPU keep-alive cannot start; missing python or script." >&2
  fi

  exit "$exit_code"
}

trap run_gpuon EXIT

export CUDA_VISIBLE_DEVICES="$GPU_IDS"

/vepfs-mlp2/c20250516/250504030/env/qmc/bin/python main.py
