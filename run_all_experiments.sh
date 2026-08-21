#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")"

KEEP_ALIVE_SCRIPT="/vepfs-mlp2/c20250516/250504030/jing/keep_gpu_alive.py"
KEEP_ALIVE_PYTHON="/vepfs-mlp2/c20250516/250504030/env/flux/bin/python"
KEEP_ALIVE_PID_FILE="/tmp/gpu_keep_alive.pid"
KEEP_ALIVE_LOG_FILE="/tmp/gpu_keep_alive.log"

start_keep_alive() {
  local experiment_exit_code=$?
  set +e

  if [ -f "$KEEP_ALIVE_PID_FILE" ] && kill -0 "$(<"$KEEP_ALIVE_PID_FILE")" 2>/dev/null; then
    echo "GPU keep-alive is already running (PID: $(<"$KEEP_ALIVE_PID_FILE"))."
  elif [ -x "$KEEP_ALIVE_PYTHON" ] && [ -f "$KEEP_ALIVE_SCRIPT" ]; then
    nohup env -u CUDA_VISIBLE_DEVICES \
      "$KEEP_ALIVE_PYTHON" "$KEEP_ALIVE_SCRIPT" all \
      >"$KEEP_ALIVE_LOG_FILE" 2>&1 &
    echo $! >"$KEEP_ALIVE_PID_FILE"
    echo "GPU keep-alive started once after all batches (PID: $!, GPUs: all, log: $KEEP_ALIVE_LOG_FILE)."
  else
    echo "WARNING: GPU keep-alive could not start; Python or script is missing." >&2
  fi

  exit "$experiment_exit_code"
}
trap start_keep_alive EXIT

FAILED_BATCHES=()
LAYER_DIMS="8,48,48,48,64"

run_batch() {
  local name="$1"
  shift
  echo
  echo "================ Batch: ${name} ================"
  if "$@"; then
    echo "BATCH SUCCESS: ${name}"
  else
    local exit_code=$?
    FAILED_BATCHES+=("${name}:${exit_code}")
    echo "BATCH FAILED: ${name} (exit code: ${exit_code}); continuing." >&2
  fi
}

# run_batch "Be_basis_degree" \
#   bash experiment_sweep.sh 0,1,2,3 \
#     --atoms Be \
#     --bases chebyshev,legendre,spline,rbf,fastkan,sine,fourier \
#     --degrees 3,5,7,10 \
#     --layer-dims  8,48,48,48,32 \
#     --envelope-degrees Be:0

# # Example of another batch:
# run_batch "B_basis_degree" \
#   bash experiment_sweep.sh 0,1,2,3 \
#     --atoms B \
#     --bases chebyshev,legendre,spline,rbf,fastkan,sine,fourier \
#     --degrees 3,5,7,10 \
#     --layer-dims  8,48,48,48,40 \
#     --envelope-degrees B:0

# run_batch "N_basis_degree" \
#   bash experiment_sweep.sh 0,1,2,3 \
#     --atoms N \
#     --bases chebyshev,legendre,spline,rbf,fastkan,sine,fourier \
#     --degrees 3,5,7,10 \
#     --layer-dims  8,48,48,48,56 \
#     --envelope-degrees N:1
    
# run_batch "O_basis_degree" \
#   bash experiment_sweep.sh 0,1,2,3 \
#     --atoms O \
#     --bases chebyshev,legendre,spline,rbf,fastkan,sine,fourier \
#     --degrees 3,5,7,10 \
#     --layer-dims  8,48,48,48,64 \
#     --envelope-degrees O:1

# run_batch "F_basis_degree" \
#   bash experiment_sweep.sh 0,1,2,3 \
#     --atoms F \
#     --bases chebyshev,legendre,spline,rbf,fastkan,sine,fourier \
#     --degrees 3,5,7,10 \
#     --layer-dims  8,48,48,48,72 \
#     --envelope-degrees F:1

run_batch "Ne_basis_degree" \
  bash experiment_sweep.sh 0,1,2,3 \
    --atoms Ne \
    --bases chebyshev,legendre,spline,rbf,fastkan,sine,fourier \
    --degrees 3,5,7,10 \
    --layer-dims  8,48,48,48,80 \
    --envelope-degrees Ne:1

echo
echo "================ All batch summary ================"
if [ "${#FAILED_BATCHES[@]}" -eq 0 ]; then
  echo "All experiment batches completed successfully."
  exit 0
fi

echo "Failed batches:"
for batch in "${FAILED_BATCHES[@]}"; do
  echo "  - ${batch}"
done
exit 1
