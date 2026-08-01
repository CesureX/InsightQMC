#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")"

GPU_IDS="${1:-${CUDA_VISIBLE_DEVICES:-}}"
if [ -z "$GPU_IDS" ]; then
  echo "Usage: $0 <gpu_ids> [basis ...]"
  echo "Example: $0 0"
  echo "Example: $0 0,1 base spline rbf"
  exit 2
fi
if [ "$#" -gt 0 ]; then
  shift
fi

restore_all_gpus() {
  local test_exit_code=$?
  if command -v gpu-on >/dev/null 2>&1; then
    if ! gpu-on all; then
      echo "WARNING: 'gpu-on all' failed." >&2
    fi
  fi
  exit "$test_exit_code"
}
trap restore_all_gpus EXIT

if [ "$#" -gt 0 ]; then
  BASES=("$@")
else
  BASES=(
    base
    spline
    chebyshev
    legendre
    rbf
    sine
    fourier
    fastkan
    relukan
    wavkan
  )
fi

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
PYTHON="${PYTHON:-/vepfs-mlp2/c20250516/250504030/env/qmc/bin/python}"
FAILED_BASES=()
FAILED_CODES=()

for BASIS in "${BASES[@]}"; do
  echo
  echo "========== Testing basis: ${BASIS} =========="

  if BASIS="$BASIS" "$PYTHON" - <<'PY'
import os
from pathlib import Path

import jax

from train import config, train


jax.config.update("jax_traceback_filtering", "off")

basis = os.environ["BASIS"]
cfg = config.default()

if basis not in cfg.mkan.basis_required_parameters:
    raise ValueError(f"No basis parameters configured for type: {basis}")

cfg.mkan.layer_type = basis
cfg.mkan.required_parameters = dict(cfg.mkan.basis_required_parameters[basis])
cfg.output.resume = False

original_output = Path(cfg.output.root_dir)
cfg.output.root_dir = str(
    original_output.parent / "basis_test" / basis / original_output.name
)

print(f"basis: {basis}")
print(f"required_parameters: {dict(cfg.mkan.required_parameters)}")
print(f"output: {cfg.output.root_dir}")
train.train(cfg)
PY
  then
    echo "SUCCESS: basis type '${BASIS}' completed."
  else
    exit_code=$?
    FAILED_BASES+=("$BASIS")
    FAILED_CODES+=("$exit_code")
    echo "ERROR: basis type '${BASIS}' failed (exit code: ${exit_code}); continuing." >&2
  fi
done

echo
echo "========== Basis test summary =========="
if [ "${#FAILED_BASES[@]}" -eq 0 ]; then
  echo "All basis types completed successfully."
  exit 0
fi

echo "Failed basis types:"
for i in "${!FAILED_BASES[@]}"; do
  echo "  - ${FAILED_BASES[$i]} (exit code: ${FAILED_CODES[$i]})"
done
exit 1

# bash basis_test.sh 0 spline
# bash basis_test.sh 0,1,2,3 spline
# bash basis_test.sh 0,1,2,3
