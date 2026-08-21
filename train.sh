#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

GPU_IDS="${1:-${CUDA_VISIBLE_DEVICES:-}}"
if [ -z "$GPU_IDS" ]; then
  echo "Usage: $0 <gpu_ids>"
  echo "Example: $0 0 or $0 0,1,2,3"
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$GPU_IDS"

/vepfs-mlp2/c20250516/250504030/env/qmc/bin/python main.py
