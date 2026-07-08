#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
GPU_IDS="${1:-0}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

cd "${SCRIPT_DIR}"
python main.py

if [ -f "${PROJECT_DIR}/keep_run.py" ]; then
  cd "${PROJECT_DIR}"
  python keep_run.py "${GPU_IDS%%,*}"
fi
