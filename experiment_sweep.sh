#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")"

usage() {
  cat <<'EOF'
Usage:
  experiment_sweep.sh <gpu_ids> [options]

Options:
  --atoms LIST    Comma-separated atoms (default: Be,B)
  --bases LIST    Comma-separated basis types (default: chebyshev,legendre,rbf)
  --degrees LIST  Comma-separated positive integers (default: 3,5,7,10,14)
  --layer-dims LIST
                  Comma-separated MKAN dimensions, e.g. 8,48,48,48,64
  --envelope-degrees MAP
                  Per-atom envelope degrees, e.g. Be:1,B:2 (default: *:1)
  --preset-only   Run each basis once with basis_required_parameters from config
  --dry-run       Print experiments and output paths without training
  -h, --help      Show this help

Examples:
  bash experiment_sweep.sh 0,1,2,3
  bash experiment_sweep.sh 0,1,2,3 --atoms C --bases spline,chebyshev,rbf --preset-only
  bash experiment_sweep.sh 0,1,2,3 --atoms Be,B --bases chebyshev,legendre,rbf --degrees 3,5,7,10,14 --layer-dims 8,48,48,48,64 --envelope-degrees Be:1,B:2

For chebyshev/legendre/rbf/sine/fourier/fastkan, the swept value is D.
For base/spline/relukan, the swept value is G and k remains at its preset value.
WavKAN has no D/G/k capacity parameter and is run once per atom.
EOF
}

if [ "$#" -eq 0 ]; then
  usage
  exit 2
fi

GPU_IDS="$1"
shift
ATOMS_CSV="Be,B"
BASES_CSV="chebyshev,legendre,rbf"
DEGREES_CSV="3,5,7,10,14"
ENVELOPE_DEGREES_CSV="*:1"
LAYER_DIMS_CSV=""
DRY_RUN=0
PRESET_ONLY=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --atoms)
      [ "$#" -ge 2 ] || { echo "ERROR: --atoms requires a value." >&2; exit 2; }
      ATOMS_CSV="$2"
      shift 2
      ;;
    --bases)
      [ "$#" -ge 2 ] || { echo "ERROR: --bases requires a value." >&2; exit 2; }
      BASES_CSV="$2"
      shift 2
      ;;
    --degrees)
      [ "$#" -ge 2 ] || { echo "ERROR: --degrees requires a value." >&2; exit 2; }
      DEGREES_CSV="$2"
      shift 2
      ;;
    --layer-dims)
      [ "$#" -ge 2 ] || { echo "ERROR: --layer-dims requires a value." >&2; exit 2; }
      LAYER_DIMS_CSV="$2"
      shift 2
      ;;
    --envelope-degrees)
      [ "$#" -ge 2 ] || { echo "ERROR: --envelope-degrees requires a value." >&2; exit 2; }
      ENVELOPE_DEGREES_CSV="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --preset-only)
      PRESET_ONLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown option '$1'." >&2
      usage
      exit 2
      ;;
  esac
done

[ -n "$GPU_IDS" ] || { echo "ERROR: gpu_ids cannot be empty." >&2; exit 2; }

IFS=',' read -r -a ATOMS <<<"$ATOMS_CSV"
IFS=',' read -r -a BASES <<<"$BASES_CSV"
IFS=',' read -r -a DEGREES <<<"$DEGREES_CSV"
IFS=',' read -r -a ENVELOPE_DEGREE_ENTRIES <<<"$ENVELOPE_DEGREES_CSV"

if [ -n "$LAYER_DIMS_CSV" ]; then
  IFS=',' read -r -a LAYER_DIMS <<<"$LAYER_DIMS_CSV"
  if [ "${#LAYER_DIMS[@]}" -lt 2 ]; then
    echo "ERROR: --layer-dims requires at least input and output dimensions." >&2
    exit 2
  fi
  for DIM in "${LAYER_DIMS[@]}"; do
    if ! [[ "$DIM" =~ ^[1-9][0-9]*$ ]]; then
      echo "ERROR: layer dimensions must be positive integers, got '$DIM'." >&2
      exit 2
    fi
  done
fi

for DEGREE in "${DEGREES[@]}"; do
  if ! [[ "$DEGREE" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: degree values must be positive integers, got '$DEGREE'." >&2
    exit 2
  fi
done

for ENTRY in "${ENVELOPE_DEGREE_ENTRIES[@]}"; do
  if ! [[ "$ENTRY" =~ ^([A-Z][a-z]?|\*):[0-9]+$ ]]; then
    echo "ERROR: envelope-degree entries must look like Be:1,B:2 or *:1; got '$ENTRY'." >&2
    exit 2
  fi
done

envelope_degree_for_atom() {
  local atom="$1"
  local entry key value
  local fallback=""
  for entry in "${ENVELOPE_DEGREE_ENTRIES[@]}"; do
    key="${entry%%:*}"
    value="${entry#*:}"
    if [ "$key" = "$atom" ]; then
      echo "$value"
      return 0
    fi
    if [ "$key" = "*" ]; then
      fallback="$value"
    fi
  done
  if [ -n "$fallback" ]; then
    echo "$fallback"
    return 0
  fi
  echo "ERROR: no envelope degree configured for atom '$atom'." >&2
  return 2
}

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
PYTHON="/vepfs-mlp2/c20250516/250504030/env/qmc/bin/python"
FAILED_EXPERIMENTS=()
FAILED_CODES=()
SUCCESSFUL_EXPERIMENTS=()

run_experiment() {
  local atom="$1"
  local basis="$2"
  local capacity_name="$3"
  local capacity_value="$4"
  local envelope_degree="$5"
  local experiment="${atom}/${basis}/envelope_degree_${envelope_degree}/${capacity_name}_${capacity_value}"

  echo
  echo "========== Experiment: ${experiment} =========="

  if [ "$DRY_RUN" -eq 1 ]; then
    ATOM="$atom" BASIS="$basis" CAPACITY_NAME="$capacity_name" CAPACITY_VALUE="$capacity_value" ENVELOPE_DEGREE="$envelope_degree" LAYER_DIMS_CSV="$LAYER_DIMS_CSV" \
      "$PYTHON" - <<'PY'
import os
from pathlib import Path

from train import config

atom = os.environ["ATOM"]
basis = os.environ["BASIS"]
capacity_name = os.environ["CAPACITY_NAME"]
capacity_value = os.environ["CAPACITY_VALUE"]
envelope_degree = os.environ["ENVELOPE_DEGREE"]
layer_dims_csv = os.environ["LAYER_DIMS_CSV"]
cfg = config.default()
layer_dims = [int(value) for value in layer_dims_csv.split(",")] if layer_dims_csv else list(cfg.layer_dims)
original_output = Path(cfg.output.root_dir)
output = (
    original_output.parent.parent
    / atom
    / basis
    / f"{capacity_name}_{capacity_value}"
    / original_output.name
)
print(f"DRY RUN output: {output}")
print(f"DRY RUN effective layer_dims: {layer_dims}")
PY
    SUCCESSFUL_EXPERIMENTS+=("$experiment")
    return 0
  fi

  if ATOM="$atom" BASIS="$basis" CAPACITY_NAME="$capacity_name" CAPACITY_VALUE="$capacity_value" ENVELOPE_DEGREE="$envelope_degree" LAYER_DIMS_CSV="$LAYER_DIMS_CSV" \
    "$PYTHON" - <<'PY'
import os
from pathlib import Path

import jax

from tools.utils import system
from train import config, train


jax.config.update("jax_traceback_filtering", "off")

ELECTRONS = {
    "H": (1, 0),
    "He": (1, 1),
    "Li": (2, 1),
    "Be": (2, 2),
    "B": (3, 2),
    "C": (4, 2),
    "N": (5, 2),
    "O": (5, 3),
    "F": (5, 4),
    "Ne": (5, 5),
}

atom = os.environ["ATOM"].strip()
basis = os.environ["BASIS"].strip().lower()
capacity_name = os.environ["CAPACITY_NAME"]
capacity_value = os.environ["CAPACITY_VALUE"]
envelope_degree = int(os.environ["ENVELOPE_DEGREE"])
layer_dims_csv = os.environ["LAYER_DIMS_CSV"]

if atom not in ELECTRONS:
    raise ValueError(
        f"Unsupported atom {atom!r}; supported atoms are {', '.join(ELECTRONS)}"
    )

cfg = config.default()
if layer_dims_csv:
    cfg.layer_dims = [int(value) for value in layer_dims_csv.split(",")]
if basis not in cfg.mkan.basis_required_parameters:
    raise ValueError(f"Unsupported basis {basis!r}")

required_parameters = dict(cfg.mkan.basis_required_parameters[basis])
if capacity_name != "fixed":
    required_parameters[capacity_name] = int(capacity_value)

cfg.system.molecule = [system.Atom(atom, (0, 0, 0))]
cfg.system.electrons = ELECTRONS[atom]
cfg.envelope_degree = envelope_degree
cfg.mkan.layer_type = basis
cfg.mkan.required_parameters = required_parameters
cfg.output.resume = False

original_output = Path(cfg.output.root_dir)
cfg.output.root_dir = str(
    original_output.parent.parent
    / atom
    / basis
    / f"{capacity_name}_{capacity_value}"
    / original_output.name
)

print(f"atom: {atom}")
print(f"electrons: {tuple(cfg.system.electrons)}")
print(f"envelope_degree: {cfg.envelope_degree}")
print(f"layer_dims: {list(cfg.layer_dims)}")
print(f"basis: {basis}")
print(f"required_parameters: {dict(cfg.mkan.required_parameters)}")
print(f"output: {cfg.output.root_dir}")
train.train(cfg)
PY
  then
    SUCCESSFUL_EXPERIMENTS+=("$experiment")
    echo "SUCCESS: ${experiment} completed."
  else
    local exit_code=$?
    FAILED_EXPERIMENTS+=("$experiment")
    FAILED_CODES+=("$exit_code")
    echo "ERROR: ${experiment} failed (exit code: ${exit_code}); continuing." >&2
  fi
}

for ATOM in "${ATOMS[@]}"; do
  ATOM="${ATOM//[[:space:]]/}"
  [ -n "$ATOM" ] || continue
  ENVELOPE_DEGREE="$(envelope_degree_for_atom "$ATOM")"
  for BASIS in "${BASES[@]}"; do
    BASIS="${BASIS//[[:space:]]/}"
    BASIS="${BASIS,,}"
    [ -n "$BASIS" ] || continue

    case "$BASIS" in
      chebyshev|legendre|rbf|sine|fourier|fastkan)
        if [ "$PRESET_ONLY" -eq 1 ]; then
          run_experiment "$ATOM" "$BASIS" "fixed" "preset" "$ENVELOPE_DEGREE"
        else
          for DEGREE in "${DEGREES[@]}"; do
            run_experiment "$ATOM" "$BASIS" "D" "$DEGREE" "$ENVELOPE_DEGREE"
          done
        fi
        ;;
      base|spline|relukan)
        if [ "$PRESET_ONLY" -eq 1 ]; then
          run_experiment "$ATOM" "$BASIS" "fixed" "preset" "$ENVELOPE_DEGREE"
        else
          for DEGREE in "${DEGREES[@]}"; do
            run_experiment "$ATOM" "$BASIS" "G" "$DEGREE" "$ENVELOPE_DEGREE"
          done
        fi
        ;;
      wavkan)
        run_experiment "$ATOM" "$BASIS" "fixed" "default" "$ENVELOPE_DEGREE"
        ;;
      *)
        FAILED_EXPERIMENTS+=("${ATOM}/${BASIS}/unsupported")
        FAILED_CODES+=("2")
        echo "ERROR: unsupported basis '${BASIS}'; continuing." >&2
        ;;
    esac
  done
done

echo
echo "========== Experiment sweep summary =========="
echo "Successful experiments: ${#SUCCESSFUL_EXPERIMENTS[@]}"
for experiment in "${SUCCESSFUL_EXPERIMENTS[@]}"; do
  echo "  - ${experiment}"
done

if [ "${#FAILED_EXPERIMENTS[@]}" -eq 0 ]; then
  echo "All experiments completed successfully."
  exit 0
fi

echo "Failed experiments: ${#FAILED_EXPERIMENTS[@]}"
for i in "${!FAILED_EXPERIMENTS[@]}"; do
  echo "  - ${FAILED_EXPERIMENTS[$i]} (exit code: ${FAILED_CODES[$i]})"
done
exit 1
