#!/usr/bin/env bash
# Run matched, independent COSMOS TDV-QSM overfit experiments for S=1,2,3.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/.."

cosmos_data="/cosmos_data"
output_root="runs/cosmos-step-sweep3"
iterations=2000
total_epochs=""
snr=70
seed=0
learning_rate=1e-3
max_gradient_norm=1.0
data_consistency_weight=0.0
features=1
macro_blocks=1
cg_iterations=5
max_time=0.25
phase_scale=1.0
log_every=1
epoch_chunk_size=2
remat_force=true
include_magnitude_in_weight=true
voxel_size=(1.0 1.0 1.0)
b0_direction=(0.0 0.0 1.0)

usage() {
    cat <<'EOF'
Usage: scripts/run_cosmos_step_sweep.sh [options]

Run fresh, matched COSMOS overfit experiments with S=1, S=2, and S=3.
Each run has an isolated output directory under --output-root. Existing
step directories are refused to prevent accidental checkpoint replacement.

Options:
  --cosmos-data DIR                 Directory containing chi_cosmos.mat, magn.mat, and msk.mat
  --output-root DIR                 Parent directory for steps-1, steps-2, and steps-3
  --iterations N                    Optimizer epochs per run (default: 100)
  --total-epochs N                  Progress denominator (default: --iterations)
  --snr VALUE                       COSMOS simulated SNR (default: 70)
  --seed N                          Shared simulation and initialization seed (default: 0)
  --learning-rate VALUE             Adam learning rate (default: 1e-3)
  --max-gradient-norm VALUE         Global gradient clipping threshold (default: 1.0)
  --data-consistency-weight VALUE   Optional terminal consistency-loss weight (default: 0)
  --features N                      TDV feature count (default: 1)
  --macro-blocks N                  TDV macro-block count (default: 1)
  --cg-iterations N                 Fixed CG iterations per TDV step (default: 12)
  --max-time VALUE                  Maximum learned stopping time (default: 0.25)
  --phase-scale VALUE               Radians per susceptibility unit (default: 1.0)
  --voxel-size VZ VY VX             Voxel size in z,y,x order
  --b0-direction BZ BY BX           B0 direction in z,y,x order
  --log-every N                     Epoch logging interval (default: 1)
  --epoch-chunk-size N              Jitted update chunk size (default: 2)
  --no-remat-force                  Disable TDV-force rematerialization
  --no-include-magnitude-in-weight  Use W=mask instead of W=mask*magnitude
  -h, --help                        Show this help
EOF
}

require_value() {
    if (($# < 2)); then
        echo "Missing value for $1" >&2
        exit 2
    fi
}

while (($#)); do
    case "$1" in
        --cosmos-data) require_value "$@"; cosmos_data="$2"; shift 2 ;;
        --output-root) require_value "$@"; output_root="$2"; shift 2 ;;
        --iterations) require_value "$@"; iterations="$2"; shift 2 ;;
        --total-epochs) require_value "$@"; total_epochs="$2"; shift 2 ;;
        --snr) require_value "$@"; snr="$2"; shift 2 ;;
        --seed) require_value "$@"; seed="$2"; shift 2 ;;
        --learning-rate) require_value "$@"; learning_rate="$2"; shift 2 ;;
        --max-gradient-norm) require_value "$@"; max_gradient_norm="$2"; shift 2 ;;
        --data-consistency-weight) require_value "$@"; data_consistency_weight="$2"; shift 2 ;;
        --features) require_value "$@"; features="$2"; shift 2 ;;
        --macro-blocks) require_value "$@"; macro_blocks="$2"; shift 2 ;;
        --cg-iterations) require_value "$@"; cg_iterations="$2"; shift 2 ;;
        --max-time) require_value "$@"; max_time="$2"; shift 2 ;;
        --phase-scale) require_value "$@"; phase_scale="$2"; shift 2 ;;
        --log-every) require_value "$@"; log_every="$2"; shift 2 ;;
        --epoch-chunk-size) require_value "$@"; epoch_chunk_size="$2"; shift 2 ;;
        --voxel-size)
            if (($# < 4)); then echo "--voxel-size requires VZ VY VX" >&2; exit 2; fi
            voxel_size=("$2" "$3" "$4"); shift 4 ;;
        --b0-direction)
            if (($# < 4)); then echo "--b0-direction requires BZ BY BX" >&2; exit 2; fi
            b0_direction=("$2" "$3" "$4"); shift 4 ;;
        --no-remat-force) remat_force=false; shift ;;
        --no-include-magnitude-in-weight) include_magnitude_in_weight=false; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$total_epochs" ]]; then
    total_epochs="$iterations"
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required; install it or run this script from an environment that provides uv." >&2
    exit 1
fi

# Match the COSMOS adapter's documented workspace fallback.
if [[ ! -f "$cosmos_data/chi_cosmos.mat" && -f "cosmos_data/chi_cosmos.mat" ]]; then
    cosmos_data="cosmos_data"
fi
for required_file in chi_cosmos.mat magn.mat msk.mat; do
    if [[ ! -f "$cosmos_data/$required_file" ]]; then
        echo "Missing required COSMOS input: $cosmos_data/$required_file" >&2
        exit 1
    fi
done

for steps in 1 2 3; do
    run_dir="$output_root/steps-$steps"
    if [[ -e "$run_dir" ]]; then
        echo "Refusing to overwrite existing run directory: $run_dir" >&2
        exit 1
    fi
done

mkdir -p "$output_root"
for steps in 1 2 3; do
    run_dir="$output_root/steps-$steps"
    command=(
        uv run python scripts/train_cosmos_overfit.py
        --cosmos-data "$cosmos_data"
        --output-dir "$run_dir"
        --snr "$snr"
        --seed "$seed"
        --voxel-size "${voxel_size[@]}"
        --b0-direction "${b0_direction[@]}"
        --phase-scale "$phase_scale"
        --iterations "$iterations"
        --total-epochs "$total_epochs"
        --learning-rate "$learning_rate"
        --max-gradient-norm "$max_gradient_norm"
        --data-consistency-weight "$data_consistency_weight"
        --features "$features"
        --macro-blocks "$macro_blocks"
        --steps "$steps"
        --cg-iterations "$cg_iterations"
        --max-time "$max_time"
        --log-every "$log_every"
        --epoch-chunk-size "$epoch_chunk_size"
    )
    if [[ "$remat_force" == true ]]; then
        command+=(--remat-force)
    fi
    if [[ "$include_magnitude_in_weight" == false ]]; then
        command+=(--no-include-magnitude-in-weight)
    fi

    echo "Starting independent COSMOS run with S=$steps: $run_dir"
    "${command[@]}"
done

echo "Completed matched COSMOS sweep: $output_root/steps-{1,2,3}"
