#!/bin/bash
# =============================================================================
# TimesFM-2.0 Evaluation Script
# =============================================================================
# This script sets up the environment and runs TimesFM-2.0 experiments
# for selected datasets and terms.
#
# Usage:
#   Arguments:
#     $1: datasets (space-separated, use quotes for multiple datasets)
#     $2: terms (comma-separated: short,medium,long)
#
#   Single argument behavior:
#     - If contains "/" -> treated as datasets (e.g., "SG_Weather/D")
#     - If contains "," -> treated as terms (e.g., "short,medium")
#     - If is "short"/"medium"/"long" -> treated as terms
#     - Otherwise -> treated as datasets
#
#   Environment variables (alternative to command line args):
#     DATASETS="SG_Weather/D SG_PM25/H" ./scripts/run_timesfm2.sh
#     TERMS="short,medium" ./scripts/run_timesfm2.sh
#
#   Environment setup:
#     ENV_NAME (default: timesfm2)
#     PYTHON_VERSION (default: 3.11)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DEFAULT_DATASETS=(
    "SG_Weather/D"
    "SG_PM25/H"
    "Water_Quality_Darwin/15T"
    "Coastal_T_S/5T"
    "Coastal_T_S/15T"
    "Coastal_T_S/20T"
)

if [ $# -ge 2 ]; then
    if [ -n "$1" ]; then
        IFS=' ' read -ra DATASET_LIST <<< "$1"
    else
        DATASET_LIST=("${DEFAULT_DATASETS[@]}")
    fi
    IFS=',' read -ra TERMS <<< "$2"
elif [ $# -eq 1 ]; then
    if [[ "$1" == *"/"* ]]; then
        IFS=' ' read -ra DATASET_LIST <<< "$1"
        TERMS=("short" "medium" "long")
    elif [[ "$1" == *","* ]]; then
        DATASET_LIST=("${DEFAULT_DATASETS[@]}")
        IFS=',' read -ra TERMS <<< "$1"
    elif [[ "$1" == *" "* ]]; then
        IFS=' ' read -ra DATASET_LIST <<< "$1"
        TERMS=("short" "medium" "long")
    elif [[ "$1" == "short" ]] || [[ "$1" == "medium" ]] || [[ "$1" == "long" ]]; then
        DATASET_LIST=("${DEFAULT_DATASETS[@]}")
        TERMS=("$1")
    else
        IFS=' ' read -ra DATASET_LIST <<< "$1"
        TERMS=("short" "medium" "long")
    fi
else
    if [ -n "$DATASETS" ]; then
        IFS=' ' read -ra DATASET_LIST <<< "$DATASETS"
    else
        DATASET_LIST=("${DEFAULT_DATASETS[@]}")
    fi

    if [ -n "$TERMS" ]; then
        IFS=',' read -ra TERMS <<< "$TERMS"
    else
        TERMS=("short" "medium" "long")
    fi
fi

ENV_NAME="${ENV_NAME:-timesfm2}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
MODEL_SIZE="${MODEL_SIZE:-base}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/results/timesfm2.0_${MODEL_SIZE}}"
BATCH_SIZE="${BATCH_SIZE:-512}"
PER_CORE_BATCH_SIZE="${PER_CORE_BATCH_SIZE:-32}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-2048}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
CONFIG_PATH="${CONFIG_PATH:-}"
USE_VAL="${USE_VAL:-false}"
QUANTILES="${QUANTILES:-}"

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" >&2
}

log_section() {
    echo ""
    echo "============================================================================="
    echo "  $1"
    echo "============================================================================="
}

ensure_conda_env() {
    if ! command -v conda >/dev/null 2>&1; then
        log_error "Conda not found. Please install conda first."
        exit 1
    fi

    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"

    if conda env list | awk '{print $1}' | grep -x "$ENV_NAME" >/dev/null 2>&1; then
        log_info "Activating conda env: $ENV_NAME"
        conda activate "$ENV_NAME"
    else
        log_info "Creating conda env: $ENV_NAME"
        conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
        conda activate "$ENV_NAME"
        pip install torch datasets gluonts python-dotenv huggingface_hub pandas numpy utilsforecast
    fi
}

# -----------------------------------------------------------------------------
# Main Pipeline
# -----------------------------------------------------------------------------
main() {
    cd "$ROOT_DIR"

    log_section "TimesFM-2.0 Evaluation Pipeline"
    echo "Datasets: ${DATASET_LIST[*]}"
    echo "Terms: ${TERMS[*]}"
    echo "Model size: $MODEL_SIZE"
    echo "Output directory: $OUTPUT_DIR"
    echo "Batch size: $BATCH_SIZE"
    echo "Per-core batch size: $PER_CORE_BATCH_SIZE"
    echo "Context length: $CONTEXT_LENGTH"
    echo "CUDA device: $CUDA_DEVICE"
    echo "Conda env: $ENV_NAME (python $PYTHON_VERSION)"
    echo "Use val: $USE_VAL"
    if [ -n "$QUANTILES" ]; then
        echo "Quantiles: $QUANTILES"
    fi

    ensure_conda_env
    pip install utilsforecast

    mkdir -p "$OUTPUT_DIR"

    extra_args=()
    if [ -n "$CONFIG_PATH" ]; then
        extra_args+=(--config "$CONFIG_PATH")
    fi
    if [ "$USE_VAL" = "true" ]; then
        extra_args+=(--val)
    fi
    if [ -n "$QUANTILES" ]; then
        IFS=',' read -ra QUANTILE_LIST <<< "$QUANTILES"
        extra_args+=(--quantiles "${QUANTILE_LIST[@]}")
    fi

    total_datasets=${#DATASET_LIST[@]}
    total_terms=${#TERMS[@]}
    total_experiments=$((total_datasets * total_terms))
    current=0

    for dataset in "${DATASET_LIST[@]}"; do
        for term in "${TERMS[@]}"; do
            current=$((current + 1))
            log_section "[$current/$total_experiments] $dataset / $term"

            python experiments/timesfm2.0.py \
                --dataset "$dataset" \
                --terms "$term" \
                --model-size "$MODEL_SIZE" \
                --output-dir "$OUTPUT_DIR" \
                --batch-size "$BATCH_SIZE" \
                --per-core-batch-size "$PER_CORE_BATCH_SIZE" \
                --context-length "$CONTEXT_LENGTH" \
                --cuda-device "$CUDA_DEVICE" \
                "${extra_args[@]}" \
                || {
                    log_error "Run failed for $dataset / $term"
                    continue
                }

            log_info "Completed $dataset / $term"
        done
    done

    log_section "Pipeline Completed!"
    echo "Results saved to: $OUTPUT_DIR"
}

main "$@"
