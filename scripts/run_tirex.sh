#!/bin/bash
# =============================================================================
# TiRex Evaluation Script
# =============================================================================
# This script sets up the TiRex environment and runs the TiRex pipeline for
# selected datasets and terms.
# It ensures a conda environment exists (creates it if missing).
#
# Usage:
#   Arguments:
#     $1: datasets (space-separated, use quotes for multiple datasets)
#     $2: terms (comma-separated: short,medium,long)
#
#   Single argument behavior:
#     - If contains "/" -> treated as datasets (e.g., "IMOS/15T")
#     - If contains "," -> treated as terms (e.g., "short,medium")
#     - If is "short"/"medium"/"long" -> treated as terms
#     - Otherwise -> treated as datasets
#
#   Environment variables (alternative to command line args):
#     DATASETS="IMOS/15T" ./scripts/run_tirex.sh
#     TERMS="short,medium" ./scripts/run_tirex.sh
#
#   Environment setup:
#     ENV_NAME (default: tirex)
#     PYTHON_VERSION (default: 3.11)
# =============================================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Default datasets (can be overridden by command line or environment variable)
DEFAULT_DATASETS=(
    "SG_Weather/D"
    "SG_PM25/H"
    "Water_Quality_Darwin/15T"
    "Coastal_T_S/5T"
    "Coastal_T_S/15T"
    "Coastal_T_S/20T"
    # "Australia_Solar/H"
    # "epf_electricity_price/H"
    # "Finland_Traffic/15T"
    # "ECDC_COVID/D"
    # "NE_China_Wind/H"
)

# Parse command line arguments
# $1: datasets (space-separated string, use quotes for multiple, empty string for defaults)
# $2: terms (comma-separated string)
if [ $# -ge 2 ]; then
    # Two or more arguments provided: $1 = datasets, $2 = terms
    if [ -n "$1" ]; then
        # Non-empty datasets string
        IFS=' ' read -ra DATASET_LIST <<< "$1"
    else
        # Empty string -> use defaults
        DATASET_LIST=("${DEFAULT_DATASETS[@]}")
    fi
    IFS=',' read -ra TERMS <<< "$2"
elif [ $# -eq 1 ]; then
    # Only one argument: determine if it's datasets or terms
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
    # No arguments: use environment variables or defaults
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

# Other configurations
ENV_NAME="${ENV_NAME:-tirex}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
MODEL_SIZE="${MODEL_SIZE:-base}"
MODEL_ID="${MODEL_ID:-}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/results/tirex}"
BATCH_SIZE="${BATCH_SIZE:-128}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-4000}"
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
    fi
}

# -----------------------------------------------------------------------------
# Main Pipeline
# -----------------------------------------------------------------------------
main() {
    cd "$ROOT_DIR"

    log_section "TiRex Evaluation Pipeline"
    echo "Datasets: ${DATASET_LIST[*]}"
    echo "Terms: ${TERMS[*]}"
    echo "Model size: $MODEL_SIZE"
    echo "Model id: ${MODEL_ID:-auto}"
    echo "Output directory: $OUTPUT_DIR"
    echo "Batch size: $BATCH_SIZE"
    echo "Context length: $CONTEXT_LENGTH"
    echo "CUDA device: $CUDA_DEVICE"
    echo "Conda env: $ENV_NAME (python $PYTHON_VERSION)"
    echo "Use val: $USE_VAL"
    if [ -n "$QUANTILES" ]; then
        echo "Quantiles: $QUANTILES"
    fi

    ensure_conda_env

    log_info "Installing dependencies..."
    pip install tirex-ts
    pip install dotenv
    pip install gluonts
    pip install datasets

    # Create output directory
    mkdir -p "$OUTPUT_DIR"

    extra_args=()
    if [ -n "$CONFIG_PATH" ]; then
        extra_args+=(--config "$CONFIG_PATH")
    fi
    if [ "$USE_VAL" = "true" ]; then
        extra_args+=(--val)
    fi
    if [ -n "$MODEL_ID" ]; then
        extra_args+=(--model-id "$MODEL_ID")
    else
        extra_args+=(--model-size "$MODEL_SIZE")
    fi
    if [ -n "$QUANTILES" ]; then
        IFS=',' read -ra QUANTILE_LIST <<< "$QUANTILES"
        extra_args+=(--quantiles "${QUANTILE_LIST[@]}")
    fi

    # Track progress
    total_datasets=${#DATASET_LIST[@]}
    total_terms=${#TERMS[@]}
    total_experiments=$((total_datasets * total_terms))
    current=0

    # Iterate over datasets and terms
    for dataset in "${DATASET_LIST[@]}"; do
        for term in "${TERMS[@]}"; do
            current=$((current + 1))
            log_section "[$current/$total_experiments] $dataset / $term"

            python experiments/tirex_model.py \
                --dataset "$dataset" \
                --terms "$term" \
                --output-dir "$OUTPUT_DIR" \
                --batch-size "$BATCH_SIZE" \
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

# Run main function
main "$@"

