#!/bin/bash
# =============================================================================
# PatchTST Training Pipeline Script
# =============================================================================
# This script runs the PatchTST pipeline:
# 1. Hyperparameter tuning (tune) - find best hyperparameters using Optuna
# 2. Testing (test) - train with best hyperparameters and evaluate
#
# Note: Phase 2 (train) is skipped by default because test mode already
# trains the model internally. Set SKIP_TRAIN=false to enable it.
#
# Usage:
#   Arguments:
#     $1: datasets (space-separated, use quotes for multiple datasets)
#     $2: terms (comma-separated: short,medium,long)
#
#   Single argument behavior:
#     - If contains "/" -> treated as datasets (e.g., "Water_Quality_Darwin/15T")
#     - If contains "," -> treated as terms (e.g., "short,medium")
#     - If is "short"/"medium"/"long" -> treated as terms
#     - Otherwise -> treated as datasets
#
#   Examples:
#     ./scripts/run_patch_tst.sh                                    # All default datasets, all terms
#     ./scripts/run_patch_tst.sh "Water_Quality_Darwin/15T"         # One dataset, all terms
#     ./scripts/run_patch_tst.sh "SG_Weather/D SG_PM25/H"          # Multiple datasets, all terms
#     ./scripts/run_patch_tst.sh "" short                            # All datasets, one term
#     ./scripts/run_patch_tst.sh "SG_Weather/D" short               # One dataset, one term
#     ./scripts/run_patch_tst.sh "SG_Weather/D SG_PM25/H" short,medium  # Multiple datasets, multiple terms
#     ./scripts/run_patch_tst.sh short                               # All datasets, one term (backward compatible)
#
#   Environment variables (alternative to command line args):
#     DATASETS="SG_Weather/D SG_PM25/H" ./scripts/run_patch_tst.sh
#     TERMS="short,medium" ./scripts/run_patch_tst.sh
# =============================================================================

set -e  # Exit on error

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
    # Only one argument: need to determine if it's datasets or terms
    # Priority: check for dataset format (contains "/") or term format (short/medium/long or comma-separated)
    if [[ "$1" == *"/"* ]]; then
        # Contains "/" -> treat as datasets (e.g., "SG_Weather/D" or "Water_Quality_Darwin/15T")
        IFS=' ' read -ra DATASET_LIST <<< "$1"
        TERMS=("short" "medium" "long")
    elif [[ "$1" == *","* ]]; then
        # Contains comma -> treat as terms (e.g., "short,medium")
        DATASET_LIST=("${DEFAULT_DATASETS[@]}")
        IFS=',' read -ra TERMS <<< "$1"
    elif [[ "$1" == *" "* ]]; then
        # Contains spaces -> treat as datasets (multiple datasets)
        IFS=' ' read -ra DATASET_LIST <<< "$1"
        TERMS=("short" "medium" "long")
    elif [[ "$1" == "short" ]] || [[ "$1" == "medium" ]] || [[ "$1" == "long" ]]; then
        # Single term value -> treat as terms (backward compatibility)
        DATASET_LIST=("${DEFAULT_DATASETS[@]}")
        TERMS=("$1")
    else
        # Default: treat as datasets (most common use case when passing dataset name)
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
OUTPUT_DIR="${OUTPUT_DIR:-./output/results/patch_tst}"
N_TRIALS="${N_TRIALS:-15}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
SKIP_TUNE="${SKIP_TUNE:-false}"  # Set to "true" to skip tuning
# SKIP_TRAIN defaults to true because Phase 2 is redundant:
# - test mode already calls run_training() internally
# - we don't save checkpoints between train and test
# Set to "false" if you want to run training separately (e.g., for debugging)
SKIP_TRAIN="${SKIP_TRAIN:-true}"

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

# Get the script directory (resolved to absolute path)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATASETS_YAML="$PROJECT_ROOT/src/timebench/config/datasets.yaml"

# Get available terms for a dataset from datasets.yaml
get_available_terms() {
    local dataset="$1"
    python3 -c "
import yaml

with open('$DATASETS_YAML') as f:
    config = yaml.safe_load(f)

dataset = '$dataset'
datasets = config.get('datasets', {})

if dataset in datasets:
    ds_config = datasets[dataset]
    terms = []
    for term in ['short', 'medium', 'long']:
        if term in ds_config:
            terms.append(term)
    print(' '.join(terms))
else:
    # Dataset not found, return all terms as fallback
    print('short medium long')
"
}

# -----------------------------------------------------------------------------
# Main Pipeline
# -----------------------------------------------------------------------------
main() {
    log_section "PatchTST Training Pipeline"
    echo "Datasets: ${DATASET_LIST[*]}"
    echo "Terms: ${TERMS[*]}"
    echo "Output directory: $OUTPUT_DIR"
    echo "N trials: $N_TRIALS"
    echo "CUDA device: $CUDA_DEVICE"
    echo "Skip tune: $SKIP_TUNE"
    echo "Skip train: $SKIP_TRAIN"

    # Create output directory
    mkdir -p "$OUTPUT_DIR"

    # Calculate total experiments (considering available terms per dataset)
    total_experiments=0
    for dataset in "${DATASET_LIST[@]}"; do
        available_terms_str=$(get_available_terms "$dataset")
        IFS=' ' read -ra available_terms <<< "$available_terms_str"
        for term in "${TERMS[@]}"; do
            for avail_term in "${available_terms[@]}"; do
                if [ "$term" == "$avail_term" ]; then
                    total_experiments=$((total_experiments + 1))
                    break
                fi
            done
        done
    done
    current=0
    echo "Total experiments (filtered by datasets.yaml): $total_experiments"

    # Iterate over datasets and terms
    for dataset in "${DATASET_LIST[@]}"; do
        # Get available terms for this dataset from datasets.yaml
        available_terms_str=$(get_available_terms "$dataset")
        IFS=' ' read -ra available_terms <<< "$available_terms_str"

        for term in "${TERMS[@]}"; do
            # Check if the term is available for this dataset
            term_available=false
            for avail_term in "${available_terms[@]}"; do
                if [ "$term" == "$avail_term" ]; then
                    term_available=true
                    break
                fi
            done

            if [ "$term_available" != "true" ]; then
                log_info "Skipping $dataset / $term (term not defined in datasets.yaml)"
                continue
            fi

            current=$((current + 1))

            log_section "[$current/$total_experiments] $dataset / $term"

            # -------------------------------------------------------------
            # Phase 1: Hyperparameter Tuning
            # -------------------------------------------------------------
            if [ "$SKIP_TUNE" != "true" ]; then
                log_info "Phase 1: Hyperparameter Tuning"

                python experiments/patch_tst.py \
                    --dataset "$dataset" \
                    --terms "$term" \
                    --mode tune \
                    --output-dir "$OUTPUT_DIR" \
                    --n-trials "$N_TRIALS" \
                    --cuda-device "$CUDA_DEVICE" \
                    || {
                        log_error "Tuning failed for $dataset / $term"
                        continue
                    }

                log_info "Tuning completed"
            else
                log_info "Skipping tuning (SKIP_TUNE=true)"
            fi

            # -------------------------------------------------------------
            # Phase 2: Training with Best Hyperparameters
            # -------------------------------------------------------------
            if [ "$SKIP_TRAIN" != "true" ]; then
                log_info "Phase 2: Training with Best Hyperparameters"

                python experiments/patch_tst.py \
                    --dataset "$dataset" \
                    --terms "$term" \
                    --mode train \
                    --output-dir "$OUTPUT_DIR" \
                    --cuda-device "$CUDA_DEVICE" \
                    || {
                        log_error "Training failed for $dataset / $term"
                        continue
                    }

                log_info "Training completed"
            else
                log_info "Skipping training (SKIP_TRAIN=true)"
            fi

            # -------------------------------------------------------------
            # Phase 3: Testing and Saving Predictions
            # -------------------------------------------------------------
            log_info "Phase 3: Testing and Saving Predictions"

            python experiments/patch_tst.py \
                --dataset "$dataset" \
                --terms "$term" \
                --mode test \
                --output-dir "$OUTPUT_DIR" \
                --cuda-device "$CUDA_DEVICE" \
                || {
                    log_error "Testing failed for $dataset / $term"
                    continue
                }

            log_info "Testing completed"
            log_info "Progress: $current/$total_experiments experiments done"
        done
    done

    log_section "Pipeline Completed!"
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Output structure:"
    echo "  $OUTPUT_DIR/"
    echo "  ├── optuna/              # Optuna study databases"
    echo "  ├── hparams/             # Best hyperparameters (JSON)"
    echo "  ├── checkpoints/         # Model checkpoints"
    echo "  └── {dataset}/{term}/    # Predictions and metrics"
}

# Run main function
main "$@"

