#!/bin/bash
# Simons experiments for all datasets (https://github.com/Abel-ai-lab/Simons)
# Runs every TIME task for one checkpoint: MODEL=abel-lab/Simons-25M, -60M, -178M or -297M (default).
# Results go to output/results/<model name>/ (e.g. output/results/Simons-297M).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_NAME="${ENV_NAME:-simons}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
MODEL="${MODEL:-abel-lab/Simons-297M}"
SIMONS="simons @ git+https://github.com/Abel-ai-lab/Simons.git"

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1"
}

setup_conda_env() {
    source "$(conda info --base)/etc/profile.d/conda.sh"

    if conda env list | awk '{print $1}' | grep -x "$ENV_NAME" >/dev/null 2>&1; then
        log_info "Activating existing env: $ENV_NAME"
        conda activate "$ENV_NAME"
    else
        log_info "Creating new env: $ENV_NAME"
        conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
        conda activate "$ENV_NAME"

        log_info "Installing dependencies..."
        pip install "$SIMONS" torch datasets gluonts dotenv
    fi
}

setup_conda_env
cd "$ROOT_DIR"

# Fixed settings: the checkpoint's own context length and forecast-time rules, the nine quantiles
# 0.1, ..., 0.9, and 256 series per forward pass (lower --batch-size if the GPU runs out of memory).
log_info "Running $MODEL on all datasets"
python experiments/simons_model.py --model "$MODEL" --dataset all_datasets --batch-size 256
