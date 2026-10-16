#!/usr/bin/env bash
set -euo pipefail

# Run Timestral on the full TIME benchmark using uv (Python 3.11, as in the README).
# Pick the model size with SIZE (5M, 28M, 123M or 209M), for example:
#   SIZE=5M bash scripts/run_timestral.sh
# Results go to output/results/Timestral-${SIZE}/.

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is required to run this script" >&2
  echo "install it from https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

SIZE="${SIZE:-209M}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/results}"
TIMESTRAL="timestral @ git+https://github.com/mistralai/timestral.git@a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff"

# pyarrow 26 needs NumPy 2, and TIME pins NumPy 1.26.
uv run --python 3.11 --with "${TIMESTRAL}" --with "pyarrow<26" python experiments/timestral_model.py \
  --model-id "mistralai/Timestral-${SIZE}" \
  --dataset all_datasets \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}"
