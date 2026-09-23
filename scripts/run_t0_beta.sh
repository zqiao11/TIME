#!/usr/bin/env bash
set -euo pipefail

# Run t0-beta on the full TIME benchmark using uv and the PyTorch tfc-t0 runtime.
# Override these environment variables as needed, for example:
#   BATCH_SIZE=8 DEVICE=cuda bash scripts/run_t0_beta.sh

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is required to run this script" >&2
  echo "install it from https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

BATCH_SIZE="${BATCH_SIZE:-32}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-8192}"
DEVICE="${DEVICE:-auto}"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/results/t0_beta}"

uv run --extra t0 python experiments/t0_beta.py \
  --dataset all_datasets \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --context-length "${CONTEXT_LENGTH}" \
  --device "${DEVICE}" \
  --torch-dtype "${TORCH_DTYPE}"
