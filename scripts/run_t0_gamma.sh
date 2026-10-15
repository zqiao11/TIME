#!/usr/bin/env bash
set -euo pipefail

# Set TIME_DATASET to the downloaded Real-TSF/TIME snapshot before running.
# Override DEVICE, BATCH_SIZE, CONTEXT_LENGTH, or OUTPUT_DIR as needed.
: "${TIME_DATASET:?Set TIME_DATASET to the downloaded Real-TSF/TIME dataset path}"

device_args=()
if [[ -n "${DEVICE:-}" ]]; then
  device_args=(--device "$DEVICE")
fi

uv run --extra t0-gamma python experiments/t0_gamma.py \
  "${device_args[@]}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --context-length "${CONTEXT_LENGTH:-8192}" \
  --output-dir "${OUTPUT_DIR:-output/results/t0_gamma}" \
  "$@"
