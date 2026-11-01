#!/usr/bin/env bash
set -euo pipefail

# FLASHMIND_API_KEY must be an evaluator key. Defaults to the full Nano suite.
# Example subset: bash scripts/run_flashpdm.sh --model pdm-pico --dataset Crypto/D --terms short
# TIME_DATASET optionally points to an existing dataset download.
uv run --python 3.12 --with 'flashpdm[batch]==0.1.12' --with 'pyarrow==17.0.0' \
  --with 'huggingface-hub<1' python experiments/flashpdm_model.py "$@"
