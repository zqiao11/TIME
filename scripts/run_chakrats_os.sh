#!/usr/bin/env bash
set -euo pipefail

# Run ChakraTS-OS-10M on the full TIME benchmark using uv (Python 3.11, as in the README).
# Results go to output/results/ChakraTS-OS-10M/. Arguments go to experiments/chakrats_os.py:
#   bash scripts/run_chakrats_os.sh --dataset Crypto/D --terms short

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is required to run this script" >&2
  echo "install it from https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

YHATMODELS="yhatmodels==0.1.0"

# pyarrow 26 needs NumPy 2, and TIME pins NumPy 1.26.
uv run --python 3.11 --with "${YHATMODELS}" --with "pyarrow<26" python experiments/chakrats_os.py "$@"
