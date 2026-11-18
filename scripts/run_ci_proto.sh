set -euo pipefail

# Run CI-Proto on the full TIME benchmark using uv (Python 3.11, as in the README).
# Results go to output/results/ci-proto/. Arguments go to experiments/ci_proto.py:
#   bash scripts/run_ci_proto.sh --dataset Crypto/D --terms short

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is required to run this script" >&2
  echo "install it from https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

CONTINUOUSINTELLIGENCE="continuousintelligence==0.1.1"

# The release checkpoint was validated with PyTorch 2.13.0 from PyPI (CUDA on Linux). uv installs
# the package with its own requirements, such as einops 0.8, on top of TIME's environment.
# pyarrow 26 needs NumPy 2, and TIME pins NumPy 1.26.
uv run --python 3.11 \
  --with "${CONTINUOUSINTELLIGENCE}" --with "torch==2.13.0" --with "pyarrow<26" \
  python experiments/ci_proto.py "$@"
