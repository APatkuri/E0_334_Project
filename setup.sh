#!/usr/bin/env bash
# Set up the project on a fresh Linux machine: virtual environment, packages, datasets and weights.
#
#   DATA_ROOT=$(realpath ../datasets) bash setup.sh          # everything, data outside the repo
#   bash setup.sh --glue rte mrpc cola --no_vision           # arguments go to prepare_data.py
#
# Environment variables:
#   DATA_ROOT        where datasets and model weights go (default: ./data)
#                      $DATA_ROOT/huggingface  GLUE and BERT (used as HF_HOME)
#                      $DATA_ROOT/vision       CIFAR-10 and Fashion-MNIST
#   TORCH_INDEX_URL  PyTorch wheel index for a specific CUDA build, e.g.
#                    https://download.pytorch.org/whl/cu121 (see pytorch.org/get-started)
#   VENV             where to create the virtual environment (default: .venv)
#   PYTHON           Python interpreter to use (default: python3, needs >= 3.9)
set -euo pipefail
cd "$(dirname "$0")"

VENV=${VENV:-.venv}
PYTHON=${PYTHON:-python3}
mkdir -p "${DATA_ROOT:=$PWD/data}"
DATA_ROOT=$(cd "$DATA_ROOT" && pwd)
export DATA_ROOT HF_HOME="$DATA_ROOT/huggingface"

# Debian/Ubuntu ship venv without pip unless python3-venv is installed; fall back to virtualenv.
if ! "$PYTHON" -m venv "$VENV" 2>/dev/null; then
    rm -rf "$VENV"
    "$PYTHON" -m virtualenv --version >/dev/null 2>&1 || "$PYTHON" -m pip install --user virtualenv
    "$PYTHON" -m virtualenv "$VENV"
fi

# Activating the environment also points every script at the data.
if ! grep -q "export DATA_ROOT=" "$VENV/bin/activate"; then
    printf '\nexport DATA_ROOT="%s"\nexport HF_HOME="%s"\n' "$DATA_ROOT" "$HF_HOME" >> "$VENV/bin/activate"
fi
source "$VENV/bin/activate"
pip install --upgrade pip

if [ -n "${TORCH_INDEX_URL:-}" ]; then
    pip install torch torchvision --index-url "$TORCH_INDEX_URL"
fi
pip install -r requirements.txt

python prepare_data.py "$@"

echo
echo "Done. Data is in $DATA_ROOT"
echo "Activate the environment with:  source $VENV/bin/activate"
