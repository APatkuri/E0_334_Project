#!/usr/bin/env bash
# Reduced X-SAM Table 1 (ResNet-18 + AlexNet) on one GPU, seed 0 only. Every run keeps the paper's
# protocol; only the number of runs is cut (see README.md). Run inside tmux, from any directory:
#
#   bash SAM/reduced/run_images.sh 2          # GPU index (default 0)
#
# Finished runs are skipped, so the script can be stopped and restarted at any time.
set -uo pipefail
cd "$(dirname "$0")/../.."
GPU=${1:-0}
PY=${PYTHON:-python}
XSAM_T1='^(?!.*(optimsgd|datasetsvhn))'  # X-SAM Table 1 rows only: no SGD or SVHN (Eigen-SAM's)

# 1. AlexNet, seed 0: 5 methods x 3 datasets (15 runs). Small model, so 3 at a time.
$PY sweep.py sam-alexnet --gpus "$GPU" --per_gpu 3 --match '_s0$'
# 2. ResNet-18, seed 0: 5 methods x 3 datasets (15 runs)
$PY sweep.py sam-r18 --gpus "$GPU" --per_gpu 1 --match "$XSAM_T1.*_s0\$"
