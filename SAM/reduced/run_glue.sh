#!/usr/bin/env bash
# All five SAM variants on GLUE (our experiments), seed 0: AdamW lr, then SAM rho, then GSAM, WSAM,
# Eigen-SAM and X-SAM with SAM's rho, each tuning its own knob. Each stage picks its settings from
# the previous one. Entropy-Adam (not a SAM variant) is skipped. Run inside tmux, from any directory:
#
#   bash SAM/reduced/run_glue.sh 6            # GPU index (default 0); needs ~40 GB at 4 jobs per GPU
set -euo pipefail
cd "$(dirname "$0")/../.."
GPU=${1:-0}
PY=${PYTHON:-python}

$PY sweep.py glue-lr       --gpus "$GPU" --per_gpu 4                   # 20 runs
$PY sweep.py glue-rho      --gpus "$GPU" --per_gpu 4 --match '_sam_'   # 30 runs
$PY sweep.py glue-variants --gpus "$GPU" --per_gpu 4                   # 85 runs: GSAM 15, WSAM 50, Eigen-SAM 15, X-SAM 5
