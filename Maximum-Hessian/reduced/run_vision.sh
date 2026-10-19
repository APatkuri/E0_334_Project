#!/usr/bin/env bash
# Reduced Kaur et al. (2023) reproduction on one GPU, seed 0 only. Every run keeps the paper's
# protocol; only the number of runs is cut (see README.md). Run inside tmux, from any directory:
#
#   bash Maximum-Hessian/reduced/run_vision.sh 0          # GPU index (default 0)
#
# Finished runs are skipped, so the script can be stopped and restarted at any time.
# Steps go cheapest / most central first, so stopping early still leaves complete figures.
set -uo pipefail
cd "$(dirname "$0")/../.."
GPU=${1:-0}
PY=${PYTHON:-python}

# 0. Smoke test (2 runs, minutes)
$PY sweep.py vision-smoke --gpus "$GPU"
# 1. Figs. 8-9: MLP + dropout on Fashion-MNIST (32 runs). Tiny model, so 3 at a time.
$PY sweep.py vision-3.4 --gpus "$GPU" --per_gpu 3 --match '_s0$'
# 2. Figs. 4-5: constant batch/lr ratio, lambda_max curve for 20000 (6 runs)
$PY sweep.py vision-3.2 --gpus "$GPU" --per_gpu 2 --match '_s0$'
# 3. Figs. 1-2: lr sweep at batch 100 and 5000 (14 runs)
$PY sweep.py vision-3.1 --gpus "$GPU" --per_gpu 2 --match '^batch_size(100|5000)_.*_s0$'
# 4. Figs. 6-7: SAM rho sweep (24 runs)
$PY sweep.py vision-3.3 --gpus "$GPU" --per_gpu 2 --match '_s0$'
# 5. Figs. 10-11: VGG-11 with and without BN (20 runs)
$PY sweep.py vision-3.5 --gpus "$GPU" --per_gpu 2 --match '_s0$'
# 6. Fig. 14 (appendix): VGG-11 + BN with SAM (24 runs; the paper also uses 1 seed here)
$PY sweep.py vision-a3 --gpus "$GPU" --per_gpu 2 --match '_s0$'
