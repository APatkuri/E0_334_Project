# Reduced Maximum-Hessian runs

A smaller version of the full plan that fits before the deadline: **one seed (seed 0)**, no Fig. 3 extras. Each run follows the paper's protocol (SGD/SAM at constant lr to 99% train accuracy); only the number of runs is cut. Job names match `../../sweep.py`, so these results count towards the full phases.

| Step | Figs. | Runs | Est. time (1 × RTX 3090) |
|------|-------|------|--------------------------|
| `vision-smoke` | — | 2 | minutes |
| `vision-3.4` | 8–9 | 32 | ~1 h |
| `vision-3.2` | 4–5 | 6 | ~1 h |
| `vision-3.1`, batch 100 and 5000 | 1–2 | 14 | ~4–5 h |
| `vision-3.3` | 6–7 | 24 | ~7–10 h |
| `vision-3.5` | 10–11 | 20 | ~4–5 h |
| `vision-a3` | 14 | 24 | ~5–8 h |
| **Total** | | **122** (full: 503) | **~1–1.5 days** |

Times are estimates, not yet measured; check them against the first logs.

## Usage

```bash
tmux new -s kaur
bash Maximum-Hessian/reduced/run_vision.sh 0      # argument: GPU index
cd Maximum-Hessian && python plot_results.py ../results/vision/*.jsonl ...   # see ../README.md
```

## Differences from the paper

- 1 seed per setting (paper: 3–5; Fig. 14 already uses 1), so no error bars.
- Not run: Fig. 3's batch 200 / 500 and full-batch GD.
- Batch-5000 runs at small lr may hit the 1000-epoch cap (10k steps) before 99% train accuracy; they are flagged `reached_target: false` and left out of the plots.

## Later, if time allows

```bash
python sweep.py vision-3.1 --gpus 0 --per_gpu 2 --match '^batch_size(200|500)_.*_s0$'   # Fig. 3 (14 runs)
python sweep.py vision-3.1 --gpus 0 --per_gpu 2 --match '^batch_size(100|5000)_.*_s1$'  # seed 1, Figs. 1-2
```
