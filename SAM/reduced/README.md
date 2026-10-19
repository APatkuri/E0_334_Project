# Reduced SAM-variant runs

A smaller version of the full plan that fits before the deadline: **one seed (seed 0) of everything**, all five variants (SAM, GSAM, WSAM, Eigen-SAM, X-SAM). Each run follows the paper's protocol (epochs, lr, ρ, batch size); only the number of runs is cut. Job names match `../../sweep.py`, so these results count towards the full phases.

| Script | Machine | Runs | Est. time |
|--------|---------|------|-----------|
| `run_images.sh` | 1 × RTX 3090 | X-SAM Table 1, seed 0: AlexNet (15), ResNet-18 (15) | ~1.2 days |
| `run_glue.sh` | 1 × 48 GB GPU | GLUE seed 0: AdamW lr (20), SAM ρ (30), then GSAM α, WSAM γ, Eigen-SAM α, X-SAM (85) | < 1 day |

## Usage

```bash
tmux new -s xsam
bash SAM/reduced/run_images.sh 2      # argument: GPU index
python SAM/analyze_image.py           # tables: ours | paper
```

## Differences from the paper

- 1 seed per setting (papers: 3 or 5), so no standard deviations.
- Not run: ResNet-50 (its λmax needs > 24 GB), WRN-28-10 (too slow).
- GLUE: tuning runs only (`glue-final`'s 5 seeds not run), no Entropy-Adam.

## Later, if time allows

```bash
# seeds 1-2 for SAM and X-SAM, ResNet-18 (12 runs, ~0.9 days)
python sweep.py sam-r18 --gpus 2 --match '^(?!.*datasetsvhn).*optim(sam|xsam)_.*_s[12]$'
# seeds 1-2 for everything in X-SAM Table 1 (AlexNet 30 runs, ResNet-18 30 runs)
python sweep.py sam-alexnet --gpus 2 --per_gpu 3
python sweep.py sam-r18 --gpus 2 --match '^(?!.*(optimsgd|datasetsvhn))'
```
