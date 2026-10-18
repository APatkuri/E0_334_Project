# Entropy-SGD

Self-contained PyTorch implementation of Chaudhari et al. 2017 (`Papers/Entropy-SGD.pdf`), with the same wrapper interface as `../SAM`.

**Update:** run `L` SGLD steps from `x' = x` on `f(x') + γ/2‖x − x'‖²` (new mini-batch each), averaging `μ ← (1−α)μ + αx'`. Then step the outer optimizer with `x − μ`. The scope grows as `γ_t = γ₀(1+γ₁)^t`.

## Usage

The closure returns the loss, does **not** call `backward()`, and must draw a **new mini-batch on every call** (it runs `L` times per step).

```python
from entropy_optim import EntropySGD

opt = EntropySGD(model.parameters(), torch.optim.AdamW, sgld_steps=5, gamma=0.03,
                 gamma_growth=1e-3, sgld_optimizer=torch.optim.Adam, sgld_lr=2e-5,
                 lr=2e-5, weight_decay=0.01)
stream = iter(endless_loader)
for _ in range(num_updates):
    loss = opt.step(lambda: criterion(*next(stream)))
    scheduler.step()
```

Extra kwargs go to the outer optimizer; schedulers affect only the outer lr.

| Argument | Paper | Meaning |
|----------|-------|---------|
| `sgld_steps` | L | SGLD steps (mini-batches) per update |
| `gamma`, `gamma_growth` | γ₀, γ₁ | scope and its growth per update |
| `sgld_lr` | η′ | SGLD step size (fixed) |
| `noise` | ε | thermal noise: adds `√η′ · ε · N(0, I)` |
| `alpha` | α | weight of newest sample in μ |
| `sgld_optimizer`, `sgld_kwargs` | — | inner optimizer: SGD (default), SGD + Nesterov, or Adam |
| `scale_by_gamma` | — | outer gradient `γ(x − μ)` if `True`, else `x − μ` (default) |
| `grad_clip` | — | clip ‖∇f(x′)‖ inside the SGLD loop (RNNs) |

```bash
python train_glue.py --task sst2 --optim adamw              # baseline
python train_glue.py --task sst2 --optim entropysgd --sgld_steps 5 --gamma 0.03 --gamma_growth 1e-3
python train_glue.py --task rte  --optim entropysgd --sgld_optim sgd --sgld_lr 1e-3 --sgld_momentum 0.9
```

### GLUE outputs and analysis

Run from this directory. By default, every GLUE run gets a unique folder under
`../results/glue/`; override the destination with `--output_dir`.

```bash
python train_glue.py --task sst2 --optim entropysgd
python train_glue.py --task sst2 --optim adamw --seed 43
python analyze_glue.py
```

Each run folder contains `best_model/` (the tokenizer and model checkpoint selected by the task's
validation metric), `metrics.json` (arguments and per-epoch train/validation history),
`steps.jsonl` (periodic training losses), `training_curves.png`, `confusion_matrix.png`, and
`best_validation_predictions.csv`. For Entropy-SGD runs, `entropy_dynamics.png` tracks the scoping
parameter $\gamma_t$ and gradient ratio $\|x - \mu\| / \|\nabla f\|$.

When `--eval_landscape` is enabled (default), the following additional sharpness artifacts are generated:
- `robustness_perturbation.png`: Validation loss/accuracy under increasing Gaussian weight perturbations (empirical flatness).
- `loss_landscape_1d.png`: 1D cross-section of the loss surface along a filter-normalized direction.
- `landscape_metrics.json`: Base loss, accuracy, top Hessian eigenvalue ($\lambda_{\max}$ via Pearlmutter's power iteration), and noise curves.

To interpolate directly between two fine-tuned checkpoints (e.g. AdamW vs. Entropy-SGD):
```bash
python analyze_landscape.py \
    --checkpoint ../results/glue/<esgd_run>/best_model \
    --checkpoint2 ../results/glue/<adamw_run>/best_model \
    --task sst2
```

`analyze_glue.py` aggregates completed runs by task, optimizer, and settings, reporting mean validation score, accuracy, and wall-clock training time (`mean_time_sec`), then writes `summary.csv`, `summary.json`, and `optimizer_comparison.png` under `../results/glue/`. These are validation results, not hidden GLUE test-set scores.

### Paper reproduction results

The GLUE/BERT experiment is separate from the paper's image and language-model experiments. The
paper reproduction scripts already write per-run JSON files under `../results/esgd/` when launched
through the repository-level sweep; the analyzer compares those results to the paper's Table 1 and
generates its reproduction plots:

```bash
# Run from the repository root, not this directory:
python sweep.py esgd-smoke --gpus 0
python Entropy-SGD/analyze.py --plot
```

`esgd-smoke` checks code paths with shortened runs; it is not a full paper reproduction. The
`esgd-mnist`, `esgd-cifar`, `esgd-ptb`, `esgd-char`, and `esgd-5.1` phases run the experiments
listed above and can take substantially longer.

## Paper defaults

- L ∈ [5, 20], α = 0.75, ε ∈ [1e-4, 1e-3], η′ ∈ [0.1, 1] with SGD.
- γ₀ = 1e-4 (MNIST) or 0.03 (CIFAR), γ₁ = 1e-3. Fixed γ for RNNs (1e-3 PTB, 0.01 char-LSTM).
- Momentum 0.9 for CNNs, 0.5 for RNNs (β₁ = 0.5 for Entropy-Adam).
- **Tuning γ:** pick γ so `opt.grad_ratio` (‖outer grad‖ / ‖∇f‖) is about 1.

## Reproducing the paper

| File | Purpose |
|------|---------|
| `train_image.py` | Sec. 5.2–5.3: mnistfc, LeNet (MNIST), All-CNN-BN (CIFAR-10, GCN + ZCA) |
| `train_lm.py` | Sec. 5.4: PTB-LSTM (word level), char-LSTM (War and Peace) |
| `spectrum.py` | Sec. 5.1: exact Hessian eigenvalues of small nets, diagonal Fisher of All-CNN-BN |
| `analyze.py` | Table 1 next to the paper's numbers; `--plot` for Figs. 1, 3–6 |

Paper settings are the scripts' defaults (`PAPER` tables, with sections), so a run needs only `--model`/`--task`, `--optim`, `--seed`. Run from the repo root:

| Phase | Paper | Runs |
|-------|-------|------|
| `esgd-smoke` | every code path, ~10 min | 11 |
| `esgd-mnist` | Sec. 5.2, Table 1, Fig. 4 | 2 nets × {Adam, Entropy-SGD} × 5 seeds |
| `esgd-cifar` | Sec. 5.3, Table 1, Fig. 5 | {SGD, Entropy-SGD} × 5 |
| `esgd-ptb` | Sec. 5.4.1, Table 1, Fig. 6a | {SGD, Entropy-SGD} × 5 |
| `esgd-char` | Sec. 5.4.2, Table 1, Fig. 6b | {Adam, Entropy-Adam} × 4 |
| `esgd-5.1` | Sec. 5.1, Figs. 1, 3 | 4 nets, 1–5 runs each |

```bash
python sweep.py esgd-smoke --gpus 0
python sweep.py esgd-mnist --gpus 0 --per_gpu 3      # then esgd-cifar, esgd-ptb, esgd-char, esgd-5.1
python Entropy-SGD/analyze.py --plot
```

**Not stated in the paper (ours):** batch size 128 (images), MNIST standardization, LeNet layer sizes (Caffe 20-50-500), GCN scale 55 and ZCA bias 0.1 (Goodfellow et al.'s pylearn2), weight decay inside the SGLD loop, number of runs for CIFAR/PTB (5), PTB batch/dropout/init/clip and loss summed over the 35 time steps from Zaremba et al., char-LSTM batch 50 / clip 5 / init from char-rnn, Entropy-Adam's η′ = 1 and ε = 1e-4, momentum 0.5 in both loops for RNNs, Sec. 5.1 training rates and the 10k-image Hessian subset.

**Known differences:**
- War and Peace (Karpathy's file) has 83 distinct characters, not 87.
- small-mnistfc: the text says 32 hidden units, but 50,890 parameters means 64. We use 64.
- All-CNN-BN has 1.37M parameters, not "≈ 1.6M". The architecture is All-CNN-C plus BN.
- Table 1 lists 120 and 80 effective epochs for mnistfc and LeNet, but the text gives 5 epochs × L = 20 = 100. We follow the text.
- The char-LSTM "perplexity" of ≈ 1.22 is a per-character loss in nats. `analyze.py` compares it to `test_loss`.

## Notes

- **Cost:** L forward/backward passes per update. `--epochs` in `train_glue.py` counts data passes, so Entropy-SGD makes L× fewer updates than AdamW.
- **Memory:** two extra weight copies (`x`, `μ`) plus inner optimizer state.
- **GLUE defaults are untuned.** The paper's η′ suits SGD on CNNs, so `train_glue.py` uses Adam inside with `η′ = --lr`.
- Inner optimizer is rebuilt every step (as in the reference code). In `train_glue.py` weight decay applies only to the outer step; in `train_image.py` it is part of f, inside the SGLD loop.
- `opt.gamma` holds the current scope. fp32, single-device only (no AMP/DDP).
