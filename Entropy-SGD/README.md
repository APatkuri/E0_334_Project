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

```bash
python train_glue.py --task sst2 --optim adamw              # baseline
python train_glue.py --task sst2 --optim entropysgd --sgld_steps 5 --gamma 0.03 --gamma_growth 1e-3
python train_glue.py --task rte  --optim entropysgd --sgld_optim sgd --sgld_lr 1e-3 --sgld_momentum 0.9
```

## Paper defaults

- L ∈ [5, 20], α = 0.75, ε ∈ [1e-4, 1e-3], η′ ∈ [0.1, 1] with SGD.
- γ₀ = 1e-4 (MNIST) or 0.03 (CIFAR), γ₁ = 1e-3. Fixed γ for RNNs (1e-3 PTB, 0.01 char-LSTM).
- Momentum 0.9 for CNNs, 0.5 for RNNs (β₁ = 0.5 for Entropy-Adam).
- **Tuning γ:** pick γ so `opt.grad_ratio` (‖outer grad‖ / ‖∇f‖) is about 1.

## Notes

- **Cost:** L forward/backward passes per update. `--epochs` in `train_glue.py` counts data passes, so Entropy-SGD makes L× fewer updates than AdamW.
- **Memory:** two extra weight copies (`x`, `μ`) plus inner optimizer state.
- **GLUE defaults are untuned.** The paper's η′ suits SGD on CNNs, so `train_glue.py` uses Adam inside with `η′ = --lr`.
- Inner optimizer is rebuilt every step (as in the reference code); weight decay applies only to the outer step.
- `opt.gamma` holds the current scope. fp32, single-device only (no AMP/DDP).
