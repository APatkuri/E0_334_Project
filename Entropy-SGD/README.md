# Entropy-SGD in PyTorch

A simple, self-contained implementation of the local-entropy optimizer from `Papers/`. It has the same wrapper interface as the SAM variants in `../SAM`.

| Class        | Paper | Update rule |
|--------------|-------|-------------|
| `EntropySGD` | Chaudhari et al. 2017 (`Entropy-SGD.pdf`) | `L` SGLD steps from `x' = x` on `f(x') + γ/2‖x − x'‖²`, each on a new mini-batch, with running average `μ ← (1−α)μ + αx'`. Then step with `x − μ` (or `γ(x − μ)`). Scope `γ_t = γ₀(1+γ₁)^t` |

## Usage

The closure returns the loss and does **not** call `backward()`. Unlike SAM, it is called `L` times per step, and it should draw a **new mini-batch on every call** (Alg. 1, line 3).

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

Extra keyword arguments (`lr`, `momentum`, `weight_decay`, ...) go to the outer base optimizer. Parameter groups and LR schedulers work as usual, and they affect only the outer learning rate η. The SGLD optimizer is built over the same parameters at every step, with `lr=sgld_lr` and `sgld_kwargs`.

| Argument | Paper | Meaning |
|----------|-------|---------|
| `sgld_steps` | L | SGLD steps (= mini-batches) per update |
| `gamma`, `gamma_growth` | γ₀, γ₁ | scope and its exponential growth per update |
| `sgld_lr` | η′ | SGLD step size, kept fixed |
| `noise` | ε | thermal noise; the SGLD step adds `√η′ · ε · N(0, I)` |
| `alpha` | α | weight of the newest sample in μ |
| `sgld_optimizer`, `sgld_kwargs` | — | SGD (default), SGD + Nesterov, or Adam for the inner loop (Sec. 4.2) |
| `scale_by_gamma` | — | `True`: outer gradient `γ(x − μ)` (Alg. 1 line 7). `False`: `x − μ` (Sec. 4.3, the default) |

## Scripts

```bash
python train_glue.py --task sst2 --optim adamw              # baseline
python train_glue.py --task sst2 --optim entropysgd --sgld_steps 5 --gamma 0.03 --gamma_growth 1e-3
python train_glue.py --task rte  --optim entropysgd --sgld_optim sgd --sgld_lr 1e-3 --sgld_momentum 0.9
```

## Defaults from the paper

- L ∈ [5, 20]; α = 0.75; ε ∈ [1e-4, 1e-3]; η′ ∈ [0.1, 1] with SGD, never annealed.
- Scoping: γ₀ = 1e-4 (MNIST) or 0.03 (CIFAR), growing by a factor of 1.001 per update (γ₁ = 1e-3). For RNNs γ is fixed (1e-3 for PTB, 0.01 for char-LSTM).
- Momentum mattered: 0.9 for CNNs, 0.5 for RNNs (β₁ = 0.5 for Entropy-Adam).
- **Tuning γ** (Sec. 4.2): fix L, ε and η′, then pick γ so that the outer gradient has a magnitude comparable to the plain SGD gradient. `opt.grad_ratio` logs `‖outer gradient‖ / ‖∇f(x)‖` for this purpose. Aim for about 1.

## Notes

- **Cost.** Each update costs L forward/backward passes, one per mini-batch. Compare against other methods by gradient evaluations ("epochs × L" in the paper's plots). `train_glue.py` does this: `--epochs` counts passes over the data, so Entropy-SGD makes L× fewer updates than AdamW. Tune `--lr` or `--epochs` with that in mind.
- **Memory.** Two extra copies of the weights (`x` and `μ`), plus the state of the SGLD optimizer.
- **GLUE defaults are not from the paper.** η′ ∈ [0.1, 1] is meant for plain SGD on CNNs and is likely too large for fine-tuning a transformer. `train_glue.py` therefore uses Adam for the inner loop by default, with `η′ = --lr`. These are starting points and have not been tuned.
- **Inner-loop state is reset every step.** A fresh SGLD optimizer is built for each update, which matches the reference code. Weight decay applies only in the outer step.
- `opt.gamma` holds the current scope. Everything is fp32 and single-device. AMP and DDP are not handled.
