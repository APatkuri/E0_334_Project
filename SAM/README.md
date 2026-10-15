# SAM variants

Self-contained PyTorch implementations of the SAM optimizers in `Papers/`.

| Class | Paper | Update (g = gradient, g_adv = gradient at perturbed weights) |
|-------|-------|--------------------------------------------------------------|
| `SAM` | Foret et al. 2021, Bahri et al. 2022 | `w_adv = w + ρ g/‖g‖`, step with `g_adv`. `adaptive=True` → ASAM |
| `GSAM` | Zhuang et al. 2022 | step with `g_adv − α g⊥` (`g⊥` = part of `g` orthogonal to `g_adv`) |
| `WSAM` | Yue et al. 2023 | step with `g`, then `− lr·γ/(1−γ)·(g_adv − g)` |
| `EigenSAM` | Luo et al. 2025 | perturb by `ρ (g/‖g‖ + α·sign⟨g,v⟩·v⊥)`, `v` = top Hessian eigenvector |
| `XSAM` | Duan et al. 2026 | step with `(I − α v vᵀ) g_adv/‖g_adv‖` |

## Usage

The closure returns the loss and does **not** call `backward()`.

```python
from sam_optim import SAM, GSAM, WSAM, EigenSAM, XSAM

opt = SAM(model.parameters(), torch.optim.AdamW, rho=0.05, lr=2e-5, weight_decay=0.01)
for x, y in loader:
    loss = opt.step(lambda: criterion(model(x), y))
    scheduler.step()
```

Extra kwargs go to the base optimizer; parameter groups and schedulers work as usual.

```bash
python train_glue.py --task sst2 --optim sam --ascent_frac 0.25
python train_glue.py --task rte  --optim gsam --rho_min 0.01 --alpha 0.1
python train_glue.py --task mrpc --optim wsam --gamma 0.9
python train_glue.py --task sst2 --optim eigensam --alpha 0.2
python train_glue.py --task sst2 --optim xsam --alpha 0.2
```

## Paper defaults

| Method | Hyperparameters |
|--------|-----------------|
| SAM | ρ = 0.05 (vision); ρ = 0.15 + ascent micro-batch of 1/4 for T5 (Bahri et al.) |
| GSAM | ρ_max ∈ [0.04, 0.6], ρ_min ∈ [0, 0.2], α ∈ [0.005, 1.0] (Table 4) |
| WSAM | γ ∈ [0, 1); γ = 0.5 is SAM, γ > 0.5 weights sharpness more |
| Eigen-SAM | α = 0.2, `eig_every` p = 100, `power_iters` q = 5; α = 0 is SAM |
| X-SAM | α = 0.2, ρ as for SAM |

## Notes

- **Cost:** ~2 forward/backward passes per step. Eigen-SAM and X-SAM add `q` HVPs every `p` steps; these need `attn_implementation="eager"` (set automatically in `train_glue.py`).
- **X-SAM:** uses the projector `I − α v vᵀ` (Thm. 5.1) instead of Alg. 2's sign term, so the sharp component always shrinks. Gradient is normalized (Eq. 9); pass `normalize_grad=False` to disable.
- **WSAM:** `decouple=False` gives Coupled-WSAM.
- **BatchNorm:** both forward passes update running stats. Irrelevant for transformers.
- `EigenSAM.top_eigenvalue` / `XSAM.top_eigenvalue` hold the latest λmax estimate.
- fp32, single-device only (no AMP/DDP).
