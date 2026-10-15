# SAM variants in PyTorch

Simple, self-contained implementations of the SAM optimizers from the papers in `Papers/`.

| Class      | Paper | Update rule (g = clean gradient, g_adv = gradient at w + perturbation) |
|------------|-------|-----------------------------------------------------------------------|
| `SAM`      | Foret et al. 2021, Bahri et al. 2022 (`SAM.pdf`) | `w_adv = w + ρ g/‖g‖`; step with `g_adv`. Optional ascent micro-batch; `adaptive=True` gives ASAM |
| `GSAM`     | Zhuang et al. 2022 (`GSAM.pdf`) | step with `g_adv − α g⊥`, where `g⊥` is the part of `g` orthogonal to `g_adv`; optional `ρ_t` tied to the learning rate |
| `WSAM`     | Yue et al. 2023 (`WSAM.pdf`) | base step with `g`, then `− lr·γ/(1−γ)·(g_adv − g)` (decoupled; `decouple=False` for Coupled-WSAM) |
| `EigenSAM` | Luo et al. 2025 (`EigenSAM.pdf`) | perturbation `ρ (g/‖g‖ + α·sign⟨g,v⟩·v⊥)`; `v` = top Hessian eigenvector from power iteration |
| `XSAM`     | Duan et al. 2026 (`XSAM.pdf`) | step with `(I − α v vᵀ) g_adv/‖g_adv‖` |

## Usage

The closure returns the loss and does **not** call `backward()`. Each optimizer computes its own gradients.

```python
from sam_optim import SAM, GSAM, WSAM, EigenSAM, XSAM

opt = SAM(model.parameters(), torch.optim.AdamW, rho=0.05, lr=2e-5, weight_decay=0.01)
for x, y in loader:
    loss = opt.step(lambda: criterion(model(x), y))
    scheduler.step()
```

Any extra keyword arguments (`lr`, `momentum`, `weight_decay`, ...) go to the base optimizer. Parameter groups and LR schedulers work as usual.

## Scripts

```bash
python train_glue.py --task sst2 --optim sam --ascent_frac 0.25
python train_glue.py --task rte  --optim gsam --rho_min 0.01 --alpha 0.1
python train_glue.py --task mrpc --optim wsam --gamma 0.9
python train_glue.py --task sst2 --optim eigensam --alpha 0.2
python train_glue.py --task sst2 --optim xsam --alpha 0.2
```

## Defaults from the papers

- SAM: ρ = 0.05 (vision); Bahri et al. use ρ = 0.15 with an ascent micro-batch of 1/4 of the batch for T5.
- GSAM: ρ_max ∈ [0.04, 0.6], ρ_min ∈ [0, 0.2], α ∈ [0.005, 1.0], depending on the model (Table 4).
- WSAM: γ ∈ [0, 1). γ = 0.5 is SAM, and γ > 0.5 weights sharpness more.
- Eigen-SAM: α = 0.2, p = 100 (`eig_every`), q = 5 (`power_iters`). α = 0 is SAM.
- X-SAM: α = 0.2, same ρ as SAM.

## Notes

- **Cost.** Every variant costs about 2 forward/backward passes per step. Eigen-SAM and X-SAM also run `q` Hessian-vector products every `p` steps. The HVPs need double backward, so `train_glue.py` loads the model with `attn_implementation="eager"` for those two.
- **X-SAM sign term.** Alg. 2 writes the correction as `α·sign⟨g,v⟩·g∥`. Since `g∥ = ⟨g,v⟩v` does not depend on the sign of `v`, the code uses the projector `I − α v vᵀ` from Thm. 5.1, which always shrinks the sharp component. By default the gradient is normalized as in Eq. 9, so with SGD every step has length `lr`. Pass `normalize_grad=False` to keep its scale.
- **BatchNorm.** The two forward passes both update BN running statistics. This doesn't matter for transformers (LayerNorm). For CNNs, you can freeze BN momentum during the perturbed pass if needed.
- `EigenSAM.top_eigenvalue` and `XSAM.top_eigenvalue` hold the latest λ_max estimate, which is useful for tracking sharpness.
- Everything is fp32 and single-device. AMP and DDP are not handled.
