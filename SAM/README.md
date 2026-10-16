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

## Reproducing the papers' image tables

`train_image.py` trains any model/optimizer pair (SGD + momentum 0.9, cosine schedule, crop + flip on the GPU) and records test accuracy, final training loss and λmax. `analyze_image.py` prints every table as `ours mean ± std (n) | paper`. Protocols and their sources are in `../sweep.py`. Phases, cheapest first:

| Phase | Tables | Runs |
|-------|--------|------|
| `sam-smoke` | every model, optimizer and dataset, a few steps | 12 |
| `sam-r18` | X-SAM T1 + Eigen-SAM T1, ResNet-18 (one shared protocol) | 63 |
| `sam-alexnet` | X-SAM T1, AlexNet | 45 |
| `sam-vit` | Eigen-SAM T2, ViT-B/16 fine-tuning | 18 |
| `wsam-r18` | WSAM T2, T8 (Coupled-WSAM), T9 (minima), ResNet-18 | 50 |
| `wsam-noise` | WSAM T5–T6, label noise | 80 |
| `sam-r50` | X-SAM T1 + Eigen-SAM T1, ResNet-50 | 63 |
| `sam-wrn`, `wsam-wrn` | WRN-28-10 rows (very slow) | 63, 50 |

Seeds are the outer loop, so seed 0 of a whole phase finishes first.

**Not stated in the papers (ours):**
- Architectures: CIFAR ResNets, WRN without dropout, AlexNet for 32×32 inputs (`image_models.py`).
- Input normalization. The cosine schedule is applied per step.
- Extra SAM forward passes don't update BN running stats.
- X-SAM's GSAM α (0.02), WSAM γ (0.88), and its p, q (100, 5, from Eigen-SAM). Its Appendix A is missing from the PDF.
- GSAM uses a constant ρ.
- Coupled-WSAM reuses WSAM's γ.
- Label-noise corruption is fixed across runs, with 5 seeds.
- ViT fine-tuning: batch 64, no weight decay, 224 px, and ρ taken from Eigen-SAM App. E.

**Not reproduced:**
- GSAM's main tables (ImageNet from scratch; transfer from those weights) and WSAM T4 (DeiT on ImageNet): ImageNet is beyond our compute.
- Bahri et al. (T5 Small–XL on TPUs).
- WSAM T7 (Fisher-SAM), the ESAM rows of WSAM T2/T6, and sensitivity figures.

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
- **BatchNorm:** both forward passes update running stats unless the closure prevents it (`train_image.py` freezes them after the first pass). Irrelevant for transformers.
- `EigenSAM.top_eigenvalue` / `XSAM.top_eigenvalue` hold the latest λmax estimate.
- fp32, single-device only (no AMP/DDP).
