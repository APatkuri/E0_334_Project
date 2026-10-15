# Maximum Hessian eigenvalue

Tools and experiments for Kaur, Cohen & Lipton (2023), *On the Maximum Hessian Eigenvalue and Generalization*. The paper measures **λmax** (largest Hessian eigenvalue of the training loss) and shows five training changes where λmax and test accuracy move independently.

| File | Purpose |
|------|---------|
| `hessian_tools/` | λmax (power iteration), Hessian trace (Hutchinson), λmax between iterates, `hessian_mode`. Works on any model, including the GLUE ones |
| `train_vision.py` | Paper protocol: SGD or SAM at constant lr to 99% train accuracy, then log test accuracy and λmax (one JSON line per run) |
| `plot_results.py` | Paper-style figures: metric vs. hyperparameter, or λmax-vs-accuracy scatter |

## Measuring curvature

```python
from hessian_tools import hessian_mode, top_eigenvalue, hessian_trace

params = [p for p in model.parameters() if p.requires_grad]
batches = [...]                                  # fixed training subset, equal-size batches
loss_fn = lambda b: model(**b).loss              # mean loss on one batch

with hessian_mode(model):                        # dropout off; BN uses batch statistics
    lmax, v = top_eigenvalue(params, loss_fn, batches, iters=20)
    trace, stderr = hessian_trace(params, loss_fn, batches, samples=50)
```

- **Cost:** one HVP per batch per iteration (~3× a training step). `lmax_between` evaluates 8 points.
- **Transformers:** load with `attn_implementation="eager"` (HVPs need double backward).
- **Sign:** always returns the largest *algebraic* eigenvalue, running a shifted second pass if needed.
- **BatchNorm:** stays in train mode inside `hessian_mode`; running stats are restored afterwards.

## Reproducing the paper

`../sweep.py` runs each section as a GPU job queue, one JSON line per run in `../results/vision/`. Settings follow the paper; "ours" marks what it leaves unspecified.

| Phase | Figs. | Setting (VGG-11 on CIFAR-10 unless noted) | Seeds |
|-------|-------|--------------------------------------------|-------|
| `vision-3.1` | 1, 3 | no BN, batch {100, 200, 500, 5000}, lr {1e-4, 3e-4, …, 0.1} (axis 1e-4..1e-1; values ours) | 5 |
| `vision-3.1-fullbatch` | 3 | as above, batch 50000 (GD); very slow | 5 |
| `vision-3.2` | 4, 5 | batch/lr = 10000 and 20000 with batch {100, 200, 500}; λmax every 500 steps for 20000 (Fig. 5) | 5 |
| `vision-3.3` | 6, 7 | SAM ρ {0, 0.02, …, 0.1}, lr {0.005, 0.05}, batch {100, 5000} | 4 |
| `vision-3.4` | 8, 9 | MLP (2 hidden layers, dropout on both) on Fashion-MNIST, p {0, 0.1, …, 0.7}, lr {0.01, 0.1}, batch {100, 5000 (ours)} | 3 |
| `vision-3.5` | 10, 11 | VGG-11 with and without BN, lr {1e-3, 3e-3, …, 0.1} (values ours), batch {100, 5000}, λmax between iterates | 4 |
| `vision-a3` | 14 | VGG-11 + BN with SAM, ρ {0, …, 0.1}, lr {0.005, 0.05}, batch {100, 5000} | 1 |

```bash
cd .. && python sweep.py vision-smoke --gpus 3      # then vision-3.1 ... vision-a3
cd Maximum-Hessian
python plot_results.py ../results/vision/*.jsonl --x lr --group batch_size --where model=vgg11 --where rho=0.0 --where between_iterates=False --out fig1.png
python plot_results.py ../results/vision/*.jsonl --x rho --group batch_size --where lr=0.005 --where model=vgg11 --where between_iterates=False --out fig6.png
python plot_results.py ../results/vision/*.jsonl --x dropout --group lr --where model=mlp --where batch_size=100 --out fig8.png
python plot_results.py ../results/vision/*.jsonl --x lr --group model --where batch_size=100 --where between_iterates=True --out fig10.png
python plot_results.py ../results/vision/*.jsonl --scatter --group batch_size --where model=vgg11 --where rho=0.0 --where between_iterates=False --out fig2.png
```

Not reproduced: Fixup ResNet-32 (Figs. 12, 13) and VGG-11 with GeLU (Fig. 15), both appendix-only.

## Notes

- **Protocol:** plain SGD (no momentum, weight decay, schedule or augmentation). Stops at `--target_acc 0.99` or `--max_epochs`; results flag `reached_target: false` or `diverged: true`.
- **Our choices** (unspecified in the paper): λmax on 5000 training examples (`--lmax_examples`), MLP width 512 (`--hidden`), the exact lr grids, the large batch for dropout (5000), and `--max_epochs 1000` as a cap (runs that hit it have `reached_target: false`).
- **Large batches** are accumulated in chunks of `--chunk` examples, so full-batch GD fits in memory. Lower it (e.g. 250) on small GPUs.
- **SAM** here is plain SAM + SGD, inlined to support chunking. The full family is in `../SAM`.
- fp32, single-device only.
