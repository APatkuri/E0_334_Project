# Maximum Hessian eigenvalue in PyTorch

Tools and scripts for Kaur, Cohen and Lipton (2023), *On the Maximum Hessian Eigenvalue and Generalization* (`Papers/`).

This paper proposes no optimizer. It measures **λmax**, the largest eigenvalue of the Hessian of the training loss, and shows five training changes under which λmax and test accuracy move independently. This folder has:

| File | What it does |
|------|--------------|
| `hessian_tools/` | λmax (power iteration), Hessian trace (Hutchinson), λmax between iterates, and `hessian_mode` (dropout off). Works for any model, including the GLUE models in `../SAM` and `../Entropy-SGD` |
| `train_vision.py` | The paper's protocol: SGD or SAM at a constant lr until 99% train accuracy, then record test accuracy and λmax. Writes one JSON line per run |
| `plot_results.py` | Figures in the paper's layout: accuracy and λmax vs. a swept hyperparameter, or a λmax-vs-accuracy scatter |

## Measuring curvature on any model

```python
from hessian_tools import hessian_mode, top_eigenvalue, hessian_trace

params = [p for p in model.parameters() if p.requires_grad]
batches = [...]                                  # fixed training subset, equal-size batches
loss_fn = lambda b: model(**b).loss              # mean loss on one batch

with hessian_mode(model):                        # dropout off; BN uses batch statistics
    lmax, v = top_eigenvalue(params, loss_fn, batches, iters=20)
    trace, stderr = hessian_trace(params, loss_fn, batches, samples=50)
```

- **Cost.** Each power iteration does one Hessian-vector product per batch, about 3× a training step. `lmax_between(...)` repeats `top_eigenvalue` at 8 points.
- **Transformers.** Load Hugging Face models with `attn_implementation="eager"`. Hessian-vector products need double backward, which the fused attention kernels don't support.
- **Negative eigenvalues.** Power iteration finds the eigenvalue of largest *magnitude*. If that one is negative, `top_eigenvalue` runs a second, shifted pass, so it always returns the largest *algebraic* eigenvalue.
- **`hessian_mode`.** Kaur et al. turn dropout off when computing λmax. With BatchNorm, the training loss uses batch statistics, so BN stays in train mode and its running statistics are restored afterwards.

## Reproducing the paper

Each run appends to `results.jsonl`. The values are the paper's ranges; use several `--seed` values (the paper uses 3–5).

```bash
# 3.1  Learning rate, small vs. large batch: VGG-11 without BN on CIFAR-10 (Fig. 1, 3)
for bs in 100 5000; do for lr in 0.0001 0.0003 0.001 0.003 0.01 0.03 0.1; do
  python train_vision.py --model vgg11 --batch_size $bs --lr $lr; done; done
python plot_results.py results.jsonl --x lr --group batch_size --where model=vgg11 --where rho=0.0 --out fig1.png

# 3.2  Linear scaling rule: fixed batch/lr ratio = 20000 (Fig. 4, 5)
for pair in "100 0.005" "200 0.01" "500 0.025"; do set -- $pair
  python train_vision.py --model vgg11 --batch_size $1 --lr $2; done

# 3.3  SAM, small vs. large batch (Fig. 6, 7)
for bs in 100 5000; do for rho in 0 0.02 0.04 0.06 0.08 0.1; do
  python train_vision.py --model vgg11 --batch_size $bs --lr 0.005 --rho $rho; done; done
python plot_results.py results.jsonl --x rho --group batch_size --where lr=0.005 --out fig6.png

# 3.4  Dropout: 2-hidden-layer MLP on Fashion-MNIST (Fig. 8, 9)
for lr in 0.01 0.1; do for p in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7; do
  python train_vision.py --dataset fashion_mnist --model mlp --lr $lr --batch_size 100 --dropout $p; done; done
python plot_results.py results.jsonl --x dropout --group lr --where model=mlp --out fig8.png

# 3.5  BatchNorm vs. no BatchNorm (Fig. 10, 11)
for m in vgg11 vgg11_bn; do for lr in 0.001 0.003 0.01 0.03 0.1; do
  python train_vision.py --model $m --batch_size 100 --lr $lr --between_iterates; done; done
python plot_results.py results.jsonl --x lr --group model --where batch_size=100 --out fig10.png

# λmax vs. test accuracy over all runs (Fig. 2)
python plot_results.py results.jsonl --scatter --group batch_size --out fig2.png
```

## Notes

- **Protocol.** Plain SGD (no momentum, no weight decay, no lr schedule, no augmentation). Training stops at `--target_acc 0.99` or after `--max_epochs`. Runs that hit the cap are marked `reached_target: false`, and runs whose loss becomes non-finite are marked `diverged: true`.
- **Choices the paper does not specify.**
  - λmax uses a fixed subset of 5000 training examples (`--lmax_examples`); the paper only says "a subset".
  - The MLP width is 512 (`--hidden`).
  - The large batch is 5000 (1000 for Fixup ResNet in the paper; Fixup ResNet is not included here).
- **Large batches** are accumulated over chunks of `--chunk` examples, so full-batch GD (`--batch_size 50000`) fits in memory. For BN networks each chunk uses its own batch statistics. On a small GPU, lower `--chunk` (for example 250).
- **SAM** here is plain SAM with SGD (ρ·g/‖g‖ ascent, then descend with the gradient there), as in the paper. It is written inline so the gradient can be accumulated over chunks. The general SAM family is in `../SAM`.
- Everything is fp32 and single-device.
