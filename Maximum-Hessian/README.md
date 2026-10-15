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

Runs append to `results.jsonl`. Use 3–5 `--seed` values per setting.

```bash
# 3.1  Learning rate, small vs. large batch: VGG-11 on CIFAR-10 (Fig. 1, 3)
for bs in 100 5000; do for lr in 0.0001 0.0003 0.001 0.003 0.01 0.03 0.1; do
  python train_vision.py --model vgg11 --batch_size $bs --lr $lr; done; done
python plot_results.py results.jsonl --x lr --group batch_size --where model=vgg11 --where rho=0.0 --out fig1.png

# 3.2  Linear scaling rule: batch/lr = 20000 (Fig. 4, 5)
for pair in "100 0.005" "200 0.01" "500 0.025"; do set -- $pair
  python train_vision.py --model vgg11 --batch_size $1 --lr $2; done

# 3.3  SAM, small vs. large batch (Fig. 6, 7)
for bs in 100 5000; do for rho in 0 0.02 0.04 0.06 0.08 0.1; do
  python train_vision.py --model vgg11 --batch_size $bs --lr 0.005 --rho $rho; done; done
python plot_results.py results.jsonl --x rho --group batch_size --where lr=0.005 --out fig6.png

# 3.4  Dropout: MLP on Fashion-MNIST (Fig. 8, 9)
for lr in 0.01 0.1; do for p in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7; do
  python train_vision.py --dataset fashion_mnist --model mlp --lr $lr --batch_size 100 --dropout $p; done; done
python plot_results.py results.jsonl --x dropout --group lr --where model=mlp --out fig8.png

# 3.5  BatchNorm vs. none (Fig. 10, 11)
for m in vgg11 vgg11_bn; do for lr in 0.001 0.003 0.01 0.03 0.1; do
  python train_vision.py --model $m --batch_size 100 --lr $lr --between_iterates; done; done
python plot_results.py results.jsonl --x lr --group model --where batch_size=100 --out fig10.png

# λmax vs. test accuracy, all runs (Fig. 2)
python plot_results.py results.jsonl --scatter --group batch_size --out fig2.png
```

## Notes

- **Protocol:** plain SGD (no momentum, weight decay, schedule or augmentation). Stops at `--target_acc 0.99` or `--max_epochs`; results flag `reached_target: false` or `diverged: true`.
- **Our choices** (unspecified in the paper): λmax on 5000 training examples (`--lmax_examples`), MLP width 512 (`--hidden`), large batch 5000. Fixup ResNet not included.
- **Large batches** are accumulated in chunks of `--chunk` examples, so full-batch GD fits in memory. Lower it (e.g. 250) on small GPUs.
- **SAM** here is plain SAM + SGD, inlined to support chunking. The full family is in `../SAM`.
- fp32, single-device only.
