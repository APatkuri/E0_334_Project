"""Reproduce the experiments of Kaur et al. (2023), "On the Maximum Hessian Eigenvalue and
Generalization".

Protocol (Sec. 3): minibatch SGD (or SAM) at a constant learning rate, no schedule, no momentum,
trained until 99% training accuracy. Then test accuracy and lambda_max of the training loss are
recorded, with lambda_max computed on a fixed subset of the training set and dropout off.

Each run appends one JSON line to --out, so a sweep is a loop over this script (see README.md).

Examples:
    python train_vision.py --model vgg11 --lr 0.01 --batch_size 100                  # Sec. 3.1
    python train_vision.py --model vgg11 --lr 0.01 --batch_size 5000                 # large batch
    python train_vision.py --model vgg11 --lr 0.005 --batch_size 100 --rho 0.05      # Sec. 3.3, SAM
    python train_vision.py --dataset fashion_mnist --model mlp --dropout 0.3 --lr 0.1   # Sec. 3.4
    python train_vision.py --model vgg11_bn --lr 0.001 --between_iterates            # Sec. 3.5
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import tracking
from hessian_tools import hessian_mode, hessian_trace, lmax_between, top_eigenvalue

VGG11 = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cifar10", choices=["cifar10", "fashion_mnist"])
    p.add_argument("--model", default="vgg11", choices=["vgg11", "vgg11_bn", "mlp"])
    p.add_argument("--hidden", type=int, default=512, help="MLP width")
    p.add_argument("--dropout", type=float, default=0.0, help="MLP dropout probability p")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--batch_size", type=int, default=100, help="use 50000 (60000) for full-batch GD")
    p.add_argument("--rho", type=float, default=0.0, help="SAM radius; 0 means plain SGD")
    p.add_argument("--target_acc", type=float, default=0.99)
    p.add_argument("--max_epochs", type=int, default=1000)
    p.add_argument("--chunk", type=int, default=1000,
                   help="examples per forward pass; larger batches are accumulated over chunks")
    p.add_argument("--seed", type=int, default=0)
    # lambda_max measurement
    p.add_argument("--lmax_examples", type=int, default=5000, help="training subset for the Hessian")
    p.add_argument("--lmax_iters", type=int, default=20)
    p.add_argument("--trace_samples", type=int, default=0, help="Hutchinson samples (0 = skip)")
    p.add_argument("--between_iterates", action="store_true",
                   help="max lambda_max over 8 points between the final iterate and one extra step (BN)")
    p.add_argument("--lmax_every", type=int, default=0,
                   help="also record lambda_max every this many steps during training (Fig. 5; 0 = off)")
    p.add_argument("--data_dir", default=os.path.join(os.environ.get("DATA_ROOT", "."), "vision"),
                   help="default: $DATA_ROOT/vision (set by setup.sh)")
    p.add_argument("--out", default="results.jsonl")
    return p.parse_args()


# ---- data and models -----------------------------------------------------------

def load_data(name, root):
    """Whole dataset as tensors (normalized, no augmentation)."""
    if name == "cifar10":
        cls, norm = datasets.CIFAR10, transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    else:
        cls, norm = datasets.FashionMNIST, transforms.Normalize((0.2860,), (0.3530,))
    tf = transforms.Compose([transforms.ToTensor(), norm])

    def to_tensors(train):
        ds = cls(root, train=train, download=True, transform=tf)
        loader = torch.utils.data.DataLoader(ds, batch_size=len(ds))
        return next(iter(loader))

    return (*to_tensors(True), *to_tensors(False))


def build_model(args, in_shape, num_classes=10):
    if args.model == "mlp":
        d = math.prod(in_shape)
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(d, args.hidden), nn.ReLU(), nn.Dropout(args.dropout),
            nn.Linear(args.hidden, args.hidden), nn.ReLU(), nn.Dropout(args.dropout),
            nn.Linear(args.hidden, num_classes))
    layers, c = [], in_shape[0]
    for v in VGG11:
        if v == "M":
            layers.append(nn.MaxPool2d(2))
            continue
        layers.append(nn.Conv2d(c, v, 3, padding=1))
        if args.model == "vgg11_bn":
            layers.append(nn.BatchNorm2d(v))
        layers.append(nn.ReLU())
        c = v
    return nn.Sequential(*layers, nn.Flatten(), nn.Linear(512, num_classes))


# ---- training ------------------------------------------------------------------

def gradient(model, params, x, y, chunk, device):
    """Mean cross-entropy gradient over (x, y), accumulated over chunks of `chunk` examples.

    For BN networks each chunk uses its own batch statistics ("ghost" batch norm)."""
    grads = [torch.zeros_like(p) for p in params]
    total = 0.0
    for i in range(0, len(x), chunk):
        xb, yb = x[i:i + chunk].to(device), y[i:i + chunk].to(device)
        loss = F.cross_entropy(model(xb), yb, reduction="sum")
        for g, gi in zip(grads, torch.autograd.grad(loss, params)):
            g.add_(gi)
        total += loss.item()
    return [g / len(x) for g in grads], total / len(x)


def train_step(model, params, x, y, args, device):
    """One SGD step, or one SAM step (Foret et al.) when rho > 0. Returns the loss at w."""
    g, loss = gradient(model, params, x, y, args.chunk, device)
    if args.rho > 0:
        scale = args.rho / (torch.sqrt(sum((gi * gi).sum() for gi in g)) + 1e-12)
        e = [gi * scale for gi in g]
        with torch.no_grad():
            for p, ei in zip(params, e):
                p.add_(ei)
        g, _ = gradient(model, params, x, y, args.chunk, device)
        with torch.no_grad():
            for p, ei in zip(params, e):
                p.sub_(ei)
    with torch.no_grad():
        for p, gi in zip(params, g):
            p.sub_(gi, alpha=args.lr)
    return loss


@torch.no_grad()
def accuracy(model, x, y, chunk, device):
    was_training = model.training
    model.eval()
    correct = sum((model(x[i:i + chunk].to(device)).argmax(-1) == y[i:i + chunk].to(device)).sum().item()
                  for i in range(0, len(x), chunk))
    model.train(was_training)
    return correct / len(x)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    tracking.init(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train, y_train, x_test, y_test = load_data(args.dataset, args.data_dir)
    n = len(x_train)

    model = build_model(args, x_train.shape[1:]).to(device)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]

    # lambda_max on a fixed training subset (same subset for every run)
    sub = torch.randperm(n, generator=torch.Generator().manual_seed(1234))[:args.lmax_examples]
    batches = [(x_train[sub[i:i + args.chunk]].to(device), y_train[sub[i:i + args.chunk]].to(device))
               for i in range(0, len(sub), args.chunk)]
    loss_fn = lambda b: F.cross_entropy(model(b[0]), b[1])

    start_time = time.time()
    steps, epoch, train_acc, diverged, lmax_curve = 0, 0, 0.0, False, []
    while epoch < args.max_epochs:
        epoch += 1
        perm = torch.randperm(n)
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            loss = train_step(model, params, x_train[idx], y_train[idx], args, device)
            steps += 1
            if not math.isfinite(loss):
                diverged = True
                break
            if args.lmax_every and steps % args.lmax_every == 0:
                with hessian_mode(model):
                    lmax_curve.append((steps, top_eigenvalue(params, loss_fn, batches, iters=args.lmax_iters)[0]))
                tracking.log(dict(steps=steps, lmax=lmax_curve[-1][1]))
        if diverged:
            break
        train_acc = accuracy(model, x_train, y_train, args.chunk, device)
        print(f"epoch {epoch} steps {steps} loss {loss:.4f} train acc {train_acc:.4f}", flush=True)
        tracking.log(dict(epoch=epoch, steps=steps, loss=loss, train_acc=train_acc))
        if train_acc >= args.target_acc:
            break

    result = dict(vars(args), epochs=epoch, steps=steps, train_acc=train_acc, diverged=diverged,
                  reached_target=train_acc >= args.target_acc, two_over_lr=2.0 / args.lr)
    if lmax_curve:
        result["lmax_curve"] = lmax_curve
    if not diverged:
        result["test_acc"] = accuracy(model, x_test, y_test, args.chunk, device)
        if args.between_iterates:  # one extra ordinary training step (dropout on)
            start = [p.detach().clone() for p in params]
            idx = torch.randperm(n)[:args.batch_size]
            train_step(model, params, x_train[idx], y_train[idx], args, device)
            end = [p.detach().clone() for p in params]
        with hessian_mode(model):
            if args.between_iterates:
                result["lmax"] = lmax_between(params, loss_fn, batches, start, end, iters=args.lmax_iters)
            else:
                result["lmax"] = top_eigenvalue(params, loss_fn, batches, iters=args.lmax_iters)[0]
            if args.trace_samples > 0:
                result["trace"], result["trace_se"] = hessian_trace(
                    params, loss_fn, batches, samples=args.trace_samples)
    result["seconds"] = round(time.time() - start_time, 1)

    print(json.dumps({k: result[k] for k in ("train_acc", "test_acc", "lmax", "two_over_lr") if k in result}))
    with open(args.out, "a") as f:
        f.write(json.dumps(result) + "\n")
    tracking.finish(result)


if __name__ == "__main__":
    main()
