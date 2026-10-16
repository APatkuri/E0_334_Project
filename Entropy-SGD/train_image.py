"""Reproduce the image experiments of Chaudhari et al. (2017), "Entropy-SGD" (Table 1, Figs. 4-5).

    Sec. 5.2  mnistfc and LeNet on MNIST:   Adam baseline vs. Entropy-SGD
    Sec. 5.3  All-CNN-BN on CIFAR-10:       SGD baseline vs. Entropy-SGD
              (global contrast normalization + ZCA whitening, no augmentation)

Every setting the paper states is a default below (PAPER, with its section); pass only --model,
--optim and --seed. One JSON file per run with the test error after every epoch. For Entropy-SGD
an epoch is one pass of outer updates, each using L mini-batches, so it reads the data L times:
"effective epochs" = epochs x L (Sec. 5.2, remark on complexity).

    python train_image.py --model mnistfc --optim adam --out a.json
    python train_image.py --model lenet --optim entropysgd --out b.json
    python train_image.py --model allcnn --optim sgd --out c.json
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

from entropy_optim import EntropySGD

# Paper settings per (model, optimizer). "ours" marks what the paper leaves open.
#   milestones: epochs at which the lr is multiplied by lr_factor
PAPER = {
    # Sec. 5.2: Adam, lr 1e-3 dropping by 5x every 30 epochs, 100 epochs, dropout 0.5
    ("mnistfc", "adam"): dict(epochs=100, lr=1e-3, lr_factor=0.2, milestones=[30, 60, 90], dropout=0.5),
    ("lenet", "adam"): dict(epochs=100, lr=1e-3, lr_factor=0.2, milestones=[30, 60, 90], dropout=0.5),
    # Sec. 5.2: 5 epochs, L = 20, dropout 0.15 / 0.25, eta' = 0.1, eta = 1 dropping 10x after the
    # second epoch, Nesterov momentum in both loops, noise 1e-3, gamma0 = 1e-4 growing 1.001x per update
    ("mnistfc", "entropysgd"): dict(epochs=5, lr=1.0, lr_factor=0.1, milestones=[2], dropout=0.15, L=20,
                                    sgld_lr=0.1, noise=1e-3, gamma=1e-4, gamma_growth=1e-3),
    ("lenet", "entropysgd"): dict(epochs=5, lr=1.0, lr_factor=0.1, milestones=[2], dropout=0.25, L=20,
                                  sgld_lr=0.1, noise=1e-3, gamma=1e-4, gamma_growth=1e-3),
    # Sec. 5.3: SGD + Nesterov, 200 epochs, lr 0.1 dropping 5x every 60 epochs; weight decay 1e-3
    # and dropout from All-CNN-C (Springenberg et al. 2014), "all other hyper-parameters kept the same"
    ("allcnn", "sgd"): dict(epochs=200, lr=0.1, lr_factor=0.2, milestones=[60, 120, 180], dropout=0.5,
                            weight_decay=1e-3),
    # Sec. 5.3: L = 20, 10 epochs, dropout 0.5, eta = 1 dropping 5x every 4 epochs, eta' = 0.1,
    # noise 1e-4, gamma0 = 0.03 growing 1.001x per update
    ("allcnn", "entropysgd"): dict(epochs=10, lr=1.0, lr_factor=0.2, milestones=[4, 8], dropout=0.5, L=20,
                                   sgld_lr=0.1, noise=1e-4, gamma=0.03, gamma_growth=1e-3, weight_decay=1e-3),
}
MOMENTUM = 0.9  # Sec. 5.4.2: "all our experiments on CNNs used a momentum of 0.9" (Nesterov, Sec. 5.2-5.3)
ALPHA = 0.75    # Sec. 4.2: exponential averaging of mu


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mnistfc", choices=["mnistfc", "lenet", "allcnn"])
    p.add_argument("--optim", default="adam", choices=["adam", "sgd", "entropysgd"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=128, help="not stated in the paper (ours)")
    # None = paper value from PAPER
    for k in ("epochs", "lr", "lr_factor", "dropout", "L", "sgld_lr", "noise", "gamma", "gamma_growth",
              "weight_decay"):
        p.add_argument(f"--{k}", type=int if k in ("epochs", "L") else float, default=None)
    p.add_argument("--limit", type=int, default=0, help="updates per epoch (0 = full epoch; for smoke tests)")
    p.add_argument("--data_dir", default=os.path.join(os.environ.get("DATA_ROOT", "."), "vision"))
    p.add_argument("--out", default="result.json")
    args = p.parse_args()
    paper = PAPER[(args.model, args.optim)]
    for k, v in paper.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)
    for k in ("L", "sgld_lr", "noise", "gamma", "gamma_growth"):
        if getattr(args, k) is None:
            setattr(args, k, 0)
    args.weight_decay = args.weight_decay or 0.0  # MNIST: none stated (ours)
    return args


# ---- data ----------------------------------------------------------------------

def gcn_zca(train, test, scale=55.0, filter_bias=0.1):
    """Global contrast normalization and ZCA whitening as in Goodfellow et al. (2013)'s pylearn2
    pipeline (scale 55, ZCA filter bias 0.1), fitted on the training set."""
    def gcn(x):
        x = x.flatten(1).double()
        x = x - x.mean(1, keepdim=True)
        return scale * x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)
    tr, te = gcn(train), gcn(test)
    mean = tr.mean(0)
    tr = tr - mean
    e, V = torch.linalg.eigh(tr.T @ tr / len(tr))
    P = (V * (e + filter_bias).rsqrt()) @ V.T
    return (tr @ P).float().view_as(train), ((te - mean) @ P).float().view_as(test)


def load_data(model, root, device):
    if model == "allcnn":
        tr, te = datasets.CIFAR10(root, train=True, download=True), datasets.CIFAR10(root, train=False, download=True)
        to_x = lambda d: torch.tensor(d.data, device=device).permute(0, 3, 1, 2).float()
        x_tr, x_te = gcn_zca(to_x(tr), to_x(te))
    else:  # MNIST, standardized (ours; the paper does not say)
        tr, te = datasets.MNIST(root, train=True, download=True), datasets.MNIST(root, train=False, download=True)
        to_x = lambda d: ((d.data.float() / 255 - 0.1307) / 0.3081).unsqueeze(1).to(device)
        x_tr, x_te = to_x(tr), to_x(te)
    y = lambda d: torch.as_tensor(d.targets, device=device)
    return x_tr, y(tr), x_te, y(te)


# ---- models (Sec. 5.2-5.3) -----------------------------------------------------

def build_model(name, p):
    if name == "mnistfc":  # two hidden layers of 1024, dropout after each layer
        return nn.Sequential(nn.Flatten(), nn.Linear(784, 1024), nn.ReLU(), nn.Dropout(p),
                             nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(p), nn.Linear(1024, 10))
    if name == "lenet":  # LeNet sizes (20-50-500, Caffe), with batch normalization and dropout
        return nn.Sequential(
            nn.Conv2d(1, 20, 5), nn.BatchNorm2d(20), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(p),
            nn.Conv2d(20, 50, 5), nn.BatchNorm2d(50), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(p),
            nn.Flatten(), nn.Linear(800, 500), nn.BatchNorm1d(500), nn.ReLU(), nn.Dropout(p), nn.Linear(500, 10))
    # All-CNN-C (Springenberg et al. 2014, Table 2) with BN after every convolution;
    # dropout 20% on the input and p after each stride-2 layer, as in the original
    def conv(cin, cout, k, stride=1, pad=1):
        return [nn.Conv2d(cin, cout, k, stride, pad), nn.BatchNorm2d(cout), nn.ReLU()]
    return nn.Sequential(
        nn.Dropout(0.2),
        *conv(3, 96, 3), *conv(96, 96, 3), *conv(96, 96, 3, 2), nn.Dropout(p),
        *conv(96, 192, 3), *conv(192, 192, 3), *conv(192, 192, 3, 2), nn.Dropout(p),
        *conv(192, 192, 3, pad=0), *conv(192, 192, 1, pad=0),
        nn.Conv2d(192, 10, 1), nn.BatchNorm2d(10), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten())


def build_optimizer(args, model):
    params = model.parameters()
    if args.optim == "adam":
        return torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    if args.optim == "sgd":
        return torch.optim.SGD(params, lr=args.lr, momentum=MOMENTUM, nesterov=True, weight_decay=args.weight_decay)
    # Entropy-SGD: Nesterov momentum in both loops (Sec. 5.2). Weight decay is part of f, so it goes
    # in the SGLD loop (ours). Outer gradient x - mu, not gamma (x - mu) (Sec. 4.3).
    return EntropySGD(params, torch.optim.SGD, sgld_steps=args.L, gamma=args.gamma, gamma_growth=args.gamma_growth,
                      sgld_lr=args.sgld_lr, noise=args.noise, alpha=ALPHA,
                      sgld_kwargs=dict(momentum=MOMENTUM, nesterov=True, weight_decay=args.weight_decay),
                      lr=args.lr, momentum=MOMENTUM, nesterov=True)


# ---- training ------------------------------------------------------------------

def batches(n, bs, device):
    """Endless stream of shuffled index batches."""
    while True:
        perm = torch.randperm(n, device=device)
        for i in range(0, n - bs + 1, bs):
            yield perm[i:i + bs]


@torch.no_grad()
def error(model, x, y, bs=1000):
    model.eval()
    wrong = sum((model(x[i:i + bs]).argmax(-1) != y[i:i + bs]).sum().item() for i in range(0, len(x), bs))
    model.train()
    return wrong / len(x)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_tr, y_tr, x_te, y_te = load_data(args.model, args.data_dir, device)
    model = build_model(args.model, args.dropout).to(device)
    opt = build_optimizer(args, model)
    stream = batches(len(x_tr), args.batch_size, device)
    per_epoch = args.limit or len(x_tr) // args.batch_size
    L = max(args.L, 1)

    def closure():
        idx = next(stream)
        return F.cross_entropy(model(x_tr[idx]), y_tr[idx])

    curve, start, diverged = [], time.time(), False
    for epoch in range(args.epochs):
        lr = args.lr * args.lr_factor ** sum(epoch >= m for m in args.milestones)
        for g in opt.param_groups:
            g["lr"] = lr
        total = 0.0
        for _ in range(per_epoch):
            if args.optim == "entropysgd":
                loss = opt.step(closure)
            else:
                opt.zero_grad()
                loss = closure()
                loss.backward()
                opt.step()
            total += loss.item()
        if not math.isfinite(total):
            diverged = True
            break
        row = dict(epoch=epoch + 1, effective_epochs=(epoch + 1) * L, lr=lr, train_loss=total / per_epoch,
                   test_err=error(model, x_te, y_te))
        if args.optim == "entropysgd":
            row.update(gamma=opt.gamma, grad_ratio=opt.grad_ratio)
        curve.append(row)
        print(json.dumps(row), flush=True)

    result = dict(vars(args), curve=curve, diverged=diverged, seconds=round(time.time() - start, 1))
    if curve:
        best = min(curve, key=lambda r: r["test_err"])
        result.update(test_err=curve[-1]["test_err"], best_test_err=best["test_err"],
                      best_effective_epochs=best["effective_epochs"])
    with open(args.out, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
