"""Image classification with SGD and the SAM variants, for the tables of the SAM-variant papers:

    X-SAM Table 1       SAM, WSAM, GSAM, Eigen-SAM, X-SAM; ResNet-18/50, WRN-28-10, AlexNet;
                        CIFAR-10/100, Fashion-MNIST (Sec. 6.1)
    Eigen-SAM Table 1   SGD, SAM, Eigen-SAM; ResNet-18/50, WRN-28-10; + SVHN (Sec. 6.2, App. E)
    Eigen-SAM Table 2   fine-tuning ImageNet ViT-B/16 on CIFAR-10/100 (Sec. 6.3)
    WSAM Tables 2, 6, 9 SgdM, SAM, GSAM, WSAM; ResNet-18, WRN-28-10; label noise
                        (Sec. 6.1, 6.3, 6.6)

The protocol of each table (and its source) is in ../sweep.py; this script takes it as flags.
SGD with momentum 0.9 and a per-step cosine schedule, crop (4-pixel zero padding) + flip
augmentation, done on the GPU. One JSON file per run: test accuracy after every epoch, then the
final test accuracy, the training loss and lambda_max on clean training data (WSAM Table 9).

    python train_image.py --dataset cifar10 --model resnet18 --optim sam --rho 0.05 --out a.json
"""
import argparse
import contextlib
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "Maximum-Hessian"))
sys.path.append(ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

import tracking
from hessian_tools import hessian_mode, top_eigenvalue
from image_models import MODELS, build_model
from sam_optim import GSAM, SAM, WSAM, EigenSAM, XSAM

STATS = {  # per-channel mean / std of the training sets (ours: the papers do not say)
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    "fashion_mnist": ((0.2860,), (0.3530,)),
    "svhn": ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
}
IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ViT-B/16 checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cifar10", choices=list(STATS))
    p.add_argument("--model", default="resnet18", choices=MODELS)
    p.add_argument("--optim", default="sgd", choices=["sgd", "sam", "gsam", "wsam", "eigensam", "xsam"])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--steps", type=int, default=0, help="train for this many steps instead (ViT)")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=5e-5)
    p.add_argument("--rho", type=float, default=0.05)
    p.add_argument("--alpha", type=float, default=0.2, help="GSAM / Eigen-SAM / X-SAM strength")
    p.add_argument("--gamma", type=float, default=0.9, help="WSAM sharpness weight")
    p.add_argument("--eig_every", type=int, default=100, help="Eigen-SAM / X-SAM: p")
    p.add_argument("--power_iters", type=int, default=5, help="Eigen-SAM / X-SAM: q")
    p.add_argument("--label_noise", type=float, default=0.0, help="symmetric noise rate (WSAM Sec. 6.3)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval_every", type=int, default=0, help="steps between test evaluations (0 = each epoch)")
    p.add_argument("--lmax_examples", type=int, default=5000, help="0 = skip the final lambda_max")
    p.add_argument("--lmax_every", type=int, default=0,
                   help="also track lambda_max every this many epochs (X-SAM Fig. 2; 0 = off)")
    p.add_argument("--lmax_track_examples", type=int, default=1000)
    p.add_argument("--limit", type=int, default=0, help="steps per epoch (0 = full epoch; for smoke tests)")
    p.add_argument("--data_dir", default=os.path.join(os.environ.get("DATA_ROOT", "."), "vision"))
    p.add_argument("--out", default="result.json")
    return p.parse_args()


# ---- data (all on the GPU) -----------------------------------------------------

def load_split(name, root, train):
    if name == "svhn":
        d = datasets.SVHN(root, split="train" if train else "test", download=True)
        return torch.from_numpy(d.data), torch.from_numpy(d.labels).long()
    cls = {"cifar10": datasets.CIFAR10, "cifar100": datasets.CIFAR100, "fashion_mnist": datasets.FashionMNIST}[name]
    d = cls(root, train=train, download=True)
    x = torch.as_tensor(d.data)
    x = x.unsqueeze(1) if x.dim() == 3 else x.permute(0, 3, 1, 2)
    return x, torch.as_tensor(d.targets).long()


def symmetric_noise(y, rate, num_classes, seed=0):
    """Replace a `rate` fraction of the labels by uniformly random classes (possibly the true one),
    as in Arazo et al. (2019). The corruption is fixed (seed 0) across methods and runs (ours)."""
    g = torch.Generator().manual_seed(seed)
    y = y.clone()
    idx = torch.randperm(len(y), generator=g)[:int(round(rate * len(y)))]
    y[idx] = torch.randint(num_classes, (len(idx),), generator=g)
    return y


def augment(x):
    """Random crop with 4-pixel zero padding and horizontal flip, per image (x: [B, C, H, W] in [0, 1])."""
    b, _, h, w = x.shape
    padded = F.pad(x, (4, 4, 4, 4))
    i = torch.randint(0, 9, (b,), device=x.device)
    j = torch.randint(0, 9, (b,), device=x.device)
    rows = (i[:, None] + torch.arange(h, device=x.device))[:, :, None]
    cols = (j[:, None] + torch.arange(w, device=x.device))[:, None, :]
    out = padded[torch.arange(b, device=x.device)[:, None, None], :, rows, cols].permute(0, 3, 1, 2)
    flip = torch.rand(b, device=x.device) < 0.5
    return torch.where(flip[:, None, None, None], out.flip(3), out)


class Preprocess:
    """uint8 images -> normalized float, with optional augmentation and resizing (ViT: 224)."""

    def __init__(self, mean, std, size, device):
        self.mean = torch.tensor(mean, device=device).view(1, -1, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, -1, 1, 1)
        self.size = size

    def __call__(self, x, train):
        x = x.float() / 255
        if train:
            x = augment(x)
        if self.size is not None:
            x = F.interpolate(x, size=self.size, mode="bilinear", align_corners=False)
            return (x - self.mean) / self.std
        return ((x - self.mean) / self.std).contiguous(memory_format=torch.channels_last)  # CNNs


# ---- training ------------------------------------------------------------------

def build_optimizer(args, model):
    kw = dict(lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    params, base = model.parameters(), torch.optim.SGD
    if args.optim == "sgd":
        return base(params, **kw)
    if args.optim == "sam":
        return SAM(params, base, rho=args.rho, **kw)
    if args.optim == "gsam":  # constant rho (X-SAM Sec. 6.1: "the same perturbation radii as SAM")
        return GSAM(params, base, rho=args.rho, alpha=args.alpha, **kw)
    if args.optim == "wsam":
        return WSAM(params, base, rho=args.rho, gamma=args.gamma, **kw)
    if args.optim == "eigensam":
        return EigenSAM(params, base, rho=args.rho, alpha=args.alpha, eig_every=args.eig_every,
                        power_iters=args.power_iters, **kw)
    return XSAM(params, base, rho=args.rho, alpha=args.alpha, eig_every=args.eig_every,
                power_iters=args.power_iters, **kw)


@contextlib.contextmanager
def frozen_bn_stats(model, frozen):
    """Extra forward passes of a SAM step (perturbed weights, Hessian products) use batch statistics
    but leave the BN running statistics alone, so they track only the actual weights."""
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)] if frozen else []
    saved = [m.momentum for m in bns]
    for m in bns:
        m.momentum = 0.0
    try:
        yield
    finally:
        for m, mom in zip(bns, saved):
            m.momentum = mom


@torch.no_grad()
def evaluate(model, x, y, prep, bs=500):
    model.eval()
    correct, loss = 0, 0.0
    for i in range(0, len(x), bs):
        out = model(prep(x[i:i + bs], train=False))
        correct += (out.argmax(-1) == y[i:i + bs]).sum().item()
        loss += F.cross_entropy(out, y[i:i + bs], reduction="sum").item()
    model.train()
    return correct / len(x), loss / len(x)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    tracking.init(args)
    torch.backends.cuda.matmul.allow_tf32 = True  # convolutions already use TF32 by default
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_tr, y_tr = load_split(args.dataset, args.data_dir, True)
    x_te, y_te = load_split(args.dataset, args.data_dir, False)
    num_classes = int(y_tr.max()) + 1
    y_clean = y_tr.clone()
    if args.label_noise > 0:
        y_tr = symmetric_noise(y_tr, args.label_noise, num_classes)
    x_tr, y_tr, y_clean, x_te, y_te = (t.to(device) for t in (x_tr, y_tr, y_clean, x_te, y_te))

    vit = args.model == "vit_b_16"
    prep = Preprocess(*(IMAGENET if vit else STATS[args.dataset]), 224 if vit else None, device)
    model = build_model(args.model, x_tr.size(1), num_classes).to(device)
    if not vit:
        model = model.to(memory_format=torch.channels_last)
    model.train()
    opt = build_optimizer(args, model)

    n = len(x_tr)
    per_epoch = args.limit or math.ceil(n / args.batch_size)
    total = args.steps or args.epochs * per_epoch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)
    # Hessian-vector products need double backward, which the fused attention kernels lack
    attention = contextlib.nullcontext
    if vit and args.optim in ("eigensam", "xsam"):
        from torch.nn.attention import SDPBackend, sdpa_kernel
        attention = lambda: sdpa_kernel(SDPBackend.MATH)

    # lambda_max during training: fixed clean training subset, 10 power iterations (ours)
    params = [p for p in model.parameters() if p.requires_grad]
    loss_fn = lambda b: F.cross_entropy(model(b[0]), b[1])
    track = torch.randperm(n, generator=torch.Generator().manual_seed(4321))[:args.lmax_track_examples].to(device)
    track_batches = [(prep(x_tr[track[i:i + 500]], train=False), y_clean[track[i:i + 500]])
                     for i in range(0, len(track), 500)] if args.lmax_every else []

    curve, step, start, diverged = [], 0, time.time(), False
    epoch, running, seen = 0, 0.0, 0
    while step < total and not diverged:
        epoch += 1
        perm = torch.randperm(n, device=device)
        for k in range(per_epoch):
            if step == total:
                break
            idx = perm[k * args.batch_size:(k + 1) * args.batch_size]
            x, y = prep(x_tr[idx], train=True), y_tr[idx]
            calls = 0

            def closure():
                nonlocal calls
                calls += 1
                with frozen_bn_stats(model, calls > 1):
                    return F.cross_entropy(model(x), y)

            with attention():
                if args.optim == "sgd":
                    opt.zero_grad(set_to_none=True)
                    loss = closure()
                    loss.backward()
                    opt.step()
                else:
                    loss = opt.step(closure)
            sched.step()
            step += 1
            loss = loss.item()
            if not math.isfinite(loss):
                diverged = True
                break
            running, seen = running + loss, seen + 1
            end_of_epoch = k == per_epoch - 1 or step == total
            if (args.eval_every and step % args.eval_every == 0) or (not args.eval_every and end_of_epoch) \
                    or step == total:
                acc, test_loss = evaluate(model, x_te, y_te, prep)
                row = dict(epoch=epoch, step=step, lr=sched.get_last_lr()[0], train_loss=running / seen, test_acc=acc,
                           test_loss=test_loss, minutes=round((time.time() - start) / 60, 2))
                if getattr(opt, "top_eigenvalue", None) is not None:
                    row["lmax_estimate"] = opt.top_eigenvalue  # Eigen-SAM / X-SAM's own power iteration
                if track_batches and end_of_epoch and (epoch % args.lmax_every == 0 or step == total):
                    with hessian_mode(model), attention():
                        row["lmax"] = top_eigenvalue(params, loss_fn, track_batches, iters=10)[0]
                curve.append(row)
                running, seen = 0.0, 0
                print(json.dumps(row), flush=True)
                tracking.log(row)

    result = dict(vars(args), num_classes=num_classes, steps_done=step, curve=curve, diverged=diverged)
    if curve and not diverged:
        result.update(test_acc=curve[-1]["test_acc"], best_test_acc=max(r["test_acc"] for r in curve))
        # minima analysis (WSAM Table 9): loss and lambda_max on clean, unaugmented training data
        result["train_acc"], result["train_loss"] = evaluate(model, x_tr, y_clean, prep)
        if args.lmax_examples:
            sub = torch.randperm(n, generator=torch.Generator().manual_seed(1234))[:args.lmax_examples].to(device)
            batches = [(prep(x_tr[sub[i:i + 500]], train=False), y_clean[sub[i:i + 500]])
                       for i in range(0, len(sub), 500)]
            with hessian_mode(model), attention():
                result["lmax"] = top_eigenvalue(params, loss_fn, batches, iters=20)[0]
    result["minutes"] = round((time.time() - start) / 60, 2)
    print(json.dumps({k: result.get(k) for k in ("test_acc", "best_test_acc", "train_loss", "lmax", "minutes")}))
    with open(args.out, "w") as f:
        json.dump(result, f)
    tracking.finish(result)


if __name__ == "__main__":
    main()
