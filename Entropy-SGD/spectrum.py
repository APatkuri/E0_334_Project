"""Reproduce Sec. 5.1 of Chaudhari et al. (2017), "Universality of the Hessian at local minima"
(Figs. 1 and 3): the eigenspectrum of the exact Hessian at the end of training, for networks small
enough to form it, and the diagonal Fisher information for All-CNN-BN.

    small_lenet     LeNet-like, 10 and 20 conv channels, 128 hidden units (47,658 parameters), Adam
    small_mnistfc   one hidden layer (50,890 parameters), momentum SGD
    small_charlstm  LSTM with 48 hidden units (32,640 parameters), Adam, memorizing 256 lines of
                    32 one-hot characters (96 symbols)
    allcnn_fisher   All-CNN-BN on CIFAR-10 trained with Adam; diag(I) = E[g^2] - (E g)^2 from Adam's
                    moment estimates

Each network is trained until it reaches the error the paper reports (test error 2.4%, 2.5%,
11.2%; the char-LSTM until it reproduces the text), then the eigenvalues are saved to
<out>.npy and a summary to <out>.

Not in the paper (ours): the training hyperparameters, and the Hessian is computed on a fixed
subset of --hessian_examples training images (MNIST). small_mnistfc: the paper says 32 hidden units,
but 50,890 parameters is one hidden layer of 64 units (784*64 + 64 + 64*10 + 10), which we use.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad, vjp, vmap
from torchvision import datasets

import train_image

TARGET_ERR = {"small_lenet": 0.024, "small_mnistfc": 0.025, "allcnn_fisher": 0.112}  # Sec. 5.1 (i), (ii), (iv)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="small_mnistfc",
                   choices=["small_lenet", "small_mnistfc", "small_charlstm", "allcnn_fisher"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_epochs", type=int, default=100, help="cap if the target error is not reached")
    p.add_argument("--hessian_examples", type=int, default=10000)
    p.add_argument("--chunk", type=int, default=128, help="Hessian rows per vmap call")
    p.add_argument("--vision_dir", default=os.path.join(os.environ.get("DATA_ROOT", "."), "vision"))
    p.add_argument("--text_dir", default=os.path.join(os.environ.get("DATA_ROOT", "."), "text"))
    p.add_argument("--out", default="spectrum.json")
    return p.parse_args()


# ---- small networks ------------------------------------------------------------

def small_lenet():
    return nn.Sequential(nn.Conv2d(1, 10, 5), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(10, 20, 5), nn.ReLU(),
                         nn.MaxPool2d(2), nn.Flatten(), nn.Linear(320, 128), nn.ReLU(), nn.Linear(128, 10))


def small_mnistfc():
    return nn.Sequential(nn.Flatten(), nn.Linear(784, 64), nn.ReLU(), nn.Linear(64, 10))


class SmallCharLSTM(nn.Module):
    """One-layer LSTM written out step by step so that double backward works (cuDNN's does not).
    Same parameters as nn.LSTM(96, 48) plus a bias-free 48 -> 96 readout: 32,640 in total."""

    def __init__(self, vocab=96, hidden=48):
        super().__init__()
        self.vocab, self.hidden = vocab, hidden
        self.weight_ih = nn.Parameter(torch.empty(4 * hidden, vocab))
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden, hidden))
        self.bias_ih = nn.Parameter(torch.empty(4 * hidden))
        self.bias_hh = nn.Parameter(torch.empty(4 * hidden))
        self.out = nn.Linear(hidden, vocab, bias=False)
        for p in self.parameters():
            nn.init.uniform_(p, -1 / math.sqrt(hidden), 1 / math.sqrt(hidden))

    def forward(self, x):  # x: [batch, time] character ids -> logits [batch, time, vocab]
        h = c = x.new_zeros(x.size(0), self.hidden, dtype=torch.float)
        xs = F.one_hot(x, self.vocab).float() @ self.weight_ih.T + self.bias_ih
        outs = []
        for t in range(x.size(1)):
            i, f, g, o = (xs[:, t] + h @ self.weight_hh.T + self.bias_hh).chunk(4, 1)
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
            h = torch.sigmoid(o) * torch.tanh(c)
            outs.append(h)
        return self.out(torch.stack(outs, 1))


def char_lines(text_dir, lines=256, length=32):
    """The first 256 x 32 characters of War and Peace over printable ASCII plus newline (96 symbols)."""
    symbols = "\n" + "".join(chr(c) for c in range(32, 127))
    text = [c for c in open(os.path.join(text_dir, "warpeace_input.txt")).read() if c in symbols]
    ids = torch.tensor([symbols.index(c) for c in text[:lines * length]])
    return ids.view(lines, length)


# ---- exact Hessian -------------------------------------------------------------

def full_hessian(model, loss_fn, batches, chunk):
    """Exact Hessian of the mean loss over `batches`, built row by row from Hessian-vector products."""
    model.eval()
    names, shapes = zip(*[(n, p.shape) for n, p in model.named_parameters()])
    theta = torch.cat([p.detach().flatten() for p in model.parameters()])
    buffers = dict(model.named_buffers())
    n = theta.numel()

    def unflat(t):
        out, i = {}, 0
        for name, s in zip(names, shapes):
            k = math.prod(s)
            out[name] = t[i:i + k].view(s)
            i += k
        return out

    H = torch.zeros(n, n, device=theta.device)
    for x, y in batches:
        f = lambda t: loss_fn(functional_call(model, {**unflat(t), **buffers}, (x,)), y)
        _, row = vjp(grad(f), theta)  # row(v) = H v = v^T H
        for i in range(0, n, chunk):
            m = min(chunk, n - i)
            e = torch.zeros(m, n, device=theta.device)
            e[torch.arange(m), i + torch.arange(m)] = 1
            H[i:i + m] += vmap(row)(e)[0] / len(batches)
    return H


def eigenvalues(H):
    """Eigenvalues of the symmetric H (lower triangle), on the GPU if it fits."""
    try:
        return torch.linalg.eigvalsh(H).cpu().numpy()
    except torch.cuda.OutOfMemoryError:
        return torch.linalg.eigvalsh(H.cpu()).numpy()


# ---- training ------------------------------------------------------------------

@torch.no_grad()
def test_error(model, x, y):
    return train_image.error(model, x, y)


def train_vision(args, device):
    if args.model == "allcnn_fisher":
        x_tr, y_tr, x_te, y_te = train_image.load_data("allcnn", args.vision_dir, device)
        model = train_image.build_model("allcnn", 0.5).to(device)
    else:
        x_tr, y_tr, x_te, y_te = train_image.load_data("mnistfc", args.vision_dir, device)
        model = (small_lenet() if args.model == "small_lenet" else small_mnistfc()).to(device)
    # Sec. 5.1: Adam for small-LeNet and All-CNN-BN, momentum SGD for small-mnistfc (rates ours)
    opt = (torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9) if args.model == "small_mnistfc"
           else torch.optim.Adam(model.parameters(), lr=1e-3))
    err, epochs = 1.0, 0
    while err > TARGET_ERR[args.model] and epochs < args.max_epochs:
        epochs += 1
        perm = torch.randperm(len(x_tr), device=device)
        for i in range(0, len(x_tr), 128):
            idx = perm[i:i + 128]
            opt.zero_grad()
            F.cross_entropy(model(x_tr[idx]), y_tr[idx]).backward()
            opt.step()
        err = test_error(model, x_te, y_te)
        print(f"epoch {epochs} test error {err:.4f}", flush=True)
    return model, opt, x_tr, y_tr, dict(epochs=epochs, test_err=err)


def train_char(args, device):
    data = char_lines(args.text_dir).to(device)
    x, y = data[:, :-1], data[:, 1:]
    model = SmallCharLSTM().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for step in range(1, 20001):  # full batch, until every character is reproduced (ours)
        opt.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        loss.backward()
        opt.step()
        acc = (logits.argmax(-1) == y).float().mean().item()
        if step % 500 == 0:
            print(f"step {step} loss {loss.item():.4f} char accuracy {acc:.4f}", flush=True)
        if acc == 1.0:
            break
    return model, x, y, dict(steps=step, train_loss=loss.item(), train_char_acc=acc)


def summary(eig):
    out = dict(n=int(eig.size), min=float(eig.min()), max=float(eig.max()), negative=int((eig < 0).sum()))
    for thr in (1e-5, 1e-4, 1e-3, 1e-2):
        out[f"frac_abs_le_{thr:g}"] = float((np.abs(eig) <= thr).mean())
    return out


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start = time.time()

    if args.model == "small_charlstm":
        model, x, y, info = train_char(args, device)
        lm_loss = lambda logits, t: F.cross_entropy(logits.flatten(0, 1), t.flatten())
        eig = eigenvalues(full_hessian(model, lm_loss, [(x, y)], args.chunk))
    elif args.model == "allcnn_fisher":
        model, opt, *_, info = train_vision(args, device)
        b1, b2 = opt.defaults["betas"]
        diag = []
        for p in model.parameters():
            s = opt.state[p]
            t = float(s["step"])
            m, v = s["exp_avg"] / (1 - b1 ** t), s["exp_avg_sq"] / (1 - b2 ** t)
            diag.append((v - m * m).flatten())
        eig = torch.cat(diag).cpu().numpy()
    else:
        model, _, x_tr, y_tr, info = train_vision(args, device)
        sub = torch.randperm(len(x_tr), generator=torch.Generator().manual_seed(1234))[:args.hessian_examples]
        sub = sub.to(device)
        batches = [(x_tr[sub[i:i + 1000]], y_tr[sub[i:i + 1000]]) for i in range(0, len(sub), 1000)]
        eig = eigenvalues(full_hessian(model, F.cross_entropy, batches, args.chunk))

    np.save(os.path.splitext(args.out)[0] + ".npy", eig)
    result = dict(vars(args), **info, params=sum(p.numel() for p in model.parameters()), **summary(eig),
                  seconds=round(time.time() - start, 1))
    print(json.dumps(result))
    with open(args.out, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
