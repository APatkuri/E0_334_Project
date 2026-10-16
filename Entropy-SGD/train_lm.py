"""Reproduce the RNN experiments of Chaudhari et al. (2017), "Entropy-SGD" (Table 1, Fig. 6).

    Sec. 5.4.1  PTB-LSTM, word level (Zaremba et al. 2014, "large"):  SGD vs. Entropy-SGD
    Sec. 5.4.2  char-LSTM on War and Peace (Karpathy et al. 2015):     Adam vs. Entropy-Adam

Every setting the paper states is a default below (PAPER); pass only --task, --optim and --seed.
One JSON file per run with validation and test loss/perplexity after every epoch. As in
train_image.py, an Entropy-SGD epoch reads the data L times ("effective epochs" = epochs x L).

The paper's char-LSTM "perplexities" (about 1.22) are the size of a per-character cross-entropy in
nats, not exp() of it, so both are recorded (`loss` and `ppl`).

    python train_lm.py --task ptb --optim sgd --out a.json
    python train_lm.py --task char --optim entropyadam --out b.json
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

import tracking
from entropy_optim import EntropySGD

# Paper settings per (task, optimizer). "ours" marks what the paper leaves open.
PAPER = {
    # Sec. 5.4.1: Zaremba et al.'s pipeline: SGD without momentum, 2 x 1500 LSTM unrolled 35 steps,
    # 55 epochs. From Zaremba et al. (large model): batch 20, dropout 0.65, init U(-0.04, 0.04),
    # lr 1 decaying 1/1.15 per epoch after epoch 14, gradient norm clipped at 10.
    ("ptb", "sgd"): dict(epochs=55, lr=1.0, decay=1 / 1.15, decay_after=14, decay_every=1),
    # Sec. 5.4.1: 5 epochs, L = 5, fixed gamma = 1e-3, eta = 1 dropping 10x each epoch after the
    # third, eta' = 1, noise 1e-4; momentum 0.5 (Sec. 5.4.2, used here in both loops: ours)
    ("ptb", "entropysgd"): dict(epochs=5, lr=1.0, decay=0.1, decay_after=3, decay_every=1, L=5, gamma=1e-3,
                                sgld_lr=1.0, noise=1e-4, momentum=0.5),
    # Sec. 5.4.2: Adam, 50 epochs, lr 0.002 halving every 5 epochs
    ("char", "adam"): dict(epochs=50, lr=2e-3, decay=0.5, decay_after=5, decay_every=5),
    # Sec. 5.4.2: Entropy-Adam, 5 epochs, L = 5, fixed gamma = 0.01, lr 0.01 halving every epoch,
    # beta1 = 0.5. eta' and noise are not given: eta' = 1 as for PTB and noise 1e-4 (Sec. 4.2 ranges; ours)
    ("char", "entropyadam"): dict(epochs=5, lr=1e-2, decay=0.5, decay_after=1, decay_every=1, L=5, gamma=0.01,
                                  sgld_lr=1.0, noise=1e-4, momentum=0.5, beta1=0.5),
}
TASK = {
    # Zaremba et al. (2014), large: 2 x 1500, embedding 1500, 35 steps, batch 20, dropout 0.65, clip 10.
    # Their loss is summed over the 35 time steps and averaged over the batch (time_sum), which lr 1,
    # clip 10 and the paper's gamma / eta' assume
    "ptb": dict(layers=2, hidden=1500, bptt=35, batch_size=20, dropout=0.65, init=0.04, clip=10.0, time_sum=True),
    # Sec. 5.4.2: 2 x 128 unrolled 50 steps; batch 50, no dropout, init U(-0.08, 0.08) and clip 5
    # are char-rnn's defaults (Karpathy et al. 2015; ours). char-rnn averages the loss over time
    "char": dict(layers=2, hidden=128, bptt=50, batch_size=50, dropout=0.0, init=0.08, clip=5.0, time_sum=False),
}
ALPHA = 0.75  # Sec. 4.2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="ptb", choices=["ptb", "char"])
    p.add_argument("--optim", default="sgd", choices=["sgd", "adam", "entropysgd", "entropyadam"])
    p.add_argument("--seed", type=int, default=0)
    for k in ("epochs", "lr", "decay", "decay_after", "decay_every", "L", "gamma", "sgld_lr", "noise",
              "momentum", "beta1"):
        p.add_argument(f"--{k}", type=int if k in ("epochs", "decay_after", "decay_every", "L") else float, default=None)
    p.add_argument("--limit", type=int, default=0, help="updates per epoch (0 = full epoch; for smoke tests)")
    p.add_argument("--data_dir", default=os.path.join(os.environ.get("DATA_ROOT", "."), "text"))
    p.add_argument("--out", default="result.json")
    args = p.parse_args()
    for k, v in {**PAPER[(args.task, args.optim)], **TASK[args.task]}.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)
    for k, v in dict(L=1, gamma=0.0, sgld_lr=0.0, noise=0.0, momentum=0.0, beta1=0.9).items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    return args


# ---- data ----------------------------------------------------------------------

def load_corpus(task, root):
    """Token id tensors (train, valid, test) and the vocabulary size."""
    if task == "ptb":  # Mikolov's PTB as distributed by Zaremba et al.: one sentence per line + <eos>
        splits = [open(os.path.join(root, f"ptb.{s}.txt")).read().replace("\n", " <eos> ").split()
                  for s in ("train", "valid", "test")]
        vocab = {w: i for i, w in enumerate(sorted(set(splits[0])))}
    else:  # War and Peace, split 80 / 10 / 10 (Sec. 5.4.2)
        text = open(os.path.join(root, "warpeace_input.txt")).read()
        vocab = {c: i for i, c in enumerate(sorted(set(text)))}
        a, b = int(0.8 * len(text)), int(0.9 * len(text))
        splits = [text[:a], text[a:b], text[b:]]
    return [torch.tensor([vocab[t] for t in s]) for s in splits], len(vocab)


def batchify(ids, bs, device):
    n = len(ids) // bs
    return ids[:n * bs].view(bs, n).t().contiguous().to(device)  # [time, batch]


# ---- model ---------------------------------------------------------------------

class LSTMLM(nn.Module):
    def __init__(self, vocab, task, hidden, layers, dropout, init):
        super().__init__()
        self.onehot = task == "char"  # char-rnn feeds one-hot characters; PTB uses a word embedding
        self.vocab = vocab
        self.embed = None if self.onehot else nn.Embedding(vocab, hidden)
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(vocab if self.onehot else hidden, hidden, layers, dropout=dropout)
        self.out = nn.Linear(hidden, vocab)
        for p in self.parameters():
            nn.init.uniform_(p, -init, init)

    def forward(self, x, state):
        e = F.one_hot(x, self.vocab).float() if self.onehot else self.embed(x)
        h, state = self.lstm(self.drop(e), state)
        return self.out(self.drop(h)), state


def detach(state):
    return None if state is None else tuple(s.detach() for s in state)


@torch.no_grad()
def evaluate(model, data, bptt):
    model.eval()
    state, total, n = None, 0.0, 0
    for i in range(0, len(data) - 1, bptt):
        x, y = data[i:i + bptt], data[i + 1:i + 1 + bptt]
        x = x[:len(y)]
        logits, state = model(x, state)
        total += F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="sum").item()
        n += y.numel()
    model.train()
    return total / n


# ---- training ------------------------------------------------------------------

def build_optimizer(args, model):
    params = model.parameters()
    if args.optim == "sgd":  # Zaremba et al.: no momentum
        return torch.optim.SGD(params, lr=args.lr)
    if args.optim == "adam":
        return torch.optim.Adam(params, lr=args.lr)
    base, kw = ((torch.optim.SGD, dict(lr=args.lr, momentum=args.momentum)) if args.optim == "entropysgd"
                else (torch.optim.Adam, dict(lr=args.lr, betas=(args.beta1, 0.999))))
    return EntropySGD(params, base, sgld_steps=args.L, gamma=args.gamma, gamma_growth=0.0, sgld_lr=args.sgld_lr,
                      noise=args.noise, alpha=ALPHA, sgld_kwargs=dict(momentum=args.momentum),
                      grad_clip=args.clip, **kw)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    tracking.init(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (tr, va, te), vocab = load_corpus(args.task, args.data_dir)
    train = batchify(tr, args.batch_size, device)
    # Zaremba et al. evaluate validation like training and test with batch 1 (char: like training; ours)
    valid = batchify(va, args.batch_size, device)
    test = batchify(te, 1 if args.task == "ptb" else args.batch_size, device)

    model = LSTMLM(vocab, args.task, args.hidden, args.layers, args.dropout, args.init).to(device)
    opt = build_optimizer(args, model)
    entropy = args.optim.startswith("entropy")
    starts = list(range(0, len(train) - 1, args.bptt))
    per_epoch = args.limit or len(starts)

    # Consecutive BPTT segments; the hidden state is carried over and reset at the end of the data.
    pos, state = 0, None

    def closure():
        nonlocal pos, state
        if pos == len(starts):
            pos, state = 0, None
        i = starts[pos]
        pos += 1
        x, y = train[i:i + args.bptt], train[i + 1:i + 1 + args.bptt]
        logits, new_state = model(x[:len(y)], state)
        state = detach(new_state)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        return loss * len(y) if args.time_sum else loss  # len(y) = time steps in this segment

    curve, start, diverged = [], time.time(), False
    for epoch in range(args.epochs):
        decays = (epoch - args.decay_after) // args.decay_every + 1 if epoch >= args.decay_after else 0
        lr = args.lr * args.decay ** decays
        for g in opt.param_groups:
            g["lr"] = lr
        if not entropy:
            pos, state = 0, None  # baselines: one pass over the data per epoch
        total = 0.0
        for _ in range(per_epoch):
            if entropy:
                loss = opt.step(closure)
            else:
                opt.zero_grad()
                loss = closure()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                opt.step()
            total += loss.item()
        if not math.isfinite(total):
            diverged = True
            break
        v, t = evaluate(model, valid, args.bptt), evaluate(model, test, args.bptt)
        train_loss = total / per_epoch / (args.bptt if args.time_sum else 1)  # per token, as val/test
        row = dict(epoch=epoch + 1, effective_epochs=(epoch + 1) * args.L, lr=lr, train_loss=train_loss,
                   val_loss=v, test_loss=t, val_ppl=math.exp(v), test_ppl=math.exp(t))
        if entropy:
            row["grad_ratio"] = opt.grad_ratio
        curve.append(row)
        print(json.dumps(row), flush=True)
        tracking.log(row)

    result = dict(vars(args), vocab=vocab, curve=curve, diverged=diverged, seconds=round(time.time() - start, 1))
    if curve:
        result.update({k: curve[-1][k] for k in ("val_loss", "test_loss", "val_ppl", "test_ppl")})
    with open(args.out, "w") as f:
        json.dump(result, f)
    tracking.finish(result)


if __name__ == "__main__":
    main()
