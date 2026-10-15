"""Fine-tune a Hugging Face encoder on a GLUE task with AdamW or Entropy-SGD.

Entropy-SGD reads L mini-batches per update (one per SGLD step). Here --epochs counts passes
over the data, so both optimizers use the same number of gradient evaluations and Entropy-SGD
makes L times fewer updates.

The outer optimizer is AdamW in both cases, so `--optim entropysgd` is Entropy-Adam (Sec. 5.4.2).

Examples:
    python train_glue.py --task sst2 --optim adamw
    python train_glue.py --task sst2 --optim entropysgd --sgld_steps 5 --gamma 0.03 --gamma_growth 1e-3
    python train_glue.py --task rte  --optim entropysgd --sgld_optim sgd --sgld_lr 1e-3 --sgld_momentum 0.9
"""
import argparse
import random

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, get_linear_schedule_with_warmup)

from entropy_optim import EntropySGD

TASKS = {  # task -> (text fields, num_labels, validation split)
    "sst2": (("sentence", None), 2, "validation"),
    "cola": (("sentence", None), 2, "validation"),
    "mrpc": (("sentence1", "sentence2"), 2, "validation"),
    "rte": (("sentence1", "sentence2"), 2, "validation"),
    "qnli": (("question", "sentence"), 2, "validation"),
    "qqp": (("question1", "question2"), 2, "validation"),
    "mnli": (("premise", "hypothesis"), 3, "validation_matched"),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="sst2", choices=TASKS)
    p.add_argument("--model", default="bert-base-uncased")
    p.add_argument("--optim", default="entropysgd", choices=["adamw", "entropysgd"])
    p.add_argument("--epochs", type=int, default=3, help="passes over the training data")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5, help="outer (AdamW) learning rate")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    # Entropy-SGD hyperparameters
    p.add_argument("--sgld_steps", type=int, default=5, help="L, SGLD steps per update")
    p.add_argument("--gamma", type=float, default=0.03, help="initial scope gamma_0")
    p.add_argument("--gamma_growth", type=float, default=0.0,
                   help="scoping gamma_1: gamma_t = gamma_0 (1 + gamma_1)^t (paper: 1e-3)")
    p.add_argument("--sgld_optim", default="adam", choices=["adam", "sgd"])
    p.add_argument("--sgld_lr", type=float, default=None,
                   help="eta', kept fixed (not annealed). Default: --lr for adam, 0.1 for sgd")
    p.add_argument("--sgld_momentum", type=float, default=0.0, help="Nesterov momentum for --sgld_optim sgd")
    p.add_argument("--noise", type=float, default=1e-4, help="thermal noise epsilon")
    p.add_argument("--alpha", type=float, default=0.75, help="averaging weight for mu")
    p.add_argument("--scale_by_gamma", action="store_true",
                   help="outer gradient gamma (x - mu) as in Alg. 1, instead of x - mu (Sec. 4.3)")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_data(args, tokenizer):
    (f1, f2), num_labels, val_split = TASKS[args.task]
    raw = load_dataset("nyu-mll/glue", args.task)

    def tokenize(batch):
        texts = (batch[f1],) if f2 is None else (batch[f1], batch[f2])
        return tokenizer(*texts, truncation=True, max_length=args.max_len)

    cols = [c for c in raw["train"].column_names if c != "label"]
    data = raw.map(tokenize, batched=True, remove_columns=cols).rename_column("label", "labels")
    collate = DataCollatorWithPadding(tokenizer)
    train = DataLoader(data["train"], batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val = DataLoader(data[val_split], batch_size=2 * args.batch_size, collate_fn=collate)
    return train, val, num_labels


def build_optimizer(args, model):
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
         "weight_decay": 0.0},
    ]
    base, kw = torch.optim.AdamW, dict(lr=args.lr)

    if args.optim == "adamw":
        return base(groups, **kw)
    if args.sgld_optim == "adam":
        sgld, sgld_kwargs = torch.optim.Adam, {}
        sgld_lr = args.sgld_lr if args.sgld_lr is not None else args.lr
    else:
        m = args.sgld_momentum
        sgld, sgld_kwargs = torch.optim.SGD, dict(momentum=m, nesterov=m > 0)
        sgld_lr = args.sgld_lr if args.sgld_lr is not None else 0.1
    return EntropySGD(groups, base, sgld_steps=args.sgld_steps, gamma=args.gamma,
                      gamma_growth=args.gamma_growth, sgld_lr=sgld_lr, noise=args.noise,
                      alpha=args.alpha, sgld_optimizer=sgld, sgld_kwargs=sgld_kwargs,
                      scale_by_gamma=args.scale_by_gamma, **kw)


def batch_stream(loader, device):
    """Endless stream of training batches. Entropy-SGD reads L of them per update."""
    while True:
        for batch in loader:
            yield {k: v.to(device) for k, v in batch.items()}


def train_step(model, optimizer, stream):
    loss_fn = lambda: model(**next(stream)).loss  # a new mini-batch on every call
    if isinstance(optimizer, EntropySGD):
        return optimizer.step(loss_fn).item()
    optimizer.zero_grad()
    loss = loss_fn()
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        preds = model(**batch).logits.argmax(-1)
        correct += (preds == batch["labels"]).sum().item()
        total += batch["labels"].numel()
    model.train()
    return correct / total


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_loader, val_loader, num_labels = build_data(args, tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=num_labels).to(device)
    model.train()

    L = args.sgld_steps if args.optim == "entropysgd" else 1
    steps_per_epoch = len(train_loader) // L
    total_steps = args.epochs * steps_per_epoch
    optimizer = build_optimizer(args, model)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_ratio * total_steps), total_steps)
    log_every = max(1, 100 // L)  # log at the same rate of data seen for both optimizers

    stream = batch_stream(train_loader, device)
    step = 0
    for epoch in range(args.epochs):
        for _ in range(steps_per_epoch):
            loss = train_step(model, optimizer, stream)
            scheduler.step()
            step += 1
            if step % log_every == 0:
                msg = f"epoch {epoch} step {step}/{total_steps} loss {loss:.4f}"
                if isinstance(optimizer, EntropySGD):
                    msg += f" gamma {optimizer.gamma:.3g} grad_ratio {optimizer.grad_ratio:.3f}"
                print(msg, flush=True)
        print(f"epoch {epoch}: val acc = {evaluate(model, val_loader, device):.4f}", flush=True)


if __name__ == "__main__":
    main()
