"""Fine-tune a Hugging Face encoder on a GLUE task with AdamW or any SAM variant.

Examples:
    python train_glue.py --task sst2 --optim adamw
    python train_glue.py --task sst2 --optim sam --rho 0.05 --ascent_frac 0.25
    python train_glue.py --task rte  --optim gsam --rho 0.05 --rho_min 0.01 --alpha 0.1
    python train_glue.py --task mrpc --optim wsam --gamma 0.9
    python train_glue.py --task sst2 --optim eigensam --alpha 0.2 --eig_every 100
    python train_glue.py --task sst2 --optim xsam --alpha 0.2 --eig_every 100
"""
import argparse
import random

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, get_linear_schedule_with_warmup)

from sam_optim import GSAM, SAM, WSAM, EigenSAM, SAMBase, XSAM

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
    p.add_argument("--optim", default="sam",
                   choices=["adamw", "sam", "gsam", "wsam", "eigensam", "xsam"])
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    # SAM-family hyperparameters
    p.add_argument("--rho", type=float, default=0.05)
    p.add_argument("--ascent_frac", type=float, default=1.0,
                   help="SAM only: fraction of the batch used for the ascent step (Bahri et al. use 0.25)")
    p.add_argument("--alpha", type=float, default=None,
                   help="GSAM / Eigen-SAM / X-SAM strength (defaults: 0.01 / 0.2 / 0.2)")
    p.add_argument("--rho_min", type=float, default=None, help="GSAM: enables the lr-tied rho schedule")
    p.add_argument("--gamma", type=float, default=0.9, help="WSAM sharpness weight")
    p.add_argument("--eig_every", type=int, default=100, help="Eigen-SAM / X-SAM: p")
    p.add_argument("--power_iters", type=int, default=5, help="Eigen-SAM / X-SAM: q")
    p.add_argument("--no_normalize_grad", action="store_true", help="X-SAM: keep gradient scale")
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
    if args.optim == "sam":
        return SAM(groups, base, rho=args.rho, **kw)
    if args.optim == "gsam":
        return GSAM(groups, base, rho=args.rho, alpha=args.alpha if args.alpha is not None else 0.01,
                    rho_min=args.rho_min, lr_max=args.lr, lr_min=0.0, **kw)
    if args.optim == "wsam":
        return WSAM(groups, base, rho=args.rho, gamma=args.gamma, **kw)
    alpha = args.alpha if args.alpha is not None else 0.2
    if args.optim == "eigensam":
        return EigenSAM(groups, base, rho=args.rho, alpha=alpha,
                        eig_every=args.eig_every, power_iters=args.power_iters, **kw)
    return XSAM(groups, base, rho=args.rho, alpha=alpha, eig_every=args.eig_every,
                power_iters=args.power_iters, normalize_grad=not args.no_normalize_grad, **kw)


def train_step(model, optimizer, batch, args):
    loss_fn = lambda: model(**batch).loss
    if not isinstance(optimizer, SAMBase):
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        optimizer.step()
        return loss.item()
    if isinstance(optimizer, SAM) and args.ascent_frac < 1.0:
        m = max(1, int(batch["labels"].size(0) * args.ascent_frac))
        micro = {k: v[:m] for k, v in batch.items()}
        return optimizer.step(loss_fn, ascent_closure=lambda: model(**micro).loss).item()
    return optimizer.step(loss_fn).item()


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

    model_kwargs = {}
    if args.optim in ("eigensam", "xsam"):
        # Hessian-vector products need double backward, which fused SDPA / flash kernels lack.
        model_kwargs["attn_implementation"] = "eager"
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=num_labels, **model_kwargs).to(device)
    model.train()

    optimizer = build_optimizer(args, model)
    total_steps = args.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_ratio * total_steps), total_steps)

    step = 0
    for epoch in range(args.epochs):
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = train_step(model, optimizer, batch, args)
            scheduler.step()
            step += 1
            if step % 100 == 0:
                msg = f"epoch {epoch} step {step}/{total_steps} loss {loss:.4f}"
                if getattr(optimizer, "top_eigenvalue", None) is not None:
                    msg += f" lambda_max {optimizer.top_eigenvalue:.2f}"
                print(msg, flush=True)
        print(f"epoch {epoch}: val acc = {evaluate(model, val_loader, device):.4f}", flush=True)


if __name__ == "__main__":
    main()
