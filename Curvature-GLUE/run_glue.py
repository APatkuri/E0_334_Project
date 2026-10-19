"""One GLUE fine-tuning run for our contribution: any optimizer from the talk, one protocol, and
both generalization and curvature measured. Each run writes one JSON file (--out).
../sweep.py runs the full experiment design; see README.md for where each setting comes from.

    python run_glue.py --task rte --optim sam --rho 0.15 --ascent_frac 0.25 --out test.json
    python run_glue.py --task sst2 --train_frac 0.05 --optim entropysgd --gamma 0.01 --beta1 0.5 --out test.json

Compute budget (--budget), in units of --epochs * len(train_loader):
    grads  same number of gradient evaluations for every optimizer. One update costs 1 (AdamW),
           2 (SAM, GSAM, WSAM), 2 + (1 + 2q)/p (Eigen-SAM, X-SAM) or L (Entropy-SGD)
    steps  same number of updates for every optimizer

Measured:
    dev metric at --evals evenly spaced points (accuracy; F1 for MRPC/QQP; Matthews corr. for CoLA)
    train metric on a fixed training subset, for the train-dev gap
    final weights, on a fixed --curv_examples training subset with dropout off: lambda_max (power
        iteration), tr(H) (Hutchinson) and SAM sharpness L(w + rho g/|g|) - L(w)
    --track_points times during training, on a fixed --track_examples subset: lambda_max, and
        |cos| between the top eigenvector v1 and both the gradient and the actual update (RQ4)

All models use eager attention, which Hessian-vector products need (double backward).
"""
import argparse
import json
import math
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("SAM", "Entropy-SGD", "Maximum-Hessian"):
    sys.path.insert(0, os.path.join(ROOT, sub))
sys.path.append(ROOT)

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          get_constant_schedule, get_linear_schedule_with_warmup)

import tracking
from entropy_optim import EntropySGD
from hessian_tools import hessian_mode, hessian_trace, top_eigenvalue
from hessian_tools.spectrum import dot
from sam_optim import GSAM, SAM, WSAM, EigenSAM, SAMBase, XSAM

TASKS = {  # task -> (text fields, num_labels, dev split, primary metric)
    "rte": (("sentence1", "sentence2"), 2, "validation", "acc"),
    "mrpc": (("sentence1", "sentence2"), 2, "validation", "f1"),
    "cola": (("sentence", None), 2, "validation", "mcc"),
    "sst2": (("sentence", None), 2, "validation", "acc"),
    "qnli": (("question", "sentence"), 2, "validation", "acc"),
    "qqp": (("question1", "question2"), 2, "validation", "f1"),
    "mnli": (("premise", "hypothesis"), 3, "validation_matched", "acc"),
}
OPTIMS = ["adamw", "sam", "gsam", "wsam", "eigensam", "xsam", "entropysgd"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="rte", choices=TASKS)
    p.add_argument("--train_frac", type=float, default=1.0, help="subsample the training set (e.g. 0.05)")
    p.add_argument("--subset_seed", type=int, default=0, help="fixes the subsample across optimizers and seeds")
    p.add_argument("--model", default="bert-base-uncased")
    p.add_argument("--optim", default="adamw", choices=OPTIMS)
    p.add_argument("--budget", default="grads", choices=["grads", "steps"])
    p.add_argument("--epochs", type=float, default=3, help="budget, in AdamW epochs")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1 (Entropy-Adam uses 0.5, Sec. 5.4.2)")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--schedule", default="linear", choices=["linear", "constant"],
                   help="linear: warmup then linear decay (BERT); constant: fixed lr, no warmup (Kaur et al.)")
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--target_train_acc", type=float, default=0.0,
                   help="stop once train accuracy reaches this, checked every epoch (Kaur et al.: 0.99; 0 = off)")
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--evals", type=int, default=3, help="dev evaluations, evenly spaced; the last is final")
    # SAM family
    p.add_argument("--rho", type=float, default=0.05)
    p.add_argument("--ascent_frac", type=float, default=1.0, help="SAM: batch fraction for the ascent step")
    p.add_argument("--alpha", type=float, default=None, help="GSAM / Eigen-SAM / X-SAM (default 0.01 / 0.2 / 0.2)")
    p.add_argument("--rho_min", type=float, default=None, help="GSAM: enables the lr-tied rho schedule")
    p.add_argument("--eig_every", type=int, default=100, help="Eigen-SAM / X-SAM: p")
    p.add_argument("--power_iters", type=int, default=5, help="Eigen-SAM / X-SAM: q")
    p.add_argument("--no_normalize_grad", action="store_true", help="X-SAM: keep gradient scale")
    # shared by WSAM (sharpness weight, default 0.9) and Entropy-SGD (scope gamma_0, default 0.03)
    p.add_argument("--gamma", type=float, default=None)
    # Entropy-SGD (Entropy-Adam: AdamW outside, Adam inside with eta' = --lr)
    p.add_argument("--sgld_steps", type=int, default=5, help="L")
    p.add_argument("--gamma_growth", type=float, default=0.0)
    p.add_argument("--noise", type=float, default=1e-4)
    p.add_argument("--mu_alpha", type=float, default=0.75, help="averaging weight for mu")
    # measurements
    p.add_argument("--curv_examples", type=int, default=1000)
    p.add_argument("--lmax_iters", type=int, default=20)
    p.add_argument("--trace_samples", type=int, default=20, help="Hutchinson samples (0 = skip)")
    p.add_argument("--sharp_rho", type=float, default=0.05)
    p.add_argument("--track_points", type=int, default=5, help="lambda_max / angle checks during training (0 = off)")
    p.add_argument("--track_examples", type=int, default=256)
    p.add_argument("--track_iters", type=int, default=10)
    p.add_argument("--train_eval_examples", type=int, default=5000)
    p.add_argument("--out", required=True, help="JSON file for this run")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---- data ----------------------------------------------------------------------

def build_data(args, tokenizer):
    (f1, f2), num_labels, dev_split, _ = TASKS[args.task]
    raw = load_dataset("nyu-mll/glue", args.task)
    train = raw["train"]
    if args.train_frac < 1.0:
        train = train.shuffle(seed=args.subset_seed).select(range(int(len(train) * args.train_frac)))

    def tokenize(batch):
        texts = (batch[f1],) if f2 is None else (batch[f1], batch[f2])
        return tokenizer(*texts, truncation=True, max_length=args.max_len)

    cols = [c for c in train.column_names if c != "label"]
    prep = lambda ds: ds.map(tokenize, batched=True, remove_columns=cols).rename_column("label", "labels")
    return prep(train), prep(raw[dev_split]), num_labels


def subset(ds, n, seed):
    """The same n examples for every run (independent of --seed)."""
    idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(seed))[:n]
    return ds.select(idx.tolist())


def on_device(loader, device):
    return [{k: v.to(device) for k, v in b.items()} for b in loader]


def batch_stream(loader, device):
    while True:
        for batch in loader:
            yield {k: v.to(device) for k, v in batch.items()}


# ---- optimizers ----------------------------------------------------------------

def build_optimizer(args, model):
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
         "weight_decay": 0.0},
    ]
    base, kw = torch.optim.AdamW, dict(lr=args.lr, betas=(args.beta1, 0.999))
    alpha = lambda default: args.alpha if args.alpha is not None else default
    gamma = lambda default: args.gamma if args.gamma is not None else default

    if args.optim == "adamw":
        return base(groups, **kw)
    if args.optim == "sam":
        return SAM(groups, base, rho=args.rho, **kw)
    if args.optim == "gsam":
        return GSAM(groups, base, rho=args.rho, alpha=alpha(0.01), rho_min=args.rho_min,
                    lr_max=args.lr, lr_min=0.0, **kw)
    if args.optim == "wsam":
        return WSAM(groups, base, rho=args.rho, gamma=gamma(0.9), **kw)
    if args.optim == "eigensam":
        return EigenSAM(groups, base, rho=args.rho, alpha=alpha(0.2),
                        eig_every=args.eig_every, power_iters=args.power_iters, **kw)
    if args.optim == "xsam":
        return XSAM(groups, base, rho=args.rho, alpha=alpha(0.2), eig_every=args.eig_every,
                    power_iters=args.power_iters, normalize_grad=not args.no_normalize_grad, **kw)
    return EntropySGD(groups, base, sgld_steps=args.sgld_steps, gamma=gamma(0.03),
                      gamma_growth=args.gamma_growth, sgld_lr=args.lr, noise=args.noise,
                      alpha=args.mu_alpha, sgld_optimizer=torch.optim.Adam, **kw)


def update_cost(args):
    """Gradient evaluations per update (a Hessian-vector product counts as two)."""
    if args.optim == "adamw":
        return 1.0
    if args.optim == "entropysgd":
        return float(args.sgld_steps)
    if args.optim in ("eigensam", "xsam"):
        return 2.0 + (1 + 2 * args.power_iters) / args.eig_every
    if args.optim == "sam" and args.ascent_frac < 1.0:
        return 1.0 + args.ascent_frac
    return 2.0


def train_step(model, optimizer, stream, args):
    if isinstance(optimizer, EntropySGD):  # a new mini-batch for each of the L SGLD steps
        return optimizer.step(lambda: model(**next(stream)).loss).item()
    batch = next(stream)
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


# ---- metrics -------------------------------------------------------------------

def scores(pred, label, loss):
    out = {"acc": float((pred == label).mean()), "loss": loss}
    if pred.max(initial=0) <= 1 and label.max(initial=0) <= 1:  # binary tasks
        tp, tn = float(((pred == 1) & (label == 1)).sum()), float(((pred == 0) & (label == 0)).sum())
        fp, fn = float(((pred == 1) & (label == 0)).sum()), float(((pred == 0) & (label == 1)).sum())
        out["f1"] = 2 * tp / max(2 * tp + fp + fn, 1.0)
        den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        out["mcc"] = (tp * tn - fp * fn) / den if den > 0 else 0.0
    return out


@torch.no_grad()
def evaluate(model, loader, device):
    was_training = model.training
    model.eval()
    preds, labels, loss, n = [], [], 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        preds.append(out.logits.argmax(-1).cpu().numpy())
        labels.append(batch["labels"].cpu().numpy())
        loss += out.loss.item() * len(labels[-1])
        n += len(labels[-1])
    model.train(was_training)
    return scores(np.concatenate(preds), np.concatenate(labels), loss / n)


# ---- curvature -----------------------------------------------------------------

def mean_gradient(params, loss_fn, batches):
    grads = [torch.zeros_like(p) for p in params]
    loss = 0.0
    for b in batches:
        l = loss_fn(b)
        for g, gi in zip(grads, torch.autograd.grad(l, params, allow_unused=True)):
            if gi is not None:
                g.add_(gi, alpha=1.0 / len(batches))
        loss += l.item() / len(batches)
    return grads, loss


def sam_sharpness(params, loss_fn, batches, rho):
    """First-order SAM sharpness: L(w + rho g/|g|) - L(w), with g the gradient over `batches`."""
    g, loss = mean_gradient(params, loss_fn, batches)
    scale = rho / (math.sqrt(dot(g, g).item()) + 1e-12)
    backup = [p.detach().clone() for p in params]
    with torch.no_grad():
        for p, gi in zip(params, g):
            p.add_(gi, alpha=scale)
        loss_adv = sum(loss_fn(b).item() for b in batches) / len(batches)
        for p, b in zip(params, backup):
            p.copy_(b)
    return loss_adv - loss


def abs_cos(a, b):
    return abs(dot(a, b).item()) / (math.sqrt(dot(a, a).item() * dot(b, b).item()) + 1e-12)


def main():
    args = parse_args()
    set_seed(args.seed)
    tracking.init(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.time()
    primary = TASKS[args.task][3]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_ds, dev_ds, num_labels = build_data(args, tokenizer)
    collate = DataCollatorWithPadding(tokenizer)
    loader = lambda ds, bs, shuffle=False: DataLoader(ds, batch_size=bs, shuffle=shuffle, collate_fn=collate)
    train_loader = loader(train_ds, args.batch_size, shuffle=True)
    dev_loader = loader(dev_ds, 2 * args.batch_size)
    train_eval_loader = loader(subset(train_ds, args.train_eval_examples, 1234), 2 * args.batch_size)
    curv_batches = on_device(loader(subset(train_ds, args.curv_examples, 1234), 32), device)
    track_batches = on_device(loader(subset(train_ds, args.track_examples, 4321), 32), device)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=num_labels, attn_implementation="eager").to(device)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    loss_fn = lambda b: model(**b).loss

    cost = update_cost(args)
    updates = args.epochs * len(train_loader) / (cost if args.budget == "grads" else 1.0)
    updates = max(1, round(updates))
    optimizer = build_optimizer(args, model)
    if args.schedule == "constant":
        scheduler = get_constant_schedule(optimizer)
    else:
        scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_ratio * updates), updates)
    eval_at = {round((i + 1) * updates / args.evals) for i in range(args.evals)}
    track_at = {int(i * updates / args.track_points) for i in range(args.track_points)}
    epoch_updates = max(1, round(len(train_loader) / (cost if args.budget == "grads" else 1.0)))
    log_every = max(1, updates // 20)
    print(f"{args.task} (train {len(train_ds)}) | {args.optim} | {updates} updates x {cost:.2f} grads", flush=True)

    stream = batch_stream(train_loader, device)
    dev_curve, track, ratios, diverged, train_acc = [], [], [], False, None
    for step in range(updates):
        if step in track_at:  # lambda_max, v1 and gradient on the tracking subset, dropout off
            with hessian_mode(model):
                lam, v = top_eigenvalue(params, loss_fn, track_batches, iters=args.track_iters)
                g, _ = mean_gradient(params, loss_fn, track_batches)
            before = [p.detach().clone() for p in params]

        loss = train_step(model, optimizer, stream, args)
        scheduler.step()
        if isinstance(optimizer, EntropySGD):
            ratios.append(optimizer.grad_ratio)

        if step in track_at:
            dw = [p.detach() - b for p, b in zip(params, before)]
            track.append(dict(step=step, lmax=lam, cos_grad_v=abs_cos(g, v), cos_update_v=abs_cos(dw, v)))
            print(f"step {step}: lambda_max {lam:.2f} |cos(update, v1)| {track[-1]['cos_update_v']:.4f}", flush=True)
            tracking.log(track[-1])
            del v, g, before, dw
        if not math.isfinite(loss):
            diverged = True
            print(f"step {step}: loss is {loss}, stopping", flush=True)
            break
        if (step + 1) % log_every == 0:
            print(f"step {step + 1}/{updates} loss {loss:.4f}", flush=True)
            tracking.log(dict(step=step + 1, loss=loss, lr=scheduler.get_last_lr()[0]))
        if step + 1 in eval_at:
            dev = evaluate(model, dev_loader, device)
            dev_curve.append(dict(step=step + 1, **dev))
            print(f"step {step + 1}: dev {primary} {dev[primary]:.4f}", flush=True)
            tracking.log(dict(step=step + 1, dev=dev))
        if args.target_train_acc > 0 and (step + 1) % epoch_updates == 0:
            train_acc = evaluate(model, train_eval_loader, device)["acc"]
            print(f"step {step + 1}: train acc {train_acc:.4f}", flush=True)
            tracking.log(dict(step=step + 1, train_acc=train_acc))
            if train_acc >= args.target_train_acc:
                break
    done = step + 1

    result = dict(vars(args), train_size=len(train_ds), updates=done, max_updates=updates, cost_per_update=cost,
                  grad_evals=round(done * cost), primary=primary, diverged=diverged,
                  dev_curve=dev_curve, track=track)
    if args.target_train_acc > 0:
        result["reached_target"] = train_acc is not None and train_acc >= args.target_train_acc
    if ratios:
        result["grad_ratio"] = float(np.mean(ratios))  # Sec. 4.2: tune gamma so that this is about 1
    if not diverged:
        if not dev_curve or dev_curve[-1]["step"] != done:
            dev_curve.append(dict(step=done, **evaluate(model, dev_loader, device)))
        result["dev"] = dev_curve[-1]
        result["train"] = evaluate(model, train_eval_loader, device)
        result["gap"] = result["train"][primary] - result["dev"][primary]
        with hessian_mode(model):
            result["lmax"] = top_eigenvalue(params, loss_fn, curv_batches, iters=args.lmax_iters)[0]
            print(f"final lambda_max {result['lmax']:.3f}", flush=True)
            if args.trace_samples > 0:
                result["trace"], result["trace_se"] = hessian_trace(
                    params, loss_fn, curv_batches, samples=args.trace_samples)
            result["sharpness"] = sam_sharpness(params, loss_fn, curv_batches, args.sharp_rho)
    result["seconds"] = round(time.time() - start_time, 1)
    result["gpu"] = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"

    print(json.dumps({k: result.get(k) for k in ("dev", "gap", "lmax", "trace", "sharpness", "seconds")}))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out + ".tmp", "w") as f:
        json.dump(result, f, indent=1)
    os.replace(args.out + ".tmp", args.out)  # a JSON file exists only for finished runs
    tracking.finish(result)


if __name__ == "__main__":
    main()
