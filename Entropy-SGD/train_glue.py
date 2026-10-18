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
import csv
import json
import os
import random
import time

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
    p.add_argument("--eval_landscape", action="store_true", default=True,
                   help="run post-training loss landscape & sharpness analysis on best model")
    p.add_argument("--output_dir", default=None,
                   help="run output directory (default: a unique directory under ../results/glue)")
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


def classification_metrics(task, labels, predictions):
    total = len(labels)
    correct = sum(label == prediction for label, prediction in zip(labels, predictions))
    metrics = {"accuracy": correct / total}
    if task in ("mrpc", "qqp", "cola"):
        tp = sum(label == 1 and prediction == 1 for label, prediction in zip(labels, predictions))
        fp = sum(label != 1 and prediction == 1 for label, prediction in zip(labels, predictions))
        fn = sum(label == 1 and prediction != 1 for label, prediction in zip(labels, predictions))
        tn = total - tp - fp - fn
        if task in ("mrpc", "qqp"):
            f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
            metrics.update(f1=f1, combined_score=(metrics["accuracy"] + f1) / 2)
        else:
            denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
            metrics["matthews_correlation"] = ((tp * tn - fp * fn) / denominator) if denominator else 0.0
    return metrics


def primary_metric(task):
    if task == "cola":
        return "matthews_correlation"
    if task in ("mrpc", "qqp"):
        return "combined_score"
    return "accuracy"


@torch.no_grad()
def evaluate(model, loader, device, task):
    was_training = model.training
    model.eval()
    total_loss, total = 0.0, 0
    labels, predictions = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model(**batch)
        batch_labels = batch["labels"]
        batch_predictions = output.logits.argmax(-1)
        count = batch_labels.numel()
        total_loss += output.loss.item() * count
        total += count
        labels.extend(batch_labels.cpu().tolist())
        predictions.extend(batch_predictions.cpu().tolist())
    model.train(was_training)
    return total_loss / total, classification_metrics(task, labels, predictions), labels, predictions


def save_json(path, value):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2)
        f.write("\n")


def save_predictions(path, labels, predictions):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(("example", "label_id", "prediction_id", "correct"))
        writer.writerows((i, label, prediction, label == prediction)
                         for i, (label, prediction) in enumerate(zip(labels, predictions)))


def plot_history(path, history, score_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    fig, (loss_ax, score_ax) = plt.subplots(1, 2, figsize=(10, 4))
    loss_ax.plot(epochs, [row["train_loss"] for row in history], marker="o", label="train")
    loss_ax.plot(epochs, [row["validation_loss"] for row in history], marker="o", label="validation")
    loss_ax.set(xlabel="Epoch", ylabel="Loss", title="Training and validation loss")
    loss_ax.legend()
    score_ax.plot(epochs, [row["validation_metrics"][score_name] for row in history], marker="o")
    score_ax.set(xlabel="Epoch", ylabel=score_name.replace("_", " ").title(),
                 title=f"Validation {score_name.replace('_', ' ')}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(path, labels, predictions, num_labels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = [[0 for _ in range(num_labels)] for _ in range(num_labels)]
    for label, prediction in zip(labels, predictions):
        matrix[label][prediction] += 1
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, label="Examples")
    ax.set(xlabel="Predicted label", ylabel="True label", title="Best validation confusion matrix",
           xticks=range(num_labels), yticks=range(num_labels))
    for i in range(num_labels):
        for j in range(num_labels):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_entropy_dynamics(path, step_rows):
    if not step_rows or "gamma" not in step_rows[0]:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [r["step"] for r in step_rows]
    gammas = [r["gamma"] for r in step_rows]
    ratios = [r["grad_ratio"] for r in step_rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(steps, gammas, color="tab:purple", linewidth=1.8)
    ax1.set(xlabel="Optimizer Step", ylabel="Scope (gamma)", title="Scoping Schedule (gamma_t)")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(steps, ratios, color="tab:olive", linewidth=1.8)
    ax2.axhline(1.0, color="gray", linestyle="--", alpha=0.7, label="Target ratio ~ 1.0")
    ax2.set(xlabel="Optimizer Step", ylabel="||outer grad|| / ||grad f||", title="Gradient Ratio Dynamics")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def create_output_dir(args):
    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
    else:
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "glue")
        run_name = f"{args.task}_{args.optim}_seed{args.seed}_{time.time_ns()}"
        output_dir = os.path.join(root, run_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def main():
    args = parse_args()
    output_dir = create_output_dir(args)
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
    if total_steps < 1:
        raise ValueError("Training would have zero updates; reduce --sgld_steps or increase the dataset size.")
    optimizer = build_optimizer(args, model)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_ratio * total_steps), total_steps)
    log_every = max(1, 100 // L)  # log at the same rate of data seen for both optimizers

    stream = batch_stream(train_loader, device)
    score_name = primary_metric(args.task)
    history, best_score, best_epoch = [], float("-inf"), None
    start = time.time()
    metadata = {
        "args": vars(args),
        "output_dir": output_dir,
        "device": str(device),
        "train_examples": len(train_loader.dataset),
        "validation_examples": len(val_loader.dataset),
        "total_optimizer_steps": total_steps,
        "training_batches": total_steps * L,
        "inner_steps_per_update": L,
        "primary_metric": score_name,
        "status": "running",
    }
    save_json(os.path.join(output_dir, "metrics.json"), metadata)
    step_rows = []
    with open(os.path.join(output_dir, "steps.jsonl"), "w", encoding="utf-8") as step_log:
        step = 0
        for epoch in range(args.epochs):
            epoch_start = time.time()
            losses = []
            for _ in range(steps_per_epoch):
                loss = train_step(model, optimizer, stream)
                losses.append(loss)
                scheduler.step()
                step += 1
                if step % log_every == 0:
                    row = {"epoch": epoch + 1, "step": step, "total_steps": total_steps, "loss": loss}
                    if isinstance(optimizer, EntropySGD):
                        row.update(gamma=optimizer.gamma, grad_ratio=optimizer.grad_ratio)
                    step_rows.append(row)
                    step_log.write(json.dumps(row) + "\n")
                    step_log.flush()
                    msg = f"epoch {epoch + 1} step {step}/{total_steps} loss {loss:.4f}"
                    if isinstance(optimizer, EntropySGD):
                        msg += f" gamma {optimizer.gamma:.3g} grad_ratio {optimizer.grad_ratio:.3f}"
                    print(msg, flush=True)

            validation_loss, validation_metrics, labels, predictions = evaluate(
                model, val_loader, device, args.task)
            row = {
                "epoch": epoch + 1,
                "global_step": step,
                "train_loss": float(np.mean(losses)),
                "validation_loss": validation_loss,
                "validation_metrics": validation_metrics,
                "epoch_seconds": time.time() - epoch_start,
            }
            if isinstance(optimizer, EntropySGD):
                row.update(gamma=optimizer.gamma, grad_ratio=optimizer.grad_ratio)
            history.append(row)

            score = validation_metrics[score_name]
            if score > best_score:
                best_score, best_epoch = score, epoch + 1
                best_model_dir = os.path.join(output_dir, "best_model")
                model.save_pretrained(best_model_dir)
                tokenizer.save_pretrained(best_model_dir)
                save_predictions(os.path.join(output_dir, "best_validation_predictions.csv"),
                                 labels, predictions)
                plot_confusion_matrix(os.path.join(output_dir, "confusion_matrix.png"),
                                      labels, predictions, num_labels)

            metadata.update(
                status="running",
                history=history,
                best_epoch=best_epoch,
                best_validation_metrics=history[best_epoch - 1]["validation_metrics"],
                best_model_dir=os.path.join(output_dir, "best_model"),
                elapsed_seconds=time.time() - start,
            )
            save_json(os.path.join(output_dir, "metrics.json"), metadata)
            print(f"epoch {epoch + 1}: val loss = {validation_loss:.4f}, "
                  f"val {score_name} = {score:.4f}", flush=True)

    plot_history(os.path.join(output_dir, "training_curves.png"), history, score_name)
    if isinstance(optimizer, EntropySGD):
        plot_entropy_dynamics(os.path.join(output_dir, "entropy_dynamics.png"), step_rows)

    if args.eval_landscape and os.path.exists(os.path.join(output_dir, "best_model")):
        try:
            from analyze_landscape import run_landscape_analysis
            print("\n--- Running Post-Training Landscape and Sharpness Analysis ---")
            run_landscape_analysis(
                checkpoint_dir=os.path.join(output_dir, "best_model"),
                task=args.task,
                output_dir=output_dir,
                eval_samples=min(500, len(val_loader.dataset)),
                power_iter=10
            )
        except Exception as e:
            print(f"Warning: Landscape analysis encountered an error: {e}")

    metadata.update(status="completed", elapsed_seconds=time.time() - start)
    save_json(os.path.join(output_dir, "metrics.json"), metadata)
    print(f"Best epoch: {best_epoch} ({score_name} = {best_score:.4f})", flush=True)
    print(f"Saved run artifacts to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
