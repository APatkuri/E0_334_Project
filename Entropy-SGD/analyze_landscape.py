"""Analyze sharpness, loss landscape, and curvature of fine-tuned GLUE checkpoints.

Reproduces the core thesis of Chaudhari et al. (2017) (flatness & wide valleys)
for Transformers fine-tuned on GLUE:
1. 1D Loss Landscape Slice: Loss profile around the minimum along random filter-normalized directions.
2. 1D Linear Interpolation: If two model checkpoints are provided (e.g. AdamW vs. Entropy-SGD),
   evaluates loss along w(alpha) = (1-alpha)*w1 + alpha*w2.
3. Parameter Noise Robustness (eps-sharpness): Tests accuracy and loss degradation under
   increasing Gaussian weight perturbations.
4. Top Hessian Eigenvalue (lambda_max): Computes the dominant curvature using power iteration
   with Hessian-Vector Products (Pearlmutter trick) without storing the Hessian matrix.

Usage:
    python analyze_landscape.py --checkpoint ../results/glue/<run>/best_model --task sst2
    python analyze_landscape.py --checkpoint ../results/glue/<run_esgd>/best_model \
                                --checkpoint2 ../results/glue/<run_adamw>/best_model --task sst2
"""
import argparse
import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

TASKS = {
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
    p.add_argument("--checkpoint", required=True, help="Path to best_model directory")
    p.add_argument("--checkpoint2", default=None, help="Optional second checkpoint for 1D interpolation")
    p.add_argument("--task", default="sst2", choices=TASKS)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--eval_samples", type=int, default=500, help="Max validation samples for fast evaluation")
    p.add_argument("--power_iter", type=int, default=15, help="Iterations for top Hessian eigenvalue")
    p.add_argument("--output_dir", default=None, help="Directory to save plots and json summary")
    return p.parse_args()


def load_val_data(task, tokenizer, max_len, batch_size, eval_samples=500):
    (f1, f2), num_labels, val_split = TASKS[task]
    raw = load_dataset("nyu-mll/glue", task)
    val_data = raw[val_split]
    if eval_samples and len(val_data) > eval_samples:
        val_data = val_data.select(range(eval_samples))

    def tokenize(batch):
        texts = (batch[f1],) if f2 is None else (batch[f1], batch[f2])
        return tokenizer(*texts, truncation=True, max_length=max_len)

    cols = [c for c in val_data.column_names if c != "label"]
    data = val_data.map(tokenize, batched=True, remove_columns=cols).rename_column("label", "labels")
    collate = DataCollatorWithPadding(tokenizer)
    return DataLoader(data, batch_size=batch_size, shuffle=False, collate_fn=collate), num_labels


@torch.no_grad()
def eval_loss_acc(model, loader, device):
    was_training = model.training
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        count = batch["labels"].numel()
        total_loss += out.loss.item() * count
        preds = out.logits.argmax(dim=-1)
        total_correct += (preds == batch["labels"]).sum().item()
        total += count
    model.train(was_training)
    return total_loss / total, total_correct / total


def compute_top_eigenvalue(model, loader, device, max_iter=15):
    """Compute lambda_max of the Hessian using Pearlmutter's power iteration with HVPs."""
    model.eval()
    batches = []
    # Collect a few batches for Hessian-vector product
    for i, batch in enumerate(loader):
        batches.append({k: v.to(device) for k, v in batch.items()})
        if i >= 4:  # 5 batches is enough to estimate curvature robustly
            break

    params = [p for p in model.parameters() if p.requires_grad]
    # Initialize random unit vector v
    v = [torch.randn_like(p) for p in params]
    total_norm = torch.sqrt(sum(torch.sum(vi ** 2) for vi in v))
    v = [vi / (total_norm + 1e-12) for vi in v]

    eigenval = 0.0
    # Enable math SDPA because PyTorch's Flash/mem-efficient attention kernels do not implement 2nd derivatives
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
        for it in range(max_iter):
            Hv = [torch.zeros_like(p) for p in params]
            for batch in batches:
                model.zero_grad()
                out = model(**batch)
                loss = out.loss
                grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
                # inner product <grads, v>
                grad_v = sum(torch.sum(g * vi) for g, vi in zip(grads, v))
                hvp = torch.autograd.grad(grad_v, params, retain_graph=False)
                for hv_i, h in zip(Hv, hvp):
                    hv_i.add_(h.detach() / len(batches))

            # Rayleigh quotient
            eigenval = sum(torch.sum(vi * hvi) for vi, hvi in zip(v, Hv)).item()
            hv_norm = torch.sqrt(sum(torch.sum(hvi ** 2) for hvi in Hv))
            v = [hvi / (hv_norm + 1e-12) for hvi in Hv]

    model.zero_grad()
    return float(eigenval)


def measure_noise_robustness(model, loader, device, sigmas=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 2e-2, 5e-2]):
    """Evaluate loss and accuracy under increasing Gaussian weight perturbation."""
    orig_weights = [p.data.clone() for p in model.parameters()]
    results = []
    for sigma in sigmas:
        for p, w0 in zip(model.parameters(), orig_weights):
            noise = torch.randn_like(w0) * sigma
            p.data.copy_(w0 + noise)
        loss, acc = eval_loss_acc(model, loader, device)
        results.append({"sigma": sigma, "loss": loss, "accuracy": acc})
    # Restore original weights
    for p, w0 in zip(model.parameters(), orig_weights):
        p.data.copy_(w0)
    return results


def evaluate_1d_slice(model, loader, device, alphas=np.linspace(-0.5, 0.5, 21)):
    """1D loss cross-section along a filter-normalized random direction vector."""
    orig_weights = [p.data.clone() for p in model.parameters()]
    # Generate filter-normalized random direction
    directions = []
    for p in orig_weights:
        d = torch.randn_like(p)
        # Normalize direction relative to weight norm
        norm_w = p.norm()
        norm_d = d.norm() + 1e-12
        directions.append(d * (norm_w / norm_d))

    losses, accs = [], []
    for alpha in alphas:
        for p, w0, d in zip(model.parameters(), orig_weights, directions):
            p.data.copy_(w0 + alpha * d)
        l, a = eval_loss_acc(model, loader, device)
        losses.append(l)
        accs.append(a)

    for p, w0 in zip(model.parameters(), orig_weights):
        p.data.copy_(w0)
    return list(alphas), losses, accs


def evaluate_interpolation(model1, model2, loader, device, alphas=np.linspace(-0.2, 1.2, 29)):
    """Evaluate along the line connecting model1 and model2: w(alpha) = (1-alpha)*w1 + alpha*w2."""
    w1 = [p.data.clone() for p in model1.parameters()]
    w2 = [p.data.clone() for p in model2.parameters()]

    losses, accs = [], []
    for alpha in alphas:
        for p, p1, p2 in zip(model1.parameters(), w1, w2):
            p.data.copy_((1.0 - alpha) * p1 + alpha * p2)
        l, a = eval_loss_acc(model1, loader, device)
        losses.append(l)
        accs.append(a)

    for p, p1 in zip(model1.parameters(), w1):
        p.data.copy_(p1)
    return list(alphas), losses, accs


def plot_robustness(robustness_data, out_path, title="Weight Perturbation Robustness (Flatness)"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sigmas = [r["sigma"] for r in robustness_data]
    losses = [r["loss"] for r in robustness_data]
    accs = [r["accuracy"] for r in robustness_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(sigmas, losses, marker="o", color="crimson")
    ax1.set(xscale="log", xlabel="Noise std (sigma)", ylabel="Validation Loss", title="Loss vs. Weight Noise")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(sigmas, accs, marker="s", color="royalblue")
    ax2.set(xscale="log", xlabel="Noise std (sigma)", ylabel="Validation Accuracy", title="Accuracy vs. Weight Noise")
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_slice_1d(alphas, losses, out_path, title="1D Loss Landscape Profile"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(alphas, losses, marker="o", color="darkorange", linewidth=2)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.7, label="Minimum (alpha=0)")
    ax.set(xlabel="Perturbation step alpha", ylabel="Validation Loss", title=title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_interpolation_curve(alphas, losses, accs, out_path, label1="Checkpoint 1", label2="Checkpoint 2"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(alphas, losses, marker="o", color="purple", linewidth=2)
    ax1.axvline(0, color="blue", linestyle="--", label=f"alpha=0 ({label1})")
    ax1.axvline(1, color="red", linestyle="--", label=f"alpha=1 ({label2})")
    ax1.set(xlabel="Interpolation factor alpha", ylabel="Validation Loss", title="Loss Landscape Interpolation")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(alphas, accs, marker="s", color="teal", linewidth=2)
    ax2.axvline(0, color="blue", linestyle="--", label=f"alpha=0 ({label1})")
    ax2.axvline(1, color="red", linestyle="--", label=f"alpha=1 ({label2})")
    ax2.set(xlabel="Interpolation factor alpha", ylabel="Validation Accuracy", title="Accuracy Interpolation")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_landscape_analysis(checkpoint_dir, task, checkpoint2_dir=None, output_dir=None, eval_samples=500, power_iter=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = output_dir or checkpoint_dir
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading checkpoint from {checkpoint_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    val_loader, num_labels = load_val_data(task, tokenizer, max_len=128, batch_size=32, eval_samples=eval_samples)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir).to(device)

    base_loss, base_acc = eval_loss_acc(model, val_loader, device)
    print(f"Checkpoint evaluated: Val Loss = {base_loss:.4f}, Val Accuracy = {base_acc:.4f}")

    # 1. Top Hessian eigenvalue
    print(f"Computing top Hessian eigenvalue lambda_max with {power_iter} power iterations...")
    lambda_max = compute_top_eigenvalue(model, val_loader, device, max_iter=power_iter)
    print(f"Top Hessian Eigenvalue (sharpness lambda_max): {lambda_max:.4f}")

    # 2. Perturbation robustness
    print("Measuring Gaussian weight noise robustness...")
    robustness = measure_noise_robustness(model, val_loader, device)
    robustness_plot_path = os.path.join(out_dir, "robustness_perturbation.png")
    plot_robustness(robustness, robustness_plot_path)

    # 3. 1D Loss slice
    print("Evaluating 1D loss landscape cross-section...")
    alphas, slice_losses, slice_accs = evaluate_1d_slice(model, val_loader, device)
    slice_plot_path = os.path.join(out_dir, "loss_landscape_1d.png")
    plot_slice_1d(alphas, slice_losses, slice_plot_path)

    summary = {
        "checkpoint": checkpoint_dir,
        "task": task,
        "eval_samples": eval_samples,
        "base_loss": base_loss,
        "base_accuracy": base_acc,
        "lambda_max": lambda_max,
        "robustness_curve": robustness,
        "slice_1d": {"alphas": alphas, "losses": slice_losses, "accuracies": slice_accs},
    }

    # 4. Optional 2-checkpoint interpolation
    if checkpoint2_dir and os.path.exists(checkpoint2_dir):
        print(f"Loading second checkpoint from {checkpoint2_dir} for interpolation...")
        model2 = AutoModelForSequenceClassification.from_pretrained(checkpoint2_dir).to(device)
        interp_alphas, interp_losses, interp_accs = evaluate_interpolation(model, model2, val_loader, device)
        interp_plot_path = os.path.join(out_dir, "loss_interpolation.png")
        plot_interpolation_curve(interp_alphas, interp_losses, interp_accs, interp_plot_path,
                                label1=os.path.basename(os.path.dirname(checkpoint_dir)),
                                label2=os.path.basename(os.path.dirname(checkpoint2_dir)))
        summary["interpolation"] = {
            "checkpoint2": checkpoint2_dir,
            "alphas": interp_alphas,
            "losses": interp_losses,
            "accuracies": interp_accs,
        }

    summary_json_path = os.path.join(out_dir, "landscape_metrics.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Analysis complete! Artifacts written to {out_dir}")
    return summary


def main():
    args = parse_args()
    run_landscape_analysis(
        checkpoint_dir=args.checkpoint,
        task=args.task,
        checkpoint2_dir=args.checkpoint2,
        output_dir=args.output_dir,
        eval_samples=args.eval_samples,
        power_iter=args.power_iter,
    )


if __name__ == "__main__":
    main()

