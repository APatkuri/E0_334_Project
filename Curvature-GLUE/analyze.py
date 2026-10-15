"""Summarize the GLUE runs written by ../sweep.py (results/glue/*.json).

    python analyze.py                  # per task and method: dev score, train-dev gap, lambda_max, tr(H)
    python analyze.py --plot           # also lambda_max vs dev score, one panel per task

Tables use the setting with the most seeds for each task and method (the `final` runs once they
exist), as mean +- std over seeds. The rank correlation uses every finished run of a task.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METHODS = ["adamw", "sam", "asam", "gsam", "wsam", "eigensam", "xsam", "entropysgd"]


def load(results):
    runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(results, "*.json")))]
    return [r for r in runs if not r["diverged"]]


def setting(r):
    """Everything that defines a run except the seed."""
    return tuple(sorted((k, str(v)) for k, v in r.items()
                        if k in ("lr", "rho", "alpha", "gamma", "batch_size", "epochs", "budget", "schedule")))


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1]) if len(x) > 2 else float("nan")


def fmt(vals, digits=3):
    return f"{np.mean(vals):.{digits}f} ± {np.std(vals):.{digits}f}" if vals else "-"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=os.path.join(ROOT, "results", "glue"))
    p.add_argument("--budget", default="grads", choices=["grads", "steps"])
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    runs = load(args.results)
    by_task = defaultdict(list)
    for r in runs:
        by_task[f"{r['task']}-f{r['train_frac']:g}"].append(r)

    for task, rs in sorted(by_task.items()):
        primary = rs[0]["primary"]
        print(f"\n== {task} (dev {primary}, budget {args.budget}) ==")
        print(f"{'method':11s} {'seeds':>5s}  {'dev':>15s}  {'gap':>15s}  {'lambda_max':>17s}  {'tr(H)':>17s}")
        main_runs = [r for r in rs if r["budget"] == args.budget and r["schedule"] == "linear"]
        for m in METHODS:
            groups = defaultdict(list)
            for r in main_runs:
                if r["optim"] == m:
                    groups[setting(r)].append(r)
            if not groups:
                continue
            best = max(groups.values(), key=len)
            col = lambda key: [r[key] for r in best if r.get(key) is not None]
            print(f"{m:11s} {len(best):5d}  {fmt([r['dev'][primary] for r in best]):>15s}  {fmt(col('gap')):>15s}"
                  f"  {fmt(col('lmax'), 2):>17s}  {fmt(col('trace'), 1):>17s}")
        x = [r["lmax"] for r in rs]
        y = [r["dev"][primary] for r in rs]
        print(f"Spearman(lambda_max, dev) over all {len(rs)} runs: {spearman(x, y):+.3f}")
        if any(r.get("trace") is not None for r in rs):
            t = [(r["trace"], r["dev"][primary]) for r in rs if r.get("trace") is not None]
            print(f"Spearman(tr(H), dev) over {len(t)} runs: {spearman(*zip(*t)):+.3f}")

    kaur = [r for r in runs if r["schedule"] == "constant"]
    if kaur:
        print("\n== Kaur-style test: SAM rho at small vs large batch, constant lr, to 99% train acc ==")
        table = defaultdict(list)
        for r in kaur:
            table[(r["lr"], r["batch_size"], r["rho"] if r["optim"] == "sam" else 0.0)].append(r)
        for (lr, bs, rho), rs in sorted(table.items()):
            primary = rs[0]["primary"]
            print(f"lr {lr:g} batch {bs:4d} rho {rho:<5g} seeds {len(rs)}  dev {fmt([r['dev'][primary] for r in rs])}"
                  f"  lambda_max {fmt([r['lmax'] for r in rs], 2)}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(by_task), figsize=(4 * len(by_task), 3.5), squeeze=False)
        for ax, (task, rs) in zip(axes[0], sorted(by_task.items())):
            for m in METHODS:
                pts = [(r["lmax"], r["dev"][r["primary"]]) for r in rs if r["optim"] == m]
                if pts:
                    ax.scatter(*zip(*pts), s=14, label=m)
            ax.set(title=task, xlabel="λmax (final)", ylabel=f"dev {rs[0]['primary']}")
        axes[0][-1].legend(fontsize=7)
        fig.tight_layout()
        out = os.path.join(args.results, "lmax_vs_dev.png")
        fig.savefig(out, dpi=150)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
