"""Plot results written by train_vision.py, in the layout of Kaur et al.'s figures.

Sweep mode (Figs. 1, 3, 6-11): test accuracy and lambda_max against one hyperparameter, one line
per value of --group, averaged over seeds.
    python plot_results.py results.jsonl --x lr --group batch_size --out fig1.png
    python plot_results.py results.jsonl --x rho --group batch_size --out fig6.png
    python plot_results.py results.jsonl --x dropout --group lr --out fig8.png

Scatter mode (Figs. 2, 4): lambda_max against test accuracy, one point per run.
    python plot_results.py results.jsonl --scatter --group batch_size --out fig2.png

--where key=value keeps only matching runs, e.g. --where model=vgg11 --where lr=0.005.
Several result files can be given, e.g. ../results/vision/*.jsonl from ../sweep.py.
"""
import argparse
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("results", nargs="+")
    p.add_argument("--x", default="lr", help="swept hyperparameter (lr, rho, dropout, batch_size)")
    p.add_argument("--group", default="batch_size", help="one line per value of this field")
    p.add_argument("--scatter", action="store_true", help="plot lambda_max vs test accuracy instead")
    p.add_argument("--where", action="append", default=[], help="filter, e.g. model=vgg11")
    p.add_argument("--out", default="figure.png")
    return p.parse_args()


def load(paths, where):
    runs = [json.loads(line) for path in paths for line in open(path) if line.strip()]
    runs = [r for r in runs if "lmax" in r]  # skip diverged runs
    for cond in where:
        key, value = cond.split("=", 1)
        runs = [r for r in runs if str(r.get(key)) == value]
    return runs


def mean(values):
    return sum(values) / len(values)


def main():
    args = parse_args()
    runs = load(args.results, args.where)
    groups = defaultdict(list)
    for r in runs:
        groups[r[args.group]].append(r)

    if args.scatter:
        fig, ax = plt.subplots(figsize=(5, 4))
        for g, rs in sorted(groups.items()):
            ax.scatter([r["lmax"] for r in rs], [r["test_acc"] for r in rs], label=f"{args.group} = {g}", s=18)
        ax.set_xscale("log")
        ax.set_xlabel(r"$\lambda_{\max}$")
        ax.set_ylabel("final test accuracy")
        ax.legend()
    else:
        fig, (ax_acc, ax_lmax) = plt.subplots(1, 2, figsize=(10, 4))
        for g, rs in sorted(groups.items()):
            by_x = defaultdict(list)
            for r in rs:
                by_x[r[args.x]].append(r)
            xs = sorted(by_x)
            ax_acc.plot(xs, [mean([r["test_acc"] for r in by_x[x]]) for x in xs], "o-", label=f"{args.group} = {g}")
            ax_lmax.plot(xs, [mean([r["lmax"] for r in by_x[x]]) for x in xs], "o-", label=f"{args.group} = {g}")
        if args.x == "lr":  # edge of stability: full-batch GD hovers at lambda_max = 2 / lr
            xs = sorted({r["lr"] for r in runs})
            ax_lmax.plot(xs, [2 / x for x in xs], "k--", label=r"$2/\eta$")
            for ax in (ax_acc, ax_lmax):
                ax.set_xscale("log")
        ax_lmax.set_yscale("log")
        ax_acc.set_ylabel("final test accuracy")
        ax_lmax.set_ylabel(r"$\lambda_{\max}$")
        for ax in (ax_acc, ax_lmax):
            ax.set_xlabel(args.x)
            ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out} ({len(runs)} runs)")


if __name__ == "__main__":
    main()
