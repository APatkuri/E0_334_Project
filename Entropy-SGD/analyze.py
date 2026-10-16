"""Summarize the Entropy-SGD reproduction against the paper.

    python analyze.py                      # Table 1 from ../results/esgd/*.json
    python analyze.py --plot               # also spectra (Figs. 1, 3) and curves (Figs. 4-6) as PNGs

Table 1 reports the metric at the end of training (test error %, or test perplexity / char loss),
mean +- std over seeds, next to the paper's value.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# Table 1 of the paper: (Entropy-SGD value, effective epochs, baseline value, epochs)
PAPER = {"mnistfc": (1.37, 120, 1.39, 66), "lenet": (0.50, 80, 0.51, 100), "allcnn": (7.81, 160, 7.71, 180),
         "ptb": (77.656, 25, 78.6, 55), "char": (1.217, 25, 1.226, 40)}


def load(results):
    runs = defaultdict(list)
    for path in glob.glob(os.path.join(results, "*.json")):
        r = json.load(open(path))
        if r.get("curve") and not r.get("diverged"):
            key = r.get("model") or r.get("task")
            runs[(key, r["optim"].startswith("entropy"))].append(r)
    return runs


def metric(r):
    if "test_err" in r:
        return 100 * r["test_err"]
    return r["test_ppl"] if r["task"] == "ptb" else r["test_loss"]  # char: the paper's number is a loss


def table(runs):
    print(f"{'model':8s} {'method':12s} {'ours (mean +- std)':>20s} {'n':>2s} {'eff. epochs':>11s} {'paper':>8s}")
    for key, (pe, pe_ep, pb, pb_ep) in PAPER.items():
        for entropy in (True, False):
            rs = runs.get((key, entropy), [])
            paper, paper_ep = (pe, pe_ep) if entropy else (pb, pb_ep)
            name = ("Entropy-" + ("Adam" if key == "char" else "SGD")) if entropy else \
                {"mnistfc": "Adam", "lenet": "Adam", "char": "Adam"}.get(key, "SGD")
            if not rs:
                print(f"{key:8s} {name:12s} {'-':>20s} {0:2d} {'-':>11s} {paper:8.3f}")
                continue
            v = np.array([metric(r) for r in rs])
            ep = rs[0]["curve"][-1]["effective_epochs"]
            print(f"{key:8s} {name:12s} {v.mean():12.3f} +- {v.std():5.3f} {len(v):2d} {ep:11d} {paper:8.3f}")


def plot(runs, results, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Figs. 4-6: test metric vs. effective epochs, seed 0
    keys = [k for k in PAPER if any(runs.get((k, e)) for e in (True, False))]
    if keys:
        fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3), squeeze=False)
        for ax, key in zip(axes[0], keys):
            for entropy in (False, True):
                for i, r in enumerate(sorted(runs.get((key, entropy), []), key=lambda r: r["seed"])):
                    ys = [100 * c["test_err"] if "test_err" in c else (c["test_ppl"] if key == "ptb" else c["test_loss"])
                          for c in r["curve"]]
                    ax.plot([c["effective_epochs"] for c in r["curve"]], ys, color="C1" if entropy else "C0",
                            alpha=0.6, label=("Entropy" if entropy else "baseline") if i == 0 else None)
            ax.set(title=key, xlabel="epochs x L", ylabel="test error (%)" if key in ("mnistfc", "lenet", "allcnn")
                   else "test perplexity" if key == "ptb" else "test loss")
            ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out, "esgd_curves.png"), dpi=120)

    # Figs. 1 and 3: eigenvalue histograms (log-count), seed 0 of each network
    spectra = sorted(glob.glob(os.path.join(results, "spectrum_*_s0.npy")))
    if spectra:
        fig, axes = plt.subplots(1, len(spectra), figsize=(4 * len(spectra), 3), squeeze=False)
        for ax, path in zip(axes[0], spectra):
            eig = np.load(path)
            ax.hist(eig, bins=200, log=True)
            ax.set(title=os.path.basename(path)[len("spectrum_model"):-len("_s0.npy")], xlabel="eigenvalue",
                   ylabel="count")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "esgd_spectra.png"), dpi=120)
    print(f"figures in {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=os.path.join(HERE, "..", "results", "esgd"))
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()
    runs = load(args.results)
    table(runs)
    for path in sorted(glob.glob(os.path.join(args.results, "spectrum_*.json"))):
        r = json.load(open(path))
        print(f"{r['model']:15s} seed {r['seed']}: {r['params']} params, {r['negative']} negative, "
              f"|lambda| <= 1e-2: {r['frac_abs_le_0.01']:.3f}, <= 1e-5: {r['frac_abs_le_1e-05']:.3f}, "
              f"max {r['max']:.3g}")
    if args.plot:
        plot(runs, args.results, args.results)


if __name__ == "__main__":
    main()
