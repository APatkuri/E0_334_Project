"""Tables of the SAM-variant image experiments next to the papers' numbers.

    python analyze_image.py            # reads ../results/img/*.json

Each cell is "ours mean +- std (n runs) | paper", final test accuracy (WSAM: top-1 error, as the
paper reports it). Missing runs show as "-".
"""
import argparse
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = ["resnet18", "resnet50", "wrn28_10", "alexnet"]
METHODS = {"sgd": "SGD", "sam": "SAM", "gsam": "GSAM", "wsam": "WSAM", "eigensam": "Eigen-SAM", "xsam": "X-SAM"}

# X-SAM Table 1: accuracy (%) for ResNet-18, ResNet-50, WRN-28-10, AlexNet (None: did not converge)
XSAM_T1 = {
    "cifar10": {"sam": [93.64, 92.33, 95.01, 88.66], "wsam": [93.01, 91.68, 94.23, 90.08],
                "gsam": [93.80, 92.16, 94.67, 88.59], "eigensam": [93.82, 92.57, 95.11, 88.79],
                "xsam": [94.71, 93.76, 95.53, 90.73]},
    "cifar100": {"sam": [74.00, 72.01, 77.96, 60.26], "wsam": [73.65, 71.04, 77.19, 65.90],
                 "gsam": [73.55, 71.61, 78.15, 60.14], "eigensam": [73.74, 71.90, 78.20, 60.49],
                 "xsam": [74.10, 74.08, 78.23, 68.82]},
    "fashion_mnist": {"sam": [95.22, 94.46, 95.56, 94.34], "wsam": [95.32, 94.57, 95.40, None],
                      "gsam": [95.39, 94.94, 95.68, 94.61], "eigensam": [95.36, 94.76, 95.61, None],
                      "xsam": [95.52, 95.16, 95.53, 94.75]}}
# Eigen-SAM Table 1: accuracy (%) for CIFAR-10, CIFAR-100, Fashion-MNIST, SVHN
EIGEN_T1 = {
    "resnet18": {"sgd": [94.8, 74.6, 94.9, 96.1], "sam": [95.5, 77.4, 95.4, 96.3], "eigensam": [95.9, 78.3, 95.6, 96.5]},
    "resnet50": {"sgd": [95.0, 76.6, 94.8, 96.1], "sam": [95.6, 79.0, 95.4, 96.4], "eigensam": [96.2, 79.7, 95.7, 96.6]},
    "wrn28_10": {"sgd": [95.7, 79.8, 95.1, 96.2], "sam": [96.5, 82.0, 95.6, 96.4], "eigensam": [96.8, 82.8, 95.9, 96.7]}}
EIGEN_DATASETS = ["cifar10", "cifar100", "fashion_mnist", "svhn"]
# Eigen-SAM Table 2: ViT-B/16 fine-tuning accuracy (%) on CIFAR-10, CIFAR-100
EIGEN_T2 = {"sgd": [98.0, 88.6], "sam": [98.4, 89.5], "eigensam": [98.5, 89.8]}
# WSAM Table 2: top-1 error (%) on CIFAR-10, CIFAR-100
WSAM_T2 = {
    "resnet18": {"sgd": [4.32, 20.51], "sam": [3.68, 19.97], "gsam": [3.71, 19.84], "wsam": [3.62, 19.42]},
    "wrn28_10": {"sgd": [3.73, 19.27], "sam": [2.94, 16.49], "gsam": [2.91, 16.36], "wsam": [2.74, 16.33]}}
# WSAM Table 6: accuracy (%) with 20, 40, 60, 80% symmetric label noise
WSAM_T6 = {"sgd": [88.06, 84.11, 79.15, 69.07], "sam": [94.99, 93.28, 88.32, 77.57],
           "gsam": [95.11, 93.25, 89.90, 79.09], "wsam": [95.18, 93.33, 89.95, 78.30]}
# WSAM Table 9: ResNet-18 on CIFAR-10, (training loss, accuracy, lambda_max)
WSAM_T9 = {"sgd": (0.0026, 0.9575, 62.58), "sam": (0.0339, 0.9622, 22.67), "wsam": (0.0089, 0.9654, 23.97)}


def load(path):
    runs = []
    for f in glob.glob(os.path.join(path, "*.json")):
        r = json.load(open(f))
        if "curve" in r:  # also runs that diverged before their first evaluation (empty curve)
            r["method"] = r["optim"]
            runs.append(r)
    return runs


def pick(runs, **cond):
    return [r for r in runs if all(r.get(k) == v for k, v in cond.items())]


def cell(rs, paper, error=False):
    ok = [r for r in rs if not r["diverged"]]
    vals = np.array([100 * ((1 - r["test_acc"]) if error else r["test_acc"]) for r in ok])
    ours = f"{vals.mean():6.2f} +- {vals.std():4.2f} ({len(vals)})" if len(vals) else f"{'-':>17s}"
    if len(rs) > len(ok):
        ours += f" [{len(rs) - len(ok)} diverged]"
    return f"{ours} | {'n/c' if paper is None else paper}"


def header(title):
    print(f"\n== {title} ==")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=os.path.join(HERE, "..", "results", "img"))
    runs = load(p.parse_args().results)
    xe = [r for r in runs if r["weight_decay"] == 5e-5 and r["model"] != "vit_b_16"]  # X-SAM / Eigen-SAM protocol
    ws = [r for r in runs if r["weight_decay"] == 1e-3]                                   # WSAM protocol

    for d, table in XSAM_T1.items():
        header(f"X-SAM Table 1, {d}: accuracy (%), ours | paper")
        for m, vals in table.items():
            print(f"{METHODS[m]:13s}" + "  ".join(f"{mod}: {cell(pick(xe, dataset=d, model=mod, method=m, label_noise=0.0), v)}"
                                                  for mod, v in zip(MODELS, vals)))

    for mod, table in EIGEN_T1.items():
        header(f"Eigen-SAM Table 1, {mod}: accuracy (%), ours | paper")
        for m, vals in table.items():
            print(f"{METHODS[m]:13s}" + "  ".join(f"{d}: {cell(pick(xe, dataset=d, model=mod, method=m), v)}"
                                                  for d, v in zip(EIGEN_DATASETS, vals)))

    header("Eigen-SAM Table 2, ViT-B/16: accuracy (%), ours | paper")
    for m, vals in EIGEN_T2.items():
        print(f"{METHODS[m]:13s}" + "  ".join(f"{d}: {cell(pick(runs, dataset=d, model='vit_b_16', method=m), v)}"
                                              for d, v in zip(["cifar10", "cifar100"], vals)))

    for mod, table in WSAM_T2.items():
        header(f"WSAM Table 2, {mod}: top-1 error (%), ours | paper")
        for m, vals in table.items():
            print(f"{METHODS[m]:13s}" + "  ".join(
                f"{d}: {cell(pick(ws, dataset=d, model=mod, method=m, label_noise=0.0), v, error=True)}"
                for d, v in zip(["cifar10", "cifar100"], vals)))

    header("WSAM Table 6, ResNet-18 CIFAR-10 with label noise: accuracy (%), ours | paper")
    for m, vals in WSAM_T6.items():
        print(f"{METHODS[m]:13s}" + "  ".join(f"{int(100 * n)}%: {cell(pick(ws, label_noise=n, method=m), v)}"
                                              for n, v in zip([0.2, 0.4, 0.6, 0.8], vals)))

    header("WSAM Table 9, ResNet-18 CIFAR-10 minima: train loss / accuracy / lambda_max, ours | paper")
    for m, paper in WSAM_T9.items():
        rs = [r for r in pick(ws, dataset="cifar10", model="resnet18", method=m, label_noise=0.0) if "lmax" in r]
        if rs:
            loss, acc, lmax = (np.mean([r[k] for r in rs]) for k in ("train_loss", "test_acc", "lmax"))
            print(f"{METHODS[m]:13s} {loss:.4f} / {acc:.4f} / {lmax:6.2f} ({len(rs)}) | "
                  f"{paper[0]} / {paper[1]} / {paper[2]}")
        else:
            print(f"{METHODS[m]:13s} - | {paper[0]} / {paper[1]} / {paper[2]}")

    minutes = [r["minutes"] for r in runs if "minutes" in r and r.get("steps_done")]
    if minutes:
        print(f"\n{len(runs)} runs, {sum(minutes) / 60:.1f} GPU-hours so far")


if __name__ == "__main__":
    main()
