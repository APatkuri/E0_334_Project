"""Aggregate GLUE runs produced by train_glue.py and plot validation-score comparisons.

    python analyze_glue.py
    python analyze_glue.py --results ../results/glue

The analysis uses the best validation checkpoint from each run; it does not use hidden GLUE test labels.
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SETTING_KEYS = ("lr", "weight_decay", "warmup_ratio", "batch_size", "epochs", "max_len", "sgld_steps",
                "gamma", "gamma_growth", "sgld_optim", "sgld_lr", "sgld_momentum", "noise", "alpha",
                "scale_by_gamma")


def load_runs(results_dir):
    paths = sorted(glob.glob(os.path.join(results_dir, "*", "metrics.json")))
    runs = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            run = json.load(f)
        if run.get("status") != "completed":
            continue
        run["run_dir"] = os.path.dirname(path)
        runs.append(run)
    return runs


def aggregate(runs):
    groups = defaultdict(list)
    for run in runs:
        args = run["args"]
        setting = tuple((key, args.get(key)) for key in SETTING_KEYS if key in args)
        key = (args["task"], args["optim"], args["model"], run["primary_metric"], setting)
        groups[key].append(run)

    rows = []
    for (task, optim, model, metric, setting), members in sorted(groups.items(), key=lambda item: str(item[0])):
        scores = [run["best_validation_metrics"][metric] for run in members]
        accuracies = [run["best_validation_metrics"]["accuracy"] for run in members]
        times = [run.get("elapsed_seconds", 0.0) for run in members]
        rows.append({
            "task": task,
            "optimizer": optim,
            "model": model,
            "primary_metric": metric,
            "setting": dict(setting),
            "runs": len(members),
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "mean_accuracy": float(np.mean(accuracies)),
            "std_accuracy": float(np.std(accuracies)),
            "mean_time_sec": float(np.mean(times)),
        })
    return rows


def write_summary(results_dir, rows):
    csv_path = os.path.join(results_dir, "summary.csv")
    fields = ("task", "optimizer", "model", "primary_metric", "setting", "runs",
              "mean_score", "std_score", "mean_accuracy", "std_accuracy", "mean_time_sec")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "setting": json.dumps(row["setting"], sort_keys=True)})
    json_path = os.path.join(results_dir, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
        f.write("\n")
    return csv_path, json_path


def plot_comparison(results_dir, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = sorted({row["task"] for row in rows})
    fig, axes = plt.subplots(1, len(tasks), figsize=(max(5, 4 * len(tasks)), 4), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        task_rows = [row for row in rows if row["task"] == task]
        labels = [
            f"{row['optimizer']} lr={row['setting'].get('lr', 'n/a')} n={row['runs']}"
            for row in task_rows
        ]
        scores = [row["mean_score"] for row in task_rows]
        errors = [row["std_score"] for row in task_rows]
        ax.bar(range(len(task_rows)), scores, yerr=errors, capsize=4)
        ax.set(title=task, ylabel=f"Best validation {task_rows[0]['primary_metric']}",
               xticks=range(len(task_rows)), xticklabels=labels)
        ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    path = os.path.join(results_dir, "optimizer_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=os.path.join(HERE, "..", "results", "glue"))
    args = parser.parse_args()
    results_dir = os.path.abspath(args.results)
    runs = load_runs(results_dir)
    if not runs:
        raise SystemExit(f"No completed GLUE runs found under {results_dir}")

    rows = aggregate(runs)
    print(f"{'task':8s} {'optimizer':12s} {'metric':20s} {'score (mean ± std)':>20s} {'accuracy':>16s} {'time (s)':>10s} {'n':>3s}")
    for row in rows:
        print(f"{row['task']:8s} {row['optimizer']:12s} {row['primary_metric']:20s} "
              f"{row['mean_score']:.4f} ± {row['std_score']:.4f} "
              f"{row['mean_accuracy']:.4f} ± {row['std_accuracy']:.4f} "
              f"{row['mean_time_sec']:10.1f} {row['runs']:3d}")

    csv_path, json_path = write_summary(results_dir, rows)
    plot_path = plot_comparison(results_dir, rows)
    print(f"Wrote {csv_path}, {json_path}, and {plot_path}")


if __name__ == "__main__":
    main()
