"""Run the project's experiments as a queue of jobs over several GPUs.

    python sweep.py PHASE --gpus 3 4 --per_gpu 2      # run a phase
    python sweep.py PHASE --dry_run                   # list its jobs

Every run writes results/<group>/<name>.json(l) and logs to results/logs/<phase>/<name>.log.
Finished runs are skipped, so a phase can be stopped (Ctrl-C) and restarted at any time.

Our contribution: BERT-base on GLUE (Curvature-GLUE/run_glue.py). The stages follow the tuning
protocol of the SAM-variant papers: tune the base optimizer's lr first and keep it fixed (GSAM
App. B.1), then tune SAM's rho, and reuse that rho for GSAM, WSAM, Eigen-SAM and X-SAM, which tune
only their own knob (WSAM Sec. 6.2-6.3, Eigen-SAM App. E, X-SAM Sec. 6.1). Run in order:
    glue-smoke     every optimizer once, very short
    glue-lr        AdamW lr
    glue-rho       SAM and ASAM rho, Entropy-Adam gamma      (needs glue-lr)
    glue-variants  GSAM, WSAM, Eigen-SAM, X-SAM              (needs glue-rho)
    glue-final     best setting of every method, 5 seeds     (needs glue-variants)
    glue-kaur      Kaur et al.'s SAM test transferred to GLUE (independent)

Reproduction of Kaur et al. (2023), Maximum-Hessian/train_vision.py, any order:
    vision-smoke, vision-3.1 (Figs. 1-3), vision-3.1-fullbatch (Fig. 3, batch 50000; very slow),
    vision-3.2 (Figs. 4-5), vision-3.3 (Figs. 6-7), vision-3.4 (Figs. 8-9), vision-3.5 (Figs. 10-11),
    vision-a3 (Fig. 14)
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
GLUE_SCRIPT = os.path.join(ROOT, "Curvature-GLUE", "run_glue.py")
VISION_SCRIPT = os.path.join(ROOT, "Maximum-Hessian", "train_vision.py")

# ---- GLUE: settings and where they come from -------------------------------------------
TASKS = [("rte", 1.0), ("mrpc", 1.0), ("cola", 1.0), ("sst2", 0.05), ("sst2", 0.2)]  # our design
SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 3                                     # Devlin et al. (2019) Sec. 4.1: batch 32, 3 epochs
LRS = [2e-5, 3e-5, 4e-5, 5e-5]                 # Devlin et al. Sec. 4.1, best on dev
SAM_RHOS = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3]   # Bahri et al. (2022) Sec. 4.1
SAM_ARGS = {"ascent_frac": 0.25}               # Bahri et al. Alg. 1: ascent on 1/4 of the batch
ASAM_RHOS = [10 * r for r in SAM_RHOS]         # GSAM App. B.1: ASAM uses 10x SAM's rho (Kwon et al.)
GSAM_ALPHAS = [0.1, 0.2, 0.3]                  # GSAM App. B.1
GSAM_ARGS = {"rho_min": 0.0}                   # GSAM Table 4: rho follows the lr; rho_min = 0 in most AdamW rows
WSAM_GAMMAS = [round(0.80 + 0.02 * i, 2) for i in range(10)]  # WSAM Sec. 6.2: 0.80 to 0.98, step 0.02
EIGENSAM_ALPHAS = [0.05, 0.1, 0.2]             # Eigen-SAM App. E (p = 100, q = 5 are run_glue.py defaults)
XSAM_ALPHAS = [0.2]                            # X-SAM Sec. 6.1: alpha = 0.2, not tuned
ENTROPY_GAMMAS = [0.003, 0.01, 0.03]           # around Entropy-Adam's gamma = 0.01 (Chaudhari et al. Sec. 5.4.2)
ENTROPY_ARGS = {"sgld_steps": 5, "beta1": 0.5, "noise": 1e-4, "mu_alpha": 0.75}  # Sec. 5.4.2 and 4.2
# Kaur et al. Sec. 3.3 transferred to GLUE: constant lr, train to 99% train accuracy, rho from 0 to
# 0.1, two learning rates, 4 seeds, plain SAM (full-batch ascent). Our choices: AdamW, BERT learning
# rates, and batch 256 as the large batch (Kaur's 10% of the data does not fit without accumulation).
KAUR_TASK = ("sst2", 0.2)
KAUR_RHOS = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1]
KAUR_LRS = [2e-5, 5e-5]
KAUR_BATCHES = [32, 256]
KAUR_SEEDS = [0, 1, 2, 3]
KAUR_ARGS = {"schedule": "constant", "budget": "steps", "epochs": 30, "target_train_acc": 0.99,
             "train_eval_examples": 100000, "evals": 1}


def fmt(v):
    return f"{v:g}" if isinstance(v, float) else str(v)


def glue_job(task, frac, optim, seed=0, **flags):
    flags = {"epochs": EPOCHS, "batch_size": 32, **flags}
    name = f"{task}-f{frac:g}_{optim}_" + "_".join(f"{k}{fmt(v)}" for k, v in sorted(flags.items())) + f"_s{seed}"
    argv = ["--task", task, "--train_frac", str(frac), "--optim", optim, "--seed", str(seed)]
    for k, v in flags.items():
        argv += [f"--{k}", str(v)]
    return name, GLUE_SCRIPT, argv, "glue", ".json"


def pick(candidates, stage):
    """Best value among (value, optim, flags, task, frac) candidates by dev score of their seed-0 run."""
    scored, missing = [], 0
    for value, optim, flags, task, frac in candidates:
        path = os.path.join(RESULTS, "glue", glue_job(task, frac, optim, **flags)[0] + ".json")
        if not os.path.exists(path):
            missing += 1
            continue
        r = json.load(open(path))
        if not r["diverged"]:
            scored.append((r["dev"][r["primary"]], value))
    if missing or not scored:
        sys.exit(f"{missing} of {len(candidates)} {stage} runs for {task}-f{frac:g} {optim} are missing: "
                 f"run `python sweep.py {stage}` first")
    return max(scored)[1]


def lr_stage(t, f):
    return {"adamw": [(lr, "adamw", {"lr": lr}, t, f) for lr in LRS]}


def rho_stage(t, f):
    lr = pick(lr_stage(t, f)["adamw"], "glue-lr")
    return {"sam": [(r, "sam", {"lr": lr, "rho": r, **SAM_ARGS}, t, f) for r in SAM_RHOS],
            "asam": [(r, "asam", {"lr": lr, "rho": r}, t, f) for r in ASAM_RHOS],
            "entropysgd": [(g, "entropysgd", {"lr": lr, "gamma": g, **ENTROPY_ARGS}, t, f) for g in ENTROPY_GAMMAS]}


def variant_stage(t, f):
    lr = pick(lr_stage(t, f)["adamw"], "glue-lr")
    rho = pick(rho_stage(t, f)["sam"], "glue-rho")
    base = {"lr": lr, "rho": rho}
    return {"gsam": [(a, "gsam", {**base, "alpha": a, **GSAM_ARGS}, t, f) for a in GSAM_ALPHAS],
            "wsam": [(g, "wsam", {**base, "gamma": g}, t, f) for g in WSAM_GAMMAS],
            "eigensam": [(a, "eigensam", {**base, "alpha": a}, t, f) for a in EIGENSAM_ALPHAS],
            "xsam": [(a, "xsam", {**base, "alpha": a}, t, f) for a in XSAM_ALPHAS]}


def stage_jobs(stage):
    return [glue_job(t, f, o, **flags) for t, f in TASKS for cands in stage(t, f).values()
            for _, o, flags, _, _ in cands]


def final_jobs():
    jobs = []
    for t, f in TASKS:
        stages = {**lr_stage(t, f), **rho_stage(t, f), **variant_stage(t, f)}
        for optim, cands in stages.items():
            best = pick(cands, "glue-lr" if optim == "adamw" else "glue-rho" if optim in ("sam", "asam", "entropysgd")
                        else "glue-variants")
            flags = next(fl for v, _, fl, _, _ in cands if v == best)
            print(f"{t}-f{f:g} {optim}: {flags}")
            jobs += [glue_job(t, f, optim, seed=s, **flags) for s in SEEDS]
    return jobs


def kaur_glue_jobs():
    t, f = KAUR_TASK
    return [glue_job(t, f, "adamw" if rho == 0 else "sam", seed=s, lr=lr, batch_size=bs,
                     **({} if rho == 0 else {"rho": rho}), **KAUR_ARGS)
            for lr in KAUR_LRS for bs in KAUR_BATCHES for rho in KAUR_RHOS for s in KAUR_SEEDS]


def glue_smoke_jobs():
    quick = {"epochs": 0.5, "evals": 1, "track_points": 2, "track_iters": 3, "lmax_iters": 3,
             "trace_samples": 2, "curv_examples": 128, "track_examples": 64}
    jobs = [glue_job("rte", 1.0, o, **quick, **extra) for o, extra in [
        ("adamw", {}), ("sam", SAM_ARGS), ("asam", {"rho": 0.5}), ("gsam", GSAM_ARGS), ("wsam", {}),
        ("eigensam", {"eig_every": 10}), ("xsam", {"eig_every": 10}), ("entropysgd", ENTROPY_ARGS)]]
    jobs.append(glue_job("rte", 1.0, "sam", **{**quick, **KAUR_ARGS, "epochs": 1, "target_train_acc": 0.5}))
    return [(n, s, a, "glue-smoke", e) for n, s, a, _, e in jobs]


# ---- Kaur et al. (2023) vision reproduction ------------------------------------------------
# All runs: plain SGD at a constant lr until 99% train accuracy (Sec. 3). Grids not stated in the
# paper (marked "ours") are read off the figure axes.
VISION_LRS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]   # Figs. 1, 3 axis 1e-4..1e-1 (values ours)
VISION_RHOS = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1]          # Figs. 6, 7, 14 axis 0..0.1
BN_LRS = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]                   # Figs. 10, 11 axis 1e-3..1e-1 (values ours)


def vision_job(seed, **flags):
    name = "_".join(f"{k}{fmt(v)}" for k, v in sorted(flags.items()) if not isinstance(v, bool))
    name += "".join(f"_{k}" for k, v in sorted(flags.items()) if v is True) + f"_s{seed}"
    argv = ["--seed", str(seed)]
    for k, v in flags.items():
        argv += [f"--{k}"] if v is True else [f"--{k}", str(v)]
    return name, VISION_SCRIPT, argv, "vision", ".jsonl"


def vision_phase(phase):
    if phase == "vision-smoke":
        jobs = [vision_job(0, dataset="fashion_mnist", model="mlp", lr=0.1, batch_size=100, max_epochs=2,
                           lmax_examples=1000, lmax_every=200),
                vision_job(0, model="vgg11_bn", lr=0.01, batch_size=100, rho=0.05, max_epochs=1,
                           lmax_examples=1000, between_iterates=True)]
        return [(n, s, a, "vision-smoke", e) for n, s, a, _, e in jobs]
    if phase == "vision-3.1":  # Figs. 1-3: batch 100 and 5000 (Fig. 1), plus 200 and 500 (Fig. 3); 5 seeds
        return [vision_job(s, model="vgg11", batch_size=bs, lr=lr)
                for bs in (100, 200, 500, 5000) for lr in VISION_LRS for s in range(5)]
    if phase == "vision-3.1-fullbatch":  # Fig. 3: full-batch GD
        return [vision_job(s, model="vgg11", batch_size=50000, lr=lr, max_epochs=100000)
                for lr in VISION_LRS for s in range(5)]
    if phase == "vision-3.2":  # Figs. 4-5: batch/lr = 10000 and 20000; lambda_max curve for 20000; 5 seeds
        pairs = [(bs, bs / a, a == 20000) for a in (10000, 20000) for bs in (100, 200, 500)]
        return [vision_job(s, model="vgg11", batch_size=bs, lr=lr, **({"lmax_every": 500} if curve else {}))
                for bs, lr, curve in pairs for s in range(5)]
    if phase == "vision-3.3":  # Figs. 6-7: SAM at lr 0.005 and 0.05, batch 100 and 5000; 4 seeds
        return [vision_job(s, model="vgg11", batch_size=bs, lr=lr, rho=rho)
                for lr in (0.005, 0.05) for bs in (100, 5000) for rho in VISION_RHOS for s in range(4)]
    if phase == "vision-3.4":  # Figs. 8-9: MLP, dropout 0..0.7, lr 0.01 and 0.1; large batch 5000 (ours); 3 seeds
        return [vision_job(s, dataset="fashion_mnist", model="mlp", batch_size=bs, lr=lr, dropout=p)
                for bs in (100, 5000) for lr in (0.01, 0.1) for p in [i / 10 for i in range(8)] for s in range(3)]
    if phase == "vision-3.5":  # Figs. 10-11: VGG11 with and without BN, batch 100 and 5000; 4 seeds
        return [vision_job(s, model=m, batch_size=bs, lr=lr, between_iterates=True)
                for m in ("vgg11", "vgg11_bn") for bs in (100, 5000) for lr in BN_LRS for s in range(4)]
    if phase == "vision-a3":  # Fig. 14: VGG11-BN with SAM, lr 0.005 and 0.05, batch 100 and 5000; 1 seed
        return [vision_job(0, model="vgg11_bn", batch_size=bs, lr=lr, rho=rho, between_iterates=True)
                for lr in (0.005, 0.05) for bs in (100, 5000) for rho in VISION_RHOS]
    raise KeyError(phase)


PHASES = {
    "glue-smoke": glue_smoke_jobs,
    "glue-lr": lambda: stage_jobs(lr_stage),
    "glue-rho": lambda: stage_jobs(rho_stage),
    "glue-variants": lambda: stage_jobs(variant_stage),
    "glue-final": final_jobs,
    "glue-kaur": kaur_glue_jobs,
    **{p: (lambda p=p: vision_phase(p)) for p in ("vision-smoke", "vision-3.1", "vision-3.1-fullbatch", "vision-3.2",
                                                  "vision-3.3", "vision-3.4", "vision-3.5", "vision-a3")},
}


# ---- queue ------------------------------------------------------------------------------

def run_queue(jobs, log_dir, gpus, per_gpu):
    os.makedirs(log_dir, exist_ok=True)
    out = lambda job: os.path.join(RESULTS, job[3], job[0] + job[4])
    todo = [j for j in jobs if not os.path.exists(out(j))]
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} already done, {len(todo)} to run "
          f"on GPUs {gpus} ({per_gpu} per GPU)", flush=True)
    slots = {(g, i): None for g in gpus for i in range(per_gpu)}  # slot -> (process, name, start)
    failed, done, total = [], 0, len(todo)
    try:
        while todo or any(slots.values()):
            for slot, cur in slots.items():
                if cur is not None:
                    proc, name, start = cur
                    if proc.poll() is None:
                        continue
                    done += 1
                    if proc.returncode != 0:
                        failed.append(name)
                    status = "ok" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
                    print(f"[{done}/{total}] {status} {name} ({(time.time() - start) / 60:.1f} min)", flush=True)
                    slots[slot] = None
                if todo:
                    job = todo.pop(0)
                    os.makedirs(os.path.dirname(out(job)), exist_ok=True)
                    cmd = [sys.executable, "-u", job[1], *job[2], "--out", out(job)]
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(slot[0]))
                    log = open(os.path.join(log_dir, job[0] + ".log"), "w")
                    slots[slot] = (subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                                    cwd=os.path.dirname(job[1])), job[0], time.time())
                    print(f"start {job[0]} on GPU {slot[0]}", flush=True)
            time.sleep(5)
    finally:  # Ctrl-C stops the running jobs too
        for cur in slots.values():
            if cur is not None and cur[0].poll() is None:
                cur[0].terminate()
    print(f"finished; {len(failed)} failed" + "".join(f"\n  {n}" for n in failed))
    if failed:
        print(f"logs are in {log_dir}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("phase", choices=PHASES)
    p.add_argument("--gpus", nargs="+", type=int, default=[0])
    p.add_argument("--per_gpu", type=int, default=1, help="jobs per GPU at once (a BERT job needs about 10 GB)")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    jobs = PHASES[args.phase]()
    if args.dry_run:
        for job in jobs:
            print(job[0])
        print(f"{len(jobs)} jobs")
        return
    run_queue(jobs, os.path.join(RESULTS, "logs", args.phase), args.gpus, args.per_gpu)


if __name__ == "__main__":
    main()
