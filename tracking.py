"""Optional Weights & Biases tracking for every experiment script.

Off unless WANDB_PROJECT is set, so no run ever depends on it:

    wandb login                                   # once, on a node with internet
    export WANDB_PROJECT=e0334-flat-minima        # turn tracking on
    export WANDB_MODE=offline                     # if the GPU node has no internet; later:
    wandb sync results/wandb/offline-run-*        #   upload from a node that has it

Each run logs the script's arguments as config, its progress rows (epoch / step metrics) as they
are printed, and the final scalar results as the run summary. sweep.py names each W&B run after its
job and groups runs by phase. Local W&B files go to results/wandb/.
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
_run = None


def _scalars(d, prefix=""):
    """Numbers, strings and booleans of `d`, with nested dicts flattened to "a/b" keys."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_scalars(v, f"{prefix}{k}/"))
        elif isinstance(v, (int, float, str, bool)) or v is None:
            out[prefix + k] = v
    return out


def init(args):
    """Start a run with the script's arguments as config. Does nothing if tracking is off."""
    global _run
    if not os.environ.get("WANDB_PROJECT"):
        return
    import wandb
    directory = os.path.join(ROOT, "results")
    os.makedirs(directory, exist_ok=True)
    name = os.environ.get("WANDB_NAME") or os.path.splitext(os.path.basename(getattr(args, "out", "")))[0] or None
    _run = wandb.init(config=vars(args), name=name, dir=directory)


def log(row):
    if _run is not None:
        _run.log({k: v for k, v in _scalars(row).items() if isinstance(v, (int, float)) and not isinstance(v, bool)})


def histogram(key, values):
    if _run is not None:
        import wandb
        _run.log({key: wandb.Histogram(values, num_bins=200)})


def finish(result):
    """Store the final scalar results as the run summary and close the run."""
    global _run
    if _run is not None:
        _run.summary.update(_scalars(result))
        _run.finish()
        _run = None
