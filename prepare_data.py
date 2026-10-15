"""Download every dataset and model the experiments use, so training runs need no network.

    python prepare_data.py                          # everything below
    python prepare_data.py --glue rte mrpc cola     # only some GLUE tasks
    python prepare_data.py --no_vision              # skip CIFAR-10 / Fashion-MNIST

Everything goes under $DATA_ROOT (set by setup.sh; default ./data):
    $DATA_ROOT/huggingface   GLUE and model weights (the Hugging Face cache, HF_HOME)
    $DATA_ROOT/vision        CIFAR-10 and Fashion-MNIST, where train_vision.py looks by default
"""
import argparse
import os

import torch

GLUE_TASKS = ["rte", "mrpc", "cola", "sst2", "qnli", "qqp", "mnli"]  # the tasks in train_glue.py
DATA_ROOT = os.path.abspath(os.environ.get("DATA_ROOT", "data"))
os.environ.setdefault("HF_HOME", os.path.join(DATA_ROOT, "huggingface"))  # before importing HF libraries


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--glue", nargs="*", default=GLUE_TASKS, choices=GLUE_TASKS)
    p.add_argument("--models", nargs="*", default=["bert-base-uncased"])
    p.add_argument("--vision_dir", default=os.path.join(DATA_ROOT, "vision"))
    p.add_argument("--no_vision", action="store_true")
    return p.parse_args()


def report_environment():
    import datasets
    import transformers
    print(f"torch {torch.__version__} | transformers {transformers.__version__} | datasets {datasets.__version__}")
    print(f"data root: {DATA_ROOT} (Hugging Face cache: {os.environ['HF_HOME']})")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {prop.name}, {prop.total_memory / 2**30:.1f} GiB")
    else:
        print("WARNING: no CUDA GPU visible; training will run on the CPU")


def download_glue(tasks):
    from datasets import load_dataset
    for task in tasks:
        data = load_dataset("nyu-mll/glue", task)
        sizes = ", ".join(f"{split} {len(ds)}" for split, ds in data.items())
        print(f"GLUE {task:5s}: {sizes}")


def download_models(names):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    for name in names:
        AutoTokenizer.from_pretrained(name)
        # eager attention is what Eigen-SAM, X-SAM and the Hessian tools need (double backward)
        model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2, attn_implementation="eager")
        n = sum(p.numel() for p in model.parameters())
        print(f"model {name}: {n / 1e6:.1f}M parameters")


def download_vision(root):
    from torchvision import datasets
    for cls in (datasets.CIFAR10, datasets.FashionMNIST):
        train = cls(root, train=True, download=True)
        test = cls(root, train=False, download=True)
        print(f"{cls.__name__}: train {len(train)}, test {len(test)} -> {root}")


def main():
    args = parse_args()
    report_environment()
    download_glue(args.glue)
    download_models(args.models)
    if not args.no_vision:
        download_vision(args.vision_dir)
    print("All data ready.")


if __name__ == "__main__":
    main()
