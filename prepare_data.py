"""Download every dataset and model the experiments use, so training runs need no network.

    python prepare_data.py                          # everything below
    python prepare_data.py --glue rte mrpc cola     # only some GLUE tasks
    python prepare_data.py --no_vision              # skip the image datasets and ViT weights
    python prepare_data.py --no_text                # skip PTB / War and Peace

Everything goes under $DATA_ROOT (set by setup.sh; default ./data):
    $DATA_ROOT/huggingface   GLUE and model weights (the Hugging Face cache, HF_HOME)
    $DATA_ROOT/vision        CIFAR-10/100, Fashion-MNIST, MNIST, SVHN, where the vision scripts look by default
    $DATA_ROOT/torch         ViT-B/16 ImageNet weights (SAM/train_image.py)
    $DATA_ROOT/text          PTB and War and Peace (Entropy-SGD/train_lm.py)
"""
import argparse
import os
import urllib.request

import torch

# Entropy-SGD Sec. 5.4: PTB as distributed with Zaremba et al. (2014), and the War and Peace text
# of Karpathy et al. (2015)
TEXT_URLS = {f"ptb.{s}.txt": f"https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.{s}.txt"
             for s in ("train", "valid", "test")}
TEXT_URLS["warpeace_input.txt"] = "https://cs.stanford.edu/people/karpathy/char-rnn/warpeace_input.txt"

GLUE_TASKS = ["rte", "mrpc", "cola", "sst2", "qnli", "qqp", "mnli"]  # the tasks in train_glue.py
DATA_ROOT = os.path.abspath(os.environ.get("DATA_ROOT", "data"))
os.environ.setdefault("HF_HOME", os.path.join(DATA_ROOT, "huggingface"))  # before importing HF libraries


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--glue", nargs="*", default=GLUE_TASKS, choices=GLUE_TASKS)
    p.add_argument("--models", nargs="*", default=["bert-base-uncased"])
    p.add_argument("--vision_dir", default=os.path.join(DATA_ROOT, "vision"))
    p.add_argument("--no_vision", action="store_true")
    p.add_argument("--text_dir", default=os.path.join(DATA_ROOT, "text"))
    p.add_argument("--no_text", action="store_true")
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
    for cls in (datasets.CIFAR10, datasets.CIFAR100, datasets.FashionMNIST, datasets.MNIST):
        train = cls(root, train=True, download=True)
        test = cls(root, train=False, download=True)
        print(f"{cls.__name__}: train {len(train)}, test {len(test)} -> {root}")
    train, test = (datasets.SVHN(root, split=s, download=True) for s in ("train", "test"))
    print(f"SVHN: train {len(train)}, test {len(test)} -> {root}")
    # Eigen-SAM Sec. 6.3: torchvision's ImageNet ViT-B/16 checkpoint, cached in $DATA_ROOT/torch
    from torchvision.models import ViT_B_16_Weights
    torch.hub.set_dir(os.path.join(DATA_ROOT, "torch"))
    ViT_B_16_Weights.IMAGENET1K_V1.get_state_dict(progress=False)
    print(f"ViT-B/16 ImageNet weights -> {os.path.join(DATA_ROOT, 'torch')}")


def download_text(root):
    os.makedirs(root, exist_ok=True)
    for name, url in TEXT_URLS.items():
        path = os.path.join(root, name)
        if not os.path.exists(path):
            urllib.request.urlretrieve(url, path + ".part")
            os.replace(path + ".part", path)
        text = open(path).read()
        units = len(text.split()) if name.startswith("ptb") else len(text)
        print(f"{name}: {units} {'words' if name.startswith('ptb') else 'characters'} -> {root}")


def main():
    args = parse_args()
    report_environment()
    download_glue(args.glue)
    download_models(args.models)
    if not args.no_vision:
        download_vision(args.vision_dir)
    if not args.no_text:
        download_text(args.text_dir)
    print("All data ready.")


if __name__ == "__main__":
    main()
