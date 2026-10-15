# Flat vs. Sharp Minima — E0 334 Project

PyTorch implementations for the papers in our presentation, plus the code for our contribution (curvature of GLUE fine-tuning).

| Folder | Contents |
|--------|----------|
| `SAM/` | SAM / ASAM, GSAM, WSAM, Eigen-SAM, X-SAM (`sam_optim`) and GLUE fine-tuning (`train_glue.py`) |
| `Entropy-SGD/` | Entropy-SGD / Entropy-Adam (`entropy_optim`) and GLUE fine-tuning (`train_glue.py`) |
| `Maximum-Hessian/` | λmax and Hessian-trace tools (`hessian_tools`) and the vision experiments of Kaur et al. (`train_vision.py`) |
| `project.tex` | Presentation notes (compile on Overleaf) |

## Setup on a new machine

Requirements: Linux, Python ≥ 3.9, and an NVIDIA GPU with a recent driver.

With the data in a `datasets/` folder next to the repo:

```bash
git clone https://github.com/APatkuri/E0_334_Project.git    # or: cd E0_334_Project && git pull
cd E0_334_Project
DATA_ROOT=$(realpath ../datasets) bash setup.sh
source .venv/bin/activate
```

`setup.sh` does three things:

1. creates `.venv`, falling back to `virtualenv` if `python3 -m venv` is unavailable;
2. installs `requirements.txt`;
3. runs `prepare_data.py`, which prints the GPU and package versions and downloads all the data.

Everything goes under `DATA_ROOT` (default: `./data`):

| Data | Size | Stored in |
|------|------|-----------|
| GLUE: RTE, MRPC, CoLA, SST-2, QNLI, QQP, MNLI | ≈ 1–2 GB (with tokenization caches) | `$DATA_ROOT/huggingface` |
| `bert-base-uncased` weights | ≈ 0.45 GB | `$DATA_ROOT/huggingface` |
| CIFAR-10 and Fashion-MNIST | ≈ 0.4 GB | `$DATA_ROOT/vision` |

The Python environment adds another 3–5 GB, so plan for about 10 GB free. On the GPU, BERT-base at batch 32 needs about 5–6 GB for AdamW and the SAM variants and 8–10 GB for Eigen-SAM, X-SAM and the λmax measurements. With 8 GB, halve `--batch_size` for those.

`source .venv/bin/activate` also sets `DATA_ROOT` and `HF_HOME`, so every script finds the data without extra flags.

**Options:**
- **CUDA build.** If the default PyTorch wheel doesn't match the machine's CUDA driver (check with `nvidia-smi`), choose one: `TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 bash setup.sh`.
- **Partial download.** Arguments to `setup.sh` are passed to `prepare_data.py`. For example, `bash setup.sh --glue rte mrpc cola --no_vision` downloads only what the GLUE experiments need.
- **No internet on the compute node.** Run setup on a login node, then `export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` when training.

## First runs

Each takes a few minutes on a GPU. Together they exercise every code path.

```bash
cd SAM
python train_glue.py --task rte --epochs 1 --optim adamw
python train_glue.py --task rte --epochs 1 --optim sam
python train_glue.py --task rte --epochs 1 --optim xsam          # Hessian-vector products
cd ../Entropy-SGD
python train_glue.py --task rte --epochs 1 --optim entropysgd
cd ../Maximum-Hessian
python train_vision.py --dataset fashion_mnist --model mlp --lr 0.1 --max_epochs 2
```
