# Flat vs. Sharp Minima — E0 334 Project

PyTorch implementations of the papers in our presentation, plus our contribution: curvature of GLUE fine-tuning.

| Folder | Contents |
|--------|----------|
| `SAM/` | SAM / ASAM, GSAM, WSAM, Eigen-SAM, X-SAM (`sam_optim`) + GLUE fine-tuning |
| `Entropy-SGD/` | Entropy-SGD / Entropy-Adam (`entropy_optim`) + GLUE fine-tuning |
| `Maximum-Hessian/` | λmax and Hessian-trace tools (`hessian_tools`) + Kaur et al. vision experiments |
| `project.tex` | Presentation notes (compile on Overleaf) |

## Setup

Needs Linux, Python ≥ 3.9 and an NVIDIA GPU.

```bash
git clone https://github.com/APatkuri/E0_334_Project.git
cd E0_334_Project
DATA_ROOT=$(realpath ../datasets) bash setup.sh   # DATA_ROOT defaults to ./data
source .venv/bin/activate                         # also sets DATA_ROOT and HF_HOME
```

`setup.sh` creates `.venv`, installs `requirements.txt`, and runs `prepare_data.py` to download:

| Data | Size | Location |
|------|------|----------|
| GLUE (RTE, MRPC, CoLA, SST-2, QNLI, QQP, MNLI) | 1–2 GB | `$DATA_ROOT/huggingface` |
| `bert-base-uncased` | 0.45 GB | `$DATA_ROOT/huggingface` |
| CIFAR-10, Fashion-MNIST | 0.4 GB | `$DATA_ROOT/vision` |

Plan for ~10 GB of disk including the environment.

**GPU memory** (BERT-base, batch 32): 5–6 GB for AdamW and SAM; 8–10 GB for Eigen-SAM, X-SAM and λmax. On 8 GB, halve `--batch_size`.

**Options**

| Need | Command |
|------|---------|
| Different CUDA build | `TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 bash setup.sh` |
| GLUE only | `bash setup.sh --glue rte mrpc cola --no_vision` |
| Offline compute node | Run setup on a login node, then `export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` |

## Smoke test

A few minutes each on a GPU; together they cover every code path.

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
