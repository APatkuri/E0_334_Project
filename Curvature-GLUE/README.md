# Curvature of GLUE fine-tuning (our contribution)

BERT-base fine-tuned on GLUE with every optimizer from the talk, under one protocol. Each run measures generalization and curvature (`project.tex`, Sec. "Our contribution").

| File | Purpose |
|------|---------|
| `run_glue.py` | One run: any of the 7 optimizers, compute matching, dev/train metrics, final λmax, tr(H) and SAM sharpness, λmax and the update–v₁ angle during training. Writes one JSON file |
| `analyze.py` | Tables (mean ± std over seeds), Spearman(λmax, dev), Kaur-style table, λmax-vs-dev scatter |
| `../sweep.py` | Runs the design below as a GPU job queue (`glue-*` phases) |

## Running

From the repo root, on the GPU node, in tmux:

```bash
python sweep.py glue-smoke --gpus 3                  # ~15 min, checks every code path
python sweep.py glue-lr --gpus 3 4 --per_gpu 2
python sweep.py glue-rho --gpus 3 4 --per_gpu 2
python sweep.py glue-variants --gpus 3 4 --per_gpu 2
python sweep.py glue-final --gpus 3 4 --per_gpu 2
python sweep.py glue-kaur --gpus 3 4 --per_gpu 2     # independent of the others
python Curvature-GLUE/analyze.py --plot
```

Finished runs are skipped, so any phase can be restarted.

## Protocol and sources

No paper we present fine-tunes BERT. Each method's settings come from its own paper; the tuning order is the one the SAM-variant papers use.

| Stage | Setting | Source |
|-------|---------|--------|
| Base | BERT-base-uncased, AdamW, batch 32, 3 epochs, max length 128, 10% warmup + linear decay, wd 0.01 | Devlin et al. 2019, Sec. 4.1 |
| `glue-lr` | AdamW lr ∈ {2, 3, 4, 5}e-5, best on dev; then fixed for every method | Devlin et al. Sec. 4.1; GSAM App. B.1 |
| `glue-rho` | SAM ρ ∈ {0.02, 0.05, 0.1, 0.15, 0.2, 0.3}, ascent on 1/4 of the batch | Bahri et al. 2022, Sec. 4.1, Alg. 1 |
| | Entropy-Adam: L = 5, fixed γ (no scoping), β₁ = 0.5, ε = 1e-4, α = 0.75; γ ∈ {0.003, 0.01, 0.03} | Chaudhari et al. 2017, Sec. 4.2, 5.4.2 |
| `glue-variants` | ρ = SAM's best, then: GSAM α ∈ {0.1, 0.2, 0.3} with ρ tied to the lr (ρ_min = 0) | GSAM App. B.1, Table 4 |
| | WSAM γ ∈ {0.80, 0.82, …, 0.98} | WSAM Sec. 6.2 |
| | Eigen-SAM α ∈ {0.05, 0.1, 0.2}, p = 100, q = 5 | Eigen-SAM App. E, Sec. 5 |
| | X-SAM α = 0.2 (not tuned) | X-SAM Sec. 6.1 |
| `glue-final` | best setting per task and method, 5 seeds | our design |
| Budget | equal gradient evaluations: an update costs 1 (AdamW), 1.25 (SAM with 1/4 ascent), 2 (GSAM, WSAM), 2.11 (Eigen-SAM, X-SAM), 5 (Entropy-Adam) | Eigen-SAM App. E, GSAM App. B.1, Entropy-SGD Sec. 5.2 |
| `glue-kaur` | Kaur et al. Sec. 3.3 on SST-2 (20%): constant lr, train to 99% train accuracy, ρ ∈ {0, 0.02, …, 0.1}, 2 learning rates, 4 seeds | Kaur et al. 2023 |

**Deviations we cannot avoid** (flag them in the report):

- **Model and tasks.** BERT, not T5 (Bahri) or vision models (the others). The tasks are RTE, MRPC, CoLA and SST-2 subsampled to 5% and 20%. Bahri subsamples SuperGLUE and CBQA at 2–80%, not GLUE.
- **Inner SGLD step of Entropy-Adam.** The paper's η′ ∈ [0.1, 1] is for SGD on CNNs, and it would wreck pretrained BERT. We use Adam with η′ = lr, kept fixed. The γ grid is ours; the paper tunes γ so that `grad_ratio` ≈ 1, which is recorded per run.
- **Checkpoint.** We report the final checkpoint, because curvature is measured there. Bahri reports the best one; every dev evaluation is kept in `dev_curve`.
- **X-SAM p.** Not given in the X-SAM paper; we use Eigen-SAM's p = 100.
- **Kaur-style test.** It uses AdamW with BERT learning rates (2e-5, 5e-5), not SGD. Its large batch is 256 (2% of the data), not 10%, because larger batches need gradient accumulation.

## Measurements

| Field | What |
|-------|------|
| `dev`, `dev_curve` | accuracy, F1, Matthews corr. and loss; the primary metric is accuracy (RTE, SST-2), F1 (MRPC) or MCC (CoLA) |
| `train`, `gap` | the same metrics on ≤ 5000 training examples; train − dev |
| `lmax`, `trace`, `sharpness` | final weights, 1000 fixed training examples, dropout off |
| `track` | λmax, \|cos(v₁, gradient)\| and \|cos(v₁, update)\| at 5 points (256 fixed examples); RQ4 |
| `grad_ratio` | Entropy-Adam only: ‖x − μ‖ / ‖∇f‖ |
