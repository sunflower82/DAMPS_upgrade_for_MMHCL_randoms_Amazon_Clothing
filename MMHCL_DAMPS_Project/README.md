# DAMPS-MMHCL — Spectral Domain Calibration for Multi-Modal Hypergraph Recommendation

**Reference:** *In-Depth Design Analysis Report — Upgrading the MMHCL Multi-Modal Recommendation Framework via Spectral Domain Representation Calibration (DAMPS)*, Revision 9 — 100% Compliance Check & Final Lock.

This repository contains the **production-quality reference implementation** of DAMPS-MMHCL. It is designed to drop directly into the original MMHCL training pipeline while introducing the four core upgrades described in the architecture specification:

1. **DAMPS spectral calibration block** — APC + AVRF + Residual IMCF + Soft Residual-Routing.
2. **Pattern B' (Scheduled Rebuild)** — recomputes the K-NN multi-modal hypergraph every `R` epochs from a slim momentum buffer.
3. **Slim Momentum Encoder** — EMA-smooths only the post-DAMPS `d=64` representation (~98 % VRAM saving versus naive MoCo-style momentum).
4. **Engineering hardening** — `bfloat16` mixed precision, learnable InfoNCE temperature `τ`, data-driven AVRF prior, cuFFT plan-cache lockdown, dual-path K-NN (chunked PyTorch + FAISS HNSW for `N ≥ 60 000`).

---

## 1. Repository Layout

```
MMHCL_DAMPS_Project/
├── damps/                     # Spectral calibration package
│   ├── __init__.py            # Public API exports
│   ├── core.py                # DAMPS module (APC + AVRF + IMCF + FFT/IFFT)
│   ├── momentum.py            # Slim Momentum Encoder
│   ├── graph.py               # Dual-path K-NN hypergraph builder
│   └── prior.py               # Data-driven SNR-based AVRF prior derivation
│
├── utility/                   # Shared helpers (mirrors original MMHCL)
│   ├── __init__.py
│   ├── parser.py              # CLI argument parser (DAMPS knobs incl.)
│   ├── load_data.py           # Dataset loader + raw modality features
│   ├── batch_test.py          # GPU-accelerated evaluation
│   ├── metrics.py             # Recall / NDCG / Hit / Coverage / Gini
│   └── logging.py             # Dual-destination Logger
│
├── model.py                   # DAMPS_MMHCL — full integrated network
├── train.py                   # Pattern B' training orchestrator
├── requirements.txt
└── README.md
```

---

## 2. Architecture Overview

```
                    Raw image_feat.npy        Raw text_feat.npy        (Raw audio_feat.npy — Tiktok)
                            │                         │                            │
                            ▼                         ▼                            ▼
                ┌────────────────────────────────────────────────────────────────────┐
                │       Per-modality MLP projection (eq. 4) → d = 64                 │
                └────────────────────────────────────────────────────────────────────┘
                            │                         │                            │
                            ▼                         ▼                            ▼
                ┌────────────────────────────────────────────────────────────────────┐
                │                     DAMPS spectral pipeline                        │
                │  rFFT → Metadata-Aware APC (von Mises MLE)                         │
                │       → AVRF (logit-clipped Wiener gate, per-epoch EMA MAD)        │
                │       → Residual IMCF (ASC consensus, residual)                    │
                │       → irFFT                                                       │
                └────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
                ┌────────────────────────────────────────────────────────────────────┐
                │           Soft Residual-Routing  (eq. 3)                            │
                │     h_input = h_raw + α · LayerNorm(h_cal)                          │
                └────────────────────────────────────────────────────────────────────┘
                                              │
              ┌───────────────────────────────┼───────────────────────────────┐
              ▼                               ▼                               ▼
   I2I-Hypergraph (Pattern B')   U2U Co-interaction (rw-norm)    UI Bipartite (LightGCN/NGCF)
              │                               │                               │
              ▼                               ▼                               ▼
              ii_emb                          uu_emb                          u/i_ui_emb
              │                               │                               │
              └────────────────┬──────────────┴───────────┬───────────────────┘
                               ▼                          ▼
                         CF + Hypergraph fusion       InfoNCE (learnable τ)
                               │                          │
                               └──────────► Final BPR loss + L2 reg + λ·NCE
```

---

## 3. Module-by-Module Cheatsheet

| Module | Key class / function | Spec section | Notes |
| ------ | -------------------- | ------------ | ----- |
| `damps/core.py` | `DAMPS` | §2.1 – §2.4 | 1-D rFFT, Metadata APC, AVRF logit-clipped gate, Residual IMCF |
| `damps/core.py` | `DAMPS.update_epoch_mad` | §2.3 | Per-epoch EMA MAD aggregation, β_t schedule |
| `damps/momentum.py` | `SlimMomentumEncoder` | §3.1.1 | Slim momentum on `d=64` only |
| `damps/graph.py` | `DualPathKNN` | §3.2 | Chunked PyTorch (default) + FAISS HNSW fallback |
| `damps/prior.py` | `compute_avrf_logit` | §2.3 | Data-driven SNR-based prior derivation, strict [-2, 2] clip |
| `model.py` | `DAMPS_MMHCL` | §1.3 + §3.1 | Full backbone with Soft Residual-Routing + learnable τ |
| `train.py` | `Trainer.maybe_rebuild_hypergraph` | §3.1 | Pattern B' Scheduled Rebuild every `R` epochs |
| `train.py` | `_configure_cufft_cache` | §3.3 | Permanently disable cuFFT plan cache |

---

## 4. Quick Start

### 4.1 Installation

```bash
cd MMHCL_DAMPS_Project
pip install -r requirements.txt

# Optional accelerators
pip install faiss-gpu          # auto-activates when N ≥ 60 000
pip install wandb              # for --use_wandb 1
pip install optuna             # for the Bayesian HPO loop
```

### 4.2 Dataset preparation

Place every dataset under `../data/<dataset>/` relative to this folder:

```
../data/Clothing/
├── 5-core/
│   ├── train.json
│   ├── val.json
│   └── test.json
├── image_feat.npy               # (n_items, image_dim)
├── text_feat.npy                # (n_items, text_dim)
├── meta_categories.npy          # OPTIONAL (n_items,) int — used by APC
└── audio_feat.npy               # Tiktok only
```

If `meta_categories.npy` is missing, `utility/load_data.py` falls back to a deterministic hash that still gives APC a clustering signal (no k-means is ever invoked, per §2.2 of the spec).

### 4.3 Train

```bash
# Default lock-in (all DAMPS components ON, R = 5, bfloat16 AMP)
python train.py --dataset Clothing --seed 42

# Tiktok with audio modality
python train.py --dataset Tiktok --seed 42

# Custom rebuild cadence + W&B logging
python train.py --dataset Sports --rebuild_R 5 --use_wandb 1
```

### 4.4 Ablation switchboard

| Flag | What it controls | Default |
| ---- | ---------------- | ------- |
| `--damps_apc` | Metadata-Aware APC | 1 (ON) |
| `--damps_avrf` | AVRF logit-clipped Wiener gate | 1 |
| `--damps_imcf` | Residual IMCF | 1 |
| `--damps_permutation_fft` | Permutation-FFT falsifiability test | 0 |
| `--damps_soft_routing` | Soft Residual-Routing into HGCN | 1 |
| `--damps_momentum` | Slim Momentum Encoder | 1 |
| `--damps_data_driven_prior` | SNR-based AVRF prior derivation | 1 |
| `--rebuild_R` | Pattern B' rebuild frequency (epochs) | 5 |
| `--use_amp` | bfloat16 mixed precision | 1 |

The eight-row defensive ablation table from §6 of the spec maps onto these flags directly:

1. **MMHCL Baseline** — `--damps_apc 0 --damps_avrf 0 --damps_imcf 0 --damps_soft_routing 0 --damps_momentum 0`
2. **Rebuild frequency sweep** — vary `--rebuild_R` ∈ {1, 5, 10, 20}
3. **Momentum ON/OFF** — `--damps_momentum {0, 1}`
4. **+ APC only** — `--damps_apc 1 --damps_avrf 0 --damps_imcf 0`
5. **+ AVRF only** — `--damps_apc 0 --damps_avrf 1 --damps_imcf 0`
6. **+ Residual IMCF only** — `--damps_apc 0 --damps_avrf 0 --damps_imcf 1`
7. **Full DAMPS-MMHCL** — all defaults
8. **Permutation-FFT falsification** — `--damps_permutation_fft 1`

For the dedicated paired-seed Permutation-FFT falsifiability protocol
(spec Section 6, Item 8) use the helper scripts:

```bash
# Linux / macOS
N_SEEDS=3 SEED_BASE=42 ./scripts/run_permutation_fft_ablation.sh

# Windows (cmd / PowerShell)
set N_SEEDS=3 & set SEED_BASE=42 & scripts\run_permutation_fft_ablation.bat

# Aggregate the paired runs and compute the t-test:
python scripts/_aggregate_permutation_fft.py --seeds 42 43 44
```

The aggregator parses `BEST_Test_Recall@20` / `BEST_Test_NDCG@20` from each
per-run log file, performs a paired t-test, and prints the spec's binary
verdict: switch to DCT-II if `|gap| < 1 %`, otherwise the standard 1-D FFT
is validated.

---

## 5. Diagnostic Logs Produced

Every `R`-th epoch the training loop emits two transparency probes mandated by §3.1 of the spec:

```
[Rebuild] epoch=15  NNZ=115234  avg_deg=5.04  (target K=5)
[diag epoch 15] tau=0.0987  alpha_img=0.1124 alpha_txt=0.1009
                tanh_sat: img=0.121 txt=0.302  baseline_asc=0.318
```

* **NNZ / avg_deg** — proves Pattern B' keeps the K-NN graph density anchored.
* **tanh saturation** — confirms the AVRF gate has not been cold-start-paralysed.
* **baseline_asc** — confirms the IMCF residual coefficient is correctly centred.

When `--use_wandb 1`, the same metrics stream to your W&B run for cross-experiment comparison.

### 5.1 Best-validation reporting

At the end of every run the trainer emits **four** disambiguated summary lines so the values printed to the log file match the maxima of the WandB curves exactly:

```
BEST_Val_Recall@10:  <max of val/recall@10>
BEST_Val_Recall@20:  <max of val/recall@20>     ← matches wandb chart max
BEST_Val_NDCG@10:    <max of val/ndcg@10>       ← needs the new val/ndcg@10 chart
BEST_Val_NDCG@20:    <max of val/ndcg@20>
BEST_Test_Recall@20: <test recall at recall-best val epoch>
BEST_Test_Precision@20: <test precision at recall-best val epoch>
BEST_Test_NDCG@20:   <test NDCG at ndcg-best val epoch>
```

The two `BEST_Test_*` lines are pinned to the validation epoch where the *corresponding* validation metric peaked — even if a *later* epoch only improves NDCG (or only improves Recall) and overwrites the running test snapshot. Previously the code overwrote `test_ret` on every improvement of either kind, which meant `BEST_Test_Recall@K` could end up reflecting a non-recall-optimal validation epoch. The refactored tracker fixes this corner case.

The WandB `val` section now also surfaces `val/ndcg@10` (alongside `val/recall@10`, `val/recall@20`, `val/ndcg@20`, `val/precision@20`, `val/hit@20`) so NDCG@10 can be inspected mid-training.

---

## 6. Key Engineering Safeguards

| Safeguard | Where it lives | Justification |
| --------- | -------------- | ------------- |
| `cufft_plan_cache.max_size = 1` | `train.py::_configure_cufft_cache` | Prevents cuFFT plan-cache memory leaks (Section 3.3). |
| `bfloat16` AMP | `train.py::Trainer.train` | -30 / -40 % wall-clock; -40 % VRAM on Ada Lovelace. |
| Strict `[-2, 2]` AVRF clip | `damps/core.py::_init_avrf_logit` | Avoids tanh saturation at warm-up. |
| Per-epoch EMA MAD | `damps/core.py::update_epoch_mad` | 5–7× variance reduction vs per-batch. |
| Slim Momentum (`d=64`) | `damps/momentum.py::SlimMomentumEncoder` | -98 % auxiliary VRAM vs MoCo. |
| Pattern B' rebuild | `train.py::maybe_rebuild_hypergraph` | Stable density; no NNZ explosion. |
| Dual-path K-NN | `damps/graph.py::DualPathKNN` | O(N log N) when `N ≥ 60 000`. |
| Learnable τ | `model.py::DAMPS_MMHCL.tau` | Prevents InfoNCE saturation post-EMA. |

---

## 7. Reproducibility & Statistical Reporting

Per §4 of the spec, every reported headline result must be averaged across **10 seeds** with **95 % confidence intervals** and **paired t-tests** versus the MMHCL baseline. The training script accepts `--seed` as a CLI flag; loop over your seed list and aggregate the per-run summaries written to `../<dataset>/MM/sum_<ablation_target>.txt`.

A reusable helper for the paired t-test ships at `scripts/paired_ttest.py`:

```bash
python scripts/paired_ttest.py \
    --damps damps_seeds.csv \
    --baseline mmhcl_seeds.csv \
    --column recall@20
```

The script wraps `scipy.stats.ttest_rel` (the *correct* paired test — `ttest_ind` would be wrong because the seeds are paired across methods) and additionally prints a 95 % confidence interval on the mean of the paired differences.

---

## 8. Training Speedup Toggles

The following accelerations are documented in the project's *Training Speedup Guide*. Each one is **opt-in** via a CLI flag so the locked Revision 9 architecture remains the default.

| Toggle | Default | Speedup | Where it lives |
| ------ | ------- | ------- | -------------- |
| `--use_amp 1`              | on  | -30 / -40 % wall-clock (bfloat16, no GradScaler) | `train.py` |
| `--use_torch_compile 1`    | off (CLI), **on in notebook** | +25-35 % on the DAMPS forward path | `train.py::Trainer.__init__` |
| `--torch_compile_mode`     | `reduce-overhead` | tunes `torch.compile` aggression | speedup guide §4 |
| `--faiss_use_gpu 1`        | on (when N >= `faiss_threshold`) | 5-10x vs CPU FAISS | `damps/graph.py::DualPathKNN._build_faiss` |
| `--knn_efsearch 64`        | 64 | controls HNSW recall/speed trade-off | `damps/graph.py` |
| `norm="ortho"` rFFT/irFFT  | always | improved numerical conditioning at d=64 | `damps/core.py::_fft / _ifft` |

The companion notebook `Local_Random_seeds_train_mmhcl_clothing_colab_original.ipynb` (kept off the repository for privacy reasons) wires *all* of the above through to `train.py` and runs the spec-mandated **10 seeds** (Section 4) so the paired t-test report from `scripts/paired_ttest.py` has the statistical power required by the spec.

`torch.compile` is intentionally applied **only** to the DAMPS submodule. Compiling the full forward path would force recompilation every Pattern B' rebuild, because the sparse `Item_mat` shape changes when the K-NN graph is regenerated. The DAMPS submodule has fixed input/output shapes so compilation is safe with `dynamic=True`.

**Inductor + complex FFT backward — automatic eager fallback.** Some PyTorch builds (notably Windows + CUDA) currently have an Inductor backward-compile bug for graphs that flow through complex tensors (rFFT → APC/AVRF/IMCF → iRFFT), which surfaces inside `loss.backward()` as

```
torch._inductor.exc.InductorError: AttributeError: 'complex' object has no attribute 'get_name'
```

This crash originates in AOT-autograd's `bw_compiler` chain and is **not** caught by `torch._dynamo.config.suppress_errors` (that flag only covers Dynamo forward graph-capture errors). To keep training robust, `train.py` runs a tiny end-to-end probe at startup:

```
torch.compile(rFFT → ×2 → iRFFT) → forward → backward
```

If the probe raises, the trainer logs

```
[speedup] torch.compile requested but this PyTorch build's Inductor BACKWARD
compiler cannot lower DAMPS's complex FFT region (probe failed …). Skipping
the wrap and running DAMPS in eager mode.
```

and runs DAMPS in eager mode (correct, just without compile speedup on this PyTorch build). When PyTorch ships a fix the probe will succeed automatically and the wrap is re-enabled — no notebook or CLI changes required. Users who don't want the probe overhead can pass `--use_torch_compile 0`.

### Hyperparameter optimisation

Two ready-to-run BOHB harnesses are provided. Both anchor the architecture per §4 of the spec and search only `K` (`--topk`) and the IMCF residual coefficient `lambda_coh`:

```bash
# Optuna TPE + HyperbandPruner (50 trials, headless)
python scripts/run_optuna_hpo.py --dataset Clothing --n_trials 50

# Weights & Biases Bayesian sweep (parallelisable across GPUs)
python scripts/run_wandb_sweep.py --action create
python scripts/run_wandb_sweep.py --action run --sweep <returned_id> --count 50
```

Hyperband typically prunes 60-70 % of unpromising trials, shrinking the total wall-clock cost from ~7-8 days down to 2-3 days for the recommended 50-trial budget.

### Smoke test

A tiny end-to-end CPU smoke test (~5 s) exercises every speedup toggle, including a `torch.compile` regression check:

```bash
python tests/smoke_test.py
```

---

## 9. Citation

If this implementation contributes to your research, please cite the original MMHCL paper and reference this DAMPS extension (the architecture revision PDF in the repository root).

```bibtex
@misc{damps_mmhcl_2026,
  title  = {Upgrading MMHCL via Spectral Domain Representation Calibration (DAMPS)},
  year   = {2026},
  note   = {Revision 9 — 100\% Compliance Check & Final Lock}
}
```
