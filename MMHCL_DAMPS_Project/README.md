# DAMPS-MMHCL — Spectral Domain Calibration for Multi-Modal Hypergraph Recommendation

**Reference:** *In-Depth Design Analysis Report — Upgrading the MMHCL Multi-Modal Recommendation Framework via Spectral Domain Representation Calibration (DAMPS)*, Revision 11 — Phase 1 (Quick Win) Execution Plan with Quantitative Stop-Gate. Builds on top of the Revision 9 — 100 % Compliance Check & Final Lock baseline.

This repository contains the **production-quality reference implementation** of DAMPS-MMHCL. It is designed to drop directly into the original MMHCL training pipeline while introducing the four core upgrades described in the architecture specification:

1. **DAMPS spectral calibration block** — APC + AVRF + Residual IMCF + Soft Residual-Routing.
2. **Pattern B' (Scheduled Rebuild)** — recomputes the K-NN multi-modal hypergraph every `R` epochs from a slim momentum buffer.
3. **Slim Momentum Encoder** — EMA-smooths only the post-DAMPS `d=64` representation (~98 % VRAM saving versus naive MoCo-style momentum).
4. **Engineering hardening** — `bfloat16` mixed precision, **static** InfoNCE temperature `τ` (rev44 Phase 1 default — see §6 below), data-driven AVRF prior, cuFFT plan-cache lockdown, dual-path K-NN (chunked PyTorch + FAISS HNSW for `N ≥ 60 000`).

> **Revision 11 / rev44 Phase 1 highlights**
>
> Empirical 10-seed analysis on Amazon Clothing showed the rev42/Revision 9 anchor (`Recall@20 = 0.0786 ± 0.0016`, `NDCG@20 = 0.0383 ± 0.0008`) suffers a **10.7 % asymmetric coverage loss** vs. the MMHCL paper. Root cause: the learnable τ's gradient vanishes and τ saturates at ~0.0909, triggering an embedding collapse. The Top-10 concentration signature (`Recall@10 ↔ Recall@20` gap of only ~0.027) independently corroborates the diagnosis.
>
> Phase 1 — Quick Win (`< 1 day`) consists of three zero-risk action items:
>
> 1. **Audit eval protocol** (`scripts/audit_eval_protocol.py`, 7-point checklist).
> 2. **Static τ sweep** anchored at τ = 0.3 (sweep set `{0.2, 0.3, 0.5}`); set via `--temperature 0.3 --learnable_tau 0` (now the **default**).
> 3. **AVRF ablation** — disable AVRF on sparse Clothing to recover useful signal: `--damps_avrf 0` (now the **default**).
>
> Stop-gate: Phase 1 is declared successful iff Recall@20 ≥ 0.0870 **AND** NDCG@20 ≥ 0.0390 (10-seed mean) with paired *t*-test `p < 0.05` vs the rev42 anchor. Recall@20 ≥ 0.0900 fully validates the paper contribution.

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
| `model.py` | `DAMPS_MMHCL` | §1.3 + §3.1 (+ rev44 §3) | Full backbone with Soft Residual-Routing; τ is **learnable** (rev42, `nn.Parameter`) or **static** (rev44 default, `register_buffer`) |
| `train.py` | `Trainer.maybe_rebuild_hypergraph` | §3.1 | Pattern B' Scheduled Rebuild every `R` epochs |
| `train.py` | `_configure_cufft_cache` | §3.3 | Permanently disable cuFFT plan cache |
| `scripts/audit_eval_protocol.py` | (entry point) | rev44 §4 | 7-point evaluation-protocol audit (see §6 below) |
| `scripts/day1_diagnostic_sprint.py` | (entry point) | RootCause roadmap §2 | D1 metadata audit + optional D2 top-K sweep + D3 pure-MMHCL baseline (H2/H3/H10) |
| `scripts/run_phase1_ablation.py` | (entry point) | rev44 §4 | Sweep all four Phase 1 configurations × N seeds + Bonferroni-corrected paired t-tests |
| `scripts/paired_ttest.py` | `paired_ttest_report` | rev44 §4 | Paired *t*-test with optional `--bonferroni N` correction |

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
# Default invocation == rev44 / Revision 11 Phase 1 RECOMMENDED config (d):
#   --temperature 0.3   (anchor of the static τ sweep)
#   --learnable_tau 0   (τ registered as buffer, not nn.Parameter)
#   --damps_avrf 0      (AVRF off to preserve sparse Clothing signal)
python train.py --dataset Clothing --seed 42

# Reproduce the rev42 / Revision 9 baseline anchor (variant (a)):
python train.py --dataset Clothing --seed 42 \
    --temperature 0.1 --learnable_tau 1 --damps_avrf 1

# Tiktok with audio modality
python train.py --dataset Tiktok --seed 42

# Custom rebuild cadence + W&B logging
python train.py --dataset Sports --rebuild_R 5 --use_wandb 1
```

### 4.4 Ablation switchboard

| Flag | What it controls | Default | Why |
| ---- | ---------------- | ------- | --- |
| `--damps_apc` | Metadata-Aware APC | 1 (ON) | rev42 / rev44 |
| `--damps_avrf` | AVRF logit-clipped Wiener gate | **0** | rev44 §3 — AVRF over-attenuates sparse Clothing |
| `--damps_imcf` | Residual IMCF | 1 | rev42 / rev44 |
| `--damps_permutation_fft` | Permutation-FFT falsifiability test | 0 | rev42 §6 |
| `--damps_soft_routing` | Soft Residual-Routing into HGCN | 1 | rev42 / rev44 |
| `--damps_momentum` | Slim Momentum Encoder | 1 | rev42 / rev44 |
| `--damps_data_driven_prior` | SNR-based AVRF prior derivation | 1 | rev42 |
| `--temperature` | InfoNCE τ value (or initialisation if learnable) | **0.3** | rev44 §3 — Phase 1 static-τ anchor |
| `--learnable_tau` | 1 = `nn.Parameter` (rev42), 0 = buffer (rev44) | **0** | rev44 §3 — break τ-saturation collapse |
| `--rebuild_R` | Pattern B' rebuild frequency (epochs) | 5 | rev42 §3.1 |
| `--use_amp` | bfloat16 mixed precision | 1 | rev42 §3.3 |

> **Rev44 default change-set:** `--damps_avrf 0`, `--temperature 0.3`, `--learnable_tau 0`. These are the three knobs that define the Phase 1 recommended configuration `(d)` of the four-variant sweep. The directory naming in `_experiment_paths` now also encodes `taulearn={0,1}` so each variant lands in a distinct log folder.

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

### 5.0 Day 1 diagnostic sprint (Recall@20 vs NDCG@20 — roadmap §2)

When **NDCG@20 already passes** the Phase 1 stop-gate but **Recall@20 lags**, run the cheap probes from `Phase1_RootCause_Analysis_and_Remediation_Roadmap.tex` **before** spending GPU weeks on Phase 2:

| Step | What it tests | Cost |
| ---- | ------------- | ---- |
| **D1** | `meta_categories.npy` existence, length vs `n_items`, label cardinality (H2 — APC hash fallback) | CPU-only |
| **D2** | `--topk` sweep `{10,15,20}` on the combined Phase-1 recipe (H3 — hypergraph too sparse at K=5) | ~3× one full train |
| **D3** | Pure MMHCL (`--damps_* 0`, learnable τ) on the **same** all-ranking protocol (H10 — paper vs eval scale) | 1× full train |

```bash
cd MMHCL_DAMPS_Project

# 1) Always start here (exit code 1 => roadmap suspects H2):
python scripts/day1_diagnostic_sprint.py --dataset Clothing --data_path ../data/

# 2) Preview GPU commands without executing:
python scripts/day1_diagnostic_sprint.py --dataset Clothing --data_path ../data/ \
    --print-commands --run-d2 --run-d3 --run-d1-apc-probe

# 3) Execute D2 + D3 (+ optional APC-off probe) — uses seed 737791071 by default:
python scripts/day1_diagnostic_sprint.py --dataset Clothing --data_path ../data/ \
    --run-d2 --run-d3 --run-d1-apc-probe

# Quick smoke (does NOT satisfy the full statistical protocol):
python scripts/day1_diagnostic_sprint.py --dataset Clothing --epoch 75 --run-d2 --run-d3
```

Each training run is tagged with `--ablation_target diag_*` so logs land in
separate folders under `../Clothing/`. After each job the script prints the
resolved log path and parses `BEST_Test_Recall@20` / `BEST_Test_NDCG@20` when
available.

### 5.1 rev44 / Phase 1 Quick Win — running the four-variant sweep

Section 4 of `DAMPS_to_MMHCL_architecture_revision44.tex` defines four
configurations to compare across **10 random seeds**:

| Variant | Flags | Hypothesis |
| ------- | ----- | ---------- |
| (a) anchor   | `--temperature 0.1 --learnable_tau 1 --damps_avrf 1` | rev42 baseline; *τ* saturates at ~0.0909, embedding collapses |
| (b) τ-only   | `--temperature 0.3 --learnable_tau 0 --damps_avrf 1` | static τ removes saturation, AVRF kept on |
| (c) AVRF-off | `--temperature 0.1 --learnable_tau 1 --damps_avrf 0` | AVRF off recovers sparse coverage |
| (d) combined | `--temperature 0.3 --learnable_tau 0 --damps_avrf 0` | **RECOMMENDED**: union of (b) + (c) — current default |

Step 1 — **audit the eval protocol** (zero cost, blocks the sweep):

```bash
cd MMHCL_DAMPS_Project
python scripts/audit_eval_protocol.py --dataset Clothing --data_path ../data/
```

The script verifies the 7-point checklist from rev44 §4: 5-core threshold,
split fingerprint, all-ranking vs sampled, NDCG `log2(i+2)` convention, ID
remap consistency, popularity filtering at test time, and tie-break sort
stability. The audit must report `PASS` on all seven before Phase 1 may
proceed.

Step 2 — **run the four-variant sweep across 10 seeds**:

```bash
python scripts/run_phase1_ablation.py \
    --dataset Clothing \
    --seeds 42 43 44 45 46 47 48 49 50 51 \
    --variants a b c d \
    --epoch 250 --rebuild_R 5
```

Add `--aggregate_only` to skip training and just re-aggregate metrics from
existing per-run logs. The script:

1. Spawns one `train.py` invocation per `(variant, seed)` pair via
   `subprocess`, inheriting the active Python environment.
2. Parses `BEST_Test_Recall@20` / `BEST_Test_NDCG@20` from every per-run
   log file under `../<dataset>/<damps_..._taulearn=*>/`.
3. Reports the **mean ± std** (10-seed) for each variant.
4. Runs **Bonferroni-corrected paired t-tests** for variants (b), (c), (d)
   vs anchor (a) — corrected α = 0.05 / 3 ≈ 0.0167.
5. Renders the rev44 §5.2 stop-gate verdict per variant:
   * `Recall@20 ≥ 0.0900` → **PAPER VALIDATED**.
   * `Recall@20 ≥ 0.0870 ∧ NDCG@20 ≥ 0.0390` → **PHASE 1 PASS**.
   * Else → **PHASE 1 FAIL** (re-audit eval protocol).

For an *ad-hoc* paired t-test outside the sweep helper:

```bash
python scripts/paired_ttest.py \
    --damps    variant_d_recalls.csv \
    --baseline variant_a_recalls.csv \
    --bonferroni 3
```

### 5.2 Best-validation reporting

At the end of every run the trainer emits a disambiguated summary block so the values printed to the per-run text log match the maxima of the WandB curves *and* the values surfaced in `wandb.summary` exactly:

```
BEST_Val_Recall@10:        <max of val/recall@10>
BEST_Val_Recall@20:        <max of val/recall@20>     ← matches WandB val/recall@20 max
BEST_Val_Recall_Peak_Epoch:<epoch index of that maximum>
BEST_Val_NDCG@10:          <max of val/ndcg@10>
BEST_Val_NDCG@20:          <max of val/ndcg@20>
BEST_Val_NDCG_Peak_Epoch:  <epoch index of that maximum>
BEST_Test_Recall@20:       <test recall at recall-best val epoch>
BEST_Test_Precision@20:    <test precision at recall-best val epoch>
BEST_Test_NDCG@20:         <test NDCG at ndcg-best val epoch>
```

The two `BEST_Test_*` lines are pinned to the validation epoch where the *corresponding* validation metric peaked — even if a *later* epoch only improves NDCG (or only improves Recall) and overwrites the running test snapshot. Previously the code overwrote `test_ret` on every improvement of either kind, which meant `BEST_Test_Recall@K` could end up reflecting a non-recall-optimal validation epoch. The refactored tracker fixes this corner case.

#### 5.2.1 Why a chart maximum and a `BEST_Test_*` line can disagree

The most common confusion is comparing a **WandB chart maximum** to a **`BEST_Test_*`** number and concluding that something is mis-computed. This can happen for three orthogonal reasons; the trainer now defends against all three:

1. **Validation vs. test split.** `val/recall@20` (the chart with the `val/` prefix) is the **validation** Recall, used for early stopping. `BEST_Test_Recall@20` is the **test** Recall snapshotted at the validation peak. Validation recall is typically higher than test recall on Amazon Clothing — this is *not* a bug, it is the standard "select-on-val, report-on-test" methodology. To compare apples-to-apples, look at `BEST_Val_Recall@20` (text log) or `best_val_recall@20` (WandB summary), both of which exactly match the chart maximum.
2. **`epoch` vs. WandB `_step`.** WandB's default X-axis is `_step`, a counter that increments on every `wandb.log()` call. The trainer logs train metrics, val metrics, test metrics (when validation improves), and rebuild diagnostics — all separately — so `_step` runs ~1.5–2× ahead of the true training epoch. To eliminate this confusion the trainer now calls `wandb.define_metric("*", step_metric="epoch")` immediately after `wandb.init`, which makes `epoch` the canonical X-axis on every chart. `BEST_Val_Recall_Peak_Epoch` (also written to `wandb.summary["best_val_recall_peak_epoch"]`) is the epoch at which `val/recall@20` peaked — use it to verify the chart point against the headline.
3. **Multiple variants share a seed.** The rev44 Phase 1 sweep runs the same seed under four configurations (anchor / tau03 / avrf_off / combined), each producing its own WandB run with `phase1_<variant>_seed_<N>` in the run name. A chart that aggregates across runs will show maxima coming from whichever variant scored highest — which is *not* necessarily the variant whose `BEST_Test_*` line you are comparing against. Filter by the WandB run name (or by the `ablation_target` / `damps/tau_learnable` summary keys) before reading off the maximum.

The WandB `val` section also surfaces `val/ndcg@10` (alongside `val/recall@10`, `val/recall@20`, `val/ndcg@20`, `val/precision@20`, `val/hit@20`) so NDCG@10 can be inspected mid-training.

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
| Static τ (rev44 default) | `model.py::DAMPS_MMHCL.tau` | Buffer pinned at 0.3 — breaks the rev42 τ-saturation embedding collapse documented in rev44 §3 (τ stuck at 0.0909, 10.7 % Recall@20 deficit). Toggle back to learnable τ via `--learnable_tau 1`. |

---

## 7. Reproducibility & Statistical Reporting

Per §4 of the spec, every reported headline result must be averaged across **10 seeds** with **95 % confidence intervals** and **paired t-tests** versus the MMHCL baseline. The training script accepts `--seed` as a CLI flag; loop over your seed list and aggregate the per-run summaries written to `../<dataset>/MM/sum_<ablation_target>.txt`.

When the rev44 Phase 1 sweep compares **three** variants `(b), (c), (d)` against the same anchor `(a)`, the family-wise α must be Bonferroni-corrected. `scripts/paired_ttest.py` accepts `--bonferroni N` for this; `scripts/run_phase1_ablation.py` plugs in `N = 3` automatically.

A reusable helper for the paired t-test ships at `scripts/paired_ttest.py`:

```bash
python scripts/paired_ttest.py \
    --damps damps_seeds.csv \
    --baseline mmhcl_seeds.csv \
    --bonferroni 3 \
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
