# RQ2 W&B Audit — how to run

`rq2_wandb_audit.py` reads the W&B project **`baitapck51cc-uet/damps-mmhcl-clothing`** to answer three diagnostic questions about the Amazon-Clothing RQ2 ablation:

| Question | Group audited | Answer produced |
|---|---|---|
| Q1. Was `--branchA_bcl_batchn` swept by Optuna, or hardcoded to 1? | `wave2_optuna_clothing` (35 trials) | `q1_optuna_bcl_batchn.{json,txt}` |
| Q2. Do Batch-N ON runs (full_pacer/no_logq/no_simgcl) early-stop earlier than Batch-N OFF (no_batchn)? | `wave2_rq2_ablation_clothing_5seed` (20 runs) | `q2_stop_epoch_analysis.{csv,txt}` |
| Q3. Do Batch-N ON runs show train-loss drop >> val-recall lift (overfitting-like)? | `wave2_rq2_ablation_clothing_5seed` (20 runs) | `q3_history_<variant>.csv`, `q3_summary_*.csv`, 6 PNG plots |

## Option A — run inside Colab notebook (recommended)

After Cell 6 (`wandb.login`) has completed, add a new cell:

```python
!wget -q https://YOUR_STORAGE_URL/rq2_wandb_audit.py -O rq2_wandb_audit.py
!python rq2_wandb_audit.py --out_dir rq2_audit_out
```

Or just paste the whole script into a cell and call `main()`. It reuses the login cookie set by Cell 6.

## Option B — run locally

```bash
pip install "wandb>=0.16" pandas numpy matplotlib
export WANDB_API_KEY=<your_key>          # or run `wandb login`
python rq2_wandb_audit.py \
    --entity  baitapck51cc-uet \
    --project damps-mmhcl-clothing \
    --out_dir rq2_audit_out
```

Runtime: ~2–4 minutes total (Q1: 15 s, Q2: 30 s, Q3: 1–3 min because it downloads full training histories for 20 runs).

## What each artefact tells you

### Q1 — `q1_optuna_bcl_batchn.{json,txt}`
- Reads the `config` field of every trial run in group `wave2_optuna_clothing`.
- If `bcl_batchn_unique_values == ["1"]`, Optuna never explored `bcl_batchn=0` → the sweep space was restricted → Config t0030 is a local optimum inside a subspace, not a global one. **Action:** re-run Optuna with `trial.suggest_categorical("branchA_bcl_batchn", [0, 1])`.

### Q2 — `q2_stop_epoch_analysis.{csv,txt}`
Per-run columns: `variant, seed, last_epoch, best_epoch_recall, best_epoch_ndcg, best_recall@20, best_ndcg@20, runtime_sec`.
Also prints an aggregate line comparing `mean(last_epoch)` between Batch-N ON and OFF.
- Gap > 40 epochs → Batch-N is causing meaningful premature ES.
- Gap 15–40 → worth investigating.
- Gap < 15 → ES is not the issue; Batch-N really is producing worse solutions.

### Q3 — `q3_history_*.csv`, `q3_summary_*.csv`, 6 plots
- Full training history (epoch, train/loss, train/cl_loss, train/view_loss, val/recall@20, val/ndcg@20) per variant.
- **`overfit_ratio` = train_loss_drop_pct / val_recall_lift_pct**
  - ≈1.0: healthy (train and val move together)
  - >>1.0: overfitting-like (train drops much more than val improves)
  - If **overfit_ratio(full_pacer)** ≫ **overfit_ratio(no_batchn)**, that is the smoking gun: Batch-N InfoNCE is producing a sharper loss geometry that does not transfer to val Recall.
- **Plots**:
  - `q3_train_loss_curves.png` — log-scale train/loss vs epoch (all 4 variants, ±1σ band)
  - `q3_val_recall_curves.png` — val/recall@20 vs epoch
  - `q3_val_ndcg_curves.png`   — val/ndcg@20 vs epoch
  - `q3_cl_loss_curves.png`    — contrastive-loss component (log)
  - `q3_view_loss_curves.png`  — SimGCL view-loss component (log)
  - `q3_generalization_gap.png` — `(train_drop − val_lift)` per epoch; higher = more overfitting-like

## What to look for in the plots (concrete decision rules)

| Observation | Diagnosis | Action for KSE |
|---|---|---|
| `full_pacer`'s train/loss drops faster than `no_batchn`'s, but val/recall@20 curves are similar or `no_batchn` beats `full_pacer` | Overfitting on Batch-N InfoNCE | Drop Batch-N from PACER, or soften temperature |
| Both variants have similar loss shape but `full_pacer` stops at epoch < 150 while `no_batchn` reaches 250 | Premature ES on sharper loss | Raise `--early_stopping_patience` to 40 and rerun |
| `full_pacer`'s val/recall@20 peaks then declines — `no_batchn` still climbing at 250 | Under-training of the OFF variant | 250-epoch cap is unfair; extend to 500 for OFF |
| Curves look identical, terminal metrics differ by < 1σ | Noise-level effect; Batch-N is neutral | Reframe paper: DAMPS is the contribution |

## Notes on robustness

- The script auto-detects the variant of each RQ2 run from its W&B tags (`full_pacer`, `no_logq`, `no_simgcl`, `no_batchn`). If tags were dropped, it falls back to run-name matching (`rq2_<variant>_seed<i>`).
- If any run failed (state ≠ `finished`), its row still appears — just with `last_epoch = NaN`. That's fine for the audit; it just means fewer seeds contribute to the aggregate.
- `_fetch_history` uses `run.scan_history` (unlimited) with a `run.history(samples=5000)` fallback so the raw curves are complete even for long 250-epoch runs.
