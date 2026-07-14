"""
=======================================================================
  rq2_wandb_audit.py -- Reliability audit for the RQ2 ablation on
                        Amazon Clothing (KSE 2026 PACER paper).
=======================================================================

Pulls run histories & configs from Weights & Biases and answers the
three diagnostic questions that were flagged after the initial ablation
returned a suspicious "-BatchN improves the model" pattern:

  Q1. Was `--branchA_bcl_batchn` sampled by Optuna in Section 9.7, or
      hardcoded to 1 across all trials?
  Q2. When Batch-N is ON (full_pacer, no_logq, no_simgcl), do the seeds
      early-stop meaningfully earlier than when it is OFF (no_batchn)?
  Q3. Do Batch-N-ON runs show overfitting-like curves -- train/loss
      keeps dropping while val/recall@20 plateaus -- compared to
      Batch-N-OFF?

Output artefacts (default: ./rq2_audit_out/):
  q1_optuna_bcl_batchn.{json,txt}       -- Q1 answer
  q2_stop_epoch_analysis.{csv,txt}      -- Q2 answer
  q3_history_<variant>.csv              -- raw history per variant
  q3_summary_stats.{csv,txt}            -- Q3 aggregate stats
  q3_train_loss_curves.png              -- train/loss vs epoch
  q3_val_recall_curves.png              -- val/recall@20 vs epoch
  q3_generalization_gap.png             -- gap plot

Usage:
  # Inside Colab (after cell 6 wandb.login):
  !python rq2_wandb_audit.py

  # Locally:
  export WANDB_API_KEY=<your key>
  python rq2_wandb_audit.py --entity baitapck51cc-uet \
                            --project damps-mmhcl-clothing \
                            --out_dir rq2_audit_out
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------
#  Defaults (match Cells 6, 42, 44, 46 of the notebook)
# ---------------------------------------------------------------------
DEFAULT_ENTITY: str  = "baitapck51cc-uet"
DEFAULT_PROJECT: str = "damps-mmhcl-clothing"

OPTUNA_GROUP: str = "wave2_optuna_clothing"
RQ2_GROUP:    str = "wave2_rq2_ablation_clothing_5seed"

VARIANT_TAGS: list[str] = ["full_pacer", "no_logq", "no_simgcl", "no_batchn"]

# Metrics we want from every training run's history
HIST_METRICS: list[str] = [
    "epoch",
    "train/loss",
    "train/mf_loss",
    "train/emb_loss",
    "train/cl_loss",
    "train/view_loss",
    "val/recall@20",
    "val/ndcg@20",
    "best_recall",
    "best_ndcg",
]


# =====================================================================
#                             HELPERS
# =====================================================================
def _lazy_imports():
    """Import heavy deps lazily so `--help` works without wandb."""
    global wandb, pd, np, plt
    import wandb                          # noqa: F401  # noqa: E402
    import pandas as pd                   # noqa: F401  # noqa: E402
    import numpy as np                    # noqa: F401  # noqa: E402
    import matplotlib.pyplot as plt       # noqa: F401  # noqa: E402
    return wandb, pd, np, plt


def _print_hdr(msg: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {msg}")
    print("=" * 72)


def _fetch_history(run, keys: list[str], samples: int = 5000):
    """Return a DataFrame with the requested history keys, or None."""
    import pandas as pd
    try:
        rows = list(run.scan_history(keys=keys, page_size=1000))
    except Exception:
        # Fallback to sampled history if scan is unavailable
        rows = run.history(samples=samples, keys=keys, pandas=False)
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)
    # drop rows where every metric is NaN except epoch
    non_epoch = [c for c in df.columns if c != "epoch"]
    if non_epoch:
        df = df.dropna(subset=non_epoch, how="all")
    return df


def _variant_of(run) -> str | None:
    """Extract variant name from tags: full_pacer / no_logq / no_simgcl / no_batchn."""
    tags = set(run.tags or [])
    for v in VARIANT_TAGS:
        if v in tags:
            return v
    # Fallback: try run name pattern rq2_<variant>_seed<i>
    for v in VARIANT_TAGS:
        if run.name and v in run.name:
            return v
    return None


# =====================================================================
#                Q1  --  Optuna sweep space audit
# =====================================================================
def audit_q1(api, entity: str, project: str, out_dir: Path) -> dict[str, Any]:
    _print_hdr("Q1  Was `--branchA_bcl_batchn` swept by Optuna, or fixed?")

    filters = {"group": OPTUNA_GROUP}
    runs = list(api.runs(f"{entity}/{project}", filters=filters, per_page=200))
    trial_runs = [r for r in runs if "optuna_controller" not in (r.tags or [])]

    print(f"  Fetched {len(runs)} runs in group '{OPTUNA_GROUP}' "
          f"({len(trial_runs)} trial runs after filtering out controllers).")

    bcl_batchn_values: list[Any] = []
    trials_meta: list[dict[str, Any]] = []
    for r in trial_runs:
        cfg = dict(r.config or {})
        # Field can live in either `branchA_bcl_batchn` (parsed CLI) or under
        # a nested 'params/hparams' -- inspect both.
        val = cfg.get("branchA_bcl_batchn", None)
        if val is None and "params" in cfg and isinstance(cfg["params"], dict):
            val = cfg["params"].get("branchA_bcl_batchn", None)
        bcl_batchn_values.append(val)
        trials_meta.append({
            "run_id":               r.id,
            "run_name":             r.name,
            "state":                r.state,
            "branchA_bcl_batchn":   val,
            "branchA_view_every_k": cfg.get("branchA_view_every_k"),
            "lr":                   cfg.get("lr"),
            "batch_size":           cfg.get("batch_size"),
            "regs":                 cfg.get("regs"),
            "temperature":          cfg.get("temperature"),
            "lambda_view":          cfg.get("lambda_view"),
            "simgcl_eps":           cfg.get("simgcl_eps"),
            "logq_scale":           cfg.get("logq_scale"),
        })

    unique_vals = sorted({str(v) for v in bcl_batchn_values})
    swept = len(unique_vals) > 1

    result = {
        "group":           OPTUNA_GROUP,
        "n_trial_runs":    len(trial_runs),
        "bcl_batchn_unique_values":   unique_vals,
        "swept_by_optuna": swept,
        "verdict":         (
            "Batch-N WAS swept by Optuna (multiple values observed)."
            if swept else
            "Batch-N was HARDCODED to a single value across all trials -- "
            "Optuna never explored bcl_batchn=0. Config t0030 is therefore a "
            "local optimum of a restricted search space; a re-sweep with "
            "bcl_batchn as a categorical HP is necessary before drawing any "
            "conclusion about its contribution."
        ),
    }
    (out_dir / "q1_optuna_bcl_batchn.json").write_text(json.dumps(
        {"result": result, "trials": trials_meta}, indent=2, default=str))
    with (out_dir / "q1_optuna_bcl_batchn.txt").open("w") as f:
        f.write(f"Q1. Optuna sweep audit for `--branchA_bcl_batchn`\n")
        f.write(f"-" * 72 + "\n")
        f.write(f"Group:                 {OPTUNA_GROUP}\n")
        f.write(f"Trial runs found:      {len(trial_runs)}\n")
        f.write(f"Unique bcl_batchn:     {unique_vals}\n")
        f.write(f"Swept?:                {swept}\n\n")
        f.write(f"VERDICT: {result['verdict']}\n")

    print(f"  bcl_batchn unique values across {len(trial_runs)} trials: {unique_vals}")
    print(f"  Swept?  ==>  {swept}")
    print(f"  VERDICT: {result['verdict']}")
    return result


# =====================================================================
#           Q2  --  Early-stopping epoch across RQ2 variants
# =====================================================================
def audit_q2(api, entity: str, project: str, out_dir: Path) -> dict[str, Any]:
    _print_hdr("Q2  Do Batch-N ON runs early-stop earlier than Batch-N OFF?")
    import pandas as pd
    import numpy as np

    runs = list(api.runs(f"{entity}/{project}",
                         filters={"group": RQ2_GROUP},
                         per_page=200))
    # exclude controller run
    runs = [r for r in runs if _variant_of(r) is not None]
    print(f"  Fetched {len(runs)} RQ2 seed runs in '{RQ2_GROUP}'.")

    rows: list[dict[str, Any]] = []
    for r in runs:
        variant = _variant_of(r)
        cfg = dict(r.config or {})
        max_epoch = None
        best_epoch_recall = None
        best_epoch_ndcg = None
        best_recall = None
        best_ndcg = None
        n_hist_rows = 0

        hist = _fetch_history(r, keys=["epoch", "val/recall@20", "val/ndcg@20",
                                       "best_recall", "best_ndcg"])
        if hist is not None and len(hist) > 0:
            n_hist_rows = len(hist)
            if "epoch" in hist.columns:
                max_epoch = int(hist["epoch"].max())
            if "val/recall@20" in hist.columns and hist["val/recall@20"].notna().any():
                idx = hist["val/recall@20"].idxmax()
                best_epoch_recall = int(hist.loc[idx, "epoch"])
                best_recall = float(hist.loc[idx, "val/recall@20"])
            if "val/ndcg@20" in hist.columns and hist["val/ndcg@20"].notna().any():
                idx = hist["val/ndcg@20"].idxmax()
                best_epoch_ndcg = int(hist.loc[idx, "epoch"])
                best_ndcg = float(hist.loc[idx, "val/ndcg@20"])

        rows.append({
            "variant":            variant,
            "run_name":           r.name,
            "run_id":             r.id,
            "seed":               cfg.get("seed"),
            "state":              r.state,
            "n_hist_rows":        n_hist_rows,
            "last_epoch":         max_epoch,
            "best_epoch_recall":  best_epoch_recall,
            "best_epoch_ndcg":    best_epoch_ndcg,
            "best_recall@20":     best_recall,
            "best_ndcg@20":       best_ndcg,
            "batchN":             cfg.get("branchA_bcl_batchn"),
            "runtime_sec":        (r.summary.get("_runtime")
                                   if hasattr(r, "summary") else None),
        })

    df = pd.DataFrame(rows).sort_values(["variant", "seed"]).reset_index(drop=True)
    csv_path = out_dir / "q2_stop_epoch_analysis.csv"
    df.to_csv(csv_path, index=False)

    # Aggregate per variant
    agg = (df.groupby("variant", dropna=True)
             .agg(n_seeds=("run_name", "count"),
                  last_epoch_mean=("last_epoch", "mean"),
                  last_epoch_std=("last_epoch", "std"),
                  best_epoch_recall_mean=("best_epoch_recall", "mean"),
                  best_epoch_ndcg_mean=("best_epoch_ndcg", "mean"),
                  runtime_h_mean=("runtime_sec",
                                  lambda s: (s.mean() / 3600.0)
                                            if s.notna().any() else np.nan))
             .reindex(VARIANT_TAGS))
    print(agg.to_string())

    with (out_dir / "q2_stop_epoch_analysis.txt").open("w") as f:
        f.write("Q2. Early-stopping epoch across the 4 RQ2 variants (Amazon Clothing)\n")
        f.write("-" * 72 + "\n\n")
        f.write("Per-variant aggregate (mean over 5 seeds):\n")
        f.write(agg.to_string() + "\n\n")
        f.write("Interpretation:\n")
        f.write("  If last_epoch_mean is *lower* for full_pacer / no_logq /\n")
        f.write("  no_simgcl than for no_batchn, then Batch-N is triggering the\n")
        f.write("  patience-based early stop meaningfully earlier -- suggesting\n")
        f.write("  the (batch-N) contrastive loss creates a sharper best_recall\n")
        f.write("  ridge that the patience=20 counter clears before the model\n")
        f.write("  has settled.\n\n")
        f.write("Per-seed detail:\n")
        f.write(df.to_string(index=False) + "\n")

    # Convenience print
    if "no_batchn" in agg.index and agg["last_epoch_mean"].notna().any():
        off = agg.loc["no_batchn", "last_epoch_mean"]
        on_variants = [v for v in ["full_pacer", "no_logq", "no_simgcl"] if v in agg.index]
        on_mean = agg.loc[on_variants, "last_epoch_mean"].mean()
        gap = off - on_mean if (off == off and on_mean == on_mean) else float("nan")
        print(f"\n  Mean last-epoch (Batch-N ON):  {on_mean:.1f}")
        print(f"  Mean last-epoch (Batch-N OFF): {off:.1f}")
        print(f"  Gap (OFF - ON):                {gap:.1f} epochs")
        if gap == gap:
            if gap > 40:
                verdict = "Batch-N runs stop MUCH earlier -- suggests over-sharp loss geometry."
            elif gap > 15:
                verdict = "Batch-N runs stop moderately earlier -- worth investigating."
            else:
                verdict = "No large gap in stopping epoch -- Batch-N is not causing premature ES."
            print(f"  VERDICT: {verdict}")

    return {"csv": str(csv_path), "aggregate": agg.reset_index().to_dict(orient="records")}


# =====================================================================
#           Q3  --  Overfitting-signature comparison
# =====================================================================
def audit_q3(api, entity: str, project: str, out_dir: Path) -> dict[str, Any]:
    _print_hdr("Q3  Batch-N vs no-Batch-N loss curves -- overfitting check?")
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    runs = list(api.runs(f"{entity}/{project}",
                         filters={"group": RQ2_GROUP},
                         per_page=200))
    runs = [r for r in runs if _variant_of(r) is not None]

    per_variant_hist: dict[str, list] = defaultdict(list)
    per_run_summary: list[dict[str, Any]] = []

    for r in runs:
        variant = _variant_of(r)
        hist = _fetch_history(r, keys=HIST_METRICS)
        if hist is None or len(hist) == 0:
            continue
        hist = hist.copy()
        hist["run_name"] = r.name
        hist["variant"] = variant
        per_variant_hist[variant].append(hist)

        # per-run summary stats
        first = hist.iloc[0]
        # take the last valid train/loss row
        last_train_idx = hist["train/loss"].last_valid_index() \
            if "train/loss" in hist.columns else None
        last = hist.iloc[last_train_idx] if last_train_idx is not None else hist.iloc[-1]

        best_rec_idx = (hist["val/recall@20"].idxmax()
                        if "val/recall@20" in hist.columns
                        and hist["val/recall@20"].notna().any()
                        else None)
        best_rec_epoch = int(hist.loc[best_rec_idx, "epoch"]) if best_rec_idx is not None else None
        best_rec       = float(hist.loc[best_rec_idx, "val/recall@20"]) if best_rec_idx is not None else None

        # generalization gap proxy:
        #  gap = (initial_train_loss - final_train_loss) / initial_train_loss
        #  val_lift = (best_val_recall - initial_val_recall) / initial_val_recall
        # A large train_drop with small val_lift = overfitting-like.
        def _pct_drop(col):
            if col not in hist.columns:
                return float("nan")
            s = hist[col].dropna()
            if len(s) < 2:
                return float("nan")
            return float((s.iloc[0] - s.iloc[-1]) / max(abs(s.iloc[0]), 1e-9))

        def _pct_lift(col):
            if col not in hist.columns:
                return float("nan")
            s = hist[col].dropna()
            if len(s) < 2:
                return float("nan")
            return float((s.max() - s.iloc[0]) / max(abs(s.iloc[0]), 1e-9))

        per_run_summary.append({
            "variant":            variant,
            "run_name":           r.name,
            "seed":               r.config.get("seed"),
            "n_epochs":           int(hist["epoch"].max()) if "epoch" in hist.columns else None,
            "best_recall@20":     best_rec,
            "best_epoch_recall":  best_rec_epoch,
            "train_loss_drop_pct":     _pct_drop("train/loss"),
            "cl_loss_drop_pct":        _pct_drop("train/cl_loss"),
            "view_loss_drop_pct":      _pct_drop("train/view_loss"),
            "val_recall_lift_pct":     _pct_lift("val/recall@20"),
            "val_ndcg_lift_pct":       _pct_lift("val/ndcg@20"),
            "overfit_ratio":           (_pct_drop("train/loss") /
                                        max(_pct_lift("val/recall@20"), 1e-9))
                                        if _pct_lift("val/recall@20") == _pct_lift("val/recall@20")
                                        else float("nan"),
        })

    # ---- Save raw histories per variant ----
    for v, dfs in per_variant_hist.items():
        if not dfs:
            continue
        cat = pd.concat(dfs, ignore_index=True)
        cat.to_csv(out_dir / f"q3_history_{v}.csv", index=False)

    # ---- Save per-run + aggregate summaries ----
    sdf = pd.DataFrame(per_run_summary).sort_values(["variant", "seed"]).reset_index(drop=True)
    sdf.to_csv(out_dir / "q3_summary_per_run.csv", index=False)

    numeric_cols = [c for c in sdf.columns
                    if c not in {"variant", "run_name", "seed"}
                    and pd.api.types.is_numeric_dtype(sdf[c])]
    agg = (sdf.groupby("variant")[numeric_cols]
              .agg(["mean", "std"])
              .reindex(VARIANT_TAGS))
    agg.to_csv(out_dir / "q3_summary_stats.csv")

    with (out_dir / "q3_summary_stats.txt").open("w") as f:
        f.write("Q3. Overfitting-signature comparison (Batch-N ON vs OFF)\n")
        f.write("-" * 72 + "\n\n")
        f.write("Per-run detail:\n")
        f.write(sdf.to_string(index=False) + "\n\n")
        f.write("Aggregate (mean, std over 5 seeds):\n")
        f.write(agg.to_string() + "\n\n")
        f.write("Interpretation:\n")
        f.write("  overfit_ratio = train_loss_drop_pct / val_recall_lift_pct\n")
        f.write("    * ~1.0  : train and val improve proportionally  ->  healthy\n")
        f.write("    * >>1.0 : train drops much more than val improves  ->  overfitting-like\n")
        f.write("    * <<1.0 : val gains more than train drops  ->  underfitting / regularised\n\n")
        f.write("  If full_pacer / no_logq / no_simgcl show overfit_ratio >>\n")
        f.write("  no_batchn's ratio, then Batch-N is producing a sharper train\n")
        f.write("  loss surface that doesn't transfer to val Recall -- consistent\n")
        f.write("  with the observed ablation result.\n")

    print("\nPer-run summary:")
    print(sdf.to_string(index=False))
    print("\nAggregate (mean over 5 seeds):")
    print(agg["overfit_ratio"].to_string() if "overfit_ratio" in agg.columns else "no data")

    # =================================================================
    #  Plots  --  train/loss, val/recall@20, generalization gap
    # =================================================================
    def _plot_curve(metric: str, ylabel: str, fname: str,
                    log_y: bool = False) -> None:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
        colors = {"full_pacer": "#20808D", "no_logq": "#A84B2F",
                  "no_simgcl": "#944454", "no_batchn": "#1B474D"}
        styles = {"full_pacer": "-", "no_logq": "--",
                  "no_simgcl": ":",  "no_batchn": "-."}
        for v in VARIANT_TAGS:
            dfs = per_variant_hist.get(v, [])
            if not dfs:
                continue
            # Concat, then compute mean over seeds per epoch
            cat = pd.concat(dfs, ignore_index=True)
            if metric not in cat.columns:
                continue
            grp = (cat.groupby("epoch")[metric]
                     .agg(["mean", "std"])
                     .reset_index())
            ax.plot(grp["epoch"], grp["mean"],
                    color=colors[v], linestyle=styles[v], linewidth=2, label=v)
            ax.fill_between(grp["epoch"],
                            grp["mean"] - grp["std"],
                            grp["mean"] + grp["std"],
                            color=colors[v], alpha=0.15)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{ylabel} vs epoch -- mean +/- std over 5 seeds",
                     fontsize=12)
        if log_y:
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {fname}")

    _plot_curve("train/loss",     "train/loss",     "q3_train_loss_curves.png", log_y=True)
    _plot_curve("val/recall@20",  "val/recall@20",  "q3_val_recall_curves.png")
    _plot_curve("val/ndcg@20",    "val/ndcg@20",    "q3_val_ndcg_curves.png")
    _plot_curve("train/cl_loss",  "train/cl_loss",  "q3_cl_loss_curves.png", log_y=True)
    _plot_curve("train/view_loss","train/view_loss","q3_view_loss_curves.png", log_y=True)

    # Generalization-gap plot: (train_loss normalised) vs (val_recall normalised)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    for v in VARIANT_TAGS:
        dfs = per_variant_hist.get(v, [])
        if not dfs:
            continue
        cat = pd.concat(dfs, ignore_index=True)
        if not {"epoch", "train/loss", "val/recall@20"}.issubset(cat.columns):
            continue
        grp = cat.groupby("epoch").agg(
            tl=("train/loss", "mean"),
            vr=("val/recall@20", "mean"),
        ).reset_index()
        # normalise: train_loss -> fraction dropped; val_recall -> fraction lifted
        tl_norm = (grp["tl"].iloc[0] - grp["tl"]) / max(abs(grp["tl"].iloc[0]), 1e-9)
        vr_norm = (grp["vr"] - grp["vr"].iloc[0]) / max(abs(grp["vr"].iloc[0]), 1e-9)
        ax.plot(grp["epoch"], tl_norm - vr_norm,
                linewidth=2, label=f"{v} (train_drop - val_lift)")
    ax.axhline(0, color="k", linewidth=1, alpha=0.4)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("(train_loss drop) - (val_recall@20 lift), normalised",
                  fontsize=11)
    ax.set_title("Generalisation gap -- higher = overfitting-like", fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "q3_generalization_gap.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved q3_generalization_gap.png")

    return {"summary_csv": str(out_dir / "q3_summary_stats.csv")}


# =====================================================================
#                              MAIN
# =====================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reliability audit for the KSE 2026 RQ2 ablation "
                    "(Amazon Clothing).")
    parser.add_argument("--entity",  default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--out_dir", default="rq2_audit_out")
    parser.add_argument("--skip_q1", action="store_true")
    parser.add_argument("--skip_q2", action="store_true")
    parser.add_argument("--skip_q3", action="store_true")
    args = parser.parse_args()

    wandb, pd, np, plt = _lazy_imports()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _print_hdr(f"W&B target: {args.entity}/{args.project}")
    print(f"  Optuna group:  {OPTUNA_GROUP}")
    print(f"  RQ2 group:     {RQ2_GROUP}")
    print(f"  Output dir:    {out_dir.resolve()}")

    t0 = time.time()
    api = wandb.Api(timeout=90)

    if not args.skip_q1:
        audit_q1(api, args.entity, args.project, out_dir)
    if not args.skip_q2:
        audit_q2(api, args.entity, args.project, out_dir)
    if not args.skip_q3:
        audit_q3(api, args.entity, args.project, out_dir)

    _print_hdr(f"Audit complete in {time.time()-t0:.1f}s.")
    print(f"  All artefacts written to: {out_dir.resolve()}")
    print("  Recommended next actions (based on results):")
    print("    * Q1 says 'hardcoded'  ->  rerun Optuna with bcl_batchn as HP.")
    print("    * Q2 shows large ES gap ->  raise patience / min_epochs for Batch-N.")
    print("    * Q3 shows overfit_ratio(full)>>overfit_ratio(no_batchn) ->")
    print("      Batch-N InfoNCE geometry is sharper than useful; try softening")
    print("      by (a) higher temperature, (b) lower cl_loss weight, or")
    print("      (c) drop Batch-N and keep (B, N) chunked softmax.")


if __name__ == "__main__":
    main()
