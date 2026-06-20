# =====================================================================
# Section 9.2 -- Wave 2 / M1 sweep -- Post-hoc Analysis
#
#   Inputs (produced by cell32_m1_sweep_driver.py):
#     - results/m1_sweep_results.json   (machine-readable run records)
#     - logs of individual training runs under
#       <DAMPS_DIR>/<Dataset>/damps_*_seed=<seed>_<ablation_target>/*.log
#
#   Outputs:
#     - results/m1_summary_table.csv         (Excel-friendly summary)
#     - results/m1_per_seed_table.csv        (long-form, one row per run)
#     - results/m1_val_recall_curves.png     (one panel per lambda_view)
#     - results/m1_gate_verdict.json         (gate decision + recommendation)
#
#   Why this analysis cell exists (and is not optional)
#   ---------------------------------------------------
#   The M1.5 retrospective discovered three issues that mid-run epoch
#   printouts cannot reveal:
#     (a) the BEST_Test_* metric is the only one the gate cares about;
#     (b) the val_recall_peak_epoch tells us whether SimGCL is converging
#         earlier (training-cost win) or later (potential overfit) than
#         the Wave 1 baseline -- a signal worth surfacing per lambda;
#     (c) a per-seed table flags any single FAILing seed early -- the
#         rev54 gate uses min, not mean, so one bad seed sinks the lambda.
# =====================================================================
from __future__ import annotations

# ---- auto-install missing dependencies --------------------------------
import importlib
import subprocess
import sys


def _ensure(pkg: str, mod: str | None = None) -> None:
    try:
        importlib.import_module(mod or pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


_ensure("matplotlib")
_ensure("pandas")

import csv
import json
import os
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


# ---- 0. Resolve project dir (reuse the sweep driver convention) -------
try:
    _DAMPS_DIR = DAMPS_DIR                                # type: ignore[name-defined]  # noqa: F821
except NameError:
    _DAMPS_DIR = os.getcwd()
try:
    _DATASET = dataset                                    # type: ignore[name-defined]  # noqa: F821
except NameError:
    _DATASET = "Clothing"
if os.path.normpath(os.getcwd()) != os.path.normpath(_DAMPS_DIR):
    os.chdir(_DAMPS_DIR)

RESULTS = Path(_DAMPS_DIR) / "results"
RESULTS.mkdir(exist_ok=True)

SWEEP_JSON = RESULTS / "m1_sweep_results.json"
if not SWEEP_JSON.exists():
    raise FileNotFoundError(
        f"{SWEEP_JSON} not found. Run cell32_m1_sweep_driver.py first."
    )

with open(SWEEP_JSON, encoding="utf-8") as f:
    sweep = json.load(f)

runs    = sweep["runs"]
summary = sweep["summary"]


# ---- 1. Per-seed long-form table --------------------------------------
def _long_form(runs: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(runs)
    cols = ["lambda_view", "seed",
            "recall@20", "ndcg@20", "precision@20",
            "val_recall@20", "val_ndcg@20",
            "val_recall_peak_epoch", "val_ndcg_peak_epoch",
            "runtime_s", "ok", "ablation_target"]
    return df[[c for c in cols if c in df.columns]].sort_values(
        ["lambda_view", "seed"], kind="mergesort"
    ).reset_index(drop=True)


df_long = _long_form(runs)
csv_long = RESULTS / "m1_per_seed_table.csv"
df_long.to_csv(csv_long, index=False)
print(f"[M1-A] per-seed long-form table  -> {csv_long}")


# ---- 2. Per-lambda summary table --------------------------------------
def _summary_table(summary: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(summary)
    rename = {
        "lambda_view": "lambda",
        "mean_recall": "mean Recall@20",
        "std_recall":  "std Recall@20",
        "min_recall":  "min seed Recall@20",
        "mean_ndcg":   "mean NDCG@20",
        "std_ndcg":    "std NDCG@20",
        "passed":      "Pass (min >= 0.0890)",
    }
    return df.rename(columns=rename)


df_summary = _summary_table(summary)
csv_summary = RESULTS / "m1_summary_table.csv"
df_summary.to_csv(csv_summary, index=False)
print(f"[M1-A] per-lambda summary table  -> {csv_summary}")
print()
print(df_summary.to_string(index=False))


# ---- 3. Val-Recall curves (one panel per lambda_view) -----------------
# Each run writes its epoch trace to:
#   <DAMPS_DIR>/<Dataset>/damps_*_seed=<seed>_<ablation_target>/<log>.log
# We re-discover those by ablation_target tags, then extract the
# "Val_Recall@20" series printed at every --verbose epoch.
EPOCH_PAT = re.compile(
    r"Epoch\s+(\d+).+?Val_Recall@20\s*[:=]\s*(-?[0-9]*\.?[0-9]+)",
    re.DOTALL,
)


def _find_log(ablation_target: str, seed: int) -> Path | None:
    """Locate the .log produced by train.py for a given run."""
    root = Path(_DAMPS_DIR) / str(_DATASET)
    if not root.exists():
        return None
    candidates = list(root.glob(f"damps_*seed={seed}_{ablation_target}*/**/*.log"))
    if not candidates:
        # Fallback -- some configurations write to a flat directory
        candidates = list(root.glob(f"*{ablation_target}*.log"))
    return candidates[0] if candidates else None


def _trace_val_recall(log_path: Path) -> tuple[list[int], list[float]]:
    txt = log_path.read_text(encoding="utf-8", errors="replace")
    epochs: list[int] = []
    vals:   list[float] = []
    for m in EPOCH_PAT.finditer(txt):
        epochs.append(int(m.group(1)))
        vals.append(float(m.group(2)))
    return epochs, vals


fig, axes = plt.subplots(1, len(summary), figsize=(5 * len(summary), 4),
                         sharey=True)
if len(summary) == 1:
    axes = [axes]

for ax, s in zip(axes, summary):
    lam = s["lambda_view"]
    plotted = 0
    for r in runs:
        if r["lambda_view"] != lam:
            continue
        log = _find_log(r["ablation_target"], r["seed"])
        if log is None:
            continue
        epochs, vals = _trace_val_recall(log)
        if epochs:
            ax.plot(epochs, vals, alpha=0.7, label=f"seed {r['seed']}")
            plotted += 1
    ax.axhline(0.0890, color="red",   linestyle="--", linewidth=1,
               label="M1 gate (0.0890)")
    ax.axhline(0.0909, color="black", linestyle=":",  linewidth=1,
               label="Rev45 mean (0.0909)")
    ax.set_title(f"lambda_view = {lam}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Recall@20")
    if plotted == 0:
        ax.text(0.5, 0.5, "no logs found",
                ha="center", va="center", transform=ax.transAxes,
                color="grey", fontsize=11)
    else:
        ax.legend(fontsize=7, loc="lower right")

plt.tight_layout()
fig_path = RESULTS / "m1_val_recall_curves.png"
plt.savefig(fig_path, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"[M1-A] val-Recall curves         -> {fig_path}")


# ---- 4. Peak-epoch comparison (val_recall_peak_epoch) -----------------
# Tells us whether SimGCL converges earlier or later than Wave 1.
# A statistically meaningful EARLIER peak at constant Recall is the
# "free training-cost win" advertised in rev54 line 90.
peak_summary = (df_long
                .dropna(subset=["val_recall_peak_epoch"])
                .groupby("lambda_view", as_index=False)
                .agg(mean_peak_epoch=("val_recall_peak_epoch", "mean"),
                     std_peak_epoch=("val_recall_peak_epoch", "std"),
                     median_peak_epoch=("val_recall_peak_epoch", "median")))
print("\n[M1-A] Val-Recall peak-epoch per lambda:")
print(peak_summary.to_string(index=False))


# ---- 5. Final gate verdict + recommendation ---------------------------
M1_GATE = bool(summary) and all(s["passed"] for s in summary)
best = max(summary, key=lambda s: s["mean_recall"]) if summary else None
verdict = {
    "gate":               "PASS" if M1_GATE else "FAIL",
    "n_lambda":           len(summary),
    "n_lambda_passing":   sum(1 for s in summary if s["passed"]),
    "best_lambda":        best["lambda_view"] if best else None,
    "best_mean_recall":   best["mean_recall"] if best else None,
    "best_mean_ndcg":     best["mean_ndcg"]   if best else None,
    "min_seed_recall":    min((s["min_recall"] for s in summary), default=None),
    "recommendation":     (
        "Lock lambda_view = "
        f"{best['lambda_view']} and proceed to Wave 3 (M2.1 G2-lite)."
        if M1_GATE else
        "Roll back SimGCL (set --enable_simgcl 0) and keep Wave 1 LogQ-only "
        "as the carry-forward configuration. Do NOT proceed to Wave 3."
    ),
}
verdict_path = RESULTS / "m1_gate_verdict.json"
with open(verdict_path, "w", encoding="utf-8") as f:
    json.dump(verdict, f, indent=2)

print()
print("=" * 72)
print(f"[M1] verdict:  {verdict['gate']}")
print(f"[M1] {verdict['n_lambda_passing']}/{verdict['n_lambda']} "
      f"lambda values passed the min-seed >= 0.0890 gate.")
print(f"[M1] best lambda_view (arg-max mean R@20): {verdict['best_lambda']}")
print(f"[M1] best mean R@20: {verdict['best_mean_recall']}")
print(f"[M1] best mean N@20: {verdict['best_mean_ndcg']}")
print(f"[M1] recommendation: {verdict['recommendation']}")
print(f"[M1] verdict written to: {verdict_path}")
