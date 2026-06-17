# =====================================================================
# Section 8.2 -- Wave 1 / M1.5 sweep ANALYSIS  (post-hoc, read-only)
#   For each (mode, scale, seed) directory under ../<dataset>/, extract:
#       * BEST_Test_Recall@20 / NDCG@20 / Precision@20  (val-peak snapshot)
#       * BEST_Val_Recall@20 + Peak_Epoch                (early-stop proof)
#       * Full per-epoch val/recall@20 curve             (early-stop visual)
#   Plot val-recall curves so you can visually confirm NO run was cut
#   short of its recall peak.
#
# Required upstream artifacts (produced by the fixed sweep cell):
#   ../<dataset>/m15_logq_sweep_runs.json       (optional cross-check)
#   ../<dataset>/damps_..._seed=<seed>_m15_logq_<mode>_s<scale>_seed<seed>/
#       *.txt                                  (per-epoch logger output)
# =====================================================================
from __future__ import annotations

import os
import re
import json
import glob
import math
from typing import Any, Optional

import numpy as np
import matplotlib.pyplot as plt

# ---- 0. Resolve dirs --------------------------------------------------
try:
    _DAMPS_DIR = DAMPS_DIR
except NameError:
    _DAMPS_DIR = os.getcwd()
try:
    _DATASET = dataset
except NameError:
    _DATASET = "Clothing"

if os.path.normpath(os.getcwd()) != os.path.normpath(_DAMPS_DIR):
    os.chdir(_DAMPS_DIR)

LOG_ROOT = os.path.abspath(os.path.join("..", _DATASET))
print(f"[M1.5/analysis] scanning {LOG_ROOT}")

# ---- 1. Discover run directories from ablation_target naming ---------
# Pattern produced by the fixed sweep cell:
#   damps_..._seed=<seed>_m15_logq_<mode>_s<scale>_seed<seed>/<same>.txt
DIR_RE = re.compile(
    r"damps_.*?_seed=(?P<seed>\d+)_m15_logq_(?P<mode>[a-zA-Z0-9]+)_s(?P<scale>[0-9.]+)_seed\d+$"
)

run_dirs: list[dict[str, Any]] = []
for d in sorted(glob.glob(os.path.join(LOG_ROOT, "damps_*"))):
    if not os.path.isdir(d):
        continue
    name = os.path.basename(d.rstrip("/"))
    m = DIR_RE.match(name)
    if not m:
        continue
    txt_files = glob.glob(os.path.join(d, "*.txt"))
    if not txt_files:
        continue
    run_dirs.append({
        "dir":   d,
        "txt":   txt_files[0],
        "mode":  m.group("mode"),
        "scale": float(m.group("scale")),
        "seed":  int(m.group("seed")),
    })
print(f"[M1.5/analysis] found {len(run_dirs)} M1.5 LogQ run directories")
if not run_dirs:
    print("  (no logs yet -- run the fixed sweep cell first)")

# ---- 2. Regex parsers -------------------------------------------------
_S = r"[^0-9\-]*(-?[0-9]*\.?[0-9]+)"
PAT_BEST = {
    "test_recall@20":        re.compile(r"BEST_Test_Recall@20"        + _S),
    "test_ndcg@20":          re.compile(r"BEST_Test_NDCG@20"          + _S),
    "test_precision@20":     re.compile(r"BEST_Test_Precision@20"     + _S),
    "val_recall@20":         re.compile(r"BEST_Val_Recall@20"         + _S),
    "val_ndcg@20":           re.compile(r"BEST_Val_NDCG@20"           + _S),
    "val_recall_peak_epoch": re.compile(r"BEST_Val_Recall_Peak_Epoch" + _S),
    "val_ndcg_peak_epoch":   re.compile(r"BEST_Val_NDCG_Peak_Epoch"   + _S),
}
# Per-epoch line from train.py Trainer.train():
#   Epoch 17 [12.3s + 4.5s]: loss=0.12345  recall@10=0.04210  recall@20=0.07815  ndcg@20=0.03812
PAT_EPOCH = re.compile(
    r"Epoch\s+(\d+)\s*\[[^\]]+\]:\s*loss=([\d.]+)\s+"
    r"recall@\d+=([\d.]+)\s+recall@(\d+)=([\d.]+)\s+ndcg@\d+=([\d.]+)"
)
# Early-stop marker emitted by train.py
PAT_ES_TRIGGER = re.compile(r"#####\s*Early stop triggered\s*#####")


def parse_log(path: str) -> dict[str, Any]:
    """Return BEST_* snapshot, per-epoch curves, and early-stop flag."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    best: dict[str, float] = {}
    for k, pat in PAT_BEST.items():
        hits = pat.findall(text)
        if hits:
            best[k] = float(hits[-1])
    epochs:      list[int]   = []
    val_rec20:   list[float] = []
    val_ndcg20:  list[float] = []
    losses:      list[float] = []
    for m in PAT_EPOCH.finditer(text):
        epochs.append(int(m.group(1)))
        losses.append(float(m.group(2)))
        val_rec20.append(float(m.group(5)))
        val_ndcg20.append(float(m.group(6)))
    return {
        "best":       best,
        "epochs":     epochs,
        "loss":       losses,
        "val_rec20":  val_rec20,
        "val_ndcg20": val_ndcg20,
        "early_stop_triggered": bool(PAT_ES_TRIGGER.search(text)),
        "last_epoch": (epochs[-1] if epochs else None),
    }


# ---- 3. Parse all logs -----------------------------------------------
rows: list[dict[str, Any]] = []
for r in run_dirs:
    p = parse_log(r["txt"])
    best = p["best"]
    peak = int(best.get("val_recall_peak_epoch", -1))
    last = p["last_epoch"]
    margin = (last - peak) if (last is not None and peak >= 0) else None
    rows.append({
        "mode":             r["mode"],
        "scale":            r["scale"],
        "seed":             r["seed"],
        "BEST_Test_R@20":   best.get("test_recall@20"),
        "BEST_Test_NDCG@20":best.get("test_ndcg@20"),
        "BEST_Val_R@20":    best.get("val_recall@20"),
        "val_peak_epoch":   peak if peak >= 0 else None,
        "last_epoch":       last,
        "epochs_past_peak": margin,
        "early_stop":       p["early_stop_triggered"],
        "n_val_points":     len(p["val_rec20"]),
        "_curve":           p,
    })

# ---- 4. Tabular report (BEST_Test + early-stop margin) ---------------
print("\n" + "=" * 92)
print("M1.5 per-run audit  (sorted by mode, scale, seed)")
print("=" * 92)
hdr = (f"{'mode':<9}{'scale':>7}{'seed':>12}{'BEST_Test_R@20':>17}"
       f"{'BEST_Val_R@20':>16}{'peak_ep':>9}{'last_ep':>9}"
       f"{'margin':>8}{'ES?':>5}")
print(hdr); print("-" * len(hdr))
for x in sorted(rows, key=lambda r: (r["mode"], r["scale"], r["seed"])):
    print(f"{x['mode']:<9}{x['scale']:>7}{x['seed']:>12}"
          f"{(x['BEST_Test_R@20'] or float('nan')):>17.4f}"
          f"{(x['BEST_Val_R@20']  or float('nan')):>16.4f}"
          f"{(x['val_peak_epoch'] if x['val_peak_epoch'] is not None else -1):>9}"
          f"{(x['last_epoch']     if x['last_epoch']     is not None else -1):>9}"
          f"{(x['epochs_past_peak'] if x['epochs_past_peak'] is not None else -1):>8}"
          f"{('Y' if x['early_stop'] else 'n'):>5}")

# Flag suspicious cases: peak too close to last epoch (< PATIENCE)
SUSPECT_MARGIN: int = 15
suspects = [
    x for x in rows
    if x["epochs_past_peak"] is not None and x["epochs_past_peak"] < SUSPECT_MARGIN
]
if suspects:
    print(f"\n[WARN] {len(suspects)} run(s) ended within {SUSPECT_MARGIN} epochs of "
          f"their val-recall peak -- inspect curves to rule out premature stop:")
    for x in suspects:
        print(f"  mode={x['mode']} scale={x['scale']} seed={x['seed']} "
              f"peak={x['val_peak_epoch']} last={x['last_epoch']} "
              f"margin={x['epochs_past_peak']}")
else:
    print(f"\n[OK] all runs ran >= {SUSPECT_MARGIN} epochs past their val-recall peak.")

# ---- 5. Per-(mode, scale) aggregate using BEST_Test only --------------
def _agg(vals: list[Optional[float]]) -> tuple[float, float, float]:
    v = [x for x in vals if x is not None]
    if not v:
        return float("nan"), float("nan"), float("nan")
    m = sum(v) / len(v)
    s = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) >= 2 else 0.0
    return m, s, min(v)

print("\n" + "=" * 92)
print("M1.5 aggregate from PARSED LOGS  (BEST_Test_Recall@20 only)")
print("=" * 92)
key = lambda r: (r["mode"], r["scale"])
configs = sorted(set(key(r) for r in rows))
print(f"{'mode':<9}{'scale':>7}{'n':>4}{'R@20 mean':>12}{'+/-':>9}"
      f"{'R@20 min':>11}{'NDCG mean':>11}{'PASS':>7}")
print("-" * 64)
GATE_MEAN: float = 0.0925
GATE_MIN:  float = 0.0890
log_summary: list[dict[str, Any]] = []
for mode, scale in configs:
    sub = [r for r in rows if r["mode"] == mode and r["scale"] == scale]
    rm, rs, mn = _agg([r["BEST_Test_R@20"] for r in sub])
    nm, _, _  = _agg([r["BEST_Test_NDCG@20"] for r in sub])
    passed = (rm >= GATE_MEAN) and (mn >= GATE_MIN)
    log_summary.append({
        "mode": mode, "scale": scale, "n": len(sub),
        "recall_mean": round(rm, 4), "recall_std": round(rs, 4),
        "recall_min":  round(mn, 4), "ndcg_mean":   round(nm, 4),
        "M1.5_pass":   bool(passed),
    })
    print(f"{mode:<9}{scale:>7}{len(sub):>4}{rm:>12.4f}{rs:>9.4f}"
          f"{mn:>11.4f}{nm:>11.4f}{('YES' if passed else 'no'):>7}")

# ---- 6. Visual confirmation: val-recall@20 curves --------------------
if rows:
    by_scale: dict[float, list[dict[str, Any]]] = {}
    for r in rows:
        by_scale.setdefault(r["scale"], []).append(r)
    n_scales = len(by_scale)
    ncols = min(2, n_scales)
    nrows = math.ceil(n_scales / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(7.0 * ncols, 4.2 * nrows),
                             squeeze=False)
    for ax_idx, (scale, group) in enumerate(sorted(by_scale.items())):
        ax = axes[ax_idx // ncols][ax_idx % ncols]
        for x in sorted(group, key=lambda r: r["seed"]):
            curve = x["_curve"]
            if not curve["epochs"]:
                continue
            ax.plot(curve["epochs"], curve["val_rec20"],
                    linewidth=1.4, alpha=0.85,
                    label=f"seed={x['seed']} (peak ep={x['val_peak_epoch']}, "
                          f"BEST_Test={(x['BEST_Test_R@20'] or 0):.4f})")
            if x["val_peak_epoch"] is not None:
                ax.axvline(x["val_peak_epoch"], color="grey",
                           linestyle=":", linewidth=0.7, alpha=0.6)
        ax.axhline(GATE_MIN, color="red", linestyle="--",
                   linewidth=0.9, alpha=0.7, label=f"rollback {GATE_MIN}")
        ax.axhline(GATE_MEAN, color="green", linestyle="--",
                   linewidth=0.9, alpha=0.7, label=f"M1.5 gate {GATE_MEAN}")
        ax.set_title(f"M1.5 LogQ -- scale={scale}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("val/recall@20")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")
    # blank out any spare subplot
    for j in range(len(by_scale), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle("M1.5 LogQ sweep -- val/recall@20 vs epoch  "
                 "(dotted = val-recall peak; restore_best=1 means best is recovered)",
                 y=1.02, fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig_path = os.path.join(LOG_ROOT, "m15_logq_val_recall_curves.png")
    plt.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.show()
    print(f"[plot] saved -> {fig_path}")

# ---- 7. Persist parsed analysis to JSON ------------------------------
parsed_path = os.path.join(LOG_ROOT, "m15_logq_parsed_logs.json")
to_dump = []
for r in rows:
    rr = {k: v for k, v in r.items() if k != "_curve"}
    rr["epochs"]     = r["_curve"]["epochs"]
    rr["val_rec20"]  = r["_curve"]["val_rec20"]
    rr["val_ndcg20"] = r["_curve"]["val_ndcg20"]
    to_dump.append(rr)
with open(parsed_path, "w", encoding="utf-8") as f:
    json.dump({"runs": to_dump, "summary": log_summary}, f, indent=2)
print(f"[json] saved -> {parsed_path}")
