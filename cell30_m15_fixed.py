# =====================================================================
# Section 8.2 -- Wave 1 / M1.5 sweep driver  (FIXED for rev54)
#   Matrix:  logq_scale x logq_mode x seed  -> calls train.py per config
#   Gate:    mean BEST_Test_Recall@20 >= 0.0925  AND  every seed >= 0.0890
#
# FIXES vs the previous notebook revision
#   (1) Real LogQ activation: --enable_logq 1 + --logq_mode/--logq_beta/
#       --logq_scale/--logq_clip are now ACTUALLY appended to every cmd.
#   (2) Seeds are coerced via int(round(float(s))) before being passed as
#       --seed, eliminating the "seed=1.404632241e+09" float-truncation
#       reproducibility bug.
#   (3) BEST_Test_* (snapshot at the val-recall peak epoch) is the only
#       metric reported for the M1.5 gate, NOT mid-run epoch printouts.
#   (4) ablation_target now encodes (mode, scale, seed) so per-run log
#       directories do not collide across the sweep matrix.
#   (5) early_stopping_patience lowered from 30 -> 20 to terminate cleanly
#       after convergence (optional; flip BACK_TO_PATIENCE_30 = True to
#       reproduce the previous behaviour exactly).
# =====================================================================
from __future__ import annotations

import os
import re
import sys
import json
import time
import random
import subprocess
from typing import Any

import wandb

# ---- 0. Resolve project dirs (reuse earlier-cell globals if present) ----
try:
    _DAMPS_DIR = DAMPS_DIR
except NameError:
    _DAMPS_DIR = os.getcwd()
try:
    _PYTHON = PYTHON_EXE
except NameError:
    _PYTHON = sys.executable
try:
    _DATASET = dataset
except NameError:
    _DATASET = "Clothing"
try:
    _WB_PROJECT = wandb_project
except NameError:
    _WB_PROJECT = "damps-mmhcl-clothing"
try:
    _WB_ENTITY = wandb_entity
except NameError:
    _WB_ENTITY = ""

if os.path.normpath(os.getcwd()) != os.path.normpath(_DAMPS_DIR):
    os.chdir(_DAMPS_DIR)

# ---- 1. Sweep matrix definition --------------------------------------
LOGQ_SCALES: list[float] = [0.05, 0.1, 0.3, 1.0]   # spec-mandated scale sweep
LOGQ_MODES:  list[str]   = ["laplace"]              # add "raw","sqrt" -> 24 runs
N_SEEDS: int = 5                                    # 4 scales x 1 mode x 5 = 20 runs
LOGQ_BETA: float = 1.0
LOGQ_CLIP: float = 5.0

# M1.5 acceptance gate (rev54 Section 8.2 / Section 6)
GATE_MEAN_RECALL: float = 0.0925
GATE_MIN_RECALL:  float = 0.0890

# Optional: revert patience to 30 to reproduce the previous notebook exactly
BACK_TO_PATIENCE_30: bool = False
PATIENCE: int = 30 if BACK_TO_PATIENCE_30 else 20
MIN_EPOCHS: int = 75

# ---- 1a. Seed coercion: ALWAYS int (fixes the 1.40e+09 float bug) ----
def _as_int_seed(s: Any) -> int:
    """Tolerate floats / strings / numpy scalars; emit a clean Python int."""
    return int(round(float(s)))

try:
    SEEDS = [_as_int_seed(s) for s in list(seeds)[:N_SEEDS]]   # type: ignore[name-defined]  # noqa: F821
    if len(SEEDS) < N_SEEDS:
        raise NameError
except NameError:
    random.seed(2026)
    SEEDS = [random.randint(1, 2_147_483_646) for _ in range(N_SEEDS)]
print(f"[M1.5] seeds (int) = {SEEDS}")
print(f"[M1.5] patience    = {PATIENCE}  (BACK_TO_PATIENCE_30={BACK_TO_PATIENCE_30})")
print(f"[M1.5] W&B project = {_WB_PROJECT!r}  entity={_WB_ENTITY!r}")

# ---- 2. rev45 apc_off_combined backbone flags (only delta = LogQ) -----
# Identical to the Round-2 apc_off_combined cell, except early_stopping
# patience is lowered to 20 by default.
BACKBONE_FLAGS: list[str] = [
    "--dataset",                     str(_DATASET),
    "--gpu_id",                      "0",
    "--epoch",                       "250",
    "--verbose",                     "5",
    "--batch_size",                  "1024",
    "--lr",                          "0.0001",
    "--regs",                        "0.001",
    "--clip_grad_norm",              "1.0",
    "--embed_size",                  "64",
    "--topk",                        "5",
    "--core",                        "5",
    "--User_layers",                 "3",
    "--Item_layers",                 "2",
    "--user_loss_ratio",             "0.03",
    "--item_loss_ratio",             "0.07",
    "--temperature",                 "0.3",
    "--learnable_tau",               "0",
    "--early_stopping_patience",     str(PATIENCE),
    "--early_stopping_min_epochs",   str(MIN_EPOCHS),
    "--early_stopping_min_delta",    "0.0001",
    "--early_stopping_mode",         "max",
    "--early_stopping_restore_best", "1",
    "--use_reduce_lr",               "1",
    "--reduce_lr_factor",            "0.5",
    "--reduce_lr_patience",          "3",
    "--reduce_lr_min",               "1e-06",
    # apc_off_combined backbone
    "--damps_apc",                   "0",
    "--damps_avrf",                  "0",
    "--damps_imcf",                  "1",
    "--damps_permutation_fft",       "0",
    "--damps_soft_routing",          "1",
    "--damps_momentum",              "1",
    "--damps_data_driven_prior",     "1",
    "--damps_num_categories",        "10",
    "--damps_warmup_epochs",         "10",
    "--rebuild_R",                   "5",
    "--faiss_threshold",             "60000",
    "--faiss_use_gpu",               "1",
    "--knn_chunk_size",              "4096",
    "--knn_efsearch",                "64",
    "--use_amp",                     "1",
]

# ---- 3. Metric parser: prefer BEST_Test_* (val-peak snapshot) --------
# train.py emits:
#   BEST_Val_Recall@10 / @20
#   BEST_Val_Recall_Peak_Epoch
#   BEST_Val_NDCG@10  / @20
#   BEST_Val_NDCG_Peak_Epoch
#   BEST_Test_Recall@20 / Precision@20 / NDCG@20    <-- gate uses these
_S = r"[^0-9\-]*(-?[0-9]*\.?[0-9]+)"
_PAT = {
    "recall@20":            re.compile(r"BEST_Test_Recall@20"      + _S),
    "ndcg@20":              re.compile(r"BEST_Test_NDCG@20"        + _S),
    "precision@20":         re.compile(r"BEST_Test_Precision@20"   + _S),
    "val_recall@20":        re.compile(r"BEST_Val_Recall@20"       + _S),
    "val_ndcg@20":          re.compile(r"BEST_Val_NDCG@20"         + _S),
    "val_recall_peak_epoch":re.compile(r"BEST_Val_Recall_Peak_Epoch" + _S),
    "val_ndcg_peak_epoch":  re.compile(r"BEST_Val_NDCG_Peak_Epoch"   + _S),
}

def _parse_metrics(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, pat in _PAT.items():
        hits = pat.findall(text)
        if hits:
            out[k] = float(hits[-1])      # last write = final best snapshot
    return out

# ---- 4. Run one (mode, scale, seed) configuration --------------------
def run_one(mode: str, scale: float, seed: int) -> dict[str, Any]:
    """Launch one train.py subprocess; returns a result dict."""
    seed_int = int(seed)
    # Encode mode/scale/seed into the ablation_target so per-run log
    # directories (../<dataset>/damps_..._seed=<seed>_<ablation_target>/)
    # do not collide across sweep configurations.
    abl_tag = f"m15_logq_{mode}_s{scale}_seed{seed_int}"
    run_name = abl_tag
    cmd: list[str] = [
        _PYTHON, "train.py",
        *BACKBONE_FLAGS,
        "--seed",         str(seed_int),     # <-- ALWAYS int, never 1.40e+09
        # ---- Wave 1 LogQ activation (the only delta vs rev45) ----
        "--enable_logq",  "1",
        "--logq_mode",    str(mode),
        "--logq_beta",    str(LOGQ_BETA),
        "--logq_scale",   str(scale),
        "--logq_clip",    str(LOGQ_CLIP),
        # ---- W&B per-run logging ----
        "--use_wandb",       "1",
        "--wandb_project",   _WB_PROJECT,
        "--wandb_entity",    _WB_ENTITY,
        "--wandb_run_name",  run_name,
        "--ablation_target", abl_tag,
    ]
    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.run(
        cmd, capture_output=True, text=True, cwd=_DAMPS_DIR, env=env
    )
    dt = time.time() - t0
    log = (proc.stdout or "") + "\n" + (proc.stderr or "")
    m = _parse_metrics(log)
    ok = ("recall@20" in m) and (proc.returncode == 0)
    if not ok:
        print(f"  [WARN] mode={mode} scale={scale} seed={seed_int} "
              f"rc={proc.returncode}\n{log[-400:]}")
    return {
        "mode":                  mode,
        "scale":                 scale,
        "seed":                  seed_int,
        "run_name":              run_name,
        "ablation_target":       abl_tag,
        "recall@20":             m.get("recall@20"),
        "ndcg@20":               m.get("ndcg@20"),
        "precision@20":          m.get("precision@20"),
        "val_recall@20":         m.get("val_recall@20"),
        "val_ndcg@20":           m.get("val_ndcg@20"),
        "val_recall_peak_epoch": int(m["val_recall_peak_epoch"])
                                  if "val_recall_peak_epoch" in m else None,
        "val_ndcg_peak_epoch":   int(m["val_ndcg_peak_epoch"])
                                  if "val_ndcg_peak_epoch" in m else None,
        "runtime_s":             round(dt, 1),
        "ok":                    ok,
    }

# ---- 5. Execute the full matrix --------------------------------------
all_runs: list[dict[str, Any]] = []
total = len(LOGQ_MODES) * len(LOGQ_SCALES) * len(SEEDS)
idx = 0
print("=" * 72)
print(f"M1.5 SWEEP: {total} runs "
      f"({len(LOGQ_MODES)} mode x {len(LOGQ_SCALES)} scale x {len(SEEDS)} seed)")
print("=" * 72)
for mode in LOGQ_MODES:
    for scale in LOGQ_SCALES:
        for seed in SEEDS:
            idx += 1
            print(f"[{idx}/{total}] mode={mode} scale={scale} "
                  f"seed={int(seed)} ...", flush=True)
            all_runs.append(run_one(mode, scale, seed))

# ---- 6. Aggregate per (mode, scale) and evaluate the gate ------------
def _mean(xs): return sum(xs) / len(xs) if xs else float("nan")
def _std(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5

summary: list[dict[str, Any]] = []
for mode in LOGQ_MODES:
    for scale in LOGQ_SCALES:
        rs = [r["recall@20"] for r in all_runs
              if r["mode"] == mode and r["scale"] == scale
              and r["recall@20"] is not None]
        ns = [r["ndcg@20"] for r in all_runs
              if r["mode"] == mode and r["scale"] == scale
              and r["ndcg@20"] is not None]
        if not rs:
            continue
        mean_r, min_r = _mean(rs), min(rs)
        passed = (mean_r >= GATE_MEAN_RECALL) and (min_r >= GATE_MIN_RECALL)
        summary.append({
            "mode":        mode,
            "scale":       scale,
            "n":           len(rs),
            "recall_mean": round(mean_r, 4),
            "recall_std":  round(_std(rs), 4),
            "recall_min":  round(min_r, 4),
            "ndcg_mean":   round(_mean(ns), 4),
            "ndcg_std":    round(_std(ns), 4),
            "M1.5_pass":   passed,
        })

# ---- 7. Console report -----------------------------------------------
print("\n" + "=" * 72)
print("M1.5 GATE REPORT  (mean BEST_Test_R@20 >= 0.0925  AND  min >= 0.0890)")
print("=" * 72)
hdr = (f"{'mode':<9}{'scale':>7}{'n':>4}{'R@20 mean':>12}{'+/-':>9}"
       f"{'R@20 min':>11}{'NDCG mean':>11}{'PASS':>7}")
print(hdr)
print("-" * len(hdr))
for s in summary:
    print(f"{s['mode']:<9}{s['scale']:>7}{s['n']:>4}"
          f"{s['recall_mean']:>12.4f}{s['recall_std']:>9.4f}"
          f"{s['recall_min']:>11.4f}{s['ndcg_mean']:>11.4f}"
          f"{('YES' if s['M1.5_pass'] else 'no'):>7}")

any_pass = any(s["M1.5_pass"] for s in summary)
verdict = ("PASS" if any_pass else "FAIL")
print("\n[M1.5 VERDICT]",
      "PASS -- at least one (mode,scale) cleared the gate; "
      "promote that config to Wave 2 (M1 / SimGCL)."
      if any_pass else
      "FAIL -- no (mode,scale) cleared the gate; "
      "inspect logq_scale/clip before proceeding.")

# ---- 8. Persist raw + summary JSON for audit trail -------------------
out_dir = os.path.abspath(os.path.join("..", _DATASET))
os.makedirs(out_dir, exist_ok=True)
runs_path    = os.path.join(out_dir, "m15_logq_sweep_runs.json")
summary_path = os.path.join(out_dir, "m15_logq_sweep_summary.json")
with open(runs_path, "w", encoding="utf-8") as f:
    json.dump(all_runs, f, indent=2)
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved -> {runs_path}")
print(f"Saved -> {summary_path}")

# ---- 9. W&B sweep-level summary run ----------------------------------
print("\n[W&B] Logging sweep-level summary run ...")
wb_run = wandb.init(
    project=_WB_PROJECT,
    entity=_WB_ENTITY if _WB_ENTITY else None,
    name="m15_logq_sweep_summary",
    group="wave1_m15_logq",
    tags=["wave1", "m15", "logq", "sweep_summary"],
    job_type="sweep_summary",
    config={
        "phase":            "wave1_m15",
        "logq_modes":       LOGQ_MODES,
        "logq_scales":      LOGQ_SCALES,
        "logq_beta":        LOGQ_BETA,
        "logq_clip":        LOGQ_CLIP,
        "seeds":            SEEDS,
        "n_seeds":          N_SEEDS,
        "total_runs":       total,
        "gate_mean_recall": GATE_MEAN_RECALL,
        "gate_min_recall":  GATE_MIN_RECALL,
        "dataset":          _DATASET,
        "backbone":         "apc_off_combined",
        "temperature":      0.3,
        "patience":         PATIENCE,
    },
    reinit=True,
)
cols = ["mode", "scale", "seed", "run_name",
        "recall@20", "ndcg@20", "precision@20",
        "val_recall@20", "val_ndcg@20",
        "val_recall_peak_epoch", "val_ndcg_peak_epoch",
        "runtime_s", "ok"]
runs_table = wandb.Table(columns=cols)
for r in all_runs:
    runs_table.add_data(*[r.get(c) for c in cols])
wb_run.log({"m15/all_runs": runs_table})

agg_cols = ["mode", "scale", "n",
            "recall_mean", "recall_std", "recall_min",
            "ndcg_mean", "ndcg_std", "M1.5_pass"]
agg_table = wandb.Table(columns=agg_cols)
for s in summary:
    agg_table.add_data(*[s.get(c) for c in agg_cols])
    tag = f"{s['mode']}_s{s['scale']}"
    wb_run.log({
        f"m15/{tag}/recall_mean": s["recall_mean"],
        f"m15/{tag}/recall_std":  s["recall_std"],
        f"m15/{tag}/recall_min":  s["recall_min"],
        f"m15/{tag}/ndcg_mean":   s["ndcg_mean"],
        f"m15/{tag}/gate_pass":   int(s["M1.5_pass"]),
    })
wb_run.log({"m15/aggregate_table": agg_table})

best = max(
    summary,
    key=lambda s: (s["recall_mean"], -s["recall_std"]),
    default=None,
)
wb_run.summary.update({
    "m15/gate_passed":      any_pass,
    "m15/verdict":          verdict,
    "m15/best_recall_mean": (best["recall_mean"] if best else None),
    "m15/best_scale":       (best["scale"]       if best else None),
    "m15/best_mode":        (best["mode"]        if best else None),
    "m15/n_configs_passed": sum(1 for s in summary if s["M1.5_pass"]),
    "m15/n_configs_total":  len(summary),
})

artifact = wandb.Artifact(
    name=f"m15_logq_sweep_{_DATASET}",
    type="sweep_results",
    description="Raw per-run results + per-(mode,scale) aggregates for M1.5 LogQ sweep.",
    metadata={"dataset": _DATASET, "verdict": verdict},
)
artifact.add_file(runs_path,    name="m15_logq_sweep_runs.json")
artifact.add_file(summary_path, name="m15_logq_sweep_summary.json")
wb_run.log_artifact(artifact)
wb_run.finish()
print(f"[W&B] Sweep summary run logged: {wb_run.url}")
