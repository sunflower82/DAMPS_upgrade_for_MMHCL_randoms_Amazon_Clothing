# =====================================================================
# Section 9.1 -- Wave 2 / M1 sweep driver  (SimGCL view-invariance ablation)
#
#   Matrix:   lambda_view x seed  -> calls train.py per config
#   Locked:   Wave 1 winner -> logq_scale = 1.0   (laplace, beta=1.0, clip=5.0)
#   Locked:   simgcl_eps = 0.1  (Yu et al. SIGIR 2022 default; rev54 line 113)
#   Sweep:    lambda_view in {0.01, 0.05, 0.1}                (3 values)
#   Seeds:    5 seeds reused from Wave 1                       (same 5)
#   Total:    3 * 5 = 15 runs
#
#   Gate:     every seed of every lambda_view yields BEST_Test_Recall@20
#             >= 0.0890. The rev54 spec is unambiguous: the acceptance
#             criterion is "for all three lambda" -- not "for the best
#             lambda".
#
#   Why this driver intentionally LOCKS more than Wave 1
#   ----------------------------------------------------
#   Wave 1 demonstrated that ``logq_scale`` is the dominant Wave 1 lever
#   (the s=0.3 cell fell below the gate while s=0.05/0.1/1.0 cleared it).
#   Wave 2 isolates a different surface (the propagation refactor + view
#   contrast), so we MUST hold every Wave 1 lever constant to attribute
#   any movement in Recall@20 to ``lambda_view`` alone. Bundling further
#   sweeps here -- e.g. simgcl_eps, simgcl_layers -- would invalidate the
#   "geometric decoupling" assumption that justifies disabling PCGrad
#   (rev54 lines 173-176). Eps and per-layer noise are deferred to Wave 4
#   (Optuna), where PCGrad is re-enabled.
#
#   Repeat-execution safety
#   -----------------------
#   * The per-run ablation_target encodes (lambda_view, seed) so two M1
#     runs never share a log directory (the M1.5 driver discovered this
#     bug; we apply the same fix here).
#   * Seeds are coerced to int via int(round(float(s))) to avoid the
#     "seed=1.40e+09" float-truncation bug (also from the M1.5 driver).
#   * BACK_TO_PATIENCE_30=False (default 20) matches the M1.5 driver so
#     training durations stay comparable across waves.
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
    _DAMPS_DIR = DAMPS_DIR                                # type: ignore[name-defined]  # noqa: F821
except NameError:
    _DAMPS_DIR = os.getcwd()
try:
    _PYTHON = PYTHON_EXE                                  # type: ignore[name-defined]  # noqa: F821
except NameError:
    _PYTHON = sys.executable
try:
    _DATASET = dataset                                    # type: ignore[name-defined]  # noqa: F821
except NameError:
    _DATASET = "Clothing"
try:
    _WB_PROJECT = wandb_project                           # type: ignore[name-defined]  # noqa: F821
except NameError:
    _WB_PROJECT = "damps-mmhcl-clothing"
try:
    _WB_ENTITY = wandb_entity                             # type: ignore[name-defined]  # noqa: F821
except NameError:
    _WB_ENTITY = ""

if os.path.normpath(os.getcwd()) != os.path.normpath(_DAMPS_DIR):
    os.chdir(_DAMPS_DIR)


# ---- 1. Sweep matrix definition --------------------------------------
LAMBDA_VIEWS: list[float] = [0.01, 0.05, 0.1]              # rev54 spec
SIMGCL_EPS: float          = 0.1                          # SimGCL default
N_SEEDS: int               = 5
LOGQ_SCALE_LOCK: float     = 1.0                          # Wave 1 winner
LOGQ_MODE_LOCK: str        = "laplace"
LOGQ_BETA_LOCK: float      = 1.0
LOGQ_CLIP_LOCK: float      = 5.0

# M1 acceptance gate (rev54 line 175 + line 374)
GATE_MIN_RECALL: float     = 0.0890

# Optional: revert patience to 30 to reproduce the M1.5 baseline exactly
BACK_TO_PATIENCE_30: bool  = False
PATIENCE: int              = 30 if BACK_TO_PATIENCE_30 else 20
MIN_EPOCHS: int            = 75


# ---- 1a. Seed coercion -- ALWAYS int (carries forward the M1.5 fix) ----
def _as_int_seed(s: Any) -> int:
    return int(round(float(s)))


try:
    SEEDS = [_as_int_seed(s) for s in list(seeds)[:N_SEEDS]]   # type: ignore[name-defined]  # noqa: F821
    if len(SEEDS) < N_SEEDS:
        raise NameError
except NameError:
    # Reuse the same RNG stream as the M1.5 driver for cross-wave
    # comparability. DO NOT change the seed of seed_random; it is set in
    # an earlier cell.
    random.seed(2026)
    SEEDS = [random.randint(1, 2_147_483_646) for _ in range(N_SEEDS)]
print(f"[M1] seeds (int) = {SEEDS}")
print(f"[M1] patience    = {PATIENCE}  (BACK_TO_PATIENCE_30={BACK_TO_PATIENCE_30})")
print(f"[M1] W&B project = {_WB_PROJECT!r}  entity={_WB_ENTITY!r}")


# ---- 2. rev45 apc_off_combined backbone flags (frozen across all M1 runs)
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
    # apc_off_combined backbone (frozen since Phase 1 Round-2)
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
    # ---- Wave 1 LogQ lock (scale = 1.0, Wave 1 winner) ----
    "--enable_logq",                 "1",
    "--logq_mode",                   LOGQ_MODE_LOCK,
    "--logq_beta",                   str(LOGQ_BETA_LOCK),
    "--logq_scale",                  str(LOGQ_SCALE_LOCK),
    "--logq_clip",                   str(LOGQ_CLIP_LOCK),
    # ---- Wave 2 SimGCL master switch + eps (locked) ----
    "--enable_simgcl",               "1",
    "--simgcl_eps",                  str(SIMGCL_EPS),
]


# ---- 3. Metric parser (carried over from M1.5 driver) ----------------
_S = r"[^0-9\-]*(-?[0-9]*\.?[0-9]+)"
_PAT = {
    "recall@20":             re.compile(r"BEST_Test_Recall@20"        + _S),
    "ndcg@20":               re.compile(r"BEST_Test_NDCG@20"          + _S),
    "precision@20":          re.compile(r"BEST_Test_Precision@20"     + _S),
    "val_recall@20":         re.compile(r"BEST_Val_Recall@20"         + _S),
    "val_ndcg@20":           re.compile(r"BEST_Val_NDCG@20"           + _S),
    "val_recall_peak_epoch": re.compile(r"BEST_Val_Recall_Peak_Epoch" + _S),
    "val_ndcg_peak_epoch":   re.compile(r"BEST_Val_NDCG_Peak_Epoch"   + _S),
}


def _parse_metrics(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, pat in _PAT.items():
        hits = pat.findall(text)
        if hits:
            out[k] = float(hits[-1])
    return out


# ---- 4. Run one (lambda_view, seed) configuration --------------------
def run_one(lam: float, seed: int) -> dict[str, Any]:
    seed_int = int(seed)
    abl_tag = f"m1_simgcl_lam{lam}_seed{seed_int}"
    run_name = abl_tag
    cmd: list[str] = [
        _PYTHON, "train.py",
        *BACKBONE_FLAGS,
        "--seed",            str(seed_int),
        "--lambda_view",     str(lam),
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
    env["PYTHONUNBUFFERED"]  = "1"
    proc = subprocess.run(
        cmd, capture_output=True, text=True, cwd=_DAMPS_DIR, env=env
    )
    dt = time.time() - t0
    log = (proc.stdout or "") + "\n" + (proc.stderr or "")
    m = _parse_metrics(log)
    ok = ("recall@20" in m) and (proc.returncode == 0)
    if not ok:
        print(f"  [WARN] lam={lam} seed={seed_int} rc={proc.returncode}\n{log[-400:]}")
    return {
        "lambda_view":           lam,
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
total = len(LAMBDA_VIEWS) * len(SEEDS)
idx = 0
print("=" * 72)
print(f"M1 SWEEP: {total} runs ({len(LAMBDA_VIEWS)} lam x {len(SEEDS)} seed)")
print("=" * 72)
for lam in LAMBDA_VIEWS:
    for seed in SEEDS:
        idx += 1
        print(f"[{idx}/{total}] lam={lam} seed={int(seed)} ...", flush=True)
        all_runs.append(run_one(lam, seed))


# ---- 6. Aggregate per lambda_view and evaluate the gate --------------
def _mean(xs): return sum(xs) / len(xs) if xs else float("nan")
def _std(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


summary: list[dict[str, Any]] = []
gate_passes: list[bool] = []
for lam in LAMBDA_VIEWS:
    rs = [r["recall@20"] for r in all_runs
          if r["lambda_view"] == lam and r["recall@20"] is not None]
    ns = [r["ndcg@20"] for r in all_runs
          if r["lambda_view"] == lam and r["ndcg@20"] is not None]
    if not rs:
        gate_passes.append(False)
        continue
    mean_r, min_r = _mean(rs), min(rs)
    # M1 gate (rev54): min seed >= 0.0890 (per-seed, not mean)
    passed = min_r >= GATE_MIN_RECALL
    gate_passes.append(bool(passed))
    summary.append({
        "lambda_view":  lam,
        "n_seeds":      len(rs),
        "mean_recall":  round(mean_r, 6),
        "std_recall":   round(_std(rs), 6),
        "min_recall":   round(min_r, 6),
        "mean_ndcg":    round(_mean(ns), 6) if ns else None,
        "std_ndcg":     round(_std(ns), 6) if len(ns) > 1 else 0.0,
        "passed":       bool(passed),
    })


# ---- 7. Print human-readable summary + global gate decision ----------
print()
print("=" * 72)
print(f"{'lam':>6} {'n':>3} {'mean R@20':>11} {'std R@20':>10} "
      f"{'min seed':>10} {'mean N@20':>11} {'pass?':>7}")
print("-" * 72)
for s in summary:
    flag = "PASS" if s["passed"] else "FAIL"
    print(f"{s['lambda_view']:>6} {s['n_seeds']:>3} "
          f"{s['mean_recall']:>11.6f} {s['std_recall']:>10.6f} "
          f"{s['min_recall']:>10.6f} "
          f"{(s['mean_ndcg'] if s['mean_ndcg'] is not None else 0):>11.6f} "
          f"{flag:>7}")
print("=" * 72)

# Global M1 gate: ALL three lambda values must pass (rev54 line 175).
M1_GATE = bool(gate_passes) and all(gate_passes)
print(f"\n[M1] global gate (all 3 lambda pass min-seed >= 0.0890):  "
      f"{'PASS' if M1_GATE else 'FAIL'}")

if M1_GATE:
    print("[M1] -> proceed to Wave 3 (M2.1 G2-lite Sigmoid gate).")
    print(f"[M1] lock lambda_view for downstream waves: "
          f"{max(summary, key=lambda s: s['mean_recall'])['lambda_view']} "
          f"(arg-max mean R@20).")
else:
    failing = [s['lambda_view'] for s in summary if not s['passed']]
    print(f"[M1] FAILED lambda(s): {failing}.")
    print("[M1] -> deactivate SimGCL and roll back to Wave 1 LogQ-only "
          "(rev54 line 204).")


# ---- 8. Persist the raw sweep records for the post-hoc analysis cell --
os.makedirs("results", exist_ok=True)
out_json = os.path.join("results", "m1_sweep_results.json")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump({"runs": all_runs, "summary": summary, "gate": M1_GATE}, f, indent=2)
print(f"\n[M1] raw sweep records written to: {out_json}")

# ---- 9. (Optional) close any W&B leftover from a prior cell --
try:
    wandb.finish()
except Exception:
    pass
