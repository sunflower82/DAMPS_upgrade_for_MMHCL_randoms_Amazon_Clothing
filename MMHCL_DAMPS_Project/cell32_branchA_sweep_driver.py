# =====================================================================
# Section 8.1 -- Branch A sweep driver  (Log Q + SimGCL batch-N, rev55)
#
#   Target: ~6 h / seed (down from >40 h dense Wave 2 audit)
#   Matrix:  seed only  (lambda_view locked at 0.05)
#   Locked Wave-1:  logq_scale=1.0, laplace, beta=1.0, clip=5.0
#   Branch A:      view_every_k=2, bcl_batchn=1, view/bcl bsz=2048
#
#   Gate (rev55 §8.1):
#     PASS  -> every seed BEST_Test_Recall@20 in [0.0900, 0.0945]
#     FAIL  -> any seed < 0.0890 -> roll back to Wave 1 LogQ-only
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

try:
    _DAMPS_DIR = DAMPS_DIR  # type: ignore[name-defined]  # noqa: F821
except NameError:
    _DAMPS_DIR = os.getcwd()
try:
    _PYTHON = PYTHON_EXE  # type: ignore[name-defined]  # noqa: F821
except NameError:
    _PYTHON = sys.executable
try:
    _DATASET = dataset  # type: ignore[name-defined]  # noqa: F821
except NameError:
    _DATASET = "Clothing"
try:
    _WB_PROJECT = wandb_project  # type: ignore[name-defined]  # noqa: F821
except NameError:
    _WB_PROJECT = "damps-mmhcl-clothing"
try:
    _WB_ENTITY = wandb_entity  # type: ignore[name-defined]  # noqa: F821
except NameError:
    _WB_ENTITY = ""

if os.path.normpath(os.getcwd()) != os.path.normpath(_DAMPS_DIR):
    os.chdir(_DAMPS_DIR)

LAMBDA_VIEW_LOCK: float = 0.05
SIMGCL_EPS: float = 0.1
N_SEEDS: int = 5
LOGQ_SCALE_LOCK: float = 1.0
LOGQ_MODE_LOCK: str = "laplace"
LOGQ_BETA_LOCK: float = 1.0
LOGQ_CLIP_LOCK: float = 5.0
VIEW_EVERY_K: int = 2
USE_BCL_BATCHN: int = 1
BRANCHA_VIEW_BSZ: int = 2048
BRANCHA_BCL_BSZ: int = 2048

GATE_MIN_RECALL: float = 0.0900
GATE_MAX_RECALL: float = 0.0945
GATE_FAIL_BELOW: float = 0.0890

BACK_TO_PATIENCE_30: bool = False
PATIENCE: int = 30 if BACK_TO_PATIENCE_30 else 20
MIN_EPOCHS: int = 75


def _as_int_seed(s: Any) -> int:
    return int(round(float(s)))


try:
    SEEDS = [_as_int_seed(s) for s in list(seeds)[:N_SEEDS]]  # type: ignore[name-defined]  # noqa: F821
    if len(SEEDS) < N_SEEDS:
        raise NameError
except NameError:
    random.seed(2026)
    SEEDS = [random.randint(1, 2_147_483_646) for _ in range(N_SEEDS)]

print(f"[BranchA] seeds (int) = {SEEDS}")
print(f"[BranchA] lambda_view lock = {LAMBDA_VIEW_LOCK}")
print(f"[BranchA] view_every_k={VIEW_EVERY_K}  bcl_batchn={USE_BCL_BATCHN}")

BACKBONE_FLAGS: list[str] = [
    "--dataset", str(_DATASET),
    "--gpu_id", "0",
    "--epoch", "500",
    "--verbose", "5",
    "--batch_size", "4096",
    "--lr", "0.001",
    "--regs", "0.001",
    "--clip_grad_norm", "1.0",
    "--embed_size", "64",
    "--topk", "5",
    "--core", "5",
    "--User_layers", "3",
    "--Item_layers", "2",
    "--user_loss_ratio", "0.03",
    "--item_loss_ratio", "0.07",
    "--temperature", "0.3",
    "--learnable_tau", "0",
    "--early_stopping_patience", str(PATIENCE),
    "--early_stopping_min_epochs", str(MIN_EPOCHS),
    "--early_stopping_min_delta", "0.0001",
    "--early_stopping_mode", "max",
    "--early_stopping_restore_best", "1",
    "--use_reduce_lr", "1",
    "--reduce_lr_factor", "0.5",
    "--reduce_lr_patience", "3",
    "--reduce_lr_min", "1e-06",
    "--damps_apc", "0",
    "--damps_avrf", "0",
    "--damps_imcf", "1",
    "--damps_permutation_fft", "0",
    "--damps_soft_routing", "1",
    "--damps_momentum", "1",
    "--damps_data_driven_prior", "1",
    "--damps_num_categories", "10",
    "--damps_warmup_epochs", "10",
    "--rebuild_R", "5",
    "--faiss_threshold", "60000",
    "--faiss_use_gpu", "1",
    "--knn_chunk_size", "4096",
    "--knn_efsearch", "64",
    "--use_amp", "1",
    "--enable_logq", "1",
    "--logq_mode", LOGQ_MODE_LOCK,
    "--logq_beta", str(LOGQ_BETA_LOCK),
    "--logq_scale", str(LOGQ_SCALE_LOCK),
    "--logq_clip", str(LOGQ_CLIP_LOCK),
    "--enable_simgcl", "1",
    "--simgcl_eps", str(SIMGCL_EPS),
    "--lambda_view", str(LAMBDA_VIEW_LOCK),
    "--simgcl_batch_size_user", str(BRANCHA_VIEW_BSZ),
    "--simgcl_batch_size_item", str(BRANCHA_VIEW_BSZ),
    "--branchA_view_every_k", str(VIEW_EVERY_K),
    "--branchA_bcl_batchn", str(USE_BCL_BATCHN),
    "--branchA_view_bsz", str(BRANCHA_VIEW_BSZ),
    "--branchA_bcl_bsz", str(BRANCHA_BCL_BSZ),
]

_S = r"[^0-9\-]*(-?[0-9]*\.?[0-9]+)"
_PAT = {
    "recall@20": re.compile(r"BEST_Test_Recall@20" + _S),
    "ndcg@20": re.compile(r"BEST_Test_NDCG@20" + _S),
    "precision@20": re.compile(r"BEST_Test_Precision@20" + _S),
    "val_recall@20": re.compile(r"BEST_Val_Recall@20" + _S),
    "val_ndcg@20": re.compile(r"BEST_Val_NDCG@20" + _S),
    "val_recall_peak_epoch": re.compile(r"BEST_Val_Recall_Peak_Epoch" + _S),
}


def _parse_metrics(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, pat in _PAT.items():
        hits = pat.findall(text)
        if hits:
            out[k] = float(hits[-1])
    return out


def run_one(seed: int) -> dict[str, Any]:
    seed_int = int(seed)
    abl_tag = f"branchA_lam{LAMBDA_VIEW_LOCK}_seed{seed_int}"
    cmd: list[str] = [
        _PYTHON, "train.py",
        *BACKBONE_FLAGS,
        "--seed", str(seed_int),
        "--use_wandb", "1",
        "--wandb_project", _WB_PROJECT,
        "--wandb_entity", _WB_ENTITY,
        "--wandb_run_name", abl_tag,
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
    r20 = m.get("recall@20")
    in_window = (
        r20 is not None
        and GATE_MIN_RECALL <= r20 <= GATE_MAX_RECALL
    )
    if not ok:
        print(f"  [WARN] seed={seed_int} rc={proc.returncode}\n{log[-400:]}")
    return {
        "seed": seed_int,
        "run_name": abl_tag,
        "ablation_target": abl_tag,
        "recall@20": r20,
        "ndcg@20": m.get("ndcg@20"),
        "precision@20": m.get("precision@20"),
        "val_recall@20": m.get("val_recall@20"),
        "val_ndcg@20": m.get("val_ndcg@20"),
        "val_recall_peak_epoch": int(m["val_recall_peak_epoch"])
        if "val_recall_peak_epoch" in m else None,
        "runtime_s": round(dt, 1),
        "runtime_h": round(dt / 3600.0, 2),
        "in_window": in_window,
        "ok": ok,
    }


all_runs: list[dict[str, Any]] = []
print("=" * 72)
print(f"Branch A SWEEP: {len(SEEDS)} seeds (lambda_view={LAMBDA_VIEW_LOCK})")
print("=" * 72)
for idx, seed in enumerate(SEEDS, start=1):
    print(f"[{idx}/{len(SEEDS)}] seed={int(seed)} ...", flush=True)
    all_runs.append(run_one(seed))

recalls = [r["recall@20"] for r in all_runs if r["recall@20"] is not None]
ndcgs = [r["ndcg@20"] for r in all_runs if r["ndcg@20"] is not None]


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


mean_r = _mean(recalls)
min_r = min(recalls) if recalls else float("nan")
max_r = max(recalls) if recalls else float("nan")
gate_pass = bool(recalls) and all(
    GATE_MIN_RECALL <= r <= GATE_MAX_RECALL for r in recalls
)
gate_fail_rollback = any(r < GATE_FAIL_BELOW for r in recalls)

print()
print("=" * 72)
print(f"{'seed':>8} {'R@20':>10} {'N@20':>10} {'hours':>8} {'window':>8}")
print("-" * 72)
for r in all_runs:
    flag = "PASS" if r.get("in_window") else "FAIL"
    print(
        f"{r['seed']:>8} "
        f"{(r['recall@20'] or 0):>10.6f} "
        f"{(r['ndcg@20'] or 0):>10.6f} "
        f"{r['runtime_h']:>8.2f} "
        f"{flag:>8}"
    )
print("-" * 72)
print(
    f"mean R@20 = {mean_r:.6f} +/- {_std(recalls):.6f}  "
    f"min={min_r:.6f}  max={max_r:.6f}"
)
print(
    f"[BranchA] gate [{GATE_MIN_RECALL}, {GATE_MAX_RECALL}]: "
    f"{'PASS' if gate_pass else 'FAIL'}"
)
if gate_fail_rollback:
    print(
        f"[BranchA] ROLLBACK: seed(s) below {GATE_FAIL_BELOW} — "
        "revert to Wave 1 LogQ-only."
    )
print("=" * 72)

os.makedirs("results", exist_ok=True)
out_json = os.path.join("results", "branchA_sweep_results.json")
summary = {
    "lambda_view": LAMBDA_VIEW_LOCK,
    "view_every_k": VIEW_EVERY_K,
    "bcl_batchn": USE_BCL_BATCHN,
    "mean_recall": round(mean_r, 6),
    "std_recall": round(_std(recalls), 6),
    "min_recall": round(min_r, 6),
    "max_recall": round(max_r, 6),
    "mean_ndcg": round(_mean(ndcgs), 6) if ndcgs else None,
    "gate_pass": gate_pass,
}
with open(out_json, "w", encoding="utf-8") as f:
    json.dump({"runs": all_runs, "summary": summary, "gate": gate_pass}, f, indent=2)
print(f"\n[BranchA] results -> {out_json}")

try:
    _wb = wandb.init(
        project=_WB_PROJECT,
        entity=_WB_ENTITY if _WB_ENTITY else None,
        name="branchA_sweep_summary",
        group="wave2_branchA",
        tags=["wave2", "branchA", "batchN", "sweep_summary"],
        job_type="sweep_summary",
        config={
            "phase": "branchA",
            "lambda_view": LAMBDA_VIEW_LOCK,
            "view_every_k": VIEW_EVERY_K,
            "bcl_batchn": USE_BCL_BATCHN,
            "seeds": SEEDS,
            "gate_min": GATE_MIN_RECALL,
            "gate_max": GATE_MAX_RECALL,
        },
        reinit=True,
    )
    _wb.log({
        "branchA/mean_recall": mean_r,
        "branchA/std_recall": _std(recalls),
        "branchA/min_recall": min_r,
        "branchA/gate_pass": int(gate_pass),
    })
    _wb.summary.update({"branchA/gate_passed": gate_pass})
    _artifact = wandb.Artifact(
        name=f"branchA_sweep_{_DATASET}",
        type="sweep_results",
    )
    _artifact.add_file(out_json, name="branchA_sweep_results.json")
    _wb.log_artifact(_artifact)
    _wb.finish()
except Exception as _wb_err:
    print(f"[W&B] Summary run failed (non-fatal): {_wb_err}")
