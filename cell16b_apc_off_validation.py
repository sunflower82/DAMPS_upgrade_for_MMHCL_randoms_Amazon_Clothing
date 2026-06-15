"""
rev44 Phase 1 — ROUND 2:  apc_off_combined (5-seed validation)
==============================================================

Runs immediately after the **four-variant** training cell. See
`Phase1_Day1_Final_Verdict.tex`: Day-1 APC-off probe **R@20 ≈ 0.09197** (paper
**0.0881**); **combined** on **hash-fallback** ~**0.0851** (~**−0.0069** vs
APC-off). **H3** rejected; **H10** deprioritised.

This cell trains **(e) apc_off_combined** (`--damps_apc 0`, same τ/AVRF as (d))
on the **`seeds`** list from the sweep when that cell has already run; **otherwise**
it **generates** ``n_runs`` fresh random seeds (default **5**, same scheme as §3).
**PASS Phase 1** if mean **test R@20 ≥ 0.0870** (rev44 §5.2; full NDCG verdict in §4.5). If **FAIL**, run
**`build_meta_categories.py`**, restore **`meta_categories.npy`**, keep APC on
for **combined**, and re-run Phase 1.
"""

from __future__ import annotations

import os
import random
import re
import subprocess
import sys
import time
from typing import Any

# ---------------------------------------------------------------------------
# Standalone bootstrap (no §3 cell required).
#
# IPython stores notebook variables in ``user_ns``.  Writing only into a
# nested function's ``globals()`` view can leave ``seeds`` invisible to the
# rest of this cell on some builds — so we update **both** ``user_ns`` and
# ``globals()`` here.
# ---------------------------------------------------------------------------
try:
    _USER_NS: dict[str, Any] = get_ipython().user_ns  # type: ignore[name-defined]
except Exception:
    _USER_NS = globals()

_boot: dict[str, Any] = {}

if "PROJECT_ROOT" not in _USER_NS or "DAMPS_DIR" not in _USER_NS:
    _cd = os.getcwd()
    if _cd.endswith(("codes", "MMHCL_DAMPS_Project")):
        _boot["PROJECT_ROOT"] = os.path.dirname(_cd)
    else:
        _boot["PROJECT_ROOT"] = _cd
    _boot["DAMPS_DIR"] = os.path.join(_boot["PROJECT_ROOT"], "MMHCL_DAMPS_Project")

if "seeds" not in _USER_NS or not _USER_NS.get("seeds"):
    _n = int(_USER_NS.get("n_runs", 5))
    _bs = int(time.time_ns() % (2**31))
    random.seed(_bs)
    _boot["n_runs"] = _n
    _boot["base_seed"] = _bs
    _boot["seeds"] = [random.randint(1, 2**31 - 1) for _ in range(_n)]
    print(
        f"[apc_off_combined] standalone: generated {_n} random seeds "
        f"(base_seed={_bs})"
    )
    print(f"  seeds = {_boot['seeds']}")

_defaults: dict[str, Any] = {
    "dataset": "Clothing",
    "PYTHON_EXE": sys.executable,
    "wandb_project": "damps-mmhcl-clothing",
    "wandb_entity": "baitapck51cc-uet",
    "gpu_id": 0,
    "epoch": 250,
    "verbose": 5,
    "batch_size": 1024,
    "lr": 0.0001,
    "regs": 1e-3,
    "embed_size": 64,
    "topk": 5,
    "core": 5,
    "User_layers": 3,
    "Item_layers": 2,
    "user_loss_ratio": 0.03,
    "item_loss_ratio": 0.07,
    "clip_grad_norm": 1.0,
    "early_stopping_patience": 30,
    "early_stopping_min_epochs": 75,
    "early_stopping_min_delta": 0.0001,
    "early_stopping_mode": "max",
    "early_stopping_restore_best": 1,
    "use_reduce_lr": 1,
    "reduce_lr_factor": 0.5,
    "reduce_lr_patience": 3,
    "reduce_lr_min": 1e-6,
    "damps_imcf": 1,
    "damps_permutation_fft": 0,
    "damps_soft_routing": 1,
    "damps_momentum": 1,
    "damps_data_driven_prior": 1,
    "damps_num_categories": 10,
    "damps_warmup_epochs": 10,
    "rebuild_R": 5,
    "faiss_threshold": 60_000,
    "faiss_use_gpu": 1,
    "knn_chunk_size": 4096,
    "knn_efsearch": 64,
    "use_amp": 1,
    "use_torch_compile": 1,
    "torch_compile_mode": "reduce-overhead",
    "torch_compile_dynamic": 1,
}
for _k, _v in _defaults.items():
    if _k not in _USER_NS and _k not in _boot:
        _boot[_k] = _v

if "all_variant_results" not in _USER_NS:
    _boot["all_variant_results"] = {}
if "phase1_variants" not in _USER_NS:
    _boot["phase1_variants"] = {}

_USER_NS.update(_boot)
globals().update(_boot)

PROJECT_ROOT = _USER_NS["PROJECT_ROOT"]
DAMPS_DIR = _USER_NS["DAMPS_DIR"]
PYTHON_EXE = _USER_NS["PYTHON_EXE"]
dataset = _USER_NS["dataset"]
seeds = _USER_NS["seeds"]

if os.path.normpath(os.getcwd()) != os.path.normpath(DAMPS_DIR):
    os.chdir(DAMPS_DIR)
if DAMPS_DIR not in sys.path:
    sys.path.insert(0, DAMPS_DIR)

# ---------------------------------------------------------------------------
# rev44 Phase 1 -- 5th variant config
#   Equal to (d) combined  PLUS  --damps_apc 0
# ---------------------------------------------------------------------------
variant_name: str = "apc_off_combined"
variant_label: str = "(e) static tau + AVRF off + APC OFF  (ROUND-2 Phase 1 candidate)"

v_temp:      float = 0.3   # static tau anchor
v_learn_tau: int   = 0     # static (not learnable)
v_avrf:      int   = 0     # AVRF off
v_apc:       int   = 0     # NEW: APC off  (overrides cell-16 `damps_apc = 1`)

print("\n" + "#" * 80)
print(f"# VARIANT '{variant_name}' -- {variant_label}")
print(f"#   --temperature {v_temp} --learnable_tau {v_learn_tau} "
      f"--damps_avrf {v_avrf} --damps_apc {v_apc}")
print("#" * 80)
print(f"\n  seeds          : {seeds}")
print(f"  runs           : {len(seeds)}  (single-variant 5-seed validation)")
print(f"  expected wall  : ~3 hours on RTX 5090 (bfloat16 + torch.compile)")
print(f"  stop-gate      : mean(R@20) >= 0.0870 to PASS Phase 1")

# Storage for this variant only.  Will be MERGED into the global
# `all_variant_results` from cell 16 so the §4 / §4.5 cells pick it up.
_apc_off_results: list[dict[str, Any]] = []

# Regex pre-compiled to extract metrics from the per-run log
_sep: str = r"[:=]\s*"
_re_test_r20  = re.compile(rf"BEST_Test_Recall@20{_sep}([\d.]+)")
_re_test_n20  = re.compile(rf"BEST_Test_NDCG@20{_sep}([\d.]+)")
_re_test_p20  = re.compile(rf"BEST_Test_Precision@20{_sep}([\d.]+)")
_re_val_r20   = re.compile(rf"BEST_Val_Recall@20{_sep}([\d.]+)")
_re_val_n20   = re.compile(rf"BEST_Val_NDCG@20{_sep}([\d.]+)")
_re_val_rpeak = re.compile(rf"BEST_Val_Recall_Peak_Epoch{_sep}(\d+)")
_re_val_npeak = re.compile(rf"BEST_Val_NDCG_Peak_Epoch{_sep}(\d+)")

for run_idx, seed in enumerate(seeds, 1):
    print(f"\n{'=' * 80}")
    print(f"RUN {run_idx}/{len(seeds)}  (variant '{variant_name}' seed={seed})")
    print(f"{'=' * 80}")

    cmd: list[str] = [
        PYTHON_EXE, "train.py",
        # Data / schedule
        "--dataset", dataset,
        "--gpu_id", str(gpu_id),
        "--seed", str(seed),
        "--epoch", str(epoch),
        "--verbose", str(verbose),
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--regs", str(regs),
        "--clip_grad_norm", str(clip_grad_norm),
        # Model
        "--embed_size", str(embed_size),
        "--topk", str(topk),
        "--core", str(core),
        "--User_layers", str(User_layers),
        "--Item_layers", str(Item_layers),
        "--user_loss_ratio", str(user_loss_ratio),
        "--item_loss_ratio", str(item_loss_ratio),
        # rev44 Round-2 overrides
        "--temperature", str(v_temp),
        "--learnable_tau", str(v_learn_tau),
        # Early stopping
        "--early_stopping_patience",   str(early_stopping_patience),
        "--early_stopping_min_epochs", str(early_stopping_min_epochs),
        "--early_stopping_min_delta",  str(early_stopping_min_delta),
        "--early_stopping_mode",       str(early_stopping_mode),
        "--early_stopping_restore_best", str(early_stopping_restore_best),
        # ReduceLROnPlateau
        "--use_reduce_lr",      str(use_reduce_lr),
        "--reduce_lr_factor",   str(reduce_lr_factor),
        "--reduce_lr_patience", str(reduce_lr_patience),
        "--reduce_lr_min",      str(reduce_lr_min),
        # DAMPS  --  THIS is where the variant differs from (d)
        "--damps_apc",            str(v_apc),                 # <-- NEW: 0
        "--damps_avrf",           str(v_avrf),                # 0
        "--damps_imcf",           str(damps_imcf),            # 1
        "--damps_permutation_fft", str(damps_permutation_fft),
        "--damps_soft_routing",   str(damps_soft_routing),
        "--damps_momentum",       str(damps_momentum),
        "--damps_data_driven_prior", str(damps_data_driven_prior),
        "--damps_num_categories", str(damps_num_categories),
        "--damps_warmup_epochs",  str(damps_warmup_epochs),
        # Pattern B' rebuild
        "--rebuild_R",      str(rebuild_R),
        "--faiss_threshold", str(faiss_threshold),
        "--faiss_use_gpu",   str(faiss_use_gpu),
        "--knn_chunk_size",  str(knn_chunk_size),
        "--knn_efsearch",    str(knn_efsearch),
        # Mixed precision + compile
        "--use_amp",                str(use_amp),
        "--use_torch_compile",      str(use_torch_compile),
        "--torch_compile_mode",     str(torch_compile_mode),
        "--torch_compile_dynamic",  str(torch_compile_dynamic),
        # W&B
        "--use_wandb",      "1",
        "--wandb_project",  wandb_project,
        "--wandb_entity",   wandb_entity,
        "--wandb_run_name", f"phase1_{variant_name}_seed_{seed}",
        "--ablation_target", f"phase1_{variant_name}",
    ]

    print(f"Command: {' '.join(cmd)}")
    print(f"Current directory: {os.getcwd()}\n")

    env: dict[str, str] = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"]  = "1"

    t0: float = time.time()
    result: subprocess.CompletedProcess[str] = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        encoding="utf-8",
        errors="replace",
    )
    wall: float = time.time() - t0

    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        print(f"\n[WARNING] variant '{variant_name}' seed={seed} exited with "
              f"code {result.returncode}")
        if result.stderr:
            print("\n[ERROR OUTPUT]:")
            print(result.stderr)
    else:
        print(f"\n[OK] variant '{variant_name}' seed={seed} completed in "
              f"{wall/60:.1f} min")

    # -----------------------------------------------------------------
    # Parse the per-run log (mirrors cell-16 logic; note apc=0 in name).
    # -----------------------------------------------------------------
    path_name: str = (
        f"damps_uu_ii={User_layers}_{Item_layers}"
        f"_{user_loss_ratio}_{item_loss_ratio}"
        f"_topk={topk}_t={v_temp}_taulearn={v_learn_tau}_R={rebuild_R}"
        f"_apc={v_apc}_avrf={v_avrf}_imcf={damps_imcf}"
        f"_regs={regs}_dim={embed_size}_seed={seed}_"
        f"phase1_{variant_name}"
    )
    log_file: str = f"../{dataset}/{path_name}/{path_name}.txt"

    if not os.path.exists(log_file):
        print(f"[ERROR] log file missing: {log_file}")
        continue

    with open(log_file, "r", encoding="utf-8") as f:
        log_content: str = f.read()

    def _grab(rx: re.Pattern[str], cast=float) -> Any:
        m = rx.search(log_content)
        return cast(m.group(1)) if m else None

    rec: dict[str, Any] = {
        "seed":              seed,
        "variant":           variant_name,
        "log_file":          log_file,
        "test_recall@20":    _grab(_re_test_r20),
        "test_ndcg@20":      _grab(_re_test_n20),
        "test_precision@20": _grab(_re_test_p20),
        "val_recall@20":     _grab(_re_val_r20),
        "val_ndcg@20":       _grab(_re_val_n20),
        "val_recall_peak_epoch": _grab(_re_val_rpeak, int),
        "val_ndcg_peak_epoch":   _grab(_re_val_npeak, int),
        "wall_minutes":      round(wall / 60.0, 2),
    }
    _apc_off_results.append(rec)
    print(f"  parsed: R@20={rec['test_recall@20']:.6f}  "
          f"NDCG@20={rec['test_ndcg@20']:.6f}  "
          f"(val peak epochs: R={rec['val_recall_peak_epoch']}, "
          f"N={rec['val_ndcg_peak_epoch']})")

# ---------------------------------------------------------------------------
# Merge into the global variant store for the §4 summary + §4.5 t-tests.
# ---------------------------------------------------------------------------
if "all_variant_results" not in globals():
    all_variant_results = {}
all_variant_results[variant_name] = _apc_off_results

# Also extend the variant catalogue so the §4 pretty-printer has the label.
phase1_variants[variant_name] = {
    "temperature":     v_temp,
    "learnable_tau":   v_learn_tau,
    "damps_avrf":      v_avrf,
    "damps_apc":       v_apc,
    "label":           variant_label,
}

# ---------------------------------------------------------------------------
# In-cell quick verdict (full stop-gate decision lives in §4.5)
# ---------------------------------------------------------------------------
import numpy as np  # imported lazily; cell 16 already imports it globally
_r = np.array([r["test_recall@20"] for r in _apc_off_results
               if r["test_recall@20"] is not None], dtype=float)
_n = np.array([r["test_ndcg@20"]  for r in _apc_off_results
               if r["test_ndcg@20"]  is not None], dtype=float)

print("\n" + "=" * 80)
print(f"VARIANT '{variant_name}' -- 5-seed summary (preliminary)")
print("=" * 80)
print(f"  test_Recall@20 : mean={_r.mean():.6f}  std={_r.std(ddof=1):.6f}  "
      f"min={_r.min():.6f}  max={_r.max():.6f}  values={_r.round(4).tolist()}")
print(f"  test_NDCG@20   : mean={_n.mean():.6f}  std={_n.std(ddof=1):.6f}  "
      f"min={_n.min():.6f}  max={_n.max():.6f}  values={_n.round(4).tolist()}")

_paper_r20: float = 0.0881
_gate_r20:  float = 0.0870
print(f"\n  vs MMHCL paper (R@20={_paper_r20}) : "
      f"delta = {_r.mean() - _paper_r20:+.4f}  "
      f"({(_r.mean()/_paper_r20 - 1) * 100:+.1f}%)")
print(f"  rev44 stop-gate (R@20 >= {_gate_r20}) : "
      f"{'PASS' if _r.mean() >= _gate_r20 else 'FAIL'}")

print("\nNext step:")
print("  Run the **§ 4 Results Summary** code cell (next section), then")
print("  the **§ 4.5 Bonferroni** code cell — `apc_off_combined` is now in")
print("  `all_variant_results` / `phase1_variants` (5 variants × 5 seeds).")
