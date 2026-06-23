# =====================================================================
# Section 8.2 -- Branch A post-hoc analysis  (read-only, rev55 §8.1)
#
#   Inputs:
#     results/branchA_sweep_results.json   (from Section 8.1 sweep cell)
#     ../<dataset>/damps_*_branchA_*/*.txt   (per-run training logs)
# =====================================================================
from __future__ import annotations

import glob
import json
import math
import os
import re
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np

try:
    _DAMPS_DIR = DAMPS_DIR  # type: ignore[name-defined]  # noqa: F821
except NameError:
    _DAMPS_DIR = os.getcwd()
try:
    _DATASET = dataset  # type: ignore[name-defined]  # noqa: F821
except NameError:
    _DATASET = "Clothing"

if os.path.normpath(os.getcwd()) != os.path.normpath(_DAMPS_DIR):
    os.chdir(_DAMPS_DIR)

GATE_MIN: float = 0.0900
GATE_MAX: float = 0.0945
GATE_FAIL: float = 0.0890

RESULTS_JSON = os.path.join(_DAMPS_DIR, "results", "branchA_sweep_results.json")
if not os.path.isfile(RESULTS_JSON):
    raise FileNotFoundError(
        f"{RESULTS_JSON} not found — run the Section 8.1 sweep cell first."
    )

with open(RESULTS_JSON, "r", encoding="utf-8") as f:
    payload = json.load(f)

runs: list[dict[str, Any]] = payload.get("runs", [])
summary: dict[str, Any] = payload.get("summary", {})
gate_pass: bool = bool(payload.get("gate"))

print("=" * 72)
print(f"[BranchA/analysis] gate: {'PASS' if gate_pass else 'FAIL'}")
print(
    f"  mean R@20 = {summary.get('mean_recall')} "
    f"+/- {summary.get('std_recall')}  "
    f"min={summary.get('min_recall')}  max={summary.get('max_recall')}"
)
print(
    f"  acceptance window [{GATE_MIN}, {GATE_MAX}]  "
    f"rollback if any seed < {GATE_FAIL}"
)
print("=" * 72)
print(f"{'seed':>8} {'R@20':>10} {'N@20':>10} {'hours':>8} {'window':>8}")
print("-" * 52)
for r in runs:
    r20 = r.get("recall@20")
    in_win = (
        r20 is not None and GATE_MIN <= float(r20) <= GATE_MAX
    )
    print(
        f"{r.get('seed', 0):>8} "
        f"{(r20 or 0):>10.6f} "
        f"{(r.get('ndcg@20') or 0):>10.6f} "
        f"{(r.get('runtime_h') or 0):>8.2f} "
        f"{('PASS' if in_win else 'FAIL'):>8}"
    )

# ---- Parse per-epoch curves from log directories -----------------------
LOG_ROOT = os.path.abspath(os.path.join("..", _DATASET))
DIR_RE = re.compile(
    r"damps_.*?_seed=(?P<seed>\d+)_branchA_lam[0-9.]+_seed(?P=seed)$"
)
PAT_EPOCH = re.compile(
    r"Epoch\s+(\d+)\s*\[[^\]]+\]:\s*loss=([\d.]+)\s+"
    r"recall@\d+=([\d.]+)\s+recall@(\d+)=([\d.]+)\s+ndcg@\d+=([\d.]+)"
)
PAT_VIEW = re.compile(r"loss_simgcl_view=([\d.eE+\-]+)")

curves: list[dict[str, Any]] = []
for d in sorted(glob.glob(os.path.join(LOG_ROOT, "damps_*branchA*"))):
    if not os.path.isdir(d):
        continue
    m = DIR_RE.match(os.path.basename(d.rstrip("/")))
    if not m:
        continue
    txts = glob.glob(os.path.join(d, "*.txt"))
    if not txts:
        continue
    with open(txts[0], "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    epochs: list[int] = []
    val_r20: list[float] = []
    view_loss: list[Optional[float]] = []
    for line in text.splitlines():
        em = PAT_EPOCH.search(line)
        if em:
            epochs.append(int(em.group(1)))
            val_r20.append(float(em.group(5)))
            vm = PAT_VIEW.search(line)
            view_loss.append(float(vm.group(1)) if vm else None)
    curves.append({
        "seed": int(m.group("seed")),
        "epochs": epochs,
        "val_rec20": val_r20,
        "view_loss": view_loss,
    })

print(f"\n[BranchA/analysis] parsed {len(curves)} log directories under {LOG_ROOT}")

if curves:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.5))

    ax = axes[0]
    for c in sorted(curves, key=lambda x: x["seed"]):
        if not c["epochs"]:
            continue
        ax.plot(
            c["epochs"],
            c["val_rec20"],
            linewidth=1.4,
            alpha=0.85,
            label=f"seed={c['seed']}",
        )
    ax.axhline(GATE_MIN, color="green", linestyle="--", linewidth=0.9, label="gate min")
    ax.axhline(GATE_MAX, color="green", linestyle=":", linewidth=0.9, label="gate max")
    ax.axhline(GATE_FAIL, color="red", linestyle="--", linewidth=0.9, label="rollback")
    ax.set_title("Branch A — val/recall@20 vs epoch")
    ax.set_xlabel("epoch")
    ax.set_ylabel("val/recall@20")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax2 = axes[1]
    for c in sorted(curves, key=lambda x: x["seed"]):
        eps = c["epochs"]
        vl = [v for v in c["view_loss"] if v is not None]
        if len(vl) < 2:
            continue
        ax2.plot(eps[: len(vl)], vl, linewidth=1.2, alpha=0.85, label=f"seed={c['seed']}")
    ax2.set_title("Branch A — loss_simgcl_view vs epoch")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("loss_simgcl_view")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    fig_path = os.path.join(LOG_ROOT, "branchA_val_recall_curves.png")
    plt.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.show()
    print(f"[plot] saved -> {fig_path}")

# ---- Runtime budget check ----------------------------------------------
hours = [r.get("runtime_h") for r in runs if r.get("runtime_h") is not None]
if hours:
    mean_h = float(np.mean(hours))
    print(
        f"\n[BranchA/analysis] wall-clock: mean={mean_h:.2f} h/seed  "
        f"min={min(hours):.2f}  max={max(hours):.2f}  "
        f"(target 4–8 h/seed)"
    )
    if mean_h > 8.0:
        print("[WARN] mean runtime exceeds 8 h/seed — revisit batch-N / view_every_k.")

if not gate_pass:
    print(
        "\n[BranchA/analysis] ROLLBACK: set "
        "--enable_simgcl 0 --branchA_bcl_batchn 0 --branchA_view_every_k 1 "
        "to revert to Wave 1 LogQ-only."
    )
