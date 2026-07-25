"""scripts/run_p6_4_tamer.py -- P6.4 TAMER Interest Tree grid driver.
======================================================================

Ships Priority 6.4 (P6.4) of the PACER-NRDMC upgrade roadmap.

Context
-------
P6.3 delivered convergence saturation (Recall@20 = 0.09256 for image_pca,
0.09219 for image_zca) but the Head:Tail ratio remained ~ 12.6:1 -- a
strong signal that structural item-item information is still absent from
the modality-only pipeline.  P6.4 imports the TAMER (Meng et al. MM'25)
Interest Tree augmentation as a drop-in enhancement on top of the P6.3
winner (image_pca, symmetric MACP, 100 epoch, patience 30).

P6.4 grid
---------
Two ``alpha_interest`` values x two seeds x 100 epochs, patience 30.

  * p6_4_tamer_a025 : alpha_interest=0.25 (conservative injection)
  * p6_4_tamer_a050 : alpha_interest=0.50 (aggressive injection)

Seeds: 23946202, 1557638902 (same as P6.1/P6.2/P6.3 for direct comparability).

Success criterion
-----------------
Per-cell mean R@20 strictly higher than the P6.3 image_pca mean (0.09256),
with special interest in the Tail R@20 tier which P6.3 left at 0.01129.

TAMER reference implementation
------------------------------
The BFS pruning, gamma/tau exponent and Eq. 9 alpha_hg fusion match
Z-last-ONE/TAMER models/tamer.py -- see codes/interest_tree.py for the
ports and codes/damps_tamer.py for the graph builder.

Usage (from MMHCL_DAMPS_Project/)::

    # Rebuild cache (once per dataset):
    python scripts/preprocess_interest_tree.py --dataset Clothing \\
        --data_dir ../data --core 5 \\
        --output ./results/interest_tree_clothing.npz

    # P6.4 grid:
    python scripts/run_p6_4_tamer.py
    python scripts/run_p6_4_tamer.py --dry_run 1
    python scripts/run_p6_4_tamer.py --only a025 --seeds 23946202

Wiring status
-------------
This commit ships the augmentation toolkit (codes/interest_tree.py,
codes/damps_tamer.py) and the preprocess script but does NOT yet wire the
``--tamer_interest_cache`` / ``--alpha_interest`` flags into
utility/parser.py + model.py.  ``--dry_run 1`` validates the cache and
prints the intended command line; a follow-up commit will land the
parser + model changes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
#  Locked PACER + Branch A' trunk (mirrors P6.3 base flags).
# --------------------------------------------------------------------------- #
DATASET = "Clothing"
EPOCH_DEFAULT = 100
PATIENCE_DEFAULT = 30
BATCH_SIZE = 1024
LR = 2.50995e-4
REGS = 1.20e-04
EMBED_SIZE = 128
KNN_TOPK = 10
UI_LAYERS = 3
U_LAYERS = 2
I_LAYERS = 2
SIMGCL_EPS = 0.329
LOGQ_SCALE = 0.651
LOGQ_BETA = 1.0
LOGQ_CLIP = 5.0
USE_REDUCE_LR = 1
NRDMC_LITE_LAYERS = 2
LAMBDA_VIEW = 0.10
TEMPERATURE = 0.30
R_U = 0.03
R_I = 0.07

# P6.3 winners: text=replace_pca, image=replace_pca.
TEXT_MODE = "replace_pca"
IMAGE_MODE = "replace_pca"

# TAMER hyper-params (Meng et al. MM'25 defaults on Amazon Clothing).
KNN_K_COOC = 20     # top-k for the co-occurrence graph S^c
KNN_K_MOD = 10      # top-k for the modality graph S^m (matches --topk above)
N_ORDER = 3         # BFS depth
GAMMA = 1.0         # Eq. 7 gamma
TAU = 1.0           # Eq. 7 tau

SEEDS_DEFAULT: tuple[int, ...] = (23946202, 1557638902)

OUT_JSON = "results/p6_4_tamer_clothing.json"
CACHE_PATH = "results/interest_tree_clothing.npz"

P63_MEAN_R20_IMAGE_PCA = 0.09256


@dataclass(frozen=True)
class GridConfig:
    tag: str
    alpha_interest: float


def build_configs(only: str | None = None) -> list[GridConfig]:
    all_cfgs = [
        GridConfig(tag="p6_4_tamer_a025", alpha_interest=0.25),
        GridConfig(tag="p6_4_tamer_a050", alpha_interest=0.50),
    ]
    if only is None:
        return all_cfgs
    key = only.strip().lower()
    aliases = {
        "a025": "p6_4_tamer_a025", "0.25": "p6_4_tamer_a025",
        "a050": "p6_4_tamer_a050", "0.50": "p6_4_tamer_a050",
        "0.5": "p6_4_tamer_a050",
    }
    tag = aliases.get(key, key)
    picked = [c for c in all_cfgs if c.tag == tag]
    if not picked:
        raise ValueError(f"--only got {only!r}; expected one of a025/a050.")
    return picked


def _agg(vals: list[float]) -> dict[str, float]:
    finite = [v for v in vals if not math.isnan(v)]
    if not finite:
        return {"mean": float("nan"), "std": float("nan"), "n": 0.0}
    return {
        "mean": float(statistics.mean(finite)),
        "std": float(statistics.stdev(finite)) if len(finite) > 1 else 0.0,
        "n": float(len(finite)),
    }


def _resolve_paths() -> tuple[Path, str]:
    cwd = Path.cwd().resolve()
    if cwd.name == "MMHCL_DAMPS_Project":
        damps = cwd
    elif (cwd / "MMHCL_DAMPS_Project").is_dir():
        damps = cwd / "MMHCL_DAMPS_Project"
    else:
        here = Path(__file__).resolve().parent
        damps = here.parent if here.name == "scripts" else cwd
    rtx = Path(r"c:\ProgramData\anaconda3\envs\rtx5090_dl\python.exe")
    py = str(rtx) if rtx.is_file() else sys.executable
    return damps, py


def _ensure_cache(damps_dir: Path, python_exe: str, dry_run: bool) -> Path:
    cache = damps_dir / CACHE_PATH
    if cache.is_file():
        print(f"[P6.4] using existing cache: {cache}")
        return cache
    print(f"[P6.4] cache missing -- running preprocess_interest_tree.py ...")
    cmd = [
        python_exe,
        str(damps_dir / "scripts" / "preprocess_interest_tree.py"),
        "--dataset", DATASET,
        "--data_dir", str(damps_dir.parent / "data"),
        "--core", "5",
        "--output", str(cache),
        "--knn_k_cooc", str(KNN_K_COOC),
        "--knn_k_mod", str(KNN_K_MOD),
        "--n_order", str(N_ORDER),
        "--gamma", str(GAMMA),
        "--tau", str(TAU),
        "--cooc_method", "auto",
        "--precompute_tree",
        "--tree_workers", "4",
    ]
    print("[cmd] " + " ".join(cmd))
    if dry_run:
        print("[dry_run] skipping actual preprocess execution.")
        return cache
    subprocess.check_call(cmd)
    if not cache.is_file():
        raise SystemExit(f"preprocess script did not produce {cache}")
    return cache


def _base_flags(*, cfg: GridConfig, cache_path: Path, epoch: int, patience: int) -> list[str]:
    """Mirrors P6.3 flags + new TAMER-augmentation flags.

    NOTE: ``--tamer_interest_cache`` / ``--alpha_interest`` / ``--enable_tamer``
    are placeholders here -- utility/parser.py + model.py wiring lands in a
    follow-up commit.  Until then, ``--dry_run 1`` should be used.
    """
    return [
        "--dataset", DATASET, "--gpu_id", "0",
        "--epoch", str(epoch), "--verbose", "5", "--eval_every", "5",
        "--eval_last_epochs", "30", "--use_gpu_eval", "1",
        "--use_torch_compile", "1", "--torch_compile_mode", "default",
        "--torch_compile_dynamic", "0", "--use_cuda_graph", "0",
        "--batch_size", str(BATCH_SIZE), "--lr", str(LR), "--regs", str(REGS),
        "--embed_size", str(EMBED_SIZE), "--topk", str(KNN_TOPK), "--core", "5",
        "--UI_layers", str(UI_LAYERS), "--User_layers", str(U_LAYERS),
        "--Item_layers", str(I_LAYERS),
        "--user_loss_ratio", str(R_U), "--item_loss_ratio", str(R_I),
        "--temperature", str(TEMPERATURE),
        "--enable_align", "0", "--lambda_align", "0.0", "--align_temperature", "0.2",
        "--damps_apc", "0", "--learnable_tau", "0",
        "--damps_avrf", "0", "--damps_imcf", "1",
        "--damps_soft_routing", "1", "--damps_momentum", "1",
        "--damps_data_driven_prior", "1", "--damps_permutation_fft", "0",
        "--damps_warmup_epochs", "10", "--damps_num_categories", "10",
        "--enable_logq", "1", "--logq_mode", "laplace",
        "--logq_beta", str(LOGQ_BETA), "--logq_scale", str(LOGQ_SCALE),
        "--logq_clip", str(LOGQ_CLIP),
        "--enable_simgcl", "0",
        "--enable_nrdmc_lite", "1",
        "--nrdmc_lite_layers", str(NRDMC_LITE_LAYERS),
        "--lambda_view", str(LAMBDA_VIEW),
        "--enable_ptv", "0", "--n_prototypes", "0", "--lambda_ptv", "0.0",
        "--simgcl_eps", str(SIMGCL_EPS),
        "--branchA_view_bsz", "2048", "--branchA_bcl_bsz", "2048",
        "--branchA_bcl_batchn", "1",
        "--early_stopping_patience", str(patience),
        "--early_stopping_min_epochs", "0",
        "--early_stopping_min_delta", "1e-4",
        "--early_stopping_monitor", "val_recall@20",
        "--early_stopping_mode", "max",
        "--early_stopping_restore_best", "1",
        "--use_reduce_lr", str(USE_REDUCE_LR), "--use_amp", "1",
        "--asc_gate_mode", "raw", "--asc_warmup_epochs", "0",
        "--asc_reg_l2", "0.0", "--asc_reg_target", "0.3",
        "--use_macp", "1",
        "--macp_mode", TEXT_MODE, "--macp_alpha_p", "0.0", "--macp_alpha_z", "0.0",
        "--macp_image_mode", IMAGE_MODE,
        "--macp_image_alpha_p", "0.0", "--macp_image_alpha_z", "0.0",
        "--macp_verbose", "1",
        # ---- P6.4 TAMER Interest Tree augmentation --------------------
        "--enable_tamer", "1",
        "--tamer_interest_cache", str(cache_path),
        "--alpha_interest", str(cfg.alpha_interest),
        # ---- WandB -----------------------------------------------------
        "--use_wandb", "1",
        "--wandb_project", "damps-mmhcl-clothing",
        "--wandb_entity", "baitapck51cc-uet",
        "--wandb_group", "p6_4_tamer_interest_tree",
        "--wandb_tags",
        "p6,p6_4,macp,tamer,interest_tree,long_epoch,nrdmc_lite",
    ]


def _run_one(*, python_exe: str, damps_dir: Path, cfg: GridConfig,
             cache_path: Path, epoch: int, patience: int, seed: int,
             dry_run: bool) -> dict[str, Any]:
    run_name = f"p6_4_{cfg.tag}_seed{seed}"
    cmd = [
        python_exe, "main_tercile.py",
        *_base_flags(cfg=cfg, cache_path=cache_path, epoch=epoch, patience=patience),
        "--seed", str(seed),
        "--wandb_run_name", run_name,
    ]
    print(
        f"\n{'=' * 74}\n"
        f"[P6.4] cfg={cfg.tag}  alpha_interest={cfg.alpha_interest}  "
        f"epoch={epoch}  patience={patience}  seed={seed}\n"
        f"{'=' * 74}",
        flush=True,
    )
    print("[cmd] " + " ".join(cmd[:8]) + f" ... [{len(cmd)-8} more flags]", flush=True)
    if dry_run:
        return {
            "tag": cfg.tag, "seed": seed, "exit": 0, "wall_min": 0.0,
            "dry_run": True, "alpha_interest": cfg.alpha_interest,
            "epoch_cap": epoch, "patience": patience,
            "best_test_recall20": float("nan"), "best_test_ndcg20": float("nan"),
            "best_val_recall20": float("nan"), "best_epoch": float("nan"),
        }

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    t0 = time.time()
    proc = subprocess.Popen(
        cmd, cwd=str(damps_dir), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    assert proc.stdout is not None
    chunks: list[str] = []
    for line in proc.stdout:
        chunks.append(line)
        print(line, end="", flush=True)
    rc = proc.wait()
    wall_min = (time.time() - t0) / 60.0
    if rc != 0:
        print(f"[WARN] cfg={cfg.tag} seed={seed} exited {rc}", flush=True)

    # Regex extraction copied from run_p6_3_p61_convergence.py.
    import re
    out = "".join(chunks)
    b = re.search(r"BEST_Test_Recall@20\s*[:=]\s*([-\d.eE+nan]+)", out)
    n = re.search(r"BEST_Test_NDCG@20\s*[:=]\s*([-\d.eE+nan]+)", out)
    v = re.search(r"BEST_Val_Recall@20\s*[:=]\s*([-\d.eE+nan]+)", out)
    e = re.search(r"(?:BEST_Val_Recall_Peak_Epoch|best_epoch)\s*[:=]\s*(\d+)", out)
    return {
        "tag": cfg.tag, "seed": seed, "exit": rc,
        "wall_min": wall_min, "dry_run": False,
        "alpha_interest": cfg.alpha_interest,
        "epoch_cap": epoch, "patience": patience,
        "best_test_recall20": float(b.group(1)) if b else float("nan"),
        "best_test_ndcg20": float(n.group(1)) if n else float("nan"),
        "best_val_recall20": float(v.group(1)) if v else float("nan"),
        "best_epoch": float(e.group(1)) if e else float("nan"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", type=str, default=None)
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--epoch", type=int, default=EPOCH_DEFAULT)
    p.add_argument("--patience", type=int, default=PATIENCE_DEFAULT)
    p.add_argument("--dry_run", type=int, default=0)
    p.add_argument("--out", type=str, default=OUT_JSON)
    args = p.parse_args()

    damps_dir, python_exe = _resolve_paths()
    print(f"[P6.4] damps_dir={damps_dir}  python={python_exe}")
    seeds = tuple(args.seeds) if args.seeds else SEEDS_DEFAULT
    print(f"[P6.4] seeds={seeds}")

    cache_path = _ensure_cache(damps_dir, python_exe, bool(args.dry_run))

    cfgs = build_configs(args.only)
    print(f"[P6.4] configs={[c.tag for c in cfgs]}")

    rows: list[dict[str, Any]] = []
    for cfg in cfgs:
        for seed in seeds:
            row = _run_one(
                python_exe=python_exe, damps_dir=damps_dir, cfg=cfg,
                cache_path=cache_path, epoch=args.epoch, patience=args.patience,
                seed=seed, dry_run=bool(args.dry_run),
            )
            rows.append(row)

    # Aggregate per-cell -> ranked list with delta vs P6.3.
    agg: dict[str, dict[str, Any]] = {}
    for r in rows:
        tag = r["tag"]
        agg.setdefault(tag, {"seeds": [], "recall20": [], "ndcg20": [], "val_r20": [],
                             "best_epoch": [], "alpha_interest": r["alpha_interest"]})
        agg[tag]["seeds"].append(r["seed"])
        agg[tag]["recall20"].append(r["best_test_recall20"])
        agg[tag]["ndcg20"].append(r["best_test_ndcg20"])
        agg[tag]["val_r20"].append(r["best_val_recall20"])
        agg[tag]["best_epoch"].append(r["best_epoch"])

    ranked = []
    for tag, s in agg.items():
        r20 = _agg(s["recall20"])
        n20 = _agg(s["ndcg20"])
        vr = _agg(s["val_r20"])
        be = _agg(s["best_epoch"])
        ranked.append({
            "tag": tag,
            "alpha_interest": s["alpha_interest"],
            "recall20_mean": r20["mean"], "recall20_std": r20["std"],
            "ndcg20_mean": n20["mean"], "ndcg20_std": n20["std"],
            "val_recall20_mean": vr["mean"], "best_epoch_mean": be["mean"],
            "delta_vs_p6_3": r20["mean"] - P63_MEAN_R20_IMAGE_PCA,
            "n_seeds": r20["n"],
        })
    ranked.sort(key=lambda x: (-(x["recall20_mean"] if not math.isnan(x["recall20_mean"]) else -1), x["tag"]))

    out_path = damps_dir / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump({"rows": rows, "ranked": ranked,
                   "p6_3_reference_mean_r20": P63_MEAN_R20_IMAGE_PCA}, fh, indent=2)
    print(f"\n[P6.4] wrote {out_path}")

    print("\n=== P6.4 ranked ===")
    print(f"{'tag':22s}  alpha  R@20 mean   std     dR20      N@20     best_ep")
    for r in ranked:
        print(
            f"{r['tag']:22s}  {r['alpha_interest']:5.2f}  "
            f"{r['recall20_mean']:.5f}  {r['recall20_std']:.5f}  "
            f"{r['delta_vs_p6_3']:+.4f}  {r['ndcg20_mean']:.5f}  "
            f"{r['best_epoch_mean']:.1f}"
        )


if __name__ == "__main__":
    main()
