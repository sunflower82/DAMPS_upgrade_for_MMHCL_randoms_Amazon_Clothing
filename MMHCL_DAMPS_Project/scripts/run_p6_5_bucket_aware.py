"""scripts/run_p6_5_bucket_aware.py -- P6.5' bucket-aware training driver.
============================================================================

Ships Priority 6.5' (P6.5-prime) of the PACER-NRDMC upgrade roadmap.  This
is the *cheap* diagnostic track that layers on top of the P6.4 TAMER winner
(alpha_interest=0.50) two orthogonal fixes for the Head-overfit / Tail-under-
trained pattern surfaced by the P6.4 tercile analysis:

  1. **Popularity-inverse edge reweighting** (``--pop_inverse_eta``): scales
     the interest-branch edges of the TAMER-augmented modality graph by
     ``(pop_i * pop_j) ** (-eta)``, so head-head cooc edges are down-weighted
     and tail-tail cooc edges are amplified.  Applied to BOTH the raw S^c
     branch and the coef_csr tree bonus; the intrinsic modality similarity
     ``base`` (top-k cosine) is left untouched.
  2. **Bucket-geo-mean early-stopping monitor** (``--early_stopping_monitor
     bucket_geo@20``): patience listens to the geometric mean of
     ``val/recall@20_Head/Mid/Tail`` instead of overall val/recall@20, so
     the training loop refuses to declare victory when the Head tier is
     already saturated but Tail is still climbing.

P6.5' grid
----------
Two ``pop_inverse_eta`` values x two seeds x 100 epochs, patience 30.
Both cells inherit the P6.4 winner (alpha_interest=0.50) verbatim.

  * p6_5_bucket_eta05 : pop_inverse_eta=0.5 (conservative, sqrt(pop) inverse)
  * p6_5_bucket_eta10 : pop_inverse_eta=1.0 (aggressive, linear pop inverse)

Seeds: 23946202, 1557638902 (first two locked seeds -- match P6.1/P6.2/P6.3/P6.4).

Success criterion
-----------------
Per-cell:
  (a) Overall val Recall@20 >= P6.4 winner (0.09566),
  (b) Tail Recall@20 strictly improves over P6.4 winner (0.01563),
  (c) Bucket-geo mean strictly improves (P6.4 winner ~0.03795).

Failing (a) + (b) means popularity-inverse gates the interest branch too
aggressively; drop to eta=0.25 or switch to P6.5 RSFPGrowth.

Cache reuse
-----------
Consumes the SAME ``results/interest_tree_clothing.npz`` cache written by
``scripts/preprocess_interest_tree.py`` -- popularity vectors are recomputed
inside train.py from ``data_generator.train_items`` (no cache surgery).

Usage (from MMHCL_DAMPS_Project/)::

    # Full grid (cache must exist -- reuses P6.4 cache):
    python scripts/run_p6_5_bucket_aware.py

    # Dry run:
    python scripts/run_p6_5_bucket_aware.py --dry_run 1

    # Single cell, single seed:
    python scripts/run_p6_5_bucket_aware.py --only eta05 --seeds 23946202
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
#  Locked PACER + Branch A' trunk (mirrors P6.4 base flags exactly).
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

TEXT_MODE = "replace_pca"
IMAGE_MODE = "replace_pca"

# TAMER hyper-params (unchanged from P6.4).
KNN_K_COOC = 20
KNN_K_MOD = 10
N_ORDER = 3
GAMMA = 1.0
TAU = 1.0
ALPHA_INTEREST_LOCK = 0.50   # P6.4 winner cell

SEEDS_DEFAULT: tuple[int, ...] = (23946202, 1557638902)

OUT_JSON = "results/p6_5_bucket_aware_clothing.json"
CACHE_PATH = "results/interest_tree_clothing.npz"

# References for delta computation.
P64_MEAN_R20_ALPHA050 = 0.09566
P64_MEAN_R20_TAIL_ALPHA050 = 0.01563


@dataclass(frozen=True)
class GridConfig:
    tag: str
    pop_inverse_eta: float


def build_configs(only: str | None = None) -> list[GridConfig]:
    all_cfgs = [
        GridConfig(tag="p6_5_bucket_eta05", pop_inverse_eta=0.5),
        GridConfig(tag="p6_5_bucket_eta10", pop_inverse_eta=1.0),
    ]
    if only is None:
        return all_cfgs
    key = only.strip().lower()
    aliases = {
        "eta05": "p6_5_bucket_eta05", "0.5": "p6_5_bucket_eta05",
        "eta10": "p6_5_bucket_eta10", "1.0": "p6_5_bucket_eta10",
        "1": "p6_5_bucket_eta10",
    }
    tag = aliases.get(key, key)
    picked = [c for c in all_cfgs if c.tag == tag]
    if not picked:
        raise ValueError(
            f"--only got {only!r}; expected one of eta05 / eta10."
        )
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


def _ensure_cache(
    damps_dir: Path,
    python_exe: str,
    dry_run: bool,
    cache_path: Path | None = None,
) -> Path:
    cache = Path(cache_path) if cache_path is not None else (damps_dir / CACHE_PATH)
    if not cache.is_absolute():
        cache = (damps_dir / cache).resolve()
    if cache.is_file():
        print(f"[P6.5'] reusing existing cache: {cache}")
        return cache
    print(f"[P6.5'] cache missing -- running preprocess_interest_tree.py ...")
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
    cache.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(cmd)
    if not cache.is_file():
        raise SystemExit(f"preprocess script did not produce {cache}")
    return cache


def _base_flags(*, cfg: GridConfig, cache_path: Path, epoch: int, patience: int) -> list[str]:
    """P6.4 base flags + P6.5' pop-inverse eta + bucket-geo monitor."""
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
        # P6.5' KEY CHANGE: patience listens to bucket-geo-mean, not R@20.
        "--early_stopping_monitor", "bucket_geo@20",
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
        # ---- P6.4 TAMER Interest Tree (locked to winner) --------------
        "--enable_tamer", "1",
        "--tamer_interest_cache", str(cache_path),
        "--alpha_interest", str(ALPHA_INTEREST_LOCK),
        # ---- P6.5' popularity-inverse edge reweighting ----------------
        "--pop_inverse_eta", str(cfg.pop_inverse_eta),
        # ---- WandB ----------------------------------------------------
        "--use_wandb", "1",
        "--wandb_project", "damps-mmhcl-clothing",
        "--wandb_entity", "baitapck51cc-uet",
        "--wandb_group", "p6_5_bucket_aware",
        "--wandb_tags",
        "p6,p6_5_prime,tamer,bucket_aware,pop_inverse,bucket_geo_monitor,nrdmc_lite",
    ]


def _run_one(*, python_exe: str, damps_dir: Path, cfg: GridConfig,
             cache_path: Path, epoch: int, patience: int, seed: int,
             dry_run: bool) -> dict[str, Any]:
    run_name = f"p6_5_{cfg.tag}_seed{seed}"
    cmd = [
        python_exe, "main_tercile.py",
        *_base_flags(cfg=cfg, cache_path=cache_path, epoch=epoch, patience=patience),
        "--seed", str(seed),
        "--wandb_run_name", run_name,
    ]
    print(
        f"\n{'=' * 74}\n"
        f"[P6.5'] cfg={cfg.tag}  eta={cfg.pop_inverse_eta}  "
        f"alpha_interest={ALPHA_INTEREST_LOCK}  "
        f"epoch={epoch}  patience={patience}  seed={seed}\n"
        f"{'=' * 74}",
        flush=True,
    )
    print("[cmd] " + " ".join(cmd[:8]) + f" ... [{len(cmd)-8} more flags]", flush=True)
    if dry_run:
        return {
            "tag": cfg.tag, "seed": seed, "exit": 0, "wall_min": 0.0,
            "dry_run": True, "pop_inverse_eta": cfg.pop_inverse_eta,
            "alpha_interest": ALPHA_INTEREST_LOCK,
            "epoch_cap": epoch, "patience": patience,
            "best_test_recall20": float("nan"), "best_test_ndcg20": float("nan"),
            "best_val_recall20": float("nan"), "best_epoch": float("nan"),
            "best_val_tail_recall20": float("nan"),
            "best_val_mid_recall20": float("nan"),
            "best_val_head_recall20": float("nan"),
            "best_val_bucket_geo20": float("nan"),
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

    # Regex extraction: standard PACER summary + tercile-final line.
    import re
    out = "".join(chunks)
    b = re.search(r"BEST_Test_Recall@20\s*[:=]\s*([-\d.eE+nan]+)", out)
    n = re.search(r"BEST_Test_NDCG@20\s*[:=]\s*([-\d.eE+nan]+)", out)
    v = re.search(r"BEST_Val_Recall@20\s*[:=]\s*([-\d.eE+nan]+)", out)
    e = re.search(r"(?:BEST_Val_Recall_Peak_Epoch|best_epoch)\s*[:=]\s*(\d+)", out)
    # Tercile line (main_tercile.py final summary):
    #   [tercile-final]     BEST_Recall@20_Head=.. Mid=.. Tail=..
    tercile = re.search(
        r"\[tercile-final\][^\n]*"
        r"BEST_Recall@20_Head\s*[:=]\s*([-\d.eE+nan]+)[^\n]*"
        r"BEST_Recall@20_Mid\s*[:=]\s*([-\d.eE+nan]+)[^\n]*"
        r"BEST_Recall@20_Tail\s*[:=]\s*([-\d.eE+nan]+)",
        out,
    )
    if tercile is not None:
        head = float(tercile.group(1))
        mid = float(tercile.group(2))
        tail = float(tercile.group(3))
        if head > 0 and mid > 0 and tail > 0 \
           and math.isfinite(head) and math.isfinite(mid) and math.isfinite(tail):
            bg = (head * mid * tail) ** (1.0 / 3.0)
        else:
            bg = float("nan")
    else:
        head = mid = tail = bg = float("nan")

    return {
        "tag": cfg.tag, "seed": seed, "exit": rc,
        "wall_min": wall_min, "dry_run": False,
        "pop_inverse_eta": cfg.pop_inverse_eta,
        "alpha_interest": ALPHA_INTEREST_LOCK,
        "epoch_cap": epoch, "patience": patience,
        "best_test_recall20": float(b.group(1)) if b else float("nan"),
        "best_test_ndcg20": float(n.group(1)) if n else float("nan"),
        "best_val_recall20": float(v.group(1)) if v else float("nan"),
        "best_epoch": float(e.group(1)) if e else float("nan"),
        "best_val_head_recall20": head,
        "best_val_mid_recall20": mid,
        "best_val_tail_recall20": tail,
        "best_val_bucket_geo20": bg,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", type=str, default=None)
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--epoch", type=int, default=EPOCH_DEFAULT)
    p.add_argument("--patience", type=int, default=PATIENCE_DEFAULT)
    p.add_argument("--dry_run", type=int, default=0)
    # Canonical flag is --out; --output is accepted for notebook cell 9.25.
    p.add_argument("--out", type=str, default=OUT_JSON)
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Alias for --out (used by notebook cell 9.25).",
    )
    p.add_argument(
        "--cache",
        type=str,
        default=None,
        help="Path to interest_tree_*.npz (default: results/interest_tree_clothing.npz).",
    )
    args = p.parse_args()

    damps_dir, python_exe = _resolve_paths()
    print(f"[P6.5'] damps_dir={damps_dir}  python={python_exe}")
    seeds = tuple(args.seeds) if args.seeds else SEEDS_DEFAULT
    print(f"[P6.5'] seeds={seeds}")

    out_rel = args.output if args.output else args.out
    cache_arg = Path(args.cache) if args.cache else None
    cache_path = _ensure_cache(
        damps_dir, python_exe, bool(args.dry_run), cache_path=cache_arg
    )

    cfgs = build_configs(args.only)
    print(f"[P6.5'] configs={[c.tag for c in cfgs]}")

    rows: list[dict[str, Any]] = []
    for cfg in cfgs:
        for seed in seeds:
            row = _run_one(
                python_exe=python_exe, damps_dir=damps_dir, cfg=cfg,
                cache_path=cache_path, epoch=args.epoch, patience=args.patience,
                seed=seed, dry_run=bool(args.dry_run),
            )
            rows.append(row)

    # Aggregate per-cell -> ranked list with delta vs P6.4 alpha=0.50 winner.
    agg: dict[str, dict[str, Any]] = {}
    for r in rows:
        tag = r["tag"]
        agg.setdefault(tag, {
            "seeds": [], "recall20": [], "ndcg20": [], "val_r20": [],
            "best_epoch": [], "head": [], "mid": [], "tail": [], "bucket_geo": [],
            "pop_inverse_eta": r["pop_inverse_eta"],
        })
        agg[tag]["seeds"].append(r["seed"])
        agg[tag]["recall20"].append(r["best_test_recall20"])
        agg[tag]["ndcg20"].append(r["best_test_ndcg20"])
        agg[tag]["val_r20"].append(r["best_val_recall20"])
        agg[tag]["best_epoch"].append(r["best_epoch"])
        agg[tag]["head"].append(r["best_val_head_recall20"])
        agg[tag]["mid"].append(r["best_val_mid_recall20"])
        agg[tag]["tail"].append(r["best_val_tail_recall20"])
        agg[tag]["bucket_geo"].append(r["best_val_bucket_geo20"])

    ranked = []
    for tag, s in agg.items():
        r20 = _agg(s["recall20"])
        n20 = _agg(s["ndcg20"])
        vr = _agg(s["val_r20"])
        be = _agg(s["best_epoch"])
        head = _agg(s["head"])
        mid = _agg(s["mid"])
        tail = _agg(s["tail"])
        bg = _agg(s["bucket_geo"])
        ranked.append({
            "tag": tag,
            "pop_inverse_eta": s["pop_inverse_eta"],
            "alpha_interest": ALPHA_INTEREST_LOCK,
            "recall20_mean": r20["mean"], "recall20_std": r20["std"],
            "ndcg20_mean": n20["mean"], "ndcg20_std": n20["std"],
            "val_recall20_mean": vr["mean"], "best_epoch_mean": be["mean"],
            "head_mean": head["mean"], "mid_mean": mid["mean"],
            "tail_mean": tail["mean"], "bucket_geo_mean": bg["mean"],
            "delta_vs_p6_4_r20": r20["mean"] - P64_MEAN_R20_ALPHA050,
            "delta_vs_p6_4_tail": tail["mean"] - P64_MEAN_R20_TAIL_ALPHA050,
            "n_seeds": r20["n"],
        })
    ranked.sort(key=lambda x: (
        -(x["bucket_geo_mean"] if not math.isnan(x["bucket_geo_mean"]) else -1),
        x["tag"],
    ))

    out_path = Path(out_rel)
    if not out_path.is_absolute():
        out_path = damps_dir / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump({
            "rows": rows, "ranked": ranked,
            "p6_4_reference": {
                "mean_r20_alpha050": P64_MEAN_R20_ALPHA050,
                "mean_r20_tail_alpha050": P64_MEAN_R20_TAIL_ALPHA050,
            },
        }, fh, indent=2)
    print(f"\n[P6.5'] wrote {out_path}")

    print("\n=== P6.5' ranked (by bucket-geo mean, desc) ===")
    print(
        f"{'tag':22s}  eta   R@20 mean   std     dR20      "
        f"Tail       dTail     bucket_geo  best_ep"
    )
    for r in ranked:
        print(
            f"{r['tag']:22s}  {r['pop_inverse_eta']:4.2f}  "
            f"{r['recall20_mean']:.5f}  {r['recall20_std']:.5f}  "
            f"{r['delta_vs_p6_4_r20']:+.4f}  "
            f"{r['tail_mean']:.5f}  {r['delta_vs_p6_4_tail']:+.4f}  "
            f"{r['bucket_geo_mean']:.5f}    "
            f"{r['best_epoch_mean']:.1f}"
        )


if __name__ == "__main__":
    main()
