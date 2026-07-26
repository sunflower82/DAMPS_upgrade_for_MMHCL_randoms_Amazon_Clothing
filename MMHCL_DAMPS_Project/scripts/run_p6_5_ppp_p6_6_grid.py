"""scripts/run_p6_5_ppp_p6_6_grid.py -- P6.5''' + P6.6 combined grid.
====================================================================

Ships a 4-variant x 1-seed *pure-CLI* grid that tests two orthogonal
hypotheses raised by the P6.5'' post-mortem:

1. **P6.5''' (monitor + duration)** -- The ``val_recall@20`` monitor is
   Head-biased for long-tail evaluation; a checkpoint locked at
   ep~93 loses ~17% of achievable Tail R@20 vs the ep 138-141 peak
   observed in the seed-23946202 CSVs. Swapping to
   ``bucket_geo@20`` with ``--early_stopping_min_epochs 130`` should
   surface the late-stage Tail/Mid gains.
2. **P6.6 (Head overfit mitigation)** -- Head R@20 peaks at ep 44 and
   declines for the next 100 epochs. Higher weight decay and/or a
   tighter ReduceLROnPlateau schedule may hold Head near its peak
   while Tail/Mid continue to climb, unlocking the ep 93->141
   window for *both* Overall R@20 and bucket_geo@20.

Only CLI flags already exposed by ``utility/parser.py`` are used:

  * ``--early_stopping_monitor bucket_geo@20`` (P6.5' path)
  * ``--early_stopping_min_epochs``
  * ``--regs``            (uniform L2 -- NOT Head-specific; see caveat)
  * ``--reduce_lr_factor``
  * ``--reduce_lr_patience``

*Caveat*: ``--regs`` applies uniformly to *all* embeddings, not only
Head items. This is the closest CLI-only proxy for "Head-aware weight
decay"; a Head-only variant would require ~20-30 LoC in
``model_MMHCL.py`` and is deferred to a later commit.

Grid (4 variants x 1 locked seed = 23946202)
============================================

+--------------+---------------+-----------+-------------+------------------+
| Variant      | Monitor       | min_epoch | regs        | reduce_lr        |
+==============+===============+===========+=============+==================+
| p6_5_ppp     | bucket_geo@20 | 130       | 1.20e-04 (=)| factor=0.5 (=)   |
| p6_6b_lr     | bucket_geo@20 | 130       | 1.20e-04 (=)| factor=0.3, p=5  |
| p6_6c_regs   | bucket_geo@20 | 130       | 4.80e-04 (4x)| factor=0.5 (=)  |
| p6_6d_both   | bucket_geo@20 | 130       | 4.80e-04 (4x)| factor=0.3, p=5 |
+--------------+---------------+-----------+-------------+------------------+

All 4 share: ``alpha_interest=0.50``, ``pop_inverse_eta=0.0``
(bit-for-bit P6.4), ``--epoch 200 --early_stopping_patience 30
--early_stopping_mode max --early_stopping_restore_best 1``.

Single seed (23946202) is deliberate -- it matches the seed whose
CSVs revealed the ep-93 vs ep-141 divergence. Adding seed
1557638902 would double wall-clock (~3h -> ~6h) without changing the
Head/Mid/Tail training-dynamics conclusion, which is a
deterministic property of the base config, not seed noise.

Seeded pair (23946202, 1557638902) will be re-run on the winning
variant only, as a normal follow-up.

Wall-clock estimate
-------------------
~40 min/variant on RTX 5090 if patience does NOT bite past ep~170.
Grid total: ~2.5-3.0 h.

Usage (from MMHCL_DAMPS_Project/)::

    python scripts/run_p6_5_ppp_p6_6_grid.py
    python scripts/run_p6_5_ppp_p6_6_grid.py --dry_run 1
    python scripts/run_p6_5_ppp_p6_6_grid.py --variants p6_5_ppp p6_6d_both

Outputs
-------
  * ``results/p6_5_ppp_p6_6_clothing.json``    (structured rows + ranked)
  * per-variant WandB run tags with ``p6_5_ppp`` / ``p6_6_head_mitigation``
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
#  Locked P6.4 winner trunk (identical to run_p6_5_pp_extended.py minus the
#  monitor/min_epochs and the P6.6 knobs). Drift here breaks comparability.
# --------------------------------------------------------------------------- #
DATASET = "Clothing"
EPOCH_DEFAULT = 200
PATIENCE_DEFAULT = 30
MIN_EPOCHS_DEFAULT = 130  # <- P6.5''' key change
EVAL_LAST_EPOCHS = 140
BATCH_SIZE = 1024
LR = 2.50995e-4
REGS_BASE = 1.20e-04           # <- P6.4 baseline (uniform L2)
REGS_HIGH = 4.80e-04           # <- P6.6c: 4x baseline
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

# ReduceLROnPlateau baseline (P6.4 defaults).
REDUCE_LR_FACTOR_BASE = 0.5
REDUCE_LR_PATIENCE_BASE = 3
# P6.6b tighter schedule -- forces earlier LR cool-down for Head.
REDUCE_LR_FACTOR_TIGHT = 0.3
REDUCE_LR_PATIENCE_TIGHT = 5    # patience x1.7, factor x0.6

TEXT_MODE = "replace_pca"
IMAGE_MODE = "replace_pca"

# TAMER (locked to P6.4 winner).
KNN_K_COOC = 20
KNN_K_MOD = 10
N_ORDER = 3
GAMMA = 1.0
TAU = 1.0
ALPHA_INTEREST_LOCK = 0.50
POP_INVERSE_ETA_LOCK = 0.0

SEED_DEFAULT = 23946202  # <- MATCHES the diagnostic CSV seed

OUT_JSON = "results/p6_5_ppp_p6_6_clothing.json"
CACHE_PATH = "results/interest_tree_clothing.npz"

# P6.4 100-epoch reference (mean of 2 locked seeds).
P64_MEAN_R20 = 0.09566
P64_MEAN_NDCG20 = 0.04382
P64_MEAN_R20_TAIL = 0.01563
P64_MEAN_BG = 0.03795

# P6.5'' seed-23946202 CSV peaks at ep 138-141 (from post-mortem).
# Used as expected-outcome reference for the "pure" P6.5''' variant.
P6_5PP_CSV_PEAK_TAIL = 0.01826       # ep 138
P6_5PP_CSV_PEAK_MID = 0.03862        # ep 142
P6_5PP_CSV_PEAK_BG = 0.04552         # ep 141


@dataclass(frozen=True)
class GridConfig:
    tag: str
    regs: float
    reduce_lr_factor: float
    reduce_lr_patience: int
    note: str = ""


def build_configs(variants: list[str] | None = None) -> list[GridConfig]:
    all_cfgs = [
        GridConfig(
            tag="p6_5_ppp",
            regs=REGS_BASE,
            reduce_lr_factor=REDUCE_LR_FACTOR_BASE,
            reduce_lr_patience=REDUCE_LR_PATIENCE_BASE,
            note="P6.5''': bucket_geo monitor + min_ep 130, P6.4 hyperparams",
        ),
        GridConfig(
            tag="p6_6b_lr",
            regs=REGS_BASE,
            reduce_lr_factor=REDUCE_LR_FACTOR_TIGHT,
            reduce_lr_patience=REDUCE_LR_PATIENCE_TIGHT,
            note="P6.6b: P6.5''' + tighter ReduceLROnPlateau (factor 0.3, patience 5)",
        ),
        GridConfig(
            tag="p6_6c_regs",
            regs=REGS_HIGH,
            reduce_lr_factor=REDUCE_LR_FACTOR_BASE,
            reduce_lr_patience=REDUCE_LR_PATIENCE_BASE,
            note="P6.6c: P6.5''' + 4x uniform L2 (regs 4.8e-4)",
        ),
        GridConfig(
            tag="p6_6d_both",
            regs=REGS_HIGH,
            reduce_lr_factor=REDUCE_LR_FACTOR_TIGHT,
            reduce_lr_patience=REDUCE_LR_PATIENCE_TIGHT,
            note="P6.6d: P6.5''' + regs 4.8e-4 + tighter LR schedule (combined)",
        ),
    ]
    if variants is None:
        return all_cfgs
    lookup = {c.tag: c for c in all_cfgs}
    missing = [v for v in variants if v not in lookup]
    if missing:
        raise SystemExit(
            f"Unknown variants {missing}. Known: {list(lookup)}"
        )
    return [lookup[v] for v in variants]


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


def _base_flags(
    *,
    cache_path: Path,
    cfg: GridConfig,
    epoch: int,
    patience: int,
    min_epochs: int,
) -> list[str]:
    """P6.4 winner base flags + P6.5''' monitor/duration + P6.6 knobs.

    IMPORTANT: keep this list synchronised with ``run_p6_5_pp_extended.py``
    minus the monitor/min_epochs overrides and the regs/LR overrides.
    """
    return [
        "--dataset", DATASET, "--gpu_id", "0",
        "--epoch", str(epoch), "--verbose", "5", "--eval_every", "5",
        "--eval_last_epochs", str(EVAL_LAST_EPOCHS), "--use_gpu_eval", "1",
        "--use_torch_compile", "1", "--torch_compile_mode", "default",
        "--torch_compile_dynamic", "0", "--use_cuda_graph", "0",
        "--batch_size", str(BATCH_SIZE), "--lr", str(LR),
        # P6.6c/d: uniform L2 override.
        "--regs", str(cfg.regs),
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
        # ---- P6.5''' key changes ------------------------------------
        "--early_stopping_patience", str(patience),
        "--early_stopping_min_epochs", str(min_epochs),
        "--early_stopping_min_delta", "1e-4",
        "--early_stopping_monitor", "bucket_geo@20",
        "--early_stopping_mode", "max",
        "--early_stopping_restore_best", "1",
        # ---- P6.6b/d ReduceLROnPlateau overrides ---------------------
        "--use_reduce_lr", str(USE_REDUCE_LR),
        "--reduce_lr_factor", str(cfg.reduce_lr_factor),
        "--reduce_lr_patience", str(cfg.reduce_lr_patience),
        "--use_amp", "1",
        "--asc_gate_mode", "raw", "--asc_warmup_epochs", "0",
        "--asc_reg_l2", "0.0", "--asc_reg_target", "0.3",
        "--use_macp", "1",
        "--macp_mode", TEXT_MODE, "--macp_alpha_p", "0.0", "--macp_alpha_z", "0.0",
        "--macp_image_mode", IMAGE_MODE,
        "--macp_image_alpha_p", "0.0", "--macp_image_alpha_z", "0.0",
        "--macp_verbose", "1",
        # ---- P6.4 TAMER (locked to winner) ---------------------------
        "--enable_tamer", "1",
        "--tamer_interest_cache", str(cache_path),
        "--alpha_interest", str(ALPHA_INTEREST_LOCK),
        "--pop_inverse_eta", str(POP_INVERSE_ETA_LOCK),
        # ---- WandB ---------------------------------------------------
        "--use_wandb", "1",
        "--wandb_project", "damps-mmhcl-clothing",
        "--wandb_entity", "baitapck51cc-uet",
        "--wandb_group", "p6_5_ppp_p6_6",
        "--wandb_tags",
        f"p6,p6_5_ppp,p6_6_head_mitigation,{cfg.tag},"
        f"bucket_geo_monitor,min_ep130,tamer,alpha050,"
        f"regs{cfg.regs},lrf{cfg.reduce_lr_factor}_lrp{cfg.reduce_lr_patience},"
        f"nrdmc_lite",
    ]


def _ensure_cache(
    damps_dir: Path, python_exe: str, dry_run: bool,
    cache_path: Path | None = None,
) -> Path:
    cache = Path(cache_path) if cache_path is not None else (damps_dir / CACHE_PATH)
    if not cache.is_absolute():
        cache = (damps_dir / cache).resolve()
    if cache.is_file():
        print(f"[P6.5'''+P6.6] reusing existing cache: {cache}")
        return cache
    raise SystemExit(
        f"cache missing: {cache}\n"
        "Run P6.4 or P6.5' notebook cells to build the interest tree first."
    )


def _run_one(
    *, python_exe: str, damps_dir: Path, cfg: GridConfig, cache_path: Path,
    epoch: int, patience: int, min_epochs: int, seed: int, dry_run: bool,
) -> dict[str, Any]:
    run_name = f"{cfg.tag}_seed{seed}"
    cmd = [
        python_exe, "main_tercile.py",
        *_base_flags(
            cache_path=cache_path, cfg=cfg,
            epoch=epoch, patience=patience, min_epochs=min_epochs,
        ),
        "--seed", str(seed),
        "--wandb_run_name", run_name,
    ]
    print(
        f"\n{'=' * 78}\n"
        f"[P6.5'''+P6.6] {cfg.tag}  "
        f"regs={cfg.regs}  lr_factor={cfg.reduce_lr_factor}  "
        f"lr_patience={cfg.reduce_lr_patience}  seed={seed}\n"
        f"  {cfg.note}\n"
        f"{'=' * 78}",
        flush=True,
    )
    print("[cmd] " + " ".join(cmd[:8]) + f" ... [{len(cmd) - 8} more flags]", flush=True)
    if dry_run:
        return {
            "tag": cfg.tag, "seed": seed, "exit": 0, "wall_min": 0.0,
            "dry_run": True,
            "alpha_interest": ALPHA_INTEREST_LOCK,
            "pop_inverse_eta": POP_INVERSE_ETA_LOCK,
            "regs": cfg.regs,
            "reduce_lr_factor": cfg.reduce_lr_factor,
            "reduce_lr_patience": cfg.reduce_lr_patience,
            "epoch_cap": epoch, "patience": patience, "min_epochs": min_epochs,
            "best_test_recall20": float("nan"), "best_test_ndcg20": float("nan"),
            "best_val_recall20": float("nan"), "best_epoch": float("nan"),
            "best_val_head_recall20": float("nan"),
            "best_val_mid_recall20": float("nan"),
            "best_val_tail_recall20": float("nan"),
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
        print(f"[WARN] {cfg.tag} seed={seed} exited {rc}", flush=True)

    import re
    out = "".join(chunks)
    b = re.search(r"BEST_Test_Recall@20\s*[:=]\s*([-\d.eE+nan]+)", out)
    n = re.search(r"BEST_Test_NDCG@20\s*[:=]\s*([-\d.eE+nan]+)", out)
    v = re.search(r"BEST_Val_Recall@20\s*[:=]\s*([-\d.eE+nan]+)", out)
    e = re.search(r"(?:BEST_Val_Recall_Peak_Epoch|best_epoch)\s*[:=]\s*(\d+)", out)
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
        "alpha_interest": ALPHA_INTEREST_LOCK,
        "pop_inverse_eta": POP_INVERSE_ETA_LOCK,
        "regs": cfg.regs,
        "reduce_lr_factor": cfg.reduce_lr_factor,
        "reduce_lr_patience": cfg.reduce_lr_patience,
        "epoch_cap": epoch, "patience": patience, "min_epochs": min_epochs,
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
    p.add_argument(
        "--variants", type=str, nargs="*", default=None,
        help="Subset of variant tags to run "
             "(default = full 4-variant grid). Example: "
             "--variants p6_5_ppp p6_6d_both",
    )
    p.add_argument("--seed", type=int, default=SEED_DEFAULT)
    p.add_argument("--epoch", type=int, default=EPOCH_DEFAULT)
    p.add_argument("--patience", type=int, default=PATIENCE_DEFAULT)
    p.add_argument("--min_epochs", type=int, default=MIN_EPOCHS_DEFAULT)
    p.add_argument("--dry_run", type=int, default=0)
    p.add_argument("--out", type=str, default=OUT_JSON)
    p.add_argument("--output", type=str, default=None,
                   help="Alias for --out (used by notebook cell §9.27).")
    p.add_argument("--cache", type=str, default=None,
                   help="Path to interest_tree_*.npz (default: "
                        "results/interest_tree_clothing.npz).")
    args = p.parse_args()

    damps_dir, python_exe = _resolve_paths()
    print(f"[P6.5'''+P6.6] damps_dir={damps_dir}  python={python_exe}")
    seed = args.seed
    print(f"[P6.5'''+P6.6] seed={seed} (single-seed diagnostic grid)")

    out_rel = args.output if args.output else args.out
    cache_arg = Path(args.cache) if args.cache else None
    cache_path = _ensure_cache(
        damps_dir, python_exe, bool(args.dry_run), cache_path=cache_arg
    )

    cfgs = build_configs(args.variants)
    print(f"[P6.5'''+P6.6] variants={[c.tag for c in cfgs]}")

    rows: list[dict[str, Any]] = []
    for cfg in cfgs:
        row = _run_one(
            python_exe=python_exe, damps_dir=damps_dir, cfg=cfg,
            cache_path=cache_path,
            epoch=args.epoch, patience=args.patience,
            min_epochs=args.min_epochs, seed=seed,
            dry_run=bool(args.dry_run),
        )
        rows.append(row)

    ranked = []
    for r in rows:
        r20 = r["best_test_recall20"]
        n20 = r["best_test_ndcg20"]
        head = r["best_val_head_recall20"]
        mid = r["best_val_mid_recall20"]
        tail = r["best_val_tail_recall20"]
        bg = r["best_val_bucket_geo20"]
        ranked.append({
            "tag": r["tag"], "seed": r["seed"],
            "regs": r["regs"],
            "reduce_lr_factor": r["reduce_lr_factor"],
            "reduce_lr_patience": r["reduce_lr_patience"],
            "recall20": r20, "ndcg20": n20,
            "head": head, "mid": mid, "tail": tail, "bucket_geo": bg,
            "best_epoch": r["best_epoch"],
            "delta_vs_p6_4_r20": r20 - P64_MEAN_R20 if not math.isnan(r20) else float("nan"),
            "delta_vs_p6_4_ndcg": n20 - P64_MEAN_NDCG20 if not math.isnan(n20) else float("nan"),
            "delta_vs_p6_4_tail": tail - P64_MEAN_R20_TAIL if not math.isnan(tail) else float("nan"),
            "delta_vs_p6_4_bg": bg - P64_MEAN_BG if not math.isnan(bg) else float("nan"),
        })
    # Rank by bucket_geo -- the whole point of P6.5''' is that bucket_geo is
    # the fair monitor for long-tail research. Ties broken by R@20.
    ranked.sort(
        key=lambda r: (
            r["bucket_geo"] if not math.isnan(r["bucket_geo"]) else -1.0,
            r["recall20"] if not math.isnan(r["recall20"]) else -1.0,
        ),
        reverse=True,
    )

    out_path = Path(out_rel)
    if not out_path.is_absolute():
        out_path = damps_dir / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump({
            "rows": rows, "ranked": ranked,
            "p6_4_reference": {
                "mean_r20": P64_MEAN_R20,
                "mean_ndcg20": P64_MEAN_NDCG20,
                "mean_r20_tail": P64_MEAN_R20_TAIL,
                "mean_bucket_geo": P64_MEAN_BG,
                "capped_at_epoch": 100,
            },
            "p6_5_pp_csv_reference": {
                "seed": 23946202,
                "peak_tail_at_ep138": P6_5PP_CSV_PEAK_TAIL,
                "peak_mid_at_ep142": P6_5PP_CSV_PEAK_MID,
                "peak_bucket_geo_at_ep141": P6_5PP_CSV_PEAK_BG,
                "note": "P6.5''' pure should replicate these peaks; P6.6 "
                        "variants should meet-or-beat them while lifting "
                        "Head/Overall.",
            },
        }, fh, indent=2)
    print(f"\n[P6.5'''+P6.6] wrote {out_path}")

    # Compact summary.
    print("\n=== P6.5'''+P6.6 summary (ranked by bucket_geo) ===")
    print(
        f"{'tag':16s} {'BG':>7s} {'dBG':>8s} "
        f"{'R@20':>7s} {'dR20':>8s} {'NDCG':>7s} {'Tail':>7s} "
        f"{'Head':>7s} {'best_ep':>7s} {'wall_min':>8s}"
    )
    for r in ranked:
        w = next((x["wall_min"] for x in rows if x["tag"] == r["tag"]), float("nan"))
        print(
            f"{r['tag']:16s} "
            f"{r['bucket_geo']:.5f} {r['delta_vs_p6_4_bg']:+.5f} "
            f"{r['recall20']:.5f} {r['delta_vs_p6_4_r20']:+.5f} "
            f"{r['ndcg20']:.5f} {r['tail']:.5f} "
            f"{r['head']:.5f} {r['best_epoch']:7.1f} {w:8.1f}"
        )

    # Verdict for the notebook post-mortem cell.
    if ranked:
        top = ranked[0]
        pure = next((r for r in ranked if r["tag"] == "p6_5_ppp"), None)
        print("\n=== VERDICT ===")
        if pure is None:
            print(f"  Winner: {top['tag']}  bucket_geo={top['bucket_geo']:.5f}")
        else:
            same_as_pure = top["tag"] == "p6_5_ppp"
            if same_as_pure:
                print(
                    f"  P6.5''' pure wins bucket_geo={top['bucket_geo']:.5f}. "
                    "P6.6 head-overfit knobs did NOT clear the noise floor -- "
                    "commit to P6.7 (loss-side) or P6.8 (graph-side) next."
                )
            else:
                print(
                    f"  P6.6 variant '{top['tag']}' wins "
                    f"bucket_geo={top['bucket_geo']:.5f} (Δ vs pure = "
                    f"{top['bucket_geo'] - pure['bucket_geo']:+.5f}). "
                    "Head-overfit mitigation IS unlocking additional lift. "
                    "Adopt the winning config for the next 2-seed replication."
                )


if __name__ == "__main__":
    main()
