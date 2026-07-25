"""scripts/run_p6_3_p61_convergence.py -- P6.3 P6.1-winner convergence extension.

Ships Priority 6.3 (P6.3) of the PACER-NRDMC upgrade roadmap.

Context
-------
P6.1 shifted the winner from ``p6a_pca_only`` (text=replace_pca, image=raw)
to a SYMMETRIC MACP setting where BOTH modalities are decorrelated:

  * p61_text_pca_image_pca : mean R@20 = 0.09172 +/- 0.0017 (2 seeds)
  * p61_text_pca_image_zca : mean R@20 = 0.09171 +/- 0.0005 (2 seeds)
  * p61_text_pca_only      : mean R@20 = 0.08746 +/- 0.0002 (P6.0 control)

Both winners peak at ``best_epoch ~= 55`` under the P6.0 recipe
(``epoch=60, patience=20``). The Test_Recall@20 trajectory is STILL
climbing at epoch 55:

  seed 23946202 (image_pca) : R@20 = 0.09274 @ ep55  --> BEST 0.09294 @ ep56
  seed 1557638902 (image_pca): R@20 = 0.09050 @ ep55 (val peak epoch 55)

P6.2 established (on the P6.0 winner) that raising ``epoch=100 /
patience=30`` recovers +0.25% on overall R@20 but +29% on Mid and +31%
on Tail. Since the P6.1 alpha trajectory looks even less converged than
P6.0 -- alpha_txt keeps drifting from +0.63 to +0.34 between ep20 and
ep55 -- the same lever should transfer to the P6.1 winner.

P6.3 grid
---------
Two image whitening modes x two seeds x 100 epochs, patience 30.

  * p6_3_image_pca_long : text=replace_pca image=replace_pca (P6.1 mean winner)
  * p6_3_image_zca_long : text=replace_pca image=replace_zca (P6.1 std-tightest)

Seeds: 23946202, 1557638902 (same as P6.1 for direct comparability).

Success criterion
-----------------
Per-cell mean R@20 strictly higher than the corresponding P6.1 mean:

  image_pca : > 0.09172
  image_zca : > 0.09171

Total budget: 2 cells * 2 seeds * ~30 min A100 ~= 2h.

Usage (from MMHCL_DAMPS_Project/)::

    python scripts/run_p6_3_p61_convergence.py
    python scripts/run_p6_3_p61_convergence.py --dry_run 1
    python scripts/run_p6_3_p61_convergence.py --only image_zca
    python scripts/run_p6_3_p61_convergence.py --seeds 23946202
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
#  Locked PACER + Branch A' trunk (mirrors P6.1 base flags, epoch bumped).
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

BASE_DAMPS_APC = 0
BASE_LEARNABLE_TAU = 0

R_U = 0.03
R_I = 0.07

# Fix text at the P6.0 winner (unchanged across P6.1 + P6.3).
TEXT_MODE = "replace_pca"

# First two MMHCL-paired seeds (matches P1-P6.2 ordering).
SEEDS_DEFAULT: tuple[int, ...] = (23946202, 1557638902)

OUT_JSON = "results/p6_3_p61_convergence_clothing.json"

# P6.1 grid means (for the payload / logging references).
P61_MEAN_R20_IMAGE_PCA = 0.09172
P61_MEAN_R20_IMAGE_ZCA = 0.09171

_TER_TEST_RX = re.compile(
    r"\[tercile-test-final\]\s+BEST_Test_Recall@20_Head=(?P<h>[-\d.eE+nan]+)"
    r"\s+BEST_Test_Recall@20_Mid=(?P<m>[-\d.eE+nan]+)"
    r"\s+BEST_Test_Recall@20_Tail=(?P<t>[-\d.eE+nan]+)"
)
_BEST_RX = re.compile(r"BEST_Test_Recall@20\s*[:=]\s*([-\d.eE+nan]+)")
_BEST_NDCG_RX = re.compile(r"BEST_Test_NDCG@20\s*[:=]\s*([-\d.eE+nan]+)")
_VAL_R20_RX = re.compile(r"BEST_Val_Recall@20\s*[:=]\s*([-\d.eE+nan]+)")
# The trainer emits ``BEST_Val_Recall_Peak_Epoch: <int>`` (and a matching
# NDCG_Peak_Epoch). We prefer the val-recall peak as our best_epoch signal
# because early-stopping is driven by val_recall@20. Fall back to a
# generic ``best_epoch`` for older logs.
_BEST_EPOCH_RX = re.compile(
    r"(?:BEST_Val_Recall_Peak_Epoch|best_epoch)\s*[:=]\s*(\d+)"
)


@dataclass(frozen=True)
class GridConfig:
    """One cell of the P6.3 convergence-extension grid."""
    tag: str
    image_mode: str          # 'replace_pca' | 'replace_zca'
    p61_reference_mean: float


def build_configs(only: str | None = None) -> list[GridConfig]:
    all_cfgs = [
        # P6.1 mean winner: symmetric PCA.
        GridConfig(
            tag="p6_3_image_pca_long",
            image_mode="replace_pca",
            p61_reference_mean=P61_MEAN_R20_IMAGE_PCA,
        ),
        # P6.1 std-tightest: symmetric ZCA.
        GridConfig(
            tag="p6_3_image_zca_long",
            image_mode="replace_zca",
            p61_reference_mean=P61_MEAN_R20_IMAGE_ZCA,
        ),
    ]
    if only is None:
        return all_cfgs
    key = only.strip().lower()
    aliases = {
        "pca": "p6_3_image_pca_long",
        "image_pca": "p6_3_image_pca_long",
        "p6_3_image_pca_long": "p6_3_image_pca_long",
        "zca": "p6_3_image_zca_long",
        "image_zca": "p6_3_image_zca_long",
        "p6_3_image_zca_long": "p6_3_image_zca_long",
    }
    if key not in aliases:
        raise ValueError(
            f"--only got {only!r}; expected one of pca/image_pca/zca/image_zca."
        )
    tag = aliases[key]
    return [c for c in all_cfgs if c.tag == tag]


def _f(x: str | None) -> float:
    if x is None:
        return float("nan")
    try:
        return float(x)
    except ValueError:
        return float("nan")


def _agg(vals: list[float]) -> dict[str, float]:
    finite = [v for v in vals if not math.isnan(v)]
    if not finite:
        return {"mean": float("nan"), "std": float("nan"), "n": 0.0}
    return {
        "mean": float(statistics.mean(finite)),
        "std":  float(statistics.stdev(finite)) if len(finite) > 1 else 0.0,
        "n":    float(len(finite)),
    }


def _resolve_paths() -> tuple[Path, Path, str]:
    cwd = Path.cwd().resolve()
    if cwd.name == "MMHCL_DAMPS_Project":
        damps = cwd
    elif (cwd / "MMHCL_DAMPS_Project").is_dir():
        damps = cwd / "MMHCL_DAMPS_Project"
    else:
        here = Path(__file__).resolve().parent
        damps = here.parent if here.name == "scripts" else cwd
    root = damps.parent
    rtx = Path(r"c:\ProgramData\anaconda3\envs\rtx5090_dl\python.exe")
    py = str(rtx) if rtx.is_file() else sys.executable
    return damps, root, py


def _check_macp_streams(damps_dir: Path) -> None:
    """P6.3 needs BOTH text and image whitened streams."""
    data_dir = damps_dir.parent / "data" / DATASET
    needed = (
        "text_feat_pca_ica.npy",
        "text_feat_zca.npy",
        "image_feat_pca_ica.npy",
        "image_feat_zca.npy",
    )
    missing = [str(data_dir / n) for n in needed
               if not (data_dir / n).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing MACP stream(s):\n  - "
            + "\n  - ".join(missing)
            + "\n\nRun this first (both modalities):\n"
            + f"  python scripts/preprocess_macp.py --dataset {DATASET} "
              f"--modality both"
        )


def _base_flags(
    *,
    cfg: GridConfig,
    epoch: int,
    patience: int,
    wb_project: str,
    wb_entity: str,
) -> list[str]:
    return [
        "--dataset", DATASET,
        "--gpu_id", "0",
        "--epoch", str(epoch),
        "--verbose", "5",
        "--eval_every", "5",
        "--eval_last_epochs", "30",
        "--use_gpu_eval", "1",
        "--use_torch_compile", "1",
        "--torch_compile_mode", "default",
        "--torch_compile_dynamic", "0",
        "--use_cuda_graph", "0",
        "--batch_size", str(BATCH_SIZE),
        "--lr", str(LR),
        "--regs", str(REGS),
        "--embed_size", str(EMBED_SIZE),
        "--topk", str(KNN_TOPK),
        "--core", "5",
        "--UI_layers", str(UI_LAYERS),
        "--User_layers", str(U_LAYERS),
        "--Item_layers", str(I_LAYERS),
        # ---- CL reweight (control level, unchanged since P5.1) --------
        "--user_loss_ratio", str(R_U),
        "--item_loss_ratio", str(R_I),
        "--temperature", str(TEMPERATURE),
        # ---- L_align OFF (P6.3 = P6.1 winner + longer training) -------
        "--enable_align", "0",
        "--lambda_align", "0.0",
        "--align_temperature", "0.2",
        # ---- P5.0-derived base ---------------------------------------
        "--damps_apc", str(BASE_DAMPS_APC),
        "--learnable_tau", str(BASE_LEARNABLE_TAU),
        "--damps_avrf", "0",
        "--damps_imcf", "1",
        "--damps_soft_routing", "1",
        "--damps_momentum", "1",
        "--damps_data_driven_prior", "1",
        "--damps_permutation_fft", "0",
        "--damps_warmup_epochs", "10",
        "--damps_num_categories", "10",
        "--enable_logq", "1",
        "--logq_mode", "laplace",
        "--logq_beta", str(LOGQ_BETA),
        "--logq_scale", str(LOGQ_SCALE),
        "--logq_clip", str(LOGQ_CLIP),
        "--enable_simgcl", "0",
        "--enable_nrdmc_lite", "1",
        "--nrdmc_lite_layers", str(NRDMC_LITE_LAYERS),
        "--lambda_view", str(LAMBDA_VIEW),
        "--enable_ptv", "0",
        "--n_prototypes", "0",
        "--lambda_ptv", "0.0",
        "--simgcl_eps", str(SIMGCL_EPS),
        "--branchA_view_bsz", "2048",
        "--branchA_bcl_bsz", "2048",
        "--branchA_bcl_batchn", "1",
        "--early_stopping_patience", str(patience),
        "--early_stopping_min_epochs", "0",
        "--early_stopping_min_delta", "1e-4",
        "--early_stopping_monitor", "val_recall@20",
        "--early_stopping_mode", "max",
        "--early_stopping_restore_best", "1",
        "--use_reduce_lr", str(USE_REDUCE_LR),
        "--use_amp", "1",
        "--asc_gate_mode", "raw",
        "--asc_warmup_epochs", "0",
        "--asc_reg_l2", "0.0",
        "--asc_reg_target", "0.3",
        # ---- P6.1 symmetric MACP: text=replace_pca, image=<cfg> -------
        "--use_macp", "1",
        "--macp_mode", TEXT_MODE,
        "--macp_alpha_p", "0.0",
        "--macp_alpha_z", "0.0",
        "--macp_image_mode", cfg.image_mode,
        "--macp_image_alpha_p", "0.0",
        "--macp_image_alpha_z", "0.0",
        "--macp_verbose", "1",
        # ---- WandB -----------------------------------------------------
        "--use_wandb", "1",
        "--wandb_project", wb_project,
        "--wandb_entity", wb_entity,
        "--wandb_group", "p6_3_p61_convergence",
        "--wandb_tags",
        "p6,p6_3,macp,tamer,text_whitening,image_whitening,"
        "long_epoch,nrdmc_lite",
    ]


def _run_one(
    *,
    python_exe: str,
    damps_dir: Path,
    cfg: GridConfig,
    epoch: int,
    patience: int,
    seed: int,
    wb_project: str,
    wb_entity: str,
    dry_run: bool,
) -> dict[str, Any]:
    run_name = f"p6_3_{cfg.tag}_seed{seed}"
    cmd = [
        python_exe, "main_tercile.py",
        *_base_flags(
            cfg=cfg, epoch=epoch, patience=patience,
            wb_project=wb_project, wb_entity=wb_entity,
        ),
        "--seed", str(seed),
        "--wandb_run_name", run_name,
    ]
    print(
        f"\n{'=' * 74}\n"
        f"[P6.3] cfg={cfg.tag}  text_mode={TEXT_MODE}  "
        f"image_mode={cfg.image_mode}  "
        f"epoch={epoch}  patience={patience}  seed={seed}\n"
        f"{'=' * 74}",
        flush=True,
    )
    print(
        "[cmd] " + " ".join(cmd[:8])
        + f" ... [{len(cmd) - 8} more flags]",
        flush=True,
    )
    if dry_run:
        return {
            "tag": cfg.tag, "seed": seed, "exit": 0, "wall_min": 0.0,
            "dry_run": True,
            "epoch_cap": epoch, "patience": patience,
            "text_mode": TEXT_MODE, "image_mode": cfg.image_mode,
            "test_head": float("nan"),
            "test_mid":  float("nan"),
            "test_tail": float("nan"),
            "best_test_recall20": float("nan"),
            "best_test_ndcg20":   float("nan"),
            "best_val_recall20":  float("nan"),
            "best_epoch":         float("nan"),
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
    chunks: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        chunks.append(line)
        print(line, end="", flush=True)
    rc = proc.wait()
    out = "".join(chunks)
    wall = (time.time() - t0) / 60.0
    if rc != 0:
        print(f"[WARN] cfg={cfg.tag} seed={seed} exited {rc}", flush=True)

    mt = _TER_TEST_RX.search(out)
    b = _BEST_RX.search(out)
    n = _BEST_NDCG_RX.search(out)
    v = _VAL_R20_RX.search(out)
    e = _BEST_EPOCH_RX.search(out)
    row = {
        "tag": cfg.tag, "seed": seed, "exit": rc,
        "wall_min": wall, "dry_run": False,
        "epoch_cap": epoch, "patience": patience,
        "text_mode": TEXT_MODE, "image_mode": cfg.image_mode,
        "test_head": _f(mt.group("h") if mt else None),
        "test_mid":  _f(mt.group("m") if mt else None),
        "test_tail": _f(mt.group("t") if mt else None),
        "best_test_recall20": _f(b.group(1) if b else None),
        "best_test_ndcg20":   _f(n.group(1) if n else None),
        "best_val_recall20":  _f(v.group(1) if v else None),
        "best_epoch": (float(e.group(1)) if e else float("nan")),
    }
    print(
        f"[P6.3] {cfg.tag} seed={seed}  wall={wall:.1f}m  "
        f"R@20={row['best_test_recall20']:.5f}  "
        f"N@20={row['best_test_ndcg20']:.5f}  "
        f"best_epoch={row['best_epoch']}  "
        f"H/M/T="
        f"{row['test_head']:.5f}/{row['test_mid']:.5f}/{row['test_tail']:.5f}",
        flush=True,
    )
    return row


def parse_cli(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="P6.3 P6.1-winner convergence extension (long-epoch).")
    p.add_argument("--dry_run", type=int, default=0)
    p.add_argument("--skip_preflight", type=int, default=0)
    p.add_argument("--epoch", type=int, default=EPOCH_DEFAULT,
                   help="Max epoch cap. Default 100 (up from P6.1's 60).")
    p.add_argument("--patience", type=int, default=PATIENCE_DEFAULT,
                   help="Early-stopping patience. Default 30 "
                        "(up from P6.1's 20).")
    p.add_argument("--seeds", type=int, nargs="+",
                   default=list(SEEDS_DEFAULT),
                   help="Seeds to run. Default: 2 seeds "
                        "(23946202, 1557638902) matching P6.1 order.")
    p.add_argument("--only", type=str, default=None,
                   choices=["pca", "image_pca", "zca", "image_zca",
                            "p6_3_image_pca_long", "p6_3_image_zca_long"],
                   help="Restrict to one cell. Default: both cells.")
    p.add_argument("--wandb_project", type=str,
                   default=os.environ.get("WANDB_PROJECT",
                                          "damps-mmhcl-clothing"))
    p.add_argument("--wandb_entity", type=str,
                   default=os.environ.get("WANDB_ENTITY",
                                          "baitapck51cc-uet"))
    p.add_argument("--out_json", type=str, default=OUT_JSON)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_cli(argv)
    damps_dir, _root, python_exe = _resolve_paths()
    if not (damps_dir / "main_tercile.py").is_file():
        raise FileNotFoundError(
            f"main_tercile.py missing under {damps_dir}.")
    os.chdir(damps_dir)
    Path("results").mkdir(exist_ok=True)

    if not (args.dry_run or args.skip_preflight):
        _check_macp_streams(damps_dir)

    configs = build_configs(only=args.only)
    seeds = list(args.seeds)
    dry = bool(args.dry_run)
    epoch = int(args.epoch)
    patience = int(args.patience)

    print(
        f"[P6.3] {len(configs)} config(s) x {len(seeds)} seed(s) x "
        f"{epoch} epoch cap  patience={patience}  dry_run={int(dry)}",
        flush=True,
    )
    print(f"[P6.3] text_mode fixed at '{TEXT_MODE}' (P6.0/P6.1 winner).",
          flush=True)
    print("[P6.3] Cells:", flush=True)
    for c in configs:
        print(
            f"   - {c.tag}: image_mode={c.image_mode}  "
            f"P6.1 reference R@20={c.p61_reference_mean:.5f}",
            flush=True,
        )

    per_seed: list[dict[str, Any]] = []
    total = len(configs) * len(seeds)
    step = 0
    for cfg in configs:
        for seed in seeds:
            step += 1
            print(f"\n[P6.3] progress {step}/{total}", flush=True)
            per_seed.append(
                _run_one(
                    python_exe=python_exe, damps_dir=damps_dir,
                    cfg=cfg, epoch=epoch, patience=patience,
                    seed=seed,
                    wb_project=args.wandb_project,
                    wb_entity=args.wandb_entity,
                    dry_run=dry,
                )
            )

    # Aggregate per-cell.
    ranked: list[dict[str, Any]] = []
    for cfg in configs:
        rows = [r for r in per_seed if r["tag"] == cfg.tag]
        cell = {
            "tag": cfg.tag,
            "text_mode": TEXT_MODE,
            "image_mode": cfg.image_mode,
            "epoch_cap": epoch,
            "patience": patience,
            "p61_reference_mean_r20": cfg.p61_reference_mean,
            "best_test_recall20": _agg(
                [float(r["best_test_recall20"]) for r in rows]),
            "best_test_ndcg20": _agg(
                [float(r["best_test_ndcg20"]) for r in rows]),
            "best_val_recall20": _agg(
                [float(r["best_val_recall20"]) for r in rows]),
            "test_head": _agg([float(r["test_head"]) for r in rows]),
            "test_mid":  _agg([float(r["test_mid"])  for r in rows]),
            "test_tail": _agg([float(r["test_tail"]) for r in rows]),
            "best_epoch": _agg([float(r["best_epoch"]) for r in rows]),
            "n_ok": sum(1 for r in rows if int(r["exit"]) == 0),
        }
        ranked.append(cell)

    # Sort by mean R@20 (nan last).
    ranked.sort(
        key=lambda c: (
            0 if not math.isnan(c["best_test_recall20"]["mean"]) else 1,
            -(c["best_test_recall20"]["mean"]
              if not math.isnan(c["best_test_recall20"]["mean"]) else 0.0),
        )
    )

    payload: dict[str, Any] = {
        "meta": {
            "phase": "P6.3",
            "dataset": DATASET,
            "epoch_cap": epoch,
            "patience": patience,
            "seeds": seeds,
            "text_mode": TEXT_MODE,
            "image_modes": [c.image_mode for c in configs],
            "p61_reference": {
                "image_pca_mean_r20": P61_MEAN_R20_IMAGE_PCA,
                "image_zca_mean_r20": P61_MEAN_R20_IMAGE_ZCA,
            },
            "description": (
                "P6.3 convergence extension of the P6.1 symmetric-MACP "
                "winners (text=replace_pca, image in {replace_pca, "
                "replace_zca}). Epoch cap raised from 60 to 100 and "
                "patience from 20 to 30 to let alpha_txt fully settle "
                "past its ep20 peak of +0.63 and let ReduceLROnPlateau "
                "consume its remaining cooldowns. Success = per-cell "
                "mean R@20 strictly above the P6.1 reference "
                "(0.09172 for image_pca, 0.09171 for image_zca)."
            ),
        },
        "per_seed": per_seed,
        "ranked": ranked,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n[P6.3] wrote {out_path}", flush=True)
    print("[P6.3] ranked (by mean R@20 desc):", flush=True)
    for c in ranked:
        m = c["best_test_recall20"]
        ref = c["p61_reference_mean_r20"]
        delta_pct = (
            100.0 * (m["mean"] - ref) / ref
            if not math.isnan(m["mean"]) and ref > 0 else float("nan")
        )
        be = c["best_epoch"]
        print(
            f"   [{c['tag']:24s}] R@20={m['mean']:.5f} +/- {m['std']:.5f}  "
            f"(n_ok={c['n_ok']})  vs P6.1 {ref:.5f}  "
            f"delta={delta_pct:+.2f}%  best_epoch_mean={be['mean']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
