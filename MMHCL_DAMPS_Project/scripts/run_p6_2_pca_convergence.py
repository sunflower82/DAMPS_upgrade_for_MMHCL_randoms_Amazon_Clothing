"""scripts/run_p6_2_pca_convergence.py -- P6.2 convergence extension.

Ships Priority 6.2 (P6.2) of the PACER-NRDMC upgrade roadmap.

Context
-------
P6.0 established that ``p6a_pca_only`` (text=replace_pca, image=raw) is
the winner on Amazon Clothing at 60 epochs (mean R@20 = 0.08742). Two
observations from the P6.0 log motivate a longer run:

  1. alpha_txt peaks near +0.63 around epoch 20 and then decays back
     to +0.34 by epoch 55. The model is still learning to weight the
     text signal but the CL loss is slowly pulling it back toward
     zero -- indicating capacity, not headroom, is the bottleneck.
  2. R@20 was still climbing (or plateauing very late) at epoch 55
     under ``use_reduce_lr=1``. Early-stopping triggered on val R@20
     but the LR-plateau schedule likely had another cooldown left.

P6.2 reruns the winner with:

  * ``epoch=100`` (up from 60)
  * ``patience=30`` (up from 20)
  * 1 seed (23946202) to keep budget minimal

Success criterion: strictly higher mean R@20 than the P6.0 winner's
seed-23946202 point (0.08825638). Failure = plateau confirmed -->
P6.4/P6.5 (alpha-floor, CL retune) become the next moves.

Total budget: 1 seed x ~17 s/epoch x 100 epochs ~= 28 minutes A100.

Usage (from MMHCL_DAMPS_Project/)::

    python scripts/run_p6_2_pca_convergence.py
    python scripts/run_p6_2_pca_convergence.py --dry_run 1
    python scripts/run_p6_2_pca_convergence.py --seeds 23946202 1557638902
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
#  Locked PACER + Branch A' trunk (mirrors P6.0 base flags, epoch bumped).
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

# P6.0 winner: text=replace_pca (image untouched).
TEXT_MODE = "replace_pca"
IMAGE_MODE = "raw"

SEEDS_DEFAULT: tuple[int, ...] = (23946202,)

OUT_JSON = "results/p6_2_pca_convergence_clothing.json"

_TER_TEST_RX = re.compile(
    r"\[tercile-test-final\]\s+BEST_Test_Recall@20_Head=(?P<h>[-\d.eE+nan]+)"
    r"\s+BEST_Test_Recall@20_Mid=(?P<m>[-\d.eE+nan]+)"
    r"\s+BEST_Test_Recall@20_Tail=(?P<t>[-\d.eE+nan]+)"
)
_BEST_RX = re.compile(r"BEST_Test_Recall@20\s*[:=]\s*([-\d.eE+nan]+)")
_BEST_NDCG_RX = re.compile(r"BEST_Test_NDCG@20\s*[:=]\s*([-\d.eE+nan]+)")
_VAL_R20_RX = re.compile(r"BEST_Val_Recall@20\s*[:=]\s*([-\d.eE+nan]+)")
_BEST_EPOCH_RX = re.compile(r"best_epoch\s*[:=]\s*(\d+)")


@dataclass(frozen=True)
class RunConfig:
    tag: str
    epoch: int
    patience: int


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
    """P6.2 only needs the text streams (image stays raw)."""
    data_dir = damps_dir.parent / "data" / DATASET
    missing = [str(data_dir / n) for n in
               ("text_feat_pca_ica.npy", "text_feat_zca.npy")
               if not (data_dir / n).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing MACP text stream(s):\n  - "
            + "\n  - ".join(missing)
            + "\n\nRun this first:\n"
            + f"  python scripts/preprocess_macp.py --dataset {DATASET} "
              f"--modality text"
        )


def _base_flags(
    *,
    cfg: RunConfig,
    wb_project: str,
    wb_entity: str,
) -> list[str]:
    return [
        "--dataset", DATASET,
        "--gpu_id", "0",
        "--epoch", str(cfg.epoch),
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
        # ---- CL reweight (control level) ------------------------------
        "--user_loss_ratio", str(R_U),
        "--item_loss_ratio", str(R_I),
        "--temperature", str(TEMPERATURE),
        # ---- L_align OFF (P6.2 = P6.0 winner + longer training) -------
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
        "--early_stopping_patience", str(cfg.patience),
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
        # ---- P6.0 winner: text=replace_pca, image=raw -----------------
        "--use_macp", "1",
        "--macp_mode", TEXT_MODE,
        "--macp_alpha_p", "0.0",
        "--macp_alpha_z", "0.0",
        "--macp_image_mode", IMAGE_MODE,
        "--macp_image_alpha_p", "0.0",
        "--macp_image_alpha_z", "0.0",
        "--macp_verbose", "1",
        # ---- WandB -----------------------------------------------------
        "--use_wandb", "1",
        "--wandb_project", wb_project,
        "--wandb_entity", wb_entity,
        "--wandb_group", "p6_2_pca_convergence",
        "--wandb_tags",
        "p6,p6_2,macp,tamer,text_whitening,long_epoch,nrdmc_lite",
    ]


def _run_one(
    *,
    python_exe: str,
    damps_dir: Path,
    cfg: RunConfig,
    seed: int,
    wb_project: str,
    wb_entity: str,
    dry_run: bool,
) -> dict[str, Any]:
    run_name = f"p6_2_{cfg.tag}_seed{seed}"
    cmd = [
        python_exe, "main_tercile.py",
        *_base_flags(cfg=cfg, wb_project=wb_project, wb_entity=wb_entity),
        "--seed", str(seed),
        "--wandb_run_name", run_name,
    ]
    print(
        f"\n{'=' * 74}\n"
        f"[P6.2] cfg={cfg.tag}  text_mode={TEXT_MODE}  "
        f"image_mode={IMAGE_MODE}  "
        f"epoch={cfg.epoch}  patience={cfg.patience}  seed={seed}\n"
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
            "epoch_cap": cfg.epoch, "patience": cfg.patience,
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
        "epoch_cap": cfg.epoch, "patience": cfg.patience,
        "test_head": _f(mt.group("h") if mt else None),
        "test_mid":  _f(mt.group("m") if mt else None),
        "test_tail": _f(mt.group("t") if mt else None),
        "best_test_recall20": _f(b.group(1) if b else None),
        "best_test_ndcg20":   _f(n.group(1) if n else None),
        "best_val_recall20":  _f(v.group(1) if v else None),
        "best_epoch": (float(e.group(1)) if e else float("nan")),
    }
    print(
        f"[P6.2] {cfg.tag} seed={seed}  wall={wall:.1f}m  "
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
        description="P6.2 PCA winner convergence extension (long-epoch).")
    p.add_argument("--dry_run", type=int, default=0)
    p.add_argument("--skip_preflight", type=int, default=0)
    p.add_argument("--epoch", type=int, default=EPOCH_DEFAULT,
                   help="Max epoch cap. Default 100 (up from P6.0's 60).")
    p.add_argument("--patience", type=int, default=PATIENCE_DEFAULT,
                   help="Early-stopping patience. Default 30 "
                        "(up from P6.0's 20).")
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS_DEFAULT),
                   help="Seeds to run. Default: single seed (23946202) "
                        "for a 28-min budget; pass 2 seeds for a "
                        "std-tightened estimate at 2x the cost.")
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

    cfg = RunConfig(
        tag="p6_2_pca_only_long",
        epoch=int(args.epoch),
        patience=int(args.patience),
    )
    seeds = list(args.seeds)
    dry = bool(args.dry_run)
    print(
        f"[P6.2] 1 config x {len(seeds)} seed(s) x {cfg.epoch} epoch cap  "
        f"patience={cfg.patience}  dry_run={int(dry)}",
        flush=True,
    )
    print(f"[P6.2] Config: text={TEXT_MODE}, image={IMAGE_MODE} "
          f"(P6.0 winner). Reference target: R@20 > 0.08826 "
          f"(seed 23946202 at 60 epochs).", flush=True)

    per_seed: list[dict[str, Any]] = []
    for i, seed in enumerate(seeds, 1):
        print(f"\n[P6.2] progress {i}/{len(seeds)}", flush=True)
        per_seed.append(
            _run_one(
                python_exe=python_exe, damps_dir=damps_dir,
                cfg=cfg, seed=seed,
                wb_project=args.wandb_project,
                wb_entity=args.wandb_entity,
                dry_run=dry,
            )
        )

    ranked = [{
        "tag": cfg.tag,
        "epoch_cap": cfg.epoch,
        "patience": cfg.patience,
        "best_test_recall20": _agg(
            [float(r["best_test_recall20"]) for r in per_seed]),
        "best_test_ndcg20": _agg(
            [float(r["best_test_ndcg20"]) for r in per_seed]),
        "best_val_recall20": _agg(
            [float(r["best_val_recall20"]) for r in per_seed]),
        "test_head": _agg([float(r["test_head"]) for r in per_seed]),
        "test_mid":  _agg([float(r["test_mid"])  for r in per_seed]),
        "test_tail": _agg([float(r["test_tail"]) for r in per_seed]),
        "best_epoch": _agg([float(r["best_epoch"]) for r in per_seed]),
        "n_ok": sum(1 for r in per_seed if int(r["exit"]) == 0),
    }]

    payload: dict[str, Any] = {
        "meta": {
            "phase": "P6.2",
            "dataset": DATASET,
            "epoch_cap": cfg.epoch,
            "patience": cfg.patience,
            "seeds": seeds,
            "text_mode": TEXT_MODE,
            "image_mode": IMAGE_MODE,
            "reference_p6_0_seed23946202_recall20": 0.08825638,
            "description": (
                "P6.2 convergence extension of the P6.0 winner "
                "(text=replace_pca, image=raw). Epoch cap raised to "
                "100, patience to 30. Success if mean R@20 strictly "
                "improves over the P6.0 reference (0.08826 at seed "
                "23946202, 60 epochs)."
            ),
        },
        "per_seed": per_seed,
        "ranked": ranked,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n[P6.2] wrote {out_path}", flush=True)
    r = ranked[0]
    m = r["best_test_recall20"]
    print(
        f"[P6.2] mean R@20 = {m['mean']:.5f} +/- {m['std']:.5f}  "
        f"(n_ok={r['n_ok']}, best_epoch mean={r['best_epoch']['mean']})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
