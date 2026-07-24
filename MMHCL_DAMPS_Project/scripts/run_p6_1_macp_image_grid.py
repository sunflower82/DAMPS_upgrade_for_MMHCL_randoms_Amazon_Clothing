"""scripts/run_p6_1_macp_image_grid.py -- P6.1 MACP image whitening sweep.

Ships Priority 6.1 (P6.1) of the PACER-NRDMC upgrade roadmap.

Context
-------
P6.0 established that the ``replace_pca`` text-whitening cell wins on
Amazon Clothing (mean R@20 = 0.08742 vs 0.08160 baseline = +7.13 %;
mid tercile +36.8 %, tail +46.4 %). The winning alpha_txt trajectory
peaks near +0.63 -- text under whitening becomes genuinely useful.

However, the same log shows alpha_img collapses harder under text
whitening (to -0.84 by epoch 55, vs -0.51 in the P5.1 raw-text
control). Two competing hypotheses:

  H1 (image inherently noise): raw and whitened image are both
       useless on Clothing, so the model correctly discards image
       once text carries the signal. Whitening will NOT reverse the
       collapse.
  H2 (image geometry leakage): raw image covariance also blocks
       useful factors, exactly like raw text did. Whitening will
       let alpha_img recover to positive.

P6.1 tests H1 vs H2 with a 3-cell sweep (2 seeds each) that keeps
text at the P6.0 winner and varies image mode:

    p61_text_pca_only       text=replace_pca  image=raw          (P6.0 winner replay)
    p61_text_pca_image_pca  text=replace_pca  image=replace_pca
    p61_text_pca_image_zca  text=replace_pca  image=replace_zca

Prerequisite
------------
Materialise all four MACP streams once:

    python scripts/preprocess_macp.py --dataset Clothing --modality both

which writes ``text_feat_{pca_ica,zca}.npy`` and
``image_feat_{pca_ica,zca}.npy`` next to the raw ``*_feat.npy`` files.

Total budget: 3 cells x 2 seeds x ~17 s/epoch x 60 epochs ~= 1.7 h A100.

Usage (from MMHCL_DAMPS_Project/)::

    python scripts/run_p6_1_macp_image_grid.py
    python scripts/run_p6_1_macp_image_grid.py --dry_run 1
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
#  Locked PACER + Branch A' trunk (mirrors P6.0 base flags exactly).
# --------------------------------------------------------------------------- #
DATASET = "Clothing"
EPOCH = 60
PATIENCE = 20
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

# Fix text at the P6.0 winner across all P6.1 cells.
TEXT_MODE = "replace_pca"

# First two MMHCL-paired seeds (matches P1-P6.0 ordering).
SEEDS: tuple[int, ...] = (23946202, 1557638902)

OUT_JSON = "results/p6_1_macp_image_grid_clothing.json"

_TER_TEST_RX = re.compile(
    r"\[tercile-test-final\]\s+BEST_Test_Recall@20_Head=(?P<h>[-\d.eE+nan]+)"
    r"\s+BEST_Test_Recall@20_Mid=(?P<m>[-\d.eE+nan]+)"
    r"\s+BEST_Test_Recall@20_Tail=(?P<t>[-\d.eE+nan]+)"
)
_BEST_RX = re.compile(r"BEST_Test_Recall@20\s*[:=]\s*([-\d.eE+nan]+)")
_BEST_NDCG_RX = re.compile(r"BEST_Test_NDCG@20\s*[:=]\s*([-\d.eE+nan]+)")
_VAL_R20_RX = re.compile(r"BEST_Val_Recall@20\s*[:=]\s*([-\d.eE+nan]+)")


@dataclass(frozen=True)
class GridConfig:
    """One cell of the P6.1 MACP image sweep."""
    tag: str
    image_mode: str          # 'raw' | 'replace_pca' | 'replace_zca'
    image_alpha_p: float
    image_alpha_z: float


def build_configs() -> list[GridConfig]:
    return [
        # H0: replay P6.0 winner (image raw). Sanity check under new code path.
        GridConfig("p61_text_pca_only",      "raw",         0.0, 0.0),
        # H2a: symmetric PCA->ICA on image.
        GridConfig("p61_text_pca_image_pca", "replace_pca", 0.0, 0.0),
        # H2b: symmetric ZCA on image.
        GridConfig("p61_text_pca_image_zca", "replace_zca", 0.0, 0.0),
    ]


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
    """Return (damps_dir, project_root, python_exe)."""
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
    """Fail early if any of the four MACP outputs are missing."""
    data_dir = damps_dir.parent / "data" / DATASET
    required = (
        "text_feat_pca_ica.npy", "text_feat_zca.npy",
        "image_feat_pca_ica.npy", "image_feat_zca.npy",
    )
    missing = [str(data_dir / n) for n in required
               if not (data_dir / n).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing MACP stream(s):\n  - "
            + "\n  - ".join(missing)
            + "\n\nRun this first:\n"
            + f"  python scripts/preprocess_macp.py --dataset {DATASET} "
              f"--modality both"
        )


def _base_flags(
    *,
    cfg: GridConfig,
    wb_project: str,
    wb_entity: str,
) -> list[str]:
    return [
        "--dataset", DATASET,
        "--gpu_id", "0",
        "--epoch", str(EPOCH),
        "--verbose", "5",
        "--eval_every", "5",
        "--eval_last_epochs", "20",
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
        # ---- L_align OFF (P6.1 isolates image MACP) -------------------
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
        "--early_stopping_patience", str(PATIENCE),
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
        # ---- P6.0 MACP text -- fixed at winner ------------------------
        "--use_macp", "1",
        "--macp_mode", TEXT_MODE,
        "--macp_alpha_p", "0.0",
        "--macp_alpha_z", "0.0",
        # ---- P6.1 MACP image -- swept in this driver ------------------
        "--macp_image_mode", cfg.image_mode,
        "--macp_image_alpha_p", str(float(cfg.image_alpha_p)),
        "--macp_image_alpha_z", str(float(cfg.image_alpha_z)),
        "--macp_verbose", "1",
        # ---- WandB -----------------------------------------------------
        "--use_wandb", "1",
        "--wandb_project", wb_project,
        "--wandb_entity", wb_entity,
        "--wandb_group", "p6_1_macp_image_grid",
        "--wandb_tags",
        "p6,p6_1,macp,tamer,text_whitening,image_whitening,nrdmc_lite",
    ]


def _run_one(
    *,
    python_exe: str,
    damps_dir: Path,
    cfg: GridConfig,
    seed: int,
    wb_project: str,
    wb_entity: str,
    dry_run: bool,
) -> dict[str, Any]:
    run_name = f"p6_1_{cfg.tag}_seed{seed}"
    cmd = [
        python_exe, "main_tercile.py",
        *_base_flags(cfg=cfg, wb_project=wb_project, wb_entity=wb_entity),
        "--seed", str(seed),
        "--wandb_run_name", run_name,
    ]
    print(
        f"\n{'=' * 74}\n"
        f"[P6.1] cfg={cfg.tag}  text_mode={TEXT_MODE}  "
        f"image_mode={cfg.image_mode}  "
        f"alpha_p={cfg.image_alpha_p}  alpha_z={cfg.image_alpha_z}  "
        f"seed={seed}\n{'=' * 74}",
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
            "text_mode": TEXT_MODE,
            "image_mode": cfg.image_mode,
            "image_alpha_p": cfg.image_alpha_p,
            "image_alpha_z": cfg.image_alpha_z,
            "test_head": float("nan"),
            "test_mid":  float("nan"),
            "test_tail": float("nan"),
            "best_test_recall20": float("nan"),
            "best_test_ndcg20":   float("nan"),
            "best_val_recall20":  float("nan"),
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
    row = {
        "tag": cfg.tag, "seed": seed, "exit": rc,
        "wall_min": wall, "dry_run": False,
        "text_mode": TEXT_MODE,
        "image_mode": cfg.image_mode,
        "image_alpha_p": cfg.image_alpha_p,
        "image_alpha_z": cfg.image_alpha_z,
        "test_head": _f(mt.group("h") if mt else None),
        "test_mid":  _f(mt.group("m") if mt else None),
        "test_tail": _f(mt.group("t") if mt else None),
        "best_test_recall20": _f(b.group(1) if b else None),
        "best_test_ndcg20":   _f(n.group(1) if n else None),
        "best_val_recall20":  _f(v.group(1) if v else None),
    }
    print(
        f"[P6.1] {cfg.tag} seed={seed}  wall={wall:.1f}m  "
        f"R@20={row['best_test_recall20']:.5f}  "
        f"N@20={row['best_test_ndcg20']:.5f}  "
        f"H/M/T="
        f"{row['test_head']:.5f}/{row['test_mid']:.5f}/{row['test_tail']:.5f}",
        flush=True,
    )
    return row


def _rank_configs(per_seed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_tag: dict[str, list[dict[str, Any]]] = {}
    for row in per_seed:
        by_tag.setdefault(str(row["tag"]), []).append(row)
    ranked: list[dict[str, Any]] = []
    for tag, rows in by_tag.items():
        ranked.append({
            "tag": tag,
            "text_mode":     rows[0]["text_mode"],
            "image_mode":    rows[0]["image_mode"],
            "image_alpha_p": rows[0]["image_alpha_p"],
            "image_alpha_z": rows[0]["image_alpha_z"],
            "best_test_recall20": _agg(
                [float(r["best_test_recall20"]) for r in rows]),
            "best_test_ndcg20": _agg(
                [float(r["best_test_ndcg20"]) for r in rows]),
            "best_val_recall20": _agg(
                [float(r["best_val_recall20"]) for r in rows]),
            "test_head": _agg([float(r["test_head"]) for r in rows]),
            "test_mid":  _agg([float(r["test_mid"])  for r in rows]),
            "test_tail": _agg([float(r["test_tail"]) for r in rows]),
            "n_ok": sum(1 for r in rows if int(r["exit"]) == 0),
        })
    ranked.sort(
        key=lambda d: (
            -1.0
            if math.isnan(d["best_test_recall20"]["mean"])
            else -d["best_test_recall20"]["mean"]
        )
    )
    return ranked


def parse_cli(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="P6.1 MACP image whitening sweep (3-cell grid).")
    p.add_argument("--dry_run", type=int, default=0,
                   help="1 = print commands only.")
    p.add_argument("--skip_preflight", type=int, default=0,
                   help="1 = skip the check for {text,image}_feat_"
                        "{pca_ica,zca}.npy.")
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

    configs = build_configs()
    dry = bool(args.dry_run)
    print(
        f"[P6.1] {len(configs)} configs x {len(SEEDS)} seeds x "
        f"{EPOCH} epochs  dry_run={int(dry)}",
        flush=True,
    )
    print(f"[P6.1] text_mode fixed at '{TEXT_MODE}' (P6.0 winner).",
          flush=True)
    for c in configs:
        print(
            f"   - {c.tag}: image_mode={c.image_mode}  "
            f"alpha_p={c.image_alpha_p} alpha_z={c.image_alpha_z}",
            flush=True,
        )

    per_seed: list[dict[str, Any]] = []
    total = len(configs) * len(SEEDS)
    done = 0
    for cfg in configs:
        for seed in SEEDS:
            done += 1
            print(f"\n[P6.1] progress {done}/{total}", flush=True)
            per_seed.append(
                _run_one(
                    python_exe=python_exe, damps_dir=damps_dir,
                    cfg=cfg, seed=seed,
                    wb_project=args.wandb_project,
                    wb_entity=args.wandb_entity,
                    dry_run=dry,
                )
            )

    ranked = _rank_configs(per_seed)
    payload: dict[str, Any] = {
        "meta": {
            "phase": "P6.1",
            "dataset": DATASET,
            "epoch": EPOCH,
            "patience": PATIENCE,
            "seeds": list(SEEDS),
            "text_mode": TEXT_MODE,
            "base_preset": {
                "damps_apc": BASE_DAMPS_APC,
                "learnable_tau": BASE_LEARNABLE_TAU,
                "user_loss_ratio": R_U,
                "item_loss_ratio": R_I,
                "enable_align": 0,
            },
            "description": (
                "P6.1 MACP image whitening sweep. Text is fixed at the "
                "P6.0 winner (replace_pca) across all 3 cells; image "
                "mode is swept over {raw, replace_pca, replace_zca}. "
                "The 'raw' cell reproduces the P6.0 winner and acts as "
                "the control against which alpha_img recovery is "
                "measured. R@20 uplift over the P6.0 winner (0.08742) "
                "is the primary success criterion."
            ),
        },
        "per_seed": per_seed,
        "ranked": ranked,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n[P6.1] wrote {out_path}", flush=True)
    print("[P6.1] ranked by mean test R@20:", flush=True)
    for i, r in enumerate(ranked, 1):
        m = r["best_test_recall20"]
        print(
            f"  {i}. {r['tag']:<25}  R@20={m['mean']:.5f}"
            f" +/- {m['std']:.5f}  (n_ok={r['n_ok']})",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
