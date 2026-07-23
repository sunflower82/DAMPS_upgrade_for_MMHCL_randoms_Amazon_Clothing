"""scripts/run_p5_1_align_cl_grid.py -- P5.1 L_align + CL-reweight sweep.

Implements Priority 5.1 (P5.1) of the post-P4 diagnostic roadmap
(PACER_NRDMC_lite_upgrade_analysis_EN Section 7). P5.1 ships two
mutually complementary interventions in a single 2x2 factorial grid:

    (i)   Cross-modal alignment loss L_align (NRDMC IPM 2026 Eq. 21).
          A separate item-side symmetric InfoNCE between raw projected
          image and text embeddings. The loss is a fresh gradient
          source that steers ``image_proj`` / ``text_proj`` -- exactly
          the layers the P4 diagnostic flagged as under-driven (cl
          plateau at ~102 = ln(N) for all 60 epochs).

    (ii)  Reweight the existing multi-view CL branches. P1-P4 used
          user_loss_ratio=0.03 and item_loss_ratio=0.07 (~4% of total
          loss). This puts almost no gradient budget on the CL loss
          even when it does carry signal. P5.1 probes r_u=0.10 and
          r_i=0.15 (~15% of total).

Grid design (2 x 2 factorial x 2 seeds x 60 epochs)
---------------------------------------------------

    Cell 0  p5_control          -- P5.0 winner replica  (align=OFF, r_u=0.03/r_i=0.07)
    Cell 1  p5b_align_only      -- align ON             (align=ON,  r_u=0.03/r_i=0.07)
    Cell 2  p5b_reweight_only   -- reweight only        (align=OFF, r_u=0.10/r_i=0.15)
    Cell 3  p5b_align_reweight  -- both jointly         (align=ON,  r_u=0.10/r_i=0.15)

Total budget: 4 cells x 2 seeds x ~17 s/epoch x 60 epochs ~= 2.3 h A100.

Base (--base_from_p5_0)
-----------------------
By default P5.1 layers ON the trunk from the winner of P5.0. If the
P5.0 winner is ``p5a_apc_tau``, this driver runs with
``--damps_apc 1 --learnable_tau 1`` (over-ridable via CLI). If P5.0
has not yet been run, use ``--base_from_p5_0 p5_baseline_p4`` to
reproduce the P4-baseline trunk.

Usage (from MMHCL_DAMPS_Project/)::

    # Default: layer P5.1 on top of P5.0 winner (APC=1, tau_learn=1).
    python scripts/run_p5_1_align_cl_grid.py

    # Alternatively, layer on the P4 baseline (APC=0, tau_static):
    python scripts/run_p5_1_align_cl_grid.py --base_from_p5_0 p5_baseline_p4

    # Dry run:
    python scripts/run_p5_1_align_cl_grid.py --dry_run 1
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Locked PACER + Branch A' trunk (mirrors P5.0 base; P5.1 layers on top).
# ---------------------------------------------------------------------------
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

# P5.1 knobs (paper-informed).
LAMBDA_ALIGN_ON = 0.10
ALIGN_TAU = 0.20
R_U_HIGH = 0.10           # reweight cell -- CL user share
R_I_HIGH = 0.15           # reweight cell -- CL item share
R_U_BASE = 0.03           # locked baseline
R_I_BASE = 0.07           # locked baseline

# First two MMHCL-paired seeds (matches P1-P5.0 ordering).
SEEDS: tuple[int, ...] = (23946202, 1557638902)

OUT_JSON = "results/p5_1_align_cl_grid_clothing.json"

# --- P5.0-derived base presets --------------------------------------------
# Kept explicit so this driver is invocable independent of P5.0's output.
BASE_PRESETS: dict[str, dict[str, int]] = {
    "p5_baseline_p4": {"damps_apc": 0, "learnable_tau": 0},
    "p5a_apc":        {"damps_apc": 1, "learnable_tau": 0},
    "p5a_tau":        {"damps_apc": 0, "learnable_tau": 1},
    "p5a_apc_tau":    {"damps_apc": 1, "learnable_tau": 1},
}
DEFAULT_BASE = "p5a_apc_tau"   # optimistic: P5.0 winner; over-ridable via CLI.

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
    """One cell of the P5.1 L_align + CL-reweight sweep."""

    tag: str
    enable_align: int         # {0, 1}
    lambda_align: float       # 0.0 or LAMBDA_ALIGN_ON
    user_loss_ratio: float    # 0.03 or 0.10
    item_loss_ratio: float    # 0.07 or 0.15


def build_configs() -> list[GridConfig]:
    """Return the 4 P5.1 grid cells (2x2 factorial)."""
    return [
        GridConfig("p5_control",         0, 0.0,             R_U_BASE, R_I_BASE),
        GridConfig("p5b_align_only",     1, LAMBDA_ALIGN_ON, R_U_BASE, R_I_BASE),
        GridConfig("p5b_reweight_only",  0, 0.0,             R_U_HIGH, R_I_HIGH),
        GridConfig("p5b_align_reweight", 1, LAMBDA_ALIGN_ON, R_U_HIGH, R_I_HIGH),
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
        "std": (
            float(statistics.stdev(finite)) if len(finite) > 1 else 0.0
        ),
        "n": float(len(finite)),
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


def _base_flags(
    *,
    cfg: GridConfig,
    base_preset: dict[str, int],
    wb_project: str,
    wb_entity: str,
) -> list[str]:
    """Shared CLI flags for one P5.1 cell (seed / run-name appended later)."""
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
        # ---- P5.1 CL reweight ------------------------------------------
        "--user_loss_ratio", str(float(cfg.user_loss_ratio)),
        "--item_loss_ratio", str(float(cfg.item_loss_ratio)),
        "--temperature", str(TEMPERATURE),
        # ---- P5.1 L_align ---------------------------------------------
        "--enable_align", str(int(cfg.enable_align)),
        "--lambda_align", str(float(cfg.lambda_align)),
        "--align_temperature", str(ALIGN_TAU),
        # ---- P5.0-derived base ----------------------------------------
        "--damps_apc", str(int(base_preset["damps_apc"])),
        "--learnable_tau", str(int(base_preset["learnable_tau"])),
        # ---------------------------------------------------------------
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
        # ---- WandB ----
        "--use_wandb", "1",
        "--wandb_project", wb_project,
        "--wandb_entity", wb_entity,
        "--wandb_group", "p5_1_align_cl_grid",
        "--wandb_tags", "p5,p5_1,l_align,cl_reweight,nrdmc_lite,branchA_prime",
    ]


def _run_one(
    *,
    python_exe: str,
    damps_dir: Path,
    cfg: GridConfig,
    base_preset: dict[str, int],
    seed: int,
    wb_project: str,
    wb_entity: str,
    dry_run: bool,
) -> dict[str, Any]:
    """Launch one main_tercile.py job and parse summary."""
    run_name = f"p5_1_{cfg.tag}_seed{seed}"
    cmd = [
        python_exe, "main_tercile.py",
        *_base_flags(
            cfg=cfg, base_preset=base_preset,
            wb_project=wb_project, wb_entity=wb_entity,
        ),
        "--seed", str(seed),
        "--wandb_run_name", run_name,
    ]
    print(
        f"\n{'=' * 74}\n"
        f"[P5.1] cfg={cfg.tag}  align={cfg.enable_align} "
        f"lambda_align={cfg.lambda_align}  "
        f"r_u={cfg.user_loss_ratio} r_i={cfg.item_loss_ratio}  "
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
            "tag": cfg.tag,
            "enable_align": cfg.enable_align,
            "lambda_align": cfg.lambda_align,
            "user_loss_ratio": cfg.user_loss_ratio,
            "item_loss_ratio": cfg.item_loss_ratio,
            "seed": seed,
            "exit": 0,
            "wall_min": 0.0,
            "dry_run": True,
            "test_head": float("nan"),
            "test_mid": float("nan"),
            "test_tail": float("nan"),
            "best_test_recall20": float("nan"),
            "best_test_ndcg20": float("nan"),
            "best_val_recall20": float("nan"),
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
        "tag": cfg.tag,
        "enable_align": cfg.enable_align,
        "lambda_align": cfg.lambda_align,
        "user_loss_ratio": cfg.user_loss_ratio,
        "item_loss_ratio": cfg.item_loss_ratio,
        "seed": seed,
        "exit": rc,
        "wall_min": wall,
        "dry_run": False,
        "test_head": _f(mt.group("h") if mt else None),
        "test_mid": _f(mt.group("m") if mt else None),
        "test_tail": _f(mt.group("t") if mt else None),
        "best_test_recall20": _f(b.group(1) if b else None),
        "best_test_ndcg20": _f(n.group(1) if n else None),
        "best_val_recall20": _f(v.group(1) if v else None),
    }
    print(
        f"[P5.1] {cfg.tag} seed={seed}  wall={wall:.1f}m  "
        f"R@20={row['best_test_recall20']:.5f}  "
        f"N@20={row['best_test_ndcg20']:.5f}  "
        f"H/M/T="
        f"{row['test_head']:.5f}/{row['test_mid']:.5f}/{row['test_tail']:.5f}",
        flush=True,
    )
    return row


def _rank_configs(
    per_seed: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate per config and sort by mean test Recall@20 (desc)."""
    by_tag: dict[str, list[dict[str, Any]]] = {}
    for row in per_seed:
        by_tag.setdefault(str(row["tag"]), []).append(row)

    ranked: list[dict[str, Any]] = []
    for tag, rows in by_tag.items():
        ranked.append(
            {
                "tag": tag,
                "enable_align": rows[0]["enable_align"],
                "lambda_align": rows[0]["lambda_align"],
                "user_loss_ratio": rows[0]["user_loss_ratio"],
                "item_loss_ratio": rows[0]["item_loss_ratio"],
                "best_test_recall20": _agg(
                    [float(r["best_test_recall20"]) for r in rows]
                ),
                "best_test_ndcg20": _agg(
                    [float(r["best_test_ndcg20"]) for r in rows]
                ),
                "best_val_recall20": _agg(
                    [float(r["best_val_recall20"]) for r in rows]
                ),
                "test_head": _agg([float(r["test_head"]) for r in rows]),
                "test_mid": _agg([float(r["test_mid"]) for r in rows]),
                "test_tail": _agg([float(r["test_tail"]) for r in rows]),
                "n_ok": sum(1 for r in rows if int(r["exit"]) == 0),
            }
        )
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
        description="P5.1 L_align + CL-reweight sweep (2x2 factorial)."
    )
    p.add_argument("--dry_run", type=int, default=0,
                   help="1 = print commands only.")
    p.add_argument(
        "--base_from_p5_0", type=str, default=DEFAULT_BASE,
        choices=sorted(BASE_PRESETS.keys()),
        help=(
            "Which P5.0 preset to layer P5.1 on top of. Default "
            f"'{DEFAULT_BASE}' assumes the P5.0 winner was APC+tau-learn; "
            "use 'p5_baseline_p4' to keep the strict P4 trunk."
        ),
    )
    p.add_argument(
        "--wandb_project", type=str,
        default=os.environ.get("WANDB_PROJECT", "damps-mmhcl-clothing"),
    )
    p.add_argument(
        "--wandb_entity", type=str,
        default=os.environ.get("WANDB_ENTITY", "baitapck51cc-uet"),
    )
    p.add_argument(
        "--out_json", type=str, default=OUT_JSON,
        help="Results path relative to MMHCL_DAMPS_Project/.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the 4-config x 2-seed x 60-epoch P5.1 L_align + reweight probe."""
    args = parse_cli(argv)
    damps_dir, _root, python_exe = _resolve_paths()
    if not (damps_dir / "main_tercile.py").is_file():
        raise FileNotFoundError(
            f"main_tercile.py missing under {damps_dir}."
        )
    os.chdir(damps_dir)
    Path("results").mkdir(exist_ok=True)

    base_preset = BASE_PRESETS[args.base_from_p5_0]
    configs = build_configs()
    dry = bool(args.dry_run)
    print(
        f"[P5.1] {len(configs)} configs x {len(SEEDS)} seeds x "
        f"{EPOCH} epochs  dry_run={int(dry)}",
        flush=True,
    )
    print(
        f"[P5.1] base from P5.0 = '{args.base_from_p5_0}'  "
        f"(damps_apc={base_preset['damps_apc']}, "
        f"learnable_tau={base_preset['learnable_tau']})",
        flush=True,
    )
    for c in configs:
        print(
            f"   - {c.tag}: align={c.enable_align} "
            f"lambda_align={c.lambda_align}  "
            f"r_u={c.user_loss_ratio} r_i={c.item_loss_ratio}",
            flush=True,
        )

    per_seed: list[dict[str, Any]] = []
    total = len(configs) * len(SEEDS)
    done = 0
    for cfg in configs:
        for seed in SEEDS:
            done += 1
            print(f"\n[P5.1] progress {done}/{total}", flush=True)
            per_seed.append(
                _run_one(
                    python_exe=python_exe,
                    damps_dir=damps_dir,
                    cfg=cfg, base_preset=base_preset, seed=seed,
                    wb_project=args.wandb_project,
                    wb_entity=args.wandb_entity,
                    dry_run=dry,
                )
            )

    ranked = _rank_configs(per_seed)
    payload: dict[str, Any] = {
        "meta": {
            "phase": "P5.1",
            "dataset": DATASET,
            "epoch": EPOCH,
            "patience": PATIENCE,
            "base_from_p5_0": args.base_from_p5_0,
            "base_preset": base_preset,
            "constants": {
                "lambda_align_on": LAMBDA_ALIGN_ON,
                "align_temperature": ALIGN_TAU,
                "r_u_base": R_U_BASE, "r_i_base": R_I_BASE,
                "r_u_high": R_U_HIGH, "r_i_high": R_I_HIGH,
                "temperature_init": TEMPERATURE,
                "lambda_view": LAMBDA_VIEW,
                "nrdmc_lite_layers": NRDMC_LITE_LAYERS,
            },
            "description": (
                "P5.1: 2x2 factorial over (enable_align, CL-reweight) on "
                "top of a P5.0-derived trunk. Ships NRDMC IPM 2026 "
                "Eq. 21 (L_align) as a first-class loss term."
            ),
        },
        "seeds": list(SEEDS),
        "epoch": EPOCH,
        "configs": [asdict(c) for c in configs],
        "per_seed": per_seed,
        "ranked": ranked,
        "best": ranked[0] if ranked else None,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 74, flush=True)
    print("P5.1 FINAL -- L_align + CL-reweight grid  (Amazon Clothing)",
          flush=True)
    print("=" * 74, flush=True)
    for i, row in enumerate(ranked, 1):
        r20 = row["best_test_recall20"]
        n20 = row["best_test_ndcg20"]
        h = row["test_head"]; m = row["test_mid"]; t = row["test_tail"]
        print(
            f"  #{i} {row['tag']:22s}: "
            f"align={row['enable_align']} lam={row['lambda_align']}  "
            f"r_u={row['user_loss_ratio']} r_i={row['item_loss_ratio']}  "
            f"R@20={r20['mean']:.5f}+/-{r20['std']:.5f}  "
            f"N@20={n20['mean']:.5f}+/-{n20['std']:.5f}  "
            f"H/M/T={h['mean']:.4f}/{m['mean']:.4f}/{t['mean']:.4f}  "
            f"(n_ok={int(row['n_ok'])})",
            flush=True,
        )

    ctrl = next(
        (r for r in ranked if r["tag"] == "p5_control"), None
    )
    if ctrl and not math.isnan(ctrl["best_test_recall20"]["mean"]):
        print("-" * 74, flush=True)
        ref = ctrl["best_test_recall20"]["mean"]
        print(f"  Effect size vs p5_control (R@20 = {ref:.5f}):",
              flush=True)
        for r in ranked:
            if r["tag"] == "p5_control":
                continue
            delta = r["best_test_recall20"]["mean"] - ref
            pct = 100.0 * delta / ref if ref > 0 else float("nan")
            print(
                f"    {r['tag']:22s}  dR@20 = {delta:+.5f}  ({pct:+.2f}%)",
                flush=True,
            )
    print(f"  Wrote {out_path.as_posix()}", flush=True)
    print("=" * 74, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
