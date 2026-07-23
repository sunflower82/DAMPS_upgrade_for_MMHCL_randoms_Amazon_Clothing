"""scripts/run_p3_ptv_grid.py -- P3 Prototype-Aware View (PTV) sweep.

Implements Priority 3 of the revised upgrade roadmap
(PACER_NRDMC_lite_upgrade_analysis_EN, rev56 Section 6):

* **P3** -- Re-instate the third NRDMC view (PTV) that rev55 Section 8.2
  dropped for compactness. NRDMC IPM 2026 Table 4 attributes ~30 percent of the
  Clothing R@20 gain to PTV, so this is the highest-value architectural
  addition once the P1+P2 grid confirmed the plateau is a *capacity*
  ceiling, not a lambda/tau regularisation issue.

Grid design
-----------
Base config = P1+P2 winner (lambda_view=0.10, tau=0.30) held fixed;
the sweep varies PTV knobs only.

    Cell 0 (control, K=2)  : enable_ptv=0                                 [1 cell]
    Cell 1..3 (K=32)       : lambda_ptv in {0.5, 1.0, 2.0}                [3 cells]
    Cell 4               : enable_ptv=1, K=16, lambda_ptv=1.0             [1 cell]

Total: 5 configs x 2 MMHCL-paired seeds x 100 epochs.

Baseline is included so we can compute the PTV effect size directly and
attribute any gain (or loss) unambiguously to the prototype branch.

Usage (from MMHCL_DAMPS_Project/):

    python scripts/run_p3_ptv_grid.py
    python scripts/run_p3_ptv_grid.py --dry_run 1
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
# Locked PACER + Branch A' trunk (mirrors notebook Section 9.12 / t0030)
# with the P1+P2 winner (lambda_view=0.10, tau=0.30).
# ---------------------------------------------------------------------------
DATASET = "Clothing"
EPOCH = 100
PATIENCE = 20
BATCH_SIZE = 1024
LR = 2.50995e-4
REGS = 1.20e-04
EMBED_SIZE = 128
KNN_TOPK = 10
UI_LAYERS = 3
U_LAYERS = 2
I_LAYERS = 2
SIMGCL_EPS = 0.329  # unused (simgcl off)
LOGQ_SCALE = 0.651
LOGQ_BETA = 1.0
LOGQ_CLIP = 5.0
USE_REDUCE_LR = 1
NRDMC_LITE_LAYERS = 2

# P1+P2 winner (locked here as the P3 base).
LAMBDA_VIEW = 0.10
TEMPERATURE = 0.30

# First two MMHCL-paired seeds (matches Section 9.12 ordering).
SEEDS: tuple[int, ...] = (23946202, 1557638902)

OUT_JSON = "results/p3_ptv_grid_clothing.json"

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
    """One cell of the P3 PTV sweep."""

    tag: str
    enable_ptv: int          # 0 = K=2 control, 1 = K=3 with PTV
    n_prototypes: int        # K
    lambda_ptv: float


def build_configs() -> list[GridConfig]:
    """Return the 5 P3 grid cells.

    Cell 0    : control (K=2, PTV off) - reproduces P1+P2 winner exactly.
    Cell 1..3 : K=32, sweep lambda_ptv in {0.5, 1.0, 2.0}
    Cell 4    : K=16, lambda_ptv=1.0 (checks K sensitivity)
    """
    configs: list[GridConfig] = [
        GridConfig(tag="ctrl_k2", enable_ptv=0, n_prototypes=0, lambda_ptv=0.0),
        GridConfig(tag="ptv_k32_l0p5", enable_ptv=1, n_prototypes=32,
                   lambda_ptv=0.5),
        GridConfig(tag="ptv_k32_l1p0", enable_ptv=1, n_prototypes=32,
                   lambda_ptv=1.0),
        GridConfig(tag="ptv_k32_l2p0", enable_ptv=1, n_prototypes=32,
                   lambda_ptv=2.0),
        GridConfig(tag="ptv_k16_l1p0", enable_ptv=1, n_prototypes=16,
                   lambda_ptv=1.0),
    ]
    return configs


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
    wb_project: str,
    wb_entity: str,
) -> list[str]:
    """Shared CLI flags for one grid cell (seed / run-name appended later)."""
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
        "--temperature", str(TEMPERATURE),
        "--damps_apc", "0",
        "--damps_avrf", "0",
        "--damps_imcf", "1",
        "--damps_soft_routing", "1",
        "--damps_momentum", "1",
        "--damps_data_driven_prior", "1",
        "--damps_permutation_fft", "0",
        "--damps_warmup_epochs", "10",
        "--enable_logq", "1",
        "--logq_mode", "laplace",
        "--logq_beta", str(LOGQ_BETA),
        "--logq_scale", str(LOGQ_SCALE),
        "--logq_clip", str(LOGQ_CLIP),
        "--enable_simgcl", "0",
        "--enable_nrdmc_lite", "1",
        "--nrdmc_lite_layers", str(NRDMC_LITE_LAYERS),
        "--lambda_view", str(LAMBDA_VIEW),
        "--enable_ptv", str(int(cfg.enable_ptv)),
        "--n_prototypes", str(int(cfg.n_prototypes)),
        "--lambda_ptv", str(cfg.lambda_ptv),
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
        "--use_wandb", "1",
        "--wandb_project", wb_project,
        "--wandb_entity", wb_entity,
        "--wandb_group", "p3_ptv_grid",
        "--wandb_tags", "p3,ptv,nrdmc_lite,branchA_prime",
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
    """Launch one main_tercile.py job and parse summary metrics."""
    run_name = f"p3_{cfg.tag}_seed{seed}"
    cmd = [
        python_exe, "main_tercile.py",
        *_base_flags(cfg=cfg, wb_project=wb_project, wb_entity=wb_entity),
        "--seed", str(seed),
        "--wandb_run_name", run_name,
    ]
    print(
        f"\n{'=' * 74}\n"
        f"[P3] cfg={cfg.tag}  enable_ptv={cfg.enable_ptv}  "
        f"K={cfg.n_prototypes}  lambda_ptv={cfg.lambda_ptv}  seed={seed}\n"
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
            "tag": cfg.tag,
            "enable_ptv": cfg.enable_ptv,
            "n_prototypes": cfg.n_prototypes,
            "lambda_ptv": cfg.lambda_ptv,
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
        "enable_ptv": cfg.enable_ptv,
        "n_prototypes": cfg.n_prototypes,
        "lambda_ptv": cfg.lambda_ptv,
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
        f"[P3] {cfg.tag} seed={seed}  wall={wall:.1f}m  "
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
                "enable_ptv": rows[0]["enable_ptv"],
                "n_prototypes": rows[0]["n_prototypes"],
                "lambda_ptv": rows[0]["lambda_ptv"],
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
        description="P3 Prototype-Aware View (PTV) sweep for Branch A'."
    )
    p.add_argument("--dry_run", type=int, default=0,
                   help="1 = print commands only.")
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
    """Run the 5-config x 2-seed x 100-epoch P3 PTV probe."""
    args = parse_cli(argv)
    damps_dir, _root, python_exe = _resolve_paths()
    if not (damps_dir / "main_tercile.py").is_file():
        raise FileNotFoundError(
            f"main_tercile.py missing under {damps_dir}."
        )
    os.chdir(damps_dir)
    Path("results").mkdir(exist_ok=True)

    configs = build_configs()
    dry = bool(args.dry_run)
    print(
        f"[P3] {len(configs)} configs x {len(SEEDS)} seeds x "
        f"{EPOCH} epochs  dry_run={int(dry)}",
        flush=True,
    )
    print(
        f"[P3] base HP (locked): lambda_view={LAMBDA_VIEW}  tau={TEMPERATURE}",
        flush=True,
    )
    for c in configs:
        print(
            f"   - {c.tag}: enable_ptv={c.enable_ptv}  K={c.n_prototypes}  "
            f"lambda_ptv={c.lambda_ptv}",
            flush=True,
        )

    per_seed: list[dict[str, Any]] = []
    total = len(configs) * len(SEEDS)
    done = 0
    for cfg in configs:
        for seed in SEEDS:
            done += 1
            print(f"\n[P3] progress {done}/{total}", flush=True)
            per_seed.append(
                _run_one(
                    python_exe=python_exe,
                    damps_dir=damps_dir,
                    cfg=cfg, seed=seed,
                    wb_project=args.wandb_project,
                    wb_entity=args.wandb_entity,
                    dry_run=dry,
                )
            )

    ranked = _rank_configs(per_seed)
    payload: dict[str, Any] = {
        "variant": "PACER+BranchA_prime_P3_ptv_grid",
        "roadmap": {
            "P3": "Re-instate PTV (K=3 fusion, NRDMC Eq. 20-22).",
            "base": f"lambda_view={LAMBDA_VIEW}, tau={TEMPERATURE} (P1+P2 winner).",
            "grid": "5 cells: 1 K=2 control + 3 lambda_ptv sweep at K=32 + 1 K=16 check.",
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
    print(
        "P3 FINAL -- PTV grid  (Amazon Clothing)",
        flush=True,
    )
    print("=" * 74, flush=True)
    for i, row in enumerate(ranked, 1):
        r20 = row["best_test_recall20"]
        n20 = row["best_test_ndcg20"]
        print(
            f"  #{i} {row['tag']}: "
            f"K={row['n_prototypes']}  lambda_ptv={row['lambda_ptv']}  "
            f"R@20={r20['mean']:.5f}+/-{r20['std']:.5f}  "
            f"N@20={n20['mean']:.5f}+/-{n20['std']:.5f}  "
            f"(n_ok={int(row['n_ok'])})",
            flush=True,
        )

    # Report effect size vs the control cell (ctrl_k2).
    ctrl = next((r for r in ranked if r["tag"] == "ctrl_k2"), None)
    if ctrl and not math.isnan(ctrl["best_test_recall20"]["mean"]):
        print("-" * 74, flush=True)
        print(
            f"  Effect size vs control (R@20 = "
            f"{ctrl['best_test_recall20']['mean']:.5f}):",
            flush=True,
        )
        for r in ranked:
            if r["tag"] == "ctrl_k2":
                continue
            delta = (
                r["best_test_recall20"]["mean"]
                - ctrl["best_test_recall20"]["mean"]
            )
            pct = (
                100.0 * delta / ctrl["best_test_recall20"]["mean"]
                if ctrl["best_test_recall20"]["mean"] > 0 else float("nan")
            )
            print(
                f"    {r['tag']:20s}  dR@20 = {delta:+.5f}  ({pct:+.2f}%)",
                flush=True,
            )
    print(f"  Wrote {out_path.as_posix()}", flush=True)
    print("=" * 74, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
