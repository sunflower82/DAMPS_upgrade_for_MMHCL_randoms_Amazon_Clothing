"""scripts/run_p1_p2_nrdmc_grid.py — P1+P2 joint λ_view × τ probe.

Implements the upgrade-doc roadmap (PACER_NRDMC_lite_upgrade_analysis_EN):

* **P1** — lower ``λ_view`` from the aggressive 0.2 baseline to
  ``{0.05, 0.10}`` so BPR is not drowned by the view InfoNCE term.
* **P2** — raise InfoNCE temperature ``τ`` to ``{0.30, 0.40}`` to soften
  contrastive gradients and stabilise Head Recall@20.

Run them **jointly** as a 2×2 grid (4 configs) × 2 MMHCL-paired seeds ×
100 epochs, then rank by mean ``BEST_Test_Recall@20``.

Usage (from ``MMHCL_DAMPS_Project/``)::

    python scripts/run_p1_p2_nrdmc_grid.py
    python scripts/run_p1_p2_nrdmc_grid.py --dry_run 1
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
# Locked PACER + Branch A' trunk (mirrors notebook §9.12 / t0030)
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

# P1 × P2 joint grid (upgrade-doc strategic recommendation).
LAMBDA_GRID: tuple[float, ...] = (0.05, 0.10)
TAU_GRID: tuple[float, ...] = (0.30, 0.40)

# First two MMHCL-paired seeds (matches §9.12 ordering).
SEEDS: tuple[int, ...] = (23946202, 1557638902)

OUT_JSON = "results/p1_p2_lambda_tau_grid_clothing.json"

_TER_TEST_RX = re.compile(
    r"\[tercile-test-final\]\s+BEST_Test_Recall@20_Head=(?P<h>[-\d.eE+nan]+)"
    r"\s+BEST_Test_Recall@20_Mid=(?P<m>[-\d.eE+nan]+)"
    r"\s+BEST_Test_Recall@20_Tail=(?P<t>[-\d.eE+nan]+)"
)
# train.py logs ``BEST_Test_Recall@20: <val>`` (colon); accept ``=`` too.
_BEST_RX = re.compile(r"BEST_Test_Recall@20\s*[:=]\s*([-\d.eE+nan]+)")
_BEST_NDCG_RX = re.compile(r"BEST_Test_NDCG@20\s*[:=]\s*([-\d.eE+nan]+)")
_VAL_R20_RX = re.compile(r"BEST_Val_Recall@20\s*[:=]\s*([-\d.eE+nan]+)")


@dataclass(frozen=True)
class GridConfig:
    """One (λ_view, τ) cell of the P1×P2 joint grid."""

    tag: str
    lambda_view: float
    temperature: float


def build_configs() -> list[GridConfig]:
    """Return the 4 joint P1×P2 configurations."""
    configs: list[GridConfig] = []
    for lam in LAMBDA_GRID:
        for tau in TAU_GRID:
            tag = f"lam{lam:.2f}_tau{tau:.2f}".replace(".", "p")
            configs.append(
                GridConfig(tag=tag, lambda_view=lam, temperature=tau)
            )
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
        # scripts/ → parent is MMHCL_DAMPS_Project
        here = Path(__file__).resolve().parent
        damps = here.parent if here.name == "scripts" else cwd
    root = damps.parent
    rtx = Path(r"c:\ProgramData\anaconda3\envs\rtx5090_dl\python.exe")
    py = str(rtx) if rtx.is_file() else sys.executable
    return damps, root, py


def _base_flags(
    *,
    lambda_view: float,
    temperature: float,
    wb_project: str,
    wb_entity: str,
) -> list[str]:
    """Shared CLI flags for one grid cell (seed / run-name appended later)."""
    return [
        "--dataset",
        DATASET,
        "--gpu_id",
        "0",
        "--epoch",
        str(EPOCH),
        "--verbose",
        "5",
        "--eval_every",
        "5",
        "--eval_last_epochs",
        "20",
        "--use_gpu_eval",
        "1",
        "--use_torch_compile",
        "1",
        "--torch_compile_mode",
        "reduce-overhead",
        "--torch_compile_dynamic",
        "0",
        "--use_cuda_graph",
        "0",
        "--batch_size",
        str(BATCH_SIZE),
        "--lr",
        str(LR),
        "--regs",
        str(REGS),
        "--embed_size",
        str(EMBED_SIZE),
        "--topk",
        str(KNN_TOPK),
        "--core",
        "5",
        "--UI_layers",
        str(UI_LAYERS),
        "--User_layers",
        str(U_LAYERS),
        "--Item_layers",
        str(I_LAYERS),
        "--temperature",
        str(temperature),
        "--damps_apc",
        "0",
        "--damps_avrf",
        "0",
        "--damps_imcf",
        "1",
        "--damps_soft_routing",
        "1",
        "--damps_momentum",
        "1",
        "--damps_data_driven_prior",
        "1",
        "--damps_permutation_fft",
        "0",
        "--damps_warmup_epochs",
        "10",
        "--enable_logq",
        "1",
        "--logq_mode",
        "laplace",
        "--logq_beta",
        str(LOGQ_BETA),
        "--logq_scale",
        str(LOGQ_SCALE),
        "--logq_clip",
        str(LOGQ_CLIP),
        "--enable_simgcl",
        "0",
        "--enable_nrdmc_lite",
        "1",
        "--nrdmc_lite_layers",
        str(NRDMC_LITE_LAYERS),
        "--lambda_view",
        str(lambda_view),
        "--simgcl_eps",
        str(SIMGCL_EPS),
        "--branchA_view_bsz",
        "2048",
        "--branchA_bcl_bsz",
        "2048",
        "--branchA_bcl_batchn",
        "1",
        "--early_stopping_patience",
        str(PATIENCE),
        "--early_stopping_min_epochs",
        "0",
        "--early_stopping_min_delta",
        "1e-4",
        "--early_stopping_monitor",
        "val_recall@20",
        "--early_stopping_mode",
        "max",
        "--early_stopping_restore_best",
        "1",
        "--use_reduce_lr",
        str(USE_REDUCE_LR),
        "--use_amp",
        "1",
        "--use_wandb",
        "1",
        "--wandb_project",
        wb_project,
        "--wandb_entity",
        wb_entity,
        "--wandb_group",
        "p1_p2_nrdmc_grid",
        "--wandb_tags",
        "p1,p2,nrdmc_lite,branchA_prime",
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
    run_name = f"p1p2_{cfg.tag}_seed{seed}"
    cmd = [
        python_exe,
        "main_tercile.py",
        *_base_flags(
            lambda_view=cfg.lambda_view,
            temperature=cfg.temperature,
            wb_project=wb_project,
            wb_entity=wb_entity,
        ),
        "--seed",
        str(seed),
        "--wandb_run_name",
        run_name,
    ]
    print(
        f"\n{'=' * 74}\n"
        f"[P1+P2] cfg={cfg.tag}  lambda_view={cfg.lambda_view}  "
        f"tau={cfg.temperature}  seed={seed}\n"
        f"{'=' * 74}",
        flush=True,
    )
    print(
        "[cmd] "
        + " ".join(cmd[:8])
        + f" ... [{len(cmd) - 8} more flags]",
        flush=True,
    )
    if dry_run:
        return {
            "tag": cfg.tag,
            "lambda_view": cfg.lambda_view,
            "temperature": cfg.temperature,
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
        cmd,
        cwd=str(damps_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
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
        "lambda_view": cfg.lambda_view,
        "temperature": cfg.temperature,
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
        f"[P1+P2] {cfg.tag} seed={seed}  wall={wall:.1f}m  "
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
                "lambda_view": rows[0]["lambda_view"],
                "temperature": rows[0]["temperature"],
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
    """Parse driver CLI."""
    p = argparse.ArgumentParser(
        description="P1+P2 joint λ_view × τ grid for Branch A' NRDMC-lite."
    )
    p.add_argument("--dry_run", type=int, default=0, help="1 = print cmds only.")
    p.add_argument(
        "--wandb_project",
        type=str,
        default=os.environ.get("WANDB_PROJECT", "damps-mmhcl-clothing"),
    )
    p.add_argument(
        "--wandb_entity",
        type=str,
        default=os.environ.get("WANDB_ENTITY", "baitapck51cc-uet"),
    )
    p.add_argument(
        "--out_json",
        type=str,
        default=OUT_JSON,
        help="Results path relative to MMHCL_DAMPS_Project/.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the 4-config × 2-seed × 100-epoch P1+P2 probe."""
    args = parse_cli(argv)
    damps_dir, _root, python_exe = _resolve_paths()
    if not (damps_dir / "main_tercile.py").is_file():
        raise FileNotFoundError(
            f"main_tercile.py missing under {damps_dir}. "
            "Run notebook §9.9/§9.10 first."
        )
    os.chdir(damps_dir)
    Path("results").mkdir(exist_ok=True)

    configs = build_configs()
    dry = bool(args.dry_run)
    print(
        f"[P1+P2] {len(configs)} configs x {len(SEEDS)} seeds x "
        f"{EPOCH} epochs  dry_run={int(dry)}",
        flush=True,
    )
    for c in configs:
        print(
            f"   - {c.tag}: lambda_view={c.lambda_view}  "
            f"tau={c.temperature}",
            flush=True,
        )

    per_seed: list[dict[str, Any]] = []
    total = len(configs) * len(SEEDS)
    done = 0
    for cfg in configs:
        for seed in SEEDS:
            done += 1
            print(f"\n[P1+P2] progress {done}/{total}", flush=True)
            per_seed.append(
                _run_one(
                    python_exe=python_exe,
                    damps_dir=damps_dir,
                    cfg=cfg,
                    seed=seed,
                    wb_project=args.wandb_project,
                    wb_entity=args.wandb_entity,
                    dry_run=dry,
                )
            )

    ranked = _rank_configs(per_seed)
    payload: dict[str, Any] = {
        "variant": "PACER+BranchA_prime_P1P2_grid",
        "roadmap": {
            "P1": "lambda_view in {0.05, 0.10}",
            "P2": "tau in {0.30, 0.40}",
            "joint": "2x2 grid, 2 seeds, 100 epochs",
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
        "P1+P2 FINAL — lambda_view x tau joint grid  (Amazon Clothing)",
        flush=True,
    )
    print("=" * 74, flush=True)
    for i, row in enumerate(ranked, 1):
        r20 = row["best_test_recall20"]
        n20 = row["best_test_ndcg20"]
        print(
            f"  #{i} {row['tag']}: "
            f"lambda={row['lambda_view']}  tau={row['temperature']}  "
            f"R@20={r20['mean']:.5f}+/-{r20['std']:.5f}  "
            f"N@20={n20['mean']:.5f}+/-{n20['std']:.5f}  "
            f"(n_ok={int(row['n_ok'])})",
            flush=True,
        )
    if ranked and not math.isnan(ranked[0]["best_test_recall20"]["mean"]):
        best = ranked[0]
        print("-" * 74, flush=True)
        print(
            f"  BEST -> {best['tag']}  "
            f"(lambda_view={best['lambda_view']}, "
            f"tau={best['temperature']})",
            flush=True,
        )
        print(
            "  Next: lock these HPs, then proceed to P3/P4 per upgrade doc.",
            flush=True,
        )
    print(f"  Wrote {out_path.as_posix()}", flush=True)
    print("=" * 74, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
