"""scripts/run_p5_0_apc_tau_grid.py -- P5.0 flag-only sweep.

Implements Priority 5.0 (P5.0) of the post-P4 diagnostic roadmap
(PACER_NRDMC_lite_upgrade_analysis_EN Section 7). P5.0 does NOT touch
any source file -- it exercises two flags that are already wired into
``model.py`` / ``train.py`` but were pinned to zero throughout P1-P4.

Motivation (evidence-based)
---------------------------
The P4 ASC-gate sweep produced a within-noise grid: all 6 cells
clustered at R@20 = 0.0812 +/- 0.0006, refuting the P3-D1 hypothesis
that alpha-drift caused the ceiling. The training log revealed the
real bottleneck:

    (a) ``cl`` loss frozen at ~102 (near ln(N)) for all 60 epochs
        while ``mf`` dropped 99 -> 3. CL/view gradient is functionally
        zero -- the multi-view contrastive branch is dormant.
    (b) Head/Mid/Tail = 0.130 / 0.014 / 0.005 -- popularity collapse
        (H:T = 26x). ``damps_apc`` was OFF for every run.
    (c) val_R@20 peaks at 0.078-0.080 around epoch 40-50 and early-
        stops. The system plateaus, not overfits.

P5.0 attacks (b) and part of (a) via zero-code-change flag flips:

    * ``--damps_apc 1``     : enable Adaptive Popularity Correction
                              inside the DAMPS spectral pipeline
                              (already implemented in damps/core.py
                              _apply_apc). Expected: lift Mid/Tail.
    * ``--learnable_tau 1`` : let tau become a learnable nn.Parameter
                              (initialised at 0.30, clamped >= 0.01).
                              Expected: unlock CL gradient by letting
                              the softmax sharpness adapt over epochs.

Grid design
-----------
Four cells x 2 seeds x 60 epochs. Every cell keeps the P4 winner base
(baseline_raw + IMCF + soft-routing + momentum + data-driven prior +
LogQ + NRDMC-lite/K=2). Only APC and learnable_tau vary.

    Cell 0  p5_baseline_p4  -- P4 baseline_raw replica     (APC=0, tau_static)
    Cell 1  p5a_apc         -- APC ON                       (APC=1, tau_static)
    Cell 2  p5a_tau         -- learnable tau                (APC=0, tau_learn )
    Cell 3  p5a_apc_tau     -- APC + learnable tau (joint) (APC=1, tau_learn )

Total budget: 4 cells x 2 seeds x ~16.5 s/epoch x 60 epochs ~= 2.2 h A100.
Roughly 2/3 of the P4 budget with a much tighter causal claim: the two
knobs are semantically orthogonal (one attacks popularity, one attacks
softmax sharpness).

Usage (from MMHCL_DAMPS_Project/)::

    python scripts/run_p5_0_apc_tau_grid.py
    python scripts/run_p5_0_apc_tau_grid.py --dry_run 1
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
# Locked PACER + Branch A' trunk (mirrors P4 baseline_raw exactly).
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

# First two MMHCL-paired seeds (matches P1-P4 ordering).
SEEDS: tuple[int, ...] = (23946202, 1557638902)

OUT_JSON = "results/p5_0_apc_tau_grid_clothing.json"

_TER_TEST_RX = re.compile(
    r"\[tercile-test-final\]\s+BEST_Test_Recall@20_Head=(?P<h>[-\d.eE+nan]+)"
    r"\s+BEST_Test_Recall@20_Mid=(?P<m>[-\d.eE+nan]+)"
    r"\s+BEST_Test_Recall@20_Tail=(?P<t>[-\d.eE+nan]+)"
)
_BEST_RX = re.compile(r"BEST_Test_Recall@20\s*[:=]\s*([-\d.eE+nan]+)")
_BEST_NDCG_RX = re.compile(r"BEST_Test_NDCG@20\s*[:=]\s*([-\d.eE+nan]+)")
_VAL_R20_RX = re.compile(r"BEST_Val_Recall@20\s*[:=]\s*([-\d.eE+nan]+)")
_ALPHA_LAST_RX = re.compile(
    r"\[diag epoch\s+\d+\]\s+.*?alpha_img=(?P<ai>[-\d.eE+nan]+)"
    r"\s+alpha_txt=(?P<at>[-\d.eE+nan]+)"
)
_TAU_LAST_RX = re.compile(
    r"\"tau\"\s*:\s*(?P<tau>[-\d.eE+nan]+)"
)


@dataclass(frozen=True)
class GridConfig:
    """One cell of the P5.0 APC + learnable-tau sweep."""

    tag: str
    damps_apc: int          # {0, 1}
    learnable_tau: int      # {0, 1}


def build_configs() -> list[GridConfig]:
    """Return the 4 P5.0 grid cells (2 x 2 factorial)."""
    return [
        GridConfig("p5_baseline_p4", 0, 0),
        GridConfig("p5a_apc",        1, 0),
        GridConfig("p5a_tau",        0, 1),
        GridConfig("p5a_apc_tau",    1, 1),
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
    wb_project: str,
    wb_entity: str,
) -> list[str]:
    """Shared CLI flags for one P5.0 cell (seed / run-name appended later)."""
    return [
        "--dataset", DATASET,
        "--gpu_id", "0",
        "--epoch", str(EPOCH),
        "--verbose", "5",
        "--eval_every", "5",
        "--eval_last_epochs", "20",
        "--use_gpu_eval", "1",
        "--use_torch_compile", "1",
        # DAMPS complex FFT is unsafe under Inductor reduce-overhead
        # CUDAGraphs; mode=default is multi-step-probe-validated
        # (rev57 P4 fix, commit a0b0971).
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
        "--user_loss_ratio", "0.03",       # locked at P1-P4 default
        "--item_loss_ratio", "0.07",       # locked at P1-P4 default
        "--temperature", str(TEMPERATURE),  # tau init (fixed OR learnable)
        # ---- P5.0 knobs ------------------------------------------------
        "--damps_apc", str(int(cfg.damps_apc)),
        "--learnable_tau", str(int(cfg.learnable_tau)),
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
        # ASC gate: baseline_raw won P4; keep it here so P5.0 is a strict
        # ablation of (APC, tau) on top of the same trunk.
        "--asc_gate_mode", "raw",
        "--asc_warmup_epochs", "0",
        "--asc_reg_l2", "0.0",
        "--asc_reg_target", "0.3",
        # ---- WandB ----
        "--use_wandb", "1",
        "--wandb_project", wb_project,
        "--wandb_entity", wb_entity,
        "--wandb_group", "p5_0_apc_tau_grid",
        "--wandb_tags", "p5,p5_0,apc,learnable_tau,nrdmc_lite,branchA_prime",
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
    """Launch one main_tercile.py job and parse summary lines."""
    run_name = f"p5_0_{cfg.tag}_seed{seed}"
    cmd = [
        python_exe, "main_tercile.py",
        *_base_flags(cfg=cfg, wb_project=wb_project, wb_entity=wb_entity),
        "--seed", str(seed),
        "--wandb_run_name", run_name,
    ]
    print(
        f"\n{'=' * 74}\n"
        f"[P5.0] cfg={cfg.tag}  APC={cfg.damps_apc}  "
        f"learnable_tau={cfg.learnable_tau}  seed={seed}\n{'=' * 74}",
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
            "damps_apc": cfg.damps_apc,
            "learnable_tau": cfg.learnable_tau,
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
            "alpha_img_final": float("nan"),
            "alpha_txt_final": float("nan"),
            "tau_final": float("nan"),
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
    last_alpha_img = float("nan")
    last_alpha_txt = float("nan")
    for m in _ALPHA_LAST_RX.finditer(out):
        last_alpha_img = _f(m.group("ai"))
        last_alpha_txt = _f(m.group("at"))
    last_tau = float("nan")
    for m in _TAU_LAST_RX.finditer(out):
        last_tau = _f(m.group("tau"))
    row = {
        "tag": cfg.tag,
        "damps_apc": cfg.damps_apc,
        "learnable_tau": cfg.learnable_tau,
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
        "alpha_img_final": last_alpha_img,
        "alpha_txt_final": last_alpha_txt,
        "tau_final": last_tau,
    }
    print(
        f"[P5.0] {cfg.tag} seed={seed}  wall={wall:.1f}m  "
        f"R@20={row['best_test_recall20']:.5f}  "
        f"N@20={row['best_test_ndcg20']:.5f}  "
        f"H/M/T="
        f"{row['test_head']:.5f}/{row['test_mid']:.5f}/{row['test_tail']:.5f}  "
        f"tau_final={row['tau_final']:.4f}",
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
                "damps_apc": rows[0]["damps_apc"],
                "learnable_tau": rows[0]["learnable_tau"],
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
                "tau_final": _agg([float(r["tau_final"]) for r in rows]),
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
        description="P5.0 APC + learnable_tau sweep (flag-only)."
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
    """Run the 4-config x 2-seed x 60-epoch P5.0 flag-only probe."""
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
        f"[P5.0] {len(configs)} configs x {len(SEEDS)} seeds x "
        f"{EPOCH} epochs  dry_run={int(dry)}",
        flush=True,
    )
    print(
        f"[P5.0] base HP (locked): lambda_view={LAMBDA_VIEW}  "
        f"tau_init={TEMPERATURE}  enable_ptv=0  asc=raw",
        flush=True,
    )
    for c in configs:
        print(
            f"   - {c.tag}: damps_apc={c.damps_apc}  "
            f"learnable_tau={c.learnable_tau}",
            flush=True,
        )

    per_seed: list[dict[str, Any]] = []
    total = len(configs) * len(SEEDS)
    done = 0
    for cfg in configs:
        for seed in SEEDS:
            done += 1
            print(f"\n[P5.0] progress {done}/{total}", flush=True)
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
        "meta": {
            "phase": "P5.0",
            "dataset": DATASET,
            "epoch": EPOCH,
            "patience": PATIENCE,
            "base": {
                "lr": LR, "batch_size": BATCH_SIZE, "regs": REGS,
                "embed_size": EMBED_SIZE, "UI_layers": UI_LAYERS,
                "temperature_init": TEMPERATURE,
                "lambda_view": LAMBDA_VIEW,
                "nrdmc_lite_layers": NRDMC_LITE_LAYERS,
                "logq_scale": LOGQ_SCALE, "logq_beta": LOGQ_BETA,
                "logq_clip": LOGQ_CLIP,
                "user_loss_ratio": 0.03, "item_loss_ratio": 0.07,
                "asc_gate_mode": "raw",
            },
            "description": (
                "P5.0: flag-only 2x2 factorial over (damps_apc, "
                "learnable_tau) on top of the P4 baseline_raw trunk."
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
    print("P5.0 FINAL -- APC + learnable-tau grid  (Amazon Clothing)",
          flush=True)
    print("=" * 74, flush=True)
    for i, row in enumerate(ranked, 1):
        r20 = row["best_test_recall20"]
        n20 = row["best_test_ndcg20"]
        h = row["test_head"]; m = row["test_mid"]; t = row["test_tail"]
        print(
            f"  #{i} {row['tag']:16s}: "
            f"APC={row['damps_apc']} tau_learn={row['learnable_tau']}  "
            f"R@20={r20['mean']:.5f}+/-{r20['std']:.5f}  "
            f"N@20={n20['mean']:.5f}+/-{n20['std']:.5f}  "
            f"H/M/T={h['mean']:.4f}/{m['mean']:.4f}/{t['mean']:.4f}  "
            f"tau_final={row['tau_final']['mean']:.4f}  "
            f"(n_ok={int(row['n_ok'])})",
            flush=True,
        )

    ctrl = next(
        (r for r in ranked if r["tag"] == "p5_baseline_p4"), None
    )
    if ctrl and not math.isnan(ctrl["best_test_recall20"]["mean"]):
        print("-" * 74, flush=True)
        ref = ctrl["best_test_recall20"]["mean"]
        print(f"  Effect size vs p5_baseline_p4 (R@20 = {ref:.5f}):",
              flush=True)
        for r in ranked:
            if r["tag"] == "p5_baseline_p4":
                continue
            delta = r["best_test_recall20"]["mean"] - ref
            pct = 100.0 * delta / ref if ref > 0 else float("nan")
            print(
                f"    {r['tag']:20s}  dR@20 = {delta:+.5f}  ({pct:+.2f}%)",
                flush=True,
            )
    print(f"  Wrote {out_path.as_posix()}", flush=True)
    print("=" * 74, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
