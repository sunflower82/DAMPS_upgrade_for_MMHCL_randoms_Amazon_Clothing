"""scripts/run_kse_final_5seed.py -- Final KSE 2026 benchmark: 1 full + 3 ablations x 5 seeds.
=============================================================================

Locked configuration (from §9.29 K-block + §9.30 P8.0 capacity ridge)
--------------------------------------------------------------------
PACER-NRDMC-lite (full):
    embed_size=320, UI_layers=3, weight_size=[64,64,64], batch_size=1024,
    lr=0.000250995, regs=4.8e-4, alpha_interest=0.50, logq_scale=0.651,
    enable_tamer=1, enable_logq=1, enable_nrdmc_lite=1,
    tamer_interest_cache=results/interest_tree_clothing_rsfp_a010.npz.

Ablation grid (each drops exactly one novel component)
------------------------------------------------------
    A0  kse_full            (baseline for comparison; identical to full config)
    A1  kse_wo_rsfp         --tamer_interest_cache=results/interest_tree_clothing.npz
                             (co-occurrence cache only; no RSFPGrowth blending;
                              proves RSFP's contribution to Mid/Tail Recall@20)
    A2  kse_wo_logq         --enable_logq=0 --logq_scale=0.0
                             (proves LogQ's contribution to Head Recall@20)
    A3  kse_wo_nrdmc        --enable_nrdmc_lite=0 --nrdmc_lite_layers=0
                             (proves NRDMC-lite's contribution to overall
                              R@20/NDCG@20)

Protocol
--------
* seeds:    [1616406634, 1640104851, 52093548, 109649638, 372270914]
* epoch:    250
* early_stopping_monitor=val_recall@20, patience=5 (evaluations),
            min_epochs=30, eval_every=5.
* reduce_lr:  factor=0.5, patience=3.
* Total = 4 configs x 5 seeds = 20 runs.

Reported per run
----------------
For each seed x config we parse from stdout at the val_recall@20 peak:
    (1) test Recall@20         BEST_Test_Recall@20
    (2) test NDCG@20           BEST_Test_NDCG@20
    (3) test Precision@20      BEST_Test_Precision@20
    (4) test Recall@20 Head    BEST_Test_Recall@20_Head
    (5) test Recall@20 Mid     BEST_Test_Recall@20_Mid
    (6) test Recall@20 Tail    BEST_Test_Recall@20_Tail

Per-config we then aggregate seed-wise mean + std (unbiased, ddof=1) into
``ranked`` inside the output JSON.  A companion Markdown table is written
to ``<output>.md`` for direct copy-paste into the paper.

Usage
-----
::

    python scripts/run_kse_final_5seed.py \\
        --seeds 1616406634 1640104851 52093548 109649638 372270914 \\
        --epoch 250 \\
        --output ./results/kse_final_5seed_clothing.json \\
        --dry_run 0

Estimated wall time
-------------------
20 runs x embed=320 with patience=5 (typical stop 60-90 epochs) x ~24s/ep
= ~28 min/run x 20 = ~9 - 10 h.  Fits overnight comfortably.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Locked baseline configuration -- shared by all four variants
# ---------------------------------------------------------------------------
BASELINE_CLI = {
    "dataset": "Clothing",
    "core": 5,
    "seed": None,
    "epoch": 250,
    "batch_size": 1024,
    "lr": 0.000250995,
    "clip_grad_norm": 1.0,
    "embed_size": 320,               # p8_e320 (KSE full config)
    "weight_size": "[64,64,64]",
    "topk": 10,
    "cf_model": "LightGCN",
    "norm_type": "sym",
    "UI_layers": 3,
    "User_layers": 2,
    "Item_layers": 2,
    "user_loss_ratio": 0.03,
    "item_loss_ratio": 0.07,
    "temperature": 0.3,
    "learnable_tau": 0,
    "Ks": "[10,20]",
    "test_flag": "part",
    "use_gpu_eval": 1,
    "eval_every": 5,
    "eval_last_epochs": 60,
    "early_stopping_patience": 5,    # 5 evaluations = 25 epochs (eval_every=5)
    "early_stopping_min_epochs": 30, # warmup NRDMC + MACP
    "early_stopping_min_delta": 1e-4,
    "early_stopping_mode": "max",
    "early_stopping_restore_best": 1,
    "early_stopping_monitor": "val_recall@20",
    "use_reduce_lr": 1,
    "reduce_lr_factor": 0.5,
    "reduce_lr_patience": 3,
    "reduce_lr_min": 1e-6,
    "regs": 4.8e-4,
    "damps_apc": 0,
    "damps_avrf": 0,
    "damps_imcf": 1,
    "damps_permutation_fft": 0,
    "damps_soft_routing": 1,
    "damps_momentum": 1,
    "damps_data_driven_prior": 1,
    "damps_num_categories": 10,
    "damps_warmup_epochs": 10,
    "enable_logq": 1,
    "logq_mode": "laplace",
    "logq_beta": 1.0,
    "logq_clip": 5.0,
    "logq_scale": 0.651,
    "enable_simgcl": 0,
    "simgcl_eps": 0.329,
    "lambda_view": 0.1,
    "simgcl_batch_size_user": 4096,
    "simgcl_batch_size_item": 4096,
    "branchA_view_every_k": 2,
    "branchA_bcl_batchn": 1,
    "branchA_view_bsz": 2048,
    "branchA_bcl_bsz": 2048,
    "enable_nrdmc_lite": 1,
    "nrdmc_lite_layers": 2,
    "enable_ptv": 0,
    "n_prototypes": 0,
    "lambda_ptv": 0.0,
    "enable_align": 0,
    "lambda_align": 0.0,
    "align_temperature": 0.2,
    "use_macp": 1,
    "macp_mode": "replace_pca",
    "macp_alpha_p": 0.0,
    "macp_alpha_z": 0.0,
    "macp_image_mode": "replace_pca",
    "macp_image_alpha_p": 0.0,
    "macp_image_alpha_z": 0.0,
    "macp_verbose": 1,
    "enable_tamer": 1,
    "alpha_interest": 0.50,
    "pop_inverse_eta": 0.0,
    "rebuild_R": 5,
    "faiss_threshold": 60000,
    "knn_chunk_size": 4096,
    "faiss_use_gpu": 1,
    "knn_efsearch": 64,
    "use_amp": 1,
    "use_torch_compile": 1,
    "torch_compile_mode": "default",
    "torch_compile_dynamic": 0,
    "use_gpu_sample": 1,
    "use_cuda_graph": 0,
    "asc_gate_mode": "raw",
    "asc_warmup_epochs": 0,
    "asc_reg_l2": 0.0,
    "asc_reg_target": 0.3,
    "ablation_target": "",
    "use_wandb": 1,
    "wandb_project": "damps-mmhcl-clothing",
    "wandb_entity": "baitapck51cc-uet",
    "wandb_group": "kse_final_5seed",
}


DEFAULT_SEEDS = [1616406634, 1640104851, 52093548, 109649638, 372270914]


# ---------------------------------------------------------------------------
# 4-variant grid: 1 full + 3 ablations
# ---------------------------------------------------------------------------
def build_grid(rsfp_cache: str, base_cache: str) -> list[dict]:
    return [
        {
            "tag": "A0_kse_full",
            "block": "KSE_full",
            "label": "PACER-NRDMC-lite (full)",
            "overrides": {
                "tamer_interest_cache": rsfp_cache,
                "enable_tamer": 1,
                "enable_logq": 1,
                "logq_scale": 0.651,
                "enable_nrdmc_lite": 1,
                "nrdmc_lite_layers": 2,
            },
        },
        {
            "tag": "A1_kse_wo_rsfp",
            "block": "KSE_ablation",
            "label": "w/o RSFPGrowth (base co-occurrence cache only)",
            "overrides": {
                "tamer_interest_cache": base_cache,   # no RSFP blending
                "enable_tamer": 1,
                "enable_logq": 1,
                "logq_scale": 0.651,
                "enable_nrdmc_lite": 1,
                "nrdmc_lite_layers": 2,
            },
        },
        {
            "tag": "A2_kse_wo_logq",
            "block": "KSE_ablation",
            "label": "w/o LogQ (no popularity de-bias)",
            "overrides": {
                "tamer_interest_cache": rsfp_cache,
                "enable_tamer": 1,
                "enable_logq": 0,
                "logq_scale": 0.0,
                "enable_nrdmc_lite": 1,
                "nrdmc_lite_layers": 2,
            },
        },
        {
            "tag": "A3_kse_wo_nrdmc",
            "block": "KSE_ablation",
            "label": "w/o NRDMC-lite (no residual denoising)",
            "overrides": {
                "tamer_interest_cache": rsfp_cache,
                "enable_tamer": 1,
                "enable_logq": 1,
                "logq_scale": 0.651,
                "enable_nrdmc_lite": 0,
                "nrdmc_lite_layers": 0,
            },
        },
    ]


# ---------------------------------------------------------------------------
# CLI construction
# ---------------------------------------------------------------------------
def build_cli(python_exe: str, main_py: Path, variant: dict,
              seed: int, epoch: int) -> list[str]:
    cfg = dict(BASELINE_CLI)
    cfg["seed"] = seed
    cfg["epoch"] = epoch
    cfg.update(variant["overrides"])
    cfg["wandb_run_name"] = f"{variant['tag']}_seed{seed}"
    cfg["wandb_tags"] = ",".join([
        "kse", "kse_final_5seed", variant["block"], variant["tag"],
        f"embed{cfg['embed_size']}",
        f"logq{cfg['logq_scale']}",
        f"nrdmc{cfg['enable_nrdmc_lite']}",
        Path(cfg["tamer_interest_cache"]).stem,
        "r20_monitor", "patience5_min30",
    ])
    cmd = [python_exe, str(main_py)]
    for k, v in cfg.items():
        if isinstance(v, bool):
            v = int(v)
        cmd += [f"--{k}", str(v)]
    return cmd


# ---------------------------------------------------------------------------
# Stdout parser -- extract the 6 paper metrics from BEST_Test_* lines
# ---------------------------------------------------------------------------
def _parse_run_output(out: str) -> dict:
    """Parse the 6 KSE paper metrics + best_epoch from stdout."""
    result: dict = {}
    scalar_patterns = [
        ("best_test_recall20",    r"BEST_Test_Recall@20:\s*([\d.eE+\-]+)"),
        ("best_test_ndcg20",      r"BEST_Test_NDCG@20:\s*([\d.eE+\-]+)"),
        ("best_test_precision20", r"BEST_Test_Precision@20:\s*([\d.eE+\-]+)"),
        ("best_val_recall20",     r"BEST_Val_Recall@20:\s*([\d.eE+\-]+)"),
        ("best_val_ndcg20",       r"BEST_Val_NDCG@20:\s*([\d.eE+\-]+)"),
        ("best_epoch",            r"BEST_Val_Recall_Peak_Epoch:\s*(\d+)"),
    ]
    for key, pat in scalar_patterns:
        m = re.search(pat, out)
        if m:
            v = m.group(1)
            result[key] = int(v) if key == "best_epoch" else float(v)

    tercile_patterns = [
        ("best_test_head_recall20", r"BEST_Test_Recall@20_Head=([\d.eE+\-]+)"),
        ("best_test_mid_recall20",  r"BEST_Test_Recall@20_Mid=([\d.eE+\-]+)"),
        ("best_test_tail_recall20", r"BEST_Test_Recall@20_Tail=([\d.eE+\-]+)"),
        # val-side tercile (fallback if test not printed)
        ("best_val_head_recall20",  r"BEST_Recall@20_Head=([\d.eE+\-]+)"),
        ("best_val_mid_recall20",   r"BEST_Recall@20_Mid=([\d.eE+\-]+)"),
        ("best_val_tail_recall20",  r"BEST_Recall@20_Tail=([\d.eE+\-]+)"),
    ]
    for key, pat in tercile_patterns:
        m = re.search(pat, out)
        if m:
            result[key] = float(m.group(1))
    return result


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def _resolve_main(main_arg: Path) -> Path:
    if main_arg.is_absolute() and main_arg.is_file():
        return main_arg
    cwd = Path.cwd()
    if (cwd / main_arg).is_file():
        return (cwd / main_arg).resolve()
    if (cwd / "main_tercile.py").is_file():
        return (cwd / "main_tercile.py").resolve()
    if (cwd / "MMHCL_DAMPS_Project" / "main_tercile.py").is_file():
        return (cwd / "MMHCL_DAMPS_Project" / "main_tercile.py").resolve()
    here = Path(__file__).resolve()
    if here.parent.name == "scripts" and (here.parent.parent / "main_tercile.py").is_file():
        return (here.parent.parent / "main_tercile.py").resolve()
    raise FileNotFoundError(f"Could not locate main_tercile.py. Tried: {main_arg}")


def _run_one(cmd: list[str], log_path: Path, dry_run: bool) -> tuple[int, str, float]:
    if dry_run:
        print("[dry_run] " + " ".join(shlex.quote(c) for c in cmd))
        return 0, "", 0.0
    t0 = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[kse] running (tail): "
          f"{' '.join(shlex.quote(c) for c in cmd[-30:])}")
    print(f"[kse] log: {log_path}")
    with log_path.open("wb") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
        exit_code = proc.wait()
    wall = time.time() - t0
    out = log_path.read_text(encoding="utf-8", errors="replace")
    print(f"[kse] exit={exit_code}  wall={wall/60.0:.1f} min")
    return exit_code, out, wall


# ---------------------------------------------------------------------------
# Aggregation helpers -- seed-wise mean +/- std for the paper table
# ---------------------------------------------------------------------------
_METRIC_KEYS = [
    ("best_test_recall20",     "R@20"),
    ("best_test_ndcg20",       "NDCG@20"),
    ("best_test_precision20",  "P@20"),
    ("best_test_head_recall20", "R@20_Head"),
    ("best_test_mid_recall20",  "R@20_Mid"),
    ("best_test_tail_recall20", "R@20_Tail"),
]


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    m = sum(values) / len(values)
    if len(values) < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)  # ddof=1
    return m, math.sqrt(var)


def _aggregate(rows: list[dict], grid: list[dict]) -> list[dict]:
    """One record per variant: seed-wise mean & std for each metric."""
    per_tag: dict = {}
    for r in rows:
        per_tag.setdefault(r["tag"], []).append(r)

    tag_to_variant = {v["tag"]: v for v in grid}
    aggregated = []
    for tag, rs in per_tag.items():
        v = tag_to_variant.get(tag, {})
        rec = {
            "tag": tag,
            "block": rs[0].get("block"),
            "label": v.get("label", tag),
            "n_seeds": len(rs),
        }
        for key, _ in _METRIC_KEYS:
            vals = [r[key] for r in rs if key in r]
            m, s = _mean_std(vals)
            rec[f"{key}_mean"] = m
            rec[f"{key}_std"]  = s
            rec[f"{key}_n"]    = len(vals)
        aggregated.append(rec)

    # Rank by mean test R@20 desc.
    aggregated.sort(key=lambda r: (r.get("best_test_recall20_mean") or 0),
                    reverse=True)
    return aggregated


def _write_markdown(agg: list[dict], out_path: Path) -> None:
    header = ("| Variant | " +
              " | ".join(label for _, label in _METRIC_KEYS) +
              " |")
    sep = "|" + "|".join(["---"] * (1 + len(_METRIC_KEYS))) + "|"
    lines = [header, sep]
    for r in agg:
        cells = [f"{r['label']}"]
        for key, _ in _METRIC_KEYS:
            m = r.get(f"{key}_mean")
            s = r.get(f"{key}_std")
            if m is None or (isinstance(m, float) and math.isnan(m)):
                cells.append("—")
            else:
                cells.append(f"{m*100:.3f} ± {s*100:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--epoch", type=int, default=250)
    ap.add_argument("--python", type=str, default=sys.executable)
    ap.add_argument("--main", type=Path, default=Path("main_tercile.py"))
    ap.add_argument("--output", type=Path,
                    default=Path("./results/kse_final_5seed_clothing.json"))
    ap.add_argument("--log_dir", type=Path,
                    default=Path("./results/_kse_final_5seed_logs"))
    ap.add_argument("--rsfp_cache", type=Path,
                    default=Path("results/interest_tree_clothing_rsfp_a010.npz"))
    ap.add_argument("--base_cache", type=Path,
                    default=Path("results/interest_tree_clothing.npz"))
    ap.add_argument("--dry_run", type=int, default=0)
    ap.add_argument("--only_tags", type=str, nargs="*", default=None)
    args = ap.parse_args()

    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    main_py = _resolve_main(args.main)
    print(f"[kse] main_py: {main_py}")
    print(f"[kse] seeds:   {args.seeds}")

    grid = build_grid(str(args.rsfp_cache), str(args.base_cache))
    if args.only_tags:
        grid = [v for v in grid if v["tag"] in set(args.only_tags)]

    # Sanity: verify prerequisite caches exist (skip in dry_run).
    if not args.dry_run:
        missing = []
        for v in grid:
            p = Path(v["overrides"]["tamer_interest_cache"])
            if not p.is_file():
                missing.append(str(p))
        if missing:
            print("[kse] WARNING -- missing interest caches:")
            for m in missing:
                print(f"          {m}")
            print("[kse] Build them via scripts/build_rsfp_interest_tree.py first.")

    total_runs = len(grid) * len(args.seeds)
    print(f"[kse] total runs: {total_runs} "
          f"({len(grid)} variants x {len(args.seeds)} seeds x "
          f"epoch=<= {args.epoch}, patience=5 evals, min_epochs=30)")

    rows: list[dict] = []
    run_idx = 0

    for variant in grid:
        for seed in args.seeds:
            run_idx += 1
            print(f"\n{'='*72}\n"
                  f"[kse] {run_idx}/{total_runs}  "
                  f"tag={variant['tag']}  seed={seed}\n{'='*72}")
            cmd = build_cli(args.python, main_py, variant, seed, args.epoch)
            log_path = args.log_dir / f"{variant['tag']}_seed{seed}.log"
            exit_code, out, wall = _run_one(cmd, log_path, bool(args.dry_run))
            parsed = _parse_run_output(out) if not args.dry_run else {}
            row = {
                "tag": variant["tag"],
                "block": variant["block"],
                "label": variant["label"],
                "seed": seed,
                "epoch_cap": args.epoch,
                "wall_min": wall / 60.0,
                "exit": exit_code,
                **variant["overrides"],
                **parsed,
            }
            rows.append(row)
            # Incremental JSON save after every run.
            aggregated = _aggregate(rows, grid) if not args.dry_run else []
            with args.output.open("w", encoding="utf-8") as fh:
                json.dump({
                    "rows": rows,
                    "aggregated": aggregated,
                    "runs_completed": run_idx,
                    "total_runs": total_runs,
                    "seeds": args.seeds,
                    "epoch_cap": args.epoch,
                    "protocol": {
                        "early_stopping_monitor": "val_recall@20",
                        "early_stopping_patience_evals": 5,
                        "early_stopping_min_epochs": 30,
                        "eval_every": 5,
                        "reduce_lr_factor": 0.5,
                        "reduce_lr_patience": 3,
                    },
                }, fh, indent=2)

    if args.dry_run:
        print("[dry_run] summary skipped.")
        return

    aggregated = _aggregate(rows, grid)
    _write_markdown(aggregated, args.output.with_suffix(".md"))

    # Persist final JSON with aggregation.
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump({
            "rows": rows,
            "aggregated": aggregated,
            "runs_completed": len(rows),
            "total_runs": total_runs,
            "seeds": args.seeds,
            "epoch_cap": args.epoch,
            "protocol": {
                "early_stopping_monitor": "val_recall@20",
                "early_stopping_patience_evals": 5,
                "early_stopping_min_epochs": 30,
                "eval_every": 5,
                "reduce_lr_factor": 0.5,
                "reduce_lr_patience": 3,
            },
        }, fh, indent=2)

    print("\n=== KSE final 5-seed benchmark (seed-wise mean * 100, +/- std) ===")
    print(f"{'Variant':<48} " + " ".join(f"{lbl:>13}" for _, lbl in _METRIC_KEYS))
    for r in aggregated:
        cells = []
        for key, _ in _METRIC_KEYS:
            m = r.get(f"{key}_mean")
            s = r.get(f"{key}_std")
            if m is None or (isinstance(m, float) and math.isnan(m)):
                cells.append(f"{'—':>13}")
            else:
                cells.append(f"{m*100:>6.3f}+/-{s*100:.3f}")
        print(f"{r['label'][:47]:<48} " + " ".join(cells))

    if not aggregated:
        return
    full = next((r for r in aggregated if r["tag"] == "A0_kse_full"), None)
    if not full:
        return
    print("\n=== Ablation deltas (percentage points on * 100 scale, vs A0_kse_full) ===")
    full_r20 = (full.get("best_test_recall20_mean") or 0) * 100
    full_nd  = (full.get("best_test_ndcg20_mean")   or 0) * 100
    full_h   = (full.get("best_test_head_recall20_mean") or 0) * 100
    full_m   = (full.get("best_test_mid_recall20_mean")  or 0) * 100
    full_t   = (full.get("best_test_tail_recall20_mean") or 0) * 100
    for r in aggregated:
        if r["tag"] == "A0_kse_full":
            continue
        d_r20 = (r.get("best_test_recall20_mean") or 0) * 100 - full_r20
        d_nd  = (r.get("best_test_ndcg20_mean")   or 0) * 100 - full_nd
        d_h   = (r.get("best_test_head_recall20_mean") or 0) * 100 - full_h
        d_m   = (r.get("best_test_mid_recall20_mean")  or 0) * 100 - full_m
        d_t   = (r.get("best_test_tail_recall20_mean") or 0) * 100 - full_t
        print(f"  {r['tag']:<20}  dR@20={d_r20:+.3f}  dNDCG={d_nd:+.3f}  "
              f"dHead={d_h:+.3f}  dMid={d_m:+.3f}  dTail={d_t:+.3f}")

    print(f"\n=== Paper-ready Markdown table written to {args.output.with_suffix('.md')} ===")


if __name__ == "__main__":
    main()
