"""scripts/run_p7_0_simgcl_capacity_grid.py -- P7.0 SimGCL + P7.1 Capacity.
=============================================================================

Combined grid for the *loss-channel* and *architectural-capacity* levers,
built on top of the ``p6_6c_regs`` locked config (§9.27).

Grid design (9 variants x 1 seed x 100 epoch ~= 3.7 h)
-------------------------------------------------------

Block S -- P7.0 SimGCL enable + eps/lambda sweep (6 variants):
    Anchor:  regs=4.8e-4, logq=0.651, embed=128, UI_layers=3, rsfp=off.
    s0_ref             enable_simgcl=0  (baseline replicate)
    s1_simgcl_010      enable_simgcl=1  lambda_view=0.10 simgcl_eps=0.329
    s2_simgcl_005      enable_simgcl=1  lambda_view=0.05 simgcl_eps=0.329
    s3_simgcl_020      enable_simgcl=1  lambda_view=0.20 simgcl_eps=0.329
    s4_eps050          enable_simgcl=1  lambda_view=0.10 simgcl_eps=0.50
    s5_eps020          enable_simgcl=1  lambda_view=0.10 simgcl_eps=0.20

Block K -- P7.1 Capacity sweep (3 variants):
    Anchor:  regs=4.8e-4, logq=0.651, enable_simgcl=0, rsfp=off.
    k1_e192            embed_size=192  UI_layers=3
    k2_e256            embed_size=256  UI_layers=3
    k3_L4              embed_size=128  UI_layers=4

Total 6 + 3 = 9 variants, 1 seed (23946202) each -> 9 runs.

Usage
-----
::

    python scripts/run_p7_0_simgcl_capacity_grid.py \\
        --seeds 23946202 \\
        --epoch 100 \\
        --output ./results/p7_0_simgcl_capacity_clothing.json \\
        --dry_run 0

Notes
-----
* Capacity variants (k1/k2) may be slower per epoch (~18-25 s vs ~15 s
  baseline).  Wall estimate includes 15 % slack.
* All variants use the P6.4/p6_6c anchor cache
  ``results/interest_tree_clothing.npz`` -- no RSFP blending here.
* Sibling driver ``run_p7_0_boost_c1_grid.py`` covers the RSFP + logq
  micro-grid (block D).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Locked baseline (from p6_6c_regs § 9.27 winner)
# ---------------------------------------------------------------------------
BASELINE_CLI = {
    "dataset": "Clothing",
    "core": 5,
    "seed": None,        # filled per run
    "epoch": 100,
    "batch_size": 1024,
    "lr": 0.000250995,
    "clip_grad_norm": 1.0,
    "embed_size": 128,
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
    "early_stopping_patience": 30,
    "early_stopping_min_epochs": 0,
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
    "wandb_group": "p7_0_simgcl_capacity",
}


# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------
def build_grid(base_cache: str) -> list[dict]:
    grid: list[dict] = []

    # --- Block S: P7.0 SimGCL enable + eps/lambda sweep ---
    #   s0_ref replicates the p6_6c_regs baseline (SimGCL off).  Kept for
    #   apples-to-apples comparison with s1..s5.
    grid.append({
        "tag": "s0_ref",
        "block": "S_p7_0_simgcl",
        "overrides": {
            "enable_simgcl": 0,
            "lambda_view": 0.10,
            "simgcl_eps": 0.329,
            "embed_size": 128,
            "UI_layers": 3,
            "tamer_interest_cache": base_cache,
        },
    })
    for tag, lam, eps in [
        ("s1_simgcl_010", 0.10, 0.329),
        ("s2_simgcl_005", 0.05, 0.329),
        ("s3_simgcl_020", 0.20, 0.329),
        ("s4_eps050",     0.10, 0.50),
        ("s5_eps020",     0.10, 0.20),
    ]:
        grid.append({
            "tag": tag,
            "block": "S_p7_0_simgcl",
            "overrides": {
                "enable_simgcl": 1,
                "lambda_view": lam,
                "simgcl_eps": eps,
                "embed_size": 128,
                "UI_layers": 3,
                "tamer_interest_cache": base_cache,
            },
        })

    # --- Block K: P7.1 Capacity sweep ---
    for tag, emb, ui in [
        ("k1_e192", 192, 3),
        ("k2_e256", 256, 3),
        ("k3_L4",   128, 4),
    ]:
        grid.append({
            "tag": tag,
            "block": "K_p7_1_capacity",
            "overrides": {
                "enable_simgcl": 0,
                "embed_size": emb,
                "UI_layers": ui,
                "tamer_interest_cache": base_cache,
            },
        })

    return grid


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
        "p7", "p7_0_simgcl_capacity", variant["block"], variant["tag"],
        f"embed{cfg['embed_size']}",
        f"L{cfg['UI_layers']}",
        f"simgcl{cfg['enable_simgcl']}",
        f"lam{cfg['lambda_view']}",
        f"eps{cfg['simgcl_eps']}",
        "nrdmc_lite", "tamer", "alpha050", "r20_monitor",
    ])
    cmd = [python_exe, str(main_py)]
    for k, v in cfg.items():
        if isinstance(v, bool):
            v = int(v)
        cmd += [f"--{k}", str(v)]
    return cmd


# ---------------------------------------------------------------------------
# Parsing + runner
# ---------------------------------------------------------------------------
def _parse_run_output(out: str) -> dict:
    result: dict = {}
    for key, pat in [
        ("best_test_recall20", r"BEST_Test_Recall@20:\s*([\d.]+)"),
        ("best_test_ndcg20",   r"BEST_Test_NDCG@20:\s*([\d.]+)"),
        ("best_val_recall20",  r"BEST_Val_Recall@20:\s*([\d.]+)"),
        ("best_val_ndcg20",    r"BEST_Val_NDCG@20:\s*([\d.]+)"),
        ("best_epoch",         r"BEST_Val_Recall_Peak_Epoch:\s*(\d+)"),
    ]:
        m = re.search(pat, out)
        if m:
            v = m.group(1)
            result[key] = int(v) if key == "best_epoch" else float(v)
    for name, pat in [
        ("best_val_head_recall20", r"BEST_Recall@20_Head=([\d.]+)"),
        ("best_val_mid_recall20",  r"BEST_Recall@20_Mid=([\d.]+)"),
        ("best_val_tail_recall20", r"BEST_Recall@20_Tail=([\d.]+)"),
        ("best_test_head_recall20", r"BEST_Test_Recall@20_Head=([\d.]+)"),
        ("best_test_mid_recall20",  r"BEST_Test_Recall@20_Mid=([\d.]+)"),
        ("best_test_tail_recall20", r"BEST_Test_Recall@20_Tail=([\d.]+)"),
    ]:
        m = re.search(pat, out)
        if m:
            result[name] = float(m.group(1))
    return result


def _resolve_main(main_arg: Path) -> Path:
    """Resolve main_tercile.py location like sibling drivers."""
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
    raise FileNotFoundError(
        f"Could not locate main_tercile.py. Tried: {main_arg}"
    )


def _run_one(cmd: list[str], log_path: Path, dry_run: bool) -> tuple[int, str, float]:
    if dry_run:
        print("[dry_run] " + " ".join(shlex.quote(c) for c in cmd))
        return 0, "", 0.0
    t0 = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[grid] running (tail): "
          f"{' '.join(shlex.quote(c) for c in cmd[-30:])}")
    print(f"[grid] log: {log_path}")
    with log_path.open("wb") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
        exit_code = proc.wait()
    wall = time.time() - t0
    out = log_path.read_text(encoding="utf-8", errors="replace")
    print(f"[grid] exit={exit_code}  wall={wall/60.0:.1f} min")
    return exit_code, out, wall


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[23946202])
    ap.add_argument("--epoch", type=int, default=100)
    ap.add_argument("--python", type=str, default=sys.executable)
    ap.add_argument("--main", type=Path, default=Path("main_tercile.py"))
    ap.add_argument("--output", type=Path,
                    default=Path("./results/p7_0_simgcl_capacity_clothing.json"))
    ap.add_argument("--log_dir", type=Path,
                    default=Path("./results/_p7_0_simgcl_capacity_logs"))
    ap.add_argument("--base_cache", type=Path,
                    default=Path("results/interest_tree_clothing.npz"))
    ap.add_argument("--dry_run", type=int, default=0)
    ap.add_argument("--only_tags", type=str, nargs="*", default=None)
    args = ap.parse_args()

    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    main_py = _resolve_main(args.main)
    print(f"[grid] main_py: {main_py}")

    grid = build_grid(str(args.base_cache))
    if args.only_tags:
        grid = [v for v in grid if v["tag"] in set(args.only_tags)]

    total_runs = len(grid) * len(args.seeds)
    print(f"[grid] total runs: {total_runs} "
          f"({len(grid)} variants x {len(args.seeds)} seeds x "
          f"{args.epoch} epoch)")

    rows: list[dict] = []
    run_idx = 0

    for variant in grid:
        for seed in args.seeds:
            run_idx += 1
            print(f"\n{'='*72}\n"
                  f"[grid] {run_idx}/{total_runs}  "
                  f"tag={variant['tag']}  seed={seed}\n{'='*72}")
            cmd = build_cli(args.python, main_py, variant, seed, args.epoch)
            log_path = args.log_dir / f"{variant['tag']}_seed{seed}.log"
            exit_code, out, wall = _run_one(cmd, log_path, bool(args.dry_run))
            parsed = _parse_run_output(out) if not args.dry_run else {}
            row = {
                "tag": variant["tag"],
                "block": variant["block"],
                "seed": seed,
                "epoch_cap": args.epoch,
                "wall_min": wall / 60.0,
                "exit": exit_code,
                **{k: v for k, v in variant["overrides"].items()
                   if k not in ("tamer_interest_cache",)},
                "tamer_interest_cache": variant["overrides"]["tamer_interest_cache"],
                **parsed,
            }
            rows.append(row)
            with args.output.open("w", encoding="utf-8") as fh:
                json.dump({"rows": rows, "runs_completed": run_idx,
                           "total_runs": total_runs}, fh, indent=2)

    if args.dry_run:
        print("[dry_run] summary skipped.")
        return

    p6_4_ref = {"r20": 0.09566, "ndcg20": 0.04382}
    p6_6c_ref = {"r20": 0.09625, "ndcg20": 0.04426}

    per_tag: dict = {}
    for r in rows:
        per_tag.setdefault(r["tag"], []).append(r)

    def _mean(rs, k):
        vs = [r[k] for r in rs if k in r]
        return sum(vs) / len(vs) if vs else None

    ranked = []
    for tag, rs in per_tag.items():
        r20 = _mean(rs, "best_test_recall20")
        nd = _mean(rs, "best_test_ndcg20")
        h = (_mean(rs, "best_test_head_recall20")
             or _mean(rs, "best_val_head_recall20"))
        m = (_mean(rs, "best_test_mid_recall20")
             or _mean(rs, "best_val_mid_recall20"))
        t = (_mean(rs, "best_test_tail_recall20")
             or _mean(rs, "best_val_tail_recall20"))
        ranked.append({
            "tag": tag,
            "block": rs[0]["block"],
            "n_seeds": len(rs),
            "recall20_mean": r20,
            "ndcg20_mean": nd,
            "head_mean": h,
            "mid_mean": m,
            "tail_mean": t,
            "delta_vs_p6_4_r20": (r20 - p6_4_ref["r20"]) if r20 else None,
            "delta_vs_p6_6c_r20": (r20 - p6_6c_ref["r20"]) if r20 else None,
            "delta_vs_p6_4_ndcg": (nd - p6_4_ref["ndcg20"]) if nd else None,
        })
    ranked.sort(key=lambda r: (r["recall20_mean"] or 0), reverse=True)

    with args.output.open("w", encoding="utf-8") as fh:
        json.dump({
            "rows": rows, "ranked": ranked,
            "p6_4_reference": p6_4_ref, "p6_6c_anchor": p6_6c_ref,
            "runs_completed": len(rows), "total_runs": total_runs,
        }, fh, indent=2)

    print("\n=== P7.0 SimGCL + P7.1 Capacity grid (ranked by R@20) ===")
    print(f"{'tag':<18} {'block':<20} {'R@20':>8} {'dR@20':>9} "
          f"{'NDCG':>8} {'Head':>8} {'Mid':>8} {'Tail':>8}")
    for r in ranked:
        r20 = r["recall20_mean"] or float("nan")
        dr20 = r["delta_vs_p6_4_r20"] or 0.0
        nd = r["ndcg20_mean"] or float("nan")
        h = r["head_mean"] or float("nan")
        m = r["mid_mean"] or float("nan")
        t = r["tail_mean"] or float("nan")
        print(f"{r['tag']:<18} {r['block']:<20} {r20:>8.5f} "
              f"{dr20:+9.5f} {nd:>8.5f} {h:>8.5f} {m:>8.5f} {t:>8.5f}")

    if not ranked:
        return
    w = ranked[0]
    print(f"\n=== VERDICT ===")
    print(f"  Winner: '{w['tag']}' ({w['block']}) "
          f"R@20={w['recall20_mean']:.5f} "
          f"(vs P6.4 = {w['delta_vs_p6_4_r20']:+.5f}, "
          f"vs p6_6c anchor = {w['delta_vs_p6_6c_r20']:+.5f})")
    d = w['delta_vs_p6_6c_r20'] or 0.0
    if d > 0.0005:
        print("  Signal above noise band. Adopt config + replicate 2 seeds.")
    elif d > -0.0005:
        print("  Within noise band. No adoption without seed replication.")
    else:
        print("  Regression vs p6_6c. Keep p6_6c_regs baseline.")


if __name__ == "__main__":
    main()
