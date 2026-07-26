"""scripts/run_p6_5_p6_6_p6_9_grid.py -- P6.5 + P6.6 + P6.9 combined grid.
=============================================================================

12 variants x 2 seeds x 100 epochs (~10 h overnight on RTX 5090).

Baseline anchor (locked from P6.5'''+P6.6 previous grid, winner
``p6_6c_regs`` seed 23946202):
    alpha_interest = 0.50
    pop_inverse_eta = 0.0
    regs = 4.8e-4      (4x P6.4)
    logq_scale = 0.651
    reduce_lr_factor = 0.5, reduce_lr_patience = 3
    early_stopping_monitor = val_recall@20    <-- back to R@20 primary
    early_stopping_min_epochs = 0             <-- classical R@20 semantics

Blocks
------
A -- P6.6 regs-extended (5 variants, logq=0.651, rsfp=off)
    regs in {2.4e-4, 4.8e-4 anchor, 7.2e-4, 9.6e-4, 1.92e-3}
B -- P6.9 logq-resweep (4 variants, regs=4.8e-4, rsfp=off)
    logq_scale in {0.45, 0.55, 0.75, 0.85}
C -- P6.5 RSFP-augmented interest cache (3 variants, regs=4.8e-4, logq=0.651)
    tamer_interest_cache -> interest_tree_clothing_rsfp_a{010,020,040}.npz

Total 5+4+3 = 12 variants, 2 seeds each = 24 runs.

Usage
-----
::

    python scripts/run_p6_5_p6_6_p6_9_grid.py \\
        --seeds 23946202 1557638902 \\
        --epoch 100 \\
        --output ./results/p6_5_p6_6_p6_9_clothing.json \\
        --python "%DAMPS_PYTHON%" \\
        --dry_run 0

Prereq for block C
------------------
Run once beforehand:
::

    python scripts/build_rsfp_interest_tree.py \\
        --base_cache results/interest_tree_clothing.npz \\
        --dataset Clothing --data_dir ../data --core 5 \\
        --alphas 0.10 0.20 0.40 \\
        --min_sup 20 --min_ratio 0.4 \\
        --output_prefix results/interest_tree_clothing_rsfp
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
# Locked baseline (P6.4 winner + p6_6c_regs override)
# ---------------------------------------------------------------------------
BASELINE_CLI = {
    "dataset": "Clothing",
    "core": 5,
    "seed": None,  # filled per run
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
    # Wandb bookkeeping (per run).
    "use_wandb": 1,
    "wandb_project": "damps-mmhcl-clothing",
    "wandb_entity": "baitapck51cc-uet",
    "wandb_group": "p6_5_p6_6_p6_9",
}


# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------
def build_grid(base_cache: str, rsfp_prefix: str) -> list[dict]:
    """Return the 12-variant grid as list of override dicts."""
    grid: list[dict] = []

    # Block A: P6.6 regs-extended (5 variants).
    for tag, regs in [
        ("a1_regs2x",      2.4e-4),
        ("a2_regs4x_anch", 4.8e-4),
        ("a3_regs6x",      7.2e-4),
        ("a4_regs8x",      9.6e-4),
        ("a5_regs16x",     1.92e-3),
    ]:
        grid.append({
            "tag": tag,
            "block": "A_p6_6_regs",
            "regs": regs,
            "logq_scale": 0.651,
            "tamer_interest_cache": base_cache,
            "rsfp_alpha": 0.0,
        })

    # Block B: P6.9 logq-resweep (4 variants, at regs=4.8e-4).
    for tag, logq in [
        ("b1_logq045", 0.45),
        ("b2_logq055", 0.55),
        ("b3_logq075", 0.75),
        ("b4_logq085", 0.85),
    ]:
        grid.append({
            "tag": tag,
            "block": "B_p6_9_logq",
            "regs": 4.8e-4,
            "logq_scale": logq,
            "tamer_interest_cache": base_cache,
            "rsfp_alpha": 0.0,
        })

    # Block C: P6.5 RSFP-augmented cache (3 variants).
    for tag, alpha_pct in [
        ("c1_rsfp010", 10),
        ("c2_rsfp020", 20),
        ("c3_rsfp040", 40),
    ]:
        grid.append({
            "tag": tag,
            "block": "C_p6_5_rsfp",
            "regs": 4.8e-4,
            "logq_scale": 0.651,
            "tamer_interest_cache": f"{rsfp_prefix}_a{alpha_pct:03d}.npz",
            "rsfp_alpha": alpha_pct / 100.0,
        })

    return grid


# ---------------------------------------------------------------------------
# CLI construction
# ---------------------------------------------------------------------------
def build_cli(python_exe: str,
              main_py: Path,
              variant: dict,
              seed: int,
              output_dir: Path,
              epoch: int) -> list[str]:
    cfg = dict(BASELINE_CLI)
    cfg["seed"] = seed
    cfg["epoch"] = epoch
    cfg["regs"] = variant["regs"]
    cfg["logq_scale"] = variant["logq_scale"]
    cfg["tamer_interest_cache"] = str(variant["tamer_interest_cache"])
    cfg["wandb_run_name"] = f"{variant['tag']}_seed{seed}"
    cfg["wandb_tags"] = ",".join([
        "p6", "p6_5_p6_6_p6_9", variant["block"], variant["tag"],
        f"regs{variant['regs']:.2e}",
        f"logq{variant['logq_scale']}",
        f"rsfp{variant['rsfp_alpha']:.2f}",
        "nrdmc_lite", "tamer", "alpha050", "r20_monitor",
    ])

    cmd = [python_exe, str(main_py)]
    for k, v in cfg.items():
        if isinstance(v, bool):
            v = int(v)
        cmd += [f"--{k}", str(v)]
    return cmd


# ---------------------------------------------------------------------------
# Run parsing
# ---------------------------------------------------------------------------
def _parse_run_output(out: str) -> dict:
    """Extract BEST_* metrics + tercile results from stdout."""
    result = {}
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

    # Tercile-final line: BEST_Recall@20_Head=X BEST_Recall@20_Mid=Y BEST_Recall@20_Tail=Z
    for name, pat in [
        ("best_val_head_recall20", r"BEST_Recall@20_Head=([\d.]+)"),
        ("best_val_mid_recall20",  r"BEST_Recall@20_Mid=([\d.]+)"),
        ("best_val_tail_recall20", r"BEST_Recall@20_Tail=([\d.]+)"),
    ]:
        m = re.search(pat, out)
        if m:
            result[name] = float(m.group(1))

    for name, pat in [
        ("best_test_head_recall20", r"BEST_Test_Recall@20_Head=([\d.]+)"),
        ("best_test_mid_recall20",  r"BEST_Test_Recall@20_Mid=([\d.]+)"),
        ("best_test_tail_recall20", r"BEST_Test_Recall@20_Tail=([\d.]+)"),
    ]:
        m = re.search(pat, out)
        if m:
            result[name] = float(m.group(1))
    return result


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _run_one(cmd: list[str], log_path: Path, dry_run: bool) -> tuple[int, str, float]:
    if dry_run:
        print("[dry_run] " + " ".join(shlex.quote(c) for c in cmd))
        return 0, "", 0.0
    t0 = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[grid] running: {' '.join(shlex.quote(c) for c in cmd[-40:])}")
    print(f"[grid] logging to: {log_path}")
    with log_path.open("wb") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
        exit_code = proc.wait()
    wall = time.time() - t0
    out = log_path.read_text(encoding="utf-8", errors="replace")
    print(f"[grid] exit={exit_code}  wall={wall/60.0:.1f} min")
    return exit_code, out, wall


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[23946202, 1557638902])
    ap.add_argument("--epoch", type=int, default=100)
    ap.add_argument("--python", type=str, default=sys.executable)
    ap.add_argument("--main", type=Path, default=Path("codes/main.py"))
    ap.add_argument("--output", type=Path,
                    default=Path("./results/p6_5_p6_6_p6_9_clothing.json"))
    ap.add_argument("--log_dir", type=Path,
                    default=Path("./results/_p6_5_p6_6_p6_9_logs"))
    ap.add_argument("--base_cache", type=Path,
                    default=Path("results/interest_tree_clothing.npz"))
    ap.add_argument("--rsfp_prefix", type=Path,
                    default=Path("results/interest_tree_clothing_rsfp"))
    ap.add_argument("--dry_run", type=int, default=0)
    ap.add_argument("--only_tags", type=str, nargs="*", default=None,
                    help="Optional filter: only run these variant tags.")
    args = ap.parse_args()

    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    grid = build_grid(str(args.base_cache), str(args.rsfp_prefix))
    if args.only_tags:
        grid = [v for v in grid if v["tag"] in set(args.only_tags)]
        print(f"[grid] filtered to {len(grid)} variants: "
              f"{[v['tag'] for v in grid]}")

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
            cmd = build_cli(args.python, args.main, variant, seed,
                            args.log_dir, args.epoch)
            log_path = args.log_dir / f"{variant['tag']}_seed{seed}.log"
            exit_code, out, wall = _run_one(cmd, log_path, bool(args.dry_run))
            parsed = _parse_run_output(out) if not args.dry_run else {}
            row = {
                "tag": variant["tag"],
                "block": variant["block"],
                "seed": seed,
                "regs": variant["regs"],
                "logq_scale": variant["logq_scale"],
                "rsfp_alpha": variant["rsfp_alpha"],
                "tamer_interest_cache": str(variant["tamer_interest_cache"]),
                "exit": exit_code,
                "wall_min": wall / 60.0,
                "epoch_cap": args.epoch,
                **parsed,
            }
            rows.append(row)

            # Save incrementally so a crash mid-grid keeps prior results.
            with args.output.open("w", encoding="utf-8") as fh:
                json.dump({"rows": rows, "runs_completed": run_idx,
                           "total_runs": total_runs},
                          fh, indent=2)

    # ------ Aggregate + rank ------
    if args.dry_run:
        print("[dry_run] grid summary skipped.")
        return

    # Mean over seeds per tag.
    per_tag = {}
    for r in rows:
        t = r["tag"]
        per_tag.setdefault(t, []).append(r)

    p6_4_ref = {
        "mean_r20": 0.09566,
        "mean_ndcg20": 0.04382,
        "mean_r20_tail": 0.01563,
    }
    p6_6c_regs_anchor = {
        "mean_r20": 0.09625,
        "mean_ndcg20": 0.04426,
    }

    def _mean(rs, k):
        vals = [r[k] for r in rs if k in r]
        return sum(vals) / len(vals) if vals else None

    ranked = []
    for tag, rs in per_tag.items():
        m_r20 = _mean(rs, "best_test_recall20")
        m_nd = _mean(rs, "best_test_ndcg20")
        m_head = _mean(rs, "best_test_head_recall20") or _mean(rs, "best_val_head_recall20")
        m_mid = _mean(rs, "best_test_mid_recall20") or _mean(rs, "best_val_mid_recall20")
        m_tail = _mean(rs, "best_test_tail_recall20") or _mean(rs, "best_val_tail_recall20")
        ranked.append({
            "tag": tag,
            "block": rs[0]["block"],
            "n_seeds": len(rs),
            "recall20_mean": m_r20,
            "ndcg20_mean": m_nd,
            "head_mean": m_head,
            "mid_mean": m_mid,
            "tail_mean": m_tail,
            "delta_vs_p6_4_r20": (m_r20 - p6_4_ref["mean_r20"]) if m_r20 else None,
            "delta_vs_p6_4_ndcg": (m_nd - p6_4_ref["mean_ndcg20"]) if m_nd else None,
            "delta_vs_p6_6c_r20": (m_r20 - p6_6c_regs_anchor["mean_r20"]) if m_r20 else None,
            "delta_vs_p6_6c_ndcg": (m_nd - p6_6c_regs_anchor["mean_ndcg20"]) if m_nd else None,
        })

    ranked.sort(key=lambda r: (r["recall20_mean"] if r["recall20_mean"] else 0),
                reverse=True)

    with args.output.open("w", encoding="utf-8") as fh:
        json.dump({
            "rows": rows,
            "ranked": ranked,
            "p6_4_reference": p6_4_ref,
            "p6_6c_regs_anchor": p6_6c_regs_anchor,
            "runs_completed": len(rows),
            "total_runs": total_runs,
        }, fh, indent=2)

    # Pretty print.
    print("\n\n=== P6.5 + P6.6 + P6.9 grid summary (ranked by R@20 mean) ===")
    print(f"{'tag':<20} {'block':<15} {'R@20':>8} {'dR20':>9} {'NDCG':>8} "
          f"{'Head':>8} {'Mid':>8} {'Tail':>8}")
    for r in ranked:
        r20 = r["recall20_mean"] or float("nan")
        dr20 = r["delta_vs_p6_4_r20"] or 0.0
        nd = r["ndcg20_mean"] or float("nan")
        h = r["head_mean"] or float("nan")
        m = r["mid_mean"] or float("nan")
        t = r["tail_mean"] or float("nan")
        print(f"{r['tag']:<20} {r['block']:<15} {r20:>8.5f} "
              f"{dr20:+9.5f} {nd:>8.5f} {h:>8.5f} {m:>8.5f} {t:>8.5f}")

    # Verdict.
    if not ranked:
        return
    winner = ranked[0]
    print(f"\n=== VERDICT ===")
    print(f"  Winner: '{winner['tag']}' ({winner['block']}) "
          f"R@20={winner['recall20_mean']:.5f} "
          f"(delta vs P6.4 = {winner['delta_vs_p6_4_r20']:+.5f}, "
          f"delta vs p6_6c anchor = {winner['delta_vs_p6_6c_r20']:+.5f})")
    d = winner['delta_vs_p6_6c_r20'] or 0.0
    if d > 0.0005:
        print(f"  R@20 improved over p6_6c_regs anchor by "
              f"{d/p6_6c_regs_anchor['mean_r20']*100:.2f}%. Adopt config.")
    elif d > -0.0005:
        print("  R@20 within noise band vs p6_6c anchor. No adoption.")
    else:
        print("  R@20 regressed vs p6_6c anchor. Keep p6_6c_regs baseline.")


if __name__ == "__main__":
    main()
