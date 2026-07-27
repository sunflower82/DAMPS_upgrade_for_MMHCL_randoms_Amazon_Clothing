"""scripts/run_p7_0_boost_c1_grid.py -- P7.2 c1_rsfp010 boost micro-grid.
=============================================================================

Rationale
---------
Per §9.28 seed-mean, ``c1_rsfp010`` (rsfp_a010) delivered *stronger*
NDCG@20 / Mid@20 / Tail@20 than ``b2_logq055`` while conceding a small
R@20 gap driven by Head decay past epoch ~76.  The two winners rely on
orthogonal levers:

* ``b2_logq055``: reduces ``logq_scale`` -> softer popularity de-bias ->
  saves Head@20 late in training.
* ``c1_rsfp010``: fills the interest cache on Mid/Tail items whose
  co-occurrence support was too sparse for the base cache to reach 20
  partners -> lifts Mid/Tail@20 monotonically to epoch ~99.

Because RSFP only *fills* rows that base cache left thin (the auto-
selectivity property observed in §9.28), the two levers combine
additively rather than trading off.  Block D probes this hypothesis.

Grid design (7 variants x 1 seed x 100 epoch ~= 3 h)
-----------------------------------------------------

All variants: regs=4.8e-4, alpha_interest=0.5, pop_inverse_eta=0.0,
              embed=128, UI_layers=3, enable_simgcl=0.

    d0_c1_confirm            rsfp=0.10  logq=0.651  cache=rsfp_a010
        (single-seed replicate of §9.28 c1_rsfp010 at seed 23946202;
         serves as within-run anchor)
    d1_logq055               rsfp=0.10  logq=0.55   cache=rsfp_a010
        (main hypothesis -- Head-boost + Mid/Tail-boost)
    d2_logq060               rsfp=0.10  logq=0.60   cache=rsfp_a010
    d3_rsfp008_logq055       rsfp=0.08  logq=0.55   cache=rsfp_a008
    d4_rsfp015_logq055       rsfp=0.15  logq=0.55   cache=rsfp_a015
    d5_dense_rsfp010_logq055 rsfp=0.10  logq=0.55   cache=rsfp_dense_a010
    d6_dense_rsfp015_logq055 rsfp=0.15  logq=0.55   cache=rsfp_dense_a015

    Total 7 variants, seed 23946202 (1 seed) -> 7 runs.

Prereq caches
-------------
Before Block D, run:
::

    python scripts/build_rsfp_interest_tree.py \\
        --base_cache results/interest_tree_clothing.npz \\
        --dataset Clothing --data_dir ../data --core 5 \\
        --alphas 0.08 0.10 0.15 \\
        --min_sup 20 --min_ratio 0.4 \\
        --output_prefix results/interest_tree_clothing_rsfp

    python scripts/build_rsfp_interest_tree.py \\
        --base_cache results/interest_tree_clothing.npz \\
        --dataset Clothing --data_dir ../data --core 5 \\
        --alphas 0.10 0.15 \\
        --min_sup 10 --min_ratio 0.2 \\
        --output_prefix results/interest_tree_clothing_rsfp_dense

Usage
-----
::

    python scripts/run_p7_0_boost_c1_grid.py \\
        --seeds 23946202 \\
        --epoch 100 \\
        --output ./results/p7_0_boost_c1_clothing.json \\
        --dry_run 0
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


BASELINE_CLI = {
    "dataset": "Clothing",
    "core": 5,
    "seed": None,
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
    "wandb_group": "p7_0_boost_c1",
}


def build_grid(cache_prefix: str, dense_prefix: str) -> list[dict]:
    """Block D: c1_rsfp010 boost micro-grid, 7 variants."""

    def _std_cache(alpha_pct: str) -> str:
        return f"{cache_prefix}_a{alpha_pct}.npz"

    def _dense_cache(alpha_pct: str) -> str:
        return f"{dense_prefix}_a{alpha_pct}.npz"

    grid = [
        # d0 -- single-seed c1 anchor (§9.28 replicate)
        {"tag": "d0_c1_confirm", "block": "D_boost_c1",
         "overrides": {"logq_scale": 0.651,
                       "tamer_interest_cache": _std_cache("010")}},
        # d1 -- MAIN: Head-boost (logq=0.55) + Mid/Tail-boost (rsfp=0.10)
        {"tag": "d1_logq055", "block": "D_boost_c1",
         "overrides": {"logq_scale": 0.55,
                       "tamer_interest_cache": _std_cache("010")}},
        # d2 -- softer logq (0.60)
        {"tag": "d2_logq060", "block": "D_boost_c1",
         "overrides": {"logq_scale": 0.60,
                       "tamer_interest_cache": _std_cache("010")}},
        # d3 -- lighter RSFP (0.08) + logq=0.55
        {"tag": "d3_rsfp008_logq055", "block": "D_boost_c1",
         "overrides": {"logq_scale": 0.55,
                       "tamer_interest_cache": _std_cache("008")}},
        # d4 -- stronger RSFP (0.15) + logq=0.55
        {"tag": "d4_rsfp015_logq055", "block": "D_boost_c1",
         "overrides": {"logq_scale": 0.55,
                       "tamer_interest_cache": _std_cache("015")}},
        # d5 -- dense mining RSFP (0.10) + logq=0.55
        {"tag": "d5_dense_rsfp010_logq055", "block": "D_boost_c1",
         "overrides": {"logq_scale": 0.55,
                       "tamer_interest_cache": _dense_cache("010")}},
        # d6 -- dense mining RSFP (0.15) + logq=0.55
        {"tag": "d6_dense_rsfp015_logq055", "block": "D_boost_c1",
         "overrides": {"logq_scale": 0.55,
                       "tamer_interest_cache": _dense_cache("015")}},
    ]
    return grid


def build_cli(python_exe: str, main_py: Path, variant: dict,
              seed: int, epoch: int) -> list[str]:
    cfg = dict(BASELINE_CLI)
    cfg["seed"] = seed
    cfg["epoch"] = epoch
    cfg.update(variant["overrides"])
    cfg["wandb_run_name"] = f"{variant['tag']}_seed{seed}"
    cfg["wandb_tags"] = ",".join([
        "p7", "p7_0_boost_c1", variant["block"], variant["tag"],
        f"logq{cfg['logq_scale']}",
        Path(cfg["tamer_interest_cache"]).stem,
        "rsfp", "nrdmc_lite", "tamer", "alpha050", "r20_monitor",
    ])
    cmd = [python_exe, str(main_py)]
    for k, v in cfg.items():
        if isinstance(v, bool):
            v = int(v)
        cmd += [f"--{k}", str(v)]
    return cmd


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
    print(f"\n[grid] running (tail): "
          f"{' '.join(shlex.quote(c) for c in cmd[-24:])}")
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
                    default=Path("./results/p7_0_boost_c1_clothing.json"))
    ap.add_argument("--log_dir", type=Path,
                    default=Path("./results/_p7_0_boost_c1_logs"))
    ap.add_argument("--cache_prefix", type=str,
                    default="results/interest_tree_clothing_rsfp")
    ap.add_argument("--dense_prefix", type=str,
                    default="results/interest_tree_clothing_rsfp_dense")
    ap.add_argument("--dry_run", type=int, default=0)
    ap.add_argument("--only_tags", type=str, nargs="*", default=None)
    args = ap.parse_args()

    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    main_py = _resolve_main(args.main)
    print(f"[grid] main_py: {main_py}")

    grid = build_grid(args.cache_prefix, args.dense_prefix)
    if args.only_tags:
        grid = [v for v in grid if v["tag"] in set(args.only_tags)]

    # Sanity: verify all cache files exist (skip in dry_run).
    if not args.dry_run:
        missing = []
        for v in grid:
            p = Path(v["overrides"]["tamer_interest_cache"])
            if not p.is_file():
                missing.append(str(p))
        if missing:
            print("[grid] WARNING -- missing RSFP caches:")
            for m in missing:
                print(f"          {m}")
            print("[grid] Run scripts/build_rsfp_interest_tree.py first "
                  "(see docstring).")

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
                **variant["overrides"],
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
    c1_ref = {"r20": 0.09556, "ndcg20": 0.04389,
              "head": 0.14498, "mid": 0.03289, "tail": 0.01395}
    b2_ref = {"r20": 0.09571, "ndcg20": 0.04386,
              "head": 0.14699, "mid": 0.02942, "tail": 0.01231}

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
            "delta_vs_c1_r20":   (r20 - c1_ref["r20"])   if r20 else None,
            "delta_vs_b2_r20":   (r20 - b2_ref["r20"])   if r20 else None,
        })
    ranked.sort(key=lambda r: (r["recall20_mean"] or 0), reverse=True)

    with args.output.open("w", encoding="utf-8") as fh:
        json.dump({
            "rows": rows, "ranked": ranked,
            "p6_4_reference": p6_4_ref,
            "c1_rsfp010_anchor": c1_ref,
            "b2_logq055_anchor": b2_ref,
            "runs_completed": len(rows), "total_runs": total_runs,
        }, fh, indent=2)

    print("\n=== P7.2 c1 boost micro-grid (ranked by R@20) ===")
    print(f"{'tag':<28} {'R@20':>8} {'dc1':>8} {'db2':>8} "
          f"{'NDCG':>8} {'Head':>8} {'Mid':>8} {'Tail':>8}")
    for r in ranked:
        r20 = r["recall20_mean"] or float("nan")
        dc = r["delta_vs_c1_r20"] or 0.0
        db = r["delta_vs_b2_r20"] or 0.0
        nd = r["ndcg20_mean"] or float("nan")
        h = r["head_mean"] or float("nan")
        m = r["mid_mean"] or float("nan")
        t = r["tail_mean"] or float("nan")
        print(f"{r['tag']:<28} {r20:>8.5f} {dc:>+8.5f} {db:>+8.5f} "
              f"{nd:>8.5f} {h:>8.5f} {m:>8.5f} {t:>8.5f}")

    if not ranked:
        return
    w = ranked[0]
    print(f"\n=== VERDICT ===")
    print(f"  Winner: '{w['tag']}' "
          f"R@20={w['recall20_mean']:.5f} "
          f"(vs c1_rsfp010 = {w['delta_vs_c1_r20']:+.5f}, "
          f"vs b2_logq055 = {w['delta_vs_b2_r20']:+.5f}, "
          f"vs P6.4 = {w['delta_vs_p6_4_r20']:+.5f})")
    d_b2 = w['delta_vs_b2_r20'] or 0.0
    if d_b2 > 0.0005:
        print("  Signal above §9.28 winner. Replicate 2 seeds before adopting.")
    elif d_b2 > -0.0005:
        print("  Within §9.28 noise band. No adoption without seed replication.")
    else:
        print("  Regression vs b2_logq055. Keep §9.28 winner.")


if __name__ == "__main__":
    main()
