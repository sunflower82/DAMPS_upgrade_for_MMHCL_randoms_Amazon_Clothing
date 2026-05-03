"""
scripts/run_phase1_ablation.py -- Phase 1 / rev44 four-configuration sweep
============================================================================

Run all four configurations from ``DAMPS_to_MMHCL_architecture_revision44.tex``
Section 4 across a list of random seeds, then aggregate the per-seed
``BEST_Test_Recall@20`` / ``BEST_Test_NDCG@20`` numbers, run paired t-tests
(Bonferroni-corrected for 3 simultaneous variants vs the anchor) and check
the rev44 quantitative stop-gate:

    Recall@20 >= 0.0870 AND NDCG@20 >= 0.0390  -> minimum acceptance
    Recall@20 >= 0.0900                        -> paper-contribution validation

Configurations (rev44 Section 4)
--------------------------------
  (a) anchor   : rev42 baseline           --temperature 0.1 --learnable_tau 1 --damps_avrf 1
  (b) tau_fix  : static tau only          --temperature 0.3 --learnable_tau 0 --damps_avrf 1
  (c) avrf_off : AVRF off only            --temperature 0.1 --learnable_tau 1 --damps_avrf 0
  (d) combined : tau + AVRF off (rev44)   --temperature 0.3 --learnable_tau 0 --damps_avrf 0  <- RECOMMENDED

Usage
-----
::

    python MMHCL_DAMPS_Project/scripts/run_phase1_ablation.py \\
        --dataset Clothing \\
        --seeds 42 43 44 45 46 47 48 49 50 51 \\
        --variants a b c d

    # Only aggregate, do not re-train (logs already exist):
    python MMHCL_DAMPS_Project/scripts/run_phase1_ablation.py \\
        --dataset Clothing --seeds 42 43 44 45 46 47 48 49 50 51 \\
        --aggregate_only

The script invokes ``MMHCL_DAMPS_Project/train.py`` once per (variant, seed)
pair via ``subprocess`` so each run inherits the user's current Python
environment (no env-spawn cost on the inner job).
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import statistics
import subprocess
import sys
from typing import Iterable, Optional

# Make the parent package importable for the t-test helper
_HERE = pathlib.Path(__file__).resolve().parent
_PKG = _HERE.parent
sys.path.insert(0, str(_PKG))

# Reuse the shared paired-t implementation (with Bonferroni support).
from scripts.paired_ttest import paired_ttest_report  # noqa: E402

# ---------------------------------------------------------------------------
#  Variant table
# ---------------------------------------------------------------------------
# Each variant is a tuple of CLI flags appended verbatim to the train.py call.
VARIANTS: dict[str, dict[str, object]] = {
    "a": {
        "label": "(a) rev42 anchor",
        "flags": ["--temperature", "0.1", "--learnable_tau", "1", "--damps_avrf", "1"],
    },
    "b": {
        "label": "(b) static tau=0.3",
        "flags": ["--temperature", "0.3", "--learnable_tau", "0", "--damps_avrf", "1"],
    },
    "c": {
        "label": "(c) AVRF off",
        "flags": ["--temperature", "0.1", "--learnable_tau", "1", "--damps_avrf", "0"],
    },
    "d": {
        "label": "(d) static tau + AVRF off (RECOMMENDED)",
        "flags": ["--temperature", "0.3", "--learnable_tau", "0", "--damps_avrf", "0"],
    },
}

# Stop-gate thresholds from rev44 Section 5.2.
ACCEPTANCE_RECALL = 0.0870
ACCEPTANCE_NDCG = 0.0390
PAPER_VALIDATION_RECALL = 0.0900

_RE_RECALL = re.compile(r"BEST_Test_Recall@20[:=]\s*([\d.]+)")
_RE_NDCG = re.compile(r"BEST_Test_NDCG@20[:=]\s*([\d.]+)")


# ---------------------------------------------------------------------------
#  Per-run helpers
# ---------------------------------------------------------------------------
def _path_name(args: argparse.Namespace, variant_flags: list[str], seed: int) -> str:
    """Reconstruct the directory naming used by ``train.py::_experiment_paths``."""
    # Pull values out of the variant flag list (small, parsed by hand here).
    fmap = dict(zip(variant_flags[0::2], variant_flags[1::2]))
    temperature = fmap.get("--temperature", "0.3")
    taulearn = fmap.get("--learnable_tau", "0")
    avrf = fmap.get("--damps_avrf", "0")
    apc = "1"
    imcf = "1"
    return (
        f"damps_uu_ii={args.User_layers}_{args.Item_layers}"
        f"_{args.user_loss_ratio}_{args.item_loss_ratio}"
        f"_topk={args.topk}_t={temperature}_taulearn={taulearn}_R={args.rebuild_R}"
        f"_apc={apc}_avrf={avrf}_imcf={imcf}"
        f"_regs={args.regs}_dim={args.embed_size}_seed={seed}_"
    )


def _log_path(args: argparse.Namespace, variant_flags: list[str], seed: int) -> pathlib.Path:
    name = _path_name(args, variant_flags, seed)
    return pathlib.Path(args.log_root) / args.dataset / name / f"{name}.txt"


def _extract_metrics(log: pathlib.Path) -> Optional[tuple[float, float]]:
    """Return (recall@20, ndcg@20) extracted from a per-run log file."""
    if not log.is_file():
        return None
    text = log.read_text(encoding="utf-8", errors="replace")
    m_r = _RE_RECALL.search(text)
    m_n = _RE_NDCG.search(text)
    if not m_r or not m_n:
        return None
    return float(m_r.group(1)), float(m_n.group(1))


def _spawn_training(
    args: argparse.Namespace,
    variant_flags: list[str],
    seed: int,
) -> int:
    """Invoke ``train.py`` once with the given variant + seed."""
    cmd = [
        args.python_exe, "train.py",
        "--dataset", args.dataset,
        "--seed", str(seed),
        "--epoch", str(args.epoch),
        "--rebuild_R", str(args.rebuild_R),
        "--User_layers", str(args.User_layers),
        "--Item_layers", str(args.Item_layers),
        "--topk", str(args.topk),
        "--regs", str(args.regs),
        "--embed_size", str(args.embed_size),
        "--user_loss_ratio", str(args.user_loss_ratio),
        "--item_loss_ratio", str(args.item_loss_ratio),
    ] + list(variant_flags)
    print(f"  $ {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(_PKG))


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the rev44 / Phase 1 four-configuration ablation sweep across "
            "a list of seeds and aggregate Recall@20 / NDCG@20 with paired "
            "t-tests + Bonferroni correction."
        )
    )
    parser.add_argument("--dataset", type=str, default="Clothing")
    parser.add_argument("--seeds", type=int, nargs="+", required=True,
                        help="List of integer seeds (rev44 protocol uses 10).")
    parser.add_argument("--variants", type=str, nargs="+",
                        choices=list(VARIANTS.keys()), default=["a", "b", "c", "d"],
                        help="Subset of variants to run (default: all four).")
    parser.add_argument("--epoch", type=int, default=250)
    parser.add_argument("--rebuild_R", type=int, default=5)
    parser.add_argument("--User_layers", type=int, default=3)
    parser.add_argument("--Item_layers", type=int, default=2)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--regs", type=float, default=1e-3)
    parser.add_argument("--embed_size", type=int, default=64)
    parser.add_argument("--user_loss_ratio", type=float, default=0.03)
    parser.add_argument("--item_loss_ratio", type=float, default=0.07)
    parser.add_argument("--log_root", type=str, default="..",
                        help="Root containing ``<dataset>/<run_dir>/`` log "
                             "files. Default '..' matches the trainer's "
                             "``../{dataset}/{name}/`` convention.")
    parser.add_argument("--python_exe", type=str, default=sys.executable,
                        help="Python interpreter used for the inner train.py call.")
    parser.add_argument("--aggregate_only", action="store_true",
                        help="Skip training; just collect metrics from "
                             "existing per-run logs.")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Family-wise significance level (default 0.05).")
    args = parser.parse_args()

    if "a" not in args.variants:
        print(
            "ERROR: variant 'a' (rev42 anchor) is required for the paired "
            "t-test; please include it.",
            file=sys.stderr,
        )
        return 1

    # ------------------------------------------------------------------
    # 1. Optionally run training for each (variant, seed).
    # ------------------------------------------------------------------
    if not args.aggregate_only:
        for v in args.variants:
            print(f"\n=== Variant {v}: {VARIANTS[v]['label']} ===")
            for seed in args.seeds:
                rc = _spawn_training(args, list(VARIANTS[v]["flags"]), seed)  # type: ignore[arg-type]
                if rc != 0:
                    print(
                        f"WARNING: variant={v} seed={seed} returned exit "
                        f"code {rc}; will be excluded from aggregation."
                    )

    # ------------------------------------------------------------------
    # 2. Aggregate.
    # ------------------------------------------------------------------
    per_variant: dict[str, dict[str, list[float]]] = {}
    for v in args.variants:
        rec_list: list[float] = []
        ndcg_list: list[float] = []
        for seed in args.seeds:
            log = _log_path(args, list(VARIANTS[v]["flags"]), seed)  # type: ignore[arg-type]
            metrics = _extract_metrics(log)
            if metrics is None:
                print(f"  [skip] missing/unfinished log: {log}")
                continue
            rec_list.append(metrics[0])
            ndcg_list.append(metrics[1])
        per_variant[v] = {"recall@20": rec_list, "ndcg@20": ndcg_list}

    # ------------------------------------------------------------------
    # 3. Print per-variant summary.
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("Per-variant summary (10-seed mean +/- std):")
    print("-" * 72)
    print(f"  {'Variant':<40s}  {'Recall@20':>16s}  {'NDCG@20':>16s}  {'N':>3s}")
    for v in args.variants:
        rec = per_variant[v]["recall@20"]
        ndcg = per_variant[v]["ndcg@20"]
        if not rec:
            print(f"  {VARIANTS[v]['label']:<40s}  {'(no logs)':>16s}  {'(no logs)':>16s}    0")
            continue
        rmean = statistics.mean(rec)
        rstd = statistics.stdev(rec) if len(rec) > 1 else 0.0
        nmean = statistics.mean(ndcg)
        nstd = statistics.stdev(ndcg) if len(ndcg) > 1 else 0.0
        rec_cell = f"{rmean:.4f}+-{rstd:.4f}"
        ndcg_cell = f"{nmean:.4f}+-{nstd:.4f}"
        print(
            f"  {VARIANTS[v]['label']:<40s}  {rec_cell:>16s}  "
            f"{ndcg_cell:>16s}  {len(rec):>3d}"
        )

    # ------------------------------------------------------------------
    # 4. Paired t-tests vs anchor (a) with Bonferroni correction.
    # ------------------------------------------------------------------
    other = [v for v in args.variants if v != "a"]
    bonferroni = max(1, len(other))
    if not per_variant["a"]["recall@20"]:
        print(
            "ERROR: anchor variant (a) produced no logs; cannot run paired "
            "t-tests."
        )
        return 1

    print("\n" + "=" * 72)
    print(
        f"Paired t-tests vs (a) rev42 anchor  "
        f"[Bonferroni correction = {bonferroni}]"
    )
    for v in other:
        for col in ("recall@20", "ndcg@20"):
            a_scores = per_variant["a"][col]
            v_scores = per_variant[v][col]
            n_pairs = min(len(a_scores), len(v_scores))
            if n_pairs < 2:
                print(f"\n[skip] {v} vs a, {col}: only {n_pairs} paired seeds available")
                continue
            print(f"\n----- variant {v} vs (a), metric {col} (n={n_pairs}) -----")
            paired_ttest_report(
                damps_scores=v_scores[:n_pairs],
                baseline_scores=a_scores[:n_pairs],
                alpha=args.alpha,
                bonferroni=bonferroni,
                label_a=str(VARIANTS[v]["label"]),
                label_b="(a) rev42 anchor",
            )

    # ------------------------------------------------------------------
    # 5. Stop-gate verdict.
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("rev44 Section 5.2 stop-gate verdict:")
    for v in args.variants:
        rec = per_variant[v]["recall@20"]
        ndcg = per_variant[v]["ndcg@20"]
        if not rec:
            print(f"  variant {v}: no data")
            continue
        rmean = statistics.mean(rec)
        nmean = statistics.mean(ndcg)
        beat_paper = rmean >= PAPER_VALIDATION_RECALL
        beat_acceptance = (
            rmean >= ACCEPTANCE_RECALL and nmean >= ACCEPTANCE_NDCG
        )
        if beat_paper:
            verdict = "PAPER VALIDATED  (Recall@20 >= 0.0900)"
        elif beat_acceptance:
            verdict = "PHASE 1 PASS     (Recall@20 >= 0.0870 AND NDCG@20 >= 0.0390)"
        else:
            verdict = "PHASE 1 FAIL     (re-audit eval protocol; checklist (iii) is the prime suspect)"
        print(f"  variant {v} -- mean Recall@20={rmean:.4f}, NDCG@20={nmean:.4f}: {verdict}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
