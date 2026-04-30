"""
scripts/_aggregate_permutation_fft.py
======================================

Aggregate the seed-paired Permutation-FFT ablation runs (compliance check
INFO 4 / spec Section 6, Item 8).

For each seed we expect two per-run log files produced by ``train.py``:

    ../<dataset>/perm_fft_off_seed<S>/perm_fft_off_seed<S>.txt
    ../<dataset>/perm_fft_on_seed<S>/perm_fft_on_seed<S>.txt

Each log contains lines of the form::

    BEST_Test_Recall@20: 0.08920000
    BEST_Test_Precision@20: 0.00451000
    BEST_Test_NDCG@20: 0.04050000

We extract these, perform a paired t-test for both Recall@20 and NDCG@20,
report the percentage gap, and apply the spec's fallback rule:

    *  gap < 1 %   ->  switch the spectral basis to DCT-II (warning).
    *  gap >= 1 %  ->  standard 1-D FFT validated.

Run from the workspace root::

    python MMHCL_DAMPS_Project/scripts/_aggregate_permutation_fft.py \\
        --dataset Clothing --seeds 42 43 44
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from typing import Optional


_RE_RECALL = re.compile(r'BEST_Test_Recall@20[:=]\s*([\d.]+)')
_RE_NDCG = re.compile(r'BEST_Test_NDCG@20[:=]\s*([\d.]+)')


def _extract(log_path: str) -> Optional[tuple[float, float]]:
    """Pull (recall@20, ndcg@20) out of a per-run log file."""
    if not os.path.exists(log_path):
        return None
    with open(log_path, "r", encoding="utf-8") as f:
        text = f.read()
    r = _RE_RECALL.search(text)
    n = _RE_NDCG.search(text)
    if r is None or n is None:
        return None
    return float(r.group(1)), float(n.group(1))


def _paired_t_test(diffs: list[float]) -> tuple[float, float]:
    """
    Two-sided paired t-test for the difference series.

    Returns ``(t_stat, p_two_sided)``. Uses a small custom implementation
    so we don't introduce a hard ``scipy`` dependency.
    """
    n = len(diffs)
    if n < 2:
        return float("nan"), float("nan")
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    sd = math.sqrt(max(var, 1e-30))
    se = sd / math.sqrt(n)
    if se == 0.0:
        return float("inf"), 0.0
    t = mean / se
    # Two-sided p-value via the normal approximation (df = n-1; for n >= 5
    # the normal tail is within 5 % of the t-tail and is good enough for
    # the spec's binary "<1% or not" decision).
    z = abs(t)
    p = 2.0 * 0.5 * math.erfc(z / math.sqrt(2.0))
    return t, p


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="Clothing")
    parser.add_argument(
        "--seeds", type=int, nargs="+", required=True,
        help="List of seeds used by run_permutation_fft_ablation",
    )
    parser.add_argument(
        "--results_root", default="../",
        help="Root containing the dataset folders (default: ../, "
             "matching the train.py cwd convention).",
    )
    parser.add_argument(
        "--gap_threshold", type=float, default=1.0,
        help="Absolute percentage gap (Recall@20) below which the spec "
             "recommends falling back to DCT-II.",
    )
    args = parser.parse_args()

    dataset_root = os.path.join(args.results_root, args.dataset)

    rows: list[tuple[int, float, float, float, float]] = []
    missing: list[int] = []

    for seed in args.seeds:
        off_dir = f"perm_fft_off_seed{seed}"
        on_dir = f"perm_fft_on_seed{seed}"
        off_log = os.path.join(dataset_root, off_dir, f"{off_dir}.txt")
        on_log = os.path.join(dataset_root, on_dir, f"{on_dir}.txt")

        off = _extract(off_log)
        on = _extract(on_log)
        if off is None or on is None:
            missing.append(seed)
            continue
        rows.append((seed, off[0], off[1], on[0], on[1]))

    if not rows:
        print("[FATAL] No completed paired runs found.")
        for s in missing:
            print(f"  seed={s}: missing log file(s)")
        return 1

    print("=" * 78)
    print(f"Permutation-FFT paired ablation: dataset={args.dataset}, "
          f"n_pairs={len(rows)}")
    print("=" * 78)
    print(f"{'seed':>6}  {'R@20 (off)':>12}  {'R@20 (on)':>12}  "
          f"{'N@20 (off)':>12}  {'N@20 (on)':>12}")
    print("-" * 78)
    diffs_recall: list[float] = []
    diffs_ndcg: list[float] = []
    for seed, ro, no, ron, non_ in rows:
        diffs_recall.append(ro - ron)
        diffs_ndcg.append(no - non_)
        print(f"{seed:>6}  {ro:>12.6f}  {ron:>12.6f}  "
              f"{no:>12.6f}  {non_:>12.6f}")

    mean_off_r = sum(r[1] for r in rows) / len(rows)
    mean_on_r = sum(r[3] for r in rows) / len(rows)
    mean_off_n = sum(r[2] for r in rows) / len(rows)
    mean_on_n = sum(r[4] for r in rows) / len(rows)

    gap_r_pct = (mean_off_r - mean_on_r) / max(mean_off_r, 1e-12) * 100.0
    gap_n_pct = (mean_off_n - mean_on_n) / max(mean_off_n, 1e-12) * 100.0

    t_r, p_r = _paired_t_test(diffs_recall)
    t_n, p_n = _paired_t_test(diffs_ndcg)

    print("-" * 78)
    print(f"  mean Recall@20  : off={mean_off_r:.6f}  on={mean_on_r:.6f}  "
          f"gap={gap_r_pct:+.2f}%  t={t_r:.3f}  p={p_r:.4f}")
    print(f"  mean NDCG@20    : off={mean_off_n:.6f}  on={mean_on_n:.6f}  "
          f"gap={gap_n_pct:+.2f}%  t={t_n:.3f}  p={p_n:.4f}")

    print("=" * 78)
    threshold = args.gap_threshold
    if abs(gap_r_pct) < threshold:
        print(f"[WARN] |gap| < {threshold:.1f}%  ->  the spec recommends "
              f"switching the spectral basis to DCT-II (Section 6, Item 8).")
        verdict = "fallback_dct"
    else:
        print(f"[OK]   |gap| >= {threshold:.1f}%  ->  standard 1-D FFT path "
              f"is validated (no DCT-II fallback required).")
        verdict = "fft_validated"

    if missing:
        print()
        print("Note: the following seeds had no log file and were skipped:")
        for s in missing:
            print(f"  seed={s}")
    print(f"\nverdict = {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
