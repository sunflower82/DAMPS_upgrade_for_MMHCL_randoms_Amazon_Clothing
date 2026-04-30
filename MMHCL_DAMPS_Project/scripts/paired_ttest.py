"""
scripts/paired_ttest.py -- Paired t-test + 95% CI helper
==========================================================

Implements the paired statistical comparison required by Section 4 of the
DAMPS-MMHCL Revision 9 specification (10 seeds + paired t-test across paired
DAMPS and baseline runs). Mirrors Section 8 of the Speedup Guide.

Usage
-----
::

    python paired_ttest.py --damps  damps_seeds.csv  --baseline mmhcl_seeds.csv

Both CSV files must contain a single ``recall@20`` column with one row per
seed in the *same order*. The script prints a markdown-formatted summary and
exits with code 0 (significant) or 1 (not significant) so it can be wired
into CI gates.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    from scipy import stats
except ImportError as exc:                                       # pragma: no cover
    print("ERROR: scipy is required for paired_ttest.py", file=sys.stderr)
    raise SystemExit(2) from exc


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------
def paired_ttest_report(
    damps_scores: Sequence[float],
    baseline_scores: Sequence[float],
    alpha: float = 0.05,
    label_a: str = "DAMPS-MMHCL",
    label_b: str = "MMHCL",
) -> dict[str, float | bool]:
    """
    Run a paired t-test (``scipy.stats.ttest_rel``) between two equally-long
    score sequences and report the 95% confidence interval on the mean of
    the paired differences.

    The paired-t test is the *correct* statistical test here because the seeds
    are matched across methods (using ``ttest_ind`` would inflate variance).
    """
    a = np.asarray(damps_scores, dtype=np.float64)
    b = np.asarray(baseline_scores, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(
            f"score arrays must have identical shape, got {a.shape} vs {b.shape}"
        )
    if a.ndim != 1:
        raise ValueError(f"expected 1-D arrays, got ndim={a.ndim}")
    if len(a) < 2:
        raise ValueError("need at least 2 paired observations")

    t_stat, p_val = stats.ttest_rel(a, b)
    diff = a - b
    n = len(a)
    sem_diff = stats.sem(diff)
    ci = stats.t.interval(1.0 - alpha, df=n - 1, loc=diff.mean(), scale=sem_diff)

    sig = bool(p_val < alpha)
    tag = "[significant]" if sig else "[n.s.]"

    print(f"|  Method            |   Mean   |    Std   |   N  |")
    print(f"|--------------------|----------|----------|------|")
    print(f"|  {label_a:<18}|  {a.mean():.4f}  |  {a.std(ddof=1):.4f}  |  {n}   |")
    print(f"|  {label_b:<18}|  {b.mean():.4f}  |  {b.std(ddof=1):.4f}  |  {n}   |")
    print()
    print(f"Paired t-test : t={float(t_stat):.3f}, p={float(p_val):.4g} {tag}")
    print(f"95% CI of mean diff = [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"alpha = {alpha:g}, n = {n}")

    return {
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "mean_diff": float(diff.mean()),
        "significant": sig,
    }


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------
def _load_csv_column(path: Path, column: str) -> list[float]:
    """Load ``column`` from a CSV file and return as a Python list of floats."""
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split(",")
        if column not in header:
            raise KeyError(
                f"column '{column}' not in {path.name}; have {header}"
            )
        idx = header.index(column)
        out: list[float] = []
        for line in fh:
            cells = line.rstrip("\n").split(",")
            if len(cells) <= idx or cells[idx] == "":
                continue
            out.append(float(cells[idx]))
        return out


def _main() -> int:
    parser = argparse.ArgumentParser(description="Paired t-test (DAMPS vs baseline)")
    parser.add_argument("--damps", type=Path, required=True,
                        help="CSV file with one column of DAMPS scores per seed.")
    parser.add_argument("--baseline", type=Path, required=True,
                        help="CSV file with one column of baseline scores per seed.")
    parser.add_argument("--column", type=str, default="recall@20",
                        help="Column name in both CSV files (default: recall@20).")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level (default 0.05).")
    args = parser.parse_args()

    damps_scores = _load_csv_column(args.damps, args.column)
    base_scores = _load_csv_column(args.baseline, args.column)

    report = paired_ttest_report(
        damps_scores=damps_scores,
        baseline_scores=base_scores,
        alpha=args.alpha,
    )
    return 0 if report["significant"] else 1


if __name__ == "__main__":
    raise SystemExit(_main())
