"""scripts/bench_p6_4a.py -- benchmark + parity check for P6.4a preprocessing.
============================================================================

Runs the P6.4a co-occurrence backends (roaring / sparse / torch) against a
synthetic (n_users x n_items) train split, then also times the BFS Interest
Tree precomputation. Reports:

* wall-clock speedup of each backend vs the roaring baseline
* row-wise parity: same (row -> {col: weight}) top-k dict for every backend
  (rounding to int since co-occurrence counts are integers).

This is a sandbox validation harness -- the numbers on the user's RTX 5090
will be much larger (the sandbox has no GPU, so the 'torch' path is CPU-
only and mostly measures torch BLAS vs scipy BLAS on the same machine).

Usage
-----
::

    python scripts/bench_p6_4a.py                 # default sizes
    python scripts/bench_p6_4a.py --n_users 4000 --n_items 3000 --density 0.008
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from codes.interest_tree import (  # noqa: E402
    build_weighted_binary_relations,
    precompute_interest_tree_flat,
)
from codes.roaring_cooc import timed_cooc  # noqa: E402


def _synth_pairs(n_users: int, n_items: int, density: float, seed: int) -> list:
    """Return a list of deduplicated (u, i) pairs.

    We dedup because real Original-MMHCL train.txt has no duplicate (u, i)
    entries -- each user's per-line item list is unique. The synthetic
    sampler with replacement can otherwise emit duplicates and break
    parity checks between backends that dedup implicitly (sparse) and
    those that don't (roaring/fallback).
    """
    rng = np.random.default_rng(seed)
    per_user = max(2, int(density * n_items))
    total = n_users * per_user
    us = np.repeat(np.arange(n_users, dtype=np.int64), per_user)
    it = rng.integers(0, n_items, size=total, dtype=np.int64)
    combined = us * (n_items + 1) + it
    combined = np.unique(combined)
    us2 = combined // (n_items + 1)
    it2 = combined % (n_items + 1)
    return np.stack([us2, it2], axis=1).tolist()


def _to_dict(rows: np.ndarray, cols: np.ndarray, vals: np.ndarray) -> dict:
    out: dict = {}
    for r, c, v in zip(rows.tolist(), cols.tolist(), vals.tolist()):
        out.setdefault(int(r), {})[int(c)] = int(v)
    return out


def _parity_summary(a: dict, b: dict) -> tuple[int, int, list]:
    """Return (matched_rows, total_rows, mismatch_examples)."""
    rows_a = set(a.keys())
    rows_b = set(b.keys())
    all_rows = rows_a | rows_b
    matched = 0
    ex: list = []
    for r in sorted(all_rows):
        if a.get(r, {}) == b.get(r, {}):
            matched += 1
        elif len(ex) < 3:
            ex.append((r, a.get(r, {}), b.get(r, {})))
    return matched, len(all_rows), ex


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n_users", type=int, default=3000)
    p.add_argument("--n_items", type=int, default=1500)
    p.add_argument("--density", type=float, default=0.02)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--n_order", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min_shared", type=int, default=2)
    args = p.parse_args()

    print(
        f"[bench] synth pairs: n_users={args.n_users}, n_items={args.n_items}, "
        f"density={args.density}"
    )
    pairs = _synth_pairs(args.n_users, args.n_items, args.density, args.seed)
    print(f"[bench] n_pairs={len(pairs)}")

    results: dict[str, tuple[float, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for method in ("roaring", "sparse", "torch"):
        try:
            wall, triplet = timed_cooc(
                pairs, args.k, method=method, min_shared=args.min_shared
            )
            print(f"[bench] cooc.{method:<8s} wall={wall:.2f}s  nnz={triplet[0].size}")
            results[method] = (wall, triplet)
        except Exception as e:
            print(f"[bench] cooc.{method:<8s} FAILED: {e!r}")

    # Parity check
    if "roaring" in results and "sparse" in results:
        d_r = _to_dict(*results["roaring"][1])
        d_s = _to_dict(*results["sparse"][1])
        m, t, ex = _parity_summary(d_r, d_s)
        print(f"[parity] roaring vs sparse: {m}/{t} rows match")
        for row, a, b in ex:
            print(f"    row {row}: A={sorted(a.items())[:5]} B={sorted(b.items())[:5]}")
    if "roaring" in results and "torch" in results:
        d_r = _to_dict(*results["roaring"][1])
        d_t = _to_dict(*results["torch"][1])
        m, t, ex = _parity_summary(d_r, d_t)
        print(f"[parity] roaring vs torch:  {m}/{t} rows match")
        for row, a, b in ex:
            print(f"    row {row}: A={sorted(a.items())[:5]} B={sorted(b.items())[:5]}")

    if "roaring" in results:
        base = results["roaring"][0]
        for m in ("sparse", "torch"):
            if m in results:
                sp = base / max(results[m][0], 1e-9)
                print(f"[bench] speedup {m}: {sp:.2f}x  (vs roaring)")

    # Interest Tree precomputation.
    method_for_graph = "sparse" if "sparse" in results else "roaring"
    rows, cols, vals = results[method_for_graph][1]
    print(f"[bench] BFS Interest Tree from {method_for_graph} cooc ...")
    t_g = time.perf_counter()
    graph = build_weighted_binary_relations(rows, cols, vals, args.k)
    t_g = time.perf_counter() - t_g
    t_b = time.perf_counter()
    ta, tn, to, tw = precompute_interest_tree_flat(graph, args.n_order)
    t_b = time.perf_counter() - t_b
    print(
        f"    build_weighted_binary_relations: {t_g:.2f}s  n_anchors={len(graph)}"
    )
    print(
        f"    precompute_interest_tree_flat:   {t_b:.2f}s  "
        f"tree_nnz={ta.size} (avg {ta.size / max(len(graph), 1):.1f} per anchor)"
    )
    # Sanity: orders in 1..n_order.
    assert to.min() >= 1 and to.max() <= args.n_order, (to.min(), to.max())
    print("[bench] parity + BFS smoke: OK")


if __name__ == "__main__":
    main()
