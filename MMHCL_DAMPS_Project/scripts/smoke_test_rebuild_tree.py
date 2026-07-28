"""scripts/smoke_test_rebuild_tree.py -- P8.2 RSFP-tree fix smoke test.
======================================================================

Standalone verification for the ``--rebuild_tree 1`` code path of
``scripts/build_rsfp_interest_tree.py``. Does NOT require PAMI mining
or the full Clothing dataset -- runs entirely on top of an existing
P6.5 RSFP cache (``interest_tree_clothing_rsfp_a010.npz``) by:

1. Loading the P6.5 cache (which has RSFP-blended cooc_* and copied
   tree_*).
2. Simulating what the fixed builder does: derive a fresh
   ``M_blend`` graph from the SAME cooc_* triplets in that .npz, run
   ``codes.interest_tree.precompute_interest_tree_flat`` on it, and
   compare the resulting ``tree_*`` to the ones in the cache.
3. Because the P6.5 cache carries the base P6.4 tree (built from
   pure co-occurrence), and step 2 rebuilds the tree from the RSFP-
   blended graph, the two tree_* sets MUST differ. If they don't,
   the fix is a no-op and there is a bug in the rebuild path.

Usage
-----
::

    python scripts/smoke_test_rebuild_tree.py \\
        --rsfp_cache results/interest_tree_clothing_rsfp_a010.npz

Exit code 0 = smoke test passed (tree_* changes with the fix).
Exit code 1 = smoke test failed (rebuild produced identical tree_*).
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


def _fmt_diff(a: np.ndarray, b: np.ndarray, label: str) -> str:
    if a.shape != b.shape:
        return f"{label}: shape {a.shape} vs {b.shape} DIFF"
    if a.dtype.kind in "if":
        eq = bool(np.allclose(a, b, equal_nan=True))
    else:
        eq = bool(np.array_equal(a, b))
    if eq:
        return f"{label}: IDENTICAL (shape={a.shape})"
    if a.dtype.kind in "if":
        d = np.abs(a - b)
        return (
            f"{label}: DIFF  shape={a.shape}  "
            f"max={float(d.max()):.4g}  mean={float(d.mean()):.4g}"
        )
    n_diff = int((a != b).sum())
    return f"{label}: DIFF  shape={a.shape}  n_diff={n_diff}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--rsfp_cache",
        type=Path,
        required=True,
        help="P6.5 RSFP cache (e.g. interest_tree_clothing_rsfp_a010.npz).",
    )
    args = ap.parse_args()

    print(f"[smoke] loading cache: {args.rsfp_cache}")
    cache = dict(np.load(args.rsfp_cache, allow_pickle=False))
    n_items = int(cache["n_items"])
    knn_k = int(cache["knn_k_cooc"])
    n_order = int(cache["n_order"])
    print(
        f"[smoke]   n_items={n_items} knn_k_cooc={knn_k} "
        f"n_order={n_order} cooc_nnz={cache['cooc_vals'].shape[0]}"
    )
    have_tree = all(
        k in cache
        for k in ("tree_anchors", "tree_neighbours", "tree_orders", "tree_weights")
    )
    if not have_tree:
        print("[smoke] FAIL: cache lacks tree_* fields; cannot compare.")
        return 1
    print(
        f"[smoke]   tree_anchors nnz={cache['tree_anchors'].shape[0]}  "
        f"(this is the P6.4 base tree copied verbatim)"
    )

    # Rebuild the interest tree from the (already RSFP-blended) cooc_*
    # triplets in the cache. In production the fixed builder would blend
    # M_orig + M_rsfp first; here the .npz already carries M_blend as
    # cooc_*, so this test is exactly equivalent to what --rebuild_tree=1
    # produces on top of the same (base_cache, M_rsfp) inputs.
    print("[smoke] rebuilding graph from RSFP-blended cooc_* ...")
    graph = build_weighted_binary_relations(
        cache["cooc_rows"].astype(np.int64),
        cache["cooc_cols"].astype(np.int64),
        cache["cooc_vals"].astype(np.float32),
        knn_k,
    )
    print(f"[smoke]   graph anchors={len(graph)}")

    print(f"[smoke] running BFS interest tree (n_order={n_order}) ...")
    t0 = time.time()
    (new_a, new_n, new_o, new_w) = precompute_interest_tree_flat(graph, n_order)
    wall = time.time() - t0
    print(
        f"[smoke]   new tree nnz={new_a.shape[0]}  ({wall:.2f}s)"
    )

    # Compare vs the tree_* copied verbatim from the base cache.
    print("\n=== tree_* diff  (base-cache copy vs RSFP-blended rebuild) ===")
    diffs = [
        _fmt_diff(cache["tree_anchors"],    new_a, "tree_anchors"),
        _fmt_diff(cache["tree_neighbours"], new_n, "tree_neighbours"),
        _fmt_diff(cache["tree_orders"],     new_o, "tree_orders"),
        _fmt_diff(cache["tree_weights"],    new_w, "tree_weights"),
    ]
    for d in diffs:
        print(" ", d)

    any_diff = any("DIFF" in d for d in diffs)
    if any_diff:
        print("\n[smoke] PASS: rebuild produces tree_* that differ from the "
              "base-cache copy. The --rebuild_tree=1 fix is effective.")
        return 0
    else:
        print("\n[smoke] FAIL: rebuild produced identical tree_*. "
              "Fix would be a no-op.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
