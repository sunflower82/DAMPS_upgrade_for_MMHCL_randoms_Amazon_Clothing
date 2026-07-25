"""scripts/preprocess_interest_tree.py -- P6.4 offline interest-graph cache.
=============================================================================

Builds the item-item co-occurrence graph S^c (TAMER Eq. 3) from the training
interactions and writes a compact .npz cache consumed by
``run_p6_4_tamer.py``.  Because S^c depends only on the train split (not on
the modality features), we can pay the ~90 s CPU cost once per dataset and
reuse the cache across every P6.4 grid cell.

Usage
-----
::

    python scripts/preprocess_interest_tree.py \\
        --dataset Clothing \\
        --data_dir ./data \\
        --output   ./results/interest_tree_clothing.npz \\
        --knn_k_cooc 20 \\
        --knn_k_mod  10 \\
        --n_order    3 \\
        --gamma      1.0 \\
        --tau        1.0

Inputs
------
* ``--data_dir/<dataset>/train.txt``: one training user per line, whitespace
  separated ``user_id item_id [item_id ...]``.  This is the same file used by
  ``main_tercile.py`` and Original-MMHCL's data loader.

Outputs
-------
* .npz with fields ``{cooc_rows, cooc_cols, cooc_vals, knn_k_cooc, knn_k_mod,
  n_order, gamma, tau, n_items}`` -- see ``codes/damps_tamer.py``.
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

from codes.damps_tamer import save_interest_cache  # noqa: E402
from codes.roaring_cooc import timed_cooc  # noqa: E402


# ---------------------------------------------------------------------------
# Train.txt loader
# ---------------------------------------------------------------------------
def _load_train_pairs(path: Path):
    """Return list of (u, i) pairs from an Original-MMHCL style train.txt."""
    pairs = []
    max_i = -1
    with path.open("r", encoding="utf-8") as fh:
        for ln in fh:
            parts = ln.strip().split()
            if len(parts) < 2:
                continue
            u = int(parts[0])
            for tok in parts[1:]:
                i = int(tok)
                pairs.append((u, i))
                if i > max_i:
                    max_i = i
    return pairs, max_i + 1


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--output", required=True)
    p.add_argument("--knn_k_cooc", type=int, default=20)
    p.add_argument("--knn_k_mod", type=int, default=10)
    p.add_argument("--n_order", type=int, default=3)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--min_shared", type=int, default=2)
    args = p.parse_args()

    train_path = Path(args.data_dir) / args.dataset / "train.txt"
    if not train_path.is_file():
        raise SystemExit(f"train.txt not found at {train_path}")
    print(f"[P6.4-preprocess] loading {train_path} ...")
    t0 = time.perf_counter()
    pairs, n_items = _load_train_pairs(train_path)
    print(f"    n_pairs={len(pairs)}  n_items={n_items}  ({time.perf_counter()-t0:.1f}s)")

    print(f"[P6.4-preprocess] building co-occurrence top-{args.knn_k_cooc} ...")
    wall, (rows, cols, vals) = timed_cooc(
        pairs, args.knn_k_cooc, method="roaring", min_shared=args.min_shared
    )
    print(f"    nnz={rows.shape[0]}  ({wall:.1f}s)")

    out_path = Path(args.output)
    save_interest_cache(
        out_path,
        cooc_rows=rows,
        cooc_cols=cols,
        cooc_vals=vals,
        knn_k_cooc=args.knn_k_cooc,
        knn_k_mod=args.knn_k_mod,
        n_order=args.n_order,
        gamma=args.gamma,
        tau=args.tau,
        n_items=n_items,
    )
    print(f"[P6.4-preprocess] wrote {out_path}")


if __name__ == "__main__":
    main()
