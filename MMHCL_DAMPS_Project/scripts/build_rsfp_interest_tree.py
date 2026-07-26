"""scripts/build_rsfp_interest_tree.py -- P6.5 RSFPGrowth-augmented cache.
=============================================================================

Builds RSFP-augmented ``interest_tree_clothing_rsfp_a{alpha}.npz`` variants by
blending the pre-existing P6.4 interest cache with 2-itemset edges mined by
PAMI ``RSFPGrowth`` from the training user-item transactions.

Zero-touch to model code: output is just another interest_tree .npz that
``run_p6_4_tamer.py`` consumes via ``--tamer_interest_cache``.

Pipeline
--------
1. Load train pairs (``data_dir/<dataset>/<core>-core/train.json`` or
   fallback ``train.txt``).
2. Emit tab-separated transactional DB (one line per user, tab-separated
   item ids) to ``<work_dir>/rsfp_txn_<dataset>.tsv``.
3. Run ``PAMI.relativeFrequentPattern.RSFPGrowth`` with the given
   ``--min_sup`` / ``--min_ratio``.
4. Filter mined patterns to 2-itemsets; build symmetric COO edges
   ``(i, j, support)`` -> ``M_rsfp``.  Normalise per-row to unit max, matching
   the P6.4 cache convention.
5. Blend with the P6.4 cache edges:
   ``M_blend = (1 - alpha) * M_orig + alpha * M_rsfp``.
6. Re-top-k per row at ``knn_k_cooc`` (from the base cache) and save.

The tree_* fields from the base cache are copied through UNCHANGED --
they encode a BFS traversal that depends on the anchor set, not on the
raw edge weights.  This keeps P6.5 minimally invasive.

Usage
-----
::

    python scripts/build_rsfp_interest_tree.py \\
        --base_cache results/interest_tree_clothing.npz \\
        --dataset Clothing \\
        --data_dir ../data \\
        --core 5 \\
        --alphas 0.10 0.20 0.40 \\
        --min_sup 20 \\
        --min_ratio 0.4 \\
        --output_prefix results/interest_tree_clothing_rsfp \\
        --work_dir ./results/_rsfp_work

Outputs one .npz per alpha at ``<output_prefix>_a{alpha_pct}.npz``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Train-pair loader (mirror of preprocess_interest_tree.py)
# ---------------------------------------------------------------------------
def _resolve_train_path(data_dir: Path, dataset: str, core: int) -> Path:
    candidates = [
        data_dir / dataset / f"{core}-core" / "train.json",
        data_dir / dataset / "train.json",
        data_dir / dataset / "train.txt",
        data_dir / "train.txt",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No train file found. Searched: {[str(c) for c in candidates]}"
    )


def _load_transactions(path: Path) -> tuple[list[list[int]], int]:
    """Return (list-of-item-lists per user, n_items)."""
    txns: list[list[int]] = []
    max_iid = -1
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as fh:
            train = json.load(fh)
        for _uid, items in train.items():
            if not items:
                continue
            row = [int(x) for x in items]
            if row:
                txns.append(row)
                max_iid = max(max_iid, max(row))
    else:
        with path.open("r", encoding="utf-8") as fh:
            for ln in fh:
                parts = ln.split()
                if len(parts) < 2:
                    continue
                row = [int(x) for x in parts[1:]]
                if row:
                    txns.append(row)
                    max_iid = max(max_iid, max(row))
    return txns, (max_iid + 1)


# ---------------------------------------------------------------------------
# RSFPGrowth mining via PAMI
# ---------------------------------------------------------------------------
def _write_pami_input(txns: list[list[int]], path: Path, sep: str = "\t") -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in txns:
            fh.write(sep.join(str(x) for x in row))
            fh.write("\n")


def _mine_rsfp(input_tsv: Path, min_sup: float, min_ratio: float, sep: str = "\t"):
    """Run PAMI RSFPGrowth. Returns dict {pattern_str: support}."""
    from PAMI.relativeFrequentPattern.basic import RSFPGrowth as alg
    obj = alg.RSFPGrowth(str(input_tsv), min_sup, min_ratio, sep=sep)
    obj.mine()
    return obj.getPatterns()


# ---------------------------------------------------------------------------
# Pattern -> edge triples
# ---------------------------------------------------------------------------
def _patterns_to_2itemset_edges(patterns: dict, n_items: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter to 2-itemsets and return symmetric COO (rows, cols, vals).

    PAMI returns ``{pattern_str: support_str_or_int}``.  Pattern_str is
    typically space-separated tokens; support_str_or_int can be int/float or
    string encoded.
    """
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    dropped = 0
    for pat, sup in patterns.items():
        # Normalise pattern into list of ints.
        if isinstance(pat, (list, tuple)):
            toks = [int(t) for t in pat]
        else:
            toks = [int(t) for t in str(pat).strip().split() if t]
        if len(toks) != 2:
            continue
        i, j = toks
        if i == j or i < 0 or j < 0 or i >= n_items or j >= n_items:
            dropped += 1
            continue
        try:
            w = float(sup)
        except (TypeError, ValueError):
            w = float(str(sup).strip().split()[0])
        rows.extend([i, j])
        cols.extend([j, i])
        vals.extend([w, w])
    if dropped:
        print(f"[rsfp] dropped {dropped} out-of-range 2-itemset patterns")
    if not rows:
        return (np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32))
    return (np.asarray(rows, np.int64), np.asarray(cols, np.int64), np.asarray(vals, np.float32))


# ---------------------------------------------------------------------------
# Blend + top-k
# ---------------------------------------------------------------------------
def _row_max_normalise(M: sp.csr_matrix) -> sp.csr_matrix:
    """Divide each row by its max non-zero entry (in-place friendly)."""
    if M.nnz == 0:
        return M
    row_max = np.asarray(M.max(axis=1).todense()).ravel()
    row_max[row_max == 0] = 1.0
    inv = 1.0 / row_max
    D = sp.diags(inv)
    return (D @ M).tocsr()


def _topk_per_row(M: sp.csr_matrix, k: int) -> sp.csr_matrix:
    """Keep top-k entries per row of a CSR matrix (by value)."""
    if k <= 0 or M.nnz == 0:
        return M
    M = M.tocsr()
    new_rows, new_cols, new_vals = [], [], []
    for r in range(M.shape[0]):
        s, e = M.indptr[r], M.indptr[r + 1]
        if e - s <= k:
            new_rows.extend([r] * (e - s))
            new_cols.extend(M.indices[s:e].tolist())
            new_vals.extend(M.data[s:e].tolist())
            continue
        idx = np.argpartition(-M.data[s:e], k)[:k]
        new_rows.extend([r] * k)
        new_cols.extend(M.indices[s:e][idx].tolist())
        new_vals.extend(M.data[s:e][idx].tolist())
    return sp.csr_matrix(
        (np.asarray(new_vals, np.float32),
         (np.asarray(new_rows, np.int64), np.asarray(new_cols, np.int64))),
        shape=M.shape,
    )


def _symmetrise(M: sp.csr_matrix) -> sp.csr_matrix:
    return ((M + M.T) * 0.5).tocsr()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_cache", type=Path, required=True,
                    help="Path to the P6.4 interest_tree_<dataset>.npz.")
    ap.add_argument("--dataset", type=str, default="Clothing")
    ap.add_argument("--data_dir", type=Path, default=Path("../data"))
    ap.add_argument("--core", type=int, default=5)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.10, 0.20, 0.40])
    ap.add_argument("--min_sup", type=float, default=20.0,
                    help="PAMI RSFPGrowth minSup (absolute item count).")
    ap.add_argument("--min_ratio", type=float, default=0.4,
                    help="PAMI RSFPGrowth minRatio (relative frequent).")
    ap.add_argument("--output_prefix", type=Path,
                    default=Path("results/interest_tree_clothing_rsfp"))
    ap.add_argument("--work_dir", type=Path,
                    default=Path("./results/_rsfp_work"))
    ap.add_argument("--skip_mining_if_cached", type=int, default=1,
                    help="If patterns.pkl already exists, reuse it.")
    args = ap.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    # 1) Load base cache.
    print(f"[rsfp] loading base cache: {args.base_cache}")
    base = dict(np.load(args.base_cache, allow_pickle=False))
    n_items = int(base["n_items"])
    knn_k = int(base["knn_k_cooc"])
    print(f"[rsfp] base: n_items={n_items} knn_k_cooc={knn_k} nnz={len(base['cooc_vals'])}")

    # 2) Build M_orig (row-max-normalised for scale-matching with M_rsfp).
    M_orig = sp.coo_matrix(
        (base["cooc_vals"].astype(np.float32),
         (base["cooc_rows"].astype(np.int64), base["cooc_cols"].astype(np.int64))),
        shape=(n_items, n_items),
    ).tocsr()
    M_orig = _row_max_normalise(M_orig)

    # 3) Load transactions + write PAMI input.
    train_path = _resolve_train_path(args.data_dir, args.dataset, args.core)
    print(f"[rsfp] loading transactions: {train_path}")
    txns, n_items_data = _load_transactions(train_path)
    if n_items_data > n_items:
        raise SystemExit(
            f"[rsfp] data has n_items={n_items_data} > cache n_items={n_items}"
        )
    print(f"[rsfp] loaded {len(txns)} transactions, n_items_data={n_items_data}")

    pami_input = args.work_dir / f"rsfp_txn_{args.dataset}.tsv"
    patterns_pkl = args.work_dir / f"rsfp_patterns_{args.dataset}_ms{int(args.min_sup)}_mr{args.min_ratio}.pkl"

    if args.skip_mining_if_cached and patterns_pkl.exists():
        import pickle
        print(f"[rsfp] reusing cached patterns: {patterns_pkl}")
        with patterns_pkl.open("rb") as fh:
            patterns = pickle.load(fh)
    else:
        _write_pami_input(txns, pami_input, sep="\t")
        print(f"[rsfp] wrote PAMI input: {pami_input}")
        t0 = time.time()
        print(f"[rsfp] mining RSFPGrowth minSup={args.min_sup} minRatio={args.min_ratio} ...")
        patterns = _mine_rsfp(pami_input, args.min_sup, args.min_ratio, sep="\t")
        print(f"[rsfp] mined {len(patterns)} patterns in {time.time() - t0:.1f}s")
        import pickle
        with patterns_pkl.open("wb") as fh:
            pickle.dump(patterns, fh)

    # 4) Build M_rsfp (2-itemset edges).
    rows, cols, vals = _patterns_to_2itemset_edges(patterns, n_items)
    print(f"[rsfp] 2-itemset edges: {len(rows) // 2} unique pairs (symmetric COO nnz={len(rows)})")
    if len(rows) == 0:
        raise SystemExit(
            "[rsfp] no 2-itemset patterns found. Lower --min_sup or --min_ratio."
        )
    M_rsfp = sp.coo_matrix(
        (vals, (rows, cols)), shape=(n_items, n_items)
    ).tocsr()
    M_rsfp = _row_max_normalise(M_rsfp)

    # 5) Blend + top-k per alpha; save.
    for alpha in args.alphas:
        M_blend = ((1.0 - alpha) * M_orig + alpha * M_rsfp).tocsr()
        M_blend = _symmetrise(M_blend)
        M_blend = _topk_per_row(M_blend, knn_k)
        coo = M_blend.tocoo()
        alpha_tag = f"a{int(round(alpha * 100)):03d}"
        out_path = Path(f"{args.output_prefix}_{alpha_tag}.npz")

        kwargs = {
            "cooc_rows": coo.row.astype(np.int64),
            "cooc_cols": coo.col.astype(np.int64),
            "cooc_vals": coo.data.astype(np.float32),
            "knn_k_cooc": np.int32(knn_k),
            "knn_k_mod": base["knn_k_mod"],
            "n_order": base["n_order"],
            "gamma": base["gamma"],
            "tau": base["tau"],
            "n_items": np.int32(n_items),
        }
        # Copy tree_* fields verbatim (BFS is over anchors, not weights).
        for k in ("tree_anchors", "tree_neighbours", "tree_orders", "tree_weights"):
            if k in base:
                kwargs[k] = base[k]

        np.savez_compressed(out_path, **kwargs)
        print(f"[rsfp] wrote {out_path}  (blend alpha={alpha:.2f}, nnz={coo.nnz})")

    print("[rsfp] done.")


if __name__ == "__main__":
    main()
