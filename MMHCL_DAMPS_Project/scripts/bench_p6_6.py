"""scripts/bench_p6_6.py -- P6.6 speedup micro-benchmark.
==========================================================

Runs three isolated micro-benchmarks and emits a JSON summary:

1. **k-NN top-k**   HNSW vs. exact chunked torch.mm on synthetic modality
                    features (default 30k items x 128 dim, k=10).
2. **Co-occurrence**  roaring bitmap vs. python-set tidset intersection
                    on a synthetic Zipfian-degree user-item log.
3. **GCN propagation**  scatter_add vs. sparse.mm on the (n_items x n_items)
                    normalised modality adjacency built in step (1).

Usage
-----
::

    python scripts/bench_p6_6.py \\
        --n_items 30000 \\
        --dim 128 \\
        --k 10 \\
        --trials 3 \\
        --output results/p6_6_bench.json

Determinism
-----------
* HNSW uses ``num_threads=1, random_seed=100`` and the exact path is
  deterministic by construction (float32 topk).  Numerical recall is
  reported but not asserted (HNSW is approximate by design).
* The synthetic dataset is seeded with ``--seed`` (default 100).

Overrides for the notebook driver (cell §9.21)
----------------------------------------------
Environment variables ``P6_6_N_ITEMS``, ``P6_6_DIM``, ``P6_6_K``,
``P6_6_TRIALS``, ``P6_6_OUTPUT``, ``P6_6_DEVICE`` override the CLI flags.
This lets the notebook cell drop overrides in without editing the script.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Make ``codes/`` importable when run from MMHCL_DAMPS_Project/
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from codes.knn_hnsw import timed_knn, _HAVE_HNSW  # noqa: E402
from codes.roaring_cooc import timed_cooc, _HAVE_ROARING  # noqa: E402
from codes.scatter_gcn import (  # noqa: E402
    ScatterGCN,
    build_sym_norm_edges,
    sparse_mm_lightgcn,
    _HAVE_SCATTER,
)


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------
def synth_feats(n_items: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_items, dim)).astype(np.float32)


def synth_interactions(n_users: int, n_items: int, n_edges: int, seed: int):
    """Zipfian user * uniform item pairs -- rough Amazon Clothing shape."""
    rng = np.random.default_rng(seed)
    # Zipf s=1.2 gives median ~8 items/user, tail up to 100 -- similar to
    # the Amazon Clothing 5-core distribution.
    u_freq = rng.zipf(1.2, size=n_users).astype(np.int64)
    u_freq = np.clip(u_freq, 1, n_items - 1)
    pairs = []
    for u in range(n_users):
        picks = rng.choice(n_items, size=min(int(u_freq[u]), 200), replace=False)
        for i in picks:
            pairs.append((u, int(i)))
            if len(pairs) >= n_edges:
                break
        if len(pairs) >= n_edges:
            break
    return pairs[:n_edges]


# ---------------------------------------------------------------------------
# Benchmark blocks
# ---------------------------------------------------------------------------
def _median(times):
    return float(statistics.median(times))


def bench_knn(feats: np.ndarray, k: int, trials: int, device: str) -> dict:
    hnsw_t, exact_t = [], []
    exact_rows = exact_cols = None
    for _ in range(trials):
        gc.collect()
        t_hnsw, out_hnsw = timed_knn(feats, k, method="hnsw", num_threads=1, ef_search=64)
        hnsw_t.append(t_hnsw)
        gc.collect()
        t_exact, out_exact = timed_knn(feats, k, method="exact", device=device, chunk=4096)
        exact_t.append(t_exact)
        exact_rows, exact_cols = out_exact[0], out_exact[1]
    # recall vs. exact top-k on the last trial (approximate but adequate)
    hnsw_pairs = set(zip(out_hnsw[0].tolist(), out_hnsw[1].tolist()))
    exact_pairs = set(zip(exact_rows.tolist(), exact_cols.tolist()))
    recall = len(hnsw_pairs & exact_pairs) / max(len(exact_pairs), 1)
    return {
        "n_items": int(feats.shape[0]),
        "dim": int(feats.shape[1]),
        "k": int(k),
        "hnsw_available": bool(_HAVE_HNSW),
        "hnsw_ms_median": _median(hnsw_t) * 1e3,
        "exact_ms_median": _median(exact_t) * 1e3,
        "speedup_x": _median(exact_t) / max(_median(hnsw_t), 1e-9),
        "recall_at_k": recall,
    }


def bench_cooc(interactions, k: int, trials: int) -> dict:
    roar_t, set_t = [], []
    roar_nnz = set_nnz = 0
    for _ in range(trials):
        gc.collect()
        t_roar, out_roar = timed_cooc(interactions, k, method="roaring")
        roar_t.append(t_roar)
        roar_nnz = int(out_roar[0].shape[0])
        gc.collect()
        t_set, out_set = timed_cooc(interactions, k, method="sets")
        set_t.append(t_set)
        set_nnz = int(out_set[0].shape[0])
    return {
        "n_interactions": int(len(interactions)),
        "k": int(k),
        "roaring_available": bool(_HAVE_ROARING),
        "roaring_ms_median": _median(roar_t) * 1e3,
        "sets_ms_median": _median(set_t) * 1e3,
        "speedup_x": _median(set_t) / max(_median(roar_t), 1e-9),
        "nnz_roaring": roar_nnz,
        "nnz_sets": set_nnz,
        "nnz_match": bool(roar_nnz == set_nnz),
    }


def bench_gcn(feats: np.ndarray, k: int, n_layers: int, trials: int, device: str) -> dict:
    # Build adjacency from exact k-NN for parity across the two paths.
    from codes.knn_hnsw import knn_topk_exact

    rows, cols, vals = knn_topk_exact(feats, k, device=device, chunk=4096)
    n = feats.shape[0]
    edge_index, weight = build_sym_norm_edges(rows, cols, vals, n)
    edge_index = edge_index.to(device)
    weight = weight.to(device)

    # torch sparse tensor form for the reference
    indices = edge_index
    values = weight
    adj = torch.sparse_coo_tensor(indices, values, size=(n, n)).coalesce()

    x0 = torch.from_numpy(feats).to(device)
    scatter_mod = ScatterGCN(n_layers=n_layers).to(device)

    scat_t, mm_t = [], []
    for _ in range(trials):
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        y_scat = scatter_mod(x0, edge_index, weight)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        scat_t.append(time.perf_counter() - t0)
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        y_mm = sparse_mm_lightgcn(x0, adj, n_layers=n_layers)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        mm_t.append(time.perf_counter() - t0)
    max_abs = float((y_scat - y_mm).abs().max().item())
    return {
        "n_items": int(n),
        "n_layers": int(n_layers),
        "device": device,
        "scatter_available": bool(_HAVE_SCATTER),
        "scatter_ms_median": _median(scat_t) * 1e3,
        "sparse_mm_ms_median": _median(mm_t) * 1e3,
        "speedup_x": _median(mm_t) / max(_median(scat_t), 1e-9),
        "max_abs_diff": max_abs,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _env_or(name: str, default):
    v = os.environ.get(name)
    return type(default)(v) if v is not None else default


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_items", type=int, default=_env_or("P6_6_N_ITEMS", 30000))
    p.add_argument("--dim", type=int, default=_env_or("P6_6_DIM", 128))
    p.add_argument("--k", type=int, default=_env_or("P6_6_K", 10))
    p.add_argument("--n_layers", type=int, default=_env_or("P6_6_LAYERS", 3))
    p.add_argument("--trials", type=int, default=_env_or("P6_6_TRIALS", 3))
    p.add_argument("--seed", type=int, default=_env_or("P6_6_SEED", 100))
    p.add_argument("--n_users", type=int, default=_env_or("P6_6_N_USERS", 20000))
    p.add_argument("--n_edges", type=int, default=_env_or("P6_6_N_EDGES", 200000))
    p.add_argument("--device", type=str, default=_env_or("P6_6_DEVICE", "cpu"))
    p.add_argument(
        "--output",
        type=str,
        default=os.environ.get("P6_6_OUTPUT", str(_ROOT / "results" / "p6_6_bench.json")),
    )
    args = p.parse_args()

    print(f"[P6.6-bench] cfg={vars(args)}")
    torch.manual_seed(args.seed)

    feats = synth_feats(args.n_items, args.dim, args.seed)
    inter = synth_interactions(args.n_users, args.n_items, args.n_edges, args.seed)

    print("[P6.6-bench] (1/3) k-NN top-k ...")
    knn_res = bench_knn(feats, args.k, args.trials, args.device)
    print(
        f"    HNSW   {knn_res['hnsw_ms_median']:8.1f} ms  |  "
        f"Exact {knn_res['exact_ms_median']:8.1f} ms  |  "
        f"speedup {knn_res['speedup_x']:5.2f}x  |  recall@k {knn_res['recall_at_k']:.3f}"
    )

    print("[P6.6-bench] (2/3) co-occurrence ...")
    cooc_res = bench_cooc(inter, args.k, args.trials)
    print(
        f"    Roaring {cooc_res['roaring_ms_median']:8.1f} ms  |  "
        f"Sets  {cooc_res['sets_ms_median']:8.1f} ms  |  "
        f"speedup {cooc_res['speedup_x']:5.2f}x  |  nnz_match={cooc_res['nnz_match']}"
    )

    print("[P6.6-bench] (3/3) GCN propagation ...")
    gcn_res = bench_gcn(feats, args.k, args.n_layers, args.trials, args.device)
    print(
        f"    Scatter {gcn_res['scatter_ms_median']:8.1f} ms  |  "
        f"SpMM  {gcn_res['sparse_mm_ms_median']:8.1f} ms  |  "
        f"speedup {gcn_res['speedup_x']:5.2f}x  |  max|diff|={gcn_res['max_abs_diff']:.2e}"
    )

    summary = {
        "config": vars(args),
        "knn": knn_res,
        "cooc": cooc_res,
        "gcn": gcn_res,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[P6.6-bench] wrote {out_path}")

    # Pretty per-op speedup table for the notebook
    print()
    print("| op        | fast (ms) | ref (ms)  | speedup |")
    print("|-----------|-----------|-----------|---------|")
    print(
        f"| k-NN top-k | {knn_res['hnsw_ms_median']:9.1f} | "
        f"{knn_res['exact_ms_median']:9.1f} | {knn_res['speedup_x']:6.2f}x |"
    )
    print(
        f"| cooc top-k | {cooc_res['roaring_ms_median']:9.1f} | "
        f"{cooc_res['sets_ms_median']:9.1f} | {cooc_res['speedup_x']:6.2f}x |"
    )
    print(
        f"| GCN prop   | {gcn_res['scatter_ms_median']:9.1f} | "
        f"{gcn_res['sparse_mm_ms_median']:9.1f} | {gcn_res['speedup_x']:6.2f}x |"
    )


if __name__ == "__main__":
    main()
