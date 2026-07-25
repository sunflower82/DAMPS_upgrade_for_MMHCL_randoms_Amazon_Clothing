"""codes/damps_tamer.py -- P6.4 DAMPS-NRDMC-lite + TAMER Interest Tree fusion.
================================================================================

Extends the existing DAMPS-NRDMC-lite pipeline with the TAMER interest-tree
augmented similarity ``S_tilde`` (Eq. 8-9, Meng et al. MM'25).  Concretely,
this module contributes:

* ``build_augmented_modality_graph`` -- combines
  - the modality similarity produced by preprocess_macp (view v/p/z), and
  - the interest-tree bonus derived from the item-item co-occurrence
    graph S^c (see codes/roaring_cooc.py + codes/interest_tree.py),

  and returns a *symmetric, normalised* (n_items x n_items) sparse adjacency
  ready for the homogeneous GCN step (codes/scatter_gcn.py::ScatterGCN).

* ``TAMERAugmentedNRDMCLite`` -- thin wrapper around the existing
  ``NRDMCLiteView`` from damps/nrdmc_lite.py.  Overrides the item-item
  input graph passed to the modality LightGCN step; NRDMC view logic
  (SAV / IAV / PTV / InfoNCE) is untouched to preserve bit-for-bit compat
  with the P6.3 baseline when ``alpha_interest = 0``.

* ``load_interest_cache`` / ``save_interest_cache`` -- .npz round-trip for
  the pre-computed ``item_graph_dict`` (co-occurrence top-k) so training
  runs don't rebuild the tidset intersections on every launch.

Design notes
------------
* ``alpha_interest = 0`` recovers the exact P6.3 modality graph
  bit-for-bit (the augmentation term is skipped entirely).  This gives us
  an unambiguous ablation control cell for the P6.4 grid.
* The augmentation runs on CPU because ``scipy.sparse`` and
  ``build_interest_tree`` are pure-python; on Amazon Clothing (n=23k,
  cooc_topk=20) it takes ~90 s single-threaded, cached to disk.
* Devices: ``TAMERAugmentedNRDMCLite`` accepts a ``device`` arg; the
  augmented ``edge_index / edge_weight`` are moved once and reused.

Reference implementation
------------------------
The pruning schedule, gamma / tau exponent and Eq. 9 alpha_hg fusion match
Z-last-ONE/TAMER models/tamer.py::{build_session_tree,
find_weighted_n_order_relationships} verbatim -- see
codes/interest_tree.py for the ports.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from codes.interest_tree import (
    build_interest_tree,
    build_weighted_binary_relations,
    fuse_augmented,
)
from codes.knn_hnsw import knn_topk_exact
from codes.scatter_gcn import ScatterGCN, build_sym_norm_edges


# ---------------------------------------------------------------------------
# Cache round-trip
# ---------------------------------------------------------------------------
def save_interest_cache(
    path: str | Path,
    *,
    cooc_rows: np.ndarray,
    cooc_cols: np.ndarray,
    cooc_vals: np.ndarray,
    knn_k_cooc: int,
    knn_k_mod: int,
    n_order: int,
    gamma: float,
    tau: float,
    n_items: int,
) -> None:
    """Persist the pre-computed interest graph + hyper-params to .npz."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        p,
        cooc_rows=cooc_rows.astype(np.int64),
        cooc_cols=cooc_cols.astype(np.int64),
        cooc_vals=cooc_vals.astype(np.float32),
        knn_k_cooc=np.int32(knn_k_cooc),
        knn_k_mod=np.int32(knn_k_mod),
        n_order=np.int32(n_order),
        gamma=np.float32(gamma),
        tau=np.float32(tau),
        n_items=np.int32(n_items),
    )


def load_interest_cache(path: str | Path) -> Dict[str, np.ndarray]:
    """Restore what ``save_interest_cache`` wrote."""
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


# ---------------------------------------------------------------------------
# Sparse augmented modality graph builder
# ---------------------------------------------------------------------------
def _topk_sym_adj_from_feats(
    feats: np.ndarray,
    k: int,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return COO top-k adjacency (i, j, sim) for L2-normalised feats."""
    return knn_topk_exact(feats, k=k, device=device, chunk=4096)


def build_augmented_modality_graph(
    modality_feats: Dict[str, np.ndarray],
    cache: Dict[str, np.ndarray],
    *,
    alphas: Dict[str, float],
    alpha_interest: float,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Return ``(edge_index, edge_weight, n_items)`` for the augmented graph.

    Args:
        modality_feats : {view_name: (n, d) float32 array}.  Typical view_name
                         values in this codebase: ``{'v', 'p', 'z'}`` -- the
                         raw image plus the MACP PCA/ICA and ZCA text streams.
        cache          : output of ``load_interest_cache`` -- provides
                         cooc_rows/cols/vals + hyper-params.
        alphas         : Eq. 9 mixing weights for each modality view.
                         Missing keys default to 0.
        alpha_interest : Eq. 9 mixing weight for the interest (S^c) branch;
                         set to 0.0 to recover the P6.3 modality-only graph
                         (bit-for-bit ablation control).
        device         : torch device for the exact k-NN step.
    """
    knn_k_mod = int(cache["knn_k_mod"])
    knn_k_cooc = int(cache["knn_k_cooc"])
    n_order = int(cache["n_order"])
    gamma = float(cache["gamma"])
    tau = float(cache["tau"])
    n_items = int(cache["n_items"])

    graph = build_weighted_binary_relations(
        cache["cooc_rows"], cache["cooc_cols"], cache["cooc_vals"], knn_k_cooc
    )

    per_view_edges: Dict[str, sp.csr_matrix] = {}
    for name, feats in modality_feats.items():
        if float(alphas.get(name, 0.0)) == 0.0:
            continue
        rows, cols, vals = _topk_sym_adj_from_feats(feats, knn_k_mod, device=device)
        base = sp.csr_matrix(
            (vals.astype(np.float32) / 2.0, (rows, cols)),
            shape=(n_items, n_items),
        )
        # Interest tree bonus per anchor -- accumulate into LIL for random access.
        lil = base.tolil()
        if alpha_interest > 0.0:
            # Only augment rows that actually have anchor entries in ``graph``
            # -- items with no co-occurrence neighbours contribute 0 bonus.
            for i in graph.keys():
                tree = build_interest_tree(graph, i, n_order)
                for order, level in tree.items():
                    coef = gamma * float(np.exp(-(order - 1)))
                    for j, w in level:
                        s_ij = base[i, j]
                        if s_ij == 0.0:
                            continue
                        lil[i, j] = lil[i, j] + coef * (w ** tau) * s_ij
        per_view_edges[name] = lil.tocsr()

    # Interest-only branch (S^c symmetric adjacency, no modality gating).
    if alpha_interest > 0.0:
        s_c = sp.csr_matrix(
            (cache["cooc_vals"].astype(np.float32), (cache["cooc_rows"], cache["cooc_cols"])),
            shape=(n_items, n_items),
        )
        # Row-normalise co-occurrence weights so they live on comparable scale.
        row_sum = np.asarray(s_c.sum(axis=1)).flatten()
        row_sum[row_sum == 0.0] = 1.0
        d_inv = sp.diags(1.0 / row_sum)
        s_c = d_inv @ s_c
    else:
        s_c = None

    # Weighted fusion (Eq. 9).
    n = n_items
    fused = sp.csr_matrix((n, n), dtype=np.float32)
    for name, m in per_view_edges.items():
        fused = fused + float(alphas[name]) * m
    if s_c is not None:
        fused = fused + alpha_interest * s_c

    fused = fused.tocoo()
    edge_index, edge_weight = build_sym_norm_edges(
        fused.row.astype(np.int64),
        fused.col.astype(np.int64),
        fused.data.astype(np.float32),
        n_items,
    )
    return edge_index.to(device), edge_weight.to(device), n_items


# ---------------------------------------------------------------------------
# Thin nn.Module wrapper -- item-side GCN over the augmented graph
# ---------------------------------------------------------------------------
class TAMERAugmentedNRDMCLite(nn.Module):
    """Runs the homogeneous LightGCN step over the TAMER-augmented graph.

    Composed with the existing SAV/IAV/PTV modules (damps/nrdmc_lite.py) by
    the driver: the caller adds ``item_rep + tamer_module(item_id_emb)``
    before the BPR head, matching TAMER Eq. e_i = i_rep + i_rep^{hg}.
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        *,
        n_layers: int = 2,
    ):
        super().__init__()
        # Register as buffers so they move with .to(device) and are not learnt.
        self.register_buffer("edge_index", edge_index, persistent=False)
        self.register_buffer("edge_weight", edge_weight, persistent=False)
        self.gcn = ScatterGCN(n_layers=n_layers)

    def forward(self, item_emb: torch.Tensor) -> torch.Tensor:
        return self.gcn(item_emb, self.edge_index, self.edge_weight)


__all__ = [
    "save_interest_cache",
    "load_interest_cache",
    "build_augmented_modality_graph",
    "TAMERAugmentedNRDMCLite",
    "fuse_augmented",  # re-export for convenience
]
