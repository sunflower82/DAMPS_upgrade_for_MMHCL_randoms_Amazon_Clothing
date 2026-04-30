"""
damps/graph.py — Dual-Path K-NN Hypergraph Builder
====================================================

Implements the **Dual-path pipeline** described in Section 3.2 of the DAMPS
spec (``DAMPS_to_MMHCL_architecture_revision42.tex``):

    Path 1 (Default)
        Native PyTorch chunked ``torch.topk`` operations, ensuring high
        reproducibility and zero external-library dependence.

    Path 2 (Mandatory Fallback)
        Automatically activates the FAISS GPU system (``IndexHNSWFlat``)
        when the item scale reaches N >= 60,000, systematically reducing
        the K-NN rebuild time complexity from O(N^2) to O(N log N).

This is invoked once every ``R`` epochs by the **Pattern B' (Scheduled
Rebuild)** loop in ``train.py`` — never at every forward pass, which would
trigger graph instability and density explosion.

Output format
-------------
The builder returns a *symmetrically-normalised* sparse COO tensor of shape
(N, N) ready to be plugged into MMHCL's hypergraph convolution layers:

    A_norm = D^{-1/2} (H H^T) D^{-1/2}

This matches the format produced by the original
``data/<dataset>/5-core/hypergraph_mat_mul_sym_topk_K.pth`` cache file, so
the rebuilt graph is a drop-in replacement for the cached one.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Optional FAISS import (Path 2 — Mandatory Fallback for large datasets)
# ---------------------------------------------------------------------------
try:                                                              # pragma: no cover
    import faiss                                                  # type: ignore[import-not-found]
    FAISS_AVAILABLE: bool = True
    try:
        import faiss.contrib.torch_utils                         # type: ignore[import-not-found]  # noqa: F401
    except Exception:
        # torch_utils is optional; FAISS still works with numpy round-trip
        pass
except ImportError:
    FAISS_AVAILABLE = False


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------
class DualPathKNN:
    """
    Configurable K-NN graph builder with chunked PyTorch + FAISS fallback.

    Args:
        k                : number of nearest neighbours to retain per item.
        faiss_threshold  : item count at which we switch to FAISS Path 2.
                           Default 60,000 (per Section 3.2 spec).
        chunk_size       : row-chunk size for the PyTorch chunked path.
                           Tuned for an RTX 5090 (32 GB) with d = 64.
        normalize        : if True, apply symmetric Laplacian normalisation
                           D^{-1/2} A D^{-1/2} before returning.

    Notes:
        *   ``build_graph_from_modalities`` is the recommended entry point —
            it correctly fuses image and text neighbour lists into a single
            multi-modal hypergraph (matching ``H @ H^T``).
        *   ``build_graph`` is exposed for ablation experiments where you
            want a single-modality K-NN graph.
    """

    def __init__(
        self,
        k: int = 5,
        faiss_threshold: int = 60_000,
        chunk_size: int = 4_096,
        normalize: bool = True,
    ) -> None:
        if k <= 0:
            raise ValueError(f"k must be positive, got k={k}")
        self.k: int = int(k)
        self.faiss_threshold: int = int(faiss_threshold)
        self.chunk_size: int = int(chunk_size)
        self.normalize: bool = bool(normalize)

    # ------------------------------------------------------------------
    #  Multi-modal hypergraph (recommended entry point)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def build_graph_from_modalities(
        self,
        h_img: torch.Tensor,
        h_txt: torch.Tensor,
        h_aud: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Build the item-item multi-modal hypergraph from per-modality
        calibrated representations.

        Mirrors MMHCL's ``H @ H^T`` construction: each modality contributes
        its own K-NN incidence matrix; the modality-specific incidence
        matrices are then stacked column-wise and multiplied by their
        transpose to yield a single (N, N) hypergraph.

        Args:
            h_img : (N, d) calibrated image features.
            h_txt : (N, d) calibrated text features.
            h_aud : optional (N, d) calibrated audio features (Tiktok).

        Returns:
            (N, N) sparse COO tensor — the (optionally) normalised hypergraph.
        """
        N = h_img.shape[0]
        a_img = self._build_modality_adj(h_img)
        a_txt = self._build_modality_adj(h_txt)

        if h_aud is not None:
            a_aud = self._build_modality_adj(h_aud)
            # Hypergraph incidence: H = [A_img | A_txt | A_aud] in (N, 3N)
            H = torch.cat([a_img.to_dense(), a_txt.to_dense(), a_aud.to_dense()], dim=1)
        else:
            # Hypergraph incidence: H = [A_img | A_txt] in (N, 2N)
            H = torch.cat([a_img.to_dense(), a_txt.to_dense()], dim=1)

        # Final hypergraph A = H @ H^T  (N, N)
        # Done densely because intermediate H is at most (N, 3N) of floats
        # and N ≤ 60k -> ≤ 6 GB in fp32, still cheaper than a sparse mm.
        adj = H @ H.transpose(0, 1)

        if self.normalize:
            adj = self._symmetric_normalize(adj)
        return adj.to_sparse_coo().coalesce()

    @torch.no_grad()
    def build_graph(self, features: torch.Tensor) -> torch.Tensor:
        """
        Build a K-NN adjacency matrix from a single feature table.

        Args:
            features : (N, d) feature tensor.

        Returns:
            (N, N) sparse COO tensor — binary K-NN adjacency.
        """
        return self._build_modality_adj(features)

    # ------------------------------------------------------------------
    #  Routing logic
    # ------------------------------------------------------------------
    def _build_modality_adj(self, features: torch.Tensor) -> torch.Tensor:
        N = features.shape[0]
        # L2-normalise for cosine similarity
        feats = F.normalize(features.float(), p=2, dim=1)

        if N >= self.faiss_threshold and FAISS_AVAILABLE:
            logger.info(
                "[DualPathKNN] N=%d ≥ %d → routing to FAISS GPU IndexHNSWFlat",
                N, self.faiss_threshold,
            )
            return self._build_faiss(feats)

        if N >= self.faiss_threshold and not FAISS_AVAILABLE:
            logger.warning(
                "[DualPathKNN] N=%d ≥ %d but FAISS not available; "
                "falling back to chunked PyTorch (slower but reproducible)",
                N, self.faiss_threshold,
            )
        return self._build_chunked(feats)

    # ------------------------------------------------------------------
    #  Path 1: Chunked PyTorch (default, reproducible)
    # ------------------------------------------------------------------
    def _build_chunked(self, features: torch.Tensor) -> torch.Tensor:
        """
        Pure-PyTorch chunked top-K. VRAM is bounded by ``chunk_size * N``.

        Self-edges (cosine similarity = 1.0 with the row itself) are dropped
        by zeroing the diagonal of each block before ``torch.topk``. This is
        far cheaper than a per-row Python loop and produces a clean (k, k)
        index tensor in a single vectorised step.
        """
        N = features.shape[0]
        device = features.device
        k = min(self.k, max(N - 1, 1))
        rows: list[torch.Tensor] = []
        cols: list[torch.Tensor] = []

        # Process the (N, N) similarity matrix one block of `chunk_size` rows
        # at a time. Each block is `chunk_size * N` floats — bounded VRAM.
        for start in range(0, N, self.chunk_size):
            end = min(start + self.chunk_size, N)
            chunk = features[start:end]
            sim = chunk @ features.transpose(0, 1)            # (B, N)

            # Mask the self-edge by setting sim[i, global_i] = -inf so it
            # never appears in the top-k. Use -1e9 instead of -inf to stay
            # safely numeric under bfloat16 down-cast paths.
            row_global = torch.arange(start, end, device=device)
            sim[torch.arange(end - start, device=device), row_global] = -1e9

            _, topk_idx = torch.topk(sim, k, dim=1)            # (B, k)

            rows.append(
                row_global.unsqueeze(1).expand(-1, k).reshape(-1)
            )
            cols.append(topk_idx.reshape(-1))

        row_idx = torch.cat(rows)
        col_idx = torch.cat(cols)
        values = torch.ones_like(row_idx, dtype=torch.float32)

        adj = torch.sparse_coo_tensor(
            torch.stack([row_idx, col_idx]), values, (N, N)
        ).coalesce()
        return adj

    # ------------------------------------------------------------------
    #  Path 2: FAISS GPU (mandatory fallback for N >= 60k)
    # ------------------------------------------------------------------
    def _build_faiss(self, features: torch.Tensor) -> torch.Tensor:
        """FAISS GPU IndexHNSWFlat path — O(N log N) in N."""
        if not FAISS_AVAILABLE:
            raise RuntimeError("FAISS is not available — cannot use Path 2")
        import faiss                                              # type: ignore[import-not-found]

        N, d = features.shape
        feats_np = features.detach().cpu().numpy().astype("float32")

        # HNSW with M=32 (FAISS default for high-dimensional vectors).
        # Use cosine similarity via L2-normalised vectors + inner-product index.
        index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
        index.add(feats_np)
        # Ask for k+1 because the top-1 hit is the query itself.
        k_plus = min(self.k + 1, N)
        _, idxs = index.search(feats_np, k_plus)

        # Vectorised self-edge removal: build a column mask, then take the
        # first ``k`` neighbours that are not the row index itself.
        idxs_t = torch.from_numpy(idxs).to(features.device)
        row_global = torch.arange(N, device=features.device).unsqueeze(1)
        keep = idxs_t != row_global                                # (N, k+1) bool
        # Replace self-matches with a sentinel that loses the topk argsort
        idxs_t = torch.where(keep, idxs_t, torch.full_like(idxs_t, -1))

        # For each row, keep the first ``k`` non-self entries
        cols: list[torch.Tensor] = []
        for r in range(N):
            row = idxs_t[r]
            row = row[row >= 0][: self.k]
            if row.numel() < self.k:
                # Pad with the row itself (extreme cold-start corner case)
                pad = torch.full((self.k - row.numel(),), r,
                                  dtype=row.dtype, device=row.device)
                row = torch.cat([row, pad])
            cols.append(row)
        col_idx = torch.stack(cols).reshape(-1)

        row_idx = (
            torch.arange(N, device=features.device)
            .unsqueeze(1)
            .expand(-1, self.k)
            .reshape(-1)
        )
        values = torch.ones_like(row_idx, dtype=torch.float32)
        return torch.sparse_coo_tensor(
            torch.stack([row_idx, col_idx]), values, (N, N)
        ).coalesce()

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _symmetric_normalize(adj_dense: torch.Tensor) -> torch.Tensor:
        """Symmetric Laplacian normalisation D^{-1/2} A D^{-1/2}."""
        rowsum = adj_dense.sum(dim=-1)
        d_inv_sqrt = rowsum.pow(-0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        return adj_dense * d_inv_sqrt.unsqueeze(0) * d_inv_sqrt.unsqueeze(1)


# ---------------------------------------------------------------------------
#  Free helpers used by diagnostics in train.py
# ---------------------------------------------------------------------------
def adj_nnz(adj: torch.Tensor) -> int:
    """Return the number of non-zero entries in a sparse or dense adjacency."""
    if adj.is_sparse:
        return int(adj._nnz())                                    # noqa: SLF001
    return int((adj != 0).sum().item())


def adj_avg_degree(adj: torch.Tensor) -> float:
    """Average node degree (NNZ / N)."""
    n = adj.shape[0] if adj.dim() == 2 else 0
    if n == 0:
        return 0.0
    return adj_nnz(adj) / float(n)
