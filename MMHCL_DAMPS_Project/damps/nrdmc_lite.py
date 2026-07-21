"""
damps/nrdmc_lite.py -- Branch A' (NRDMC-lite) view generators + contrastive loss.
==================================================================================

Implements the design-doc §8.2 spec (DAMPS_to_MMHCL_architecture_revision55):
  * Light-redesign upgrade of the SimGCL view path.
  * Adds two LEARNABLE view generators over the observed U-I bipartite edges:
      - SAV (Structure-Aware View, NRDMC IPM 2026 Eq. 14)
      - IAV (Importance-Aware View,          Eq. 16)
    (PTV / Prototype-Aware View is DROPPED per §8.2 to keep the fix compact.)
  * Adaptive fusion of the two views (Eq. 17-19).
  * One-shot LightGCN pass over the resulting weighted contrastive graph
    to obtain view embeddings E_bar_u, E_bar_i.
  * Batch-N InfoNCE (Yu et al. SIGIR 2022, App. A.3) between:
        original ego-post-GCN embeddings  ê  (a.k.a. E_hat)
        contrastive-view embeddings       ē  (a.k.a. E_bar)
    on both user and item sides, giving L_mv = 0.5 * (L_mv_user + L_mv_item).

Runtime cost (Amazon Clothing, |E|=197 338, d=128, L=3, B=2048):
  * SAV+IAV+fusion : ~ 100 M ops        (< 1 % of an epoch)
  * View GCN       : ~ 75 M ops         (< 1 % of an epoch)
  * Batch-N InfoNCE: identical to the current SimGCL branch A path.

Design choices vs the paper:
  * We DROP PTV (K prototypes + soft assignment) -- design-doc §8.2 explicit.
  * We reuse the current `_lightgcn_propagate` normalisation semantics
    (symmetric D^{-1/2} A D^{-1/2}) on the LEARNED edge weights.
  * We DO NOT clamp the learned weights, but we normalise them via
    the graph Laplacian, which is empirically stable in the 197 k-edge regime.
  * The view is refreshed EVERY BATCH (unlike SimGCL's every-k-epochs cache);
    the cost is dominated by the InfoNCE matmul, so the extra 200 M ops per
    batch are lost in the noise. This matters for gradient flow into the
    SAV/IAV learnable parameters (`g`, `W_fuse`, `b_fuse`).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# View generator module
# ---------------------------------------------------------------------------
class NRDMCLiteView(nn.Module):
    """SAV + IAV view generators, adaptive fusion, and view LightGCN.

    Args:
        n_users : # user nodes.
        n_items : # item nodes.
        embed_dim: embedding dimensionality (matches the trunk).
        n_layers : # of LightGCN steps to propagate over the contrastive graph.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.embed_dim = int(embed_dim)
        self.n_layers = int(n_layers)

        # IAV attention vector g \in R^d  (Eq. 16 of NRDMC IPM 2026).
        # Xavier-like init: unit norm keeps β_{u,i} = σ((g·e_u)(g·e_i)) in [~0.5]
        # at init when e_u, e_i are L2-normalised.
        g = torch.randn(embed_dim) / (embed_dim ** 0.5)
        self.g = nn.Parameter(g)

        # Adaptive-fusion transform (Eq. 17): scalar affine per view.
        # A single (W, b) shared across the K views is the paper's default
        # (it explicitly *shares* W to encourage the shared/specific split).
        self.W_fuse = nn.Parameter(torch.ones(1))
        self.b_fuse = nn.Parameter(torch.zeros(1))

        # Edge topology cache (registered on first forward()).
        self._edge_u: Optional[torch.Tensor] = None
        self._edge_i: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    #  Edge extraction (once per model)
    # ------------------------------------------------------------------
    def _ensure_edges(self, ui_mat: torch.Tensor) -> None:
        """Extract the observed U-I edge list from the sparse bipartite adj.

        The PACER trunk builds ``UI_mat`` as an (n_u+n_i, n_u+n_i) sparse
        block-antidiagonal matrix with the top-right block = normalised R.
        We recover the raw (u, i) pairs by reading the upper block.
        """
        if self._edge_u is not None:
            return
        if not ui_mat.is_sparse:
            raise RuntimeError("NRDMCLiteView requires a sparse UI_mat input.")
        indices = ui_mat.coalesce().indices()          # (2, 2E)
        rows, cols = indices[0], indices[1]
        # Upper-right block: rows < n_users  and  cols >= n_users
        mask = (rows < self.n_users) & (cols >= self.n_users)
        edge_u = rows[mask].contiguous()               # (E,)  in [0, n_users)
        edge_i = (cols[mask] - self.n_users).contiguous()  # (E,)  in [0, n_items)
        # Register as buffers so .to(device) migrates them with the module.
        self.register_buffer("_edge_u_buf", edge_u, persistent=False)
        self.register_buffer("_edge_i_buf", edge_i, persistent=False)
        self._edge_u = self._edge_u_buf
        self._edge_i = self._edge_i_buf

    # ------------------------------------------------------------------
    #  SAV + IAV + adaptive fusion
    # ------------------------------------------------------------------
    def _edge_weights(
        self,
        e_u: torch.Tensor,
        e_i: torch.Tensor,
    ) -> torch.Tensor:
        """Compute learned edge weight w_final over the observed U-I edges.

        Args:
            e_u : (n_users, d) POST-GCN, L2-normalised user embeddings (Ẽ_u).
            e_i : (n_items, d) POST-GCN, L2-normalised item embeddings (Ẽ_i).

        Returns:
            w_final : (E,) tensor of per-edge weights, no clamp.

        Note:
            Forced to float32 so bf16/fp16 AMP cannot mix dtypes inside
            ``scatter_add_`` / ``scatter_reduce_`` (PyTorch requires
            self.dtype == src.dtype).
        """
        # fp32 for AMP-safe scatter reductions + stable softmax.
        eu = e_u[self._edge_u].float()                # (E, d)
        ei = e_i[self._edge_i].float()                # (E, d)
        g = self.g.float()
        w_fuse = self.W_fuse.float()
        b_fuse = self.b_fuse.float()

        # -- SAV (Eq. 14) --------------------------------------------------
        # NRDMC writes SAV[u,i] = σ(Ẽ_u ⊙ Ẽ_i). Since SAV must be a scalar
        # edge weight, we read this as σ(<Ẽ_u, Ẽ_i>) = σ of the dot product.
        sav = torch.sigmoid((eu * ei).sum(dim=-1))    # (E,)

        # -- IAV (Eq. 16) --------------------------------------------------
        # β_{u,i} = σ( (g^T Ẽ_u) · (g^T Ẽ_i) )
        gu = eu @ g                                   # (E,)
        gi = ei @ g                                   # (E,)
        beta = torch.sigmoid(gu * gi)                 # (E,)

        # Softmax normalise β over user's interacted-item neighbourhood.
        # Numerically stable via max-shift + scatter-add.
        beta_max = torch.full(
            (self.n_users,),
            float("-inf"),
            device=beta.device,
            dtype=torch.float32,
        )
        beta_max.scatter_reduce_(
            0, self._edge_u, beta, reduce="amax", include_self=True
        )
        # amax may leave un-touched users at -inf; mask them to 0 to avoid NaN
        beta_max = torch.where(
            torch.isinf(beta_max), torch.zeros_like(beta_max), beta_max
        )
        exp_b = torch.exp(beta - beta_max[self._edge_u])   # (E,)
        exp_sum = torch.zeros(
            self.n_users, device=beta.device, dtype=torch.float32
        )
        exp_sum.scatter_add_(0, self._edge_u, exp_b)
        iav = exp_b / (exp_sum[self._edge_u] + 1e-12)      # (E,)

        # -- Adaptive fusion (Eq. 17-19) -----------------------------------
        # Per-view scalar affine transform f_k = tanh(W * w_k + b).
        # Softmax attention over views -> shared component -> add specifics.
        f_sav = torch.tanh(w_fuse * sav + b_fuse)
        f_iav = torch.tanh(w_fuse * iav + b_fuse)
        att = torch.softmax(
            torch.stack([f_sav, f_iav], dim=0), dim=0
        )  # (2, E)
        w_shared = att[0] * sav + att[1] * iav                          # (E,)
        # Eq. 19: w_final = w_shared + Σ_k (w_k - w_shared)
        #                 = sav + iav - w_shared            (K = 2)
        w_final = sav + iav - w_shared                                  # (E,)
        return w_final

    # ------------------------------------------------------------------
    #  Contrastive-view LightGCN
    # ------------------------------------------------------------------
    def _propagate_view(
        self,
        e_u_ego: torch.Tensor,
        e_i_ego: torch.Tensor,
        w_final: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run n_layers of LightGCN over the weighted contrastive graph.

        Uses symmetric normalisation D^{-1/2} A D^{-1/2} on the LEARNED
        edge weights, followed by layer-averaging (LightGCN default).
        """
        device = e_u_ego.device
        n_total = self.n_users + self.n_items
        # Keep sparse.mm + scatter_add in fp32 under AMP (bf16/fp16).
        w_final = w_final.float()
        e_u_ego = e_u_ego.float()
        e_i_ego = e_i_ego.float()

        # Symmetric bipartite edges: [u -> i+n_users] and [i+n_users -> u]
        row = torch.cat([self._edge_u, self._edge_i + self.n_users], dim=0)
        col = torch.cat([self._edge_i + self.n_users, self._edge_u], dim=0)
        val = torch.cat([w_final, w_final], dim=0)

        # Degree from the LEARNED weights.
        deg = torch.zeros(n_total, device=device, dtype=torch.float32)
        deg.scatter_add_(0, row, val)
        # Clamp deg > 0 to avoid division blow-ups on isolated nodes.
        deg = deg.clamp_min(1e-12)
        deg_inv_sqrt = deg.pow(-0.5)
        norm_val = deg_inv_sqrt[row] * val * deg_inv_sqrt[col]         # (2E,)

        adj = torch.sparse_coo_tensor(
            torch.stack([row, col], dim=0),
            norm_val,
            size=(n_total, n_total),
        ).coalesce()

        ego = torch.cat([e_u_ego, e_i_ego], dim=0)
        embs = [ego]
        cur = ego
        for _ in range(self.n_layers):
            cur = torch.sparse.mm(adj, cur)
            embs.append(cur)
        mean = torch.stack(embs, dim=1).mean(dim=1)
        return mean[: self.n_users], mean[self.n_users:]

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------
    def forward(
        self,
        e_u_hat: torch.Tensor,
        e_i_hat: torch.Tensor,
        e_u_ego: torch.Tensor,
        e_i_ego: torch.Tensor,
        ui_mat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the contrastive-view embeddings E_bar_u, E_bar_i.

        Args:
            e_u_hat, e_i_hat : post-GCN embeddings from the PACER trunk
                               (row-L2 normalised).  Used as Ẽ in the
                               SAV/IAV formulas.
            e_u_ego, e_i_ego : ego embeddings (the trainable
                               nn.Embedding weights of the trunk).
                               Used as the STARTING point for the view GCN.
            ui_mat           : the bipartite sparse adjacency used by the
                               trunk (for extracting edge topology).

        Returns:
            (E_bar_u, E_bar_i)  — both (N_*, d), row-L2 normalised.
        """
        self._ensure_edges(ui_mat)

        # SAV / IAV / fusion are computed on the trunk's post-GCN embeddings.
        with torch.enable_grad():
            e_u_n = F.normalize(e_u_hat, dim=-1)
            e_i_n = F.normalize(e_i_hat, dim=-1)
            w_final = self._edge_weights(e_u_n, e_i_n)                # (E,)

        # View GCN propagates over the *ego* embeddings (LightGCN convention).
        e_bar_u, e_bar_i = self._propagate_view(e_u_ego, e_i_ego, w_final)
        e_bar_u = F.normalize(e_bar_u, dim=-1)
        e_bar_i = F.normalize(e_bar_i, dim=-1)
        return e_bar_u, e_bar_i


# ---------------------------------------------------------------------------
#  Batch-N InfoNCE between original ê and contrastive view ē
# ---------------------------------------------------------------------------
def _batchN_infonce(
    e_hat: torch.Tensor,
    e_bar: torch.Tensor,
    tau: torch.Tensor,
    batch_size: int = 2048,
) -> torch.Tensor:
    """Batch-N InfoNCE (symmetric) between two L2-normalised (N, d) matrices.

    L = 0.5 * ( CE(sim(e_hat, e_bar) / τ, diag) + CE(sim(e_bar, e_hat) / τ, diag) )
    computed in row-chunks of size ``batch_size`` for memory.

    Per-node losses are pooled with ``torch.cat(...).mean()`` so every row
    contributes equally — including the final partial chunk. Averaging one
    scalar ``cross_entropy`` per chunk would under-weight that remainder.

    NB: identical semantics to
    ``branchA_simgcl_batchN.simgcl_view_invariance_loss`` but hand-inlined
    here to keep the module self-contained and importable at model-init time
    without a dependency on the model.py view path.
    """
    if e_hat.shape != e_bar.shape:
        raise ValueError(
            f"shape mismatch: e_hat={tuple(e_hat.shape)} "
            f"e_bar={tuple(e_bar.shape)}"
        )
    if e_hat.dim() != 2:
        raise ValueError(f"expected 2-D tensors, got e_hat.dim()={e_hat.dim()}")

    n = e_hat.size(0)
    tau_c = torch.clamp(tau, min=0.01)
    losses: list[torch.Tensor] = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        z1 = e_hat[start:end]                          # (B, d)
        z2 = e_bar[start:end]                          # (B, d)
        sim_12 = z1 @ z2.T / tau_c                     # (B, B)
        sim_21 = z2 @ z1.T / tau_c                     # (B, B)
        log_p_12 = F.log_softmax(sim_12, dim=-1)
        log_p_21 = F.log_softmax(sim_21, dim=-1)
        diag_idx = torch.arange(end - start, device=e_hat.device)
        losses.append(
            -0.5
            * (log_p_12[diag_idx, diag_idx] + log_p_21[diag_idx, diag_idx])
        )
    return torch.cat(losses).mean()


def compute_nrdmc_view_loss(
    view_module: NRDMCLiteView,
    e_u_hat: torch.Tensor,
    e_i_hat: torch.Tensor,
    e_u_ego: torch.Tensor,
    e_i_ego: torch.Tensor,
    ui_mat: torch.Tensor,
    tau: torch.Tensor,
    batch_size: int = 2048,
) -> torch.Tensor:
    """Top-level entry called from ``model.simgcl_view_forward`` when the
    NRDMC-lite branch is active.

    Returns a scalar loss L_mv = 0.5 * (L_mv_user + L_mv_item),
    computed as batch-N InfoNCE between ê (post-GCN trunk output) and
    ē (contrastive-view LightGCN output).
    """
    # Disable autocast: scatter_add_/sparse.mm require matching dtypes, and
    # bf16 AMP otherwise mixes Parameter (fp32) with activation (bf16).
    device_type = e_u_hat.device.type
    with torch.amp.autocast(device_type=device_type, enabled=False):
        e_bar_u, e_bar_i = view_module(
            e_u_hat.float(),
            e_i_hat.float(),
            e_u_ego.float(),
            e_i_ego.float(),
            ui_mat,
        )
        e_hat_u_n = F.normalize(e_u_hat.float(), dim=-1)
        e_hat_i_n = F.normalize(e_i_hat.float(), dim=-1)
        tau_f = tau.float() if torch.is_tensor(tau) else tau
        L_user = _batchN_infonce(e_hat_u_n, e_bar_u, tau_f, batch_size)
        L_item = _batchN_infonce(e_hat_i_n, e_bar_i, tau_f, batch_size)
        return 0.5 * (L_user + L_item)


__all__ = ["NRDMCLiteView", "compute_nrdmc_view_loss"]
