"""
damps/nrdmc_lite.py -- Branch A' (NRDMC-lite) view generators + contrastive loss.
==================================================================================

**Rev55 §8.2 + P3 (rev56)**  --  Adds the Prototype-Aware View (PTV) that was
dropped in the original §8.2 to keep the fix compact.  The upgrade path is
motivated by the P1+P2 grid results (results/p1_p2_lambda_tau_grid_clothing.json):

  * P1 (lower ``lambda_view``) failed to lift the Head Recall@20 ceiling.
  * P2 (higher ``tau``) only delayed convergence -- the ceiling stayed put.
  * Diagnosis (revised, §6 of upgrade_analysis_EN.tex): the plateau at
    R@20 ~ 0.083 is a *representational-capacity* ceiling of the SAV+IAV
    fusion, NOT a lambda/tau regularisation issue.

PTV -- Prototype-Aware View
---------------------------
Following NRDMC IPM 2026 (Eq. 15, 20-22), we introduce K learnable prototypes
``P in R^{K x d}``.  For every U-I edge (u, i) we score the *prototype
compatibility* between user u and item i as

    pi_i  = softmax( E_i @ P^T / tau_p , dim=-1 )     # (n_items, K)
    pi_u  = softmax( E_u @ P^T / tau_p , dim=-1 )     # (n_users, K)
    ptv_{u,i} = < pi_u , pi_i >                        # scalar in (0, 1]

The prototype vectors ``P`` are trained end-to-end via gradient flow from the
downstream InfoNCE view loss; no k-means initialisation is required (the
NRDMC paper's Table 4 ablation confirms end-to-end updates are sufficient).

Adaptive fusion is extended from K=2 to K=3 (Eq. 19 template):

    w_final = w_shared + (w_sav - w_shared) + (w_iav - w_shared)
                        + lambda_ptv * (w_ptv - w_shared)
            = w_sav + w_iav + lambda_ptv * w_ptv - (1 + lambda_ptv) * w_shared

``lambda_ptv = 0`` recovers the exact K=2 baseline bit-for-bit (used as the
control cell in scripts/run_p3_ptv_grid.py).

Everything else -- SAV/IAV formulation, InfoNCE, LightGCN view propagation --
matches the K=2 module signature so upstream callers (model.py,
compute_nrdmc_view_loss) require ZERO changes for the K=2 path.

Runtime cost (Amazon Clothing, |E|=197 338, d=128, K=32, L=3, B=2048):
  * PTV assignment (K=32) : ~ 2.5 * n_edges * K FLOPs = ~15 M ops (< 0.1% epoch)
  * SAV+IAV+fusion         : ~ 100 M ops (unchanged)
  * View GCN               : ~ 75 M ops  (unchanged)
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
    """SAV + IAV (+ optional PTV) view generators, adaptive fusion, view LightGCN.

    Args:
        n_users : # user nodes.
        n_items : # item nodes.
        embed_dim: embedding dimensionality (matches the trunk).
        n_layers : # of LightGCN steps to propagate over the contrastive graph.
        enable_ptv     : bool, if True adds the K=3 Prototype-Aware View path.
        n_prototypes   : K, number of learnable prototypes (>=1 when enable_ptv).
        lambda_ptv     : float, PTV mixing coefficient inside Eq. 19 fusion.
                         0.0 -> exact K=2 baseline (bit-for-bit compat).
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int,
        n_layers: int = 2,
        *,
        enable_ptv: bool = False,
        n_prototypes: int = 32,
        lambda_ptv: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.embed_dim = int(embed_dim)
        self.n_layers = int(n_layers)

        # IAV attention vector g \in R^d  (Eq. 16 of NRDMC IPM 2026).
        g = torch.randn(embed_dim) / (embed_dim ** 0.5)
        self.g = nn.Parameter(g)

        # Adaptive-fusion transform (Eq. 17): scalar affine per view.
        self.W_fuse = nn.Parameter(torch.ones(1))
        self.b_fuse = nn.Parameter(torch.zeros(1))

        # -------------------------------------------------------------------
        # P3 (rev56) -- Prototype-Aware View.
        # -------------------------------------------------------------------
        self.enable_ptv: bool = bool(enable_ptv) and int(n_prototypes) > 0
        self.n_prototypes: int = int(n_prototypes) if self.enable_ptv else 0
        # lambda_ptv is intentionally kept as a plain Python float (NOT a
        # Parameter) so the driver can pin it per grid cell without letting
        # gradient descent trivially annihilate the new PTV branch.
        self.lambda_ptv: float = float(lambda_ptv) if self.enable_ptv else 0.0

        if self.enable_ptv:
            # Xavier-scale init keeps < e_i, P_k > ~ 0 at t=0 so pi_i is
            # near uniform. Training will sharpen cluster assignments.
            proto = torch.randn(self.n_prototypes, embed_dim) / (embed_dim ** 0.5)
            self.prototypes = nn.Parameter(proto)
            # tau_p controls sharpness of the prototype softmax. Initialised
            # at 1.0; a learnable scalar lets the model discover its own
            # cluster-assignment temperature. We clamp its LOWER bound to
            # 0.05 at read-time to avoid exp-overflow.
            self.log_tau_p = nn.Parameter(torch.zeros(1))
        else:
            self.prototypes = None  # type: ignore[assignment]
            self.log_tau_p = None   # type: ignore[assignment]

        # Edge topology cache (registered on first forward()).
        self._edge_u: Optional[torch.Tensor] = None
        self._edge_i: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    #  Edge extraction (once per model)
    # ------------------------------------------------------------------
    def _ensure_edges(self, ui_mat: torch.Tensor) -> None:
        if self._edge_u is not None:
            return
        if not ui_mat.is_sparse:
            raise RuntimeError("NRDMCLiteView requires a sparse UI_mat input.")
        indices = ui_mat.coalesce().indices()
        rows, cols = indices[0], indices[1]
        mask = (rows < self.n_users) & (cols >= self.n_users)
        edge_u = rows[mask].contiguous()
        edge_i = (cols[mask] - self.n_users).contiguous()
        self.register_buffer("_edge_u_buf", edge_u, persistent=False)
        self.register_buffer("_edge_i_buf", edge_i, persistent=False)
        self._edge_u = self._edge_u_buf
        self._edge_i = self._edge_i_buf

    # ------------------------------------------------------------------
    #  PTV -- prototype-aware edge weight (P3)
    # ------------------------------------------------------------------
    def _ptv_weights(
        self,
        e_u: torch.Tensor,     # (n_users, d) L2-normalised
        e_i: torch.Tensor,     # (n_items, d) L2-normalised
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute PTV edge weight and (mean) prototype-assignment entropy.

        Returns:
            ptv    : (E,) tensor of per-edge prototype-compatibility scores.
            H_mean : scalar mean assignment entropy across the item side
                     (for diagnostic logging; not backpropagated).

        Math:
            pi_i(k) = softmax_k( <E_i, P_k> / tau_p )         (n_items, K)
            pi_u(k) = softmax_k( <E_u, P_k> / tau_p )         (n_users, K)
            ptv_{u,i} = < pi_u , pi_i >   in (0, 1]           (E,)
        """
        assert self.enable_ptv, "PTV called with enable_ptv=False"
        # Clamp tau_p at read-time to avoid overflow.
        tau_p = torch.clamp(self.log_tau_p.exp(), min=0.05, max=20.0)
        # (K, d)
        proto = self.prototypes
        # Logits: (n_items, K) and (n_users, K).
        logits_i = (e_i @ proto.T) / tau_p
        logits_u = (e_u @ proto.T) / tau_p
        pi_i_full = F.softmax(logits_i, dim=-1)   # (n_items, K)
        pi_u_full = F.softmax(logits_u, dim=-1)   # (n_users, K)
        # Gather per-edge distributions and take inner product.
        pi_u_edge = pi_u_full[self._edge_u]       # (E, K)
        pi_i_edge = pi_i_full[self._edge_i]       # (E, K)
        ptv = (pi_u_edge * pi_i_edge).sum(dim=-1) # (E,) in (0, 1]
        # Diagnostic: mean entropy of item assignments (lower = sharper).
        with torch.no_grad():
            H_i = -(pi_i_full * (pi_i_full.clamp_min(1e-12)).log()).sum(dim=-1)
            H_mean = H_i.mean()
        return ptv, H_mean

    # ------------------------------------------------------------------
    #  SAV + IAV + adaptive fusion (K=2 or K=3)
    # ------------------------------------------------------------------
    def _edge_weights(
        self,
        e_u: torch.Tensor,
        e_i: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute learned edge weight w_final over the observed U-I edges.

        Returns:
            w_final : (E,) tensor of per-edge weights.
            diag    : dict of scalar diagnostics for WandB logging.
        """
        eu = e_u[self._edge_u].float()
        ei = e_i[self._edge_i].float()
        g = self.g.float()
        w_fuse = self.W_fuse.float()
        b_fuse = self.b_fuse.float()

        # -- SAV (Eq. 14) --------------------------------------------------
        sav = torch.sigmoid((eu * ei).sum(dim=-1))     # (E,)

        # -- IAV (Eq. 16) --------------------------------------------------
        gu = eu @ g
        gi = ei @ g
        beta = torch.sigmoid(gu * gi)                  # (E,)
        beta_max = torch.full(
            (self.n_users,), float("-inf"),
            device=beta.device, dtype=torch.float32,
        )
        beta_max.scatter_reduce_(
            0, self._edge_u, beta, reduce="amax", include_self=True
        )
        beta_max = torch.where(
            torch.isinf(beta_max), torch.zeros_like(beta_max), beta_max
        )
        exp_b = torch.exp(beta - beta_max[self._edge_u])
        exp_sum = torch.zeros(
            self.n_users, device=beta.device, dtype=torch.float32,
        )
        exp_sum.scatter_add_(0, self._edge_u, exp_b)
        iav = exp_b / (exp_sum[self._edge_u] + 1e-12)  # (E,)

        # -- Adaptive fusion (Eq. 17-19) -----------------------------------
        f_sav = torch.tanh(w_fuse * sav + b_fuse)
        f_iav = torch.tanh(w_fuse * iav + b_fuse)

        diag: dict = {}
        if self.enable_ptv:
            # PTV requires FULL user/item embedding tables, not per-edge.
            ptv, H_mean = self._ptv_weights(e_u.float(), e_i.float())
            f_ptv = torch.tanh(w_fuse * ptv + b_fuse)
            att = torch.softmax(
                torch.stack([f_sav, f_iav, f_ptv], dim=0), dim=0,
            )                                          # (3, E)
            w_shared = att[0] * sav + att[1] * iav + att[2] * ptv
            # Eq. 19 (K=3, lambda-weighted PTV branch):
            # w_final = w_shared
            #          + (sav  - w_shared)
            #          + (iav  - w_shared)
            #          + lambda_ptv * (ptv - w_shared)
            #        = sav + iav + lambda_ptv * ptv - (1 + lambda_ptv) * w_shared
            w_final = (
                sav + iav + self.lambda_ptv * ptv
                - (1.0 + self.lambda_ptv) * w_shared
            )
            diag["ptv_edge_mean"] = float(ptv.mean().detach())
            diag["ptv_edge_std"] = float(ptv.std().detach())
            diag["ptv_entropy_i"] = float(H_mean.detach())
            diag["ptv_tau_p"] = float(
                torch.clamp(self.log_tau_p.exp(), min=0.05, max=20.0)
                .detach()
                .item()
            )
        else:
            # K=2 legacy path (bit-for-bit identical to §8.2 rev55).
            att = torch.softmax(
                torch.stack([f_sav, f_iav], dim=0), dim=0,
            )
            w_shared = att[0] * sav + att[1] * iav
            w_final = sav + iav - w_shared

        diag["sav_mean"] = float(sav.mean().detach())
        diag["iav_mean"] = float(iav.mean().detach())
        diag["w_final_mean"] = float(w_final.mean().detach())
        return w_final, diag

    # ------------------------------------------------------------------
    #  Contrastive-view LightGCN
    # ------------------------------------------------------------------
    def _propagate_view(
        self,
        e_u_ego: torch.Tensor,
        e_i_ego: torch.Tensor,
        w_final: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = e_u_ego.device
        n_total = self.n_users + self.n_items
        w_final = w_final.float()
        e_u_ego = e_u_ego.float()
        e_i_ego = e_i_ego.float()

        row = torch.cat([self._edge_u, self._edge_i + self.n_users], dim=0)
        col = torch.cat([self._edge_i + self.n_users, self._edge_u], dim=0)
        val = torch.cat([w_final, w_final], dim=0)

        deg = torch.zeros(n_total, device=device, dtype=torch.float32)
        deg.scatter_add_(0, row, val)
        deg = deg.clamp_min(1e-12)
        deg_inv_sqrt = deg.pow(-0.5)
        norm_val = deg_inv_sqrt[row] * val * deg_inv_sqrt[col]

        adj = torch.sparse_coo_tensor(
            torch.stack([row, col], dim=0), norm_val,
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
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Compute view embeddings and return diagnostics.

        Returns:
            (E_bar_u, E_bar_i, diag)
        """
        self._ensure_edges(ui_mat)
        with torch.enable_grad():
            e_u_n = F.normalize(e_u_hat, dim=-1)
            e_i_n = F.normalize(e_i_hat, dim=-1)
            w_final, diag = self._edge_weights(e_u_n, e_i_n)
        e_bar_u, e_bar_i = self._propagate_view(e_u_ego, e_i_ego, w_final)
        e_bar_u = F.normalize(e_bar_u, dim=-1)
        e_bar_i = F.normalize(e_bar_i, dim=-1)
        return e_bar_u, e_bar_i, diag


# ---------------------------------------------------------------------------
#  Batch-N InfoNCE between original ê and contrastive view ē
# ---------------------------------------------------------------------------
def _batchN_infonce(
    e_hat: torch.Tensor,
    e_bar: torch.Tensor,
    tau: torch.Tensor,
    batch_size: int = 2048,
) -> torch.Tensor:
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
        z1 = e_hat[start:end]
        z2 = e_bar[start:end]
        sim_12 = z1 @ z2.T / tau_c
        sim_21 = z2 @ z1.T / tau_c
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
    return_diag: bool = False,
):
    """Top-level entry from ``model.simgcl_view_forward``.

    Backward-compat:
        return_diag=False  -> returns scalar loss tensor (same as rev55).
        return_diag=True   -> returns (loss, diag_dict).
    """
    device_type = e_u_hat.device.type
    with torch.amp.autocast(device_type=device_type, enabled=False):
        e_bar_u, e_bar_i, diag = view_module(
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
        loss = 0.5 * (L_user + L_item)
    if return_diag:
        return loss, diag
    return loss


__all__ = ["NRDMCLiteView", "compute_nrdmc_view_loss"]
