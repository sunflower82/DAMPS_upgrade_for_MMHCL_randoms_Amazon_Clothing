"""
damps_simgcl.py -- SimGCL view-invariance helpers (Phase 2 / Wave 2 / M1).
==========================================================================

This module is the **single source of truth** for the two SimGCL primitives
that the rev54 architecture requires at Milestone M1:

    (a) ``inject_uniform_noise(emb, eps)``
        Reference-faithful noise generator following Yu et al. (SIGIR 2022,
        eq. 4): a uniform vector is sampled, L2-normalised per-row, and
        scaled by ``eps``; its sign is then aligned with ``sign(emb)`` so
        that the perturbation never flips a coordinate.

    (b) ``simgcl_view_invariance_loss(z1, z2, tau, batch_size)``
        Symmetric, batched InfoNCE between two perturbed views of the
        same node set. The implementation re-uses the row-chunked layout
        already proven in ``model.batched_contrastive_loss`` (rev45 baseline)
        so memory peak stays identical at Amazon Clothing scale
        (n_items = 23,033).

Why ship this as a free-standing module
---------------------------------------
1. ``model.py`` already exceeds 900 lines; bolting yet another helper onto
   the class would push it past the project's per-file review budget.
2. SimGCL must be unit-testable in isolation -- see ``test_simgcl.py``.
3. The rev54 reviewer board explicitly flagged "geometric decoupling" of
   the contrastive terms as the only justification for disabling PCGrad
   during Phase 2 (rev54 lines 173-176). Keeping ``L_SimGCL`` in its own
   file makes that decoupling architecturally visible.

CAUTION
-------
* ``inject_uniform_noise`` must NEVER be called inside ``torch.no_grad()``;
  the perturbed embedding is the input to a differentiable LightGCN path.
* The "sign(e)" alignment from SimGCL means a coordinate with magnitude
  zero stays at zero. This is intentional and prevents the noise from
  resurrecting collapsed dimensions. Do not "fix" it by dropping the sign.
* ``simgcl_view_invariance_loss`` operates on row-L2-normalised inputs.
  Callers must apply ``F.normalize(z, dim=-1)`` BEFORE invoking this
  function, mirroring the existing convention in ``batched_contrastive_loss``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# (a) Uniform-noise injection -- Yu et al. SIGIR 2022, eq. (4)
# ---------------------------------------------------------------------------
def inject_uniform_noise(
    emb: torch.Tensor,
    eps: float,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Return ``emb + delta`` where ``delta`` is a SimGCL-style perturbation.

    Concretely, for each row ``e`` of ``emb`` we sample
        u ~ U(0, 1)^d,    delta_raw = u / ||u||_2,
        delta = eps * sign(e) * delta_raw.

    The L2-normalisation guarantees ``||delta||_2 = eps`` exactly, which
    matches the spectrum-bounding analysis in the SimGCL paper. The
    sign-alignment forbids the noise from flipping coordinates -- so the
    perturbed embedding stays in the same orthant as the anchor.

    Args:
        emb        : (N, d) tensor on any device; gradient flow is preserved.
        eps        : Non-negative scalar. Recommended default 0.1 per Yu et al.
                     The rev54 Optuna search range is ``eps in [0.05, 0.2]``.
        generator  : Optional ``torch.Generator`` for deterministic noise
                     (used by ``test_simgcl.py``). Pass ``None`` in training
                     so the noise re-randomises every forward pass -- this
                     is what gives the view-invariance objective its
                     denoising effect.

    Returns:
        Tensor of the same shape as ``emb``; ``emb`` itself is NOT modified.

    Raises:
        ValueError : if ``eps`` is negative or non-finite.
    """
    if not (eps >= 0.0 and torch.isfinite(torch.tensor(eps)).item()):
        raise ValueError(f"eps must be a non-negative finite scalar; got {eps!r}")
    if eps == 0.0:
        # Genuine no-op -- avoid spending RNG cycles. Useful for the
        # "anchor-view == perturbed-view" sanity check in test_simgcl.py.
        return emb

    # Sample U(0,1)^d, then row-L2 normalise. ``torch.rand_like`` honours
    # the dtype + device of ``emb`` so we never accidentally upcast to FP64
    # on GPU.
    if generator is None:
        u = torch.rand_like(emb)
    else:
        u = torch.rand(emb.shape, dtype=emb.dtype, device=emb.device, generator=generator)
    u = F.normalize(u, p=2, dim=-1)                       # ||u_i||_2 = 1

    # Sign-alignment: torch.sign(0) == 0 in PyTorch >= 1.7, so genuinely
    # collapsed dimensions stay collapsed. This is the documented SimGCL
    # behaviour -- it is NOT a bug.
    delta = eps * torch.sign(emb) * u                     # broadcast over d
    return emb + delta


# ---------------------------------------------------------------------------
# (b) View-invariance InfoNCE -- the third contrastive term of L_total
# ---------------------------------------------------------------------------
def simgcl_view_invariance_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    tau: torch.Tensor | float = 0.3,
    batch_size: int = 4096,
) -> torch.Tensor:
    """Symmetric batched InfoNCE between two perturbed views.

    Math (rev54 line 172, Yu et al. SIGIR 2022):
        L_view = (1/2N) * sum_i [ -log( exp(<z1_i, z2_i> / tau) /
                                        sum_j exp(<z1_i, z2_j> / tau) )
                                  -log( exp(<z2_i, z1_i> / tau) /
                                        sum_j exp(<z2_i, z1_j> / tau) ) ].

    Args:
        z1, z2     : (N, d) row-L2-normalised embeddings of the two
                     perturbed views. **Must** be row-normalised by the
                     caller (do ``F.normalize(z, dim=-1)`` first).
        tau        : Scalar temperature. Accepts either a Python float or
                     a 0-dim Tensor (``self.tau`` from the model). Clamped
                     internally to ``>= 0.01`` to mirror the rev45 baseline.
        batch_size : Row-chunk size. 4096 mirrors the value used by
                     ``model.batched_contrastive_loss`` and keeps peak
                     memory below 6 GiB on RTX 5090 at d=64, N<=30k.

    Returns:
        Scalar loss tensor with ``requires_grad=True`` (assuming z1,z2 do).

    Notes:
        * Both directions of the asymmetric InfoNCE are averaged so that
          gradients flow back into BOTH perturbed views in equal measure.
          This is what makes the loss truly view-invariant rather than
          "view-1-pulled-towards-view-2".
        * The "between" matrix is computed once per chunk and indexed for
          both the numerator (diagonal) and the row-sum denominator -- so
          the FLOP cost is exactly twice the one-directional InfoNCE.
    """
    if z1.shape != z2.shape:
        raise ValueError(f"shape mismatch: z1={tuple(z1.shape)} z2={tuple(z2.shape)}")
    if z1.dim() != 2:
        raise ValueError(f"expected 2-D tensors, got z1.dim()={z1.dim()}")

    if isinstance(tau, torch.Tensor):
        tau_eff = torch.clamp(tau, min=0.01)
    else:
        tau_eff = max(float(tau), 0.01)

    n = z1.size(0)
    losses: list[torch.Tensor] = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        # Cosine similarity == inner product (inputs are row-normalised).
        sim_12 = z1[start:end] @ z2.T / tau_eff           # (B, N)
        sim_21 = z2[start:end] @ z1.T / tau_eff           # (B, N)

        # log_softmax is numerically safer than the manual exp/sum/log
        # pattern used in the rev45 baseline. The diagonal entries of the
        # B x N chunk are at columns [start, end).
        idx = torch.arange(start, end, device=z1.device)
        log_p_12 = F.log_softmax(sim_12, dim=-1)
        log_p_21 = F.log_softmax(sim_21, dim=-1)
        losses.append(
            -0.5 * (log_p_12[torch.arange(end - start, device=z1.device), idx]
                    + log_p_21[torch.arange(end - start, device=z1.device), idx])
        )
    return torch.cat(losses).mean()


# ---------------------------------------------------------------------------
# (c) Convenience wrapper -- runs two perturbed propagations and returns L_view
# ---------------------------------------------------------------------------
def compute_simgcl_view_loss(
    *,
    propagate_fn,
    ego_user: torch.Tensor,
    ego_item: torch.Tensor,
    eps: float,
    tau: torch.Tensor | float,
    batch_size_user: int = 4096,
    batch_size_item: int = 4096,
) -> torch.Tensor:
    """Run two perturbed LightGCN propagations and return L_view (user+item).

    The function expects ``propagate_fn`` to be the model's extracted
    ``_lightgcn_propagate(ego_user, ego_item)`` method -- see
    ``model_patch_simgcl.py`` Block (2) for the refactor that exposes it.
    Both user and item branches contribute symmetrically; the returned
    scalar is the unweighted average of the two view-invariance losses.

    The caller is responsible for multiplying the returned value by
    ``lambda_view`` before adding it to the total loss.
    """
    # Two independent perturbations -> two independent propagations.
    u_pert_1 = inject_uniform_noise(ego_user, eps)
    i_pert_1 = inject_uniform_noise(ego_item, eps)
    u_view_1, i_view_1 = propagate_fn(u_pert_1, i_pert_1)

    u_pert_2 = inject_uniform_noise(ego_user, eps)
    i_pert_2 = inject_uniform_noise(ego_item, eps)
    u_view_2, i_view_2 = propagate_fn(u_pert_2, i_pert_2)

    # Row-normalise BEFORE the view-invariance loss -- this is what makes
    # the dot product equal to cosine similarity.
    u_view_1 = F.normalize(u_view_1, dim=-1)
    u_view_2 = F.normalize(u_view_2, dim=-1)
    i_view_1 = F.normalize(i_view_1, dim=-1)
    i_view_2 = F.normalize(i_view_2, dim=-1)

    l_user = simgcl_view_invariance_loss(u_view_1, u_view_2, tau, batch_size_user)
    l_item = simgcl_view_invariance_loss(i_view_1, i_view_2, tau, batch_size_item)
    return 0.5 * (l_user + l_item)
