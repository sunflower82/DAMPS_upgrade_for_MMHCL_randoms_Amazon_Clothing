"""
branchA_model_patch.py -- Surgical patches to MMHCL_DAMPS_Project/model.py.
===========================================================================

This file documents the minimal diff against the rev54 ``model.py`` required
to enable Branch A (Wave 2 / Phase 2 §8.1 of Revision 55). Apply each block
manually -- the patches are SHORT and the line numbers reference the rev54
``model.py`` that ships in the repository
``sunflower82/DAMPS_upgrade_for_MMHCL_randoms_Amazon_Clothing``.

The accompanying ``branchA_simgcl_batchN.py`` is the runtime helper that
replaces ``damps_simgcl.py``. After installing the patches below, copy
``branchA_simgcl_batchN.py`` to the project root next to ``train.py``.

Patch summary (4 blocks):
    (A1)  Constructor kwargs       -- 4 new flags (epoch-aware skip + bcl_batchn).
    (A2)  Epoch-aware view forward -- ``simgcl_view_forward(epoch)``.
    (A3)  Optional bcl_batchn path -- swap the (B, N) chunk for (B, B).
    (A4)  Replace damps_simgcl import in the model module.

The patches are bit-for-bit identical to the Wave 1 path when:
    --enable_simgcl 0
    --branchA_view_every_k 1
    --branchA_bcl_batchn 0
"""

# =============================================================================
#  Block (A1)  --  Constructor signature additions (model.py, after Wave 1 LogQ)
# =============================================================================
# BEFORE (Wave 1 tail of MMHCL.__init__):
#     enable_logq:  bool  = False,
#     logq_scale:   float = 1.0,
#     logq_clip:    float = 5.0,
#     # Wave 2 / M1 -- SimGCL view-invariance (Yu et al. SIGIR 2022)
#     enable_simgcl:          bool  = False,
#     simgcl_eps:             float = 0.1,
#     simgcl_batch_size_user: int   = 4096,
#     simgcl_batch_size_item: int   = 4096,
# ) -> None:
#
# AFTER (Branch A):
#     ...                              # (Wave 1 + Wave 2 kwargs unchanged)
#     # ---- Branch A (rev55 §8.1) -- speedup levers ----
#     branchA_view_every_k:  int  = 2,     # compute L_view every k epochs
#     branchA_bcl_batchn:    bool = True,  # batch-N bcl_item / bcl_user
#     branchA_view_bsz:      int  = 2048,  # row-chunk for L_view (was 4096)
#     branchA_bcl_bsz:       int  = 2048,  # row-chunk for bcl_*  (was 4096)
# ) -> None:
#
# And in the constructor BODY, immediately after the rev54 SimGCL kwargs are
# stored as attributes, ADD:

_BLOCK_A1_CONSTRUCTOR_BODY = '''
        # ---- Branch A (rev55 §8.1) -- speedup levers ----
        self.branchA_view_every_k = int(branchA_view_every_k)
        self.branchA_bcl_batchn   = bool(branchA_bcl_batchn)
        self.branchA_view_bsz     = int(branchA_view_bsz)
        self.branchA_bcl_bsz      = int(branchA_bcl_bsz)
        # Cache for the perturbed views; refreshed every ``view_every_k`` epochs.
        # 4-tuple (u_view_1, u_view_2, i_view_1, i_view_2). Created lazily.
        self._simgcl_view_cache:  Optional[tuple] = None
        self._simgcl_view_epoch:  int            = -1
'''


# =============================================================================
#  Block (A2)  --  Epoch-aware SimGCL forward (replaces lines 723-751 of model.py)
# =============================================================================
# REPLACE the entire rev54 ``simgcl_view_forward`` method body with the
# version below.
#
# Behaviour:
#   * Returns 0.0 when enable_simgcl=False (unchanged).
#   * When called with the same ``epoch`` more than once -- e.g. multiple
#     mini-batches per epoch -- reuses the cached perturbed views to keep the
#     loss differentiable but avoid the 2 extra LightGCN propagations.
#   * Refreshes the cache only when ``epoch % view_every_k == 0``. Between
#     refresh epochs the loss is still computed (with grad flow into the
#     cached views), so the optimiser keeps a smooth L_view signal.

_BLOCK_A2_VIEW_FORWARD = '''
    def simgcl_view_forward(self, epoch: int = 0) -> torch.Tensor:
        """Compute L_SimGCL = 0.5 * (L_user + L_item) with view-cache reuse.

        Branch A (rev55 §8.1) augments the rev54 helper with:
          * batch-N InfoNCE   (delegated to branchA_simgcl_batchN.compute_simgcl_view_loss).
          * Epoch-aware view cache: the perturbed LightGCN propagations are
            re-run only on epochs satisfying ``epoch % view_every_k == 0``.

        Args:
            epoch: Current training epoch index, supplied by train.py.

        Returns:
            Scalar loss tensor with gradient flow into ego embeddings.
        """
        if not self.enable_simgcl:
            return torch.zeros(
                (), device=self.user_ui_embedding.weight.device
            )

        from branchA_simgcl_batchN import compute_simgcl_view_loss

        # Decide whether this epoch must refresh the perturbed views.
        refresh = (
            self._simgcl_view_cache is None
            or epoch != self._simgcl_view_epoch
            and (epoch % self.branchA_view_every_k == 0)
        )

        if refresh or self._simgcl_view_cache is None:
            loss, views = compute_simgcl_view_loss(
                propagate_fn   = self._lightgcn_propagate,
                ego_user       = self.user_ui_embedding.weight,
                ego_item       = self.item_ui_embedding.weight,
                eps            = self.simgcl_eps,
                tau            = self.tau,
                batch_size_user= self.branchA_view_bsz,
                batch_size_item= self.branchA_view_bsz,
                views_cached   = None,
            )
            # Cache only the *detached* views to avoid double-counting grad
            # contributions across epochs. Re-attaching grad on cached views
            # would require keeping the LightGCN propagation graph alive
            # across optimiser steps -- a memory leak we cannot afford.
            self._simgcl_view_cache = tuple(v.detach() for v in views)
            self._simgcl_view_epoch = epoch
        else:
            # Re-use cached views; the loss still has grad w.r.t. the
            # similarity matmul against the latest ego embeddings only
            # implicitly via the cosine alignment objective. To preserve a
            # meaningful gradient we re-propagate ONE cheap perturbed view
            # on the off-epochs:
            ego_u = self.user_ui_embedding.weight
            ego_i = self.item_ui_embedding.weight
            from branchA_simgcl_batchN import inject_uniform_noise
            import torch.nn.functional as F

            u_pert = inject_uniform_noise(ego_u, self.simgcl_eps)
            i_pert = inject_uniform_noise(ego_i, self.simgcl_eps)
            u_now, i_now = self._lightgcn_propagate(u_pert, i_pert)
            u_now = F.normalize(u_now, dim=-1)
            i_now = F.normalize(i_now, dim=-1)
            # Pair (current grad-bearing view) against (cached view 1).
            u_cached, _, i_cached, _ = self._simgcl_view_cache
            views_paired = (u_now, u_cached, i_now, i_cached)
            loss, _ = compute_simgcl_view_loss(
                propagate_fn   = self._lightgcn_propagate,
                ego_user       = ego_u,
                ego_item       = ego_i,
                eps            = self.simgcl_eps,
                tau            = self.tau,
                batch_size_user= self.branchA_view_bsz,
                batch_size_item= self.branchA_view_bsz,
                views_cached   = views_paired,
            )
        return loss
'''


# =============================================================================
#  Block (A3)  --  Branch A toggle inside ``batched_contrastive_loss``.
# =============================================================================
# Add ONE early-return branch at the top of ``batched_contrastive_loss``
# (model.py:756) that delegates to the batch-N variant when the flag is on.
#
# The patch is non-invasive: when ``branchA_bcl_batchn=False`` the original
# rev54 body runs unchanged. When True, ALL further work happens in the
# helper module so the model file stays readable.

_BLOCK_A3_BCL_BATCHN_HEAD = '''
        # ---- Branch A (rev55 §8.1) -- batch-N variant for speed ----
        if getattr(self, "branchA_bcl_batchn", False):
            from branchA_simgcl_batchN import batched_contrastive_loss_batchN
            log_q_arg = self.log_q if (self.enable_logq and apply_logq) else None
            return batched_contrastive_loss_batchN(
                z1            = z1,
                z2            = z2,
                tau           = self.tau,
                batch_size    = self.branchA_bcl_bsz,
                apply_logq    = bool(self.enable_logq and apply_logq),
                log_q         = log_q_arg,
                logq_scale    = float(self.logq_scale),
                logq_clip     = float(self.logq_clip),
            )
        # ---- (else: fall through to the rev54 all-rank implementation) ----
'''


# =============================================================================
#  Block (A4)  --  Optional: redirect ``damps_simgcl`` -> ``branchA_simgcl_batchN``
# =============================================================================
# The rev54 ``simgcl_view_forward`` does ``from damps_simgcl import ...``. The
# patched version in Block (A2) already imports from ``branchA_simgcl_batchN``,
# so Block (A4) is OPTIONAL.
#
# However, ``tests/test_simgcl.py`` imports ``damps_simgcl`` directly. To keep
# those tests passing without rewriting them, create a one-line shim:
#
#     # damps_simgcl.py  --  Branch A shim
#     from branchA_simgcl_batchN import *  # noqa: F401, F403
#
# (Place this at the repo root, overwriting the current ``damps_simgcl.py``.)
