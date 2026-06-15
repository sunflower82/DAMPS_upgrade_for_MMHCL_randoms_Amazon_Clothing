"""
model_patch_logq.py — 1A PR Snippets for MMHCL_DAMPS_Project/model.py
=====================================================================

This file is **not** meant to be imported. It contains the four
self-contained code blocks that should be **manually merged** into
``MMHCL_DAMPS_Project/model.py`` to implement the LogQ correction
(variant "h", rev53 §3.1, eq. 1).

Merge order (do each one as a separate commit for auditable diffs):

    (1)  Constructor signature   — add `enable_logq`, `logq_scale`,
                                   `logq_clip` to `__init__`.
    (2)  Buffer registration     — register `log_q` buffer (zero-init).
    (3)  Setter                  — add `set_log_q(log_q)` method.
    (4)  Loss patch              — modify `batched_contrastive_loss`
                                   to optionally subtract log_q before
                                   the exp.

After (1)–(4), run the unit tests in ``tests/test_popularity_prior.py``
and a 1-epoch smoke training with ``--enable_logq 0`` to confirm
**bit-for-bit identical** outputs vs. the rev45 baseline. Only then
flip ``--enable_logq 1``.

Code-line evidence for the targeted hooks:
    * Constructor body where new defaults are appended:
      ``model.py:166-225``
    * τ buffer registration (analogous pattern):
      ``model.py:356-388``
    * ``set_meta_categories`` setter (analogous pattern):
      ``model.py:390-397``
    * ``batched_contrastive_loss`` body to replace:
      ``model.py:605-645``
"""

# =========================================================================
#  Block (1) — Constructor signature
# =========================================================================
# Append three NEW kwargs to the existing __init__ signature at model.py:166.
# Default values are chosen so existing call sites in train.py:279-293 keep
# working without any change.
#
# BEFORE (model.py:188-189):
#     warmup_epochs: int = 10,
#     damps_num_categories: int = 10,
#     data_driven_prior: bool = True,
# ) -> None:
#
# AFTER:
#     warmup_epochs: int = 10,
#     damps_num_categories: int = 10,
#     data_driven_prior: bool = True,
#     enable_logq: bool = False,            # rev53 §3.1 — variant "h"
#     logq_scale: float = 1.0,              # multiplier on log_q before subtraction
#     logq_clip: float = 5.0,               # symmetric clip on scale*log_q
# ) -> None:

# =========================================================================
#  Block (2) — Buffer registration
# =========================================================================
# Add the following block to the end of __init__, right after the
# tau registration at model.py:388. The buffer is registered with zeros
# so that — if set_log_q is never called — subtracting log_q is a no-op
# (exp(sim - 0) = exp(sim)), giving us a safe "uninitialised = baseline"
# fallback. We *also* guard against the silent-no-op by a fail-fast check
# inside batched_contrastive_loss (Block 4).
#
# Paste BELOW model.py:388 (after `self.register_buffer("tau", tau_tensor)`):

_BLOCK_2_INIT_TAIL = """
        # ------------------------------------------------------------------
        # 10. LogQ correction state (rev53 §3.1, eq. 1 — variant "h")
        # ------------------------------------------------------------------
        # Toggles the popularity-corrected InfoNCE proposed by Yi et al.
        # (RecSys 2019). When enable_logq=True, batched_contrastive_loss
        # subtracts ``logq_scale * clip(log_q[j], -logq_clip, +logq_clip)``
        # from per-column logits BEFORE dividing by τ and taking exp.
        #
        # The log_q buffer must be populated via set_log_q(...) before the
        # first training step; an uninitialised (all-zero) buffer triggers
        # a fail-fast inside batched_contrastive_loss to avoid silently
        # disabling the correction (rev53 §3.1, line 104).
        #
        # logq_scale and logq_clip default to (1.0, 5.0) which matches the
        # rev53 spec literally. For τ=0.3 and cosine sim ∈ [-1,1], expect
        # log_q ∈ [-12, -4] at Amazon Clothing scale, so the unscaled
        # subtraction will dominate the τ-normalised logits. The first
        # sanity sweep MUST cover logq_scale ∈ {0.05, 0.1, 0.3, 1.0} on a
        # subset of seeds before locking the spec default — see the M1.5
        # protocol in the LogQ README.
        # ------------------------------------------------------------------
        self.enable_logq: bool = bool(enable_logq)
        self.logq_scale: float = float(logq_scale)
        self.logq_clip: float = float(logq_clip)
        self.register_buffer("log_q", torch.zeros(n_items, dtype=torch.float32))
"""


# =========================================================================
#  Block (3) — Setter
# =========================================================================
# Add this method immediately AFTER set_meta_categories at model.py:397.
# Parallel structure & validation style.

_BLOCK_3_SETTER = """
    def set_log_q(self, log_q: torch.Tensor) -> None:
        \"\"\"Register the per-item log-popularity vector (n_items,).

        Must be called once after model construction, before the first
        training step, when ``enable_logq=True``. The tensor is copied
        into the registered buffer so it follows the model on .to(device)
        and is persisted by state_dict.

        Source of truth: ``damps.popularity_prior.load_or_build_log_q``.
        \"\"\"
        if log_q.shape != (self.n_items,):
            raise ValueError(
                f"log_q shape {tuple(log_q.shape)} != ({self.n_items},)"
            )
        if not torch.isfinite(log_q).all():
            n_bad = int((~torch.isfinite(log_q)).sum())
            raise ValueError(
                f"log_q contains {n_bad} non-finite value(s); "
                "rebuild with mode='laplace' and beta > 0."
            )
        self.log_q.copy_(log_q.detach().to(self.log_q.dtype))
"""


# =========================================================================
#  Block (4) — Patched batched_contrastive_loss
# =========================================================================
# REPLACE the entire body of batched_contrastive_loss at model.py:605-645
# with the version below. Backward compatibility: when called WITHOUT
# apply_logq=True (existing call site at train.py:578 for bcl_user), the
# behaviour is bit-for-bit identical to the rev45 baseline.

_BLOCK_4_LOSS = '''
    def batched_contrastive_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        batch_size: int = 4096,
        apply_logq: bool = False,
    ) -> torch.Tensor:
        """
        InfoNCE contrastive loss with optional LogQ popularity correction.

        Math (rev53 §3.1, eq. 1; variant "h"):
            L_NCEQ = - log[ exp((sim(u,i+) - s·clip(log_q(i+))) / τ) /
                            Σ_j exp((sim(u,i-) - s·clip(log_q(j))) / τ) ]
        where s = logq_scale and clip(·) = clamp(·, -logq_clip, +logq_clip).

        Args:
            z1, z2     : (N, d) row-normalised embeddings. Positive pairs
                         are the diagonal of sim(z1, z2). Rows index either
                         items (for ``bcl_item``) or users (for ``bcl_user``).
            batch_size : column-chunk size for the row-wise InfoNCE
                         (unchanged from rev45).
            apply_logq : when True AND ``self.enable_logq=True``, subtract
                         the scaled+clipped log_q from each column logit.
                         When False, the loss reduces to the original
                         rev45 baseline (bit-for-bit identical).

        Backward compatibility:
            * Existing call ``self.model.batched_contrastive_loss(z1, z2)``
              uses ``apply_logq=False`` — no behaviour change.
            * The user-branch call MUST stay at ``apply_logq=False`` because
              log_q is an item-popularity prior; applying it on users would
              double-count user activity bias.
        """
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        # ``torch.clamp`` works identically on Parameter and buffer.
        tau = torch.clamp(self.tau, min=0.01)

        # Decide once per call whether we are in LogQ mode.
        use_logq = bool(self.enable_logq and apply_logq)
        if use_logq:
            if self.log_q.shape[0] != num_nodes:
                raise ValueError(
                    f"LogQ correction requires log_q.shape[0] == z1.shape[0]; "
                    f"got log_q={self.log_q.shape[0]}, z1={num_nodes}. "
                    "Pass apply_logq=True ONLY on the item branch."
                )
            if float(self.log_q.abs().sum()) == 0.0:
                # Fail-fast (rev53 §3.1, line 104): an uninitialised log_q
                # buffer would silently disable the correction.
                raise RuntimeError(
                    "enable_logq=True but log_q is zero. "
                    "Call model.set_log_q(...) once after construction."
                )
            log_q_term = torch.clamp(
                self.logq_scale * self.log_q.to(device),
                min=-self.logq_clip,
                max=+self.logq_clip,
            )                                                 # (num_nodes,)
            # f_logq(sim_block, col_slice) = exp((sim - log_q_term[col]) / τ)
            def f_logq(sim_block: torch.Tensor,
                       col_slice: torch.Tensor) -> torch.Tensor:
                return torch.exp((sim_block - log_q_term[col_slice][None, :]) / tau)
        else:
            def f_simple(sim_block: torch.Tensor) -> torch.Tensor:
                return torch.exp(sim_block / tau)

        indices = torch.arange(0, num_nodes, device=device)
        losses: list[torch.Tensor] = []

        for i in range(num_batches):
            mask = indices[i * batch_size : (i + 1) * batch_size]
            if use_logq:
                # For refl / between sims, all columns participate so the
                # column index slice is the full ``indices`` vector.
                refl_sim = f_logq(self._sim(z1[mask], z1), indices)
                between_sim = f_logq(self._sim(z1[mask], z2), indices)
            else:
                refl_sim = f_simple(self._sim(z1[mask], z1))
                between_sim = f_simple(self._sim(z1[mask], z2))

            losses.append(
                -torch.log(
                    between_sim[:, i * batch_size : (i + 1) * batch_size].diag()
                    / (
                        refl_sim.sum(1)
                        + between_sim.sum(1)
                        - refl_sim[:, i * batch_size : (i + 1) * batch_size].diag()
                    )
                )
            )
        return torch.cat(losses).mean()
'''


# =========================================================================
#  Block (5) — Call-site change in train.py
# =========================================================================
# ONLY the item branch carries the LogQ correction (rev53 §3.1 — variant "h"
# applies to item logits in InfoNCE). User branch must stay at apply_logq=False.
#
# BEFORE (train.py:575-580):
#     bcl_item = self.model.batched_contrastive_loss(
#         out["i_ui_emb"], out["ii_emb"]
#     ) * args.item_loss_ratio
#     bcl_user = self.model.batched_contrastive_loss(
#         out["u_ui_emb"], out["uu_emb"]
#     ) * args.user_loss_ratio
#
# AFTER:
#     bcl_item = self.model.batched_contrastive_loss(
#         out["i_ui_emb"], out["ii_emb"], apply_logq=True,
#     ) * args.item_loss_ratio
#     bcl_user = self.model.batched_contrastive_loss(
#         out["u_ui_emb"], out["uu_emb"], apply_logq=False,
#     ) * args.user_loss_ratio


# =========================================================================
#  Block (6) — CLI flags in utility/parser.py
# =========================================================================
# Insert next to the existing damps_* flags around utility/parser.py:168-191.

_BLOCK_6_PARSER = '''
    # ------------------------------------------------------------------
    # rev53 §3.1 — LogQ popularity correction (variant "h")
    # ------------------------------------------------------------------
    parser.add_argument("--enable_logq", type=int, default=0,
                        help="1 enables the LogQ popularity correction in "
                             "the item-branch InfoNCE (rev53 §3.1 eq. 1). "
                             "Requires --logq_mode and --logq_beta to be set; "
                             "log_q is rebuilt and cached under the dataset "
                             "directory on first use.")
    parser.add_argument("--logq_mode", type=str, default="laplace",
                        choices=["laplace", "raw", "sqrt"],
                        help="q(i) estimator. 'laplace' = (n+β)/(N+|I|β); "
                             "'sqrt' = the DGRec WWW2024 less-aggressive "
                             "variant; 'raw' requires every item ≥ 1 "
                             "interaction (rare).")
    parser.add_argument("--logq_beta", type=float, default=1.0,
                        help="Laplace smoothing coefficient β > 0 for "
                             "logq_mode in {laplace, sqrt}. Ignored for raw.")
    parser.add_argument("--logq_scale", type=float, default=1.0,
                        help="Multiplier on log_q before subtraction. With "
                             "cosine-normalised sim and τ=0.3, the spec "
                             "default 1.0 may dominate the logits; sweep "
                             "{0.05, 0.1, 0.3, 1.0} at M1.5 before locking.")
    parser.add_argument("--logq_clip", type=float, default=5.0,
                        help="Symmetric clip on logq_scale*log_q to keep "
                             "exp(./τ) numerically safe.")
'''


# =========================================================================
#  Block (7) — train.py wiring (build log_q, pass to model)
# =========================================================================
# Insert immediately AFTER the model construction (after the
# set_meta_categories call at train.py:297-299).

_BLOCK_7_TRAIN_WIRING = '''
        # ------------------------------------------------------------------
        # rev53 §3.1 — Build the LogQ popularity prior and register it
        # on the model. Cached under data_generator.path.
        # ------------------------------------------------------------------
        if bool(args.enable_logq):
            from damps.popularity_prior import (
                load_or_build_log_q,
                describe_log_q,
                compute_item_counts,
            )
            log_q = load_or_build_log_q(
                cache_dir=data_generator.path,
                n_items=self.n_items,
                train_items=data_generator.train_items,
                beta=float(args.logq_beta),
                mode=str(args.logq_mode),
                force_rebuild=False,
            )
            self.model.set_log_q(log_q.to(self.device))
            counts = compute_item_counts(
                data_generator.train_items, self.n_items
            )
            self.logger.logging(
                "[LogQ] " + ", ".join(
                    f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in describe_log_q(log_q, counts=counts).items()
                )
            )
'''


# =========================================================================
#  END — apply the seven blocks in order and run the unit tests.
# =========================================================================
