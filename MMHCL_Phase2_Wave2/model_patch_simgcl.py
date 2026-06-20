"""
model_patch_simgcl.py -- Wave 2 / M1 PR Snippets for MMHCL_DAMPS_Project
========================================================================

This file is **not** meant to be imported. It contains the six
self-contained code blocks that should be **manually merged** into the
``MMHCL_DAMPS_Project`` repository to implement the SimGCL view-invariance
contrastive objective (rev54 §3.1, eq. on line 172, citing Yu et al.
SIGIR 2022).

Merge order (do each one as a separate commit for auditable diffs):

    (1)  Constructor signature   -- add ``enable_simgcl``, ``simgcl_eps``,
                                    ``simgcl_batch_size_user``,
                                    ``simgcl_batch_size_item`` to __init__.
    (2)  LightGCN refactor       -- extract block E (model.py:553-565)
                                    into a callable
                                    ``_lightgcn_propagate(ego_u, ego_i)``.
                                    Default forward path stays bit-for-bit
                                    identical.
    (3)  SimGCL forward method   -- add ``simgcl_view_forward(...)`` that
                                    runs two perturbed propagations and
                                    returns the view-invariance loss.
    (4)  train.py loss patch     -- add ``lambda_view * L_view`` to the
                                    total loss with a feature-flag gate.
    (5)  utility/parser.py flags -- expose ``--enable_simgcl``, ``--simgcl_eps``,
                                    ``--lambda_view``, ``--simgcl_layers``.
    (6)  Trainer wiring          -- pass the new CLI args into the model
                                    constructor.

After (1)-(6), run the unit tests in ``tests/test_simgcl.py`` and a
1-epoch smoke training with ``--enable_simgcl 0`` to confirm
**bit-for-bit identical** outputs vs. the Wave 1 baseline (LogQ-only at
scale=1.0). Only then flip ``--enable_simgcl 1`` and proceed with the
M1 sweep over ``--lambda_view in {0.01, 0.05, 0.1}``.

Code-line evidence for the targeted hooks:
    * Constructor body where Wave 1 LogQ kwargs were appended:
      ``model.py:188-198``  (Wave 1 added 3 kwargs at this position)
    * Block E -- LightGCN propagation loop to refactor:
      ``model.py:553-565``  (the for-loop over self.User_layers)
    * Total-loss assembly in train.py:
      ``train.py:575-590``  (where bcl_item / bcl_user are already wired)
    * Argument parser:
      ``utility/parser.py``  (search for ``--enable_logq`` for layout)
    * Trainer model construction:
      ``train.py:279-310``   (where Wave 1 kwargs were forwarded)
"""

# =========================================================================
#  Block (1) -- Constructor signature additions
# =========================================================================
# Append four NEW kwargs to the existing __init__ signature, immediately
# AFTER the Wave 1 LogQ kwargs added at model.py:194-197.
#
# BEFORE (Wave 1 tail of __init__ signature):
#     enable_logq: bool = False,
#     logq_scale: float = 1.0,
#     logq_clip: float = 5.0,
# ) -> None:
#
# AFTER (Wave 2):
#     enable_logq: bool = False,
#     logq_scale: float = 1.0,
#     logq_clip: float = 5.0,
#     # --- Wave 2 / M1 -- SimGCL view-invariance (Yu et al. SIGIR 2022) ---
#     enable_simgcl: bool = False,           # toggles the third contrastive term
#     simgcl_eps: float = 0.1,               # noise magnitude (rev54 default 0.1)
#     simgcl_batch_size_user: int = 4096,    # row-chunk for user-branch L_view
#     simgcl_batch_size_item: int = 4096,    # row-chunk for item-branch L_view
# ) -> None:


# =========================================================================
#  Block (2) -- LightGCN propagation refactor (model.py:553-565)
# =========================================================================
# The current block E inlines the layer loop directly into ``forward``.
# Wave 2 requires the loop to be **callable three times per forward step**:
# once on the anchor (clean) ego, once on each of the two perturbed egos.
# The refactor is purely a code-motion: NO behavioural change when
# ``enable_simgcl=False``.
#
# REPLACE (model.py:553-565, inside forward()):
#
#     ego = torch.cat([self.user_ui_embedding.weight,
#                      self.item_ui_embedding.weight], dim=0)
#     all_embs = [ego]
#     for _ in range(self.User_layers):
#         ego = torch.sparse.mm(self.UI_mat, ego)
#         all_embs.append(ego)
#     final = torch.stack(all_embs, dim=1).mean(dim=1)
#     u_ui_emb = final[:self.n_users]
#     i_ui_emb = final[self.n_users:]
#
# WITH:
#
#     u_ui_emb, i_ui_emb = self._lightgcn_propagate(
#         self.user_ui_embedding.weight,
#         self.item_ui_embedding.weight,
#     )
#
# And add the new method as a sibling of ``batched_contrastive_loss``:

_BLOCK_2_LIGHTGCN_PROPAGATE = '''
    def _lightgcn_propagate(
        self,
        ego_user: torch.Tensor,
        ego_item: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """LightGCN propagation extracted from the rev45 forward() block E.

        Concatenates user/item ego embeddings, runs ``self.User_layers``
        rounds of sparse propagation on the normalised bipartite adjacency
        ``self.UI_mat``, layer-averages the trace, and splits the result.

        This method is the **only** propagation surface in the codebase
        after the Wave 2 refactor. It is invoked exactly once per
        ``forward()`` call on the anchor ego, and twice more from
        ``simgcl_view_forward`` on perturbed egos when SimGCL is enabled.

        Backward compatibility:
            Passing the unmodified ``user_ui_embedding.weight`` and
            ``item_ui_embedding.weight`` reproduces the rev45/Wave 1
            propagation result **bit-for-bit** (see ``tests/test_simgcl.py``
            :: ``test_propagation_refactor_identity``).
        """
        ego = torch.cat([ego_user, ego_item], dim=0)         # (n_u + n_i, d)
        all_embs = [ego]
        for _ in range(self.User_layers):
            ego = torch.sparse.mm(self.UI_mat, ego)
            all_embs.append(ego)
        final = torch.stack(all_embs, dim=1).mean(dim=1)     # rev45 LightGCN avg
        return final[:self.n_users], final[self.n_users:]
'''


# =========================================================================
#  Block (3) -- SimGCL view-forward + view-loss
# =========================================================================
# Add this method immediately AFTER ``_lightgcn_propagate``. It delegates
# all the math to the standalone helper in ``damps/damps_simgcl.py``
# (shipped as ``damps_simgcl.py`` at the project root for the smoke test;
# move to ``damps/`` when promoting to a permanent module).
#
# The method is a no-op when ``enable_simgcl=False``. Importantly, the
# noise is sampled AFRESH inside this call -- callers MUST invoke it on
# the same step as the main forward pass, never on a cached trace.

_BLOCK_3_VIEW_FORWARD = '''
    def simgcl_view_forward(self) -> torch.Tensor:
        """Compute L_SimGCL = (1/2)(L_view^user + L_view^item).

        Two perturbed LightGCN propagations are run; the perturbations are
        sampled fresh inside ``inject_uniform_noise``. The output is a
        scalar tensor with gradient flow back into the ego embedding
        parameters. The caller (train.py) is responsible for multiplying
        by ``lambda_view`` before adding to the total loss.

        Returns:
            Scalar loss tensor. Returns a 0.0 tensor (no grad) when
            ``enable_simgcl=False`` so that bracketing this call inside
            the training loop is unconditionally safe.
        """
        if not self.enable_simgcl:
            return torch.zeros((), device=self.user_ui_embedding.weight.device)

        # Import inside the method to keep the constructor free of side
        # effects when SimGCL is disabled at deploy time.
        from damps_simgcl import compute_simgcl_view_loss

        return compute_simgcl_view_loss(
            propagate_fn=self._lightgcn_propagate,
            ego_user=self.user_ui_embedding.weight,
            ego_item=self.item_ui_embedding.weight,
            eps=self.simgcl_eps,
            tau=self.tau,
            batch_size_user=self.simgcl_batch_size_user,
            batch_size_item=self.simgcl_batch_size_item,
        )
'''


# =========================================================================
#  Block (4) -- train.py loss-assembly patch
# =========================================================================
# Add the SimGCL view-invariance term to the total loss at
# train.py:575-590, immediately AFTER bcl_user is computed. Crucially,
# the addition is gated by ``args.enable_simgcl`` so the Wave 1 baseline
# is reproducible by simply passing ``--enable_simgcl 0`` (the default).
#
# BEFORE (after the Wave 1 LogQ call-site patch):
#
#     bcl_item = self.model.batched_contrastive_loss(
#         out["i_ui_emb"], out["ii_emb"], apply_logq=True,
#     ) * args.item_loss_ratio
#     bcl_user = self.model.batched_contrastive_loss(
#         out["u_ui_emb"], out["uu_emb"], apply_logq=False,
#     ) * args.user_loss_ratio
#     loss = bpr_loss + reg_loss + bcl_item + bcl_user
#
# AFTER (Wave 2):

_BLOCK_4_TRAIN_PATCH = '''
    bcl_item = self.model.batched_contrastive_loss(
        out["i_ui_emb"], out["ii_emb"], apply_logq=True,
    ) * args.item_loss_ratio
    bcl_user = self.model.batched_contrastive_loss(
        out["u_ui_emb"], out["uu_emb"], apply_logq=False,
    ) * args.user_loss_ratio

    # ---- Wave 2 / M1 -- SimGCL view-invariance (Yu et al. SIGIR 2022) ----
    # ``simgcl_view_forward`` is a hard no-op when args.enable_simgcl=0
    # so the Wave 1 LogQ-only baseline is bit-for-bit reproducible.
    if args.enable_simgcl:
        l_view = self.model.simgcl_view_forward() * args.lambda_view
    else:
        l_view = torch.zeros((), device=bcl_item.device)

    loss = bpr_loss + reg_loss + bcl_item + bcl_user + l_view

    # ---- Telemetry hook -- log the raw L_view so the M1 sweep can audit
    # gradient interference at λ_view in {0.01, 0.05, 0.1}.
    if args.enable_simgcl and (it % args.verbose == 0):
        wandb.log({"loss/simgcl_view": float(l_view.detach()),
                   "loss/total":       float(loss.detach())},
                  step=epoch * len(self.train_loader) + it,
                  commit=False)
'''


# =========================================================================
#  Block (5) -- utility/parser.py argument additions
# =========================================================================
# Append the following lines immediately AFTER the Wave 1 LogQ flags.
# Defaults are chosen so existing scripts (Wave 1 sweeps, smoke tests,
# unit tests) continue to behave identically unless ``--enable_simgcl 1``
# is passed.

_BLOCK_5_PARSER_FLAGS = '''
    # ---- Wave 2 / M1 -- SimGCL view-invariance (Yu et al. SIGIR 2022) ----
    parser.add_argument(
        "--enable_simgcl", type=int, default=0,
        help="Master switch for the SimGCL view-invariance term. "
             "0 = Wave 1 LogQ-only baseline (default); 1 = Wave 2 M1.",
    )
    parser.add_argument(
        "--simgcl_eps", type=float, default=0.1,
        help="Magnitude of the uniform-noise perturbation injected into "
             "ego embeddings before the LightGCN propagation. "
             "Yu et al. (SIGIR 2022) recommend 0.1; the rev54 Optuna "
             "search range is [0.05, 0.2].",
    )
    parser.add_argument(
        "--lambda_view", type=float, default=0.05,
        help="Weight of L_SimGCL in the total loss. The M1 ablation "
             "sweep covers {0.01, 0.05, 0.1}; the rollback gate requires "
             "Recall@20 >= 0.0890 on every seed for ALL three values.",
    )
    parser.add_argument(
        "--simgcl_batch_size_user", type=int, default=4096,
        help="Row-chunk size for the user-branch view-invariance loss.",
    )
    parser.add_argument(
        "--simgcl_batch_size_item", type=int, default=4096,
        help="Row-chunk size for the item-branch view-invariance loss.",
    )
'''


# =========================================================================
#  Block (6) -- Trainer model construction
# =========================================================================
# Forward the new CLI flags into the model constructor at train.py:279-310,
# immediately AFTER the Wave 1 LogQ kwargs.
#
# BEFORE (Wave 1 tail of model kwargs):
#     enable_logq=bool(args.enable_logq),
#     logq_scale=float(args.logq_scale),
#     logq_clip=float(args.logq_clip),
# )
#
# AFTER (Wave 2):

_BLOCK_6_MODEL_CONSTRUCT = '''
    self.model = MMHCL(
        # ... (all rev45 + Wave 1 kwargs) ...
        enable_logq=bool(args.enable_logq),
        logq_scale=float(args.logq_scale),
        logq_clip=float(args.logq_clip),
        # ---- Wave 2 / M1 ----
        enable_simgcl=bool(args.enable_simgcl),
        simgcl_eps=float(args.simgcl_eps),
        simgcl_batch_size_user=int(args.simgcl_batch_size_user),
        simgcl_batch_size_item=int(args.simgcl_batch_size_item),
    )
'''


# =========================================================================
#  Post-merge sanity protocol (DO NOT SKIP)
# =========================================================================
#  S1. Run ``pytest tests/test_simgcl.py -x`` -- the four tests must pass.
#  S2. Smoke run with ``--enable_simgcl 0`` for 5 epochs. Compare every
#      BEST_Val_* and BEST_Test_* metric against the Wave 1 LogQ-only
#      baseline (logq_scale=1.0). Difference must be 0 to float precision.
#      Bit-for-bit reproduction is the gate for proceeding to S3.
#  S3. Smoke run with ``--enable_simgcl 1 --lambda_view 0.05`` for 5
#      epochs. Inspect ``loss/simgcl_view`` in W&B: it must be finite,
#      monotonically (weakly) decreasing on average over the first
#      5 epochs, and bounded above by 5.0 (typical at d=64, n<=30k).
#  S4. Only after S1-S3 succeed, launch the M1 sweep via
#      ``cell32_m1_sweep_driver.py``.
