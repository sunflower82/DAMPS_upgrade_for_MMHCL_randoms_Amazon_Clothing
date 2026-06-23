"""
branchA_train_patch.py -- Surgical patches to MMHCL_DAMPS_Project/train.py.
===========================================================================

Two MINIMAL changes are required in ``train.py`` to wire up Branch A:

(T1) Pass the current epoch into ``simgcl_view_forward``.
(T2) Forward the four Branch A CLI flags into the MMHCL constructor.

The patches are bit-for-bit identical to Wave 1 when the new flags are
left at their defaults (``--branchA_view_every_k 1``, ``--branchA_bcl_batchn 0``).
"""

# =============================================================================
#  Patch (T1) -- train.py line 625-628 -- pass ``epoch`` to view forward.
# =============================================================================
# BEFORE:
#     if args.enable_simgcl:
#         l_view = (
#             self.model.simgcl_view_forward() * args.lambda_view
#         )
#     else:
#         l_view = torch.zeros((), device=bcl_item.device)
#
# AFTER:
#     if args.enable_simgcl:
#         l_view = (
#             self.model.simgcl_view_forward(epoch=epoch) * args.lambda_view
#         )
#     else:
#         l_view = torch.zeros((), device=bcl_item.device)

_PATCH_T1 = '''
                    if args.enable_simgcl:
                        l_view = (
                            self.model.simgcl_view_forward(epoch=epoch) * args.lambda_view
                        )
                    else:
                        l_view = torch.zeros((), device=bcl_item.device)
'''


# =============================================================================
#  Patch (T2) -- train.py line 295-310 -- forward Branch A flags.
# =============================================================================
# In the ``self.model = MMHCL(...)`` constructor call, immediately AFTER the
# Wave 2 SimGCL kwargs (``simgcl_batch_size_item=...``), ADD:
#
#     # ---- Branch A (rev55 §8.1) ----
#     branchA_view_every_k = int(args.branchA_view_every_k),
#     branchA_bcl_batchn   = bool(args.branchA_bcl_batchn),
#     branchA_view_bsz     = int(args.branchA_view_bsz),
#     branchA_bcl_bsz      = int(args.branchA_bcl_bsz),

_PATCH_T2 = '''
            # ---- Branch A (rev55 §8.1) ----
            branchA_view_every_k = int(args.branchA_view_every_k),
            branchA_bcl_batchn   = bool(args.branchA_bcl_batchn),
            branchA_view_bsz     = int(args.branchA_view_bsz),
            branchA_bcl_bsz      = int(args.branchA_bcl_bsz),
'''


# =============================================================================
#  Patch (T3) -- AMP sanity (already present in rev54 train.py L566-597).
# =============================================================================
# Branch A does NOT require GradScaler because rev54 already runs the forward
# pass under ``torch.amp.autocast(dtype=torch.bfloat16)``. Make sure the user
# launches with ``--use_amp 1`` (the default) so the (B, B) matmul lands on
# Tensor Cores at bf16. No code change is needed -- only verify the CLI flag.
#
# If the user is on Ampere (RTX 3090, A100) which lacks fast bf16-on-FP32
# accumulation kernels, switch to:
#     amp_dtype = torch.float16    # train.py:566
# and add a GradScaler. RTX 5090 (Ada Lovelace + FP8 paths) handles bf16 at
# full Tensor Core rate, so the rev54 default is optimal.


# =============================================================================
#  Patch (T4) -- (RECOMMENDED) reduce data_generator batch size for SimGCL.
# =============================================================================
# The bottleneck after batch-N is the data-loading subroutine that fetches
# BPR negatives. The rev54 default ``--batch_size 4096`` is appropriate for
# BPR; the Branch A row-chunks for L_view and L_NCEQ are set independently
# via the new ``--branchA_view_bsz`` and ``--branchA_bcl_bsz`` flags, so the
# BPR batch size can stay at 4096 without affecting Branch A speed.
#
# However, if the user wants ONE more 15-20% wall-clock reduction, raising
# the BPR batch size to ``--batch_size 8192`` is safe (no OOM at d=64,
# N=23k, 24GB RTX 5090) and reduces ``n_batch`` per epoch by 50%.
