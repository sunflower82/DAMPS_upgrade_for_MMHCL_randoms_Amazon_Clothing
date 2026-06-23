"""
Branch A — Parser Patch
=======================

Adds 4 new CLI flags for Branch A speedups (rev55 §8.1, Wave 2 Phase 2):

    --branchA_view_every_k     Compute L_view every K epochs (default 2)
    --branchA_bcl_batchn       Use batch-N InfoNCE for bcl_item (default 1)
    --branchA_view_bsz         Row-chunk size for SimGCL view loss (default 2048)
    --branchA_bcl_bsz          Row-chunk size for bcl_item batch-N (default 2048)

INSTRUCTIONS
------------
Open  MMHCL_DAMPS_Project/utility/parser.py  and paste the BLOCK BELOW
immediately AFTER the existing Wave 2 SimGCL block, i.e. after the line:

    parser.add_argument(
        "--simgcl_batch_size_item", type=int, default=4096,
        help="Row-chunk size for the item-branch view-invariance loss.",
    )

(currently at parser.py line ~249).

The new block lives between the SimGCL section and the existing
"Pattern B' (Scheduled Rebuild)" section.

VERIFICATION
------------
After patching:

    python -c "from utility.parser import parse_args; \\
               a = parse_args(['--dataset','clothing']); \\
               print(a.branchA_view_every_k, a.branchA_bcl_batchn, \\
                     a.branchA_view_bsz, a.branchA_bcl_bsz)"

Expected output:

    2 1 2048 2048
"""

# =============================================================================
#  PASTE THE BLOCK BELOW INTO  utility/parser.py  AFTER LINE ~249
# =============================================================================

PATCH_PARSER = r"""
    # =====================================================================
    #  Branch A -- speedup overlays for Wave 2 SimGCL (rev55 §8.1)
    # =====================================================================
    parser.add_argument(
        "--branchA_view_every_k", type=int, default=2,
        help="Compute the SimGCL view-invariance loss every K epochs and "
             "reuse the cached perturbed views on the off-epochs. "
             "K=1 reproduces the dense Wave 2 schedule; K=2 halves the "
             "number of perturbed LightGCN propagations and is the "
             "Branch A default. Set K=1 for the S2 bit-for-bit smoke "
             "test against Wave 1.",
    )
    parser.add_argument(
        "--branchA_bcl_batchn", type=int, default=1,
        help="1 = replace the (B, N) chunked InfoNCE in "
             "batched_contrastive_loss with a batch-N InfoNCE that "
             "compares each anchor against the (B-1) other rows of the "
             "mini-batch (Branch A default; ~22x FLOPs reduction on "
             "Amazon-Clothing). 0 = keep the legacy (B, N) path used in "
             "Wave 1 / Wave 2 audit runs.",
    )
    parser.add_argument(
        "--branchA_view_bsz", type=int, default=2048,
        help="Row-chunk size used by the Branch A batch-N SimGCL view "
             "loss. Must be <= simgcl_batch_size_user / "
             "simgcl_batch_size_item; the 2048 default keeps the per-chunk "
             "(B, B) Gram matrix under 16 MB FP32.",
    )
    parser.add_argument(
        "--branchA_bcl_bsz", type=int, default=2048,
        help="Row-chunk size used by the Branch A batch-N bcl_item "
             "contrastive loss when --branchA_bcl_batchn=1. Trades VRAM "
             "for throughput; 2048 matches the SimGCL chunk for cache "
             "reuse on a single A100 / RTX 4090.",
    )
"""

# =============================================================================
#  HOW TO APPLY (one-liner, from inside MMHCL_DAMPS_Project/)
# =============================================================================
#
#   1. Open  utility/parser.py  in your editor.
#   2. Find the line:
#          help="Row-chunk size for the item-branch view-invariance loss.",
#      and the closing  )  of that  add_argument  block (line ~249).
#   3. Paste the PATCH_PARSER block (the four add_argument calls) on the
#      next blank line, BEFORE the "# Pattern B' (Scheduled Rebuild)" header.
#   4. Save. No other files in utility/ require changes.
#
#  Sanity check (must print "2 1 2048 2048"):
#
#      cd MMHCL_DAMPS_Project
#      python -c "from utility.parser import parse_args as P; a = P(['--dataset','clothing']); print(a.branchA_view_every_k, a.branchA_bcl_batchn, a.branchA_view_bsz, a.branchA_bcl_bsz)"
#
# =============================================================================

if __name__ == "__main__":
    print(__doc__)
    print("---- BLOCK TO PASTE INTO utility/parser.py ----")
    print(PATCH_PARSER)
