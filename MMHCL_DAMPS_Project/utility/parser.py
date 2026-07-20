"""
utility/parser.py — DAMPS-MMHCL command-line argument parser
==============================================================

Centralises every hyperparameter and configuration flag used by the
DAMPS-MMHCL training script (``train.py``).

Compatible with the original MMHCL CLI surface (every flag from
``codes/utility/parser.py`` is preserved) plus the new DAMPS-specific
options listed in Section 3 of the DAMPS spec.

Phase-1 / rev44 / Revision 11 defaults (Quick Win)
--------------------------------------------------
* ``--temperature 0.3`` (was 0.1)
* ``--learnable_tau 0`` (new flag; rev42 used a learnable nn.Parameter, which
  empirically saturates at ~0.0909 and triggers an embedding collapse).
* ``--damps_avrf 0`` (was 1; rev44 disables AVRF for Phase 1 to recover
  Recall@20 coverage on the sparse Amazon Clothing dataset).

Usage examples
--------------
::

    # Default invocation == rev44 Phase 1 recommended config (d):
    python train.py --dataset Clothing --seed 42

    # Reproduce the rev42 / Revision 9 baseline anchor (a):
    python train.py --dataset Clothing --seed 42 \\
        --temperature 0.1 --learnable_tau 1 --damps_avrf 1

    # Phase 1 variant (b) -- only the static-tau fix:
    python train.py --dataset Clothing --seed 42 \\
        --temperature 0.3 --learnable_tau 0 --damps_avrf 1

    # Phase 1 variant (c) -- only the AVRF-off fix:
    python train.py --dataset Clothing --seed 42 \\
        --temperature 0.1 --learnable_tau 1 --damps_avrf 0
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    """Parse and return all DAMPS-MMHCL CLI arguments."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=(
            "DAMPS-MMHCL: Multi-Modal Hypergraph Contrastive Learning "
            "with Spectral Domain Representation Calibration"
        )
    )

    # =====================================================================
    #  General / Data
    # =====================================================================
    parser.add_argument("--data_path", type=str, default="../data/",
                        help="Root path to all dataset folders.")
    parser.add_argument("--seed", type=int, default=2024,
                        help="Random seed for reproducibility.")
    parser.add_argument("--dataset", type=str, default="Clothing",
                        help="Dataset name: {Tiktok, Sports, Clothing}.")
    parser.add_argument("--core", type=int, default=5,
                        help="K-core filtering threshold.")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU index to use.")
    parser.add_argument("--debug", default="True",
                        help='If "True", enable per-run file logging.')

    # =====================================================================
    #  Training Schedule
    # =====================================================================
    parser.add_argument("--verbose", type=int, default=5,
                        help="Run validation every N epochs.")
    parser.add_argument("--epoch", type=int, default=250,
                        help="Maximum number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="Number of BPR triplets per mini-batch.")
    parser.add_argument("--regs", type=float, default=1e-3,
                        help="L2 regularisation coefficient (BPR).")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Initial learning rate for Adam.")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0,
                        help="Max L2 norm for gradient clipping (0 = disabled).")

    # =====================================================================
    #  Model Architecture (MMHCL backbone)
    # =====================================================================
    parser.add_argument("--embed_size", type=int, default=64,
                        help="User/item embedding dimensionality.")
    parser.add_argument("--weight_size", type=str, default="[64,64,64]",
                        help="Per-layer GNN sizes (NGCF only).")
    parser.add_argument("--topk", type=int, default=5,
                        help="K for the K-NN sparsification.")
    parser.add_argument("--cf_model", type=str, default="LightGCN",
                        help="CF backbone: {MF, NGCF, LightGCN}.")
    parser.add_argument("--norm_type", type=str, default="sym",
                        help="Adjacency normalisation: {sym, rw, origin}.")
    parser.add_argument("--sparse", type=int, default=0,
                        help="1 = sparse UI graph; 0 = dense.")
    parser.add_argument("--UI_layers", type=int, default=2,
                        help="GNN layers on user-item bipartite graph.")
    parser.add_argument("--User_layers", type=int, default=3,
                        help="GNN layers on user-user co-interaction graph.")
    parser.add_argument("--Item_layers", type=int, default=2,
                        help="GNN layers on item-item multi-modal hypergraph.")
    parser.add_argument("--user_loss_ratio", type=float, default=0.03,
                        help="Weight for user-side contrastive loss.")
    parser.add_argument("--item_loss_ratio", type=float, default=0.07,
                        help="Weight for item-side contrastive loss.")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="InfoNCE temperature tau. Default is 0.3 to "
                             "match the Revision 11 / rev44 Phase 1 anchor "
                             "(static tau sweep set {0.2, 0.3, 0.5}). To "
                             "reproduce the Revision 9 / rev42 baseline, "
                             "pass --learnable_tau 1 --temperature 0.1.")
    parser.add_argument("--learnable_tau", type=int, default=0,
                        help="0 = static tau (rev44 Phase 1 default; tau "
                             "is a non-trainable buffer fixed at "
                             "--temperature throughout training); "
                             "1 = learnable tau (rev42 baseline; tau is an "
                             "nn.Parameter initialised at --temperature, "
                             "clamped to >= 0.01). Set to 0 to break the "
                             "tau-saturation embedding-collapse failure mode "
                             "documented in rev44 Section 3.")

    # =====================================================================
    #  Evaluation
    # =====================================================================
    parser.add_argument("--Ks", type=str, default="[10,20]",
                        help="Cut-offs for Recall/Precision/NDCG/Hit@K.")
    parser.add_argument("--test_flag", type=str, default="part",
                        help="{part, full}. 'part' = heapq top-K (faster).")

    # =====================================================================
    #  Early Stopping
    # =====================================================================
    parser.add_argument("--early_stopping_patience", type=int, default=5,
                        help="Non-improving evaluation cycles before stop.")
    parser.add_argument("--early_stopping_min_epochs", type=int, default=0,
                        help="Min epochs before early stopping can trigger.")
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-4,
                        help="Min metric improvement to count as progress.")
    parser.add_argument("--early_stopping_mode", type=str, default="max",
                        help="{max, min}.")
    parser.add_argument("--early_stopping_restore_best", type=int, default=1,
                        help="1 = restore best weights on stop.")
    parser.add_argument(
        "--early_stopping_monitor",
        type=str,
        default="val_recall@20",
        help="Validation metric that drives the early-stopping patience "
             "counter. Accepted forms: 'val_recall@K', 'val_ndcg@K', "
             "'val_precision@K' (K must appear in --Ks), or the aliases "
             "'recall' / 'ndcg' / 'precision' (uses the last K in --Ks). "
             "Peak-snapshot bookkeeping for recall AND ndcg is unchanged; "
             "only the patience reset rule is gated by this flag. "
             "Default 'val_recall@20' matches the PACER / smoke protocol.",
    )
    parser.add_argument(
        "--adaptive_patience",
        type=int,
        default=0,
        help="0 = fixed --early_stopping_patience (default; smoke / "
             "PACER tercile protocol). "
             "1 = grow effective patience by +1 every 50 epochs after "
             "--early_stopping_min_epochs (mild schedule; keeps long "
             "runs from stopping too aggressively on noisy plateaus).",
    )

    # =====================================================================
    #  ReduceLROnPlateau
    # =====================================================================
    parser.add_argument("--use_reduce_lr", type=int, default=0)
    parser.add_argument("--reduce_lr_factor", type=float, default=0.5)
    parser.add_argument("--reduce_lr_patience", type=int, default=3)
    parser.add_argument("--reduce_lr_min", type=float, default=1e-6)

    # =====================================================================
    #  W&B Tracking
    # =====================================================================
    parser.add_argument("--use_wandb", type=int, default=0)
    parser.add_argument("--wandb_project", type=str, default="damps-mmhcl")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument(
        "--wandb_group", type=str, default="",
        help="W&B run group; groups related runs together in the W&B UI "
             "(e.g. 'wave2_branchA'). Empty string = no group.",
    )
    parser.add_argument(
        "--wandb_tags", type=str, default="",
        help="Comma-separated W&B tags attached to the run "
             "(e.g. 'branchA,batchN,rev55'). Empty string = no tags.",
    )
    parser.add_argument(
        "--wandb_job_type", type=str, default="",
        help="W&B job_type label (e.g. 'train', 'sweep_seed'). "
             "Empty string = W&B default.",
    )
    parser.add_argument(
        "--wandb_name", type=str, default="",
        help="W&B run name (alias for --wandb_run_name; takes precedence "
             "when both are set). Accepted by the RQ2 ablation runner and "
             "any script that follows the W&B CLI convention.",
    )

    # =====================================================================
    #  DAMPS-Specific (Revision 9)
    # =====================================================================
    parser.add_argument("--damps_apc", type=int, default=1,
                        help="1 = enable Metadata-Aware Adaptive Phase Calibration.")
    parser.add_argument("--damps_avrf", type=int, default=0,
                        help="0 = AVRF off (rev44 Phase 1 default; preserves "
                             "sparse signals on Amazon Clothing where AVRF "
                             "tends to over-attenuate). "
                             "1 = AVRF on (rev42 baseline; logit-clipped "
                             "Wiener gate). Phase 1 ablation explicitly sets "
                             "this to 0 to recover Recall@20 coverage.")
    parser.add_argument("--damps_imcf", type=int, default=1,
                        help="1 = enable Residual Inter-Modal Coherence Filter.")
    parser.add_argument("--damps_permutation_fft", type=int, default=0,
                        help="1 = run the falsifiable Permutation-FFT ablation.")
    parser.add_argument("--damps_soft_routing", type=int, default=1,
                        help="1 = use Soft Residual-Routing into HGCN; "
                             "0 = forward h_cal directly (will trigger over-smoothing).")
    parser.add_argument("--damps_momentum", type=int, default=1,
                        help="1 = enable Slim Momentum Encoder.")
    parser.add_argument("--damps_data_driven_prior", type=int, default=1,
                        help="1 = compute AVRF priors from raw modality features; "
                             "0 = use the hard-coded fallback (0.24/0.85).")
    parser.add_argument("--damps_num_categories", type=int, default=10,
                        help="Number of static metadata clusters for APC.")
    parser.add_argument("--damps_warmup_epochs", type=int, default=10,
                        help="Warm-up window for adaptive EMA schedules.")

    # =====================================================================
    #  rev53 §3.1 — LogQ popularity correction (variant "h")
    # =====================================================================
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
                             "variant; 'raw' requires every item >= 1 "
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

    # =====================================================================
    #  Wave 2 / M1 -- SimGCL view-invariance (Yu et al. SIGIR 2022)
    # =====================================================================
    parser.add_argument(
        "--enable_simgcl", type=int, default=0,
        help="Master switch for the SimGCL view-invariance term. "
             "0 = Wave 1 LogQ-only baseline (default); 1 = Wave 2 M1.",
    )
    parser.add_argument(
        "--simgcl_eps", type=float, default=0.1,
        help="Magnitude of the uniform-noise perturbation injected into "
             "ego embeddings before LightGCN propagation. "
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

    # =====================================================================
    #  Branch A' -- NRDMC-lite learnable view generators (rev55 §8.2)
    #  Replaces the SimGCL noise-based views with SAV + IAV + adaptive
    #  fusion (NRDMC IPM 2026 Eq. 14, 16, 17-19). PTV is dropped per §8.2.
    # =====================================================================
    parser.add_argument(
        "--enable_nrdmc_lite", type=int, default=0,
        help="1 = replace the SimGCL view path with the NRDMC-lite "
             "learnable SAV+IAV+adaptive-fusion view generators from "
             "rev55 §8.2 (Branch A'). Mutually exclusive with "
             "--enable_simgcl (train.py refuses both being on).",
    )
    parser.add_argument(
        "--nrdmc_lite_layers", type=int, default=2,
        help="Number of LightGCN steps to propagate over the learned "
             "contrastive graph inside the NRDMC-lite view module. "
             "Default 2 keeps the extra compute below 1%% of an epoch.",
    )

    # =====================================================================
    #  Pattern B' (Scheduled Rebuild)
    # =====================================================================
    parser.add_argument("--rebuild_R", type=int, default=5,
                        help="Rebuild K-NN hypergraph every R epochs.")
    parser.add_argument("--faiss_threshold", type=int, default=60_000,
                        help="N >= threshold -> switch to FAISS HNSW path.")
    parser.add_argument("--knn_chunk_size", type=int, default=4_096,
                        help="Row-chunk size for the chunked PyTorch K-NN path.")
    parser.add_argument("--faiss_use_gpu", type=int, default=1,
                        help="1 = use FAISS GPU resources when available "
                             "(StandardGpuResources + index_cpu_to_gpu); "
                             "0 = stay on CPU FAISS. Speedup guide Section 2.")
    parser.add_argument("--knn_efsearch", type=int, default=64,
                        help="HNSW efSearch parameter (controls recall/speed "
                             "trade-off). Higher = better recall, slower.")

    # =====================================================================
    #  Mixed Precision (bfloat16)
    # =====================================================================
    parser.add_argument("--use_amp", type=int, default=1,
                        help="1 = enable bfloat16 mixed precision training.")

    # =====================================================================
    #  Training Speedup Flags (Speedup Guide, Sections 1-10)
    # =====================================================================
    parser.add_argument("--use_torch_compile", type=int, default=0,
                        help="1 = wrap the DAMPS spectral block in "
                             "torch.compile (PyTorch >= 2.0). Reported "
                             "+25-35%% speedup on the GNN forward path.")
    parser.add_argument("--torch_compile_mode", type=str, default="reduce-overhead",
                        help="torch.compile mode: {default, reduce-overhead, "
                             "max-autotune}. 'reduce-overhead' is best for "
                             "medium-sized models per the speedup guide.")
    parser.add_argument("--torch_compile_dynamic", type=int, default=1,
                        help="1 = compile with dynamic=True (required because "
                             "the BPR triplet batch size can vary by 1 between "
                             "epochs).")

    # =====================================================================
    #  Ablation Tag
    # =====================================================================
    parser.add_argument("--ablation_target", type=str, default="",
                        help="Tag for the experiment directory.")

    return parser.parse_args()
