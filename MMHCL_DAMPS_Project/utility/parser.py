"""
utility/parser.py — DAMPS-MMHCL command-line argument parser
==============================================================

Centralises every hyperparameter and configuration flag used by the
DAMPS-MMHCL training script (``train.py``).

Compatible with the original MMHCL CLI surface (every flag from
``codes/utility/parser.py`` is preserved) plus the new DAMPS-specific
options listed in Section 3 of the DAMPS spec.

Usage examples
--------------
::

    python train.py --dataset Clothing --seed 42
    python train.py --dataset Tiktok --damps_apc 1 --damps_avrf 1 --damps_imcf 1
    python train.py --dataset Sports --rebuild_R 5 --use_amp 1 --use_wandb 1
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
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Initialisation value for the *learnable* "
                             "InfoNCE temperature tau (Revision 9 spec, "
                             "Section 3.1: tau is an nn.Parameter init at "
                             "0.1, dynamically clamped to >= 0.01).")

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

    # =====================================================================
    #  DAMPS-Specific (Revision 9)
    # =====================================================================
    parser.add_argument("--damps_apc", type=int, default=1,
                        help="1 = enable Metadata-Aware Adaptive Phase Calibration.")
    parser.add_argument("--damps_avrf", type=int, default=1,
                        help="1 = enable AVRF (logit-clipped Wiener gate).")
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
