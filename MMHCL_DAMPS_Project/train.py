"""
train.py — DAMPS-MMHCL Training Orchestrator
==============================================

Production entry point that wires together the rest of the package:

    parser       ──>  args
    load_data    ──>  data_generator   (raw modality + graphs)
    model        ──>  DAMPS_MMHCL      (network)
    batch_test   ──>  test_torch       (evaluation)
    logging      ──>  Logger
    DualPathKNN  ──>  Pattern B' scheduled K-NN rebuild

Per-epoch training loop
-----------------------
1.  (every R epochs after warm-up) Rebuild the item-item hypergraph via
    ``DualPathKNN`` reading from the **Slim Momentum** tables, then log NNZ.
2.  Sample a BPR triplet batch.
3.  Forward pass through ``DAMPS_MMHCL`` under **bfloat16 AMP**:
        BPR loss + L2 reg + λ_item·NCE(item) + λ_user·NCE(user)
4.  Back-prop with gradient clipping, then ``optimizer.step()``.
5.  Update per-epoch EMA MAD diagnostic.
6.  (every ``verbose`` epochs) Run validation; if it improves, run test.

CRITICAL FIX
------------
``torch.cuda.empty_cache()`` is **never** called inside the training loop —
that would shred allocator efficiency and trigger >50% wall-clock slowdowns.
The cuFFT plan cache is permanently disabled at startup via:

    torch.backends.cuda.cufft_plan_cache[device.index].max_size = 1

(per Section 3.3 of the DAMPS spec).
"""

from __future__ import annotations

import copy
import math
import os
import pathlib
import random
import re
import sys
from time import time
from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
import torch.optim as optim

# -------------------------------------------------------------------------
# Speedup guide Section F steps 1 + B.5 -- TF32 / cuDNN / reduced-precision
# -------------------------------------------------------------------------
# Blackwell fully supports TF32 (~8x fp32 matmul throughput). Fixed shapes
# throughout PACER training make cudnn.benchmark a pure win after the first
# autotune batch. Reduced-precision reductions speed sparse.mm on sm_120.
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    _rp = torch.backends.cuda.matmul
    if hasattr(_rp, "allow_fp16_reduced_precision_reduction"):
        _rp.allow_fp16_reduced_precision_reduction = True
    if hasattr(_rp, "allow_bf16_reduced_precision_reduction"):
        _rp.allow_bf16_reduced_precision_reduction = True

# Cap CPU oversubscription when many grid subprocesses share one host
# (speedup guide Section C -- torch.set_num_threads).
_cpu_threads = int(os.environ.get("PACER_NUM_THREADS", "4"))
torch.set_num_threads(max(1, _cpu_threads))

# -------------------------------------------------------------------------
# torch.compile + complex-FFT backward Inductor crash workaround
# -------------------------------------------------------------------------
# PyTorch's Inductor backward compiler on Windows/CUDA builds currently has
# a bug when lowering the backward of complex-tensor regions (rFFT -> APC/
# AVRF/IMCF -> iRFFT), which surfaces as:
#
#    torch._inductor.exc.InductorError:
#        AttributeError: 'complex' object has no attribute 'get_name'
#
# (a Python ``complex(...)`` scalar leaks into Inductor's IR aliasing
# handler, which then crashes in ``add_alias``). The forward path compiles
# fine; only the backward graph for the FFT region is affected.
#
# The PyTorch-canonical fix is ``suppress_errors``: when set, Dynamo /
# AOT-autograd catch Inductor compile errors at runtime and silently fall
# back to eager mode for the offending subgraph instead of propagating the
# crash. Net effect:
#
#   * Forward compile keeps whatever speedup Inductor can deliver.
#   * Backward of the complex FFT subgraph runs in eager mode.
#   * Training never crashes mid-step on this PyTorch build.
#
# We also set ``capture_scalar_outputs`` so the .item() graph break in
# damps/core.py:_apply_imcf doesn't introduce an extra recompilation per
# epoch when the IMCF schedule advances.
try:
    import torch._dynamo                                              # type: ignore[import-untyped]
    torch._dynamo.config.suppress_errors = True
    # Best-effort: not all torch versions expose this knob, hence the
    # outer try/except already covers AttributeError.
    if hasattr(torch._dynamo.config, "capture_scalar_outputs"):
        torch._dynamo.config.capture_scalar_outputs = True
except Exception:
    # If torch._dynamo isn't importable on this build, --use_torch_compile
    # is already a no-op so there is nothing to suppress.
    pass


def _inductor_complex_backward_supported() -> bool:
    """
    Probe whether ``torch.compile``'s Inductor BACKWARD compiler can lower
    a region that operates on complex tensors (rFFT -> ops -> iRFFT).

    On affected PyTorch builds the backward compile crashes in
    ``torch/_inductor/ir.py::add_alias`` with::

        torch._inductor.exc.InductorError:
            AttributeError: 'complex' object has no attribute 'get_name'

    The crash is raised from inside ``loss.backward()``'s
    ``_backward_impl`` -> ``aot_config.bw_compiler`` chain and is **not**
    caught by ``torch._dynamo.config.suppress_errors`` -- that flag only
    covers Dynamo forward graph-capture errors, not AOT-autograd backward
    compilation errors. Once Inductor decides to lower an FFT subgraph
    that triggers this bug it will fire on every training step.

    We therefore probe end-to-end (forward + backward through a tiny
    ``rFFT -> rot -> iRFFT`` pipeline) **lazily** the first time
    ``--use_torch_compile 1`` is requested. If the probe fails we skip
    the ``torch.compile`` wrap on the DAMPS submodule entirely and run
    it in eager mode. If the probe succeeds the wrap is applied as usual.

    The probe runs on CPU and takes ~1-3 s. It is **not** invoked when
    ``--use_torch_compile 0`` (the PACER / smoke default), so smoke logs
    are no longer flooded with Inductor backward-compile tracebacks.
    """
    if not hasattr(torch, "compile"):
        return False
    # Silence Inductor's verbose "failed to eagerly compile backwards"
    # dump during the probe — a caught failure is the expected outcome
    # on affected Windows/CUDA builds and must not look like a training
    # crash in smoke stdout.
    import contextlib
    import io
    import logging as _logging

    _log = _logging.getLogger("torch._inductor")
    _prev_level = _log.level
    try:
        _log.setLevel(_logging.CRITICAL)

        def _probe(v: torch.Tensor) -> torch.Tensor:
            # Replicates damps/core.py::_apply_apc's ``exp(1j * phi)``
            # idiom so the probe predicts the real DAMPS compile path.
            z = torch.fft.rfft(v, dim=-1, norm="ortho")
            phi = v[..., : z.shape[-1]]
            rot = torch.exp(1j * phi)
            return torch.fft.irfft(z * rot, n=8, dim=-1, norm="ortho")

        with contextlib.redirect_stderr(io.StringIO()):
            compiled = torch.compile(
                _probe, mode="reduce-overhead", dynamic=True
            )
            x = torch.randn(4, 8, requires_grad=True)
            compiled(x).sum().backward()
        return True
    except Exception:
        return False
    finally:
        _log.setLevel(_prev_level)


# Lazy cache: None = not probed yet. Avoids paying the ~1-3 s Inductor
# probe (and its stderr noise) on every import when compile is off.
_INDUCTOR_COMPLEX_BACKWARD_OK: bool | None = None


def _get_inductor_complex_backward_ok() -> bool:
    """Return (and cache) the Inductor complex-backward probe result."""
    global _INDUCTOR_COMPLEX_BACKWARD_OK
    if _INDUCTOR_COMPLEX_BACKWARD_OK is None:
        _INDUCTOR_COMPLEX_BACKWARD_OK = (
            _inductor_complex_backward_supported()
        )
    return bool(_INDUCTOR_COMPLEX_BACKWARD_OK)


from utility.parser import parse_args
from utility.batch_test import (
    BATCH_SIZE,
    ITEM_NUM,
    USR_NUM,
    Ks,
    data_generator,
    test_torch,
)
from utility.logging import Logger
from damps import DualPathKNN, adj_avg_degree, adj_nnz
from model import DAMPS_MMHCL


args = parse_args()

# Branch A' / SimGCL are mutually exclusive view paths (parser help contract).
if bool(args.enable_simgcl) and bool(getattr(args, "enable_nrdmc_lite", 0)):
    raise ValueError(
        "--enable_simgcl and --enable_nrdmc_lite are mutually exclusive; "
        "enable exactly one view path (or neither for LogQ-only)."
    )


def _resolve_early_stopping_monitor(
    monitor: str,
    ks: list[int],
) -> tuple[str, int]:
    """Parse ``--early_stopping_monitor`` into ``(metric_name, ks_index)``.

    Args:
        monitor: CLI string, e.g. ``val_recall@20`` or ``ndcg``.
        ks: Evaluation cut-offs from ``--Ks`` (e.g. ``[10, 20]``).

    Returns:
        ``(metric, idx)`` where ``metric`` is one of
        ``{"recall", "ndcg", "precision"}`` and ``idx`` indexes into
        the per-metric arrays returned by ``Trainer.test``.

    Raises:
        ValueError: If the requested ``@K`` is not present in ``ks``.
    """
    mon = str(monitor or "val_recall@20").strip().lower()
    k_match = re.search(r"@(\d+)\s*$", mon)
    if k_match is not None:
        k_req = int(k_match.group(1))
        if k_req not in ks:
            raise ValueError(
                f"--early_stopping_monitor={monitor!r} requests @{k_req}, "
                f"but --Ks={ks} does not contain that cut-off."
            )
        idx = ks.index(k_req)
    else:
        idx = len(ks) - 1

    if "ndcg" in mon:
        metric = "ndcg"
    elif "precision" in mon:
        metric = "precision"
    else:
        metric = "recall"
    return metric, idx


def _effective_patience(epoch: int) -> int:
    """Return the patience used at ``epoch`` (fixed or adaptive)."""
    base = max(1, int(args.early_stopping_patience))
    if not bool(getattr(args, "adaptive_patience", 0)):
        return base
    # Mild growth: +1 patience every 50 epochs after min_epochs.
    min_ep = max(0, int(args.early_stopping_min_epochs))
    extra = max(0, (epoch - min_ep) // 50)
    return base + extra


# ===========================================================================
#  cuFFT plan cache: PERMANENTLY DISABLED (DAMPS spec Section 3.3)
# ===========================================================================
def _configure_cufft_cache(device: torch.device) -> None:
    """Disable the cuFFT plan cache to prevent latent leaks."""
    if device.type != "cuda":
        return
    try:
        idx = device.index if device.index is not None else 0
        torch.backends.cuda.cufft_plan_cache[idx].max_size = 1
    except Exception as exc:                                       # pragma: no cover
        print(f"[train] failed to configure cuFFT plan cache: {exc}")


# ===========================================================================
#  Experiment directory
# ===========================================================================
def _experiment_paths() -> tuple[str, str, str]:
    """
    Build the per-run and aggregated log directories.

    The directory name encodes the **rev44 Phase 1 dimensions** explicitly:
    ``taulearn`` (1 = learnable τ, 0 = static τ) and the τ value, plus the
    AVRF/APC/IMCF ablation switches. This keeps log files for the four
    Phase 1 configurations in distinct directories, e.g.::

        damps_..._t=0.1_taulearn=1_R=5_apc=1_avrf=1_imcf=1_..._    # rev42 anchor (a)
        damps_..._t=0.3_taulearn=0_R=5_apc=1_avrf=1_imcf=1_..._    # variant (b)
        damps_..._t=0.1_taulearn=1_R=5_apc=1_avrf=0_imcf=1_..._    # variant (c)
        damps_..._t=0.3_taulearn=0_R=5_apc=1_avrf=0_imcf=1_..._    # variant (d) -- Phase 1 default
    """
    name = (
        f"damps_uu_ii={args.User_layers}_{args.Item_layers}"
        f"_{args.user_loss_ratio}_{args.item_loss_ratio}"
        f"_topk={args.topk}_t={args.temperature}_taulearn={args.learnable_tau}"
        f"_R={args.rebuild_R}"
        f"_apc={args.damps_apc}_avrf={args.damps_avrf}_imcf={args.damps_imcf}"
        f"_regs={args.regs}_dim={args.embed_size}_seed={args.seed}_"
        f"{args.ablation_target}"
    )
    per_run = f"../{args.dataset}/{name}/"
    shared = f"../{args.dataset}/MM/"
    pathlib.Path(per_run).mkdir(parents=True, exist_ok=True)
    pathlib.Path(shared).mkdir(parents=True, exist_ok=True)
    return name, per_run, shared


# ===========================================================================
#  Trainer
# ===========================================================================
class Trainer:
    """Encapsulates DAMPS-MMHCL training, evaluation, and diagnostics."""

    def __init__(self, data_config: dict[str, Any]) -> None:
        self.n_users: int = data_config["n_users"]
        self.n_items: int = data_config["n_items"]

        self.path_name: str
        self.path: str
        self.record_path: str
        self.path_name, self.path, self.record_path = _experiment_paths()

        # ---------------- Logger ----------------
        self.logger = Logger(
            self.path,
            is_debug=args.debug,
            target=self.path_name,
            path2=self.record_path,
            ablation_target=args.ablation_target,
        )
        self.logger.logging(f"PID: {os.getpid()}")
        self.logger.logging(str(args))

        # ---------------- Hyperparams ----------------
        self.lr: float = args.lr
        self.batch_size: int = args.batch_size
        self.regs: float = args.regs

        # ---------------- Device ----------------
        self.device: torch.device = torch.device(
            f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        _configure_cufft_cache(self.device)
        self.use_amp: bool = bool(args.use_amp) and self.device.type == "cuda"

        # ---------------- Graphs (move to GPU) ----------------
        self.UI_mat: torch.Tensor = data_config["UI_mat"].to(self.device)
        self.User_mat: torch.Tensor = data_config["User_mat"].to(self.device)
        self.Item_mat: torch.Tensor = data_config["Item_mat"].to(self.device)

        # ---------------- Modality features ----------------
        if data_generator.image_feats is None or data_generator.text_feats is None:
            raise RuntimeError(
                "DAMPS-MMHCL requires both image and text modality features."
            )
        image_feats = data_generator.image_feats.to(self.device)
        text_feats = data_generator.text_feats.to(self.device)
        audio_feats = (
            data_generator.audio_feats.to(self.device)
            if data_generator.audio_feats is not None
            else None
        )

        # ---------------- Model ----------------
        ablations = {
            "apc": bool(args.damps_apc),
            "avrf": bool(args.damps_avrf),
            "imcf": bool(args.damps_imcf),
            "permutation_fft": bool(args.damps_permutation_fft),
            "soft_routing": bool(args.damps_soft_routing),
            "momentum": bool(args.damps_momentum),
        }
        self.model: DAMPS_MMHCL = DAMPS_MMHCL(
            n_users=self.n_users,
            n_items=self.n_items,
            embedding_dim=args.embed_size,
            image_feats=image_feats,
            text_feats=text_feats,
            audio_feats=audio_feats,
            ablations=ablations,
            cf_model=args.cf_model,
            ui_layers=args.UI_layers,
            user_layers=args.User_layers,
            item_layers=args.Item_layers,
            weight_size=eval(args.weight_size),
            item_loss_ratio=args.item_loss_ratio,
            user_loss_ratio=args.user_loss_ratio,
            temperature_init=args.temperature,
            learnable_tau=bool(args.learnable_tau),
            warmup_epochs=args.damps_warmup_epochs,
            damps_num_categories=args.damps_num_categories,
            data_driven_prior=bool(args.damps_data_driven_prior),
            enable_logq=bool(args.enable_logq),
            logq_scale=float(args.logq_scale),
            logq_clip=float(args.logq_clip),
            # ---- Wave 2 / M1 -- SimGCL view-invariance ----
            enable_simgcl=bool(args.enable_simgcl),
            simgcl_eps=float(args.simgcl_eps),
            simgcl_batch_size_user=int(args.simgcl_batch_size_user),
            simgcl_batch_size_item=int(args.simgcl_batch_size_item),
            # ---- Branch A (rev55 §8.1) ----
            branchA_view_every_k=int(args.branchA_view_every_k),
            branchA_bcl_batchn=bool(args.branchA_bcl_batchn),
            branchA_view_bsz=int(args.branchA_view_bsz),
            branchA_bcl_bsz=int(args.branchA_bcl_bsz),
            # ---- Branch A' / NRDMC-lite (rev55 §8.2) ----
            enable_nrdmc_lite=bool(getattr(args, "enable_nrdmc_lite", 0)),
            nrdmc_lite_layers=int(getattr(args, "nrdmc_lite_layers", 2)),
            # ---- Branch A' / P3 (rev56) Prototype-Aware View ----
            enable_ptv=bool(getattr(args, "enable_ptv", 0)),
            n_prototypes=int(getattr(args, "n_prototypes", 32)),
            lambda_ptv=float(getattr(args, "lambda_ptv", 1.0)),
        ).to(self.device)
        self.model.set_meta_categories(
            data_generator.meta_categories.to(self.device)
        )

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

        self.logger.logging(
            f"DAMPS trainable params: {self.model.damps.num_trainable_params()}"
        )

        # ---------------- Optimizer & schedulers ----------------
        # Speedup guide Section F step 2 -- fused Adam (fewer CUDA launches;
        # especially helpful on Windows where launch latency is ~2x Linux).
        # Fall back to foreach=, then to the stock eager Adam.
        _adam_kwargs: dict[str, Any] = {"lr": self.lr}
        self.optimizer: optim.Optimizer
        try:
            self.optimizer = optim.Adam(
                self.model.parameters(), fused=True, **_adam_kwargs
            )
            self.logger.logging("[speedup] Adam fused=True")
        except (RuntimeError, TypeError, ValueError) as exc:
            try:
                self.optimizer = optim.Adam(
                    self.model.parameters(), foreach=True, **_adam_kwargs
                )
                self.logger.logging(
                    f"[speedup] Adam fused unavailable ({exc}); "
                    "using foreach=True"
                )
            except (RuntimeError, TypeError, ValueError):
                self.optimizer = optim.Adam(
                    self.model.parameters(), **_adam_kwargs
                )
                self.logger.logging(
                    f"[speedup] Adam fused/foreach unavailable ({exc}); "
                    "using eager Adam"
                )
        self.lr_scheduler: optim.lr_scheduler.LambdaLR = optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda e: 0.96 ** (e / 50)
        )
        self.reduce_lr_scheduler: Optional[optim.lr_scheduler.ReduceLROnPlateau] = None
        if args.use_reduce_lr:
            self.reduce_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=args.early_stopping_mode,
                factor=args.reduce_lr_factor,
                patience=args.reduce_lr_patience,
                min_lr=args.reduce_lr_min,
            )

        # ---------------- DualPath K-NN router ----------------
        self.knn_router = DualPathKNN(
            k=args.topk,
            faiss_threshold=args.faiss_threshold,
            chunk_size=args.knn_chunk_size,
            normalize=True,
            faiss_use_gpu=bool(args.faiss_use_gpu),
            ef_search=int(args.knn_efsearch),
        )
        self.warmup_epochs: int = max(1, int(args.damps_warmup_epochs))
        self.rebuild_R: int = max(1, int(args.rebuild_R))

        # ---------------- torch.compile (Speedup Guide Section 4) ----------------
        # We compile ONLY the DAMPS submodule -- never the full forward path:
        # the full forward consumes the periodically-rebuilt ``Item_mat``
        # (sparse tensor with changing nnz), which would trigger expensive
        # recompilations. The DAMPS submodule has fixed input/output shapes
        # so it is safe to compile with dynamic=True.
        #
        # Gate on _INDUCTOR_COMPLEX_BACKWARD_OK: on PyTorch builds where
        # Inductor's backward compiler crashes on complex-FFT regions
        # (the 'complex' object has no attribute 'get_name' bug), wrapping
        # DAMPS would kill every training step at .backward(). The probe
        # detects that condition at startup and we skip the wrap to keep
        # DAMPS in eager mode (correct, just no compile speedup).
        if bool(args.use_torch_compile) and hasattr(torch, "compile"):
            if not _get_inductor_complex_backward_ok():
                self.logger.logging(
                    "[speedup] torch.compile requested but this PyTorch "
                    "build's Inductor BACKWARD compiler cannot lower DAMPS's "
                    "complex FFT region (probe failed: 'complex' object has "
                    "no attribute 'get_name'). Skipping the wrap and running "
                    "DAMPS in eager mode."
                )
            else:
                try:
                    self.model.damps = torch.compile(           # type: ignore[assignment]
                        self.model.damps,
                        mode=str(args.torch_compile_mode),
                        dynamic=bool(args.torch_compile_dynamic),
                    )
                    self.logger.logging(
                        f"[speedup] torch.compile enabled on DAMPS submodule "
                        f"(mode={args.torch_compile_mode}, "
                        f"dynamic={bool(args.torch_compile_dynamic)})"
                    )
                except Exception as exc:                        # pragma: no cover
                    self.logger.logging(
                        f"[speedup] torch.compile failed to attach: {exc}; "
                        f"continuing in eager mode."
                    )

        # ---------------- W&B (optional) ----------------
        self.wandb: Any = None
        if args.use_wandb:
            try:
                import wandb as _wandb
                self.wandb = _wandb
            except ImportError:
                self.logger.logging("[WARN] wandb not installed; disabling W&B")

    # ------------------------------------------------------------------
    #  Pattern B' Scheduled Rebuild
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def maybe_rebuild_hypergraph(self, epoch: int) -> None:
        """
        Rebuild the item-item multi-modal hypergraph from the Slim Momentum
        tables every ``rebuild_R`` epochs *after* the warm-up phase.

        Logs NNZ and average degree for transparency (spec Table 1).
        """
        if epoch < self.warmup_epochs:
            return
        if (epoch - self.warmup_epochs) % self.rebuild_R != 0:
            return
        if self.model.momentum.initialised_count() < 0.5 * self.n_items:
            self.logger.logging(
                f"[Rebuild] epoch={epoch} skipped — "
                f"only {self.model.momentum.initialised_count()}/"
                f"{self.n_items} items touched"
            )
            return

        h_img = self.model.momentum.image_table().to(self.device)
        h_txt = self.model.momentum.text_table().to(self.device)
        h_aud: Optional[torch.Tensor] = None
        if self.model.has_audio:
            h_aud = self.model.momentum.audio_table().to(self.device)

        new_adj = self.knn_router.build_graph_from_modalities(h_img, h_txt, h_aud)
        new_adj = new_adj.to(self.device)
        self.Item_mat = new_adj

        nnz = adj_nnz(new_adj)
        deg = adj_avg_degree(new_adj)
        self.logger.logging(
            f"[Rebuild] epoch={epoch}  NNZ={nnz}  avg_deg={deg:.2f}  "
            f"(target K={args.topk})"
        )
        if self.wandb is not None:
            self.wandb.log({
                "epoch": epoch,
                "rebuild/nnz": nnz,
                "rebuild/avg_deg": deg,
            })

    # ------------------------------------------------------------------
    #  Evaluation
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def test(self, users_to_test: list[int], is_val: bool) -> dict[str, Any]:
        self.model.eval()
        out = self.model(
            self.UI_mat,
            self.Item_mat,
            self.User_mat,
            item_indices=None,
            epoch=0,
            update_momentum=False,
        )
        return test_torch(out["u_ui_emb"], out["i_ui_emb"], users_to_test, is_val)

    # ------------------------------------------------------------------
    #  BPR loss
    # ------------------------------------------------------------------
    def bpr_loss(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Standard BPR pairwise ranking loss + L2 regularisation."""
        pos = (users * pos_items).sum(dim=1)
        neg = (users * neg_items).sum(dim=1)
        regularizer = (
            users.pow(2).sum() / 2 + pos_items.pow(2).sum() / 2 + neg_items.pow(2).sum() / 2
        ) / self.batch_size
        mf = -F.logsigmoid(pos - neg).mean()
        emb = self.regs * regularizer
        return mf, emb, 0.0

    # ------------------------------------------------------------------
    #  Main training loop
    # ------------------------------------------------------------------
    def train(self) -> None:
        # ---------------- W&B init ----------------
        if self.wandb is not None:
            cfg = vars(args)
            cfg["path_name"] = self.path_name
            init_kwargs: dict[str, Any] = {
                "project": args.wandb_project,
                "config": cfg,
                "reinit": True,
            }
            if args.wandb_entity:
                init_kwargs["entity"] = args.wandb_entity
            _effective_run_name = (
                getattr(args, "wandb_name", "") or args.wandb_run_name
            )
            if _effective_run_name:
                init_kwargs["name"] = _effective_run_name
            if getattr(args, "wandb_group", ""):
                init_kwargs["group"] = args.wandb_group
            if getattr(args, "wandb_tags", ""):
                init_kwargs["tags"] = [
                    t.strip()
                    for t in args.wandb_tags.split(",")
                    if t.strip()
                ]
            if getattr(args, "wandb_job_type", ""):
                init_kwargs["job_type"] = args.wandb_job_type
            self.wandb.init(**init_kwargs)

            # ----- Make ``epoch`` the canonical X-axis for every chart -----
            # WandB defaults to ``_step`` (the monotonic counter that
            # increments by 1 on every ``wandb.log()`` call). Because we
            # call ``log()`` 2-3x per epoch (train + val + maybe test +
            # rebuild), ``_step`` runs ahead of the true training epoch
            # by ~1.5x: a 250-epoch run shows up as ``_step ~ 360+`` on
            # the chart, which has caused real users to misread the
            # X-axis as the epoch counter. The two calls below redirect
            # **every** logged metric to use ``epoch`` as the step axis,
            # so what you see on the chart is what is logged in the per
            # run text file.
            try:
                self.wandb.define_metric("epoch")
                self.wandb.define_metric("*", step_metric="epoch")
            except Exception as exc:                                # pragma: no cover
                self.logger.logging(
                    f"[wandb] define_metric() unavailable in this wandb "
                    f"version ({exc}); charts will fall back to _step."
                )

        n_batch = data_generator.n_train // self.batch_size + 1
        # ------------------------------------------------------------------
        # Best-validation tracking
        # ------------------------------------------------------------------
        # We track the **two** primary validation metrics independently so that
        # the final ``BEST_Val_*`` lines emitted to the log file (and to the
        # ``wandb.summary``) exactly match the maxima of the per-epoch
        # ``val/recall@K`` and ``val/ndcg@K`` curves on WandB. We also keep the
        # **test** snapshot captured *at the epoch where each val metric peaked*
        # so the reported ``BEST_Test_Recall@K`` (resp. ``BEST_Test_NDCG@K``)
        # is unambiguously the test-set value at the recall-best (resp.
        # ndcg-best) validation epoch -- not the test result of the *last*
        # improvement, which would be wrong whenever the final improvement was
        # ndcg-only or recall-only.
        # ------------------------------------------------------------------
        best_val_recall: float = 0.0      # max of val/recall@Ks[-1]
        best_val_ndcg: float = 0.0        # max of val/ndcg@Ks[-1]
        best_val_precision: float = 0.0   # max of val/precision@Ks[-1]
        best_val_recall_epoch: int = -1   # epoch at which best_val_recall was reached
        best_val_ndcg_epoch: int = -1     # epoch at which best_val_ndcg was reached
        best_val_at_recall_peak: Optional[dict[str, Any]] = None
        best_val_at_ndcg_peak: Optional[dict[str, Any]] = None
        test_at_recall_peak: Optional[dict[str, Any]] = None
        test_at_ndcg_peak: Optional[dict[str, Any]] = None
        stopping_step: int = 0
        best_model_state: Optional[dict[str, Any]] = None
        test_ret: Union[str, dict[str, Any]] = ""  # last test snapshot (for the run-summary line)

        amp_dtype = torch.bfloat16 if self.use_amp else torch.float32

        for epoch in range(args.epoch):
            t0 = time()

            # ---------------- Pattern B' rebuild ----------------
            self.maybe_rebuild_hypergraph(epoch)

            # ---------------- Mini-batch loop ----------------
            self.model.train()
            loss = 0.0
            mf_loss = 0.0
            emb_loss = 0.0
            cl_loss = 0.0
            view_loss = 0.0
            # P3 perf: NRDMC diag scalars only every N batches (cuts CUDA syncs).
            _nrdmc_diag_every = 50
            _last_nrdmc_diag: dict[str, float] = {}

            for batch_idx in range(n_batch):
                self.optimizer.zero_grad()

                # Speedup guide Section C / F step 7 -- GPU BPR sampling
                # (flag --use_gpu_sample; default on for CUDA).
                if bool(getattr(args, "use_gpu_sample", 0)):
                    users_t, pos_t, neg_t = data_generator.sample_gpu(
                        self.device
                    )
                else:
                    users_list, pos_list, neg_list = data_generator.sample()
                    users_t = torch.tensor(
                        users_list, dtype=torch.long, device=self.device
                    )
                    pos_t = torch.tensor(
                        pos_list, dtype=torch.long, device=self.device
                    )
                    neg_t = torch.tensor(
                        neg_list, dtype=torch.long, device=self.device
                    )

                # Items covered by this batch (for the Slim Momentum write)
                batch_items = torch.cat([pos_t, neg_t]).unique()

                with torch.amp.autocast(
                    device_type=self.device.type,
                    dtype=amp_dtype,
                    enabled=self.use_amp,
                ):
                    out = self.model(
                        self.UI_mat,
                        self.Item_mat,
                        self.User_mat,
                        item_indices=batch_items,
                        epoch=epoch,
                        update_momentum=True,
                    )

                    u_g = out["u_ui_emb"][users_t]
                    pos_g = out["i_ui_emb"][pos_t]
                    neg_g = out["i_ui_emb"][neg_t]

                    bmf, bemb, _ = self.bpr_loss(u_g, pos_g, neg_g)

                    bcl_item = self.model.batched_contrastive_loss(
                        out["i_ui_emb"], out["ii_emb"], apply_logq=True,
                    ) * args.item_loss_ratio
                    bcl_user = self.model.batched_contrastive_loss(
                        out["u_ui_emb"], out["uu_emb"], apply_logq=False,
                    ) * args.user_loss_ratio
                    bcl = bcl_item + bcl_user

                    # Wave 2 / M1 -- SimGCL view-invariance term.
                    # simgcl_view_forward() is a hard no-op when
                    # args.enable_simgcl=0, preserving the Wave 1
                    # LogQ-only baseline bit-for-bit.
                    # rev55 §8.2 -- NRDMC-lite (Branch A') uses the same
                    # hook but internally routes to learnable view generators.
                    _view_on = bool(args.enable_simgcl) or bool(
                        getattr(args, "enable_nrdmc_lite", 0)
                    )
                    if _view_on:
                        _want_diag = (
                            bool(getattr(args, "enable_nrdmc_lite", 0))
                            and (batch_idx % _nrdmc_diag_every == 0)
                        )
                        l_view = (
                            self.model.simgcl_view_forward(
                                epoch=epoch, return_diag=_want_diag
                            )
                            * args.lambda_view
                        )
                        if _want_diag:
                            _cached = getattr(
                                self.model, "_last_nrdmc_diag", None
                            )
                            if isinstance(_cached, dict) and _cached:
                                _last_nrdmc_diag = dict(_cached)
                    else:
                        l_view = torch.zeros((), device=bcl_item.device)

                    batch_total = bmf + bemb + bcl + l_view

                # Backward (works with bfloat16 AMP — no GradScaler needed)
                batch_total.backward()
                if args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=args.clip_grad_norm
                    )
                self.optimizer.step()

                loss += float(batch_total)
                mf_loss += float(bmf)
                emb_loss += float(bemb)
                cl_loss += float(bcl)
                view_loss += float(l_view)

            self.lr_scheduler.step()

            # ---------------- Per-epoch EMA MAD diagnostic ----------------
            if args.damps_avrf:
                with torch.inference_mode():
                    h_img_raw = self.model.image_proj(self.model.raw_image)
                    h_txt_raw = self.model.text_proj(self.model.raw_text)
                    h_aud_raw = (
                        self.model.audio_proj(self.model.raw_audio)             # type: ignore[union-attr]
                        if self.model.has_audio
                        else None
                    )
                    self.model.damps.update_epoch_mad(
                        epoch, h_img_raw, h_txt_raw, h_aud_raw
                    )

            # ---------------- NaN guard ----------------
            if math.isnan(loss):
                self.logger.logging("ERROR: training loss is NaN — aborting")
                if self.wandb is not None:
                    self.wandb.finish(exit_code=1)
                sys.exit(1)

            elapsed = time() - t0

            # ---------------- Diagnostic logging ----------------
            with torch.inference_mode():
                diag = self.model.diagnostics()
            if epoch % self.rebuild_R == 0:
                self.logger.logging(
                    f"[diag epoch {epoch}] tau={diag['tau_clamped']:.4f} "
                    f"({diag['tau_mode']})  "
                    f"alpha_img={diag['alpha_img']:.4f} "
                    f"alpha_txt={diag['alpha_txt']:.4f}  "
                    f"tanh_sat: img={diag['tanh_sat_img']:.3f} "
                    f"txt={diag['tanh_sat_txt']:.3f}  "
                    f"baseline_asc={diag['baseline_asc']:.4f}"
                )

            # ---------------- W&B per-epoch ----------------
            if self.wandb is not None:
                _wb_payload: dict[str, Any] = {
                    "epoch": epoch,
                    "train/loss": loss,
                    "train/mf_loss": mf_loss,
                    "train/emb_loss": emb_loss,
                    "train/cl_loss": cl_loss,
                    "train/view_loss": view_loss,
                    "loss/simgcl_view": view_loss,
                    "train/lr": self.optimizer.param_groups[0]["lr"],
                    "diag/tau": diag["tau_clamped"],
                    # 1.0 if learnable, 0.0 if static -- makes the rev44 Phase 1
                    # vs rev42 anchor distinction trivially filterable in WandB.
                    "diag/tau_learnable": float(diag["tau_mode"] == "learnable"),
                    "diag/alpha_img": diag["alpha_img"],
                    "diag/alpha_txt": diag["alpha_txt"],
                    "diag/tanh_sat_img": diag["tanh_sat_img"],
                    "diag/tanh_sat_txt": diag["tanh_sat_txt"],
                    "diag/baseline_asc": diag["baseline_asc"],
                }
                for _k, _v in _last_nrdmc_diag.items():
                    _wb_payload[f"nrdmc/{_k}"] = _v
                self.wandb.log(_wb_payload)

            # ---------------- Skip evaluation on non-eval epochs ----------------
            # --eval_every N (bottleneck guide §5.2): evaluate every N epochs,
            # but always evaluate in the last --eval_last_epochs. When
            # --eval_every is 0, preserve the legacy --verbose cadence.
            eval_every = int(getattr(args, "eval_every", 0) or 0)
            eval_last = int(getattr(args, "eval_last_epochs", 20) or 0)
            if eval_every > 0:
                in_tail = (
                    eval_last > 0 and (epoch + 1) > (args.epoch - eval_last)
                )
                do_eval = in_tail or ((epoch + 1) % eval_every == 0)
            else:
                do_eval = ((epoch + 1) % args.verbose == 0)
            if not do_eval:
                self.logger.logging(
                    f"Epoch {epoch} [{elapsed:.1f}s]: "
                    f"loss={loss:.5f}  mf={mf_loss:.5f}  emb={emb_loss:.5f}  "
                    f"cl={cl_loss:.5f}  view={view_loss:.5f}"
                )
                continue

            # ---------------- Validation ----------------
            t1 = time()
            users_to_val = list(data_generator.val_set.keys())
            users_to_test = list(data_generator.test_set.keys())
            val = self.test(users_to_val, is_val=True)
            t2 = time()

            self.logger.logging(
                f"Epoch {epoch} [{elapsed:.1f}s + {t2 - t1:.1f}s]: "
                f"loss={loss:.5f}  recall@{Ks[0]}={val['recall'][0]:.5f}  "
                f"recall@{Ks[-1]}={val['recall'][-1]:.5f}  "
                f"ndcg@{Ks[-1]}={val['ndcg'][-1]:.5f}"
            )

            # Keep the pre-update precision best so --early_stopping_monitor
            # val_precision@K can detect a true improvement this cycle.
            prev_best_val_precision = best_val_precision
            best_val_precision = max(
                best_val_precision, float(val["precision"][-1])
            )
            if self.wandb is not None:
                # NOTE: ``val/ndcg@{Ks[0]}`` (i.e. NDCG@10) is logged here in
                # addition to the @20 cut-off so the WandB ``val`` section
                # surfaces both NDCG charts side by side.
                # ``best_precision`` is the running maximum of
                # val/precision@Ks[-1] up to the current epoch, mirroring the
                # ``best_recall`` / ``best_ndcg`` series visible in the
                # Workspace Charts.
                self.wandb.log({
                    "epoch": epoch,
                    f"val/recall@{Ks[0]}": val["recall"][0],
                    f"val/recall@{Ks[-1]}": val["recall"][-1],
                    f"val/ndcg@{Ks[0]}": val["ndcg"][0],
                    f"val/ndcg@{Ks[-1]}": val["ndcg"][-1],
                    f"val/precision@{Ks[-1]}": val["precision"][-1],
                    f"val/hit@{Ks[-1]}": val["hit_ratio"][-1],
                    "best_precision": best_val_precision,
                })

            if self.reduce_lr_scheduler is not None:
                self.reduce_lr_scheduler.step(val["recall"][-1])

            # ---------------- Early stopping ----------------
            # Peak bookkeeping still tracks recall AND ndcg independently
            # (PACER BEST_Test_* semantics). The *patience* counter, however,
            # is gated solely by ``--early_stopping_monitor`` (smoke / PACER
            # tercile protocol passes ``val_recall@20``).
            recall_improved = (
                val["recall"][1] > best_val_recall + args.early_stopping_min_delta
            )
            ndcg_improved = (
                val["ndcg"][1] > best_val_ndcg + args.early_stopping_min_delta
            )
            mon_metric, mon_idx = _resolve_early_stopping_monitor(
                str(getattr(args, "early_stopping_monitor", "val_recall@20")),
                list(Ks),
            )
            mon_best = {
                "recall": best_val_recall,
                "ndcg": best_val_ndcg,
                "precision": prev_best_val_precision,
            }[mon_metric]
            monitor_improved = (
                float(val[mon_metric][mon_idx])
                > mon_best + args.early_stopping_min_delta
            )
            # Still run a test evaluation whenever recall OR ndcg improves
            # so peak snapshots stay correct; patience only listens to
            # ``monitor_improved``.
            improved = recall_improved or ndcg_improved
            if improved:
                test_ret = self.test(users_to_test, is_val=False)
                self.logger.logging(
                    f"Test_Recall@{Ks[1]}: {test_ret['recall'][1]:.8f}  "
                    f"Test_Precision@{Ks[1]}: {test_ret['precision'][1]:.8f}  "
                    f"Test_NDCG@{Ks[1]}: {test_ret['ndcg'][1]:.8f}"
                )

                # Snapshot val + test ONLY when val_recall@K hits a new high.
                # This guarantees that the final BEST_Test_Recall@K is the
                # test result at the recall-best validation epoch -- even if a
                # later epoch only improves NDCG and overwrites ``test_ret``.
                if val["recall"][1] > best_val_recall:
                    best_val_recall = float(val["recall"][1])
                    best_val_recall_epoch = int(epoch)
                    best_val_at_recall_peak = {
                        "recall": np.array(val["recall"], copy=True),
                        "ndcg": np.array(val["ndcg"], copy=True),
                        "precision": np.array(val["precision"], copy=True),
                        "hit_ratio": np.array(val["hit_ratio"], copy=True),
                    }
                    test_at_recall_peak = {
                        "recall": np.array(test_ret["recall"], copy=True),
                        "ndcg": np.array(test_ret["ndcg"], copy=True),
                        "precision": np.array(test_ret["precision"], copy=True),
                        "hit_ratio": np.array(test_ret["hit_ratio"], copy=True),
                    }
                # Same idea for the ndcg-best snapshot.
                if val["ndcg"][1] > best_val_ndcg:
                    best_val_ndcg = float(val["ndcg"][1])
                    best_val_ndcg_epoch = int(epoch)
                    best_val_at_ndcg_peak = {
                        "recall": np.array(val["recall"], copy=True),
                        "ndcg": np.array(val["ndcg"], copy=True),
                        "precision": np.array(val["precision"], copy=True),
                        "hit_ratio": np.array(val["hit_ratio"], copy=True),
                    }
                    test_at_ndcg_peak = {
                        "recall": np.array(test_ret["recall"], copy=True),
                        "ndcg": np.array(test_ret["ndcg"], copy=True),
                        "precision": np.array(test_ret["precision"], copy=True),
                        "hit_ratio": np.array(test_ret["hit_ratio"], copy=True),
                    }

                if self.wandb is not None:
                    self.wandb.log({
                        "epoch": epoch,
                        f"test/recall@{Ks[-1]}": test_ret["recall"][1],
                        f"test/ndcg@{Ks[-1]}": test_ret["ndcg"][1],
                        f"test/precision@{Ks[-1]}": test_ret["precision"][1],
                        "best_recall": best_val_recall,
                        "best_ndcg": best_val_ndcg,
                    })

            # Patience is driven ONLY by --early_stopping_monitor.
            if monitor_improved:
                stopping_step = 0
                if args.early_stopping_restore_best:
                    best_model_state = copy.deepcopy(self.model.state_dict())
            elif epoch + 1 >= args.early_stopping_min_epochs:
                stopping_step += 1
                eff_patience = _effective_patience(epoch)
                self.logger.logging(
                    f"##### Early stopping step: {stopping_step}/"
                    f"{eff_patience} (monitor="
                    f"{getattr(args, 'early_stopping_monitor', 'val_recall@20')}"
                    f") #####"
                )
                if stopping_step >= eff_patience:
                    self.logger.logging("##### Early stop triggered #####")
                    if args.early_stopping_restore_best and best_model_state is not None:
                        self.model.load_state_dict(best_model_state)
                    fname = f"Model.epoch={epoch}.pth"
                    torch.save(self.model.state_dict(), os.path.join(self.path, fname))
                    break

        # ---------------- Final summary ----------------
        # The two ``BEST_Val_*`` lines exactly match the maxima of the WandB
        # ``val/recall@K`` and ``val/ndcg@K`` curves; the ``BEST_Test_*`` lines
        # are the test-set values captured at those very same epochs (i.e. the
        # standard convention: pick the model by validation, report on test).
        if best_val_at_recall_peak is not None:
            self.logger.logging(
                f"BEST_Val_Recall@{Ks[0]}: {float(best_val_at_recall_peak['recall'][0]):.8f}"
            )
            self.logger.logging(
                f"BEST_Val_Recall@{Ks[1]}: {best_val_recall:.8f}"
            )
            self.logger.logging(
                f"BEST_Val_Recall_Peak_Epoch: {best_val_recall_epoch}"
            )
        if best_val_at_ndcg_peak is not None:
            self.logger.logging(
                f"BEST_Val_NDCG@{Ks[0]}: {float(best_val_at_ndcg_peak['ndcg'][0]):.8f}"
            )
            self.logger.logging(
                f"BEST_Val_NDCG@{Ks[1]}: {best_val_ndcg:.8f}"
            )
            self.logger.logging(
                f"BEST_Val_NDCG_Peak_Epoch: {best_val_ndcg_epoch}"
            )

        # ``BEST_Test_Recall@K`` is the test-set recall at the epoch where
        # val_recall@K peaked; ``BEST_Test_NDCG@K`` is the test-set NDCG at
        # the epoch where val_ndcg@K peaked. The two may come from different
        # epochs, which is the methodologically clean choice when reporting
        # multiple test metrics derived from a single validation-selection
        # criterion per metric.
        if test_at_recall_peak is not None:
            self.logger.logging(
                f"BEST_Test_Recall@{Ks[1]}: {float(test_at_recall_peak['recall'][1]):.8f}"
            )
            self.logger.logging(
                f"BEST_Test_Precision@{Ks[1]}: {float(test_at_recall_peak['precision'][1]):.8f}"
            )
        if test_at_ndcg_peak is not None:
            self.logger.logging(
                f"BEST_Test_NDCG@{Ks[1]}: {float(test_at_ndcg_peak['ndcg'][1]):.8f}"
            )

        if self.wandb is not None:
            if best_val_at_recall_peak is not None:
                self.wandb.summary[f"best_val_recall@{Ks[0]}"] = float(
                    best_val_at_recall_peak["recall"][0]
                )
                self.wandb.summary[f"best_val_recall@{Ks[-1]}"] = best_val_recall
                # Record the epoch at which val/recall@K peaked so a reviewer
                # can correlate the W&B chart maximum with the run-summary
                # number without doing arithmetic on the _step axis.
                self.wandb.summary["best_val_recall_peak_epoch"] = (
                    best_val_recall_epoch
                )
            if best_val_at_ndcg_peak is not None:
                self.wandb.summary[f"best_val_ndcg@{Ks[0]}"] = float(
                    best_val_at_ndcg_peak["ndcg"][0]
                )
                self.wandb.summary[f"best_val_ndcg@{Ks[-1]}"] = best_val_ndcg
                self.wandb.summary["best_val_ndcg_peak_epoch"] = (
                    best_val_ndcg_epoch
                )
            self.wandb.summary[f"best_val_precision@{Ks[-1]}"] = (
                best_val_precision
            )
            if test_at_recall_peak is not None:
                self.wandb.summary[f"best_test_recall@{Ks[-1]}"] = float(
                    test_at_recall_peak["recall"][1]
                )
                self.wandb.summary[f"best_test_precision@{Ks[-1]}"] = float(
                    test_at_recall_peak["precision"][1]
                )
            if test_at_ndcg_peak is not None:
                self.wandb.summary[f"best_test_ndcg@{Ks[-1]}"] = float(
                    test_at_ndcg_peak["ndcg"][1]
                )

        # Keep the legacy ``test_ret`` print so existing log-parsers in the
        # multi-seed notebook still find a serialisable test-result dict.
        self.logger.logging(str(test_ret))
        self.logger.logging_sum(f"{self.path_name}:{str(test_ret)}")
        if self.wandb is not None:
            self.wandb.finish()


# ===========================================================================
#  Reproducibility
# ===========================================================================
def set_seed(seed: int) -> None:
    """Set every relevant random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ===========================================================================
#  Entry point
# ===========================================================================
def main() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    set_seed(args.seed)

    config: dict[str, Any] = {
        "n_users": data_generator.n_users,
        "n_items": data_generator.n_items,
    }
    config["UI_mat"] = data_generator.get_UI_mat()
    config["User_mat"] = data_generator.get_U2U_mat()
    config["Item_mat"] = data_generator.build_static_hypergraph()

    trainer = Trainer(data_config=config)
    trainer.train()


if __name__ == "__main__":
    main()
