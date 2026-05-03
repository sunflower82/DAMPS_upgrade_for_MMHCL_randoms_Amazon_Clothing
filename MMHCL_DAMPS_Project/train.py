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
import sys
from time import time
from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
import torch.optim as optim

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
    ``rFFT -> *2.0 -> iRFFT`` pipeline) at startup. If the probe fails we
    skip the ``torch.compile`` wrap on the DAMPS submodule entirely and
    run it in eager mode. If the probe succeeds (e.g. a future PyTorch
    where the bug is fixed, or a non-Windows build) the wrap is applied
    as usual and the speedup is preserved.

    The probe runs on CPU, takes ~1-3 s, and is invoked once per
    ``train.py`` subprocess.
    """
    if not hasattr(torch, "compile"):
        return False
    try:
        # The bug fires specifically when a Python ``complex`` literal (e.g.
        # ``1j``) leaks into Inductor's ``add_alias`` via something like
        # ``torch.exp(1j * theta)`` -- which is exactly what DAMPS's APC
        # phase rotation does in ``damps/core.py::_apply_apc``::
        #     rot = torch.exp(-1j * (theta/2 + psi))   # complex tensor
        #     z   = z * rot                             # complex * complex
        # The probe replicates this idiom so it accurately predicts whether
        # the real DAMPS forward+backward will compile on this build.
        def _probe(v: torch.Tensor) -> torch.Tensor:
            z = torch.fft.rfft(v, dim=-1, norm="ortho")
            phi = v[..., : z.shape[-1]]
            rot = torch.exp(1j * phi)
            return torch.fft.irfft(z * rot, n=8, dim=-1, norm="ortho")

        compiled = torch.compile(_probe, mode="reduce-overhead", dynamic=True)
        x = torch.randn(4, 8, requires_grad=True)
        compiled(x).sum().backward()
        return True
    except Exception:
        return False


_INDUCTOR_COMPLEX_BACKWARD_OK: bool = _inductor_complex_backward_supported()


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
    """Build the per-run and aggregated log directories."""
    name = (
        f"damps_uu_ii={args.User_layers}_{args.Item_layers}"
        f"_{args.user_loss_ratio}_{args.item_loss_ratio}"
        f"_topk={args.topk}_t={args.temperature}_R={args.rebuild_R}"
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
            warmup_epochs=args.damps_warmup_epochs,
            damps_num_categories=args.damps_num_categories,
            data_driven_prior=bool(args.damps_data_driven_prior),
        ).to(self.device)
        self.model.set_meta_categories(
            data_generator.meta_categories.to(self.device)
        )

        self.logger.logging(
            f"DAMPS trainable params: {self.model.damps.num_trainable_params()}"
        )

        # ---------------- Optimizer & schedulers ----------------
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
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
            if not _INDUCTOR_COMPLEX_BACKWARD_OK:
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
    @torch.no_grad()
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
    @torch.no_grad()
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
            if args.wandb_run_name:
                init_kwargs["name"] = args.wandb_run_name
            self.wandb.init(**init_kwargs)

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
        best_val_recall: float = 0.0   # max of val/recall@Ks[-1]
        best_val_ndcg: float = 0.0     # max of val/ndcg@Ks[-1]
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

            for _ in range(n_batch):
                self.optimizer.zero_grad()

                users_list, pos_list, neg_list = data_generator.sample()
                users_t = torch.tensor(users_list, dtype=torch.long, device=self.device)
                pos_t = torch.tensor(pos_list, dtype=torch.long, device=self.device)
                neg_t = torch.tensor(neg_list, dtype=torch.long, device=self.device)

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
                        out["i_ui_emb"], out["ii_emb"]
                    ) * args.item_loss_ratio
                    bcl_user = self.model.batched_contrastive_loss(
                        out["u_ui_emb"], out["uu_emb"]
                    ) * args.user_loss_ratio
                    bcl = bcl_item + bcl_user
                    batch_total = bmf + bemb + bcl

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

            self.lr_scheduler.step()

            # ---------------- Per-epoch EMA MAD diagnostic ----------------
            if args.damps_avrf:
                with torch.no_grad():
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
            diag = self.model.diagnostics()
            if epoch % self.rebuild_R == 0:
                self.logger.logging(
                    f"[diag epoch {epoch}] tau={diag['tau_clamped']:.4f}  "
                    f"alpha_img={diag['alpha_img']:.4f} "
                    f"alpha_txt={diag['alpha_txt']:.4f}  "
                    f"tanh_sat: img={diag['tanh_sat_img']:.3f} "
                    f"txt={diag['tanh_sat_txt']:.3f}  "
                    f"baseline_asc={diag['baseline_asc']:.4f}"
                )

            # ---------------- W&B per-epoch ----------------
            if self.wandb is not None:
                self.wandb.log({
                    "epoch": epoch,
                    "train/loss": loss,
                    "train/mf_loss": mf_loss,
                    "train/emb_loss": emb_loss,
                    "train/cl_loss": cl_loss,
                    "train/lr": self.optimizer.param_groups[0]["lr"],
                    "diag/tau": diag["tau_clamped"],
                    "diag/alpha_img": diag["alpha_img"],
                    "diag/alpha_txt": diag["alpha_txt"],
                    "diag/tanh_sat_img": diag["tanh_sat_img"],
                    "diag/tanh_sat_txt": diag["tanh_sat_txt"],
                    "diag/baseline_asc": diag["baseline_asc"],
                })

            # ---------------- Skip evaluation on non-eval epochs ----------------
            if (epoch + 1) % args.verbose != 0:
                self.logger.logging(
                    f"Epoch {epoch} [{elapsed:.1f}s]: "
                    f"loss={loss:.5f}  mf={mf_loss:.5f}  emb={emb_loss:.5f}  "
                    f"cl={cl_loss:.5f}"
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

            if self.wandb is not None:
                # NOTE: ``val/ndcg@{Ks[0]}`` (i.e. NDCG@10) is logged here in
                # addition to the @20 cut-off so the WandB ``val`` section
                # surfaces both NDCG charts side by side.
                self.wandb.log({
                    "epoch": epoch,
                    f"val/recall@{Ks[0]}": val["recall"][0],
                    f"val/recall@{Ks[-1]}": val["recall"][-1],
                    f"val/ndcg@{Ks[0]}": val["ndcg"][0],
                    f"val/ndcg@{Ks[-1]}": val["ndcg"][-1],
                    f"val/precision@{Ks[-1]}": val["precision"][-1],
                    f"val/hit@{Ks[-1]}": val["hit_ratio"][-1],
                })

            if self.reduce_lr_scheduler is not None:
                self.reduce_lr_scheduler.step(val["recall"][-1])

            # ---------------- Early stopping ----------------
            # Improvement criterion: either val_recall@K OR val_ndcg@K must
            # strictly exceed its running best by at least ``min_delta``. The
            # patience counter resets on either kind of improvement (matches
            # the original MMHCL/BM3 behaviour).
            recall_improved = (
                val["recall"][1] > best_val_recall + args.early_stopping_min_delta
            )
            ndcg_improved = (
                val["ndcg"][1] > best_val_ndcg + args.early_stopping_min_delta
            )
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
                stopping_step = 0
                if args.early_stopping_restore_best:
                    best_model_state = copy.deepcopy(self.model.state_dict())
            elif epoch + 1 >= args.early_stopping_min_epochs:
                stopping_step += 1
                self.logger.logging(f"##### Early stopping step: {stopping_step} #####")
                if stopping_step >= args.early_stopping_patience:
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
        if best_val_at_ndcg_peak is not None:
            self.logger.logging(
                f"BEST_Val_NDCG@{Ks[0]}: {float(best_val_at_ndcg_peak['ndcg'][0]):.8f}"
            )
            self.logger.logging(
                f"BEST_Val_NDCG@{Ks[1]}: {best_val_ndcg:.8f}"
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
            if best_val_at_ndcg_peak is not None:
                self.wandb.summary[f"best_val_ndcg@{Ks[0]}"] = float(
                    best_val_at_ndcg_peak["ndcg"][0]
                )
                self.wandb.summary[f"best_val_ndcg@{Ks[-1]}"] = best_val_ndcg
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
