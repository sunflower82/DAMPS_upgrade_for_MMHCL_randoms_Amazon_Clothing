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
        if bool(args.use_torch_compile) and hasattr(torch, "compile"):
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
        best_recall: float = 0.0
        best_ndcg: float = 0.0
        stopping_step: int = 0
        best_model_state: Optional[dict[str, Any]] = None
        test_ret: Union[str, dict[str, Any]] = ""

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
                self.wandb.log({
                    "epoch": epoch,
                    f"val/recall@{Ks[0]}": val["recall"][0],
                    f"val/recall@{Ks[-1]}": val["recall"][-1],
                    f"val/ndcg@{Ks[-1]}": val["ndcg"][-1],
                    f"val/precision@{Ks[-1]}": val["precision"][-1],
                    f"val/hit@{Ks[-1]}": val["hit_ratio"][-1],
                })

            if self.reduce_lr_scheduler is not None:
                self.reduce_lr_scheduler.step(val["recall"][-1])

            # ---------------- Early stopping ----------------
            improved = (
                val["recall"][1] > best_recall + args.early_stopping_min_delta
                or val["ndcg"][1] > best_ndcg + args.early_stopping_min_delta
            )
            if improved:
                if val["recall"][1] > best_recall:
                    best_recall = float(val["recall"][1])
                if val["ndcg"][1] > best_ndcg:
                    best_ndcg = float(val["ndcg"][1])
                test_ret = self.test(users_to_test, is_val=False)
                self.logger.logging(
                    f"Test_Recall@{Ks[1]}: {test_ret['recall'][1]:.8f}  "
                    f"Test_Precision@{Ks[1]}: {test_ret['precision'][1]:.8f}  "
                    f"Test_NDCG@{Ks[1]}: {test_ret['ndcg'][1]:.8f}"
                )
                if self.wandb is not None:
                    self.wandb.log({
                        "epoch": epoch,
                        f"test/recall@{Ks[-1]}": test_ret["recall"][1],
                        f"test/ndcg@{Ks[-1]}": test_ret["ndcg"][1],
                        f"test/precision@{Ks[-1]}": test_ret["precision"][1],
                        "best_recall": best_recall,
                        "best_ndcg": best_ndcg,
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
        if isinstance(test_ret, dict):
            self.logger.logging(
                f"BEST_Test_Recall@{Ks[1]}: {test_ret['recall'][1]:.8f}"
            )
            self.logger.logging(
                f"BEST_Test_Precision@{Ks[1]}: {test_ret['precision'][1]:.8f}"
            )
            self.logger.logging(
                f"BEST_Test_NDCG@{Ks[1]}: {test_ret['ndcg'][1]:.8f}"
            )
            if self.wandb is not None:
                self.wandb.summary[f"best_test_recall@{Ks[-1]}"] = test_ret["recall"][1]
                self.wandb.summary[f"best_test_precision@{Ks[-1]}"] = test_ret["precision"][1]
                self.wandb.summary[f"best_test_ndcg@{Ks[-1]}"] = test_ret["ndcg"][1]

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
