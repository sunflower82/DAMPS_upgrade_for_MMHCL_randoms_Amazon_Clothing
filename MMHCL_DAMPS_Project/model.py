"""
model.py — DAMPS-MMHCL Neural Network
========================================

Implements the full **DAMPS-MMHCL** model that integrates the spectral
calibration block (``damps/core.py``) into the original MMHCL backbone via
**Soft Residual-Routing** (eq. 3 of the spec).

Forward pipeline (per epoch)
----------------------------
::

    raw_image, raw_text                 # (N, image_dim), (N, text_dim)
        │
        ▼  per-modality MLP projection
    h_img_raw, h_txt_raw                # (N, d=64)
        │
        ▼  DAMPS spectral calibration (FFT → APC → AVRF → IMCF → IFFT)
    h_img_cal, h_txt_cal                # (N, d)
        │
        ▼  Soft Residual-Routing (eq. 3)
    h_img_input  = h_img_raw + α_img · LayerNorm(h_img_cal)
    h_txt_input  = h_txt_raw + α_txt · LayerNorm(h_txt_cal)
        │
        ▼  hypergraph convolution (uses I2I_mat from Pattern B' rebuild)
    ii_emb                              # (N, d)
        │
        ▼  fuse with CF view (LightGCN/NGCF/MF on UI graph) + UU view
    final_user_emb, final_item_emb      # (n_users, d), (n_items, d)

Compared with the original MMHCL the only architectural deltas are:

1.  An **additional DAMPS pre-pass** before hypergraph propagation.
2.  A **Soft Residual-Routing** branch that mixes raw + calibrated features
    to evade the double over-smoothing of GCN stacks.
3.  A **learnable temperature** ``τ`` (Section 3.1) replacing the static
    InfoNCE temperature in ``batched_contrastive_loss``.
4.  A **Slim Momentum Encoder** that lives outside the autograd graph and
    feeds the Pattern B' rebuild (see ``train.py``).

Everything else — UI bipartite graph, U2U co-interaction graph, BPR loss,
contrastive loss — is identical to the original MMHCL.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from damps import DAMPS, SlimMomentumEncoder, compute_avrf_logit


# ===========================================================================
#  Modality MLP projection head
# ===========================================================================
class ModalityProjection(nn.Module):
    """
    Single-layer projection ``h_m = W_m x_m + b_m``  (eq. 4 of the spec).

    Mirrors the original MMHCL projection layer.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ===========================================================================
#  DAMPS_MMHCL — main model
# ===========================================================================
class DAMPS_MMHCL(nn.Module):
    """
    Full DAMPS-MMHCL recommender.

    Args:
        n_users               : number of users.
        n_items               : number of items.
        embedding_dim         : embedding dim ``d`` (default 64).
        image_feats           : (n_items, image_dim) tensor or None.
        text_feats            : (n_items, text_dim) tensor or None.
        audio_feats           : optional (n_items, audio_dim) tensor (Tiktok).
        ablations             : per-component on/off dict
                                (apc, avrf, imcf, soft_routing, momentum,
                                permutation_fft).
        cf_model              : 'LightGCN' (default), 'NGCF', or 'MF'.
        ui_layers, user_layers, item_layers : GNN depths.
        weight_size           : NGCF per-layer sizes (only used when
                                ``cf_model == 'NGCF'``).
        item_loss_ratio       : weight for item-side contrastive loss.
        user_loss_ratio       : weight for user-side contrastive loss.
        temperature_init      : initial value for the learnable τ.
        warmup_epochs         : warm-up window for the EMA schedules.
        damps_num_categories  : number of static metadata clusters.
        data_driven_prior     : if True, derive AVRF priors from raw features.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        image_feats: Optional[torch.Tensor] = None,
        text_feats: Optional[torch.Tensor] = None,
        audio_feats: Optional[torch.Tensor] = None,
        ablations: Optional[dict[str, bool]] = None,
        cf_model: str = "LightGCN",
        ui_layers: int = 2,
        user_layers: int = 3,
        item_layers: int = 2,
        weight_size: Optional[list[int]] = None,
        item_loss_ratio: float = 0.07,
        user_loss_ratio: float = 0.03,
        temperature_init: float = 0.1,
        warmup_epochs: int = 10,
        damps_num_categories: int = 10,
        data_driven_prior: bool = True,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # Config
        # ------------------------------------------------------------------
        self.n_users: int = n_users
        self.n_items: int = n_items
        self.embedding_dim: int = embedding_dim
        self.cf_model: str = cf_model
        self.ui_layers: int = ui_layers
        self.user_layers: int = user_layers
        self.item_layers: int = item_layers
        self.weight_size: list[int] = weight_size or [embedding_dim] * (ui_layers + 1)
        self.item_loss_ratio: float = item_loss_ratio
        self.user_loss_ratio: float = user_loss_ratio

        self.has_audio: bool = audio_feats is not None

        # Default ablation switches (mirrors the spec's full lock-in defaults)
        defaults: dict[str, bool] = {
            "apc": True,
            "avrf": True,
            "imcf": True,
            "permutation_fft": False,
            "soft_routing": True,
            "momentum": True,
        }
        if ablations:
            defaults.update({k: bool(v) for k, v in ablations.items()})
        self.ablations: dict[str, bool] = defaults

        # ------------------------------------------------------------------
        # 1. Learnable embedding tables (CF + hypergraph branches)
        # ------------------------------------------------------------------
        self.user_ui_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_ui_embedding = nn.Embedding(n_items, embedding_dim)
        self.uu_embedding = nn.Embedding(n_users, embedding_dim)
        self.ii_embedding = nn.Embedding(n_items, embedding_dim)
        for emb in [
            self.user_ui_embedding,
            self.item_ui_embedding,
            self.uu_embedding,
            self.ii_embedding,
        ]:
            nn.init.xavier_uniform_(emb.weight)

        # ------------------------------------------------------------------
        # 2. Optional NGCF transformation layers
        # ------------------------------------------------------------------
        self.GC_Linear_list = nn.ModuleList()
        self.Bi_Linear_list = nn.ModuleList()
        self.dropout_list = nn.ModuleList()
        if cf_model == "NGCF":
            for i in range(ui_layers):
                self.GC_Linear_list.append(
                    nn.Linear(self.weight_size[i], self.weight_size[i + 1])
                )
                self.Bi_Linear_list.append(
                    nn.Linear(self.weight_size[i], self.weight_size[i + 1])
                )
                self.dropout_list.append(nn.Dropout(0.1))

        # ------------------------------------------------------------------
        # 3. Modality projection MLPs (raw → d=64)
        # ------------------------------------------------------------------
        self.image_dim: Optional[int] = (
            image_feats.shape[-1] if image_feats is not None else None
        )
        self.text_dim: Optional[int] = (
            text_feats.shape[-1] if text_feats is not None else None
        )
        self.audio_dim: Optional[int] = (
            audio_feats.shape[-1] if audio_feats is not None else None
        )

        if self.image_dim is None or self.text_dim is None:
            raise RuntimeError(
                "DAMPS-MMHCL requires both image and text modalities. "
                "Please make sure image_feat.npy / text_feat.npy are present."
            )

        self.image_proj = ModalityProjection(self.image_dim, embedding_dim)
        self.text_proj = ModalityProjection(self.text_dim, embedding_dim)
        self.audio_proj: Optional[ModalityProjection] = (
            ModalityProjection(self.audio_dim, embedding_dim)
            if self.has_audio else None
        )

        # ------------------------------------------------------------------
        # 4. Cache raw feature buffers (read-only, for DAMPS forward pass)
        # ------------------------------------------------------------------
        self.register_buffer("raw_image", image_feats.float(), persistent=False)
        self.register_buffer("raw_text", text_feats.float(), persistent=False)
        if self.has_audio:
            self.register_buffer("raw_audio", audio_feats.float(), persistent=False)

        # ------------------------------------------------------------------
        # 5. Static metadata categories for APC (registered later via setter)
        # ------------------------------------------------------------------
        self.register_buffer(
            "meta_categories",
            torch.zeros(n_items, dtype=torch.long),
            persistent=False,
        )

        # ------------------------------------------------------------------
        # 6. DAMPS spectral calibrator
        # ------------------------------------------------------------------
        prior_image: Optional[torch.Tensor] = None
        prior_text: Optional[torch.Tensor] = None
        prior_audio: Optional[torch.Tensor] = None
        if data_driven_prior:
            # Project raw features to d=64 before computing the prior so the
            # spectral statistics match the actual DAMPS input distribution.
            #
            # Device alignment: ``image_feats`` / ``text_feats`` may already be
            # on a CUDA device because the trainer pre-moves them BEFORE
            # constructing the model (see train.py: feats.to(self.device) ->
            # DAMPS_MMHCL(...).to(self.device)). The projection MLPs above
            # were just instantiated with ``nn.Linear`` and live on CPU, so
            # we must move them to the inputs' device before calling them.
            # The trainer's outer ``.to(self.device)`` after __init__ is then
            # a no-op for these submodules.
            target_device = image_feats.device
            self.image_proj.to(target_device)
            self.text_proj.to(target_device)
            if self.has_audio and self.audio_proj is not None:
                self.audio_proj.to(target_device)
            with torch.no_grad():
                proj_img = self.image_proj(image_feats.float())
                proj_txt = self.text_proj(text_feats.float())
                prior_image = compute_avrf_logit(proj_img)
                prior_text = compute_avrf_logit(proj_txt)
                if self.has_audio and audio_feats is not None:
                    proj_aud = self.audio_proj(audio_feats.float())   # type: ignore[union-attr]
                    prior_audio = compute_avrf_logit(proj_aud)

        damps_ablations = {
            "apc": self.ablations["apc"],
            "avrf": self.ablations["avrf"],
            "imcf": self.ablations["imcf"],
            "permutation_fft": self.ablations["permutation_fft"],
        }
        self.damps = DAMPS(
            d=embedding_dim,
            num_categories=damps_num_categories,
            warmup_epochs=warmup_epochs,
            ablations=damps_ablations,
            prior_image=prior_image,
            prior_text=prior_text,
            prior_audio=prior_audio,
        )

        # ------------------------------------------------------------------
        # 7. Soft Residual-Routing (eq. 3 of the spec)
        # ------------------------------------------------------------------
        self.alpha_img = nn.Parameter(torch.tensor(0.1))
        self.alpha_txt = nn.Parameter(torch.tensor(0.1))
        self.alpha_aud = nn.Parameter(torch.tensor(0.1))
        self.ln_img = nn.LayerNorm(embedding_dim)
        self.ln_txt = nn.LayerNorm(embedding_dim)
        self.ln_aud = nn.LayerNorm(embedding_dim) if self.has_audio else None

        # ------------------------------------------------------------------
        # 8. Slim Momentum Encoder (eats h_cal_*, drives Pattern B' rebuild)
        # ------------------------------------------------------------------
        self.momentum = SlimMomentumEncoder(
            num_items=n_items,
            dim=embedding_dim,
            warmup_epochs=warmup_epochs,
            use_ema=self.ablations["momentum"],
            num_modalities=3 if self.has_audio else 2,
        )

        # ------------------------------------------------------------------
        # 9. Learnable InfoNCE temperature (Revision 9 spec, Section 3.1).
        #    Default initialisation = 0.1; can be overridden via the
        #    ``temperature_init`` constructor argument or the
        #    ``--temperature`` CLI flag in train.py. Clamped to >= 0.01 in
        #    ``batched_contrastive_loss`` to prevent division blow-ups.
        # ------------------------------------------------------------------
        self.tau = nn.Parameter(torch.tensor(float(temperature_init)))

    # ------------------------------------------------------------------
    #  Setters & accessors
    # ------------------------------------------------------------------
    def set_meta_categories(self, cats: torch.Tensor) -> None:
        """Register the static metadata category vector (n_items,) for APC."""
        if cats.shape[0] != self.n_items:
            raise ValueError(
                f"meta_categories length {cats.shape[0]} != n_items {self.n_items}"
            )
        self.meta_categories.copy_(cats.long())

    @property
    def has_item_branch(self) -> bool:
        return self.item_loss_ratio != 0.0

    @property
    def has_user_branch(self) -> bool:
        return self.user_loss_ratio != 0.0

    # ------------------------------------------------------------------
    #  DAMPS pre-pass: raw modality → calibrated d=64 representation
    # ------------------------------------------------------------------
    def _damps_calibration(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor],
               torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Project raw modality features to d=64, then run DAMPS calibration.

        Returns:
            (h_img_raw, h_txt_raw, h_aud_raw, h_img_cal, h_txt_cal, h_aud_cal)
        """
        h_img_raw = self.image_proj(self.raw_image)
        h_txt_raw = self.text_proj(self.raw_text)
        h_aud_raw: Optional[torch.Tensor] = None
        if self.has_audio:
            h_aud_raw = self.audio_proj(self.raw_audio)            # type: ignore[union-attr]

        cats = self.meta_categories if self.ablations["apc"] else None
        h_img_cal, h_txt_cal, h_aud_cal = self.damps(
            h_img_raw, h_txt_raw, item_categories=cats, h_aud=h_aud_raw
        )
        return h_img_raw, h_txt_raw, h_aud_raw, h_img_cal, h_txt_cal, h_aud_cal

    # ------------------------------------------------------------------
    #  Soft Residual-Routing (eq. 3 of the spec)
    # ------------------------------------------------------------------
    def _soft_route(
        self,
        h_raw: torch.Tensor,
        h_cal: torch.Tensor,
        ln: nn.LayerNorm,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        if not self.ablations["soft_routing"]:
            return h_cal
        return h_raw + alpha * ln(h_cal)

    # ------------------------------------------------------------------
    #  Forward pass
    # ------------------------------------------------------------------
    def forward(
        self,
        UI_mat: torch.Tensor,
        I2I_mat: torch.Tensor,
        U2U_mat: torch.Tensor,
        item_indices: Optional[torch.Tensor] = None,
        epoch: int = 0,
        update_momentum: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass that produces both the CF view and the hypergraph view.

        Args:
            UI_mat          : (n_users + n_items)² sparse — bipartite graph.
            I2I_mat         : (n_items, n_items) sparse — multi-modal hypergraph.
            U2U_mat         : (n_users, n_users) sparse — co-interaction graph.
            item_indices    : optional (B,) — items covered by this batch
                              (used by the Slim Momentum updater to limit
                              the buffer write to relevant rows).
            epoch           : current epoch (drives the EMA schedule).
            update_momentum : if False, the Slim Momentum buffers are not
                              touched (e.g. during evaluation).

        Returns:
            Dict with keys:
                u_ui_emb, i_ui_emb     : final user/item embeddings.
                ii_emb, uu_emb         : hypergraph-only embeddings.
                h_img_cal, h_txt_cal   : calibrated modality features.
                h_aud_cal              : audio (Tiktok), else None.
                damps_node_input       : the actual node features fed into HGCN.
        """
        # =====================================================================
        # (A) DAMPS calibration pass (cheap: ~5 ms for N=23k on RTX 5090)
        # =====================================================================
        # Drive the IMCF / MAD adaptive EMA schedule via the explicit epoch
        # index from the trainer (Revision 9 audit WARN 3). Without this,
        # ``_apply_imcf`` would tick its counter once per forward pass and
        # exhaust ``warmup_epochs`` inside the first real epoch.
        if self.training:
            self.damps.set_epoch(epoch)
        (
            h_img_raw,
            h_txt_raw,
            h_aud_raw,
            h_img_cal,
            h_txt_cal,
            h_aud_cal,
        ) = self._damps_calibration()

        # =====================================================================
        # (B) Slim Momentum update — only on the items covered by this batch
        # =====================================================================
        if self.training and update_momentum and item_indices is not None:
            with torch.no_grad():
                self.momentum.update(
                    item_indices=item_indices.to(h_img_cal.device),
                    h_cal_img=h_img_cal[item_indices],
                    h_cal_txt=h_txt_cal[item_indices],
                    h_cal_aud=(
                        h_aud_cal[item_indices] if h_aud_cal is not None else None
                    ),
                    epoch=epoch,
                )

        # =====================================================================
        # (C) Soft Residual-Routing → produce HGCN node features
        # =====================================================================
        node_img = self._soft_route(h_img_raw, h_img_cal, self.ln_img, self.alpha_img)
        node_txt = self._soft_route(h_txt_raw, h_txt_cal, self.ln_txt, self.alpha_txt)
        node_aud: Optional[torch.Tensor] = None
        if h_aud_cal is not None and self.ln_aud is not None:
            node_aud = self._soft_route(
                h_aud_raw,                                          # type: ignore[arg-type]
                h_aud_cal,
                self.ln_aud,
                self.alpha_aud,
            )

        # Average the per-modality node features into a single (n_items, d)
        # signal that drives the hypergraph propagation. (We *also* keep the
        # original ``ii_embedding`` table — see eq. (4)/(11) of the spec.)
        if node_aud is not None:
            damps_node_signal = (node_img + node_txt + node_aud) / 3.0
        else:
            damps_node_signal = (node_img + node_txt) / 2.0

        # =====================================================================
        # (D) Hypergraph branches (item-side and user-side)
        # =====================================================================
        # The DAMPS-fed node signal is *added* to the learnable ii embedding
        # so the hypergraph propagation has both a learnable bias and a
        # spectral-cleansed multi-modal signal (Section 1.3 of the spec).
        ii_emb = self.ii_embedding.weight + damps_node_signal
        uu_emb = self.uu_embedding.weight

        if self.has_item_branch:
            for _ in range(self.item_layers):
                ii_emb = torch.sparse.mm(I2I_mat, ii_emb)

        if self.has_user_branch:
            for _ in range(self.user_layers):
                uu_emb = torch.sparse.mm(U2U_mat, uu_emb)

        # =====================================================================
        # (E) CF branch — LightGCN / NGCF / MF
        # =====================================================================
        if self.cf_model == "LightGCN":
            ego = torch.cat(
                [self.user_ui_embedding.weight, self.item_ui_embedding.weight], dim=0
            )
            stack = [ego]
            for _ in range(self.ui_layers):
                ego = torch.sparse.mm(UI_mat, ego)
                stack.append(ego)
            mean = torch.stack(stack, dim=1).mean(dim=1)
            u_ui_emb, i_ui_emb = torch.split(mean, [self.n_users, self.n_items], dim=0)
        elif self.cf_model == "NGCF":
            ego = torch.cat(
                [self.user_ui_embedding.weight, self.item_ui_embedding.weight], dim=0
            )
            stack = [ego]
            for i in range(self.ui_layers):
                side = torch.sparse.mm(UI_mat, ego)
                sum_e = F.leaky_relu(self.GC_Linear_list[i](side))
                bi_e = F.leaky_relu(
                    self.Bi_Linear_list[i](torch.mul(ego, side))
                )
                ego = self.dropout_list[i](sum_e + bi_e)
                stack.append(F.normalize(ego, p=2, dim=1))
            mean = torch.stack(stack, dim=1).mean(dim=1)
            u_ui_emb, i_ui_emb = torch.split(mean, [self.n_users, self.n_items], dim=0)
        else:                                                       # 'MF'
            u_ui_emb = self.user_ui_embedding.weight
            i_ui_emb = self.item_ui_embedding.weight

        # =====================================================================
        # (F) Fuse hypergraph view into CF view (mirrors original MMHCL)
        # =====================================================================
        if self.has_item_branch:
            i_ui_emb = i_ui_emb + F.normalize(ii_emb, p=2, dim=1)
        if self.has_user_branch:
            u_ui_emb = u_ui_emb + F.normalize(uu_emb, p=2, dim=1)

        return {
            "u_ui_emb": u_ui_emb,
            "i_ui_emb": i_ui_emb,
            "ii_emb": ii_emb,
            "uu_emb": uu_emb,
            "h_img_cal": h_img_cal,
            "h_txt_cal": h_txt_cal,
            "h_aud_cal": h_aud_cal,
            "damps_node_signal": damps_node_signal,
        }

    # ------------------------------------------------------------------
    #  Contrastive Loss (Learnable τ — Section 3.1 of the spec)
    # ------------------------------------------------------------------
    def batched_contrastive_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        batch_size: int = 4096,
    ) -> torch.Tensor:
        """
        InfoNCE contrastive loss with a **learnable** temperature τ.

        τ is clamped at 0.01 from below to prevent division by zero / numerical
        explosion. This matches the "Learnable InfoNCE Temperature" subsection
        of the DAMPS spec.
        """
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        tau = torch.clamp(self.tau, min=0.01)

        f: Callable[[torch.Tensor], torch.Tensor] = lambda x: torch.exp(x / tau)
        indices = torch.arange(0, num_nodes, device=device)
        losses: list[torch.Tensor] = []

        for i in range(num_batches):
            mask = indices[i * batch_size : (i + 1) * batch_size]
            refl_sim = f(self._sim(z1[mask], z1))
            between_sim = f(self._sim(z1[mask], z2))

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

    @staticmethod
    def _sim(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Cosine similarity between two row-normalised tensors."""
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    # ------------------------------------------------------------------
    #  Diagnostics
    # ------------------------------------------------------------------
    @torch.no_grad()
    def diagnostics(self) -> dict[str, Any]:
        """
        Return a summary of the model's current state (used by train.py for
        per-epoch logs):

            damps_params         : trainable params inside DAMPS only.
            tau                  : current learnable temperature.
            tau_clamped          : effective temperature after clamping.
            alpha_img / alpha_txt: Soft-Routing scalars.
            tanh_saturation      : per-modality saturation rates.
            momentum_init        : how many items have been touched.
        """
        sat = self.damps.tanh_saturation_rates()
        return {
            "damps_params": self.damps.num_trainable_params(),
            "tau": float(self.tau.item()),
            "tau_clamped": float(torch.clamp(self.tau, min=0.01).item()),
            "alpha_img": float(self.alpha_img.item()),
            "alpha_txt": float(self.alpha_txt.item()),
            "alpha_aud": float(self.alpha_aud.item()) if self.has_audio else None,
            "tanh_sat_img": sat["img"],
            "tanh_sat_txt": sat["txt"],
            "tanh_sat_aud": sat["aud"] if self.has_audio else None,
            "lambda_coh": float(self.damps.lambda_coh.item()),
            "baseline_asc": float(self.damps.baseline_asc.item()),
            "imcf_epoch": int(self.damps._current_epoch.item()),
            "imcf_forward_passes": int(self.damps._imcf_update_count.item()),
            "momentum_init": self.momentum.initialised_count(),
        }
