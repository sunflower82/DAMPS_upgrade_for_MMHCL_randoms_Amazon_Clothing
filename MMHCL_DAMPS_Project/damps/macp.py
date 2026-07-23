"""damps/macp.py -- Multi-Aspect Content Preprocessing fusion.

Ships Priority 6.0 (P6.0). PACER-NRDMC treats the text modality as
frozen features loaded from ``text_feat.npy``. This module reads the
pre-computed MACP streams (see ``scripts/preprocess_macp.py``) and
returns a fused text tensor that can drop-in replace ``text_feats``
inside ``utility.load_data.Data``.

Design decisions
----------------
* **Text-only.** Image features are left raw -- P5.0 confirmed the
  model adaptively drives alpha_img to negative values on Clothing;
  whitening image would fight that correction.

* **Same input dim.** Both PCA->ICA and ZCA streams are produced at
  the input dimension by the offline script, so fusion is a straight
  additive residual with no projection.

* **Std matching.** Whitening scales the streams differently (ZCA
  shrinks; ICA amplifies). Before mixing we rescale each auxiliary
  stream to match the raw stream's row-wise L2 mean, so the fusion
  weights `alpha_p, alpha_z` act as *relative* injection ratios
  rather than absolute magnitudes.

* **Flag-off equals baseline.** ``fuse_text(..., mode='raw')`` returns
  the input tensor unmodified so any bit-exact regression test against
  the P5 baseline still passes.

Modes
-----
raw
    No change. Used to disable MACP without removing the wiring.
replace_pca
    Replace ``text_feats`` with the PCA->ICA stream.
replace_zca
    Replace ``text_feats`` with the ZCA stream.
residual
    ``t_raw + alpha_p * scale_p * t_pca_ica + alpha_z * scale_z * t_zca``
    where ``scale_*`` matches the auxiliary stream's mean row L2 to the
    raw stream. This is the P6.0 default (safe: raw is preserved as
    the dominant signal, auxiliaries inject discriminability).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


# Filenames must stay in lockstep with scripts/preprocess_macp.py.
PCA_ICA_FILE = "text_feat_pca_ica.npy"
ZCA_FILE     = "text_feat_zca.npy"

VALID_MODES = ("raw", "replace_pca", "replace_zca", "residual")


@dataclass(frozen=True)
class MacpConfig:
    """Immutable knobs consumed by :func:`fuse_text`.

    ``mode == 'raw'`` short-circuits every branch below and returns the
    input tensor untouched, so the whole module is a no-op when the
    parser flag is off.
    """

    mode: str = "raw"
    alpha_p: float = 0.10        # PCA->ICA residual weight
    alpha_z: float = 0.10        # ZCA residual weight

    def __post_init__(self) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError(
                f"MacpConfig.mode='{self.mode}' invalid; "
                f"expected one of {VALID_MODES}."
            )


def _load_stream(path: str) -> Optional[torch.Tensor]:
    """Load a MACP .npy stream or return ``None`` if absent."""
    if not os.path.isfile(path):
        return None
    arr = np.load(path)
    return torch.from_numpy(arr).float()


def _rescale_to(match: torch.Tensor, aux: torch.Tensor,
                *, eps: float = 1e-8) -> torch.Tensor:
    """Rescale *aux* so its row-wise L2 mean matches *match*.

    Multiplicative scalar; preserves geometry, only fixes magnitude.
    """
    match_norm = match.norm(dim=1).mean().clamp_min(eps)
    aux_norm   = aux.norm(dim=1).mean().clamp_min(eps)
    scale = float(match_norm / aux_norm)
    return aux * scale, scale


def load_macp_streams(
    dataset_dir: str,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Return (pca_ica_tensor, zca_tensor) or (None, None) if missing."""
    pca_ica = _load_stream(os.path.join(dataset_dir, PCA_ICA_FILE))
    zca     = _load_stream(os.path.join(dataset_dir, ZCA_FILE))
    return pca_ica, zca


def fuse_text(
    text_raw: torch.Tensor,
    *,
    dataset_dir: str,
    cfg: MacpConfig,
    verbose: bool = True,
) -> tuple[torch.Tensor, dict]:
    """Return (fused_text, diagnostics).

    Parameters
    ----------
    text_raw : (N, D) float tensor
        The tensor originally produced by ``_load_modality('text')``.
    dataset_dir : str
        Directory that holds ``text_feat.npy`` and (optionally) the
        MACP streams. Same directory the loader passed to ``np.load``.
    cfg : MacpConfig
        Which mode + weights to use. ``mode='raw'`` bypasses I/O.
    verbose : bool
        If True, prints a one-line summary. Wired to
        ``args.macp_verbose`` in the loader.
    """
    diag: dict = {"mode": cfg.mode}
    if cfg.mode == "raw" or text_raw is None:
        diag["status"] = "bypass"
        if verbose:
            print("[MACP] mode=raw -- text_feats unchanged.", flush=True)
        return text_raw, diag

    pca_ica, zca = load_macp_streams(dataset_dir)
    diag["has_pca_ica"] = pca_ica is not None
    diag["has_zca"]     = zca is not None

    if cfg.mode == "replace_pca":
        if pca_ica is None:
            raise FileNotFoundError(
                f"macp_mode=replace_pca requires {PCA_ICA_FILE} in "
                f"{dataset_dir}. Run scripts/preprocess_macp.py first."
            )
        if pca_ica.shape != text_raw.shape:
            raise ValueError(
                f"PCA->ICA stream shape {tuple(pca_ica.shape)} does not "
                f"match text_raw {tuple(text_raw.shape)}. Regenerate "
                f"with the correct text_feat.npy."
            )
        fused = pca_ica
        diag["status"] = "replaced_with_pca_ica"

    elif cfg.mode == "replace_zca":
        if zca is None:
            raise FileNotFoundError(
                f"macp_mode=replace_zca requires {ZCA_FILE} in "
                f"{dataset_dir}. Run scripts/preprocess_macp.py first."
            )
        if zca.shape != text_raw.shape:
            raise ValueError(
                f"ZCA stream shape {tuple(zca.shape)} does not match "
                f"text_raw {tuple(text_raw.shape)}."
            )
        fused = zca
        diag["status"] = "replaced_with_zca"

    elif cfg.mode == "residual":
        if pca_ica is None and zca is None:
            raise FileNotFoundError(
                f"macp_mode=residual requires at least one MACP stream "
                f"in {dataset_dir}. Run scripts/preprocess_macp.py "
                f"first."
            )
        fused = text_raw.clone()
        if pca_ica is not None and cfg.alpha_p != 0.0:
            if pca_ica.shape != text_raw.shape:
                raise ValueError(
                    f"PCA->ICA stream shape {tuple(pca_ica.shape)} "
                    f"does not match text_raw {tuple(text_raw.shape)}."
                )
            pca_ica_scaled, scale_p = _rescale_to(text_raw, pca_ica)
            fused = fused + float(cfg.alpha_p) * pca_ica_scaled
            diag["scale_p"] = scale_p
            diag["alpha_p"] = cfg.alpha_p
        if zca is not None and cfg.alpha_z != 0.0:
            if zca.shape != text_raw.shape:
                raise ValueError(
                    f"ZCA stream shape {tuple(zca.shape)} does not "
                    f"match text_raw {tuple(text_raw.shape)}."
                )
            zca_scaled, scale_z = _rescale_to(text_raw, zca)
            fused = fused + float(cfg.alpha_z) * zca_scaled
            diag["scale_z"] = scale_z
            diag["alpha_z"] = cfg.alpha_z
        diag["status"] = "residual"

    else:  # pragma: no cover -- guarded by MacpConfig.__post_init__
        raise ValueError(f"Unhandled MACP mode: {cfg.mode}")

    diag["mean_l2_raw"]   = float(text_raw.norm(dim=1).mean())
    diag["mean_l2_fused"] = float(fused.norm(dim=1).mean())
    if verbose:
        print(
            f"[MACP] mode={cfg.mode}  alpha_p={cfg.alpha_p}  "
            f"alpha_z={cfg.alpha_z}  "
            f"|t_raw|={diag['mean_l2_raw']:.3f}  "
            f"|t_fused|={diag['mean_l2_fused']:.3f}",
            flush=True,
        )
    return fused, diag
