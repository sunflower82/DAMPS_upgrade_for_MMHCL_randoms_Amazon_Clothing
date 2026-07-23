"""scripts/preprocess_macp.py -- Offline MACP text-feature whitening.

Ships Priority 6.0 (P6.0) of the PACER-NRDMC upgrade roadmap: reproduces
TAMER (MM'25) "Multi-Aspect Content Preprocessing" (MACP) for the
**text** modality only. Image features are deliberately left raw --
Amazon Clothing image embeddings are noisy (P5.0 confirmed the model
adaptively drives alpha_img -> -0.52), and whitening them would fight
that correction.

Two streams are produced from ``text_feat.npy``:

* ``text_feat_pca_ica.npy`` -- PCA (dim-preserving rotation) followed by
  FastICA. Emphasises statistically independent latent factors.
* ``text_feat_zca.npy``     -- Zero-phase Component Analysis whitening
  (Cov = U diag(lam) U^T -> W = U diag(lam^{-1/2}) U^T). Decorrelates
  while remaining as close to the raw embedding as possible in L2.

Both outputs share the input dimension so the downstream loader can
either replace ``text_feats`` in-place or perform a residual injection
``t_raw + alpha_p * t_pca_ica + alpha_z * t_zca`` without any dim
gymnastics.

Determinism
-----------
FastICA (sklearn) is seeded via ``--seed``; the ZCA path is pure
NumPy so is deterministic by construction. Reproducibility is
verified by rerunning with the same seed and diffing MD5.

Usage (from MMHCL_DAMPS_Project/)::

    # Standard: reads ../data/Clothing/text_feat.npy and writes two
    # sibling files next to it.
    python scripts/preprocess_macp.py --dataset Clothing

    # Custom paths + seed:
    python scripts/preprocess_macp.py \
        --input   ../data/Clothing/text_feat.npy \
        --out_dir ../data/Clothing/ \
        --seed 42 --ica_max_iter 500 --pca_var_floor 0.999
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
#  ZCA whitening (pure NumPy; deterministic)
# --------------------------------------------------------------------------- #
def zca_whiten(x: np.ndarray, *, eps: float = 1e-5) -> tuple[np.ndarray, dict]:
    """Return ZCA-whitened copy of *x* and diagnostics.

    Parameters
    ----------
    x : (N, D) float64 ndarray
        Row-wise samples. NOT modified in place.
    eps : float
        Regularisation added to the eigenvalues to guard against
        near-zero variance directions (typical of pre-trained embeds).

    Returns
    -------
    y : (N, D) float64 ndarray
        Whitened matrix. Has zero mean and (approximately) identity
        covariance in the same basis as *x*.
    stats : dict
        `mean_l2_before/after`, `cov_offdiag_max_before/after`,
        `eigenvalue_min/max`. Handy to log in the driver.
    """
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    mu = x.mean(axis=0, keepdims=True)                          # (1, D)
    xc = x - mu
    # Sample covariance with (N-1) normalisation matches sklearn convention.
    cov = (xc.T @ xc) / max(1, n - 1)                           # (D, D)
    # Symmetric eigendecomposition (numerical rank <= D-1 is common).
    eigvals, eigvecs = np.linalg.eigh(cov)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(eigvals, 0.0) + eps)
    w = (eigvecs * inv_sqrt) @ eigvecs.T                        # (D, D) ZCA
    y = xc @ w

    # Diagnostics -- useful to catch degenerate embeddings early.
    def _offdiag_max(m: np.ndarray) -> float:
        m = m.copy()
        np.fill_diagonal(m, 0.0)
        return float(np.abs(m).max()) if m.size else 0.0

    cov_y = (y.T @ y) / max(1, n - 1)
    stats = {
        "eigenvalue_min": float(eigvals.min()),
        "eigenvalue_max": float(eigvals.max()),
        "cov_offdiag_max_before": _offdiag_max(cov),
        "cov_offdiag_max_after": _offdiag_max(cov_y),
        "mean_l2_before": float(np.linalg.norm(xc, axis=1).mean()),
        "mean_l2_after":  float(np.linalg.norm(y,  axis=1).mean()),
    }
    return y, stats


# --------------------------------------------------------------------------- #
#  PCA (dim-preserving rotation) followed by FastICA
# --------------------------------------------------------------------------- #
def pca_ica(
    x: np.ndarray,
    *,
    seed: int,
    ica_max_iter: int = 500,
    ica_tol: float = 1e-4,
    pca_var_floor: float | None = None,
) -> tuple[np.ndarray, dict]:
    """PCA then FastICA in the input dimension.

    We deliberately keep k = D (or the effective rank if
    ``pca_var_floor`` is set) so the output is a *rotation* of the
    input embedding and can be additively fused with the raw stream
    without projection mismatch.

    Parameters
    ----------
    x : (N, D) float64 ndarray
    seed : int
        Passed to FastICA.random_state.
    ica_max_iter, ica_tol : FastICA solver knobs.
    pca_var_floor : optional cutoff on cumulative explained variance
        (e.g. 0.999). If given, PCA is truncated to the smallest k
        that reaches the floor, and the ICA output is zero-padded
        back to D so downstream shapes are stable.

    Returns
    -------
    y : (N, D) float64 ndarray
    stats : dict
    """
    from sklearn.decomposition import PCA, FastICA          # local import

    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape

    # PCA's rank is bounded by min(N-1, D). On Amazon Clothing we have
    # ~24k items and D=384 so this collapses to k=D, but small text
    # fixtures (N < D) exercise the guard below.
    k_cap = max(1, min(d, n - 1))
    if pca_var_floor is not None:
        pca = PCA(n_components=k_cap, svd_solver="full",
                  random_state=seed).fit(x)
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        k = int(np.searchsorted(cumsum, pca_var_floor) + 1)
        k = max(1, min(k_cap, k))
        pca = PCA(n_components=k, svd_solver="full",
                  random_state=seed, whiten=False).fit(x)
    else:
        k = k_cap
        pca = PCA(n_components=k, svd_solver="full",
                  random_state=seed, whiten=False).fit(x)

    xp = pca.transform(x)                                    # (N, k)
    ica = FastICA(
        n_components=k,
        whiten="unit-variance",
        random_state=seed,
        max_iter=ica_max_iter,
        tol=ica_tol,
    )
    yp = ica.fit_transform(xp)                               # (N, k)

    if k < d:
        y = np.zeros((n, d), dtype=np.float64)
        y[:, :k] = yp
    else:
        y = yp

    stats = {
        "pca_k": int(k),
        "pca_d_input": int(d),
        "explained_variance_ratio_sum": float(
            pca.explained_variance_ratio_.sum()
        ),
        "ica_n_iter": int(ica.n_iter_),
        "ica_max_iter": int(ica_max_iter),
        "mean_l2_before": float(np.linalg.norm(x - x.mean(0), axis=1).mean()),
        "mean_l2_after":  float(np.linalg.norm(y, axis=1).mean()),
    }
    return y, stats


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _parse_cli(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Offline MACP whitening for PACER-NRDMC (text only)."
    )
    p.add_argument(
        "--dataset", type=str, default="Clothing",
        help="Dataset name under ``--data_path``. Ignored when both "
             "``--input`` and ``--out_dir`` are supplied.",
    )
    p.add_argument(
        "--data_path", type=str, default="../data",
        help="Root data directory (relative to MMHCL_DAMPS_Project/).",
    )
    p.add_argument(
        "--input", type=str, default=None,
        help="Explicit path to text_feat.npy. Overrides "
             "``--data_path/--dataset``.",
    )
    p.add_argument(
        "--out_dir", type=str, default=None,
        help="Directory for the MACP outputs. Defaults to the parent "
             "of --input (or ``--data_path/--dataset``).",
    )
    p.add_argument(
        "--stream", type=str, default="both",
        choices=("pca_ica", "zca", "both"),
        help="Which whitening streams to produce.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for FastICA (ZCA is deterministic).",
    )
    p.add_argument("--ica_max_iter", type=int, default=500)
    p.add_argument("--ica_tol", type=float, default=1e-4)
    p.add_argument(
        "--pca_var_floor", type=float, default=None,
        help="Optional cumulative-variance cutoff (e.g. 0.999). If "
             "set, PCA is truncated to that many components and ICA "
             "is zero-padded back to input dim.",
    )
    p.add_argument(
        "--dtype_out", type=str, default="float32",
        choices=("float32", "float64"),
        help="Output dtype. float32 halves disk usage and matches the "
             "PACER loader default.",
    )
    p.add_argument(
        "--force", type=int, default=0,
        help="1 = overwrite existing MACP outputs.",
    )
    p.add_argument(
        "--log_json", type=str, default=None,
        help="Optional path to dump diagnostics + MD5 sums as JSON.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_cli(argv)

    if args.input is None:
        in_path = Path(args.data_path) / args.dataset / "text_feat.npy"
    else:
        in_path = Path(args.input)
    if not in_path.is_file():
        raise FileNotFoundError(f"Missing text_feat.npy: {in_path}")

    out_dir = Path(args.out_dir) if args.out_dir else in_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pca_ica = out_dir / "text_feat_pca_ica.npy"
    out_zca     = out_dir / "text_feat_zca.npy"

    print(f"[MACP] input:   {in_path}", flush=True)
    print(f"[MACP] out_dir: {out_dir}", flush=True)
    print(f"[MACP] stream:  {args.stream}  seed={args.seed}  "
          f"dtype_out={args.dtype_out}", flush=True)

    x = np.load(in_path)
    if x.ndim != 2:
        raise ValueError(
            f"text_feat.npy has shape {x.shape}; expected (N_items, D)."
        )
    print(f"[MACP] loaded: shape={x.shape}  dtype={x.dtype}", flush=True)

    diagnostics: dict[str, dict] = {}
    dtype_out = np.float32 if args.dtype_out == "float32" else np.float64

    if args.stream in ("both", "pca_ica"):
        if out_pca_ica.is_file() and not args.force:
            print(f"[MACP] SKIP pca_ica: {out_pca_ica} exists "
                  f"(use --force 1 to overwrite).", flush=True)
        else:
            t0 = time.time()
            y_pca, s_pca = pca_ica(
                x, seed=args.seed,
                ica_max_iter=args.ica_max_iter,
                ica_tol=args.ica_tol,
                pca_var_floor=args.pca_var_floor,
            )
            np.save(out_pca_ica, y_pca.astype(dtype_out))
            wall = time.time() - t0
            s_pca["wall_seconds"] = wall
            s_pca["md5"] = _md5(out_pca_ica)
            diagnostics["pca_ica"] = s_pca
            print(f"[MACP] wrote {out_pca_ica.name}  wall={wall:.1f}s  "
                  f"md5={s_pca['md5'][:12]}", flush=True)

    if args.stream in ("both", "zca"):
        if out_zca.is_file() and not args.force:
            print(f"[MACP] SKIP zca: {out_zca} exists "
                  f"(use --force 1 to overwrite).", flush=True)
        else:
            t0 = time.time()
            y_zca, s_zca = zca_whiten(x)
            np.save(out_zca, y_zca.astype(dtype_out))
            wall = time.time() - t0
            s_zca["wall_seconds"] = wall
            s_zca["md5"] = _md5(out_zca)
            diagnostics["zca"] = s_zca
            print(f"[MACP] wrote {out_zca.name}  wall={wall:.1f}s  "
                  f"md5={s_zca['md5'][:12]}", flush=True)

    if args.log_json:
        log_path = Path(args.log_json)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "input": str(in_path),
            "out_dir": str(out_dir),
            "input_shape": list(x.shape),
            "input_dtype": str(x.dtype),
            "seed": args.seed,
            "streams": diagnostics,
        }
        with log_path.open("w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"[MACP] log JSON: {log_path}", flush=True)

    for name, d in diagnostics.items():
        print(f"[MACP] {name} stats:", flush=True)
        for k, v in d.items():
            print(f"       {k:>28} = {v}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
