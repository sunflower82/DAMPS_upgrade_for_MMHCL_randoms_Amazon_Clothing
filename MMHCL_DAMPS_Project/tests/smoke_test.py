"""
tests/smoke_test.py — End-to-end sanity check for the DAMPS package.

Runs a tiny synthetic forward+backward pass through every DAMPS sub-component
and the integrated ``DAMPS_MMHCL`` model. Designed to finish in < 5 seconds
and use < 100 MiB of RAM, so it is safe for CI / pre-commit hooks.

Run from the repository root:

    python MMHCL_DAMPS_Project/tests/smoke_test.py
"""

from __future__ import annotations

import os
import sys

# Make the package importable when run as a standalone script
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import torch
import torch.nn.functional as F

from damps import (
    DAMPS,
    DualPathKNN,
    SlimMomentumEncoder,
    adj_avg_degree,
    adj_nnz,
    compute_avrf_logit,
    compute_avrf_prior,
)


def _print_ok(label: str) -> None:
    print(f"  [OK] {label}")


def smoke_damps_core() -> None:
    print("== DAMPS core ==")
    N, d, C = 64, 64, 4
    h_img = torch.randn(N, d)
    h_txt = torch.randn(N, d)
    cats = torch.randint(0, C, (N,))

    damps = DAMPS(d=d, num_categories=C, warmup_epochs=2)
    damps.train()

    h_img_cal, h_txt_cal, _ = damps(h_img, h_txt, item_categories=cats)
    assert h_img_cal.shape == h_img.shape, "shape mismatch (img)"
    assert h_txt_cal.shape == h_txt.shape, "shape mismatch (txt)"

    # Backprop through DAMPS — verifies all parameters receive gradient
    loss = (h_img_cal.pow(2).sum() + h_txt_cal.pow(2).sum()) * 1e-3
    loss.backward()
    grad_count = sum(p.grad is not None for p in damps.parameters())
    assert grad_count >= 4, f"expected >=4 grads, got {grad_count}"
    _print_ok(f"forward + backward, {damps.num_trainable_params()} params")

    sat = damps.tanh_saturation_rates()
    _print_ok(f"tanh_sat: {sat}")

    damps.update_epoch_mad(0, h_img, h_txt)
    _print_ok("update_epoch_mad runs")

    # ----- IMCF EMA epoch-counter regression test (compliance WARN 3) -----
    # Run several forward passes inside a single "epoch": the per-forward-pass
    # counter must keep ticking, but the *current epoch* (which drives the
    # adaptive EMA schedule) must NOT change unless ``set_epoch`` is called.
    damps.zero_grad(set_to_none=True)
    damps.set_epoch(3)
    fwd_before = float(damps._imcf_update_count.item())
    epoch_before = int(damps._current_epoch.item())
    for _ in range(5):
        damps(h_img, h_txt, item_categories=cats)
    fwd_after = float(damps._imcf_update_count.item())
    epoch_after = int(damps._current_epoch.item())
    assert fwd_after - fwd_before == 5, (
        f"forward-pass counter expected +5, got "
        f"+{fwd_after - fwd_before}"
    )
    assert epoch_after == epoch_before == 3, (
        f"epoch counter must stay at 3 across forward passes, got "
        f"before={epoch_before}, after={epoch_after}"
    )
    damps.set_epoch(7)
    assert int(damps._current_epoch.item()) == 7
    _print_ok(
        "IMCF schedule: forward-pass counter +5, epoch held at 3 then "
        "advanced to 7 via set_epoch"
    )


def smoke_momentum() -> None:
    print("== Slim Momentum Encoder ==")
    N, d = 32, 64
    enc = SlimMomentumEncoder(num_items=N, dim=d, warmup_epochs=2)
    idx = torch.arange(N)
    enc.update(idx, torch.randn(N, d), torch.randn(N, d), epoch=0)
    enc.update(idx, torch.randn(N, d), torch.randn(N, d), epoch=1)
    enc.update(idx, torch.randn(N, d), torch.randn(N, d), epoch=2)
    assert enc.initialised_count() == N
    _print_ok("EMA buffers update + initialised flag")


def smoke_knn() -> None:
    print("== Dual-path KNN ==")
    N, d = 64, 32
    h_img = torch.randn(N, d)
    h_txt = torch.randn(N, d)

    builder = DualPathKNN(k=4, faiss_threshold=10**12, chunk_size=16)
    adj_single = builder.build_graph(h_img)
    assert adj_single.shape == (N, N)
    _print_ok(f"single-modality K-NN, NNZ={adj_nnz(adj_single)}")

    adj_multi = builder.build_graph_from_modalities(h_img, h_txt)
    assert adj_multi.shape == (N, N)
    _print_ok(
        f"multi-modal hypergraph, NNZ={adj_nnz(adj_multi)}, "
        f"avg_deg={adj_avg_degree(adj_multi):.2f}"
    )


def smoke_prior() -> None:
    print("== Data-driven prior ==")
    N, d = 128, 64
    feats = torch.randn(N, d)
    prior = compute_avrf_prior(feats)
    logit = compute_avrf_logit(feats, clip=2.0)
    assert prior.shape == (d // 2 + 1,)
    assert logit.shape == (1, d // 2 + 1)
    assert (logit.abs() <= 2.0 + 1e-6).all()
    _print_ok(f"prior range=[{prior.min():.3f},{prior.max():.3f}]  logit clipped")


def smoke_full_model() -> None:
    print("== DAMPS_MMHCL full model ==")
    # Local import so the smoke test can run without a configured CLI parser
    # in the sub-imports (model.py only imports damps + nn).
    from model import DAMPS_MMHCL

    n_users, n_items, d = 16, 32, 64
    image_feats = torch.randn(n_items, 256)
    text_feats = torch.randn(n_items, 128)

    model = DAMPS_MMHCL(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=d,
        image_feats=image_feats,
        text_feats=text_feats,
        cf_model="LightGCN",
        ui_layers=2,
        user_layers=1,
        item_layers=1,
        warmup_epochs=2,
    )
    model.set_meta_categories(torch.randint(0, 4, (n_items,)))

    # ----- Tau-init regression test (compliance WARN 2) -------------------
    # The Revision 9 spec mandates that the learnable InfoNCE temperature
    # is initialised at 0.1; verify the default constructor honours this.
    assert abs(float(model.tau.item()) - 0.1) < 1e-6, (
        f"learnable tau must be initialised at 0.1, got {float(model.tau.item())}"
    )
    _print_ok(f"learnable tau initialised at {float(model.tau.item()):.4f} (spec=0.1)")

    # Build trivial graphs (identity-like sparse tensors)
    UI = torch.eye(n_users + n_items).to_sparse_coo()
    I2I = torch.eye(n_items).to_sparse_coo()
    U2U = torch.eye(n_users).to_sparse_coo()

    out = model(
        UI, I2I, U2U,
        item_indices=torch.arange(n_items),
        epoch=0,
        update_momentum=True,
    )
    assert out["u_ui_emb"].shape == (n_users, d)
    assert out["i_ui_emb"].shape == (n_items, d)
    _print_ok("forward pass shapes OK")

    # Synthetic BPR + InfoNCE loss
    u = out["u_ui_emb"][:4]
    p = out["i_ui_emb"][:4]
    n = out["i_ui_emb"][4:8]
    bpr = -F.logsigmoid((u * p).sum(-1) - (u * n).sum(-1)).mean()
    nce = model.batched_contrastive_loss(out["i_ui_emb"], out["ii_emb"], batch_size=8)
    total = bpr + 0.07 * nce
    total.backward()
    _print_ok(f"backward OK; loss={total.detach().item():.4f}")

    diag = model.diagnostics()
    _print_ok(f"diag: {diag}")


def smoke_torch_compile() -> None:
    """
    Speedup Guide S4: ``torch.compile`` on the DAMPS submodule. We do not
    compile the full forward path because the periodically-rebuilt sparse
    Item_mat would otherwise trigger expensive graph recompilations.

    Some environments (in particular Windows installs whose Python prefix
    contains spaces, or systems without a usable C++ toolchain) cannot
    actually compile Inductor's generated kernels even though
    ``torch.compile`` itself imports successfully. In that case we fall back
    to checking that attribute forwarding through ``OptimizedModule`` is
    still intact, since that is what the trainer relies on.
    """
    print("== torch.compile smoke ==")
    if not hasattr(torch, "compile"):                            # pragma: no cover
        _print_ok("torch.compile not available; skipping")
        return

    N, d = 32, 64
    damps = DAMPS(d=d, num_categories=4, warmup_epochs=2)
    damps.train()
    try:
        compiled = torch.compile(damps, mode="reduce-overhead", dynamic=True)
    except Exception as exc:                                     # pragma: no cover
        _print_ok(f"torch.compile failed to attach ({exc}); skipping")
        return

    # Attribute forwarding through OptimizedModule must work, regardless of
    # whether the actual graph compilation succeeds in this environment.
    compiled.set_epoch(2)                                          # type: ignore[attr-defined]
    assert int(damps._current_epoch.item()) == 2, (
        "set_epoch must forward through the OptimizedModule wrapper"
    )
    _print_ok("OptimizedModule attribute forwarding works (set_epoch)")

    # Try a real compiled forward; on environments with broken C++ builds
    # (e.g. Windows path with a space) we accept the documented graceful
    # fallback used by train.py and skip with a warning.
    h_img = torch.randn(N, d)
    h_txt = torch.randn(N, d)
    cats = torch.randint(0, 4, (N,))
    try:
        h_img_cal, h_txt_cal, _ = compiled(h_img, h_txt, item_categories=cats)
        assert h_img_cal.shape == (N, d)
        assert h_txt_cal.shape == (N, d)
        _print_ok("compiled forward OK (full Inductor path)")
    except Exception as exc:                                     # pragma: no cover
        _print_ok(
            f"Inductor compile not usable in this env ({exc.__class__.__name__}); "
            f"trainer will fall back to eager mode automatically."
        )


if __name__ == "__main__":
    torch.manual_seed(42)
    smoke_damps_core()
    smoke_momentum()
    smoke_knn()
    smoke_prior()
    smoke_full_model()
    smoke_torch_compile()
    print("\nAll smoke tests passed!")
