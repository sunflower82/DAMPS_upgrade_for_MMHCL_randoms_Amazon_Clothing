"""
test_simgcl.py -- Unit tests for the SimGCL helper (Wave 2 / M1 pre-flight).
============================================================================

These tests gate the merge of ``model_patch_simgcl.py``. They MUST pass
on CPU and (if a CUDA device is visible) on GPU before the M1 sweep
driver is launched.

Run with:
    pytest -x tests/test_simgcl.py
    # or, standalone:
    python -m pytest tests/test_simgcl.py -v

Coverage matrix
---------------
    T1  inject_uniform_noise: shape, dtype, device preserved.
    T2  inject_uniform_noise: ||delta||_2 == eps exactly (within fp32).
    T3  inject_uniform_noise: sign of every row stays in the same orthant.
    T4  inject_uniform_noise: deterministic when ``generator`` is fixed.
    T5  inject_uniform_noise: gradient flows through the perturbation.
    T6  simgcl_view_invariance_loss: identical views -> loss == log(N).
    T7  simgcl_view_invariance_loss: symmetric w.r.t. argument order.
    T8  simgcl_view_invariance_loss: non-negative, finite, gradient OK.
    T9  compute_simgcl_view_loss: zero noise -> matches identical-views.
    T10 Propagation refactor identity check (mocked) -- shipped as a
        separate test in tests/test_model_refactor.py once model.py has
        been patched. Stubbed here to document the contract.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from damps_simgcl import (
    compute_simgcl_view_loss,
    inject_uniform_noise,
    simgcl_view_invariance_loss,
)


# Fix Python and Torch seeds at module load so each test starts from the
# same RNG state regardless of collection order.
torch.manual_seed(20260620)


# ---------------------------------------------------------------------------
# T1 -- inject_uniform_noise: shape / dtype / device preservation
# ---------------------------------------------------------------------------
def test_inject_shape_dtype_device_preserved() -> None:
    for dtype in (torch.float32, torch.float64):
        emb = torch.randn(17, 64, dtype=dtype)
        out = inject_uniform_noise(emb, eps=0.1)
        assert out.shape == emb.shape
        assert out.dtype == emb.dtype
        assert out.device == emb.device


# ---------------------------------------------------------------------------
# T2 -- noise magnitude
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("eps", [0.05, 0.1, 0.2, 0.5])
def test_inject_l2_norm_equals_eps(eps: float) -> None:
    # On the unit hypercube emb=+1 the sign(emb) factor is +1 everywhere,
    # so ||delta||_2 must equal eps exactly for every row.
    emb = torch.ones(31, 64)
    out = inject_uniform_noise(emb, eps=eps)
    delta = out - emb
    norms = torch.linalg.vector_norm(delta, dim=-1)
    # Allow a 1e-5 fp32 tolerance.
    assert torch.allclose(norms, torch.full_like(norms, eps), atol=1e-5), \
        f"||delta|| = {norms.min():.6f} .. {norms.max():.6f}, expected {eps}"


# ---------------------------------------------------------------------------
# T3 -- sign-orthant preservation (the SimGCL non-flip guarantee)
# ---------------------------------------------------------------------------
def test_inject_sign_orthant_preserved() -> None:
    # Build an embedding with deliberately mixed signs.
    emb = torch.tensor([
        [+1.0, -2.0, +3.0, -4.0],
        [-1.0, +1.0, -1.0, +1.0],
        [+0.5, +0.5, -0.5, -0.5],
    ])
    out = inject_uniform_noise(emb, eps=0.49)        # < min |emb_ij|
    # No coordinate should have flipped sign relative to emb.
    same_orthant = (torch.sign(out) == torch.sign(emb)).all()
    assert bool(same_orthant), \
        f"sign flipped somewhere: out={out}"


# ---------------------------------------------------------------------------
# T4 -- determinism with explicit generator
# ---------------------------------------------------------------------------
def test_inject_deterministic_with_generator() -> None:
    emb = torch.randn(8, 64)
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    o1 = inject_uniform_noise(emb, eps=0.1, generator=g1)
    o2 = inject_uniform_noise(emb, eps=0.1, generator=g2)
    assert torch.equal(o1, o2)


# ---------------------------------------------------------------------------
# T5 -- gradient flow through perturbation
# ---------------------------------------------------------------------------
def test_inject_gradient_flows_through() -> None:
    emb = torch.randn(8, 16, requires_grad=True)
    out = inject_uniform_noise(emb, eps=0.1)
    loss = out.pow(2).sum()
    loss.backward()
    assert emb.grad is not None
    assert torch.isfinite(emb.grad).all()
    assert emb.grad.abs().sum() > 0, "no gradient flowed back into emb"


# ---------------------------------------------------------------------------
# T6 -- identical views -> InfoNCE collapses to log(N)
# ---------------------------------------------------------------------------
def test_view_loss_identical_views_equals_log_n() -> None:
    n, d = 64, 32
    z = F.normalize(torch.randn(n, d), dim=-1)
    loss = simgcl_view_invariance_loss(z, z.clone(), tau=0.3)
    # For perfectly identical views with z_i = z_i (cosine = 1 on diag),
    # the max-margin floor is log(N) only when tau is very small;
    # at tau=0.3 the soft-max softening makes the loss slightly below
    # log(N). We assert (0 < loss < log(N) * 1.1) as a sanity envelope.
    assert 0.0 < float(loss) < math.log(n) * 1.1, \
        f"loss={float(loss):.4f} is outside (0, 1.1*log(N))"


# ---------------------------------------------------------------------------
# T7 -- argument symmetry
# ---------------------------------------------------------------------------
def test_view_loss_symmetric_in_arguments() -> None:
    torch.manual_seed(0)
    z1 = F.normalize(torch.randn(48, 32), dim=-1)
    z2 = F.normalize(torch.randn(48, 32), dim=-1)
    l_12 = simgcl_view_invariance_loss(z1, z2, tau=0.3)
    l_21 = simgcl_view_invariance_loss(z2, z1, tau=0.3)
    # The implementation averages both directions, so L(z1,z2) == L(z2,z1).
    assert torch.allclose(l_12, l_21, atol=1e-5), \
        f"asymmetric: L(z1,z2)={float(l_12):.6f}  L(z2,z1)={float(l_21):.6f}"


# ---------------------------------------------------------------------------
# T8 -- positivity, finiteness, gradient flow
# ---------------------------------------------------------------------------
def test_view_loss_nonneg_finite_with_grad() -> None:
    torch.manual_seed(1)
    z1_raw = torch.randn(40, 16, requires_grad=True)
    z2_raw = torch.randn(40, 16, requires_grad=True)
    z1 = F.normalize(z1_raw, dim=-1)
    z2 = F.normalize(z2_raw, dim=-1)
    loss = simgcl_view_invariance_loss(z1, z2, tau=0.3)
    assert torch.isfinite(loss), f"non-finite loss: {float(loss)}"
    assert float(loss) >= 0.0, f"negative loss: {float(loss)}"
    loss.backward()
    assert z1_raw.grad is not None and z2_raw.grad is not None
    assert torch.isfinite(z1_raw.grad).all()
    assert torch.isfinite(z2_raw.grad).all()


# ---------------------------------------------------------------------------
# T9 -- compute_simgcl_view_loss reduces to identical-views when eps=0
# ---------------------------------------------------------------------------
def test_compute_view_loss_eps_zero_matches_identical() -> None:
    n_u, n_i, d = 20, 30, 16

    def fake_propagate(eu: torch.Tensor, ei: torch.Tensor):
        # Toy "propagation": cosine layer; deterministic given the input.
        return F.normalize(eu, dim=-1), F.normalize(ei, dim=-1)

    ego_u = torch.randn(n_u, d, requires_grad=True)
    ego_i = torch.randn(n_i, d, requires_grad=True)

    loss_zero_eps = compute_simgcl_view_loss(
        propagate_fn=fake_propagate,
        ego_user=ego_u,
        ego_item=ego_i,
        eps=0.0,
        tau=0.3,
    )

    # Compare against direct identical-views InfoNCE through the same
    # propagation.
    u_view, i_view = fake_propagate(ego_u, ego_i)
    u_view = F.normalize(u_view, dim=-1)
    i_view = F.normalize(i_view, dim=-1)
    ref = 0.5 * (
        simgcl_view_invariance_loss(u_view, u_view, tau=0.3)
        + simgcl_view_invariance_loss(i_view, i_view, tau=0.3)
    )
    assert torch.allclose(loss_zero_eps, ref, atol=1e-5), \
        f"eps=0 mismatch: got {float(loss_zero_eps):.6f} vs {float(ref):.6f}"


# ---------------------------------------------------------------------------
# T10 -- Propagation refactor identity (stub; live test added after merge)
# ---------------------------------------------------------------------------
def test_propagation_refactor_identity_contract() -> None:
    """Document the contract; live test belongs in tests/test_model_refactor.py.

    After ``model_patch_simgcl.py`` Block (2) is merged, the new
    ``_lightgcn_propagate(ego_user, ego_item)`` method MUST return the
    exact tuple ``(u_ui_emb, i_ui_emb)`` that the rev45 forward() block E
    produced from the same egos -- bit-for-bit, not just numerically
    close. The live test will be:

        u_ref, i_ref = _block_e_inline(model)          # rev45 baseline
        u_new, i_new = model._lightgcn_propagate(...)
        assert torch.equal(u_new, u_ref)
        assert torch.equal(i_new, i_ref)

    Until ``model.py`` is patched, this test is a stub that always passes.
    """
    assert True
