"""Tests for the uniaxial-material library.

These exercise the constitutive interface at the *fiber* level — every
property the fiber section then aggregates over a discretised cross
section. Two laws are validated:

* :class:`UniaxialElastic` — stateless, ``sigma = E * eps``.
* :class:`UniaxialBilinear` — bilinear with kinematic hardening; the
  ``b = 0`` limit is elastic-perfectly-plastic.

The tests mirror the algebra of the moment-rotation spring
(:class:`BilinearMomentRotationSpring`) one layer deeper, on (sigma, eps)
instead of (M, theta_h). If the return mapping ever drifts these tests
will catch it before any fiber-section result does.
"""
import numpy as np
import pytest

from femsolver import UniaxialBilinear, UniaxialElastic


# ====================================================== UniaxialElastic ===

def test_elastic_response_is_linear():
    mat = UniaxialElastic(E=2.0e11)
    for eps in (-1.0e-3, 0.0, 1.0e-3, 5.0e-3):
        sigma, Et = mat.get_response(eps)
        assert sigma == pytest.approx(mat.E * eps, rel=1e-14)
        assert Et == pytest.approx(mat.E, rel=1e-14)


def test_elastic_rejects_nonpositive_E():
    with pytest.raises(ValueError):
        UniaxialElastic(E=0.0)
    with pytest.raises(ValueError):
        UniaxialElastic(E=-1.0)


def test_elastic_commit_revert_are_no_ops():
    """An elastic material has no state; the lifecycle calls must be safe
    no-ops so it composes with the FiberSection bookkeeping."""
    mat = UniaxialElastic(E=1.0e11)
    mat.get_response(1.0e-3)
    mat.commit_state()
    mat.revert_state()
    # next call must produce the same answer as before
    sigma, _ = mat.get_response(1.0e-3)
    assert sigma == pytest.approx(1.0e11 * 1.0e-3, rel=1e-14)


def test_elastic_clone_is_independent_object():
    mat = UniaxialElastic(E=1.0e11)
    mat2 = mat.clone()
    assert mat2 is not mat
    # numerical behaviour identical
    s1, _ = mat.get_response(1e-3)
    s2, _ = mat2.get_response(1e-3)
    assert s1 == s2


# ===================================================== UniaxialBilinear ===

def test_bilinear_rejects_invalid_args():
    with pytest.raises(ValueError):
        UniaxialBilinear(E=0.0, sigma_y=1.0)
    with pytest.raises(ValueError):
        UniaxialBilinear(E=1.0, sigma_y=0.0)
    with pytest.raises(ValueError):
        UniaxialBilinear(E=1.0, sigma_y=1.0, b=-0.01)
    with pytest.raises(ValueError):
        UniaxialBilinear(E=1.0, sigma_y=1.0, b=1.0)


@pytest.mark.parametrize("b", [0.0, 0.05, 0.1])
def test_bilinear_below_yield_is_elastic(b):
    mat = UniaxialBilinear(E=2.0e11, sigma_y=400.0e6, b=b)
    eps = 0.5 * mat.sigma_y / mat.E
    sigma, Et = mat.get_response(eps)
    assert sigma == pytest.approx(mat.E * eps, rel=1e-14)
    assert Et == pytest.approx(mat.E, rel=1e-14)
    assert mat.eps_p_trial == 0.0


def test_bilinear_at_yield_is_elastic_then_yields_on_next_step():
    """At the yield surface ``f = 0`` we still consider the step elastic;
    the *next* incremental step is what triggers plastic flow."""
    mat = UniaxialBilinear(E=2.0e11, sigma_y=400.0e6, b=0.0)
    sigma, Et = mat.get_response(mat.sigma_y / mat.E)
    assert sigma == pytest.approx(mat.sigma_y, rel=1e-14)
    assert mat.eps_p_trial == 0.0
    # next step past yield
    mat.commit_state()
    sigma2, Et2 = mat.get_response(2.0 * mat.sigma_y / mat.E)
    assert sigma2 == pytest.approx(mat.sigma_y, rel=1e-14)
    assert mat.eps_p_trial > 0.0
    assert Et2 == 0.0  # EPP post-yield


def test_epp_plateau_clips_stress_at_sigma_y():
    """EPP (b=0): once yielded, sigma stays at sigma_y under monotonic
    loading, plastic strain accumulates linearly with total strain."""
    mat = UniaxialBilinear(E=2.0e11, sigma_y=400.0e6, b=0.0)
    eps_values = [2.0, 3.0, 5.0, 10.0]
    sigmas = []
    for k in eps_values:
        mat.get_response(k * mat.sigma_y / mat.E)
        mat.commit_state()
        sigmas.append(mat.sigma_trial)
    np.testing.assert_allclose(sigmas, mat.sigma_y, rtol=1e-14)
    # plastic strain at the final state ~ (eps - eps_y)
    eps_final = 10.0 * mat.sigma_y / mat.E
    eps_y = mat.sigma_y / mat.E
    assert mat.eps_p_committed == pytest.approx(eps_final - eps_y, rel=1e-14)


def test_epp_unloading_is_elastic_with_original_E():
    """After plastic flow, unloading travels down a line of slope E."""
    mat = UniaxialBilinear(E=2.0e11, sigma_y=400.0e6, b=0.0)
    mat.get_response(3.0 * mat.sigma_y / mat.E); mat.commit_state()
    eps_p = mat.eps_p_committed
    # unload partially
    eps_unload = 2.0 * mat.sigma_y / mat.E
    sigma, Et = mat.get_response(eps_unload)
    assert sigma == pytest.approx(mat.E * (eps_unload - eps_p), rel=1e-14)
    assert Et == pytest.approx(mat.E, rel=1e-14)


def test_epp_reverse_yielding_clips_at_minus_sigma_y():
    """Load + into yield, then strongly negative: stress clips at -sigma_y."""
    mat = UniaxialBilinear(E=2.0e11, sigma_y=400.0e6, b=0.0)
    mat.get_response(3.0 * mat.sigma_y / mat.E); mat.commit_state()
    mat.get_response(-3.0 * mat.sigma_y / mat.E); mat.commit_state()
    assert mat.sigma_trial == pytest.approx(-mat.sigma_y, rel=1e-14)


@pytest.mark.parametrize("b", [0.05, 0.1, 0.2])
def test_post_yield_tangent_is_b_times_E(b):
    mat = UniaxialBilinear(E=2.0e11, sigma_y=400.0e6, b=b)
    mat.get_response(3.0 * mat.sigma_y / mat.E); mat.commit_state()
    assert mat.Et == pytest.approx(b * mat.E, rel=1e-12)


def test_post_yield_slope_finite_difference():
    """Finite-difference dM/d_eps on the post-yield branch matches b*E."""
    mat = UniaxialBilinear(E=2.0e11, sigma_y=400.0e6, b=0.1)
    eps_a = 2.0 * mat.sigma_y / mat.E
    eps_b = 3.0 * mat.sigma_y / mat.E
    mat.get_response(eps_a); mat.commit_state()
    sigma_a = mat.sigma_trial
    mat.get_response(eps_b); mat.commit_state()
    sigma_b = mat.sigma_trial
    assert (sigma_b - sigma_a) / (eps_b - eps_a) == pytest.approx(
        0.1 * mat.E, rel=1e-10
    )


def test_kinematic_hardening_translates_yield_surface():
    """After plastic loading in +, the back-stress q moves positive.
    Reverse yielding then occurs near (q - sigma_y), not at -sigma_y."""
    mat = UniaxialBilinear(E=2.0e11, sigma_y=400.0e6, b=0.1)
    mat.get_response(3.0 * mat.sigma_y / mat.E); mat.commit_state()
    q = mat.q_committed
    assert q > 0.0
    # Take a small step that just crosses reverse yield
    eps_reverse = (q - 1.2 * mat.sigma_y) / mat.E + mat.eps_p_committed
    mat.get_response(eps_reverse)
    # Should now be on reverse yield surface, sigma close to q - sigma_y
    assert mat.sigma_trial < q - 0.9 * mat.sigma_y


def test_revert_undoes_uncommitted_plastic_flow():
    mat = UniaxialBilinear(E=2.0e11, sigma_y=400.0e6, b=0.0)
    mat.get_response(3.0 * mat.sigma_y / mat.E)
    assert mat.eps_p_trial > 0.0
    mat.revert_state()
    assert mat.eps_p_trial == 0.0
    assert mat.q_trial == 0.0


def test_clone_is_independent():
    """Each fiber must own its plastic state; cloning is the mechanism."""
    mat = UniaxialBilinear(E=2.0e11, sigma_y=400.0e6, b=0.0)
    mat.get_response(3.0 * mat.sigma_y / mat.E); mat.commit_state()
    mat2 = mat.clone()
    # mat2 has the same committed state at the moment of cloning
    assert mat2.eps_p_committed == mat.eps_p_committed
    # mutating the original after cloning must not change the clone
    mat.get_response(10.0 * mat.sigma_y / mat.E); mat.commit_state()
    assert mat2.eps_p_committed != mat.eps_p_committed
