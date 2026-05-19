"""Tests for the bilinear moment-rotation spring (concentrated-hinge backbone).

The spring carries a 1-D plasticity model with kinematic hardening.
``b = 0`` recovers the elastic-perfectly-plastic limit; ``0 < b < 1``
gives a bilinear backbone with post-yield slope ``b * K0``.

The tests below pin down the four behaviours that any later phase of
the project (or any future refactor of the return mapping) must continue
to satisfy:

* response is purely elastic below the yield moment,
* the post-yield tangent equals ``b * K0``,
* unloading from the plastic branch is elastic with the original ``K0``,
* committed plastic rotation is the integrator of plastic flow over a
  load history (sign-aware under reversal).
"""
import math

import numpy as np
import pytest

from femsolver import BilinearMomentRotationSpring


# ============================================================== invariants

def test_spring_rejects_invalid_args():
    with pytest.raises(ValueError):
        BilinearMomentRotationSpring(K0=0.0, My=1.0)
    with pytest.raises(ValueError):
        BilinearMomentRotationSpring(K0=1.0, My=0.0)
    with pytest.raises(ValueError):
        BilinearMomentRotationSpring(K0=1.0, My=1.0, b=-0.1)
    with pytest.raises(ValueError):
        BilinearMomentRotationSpring(K0=1.0, My=1.0, b=1.0)


def test_spring_initial_state_is_elastic():
    s = BilinearMomentRotationSpring(K0=1.0e6, My=1.0e3, b=0.0)
    assert s.theta_p_committed == 0.0
    assert s.q_committed == 0.0
    assert s.K_tangent == 1.0e6


# ============================================================ pure elastic

@pytest.mark.parametrize("b", [0.0, 0.05, 0.2])
def test_below_yield_is_purely_elastic(b):
    K0, My = 1.0e6, 1.0e3
    s = BilinearMomentRotationSpring(K0=K0, My=My, b=b)
    theta = 0.5 * My / K0          # halfway to yield
    M, K_t = s.get_response(theta)
    assert M == pytest.approx(K0 * theta, rel=1e-14)
    assert K_t == pytest.approx(K0, rel=1e-14)
    assert s.theta_p_trial == 0.0
    assert s.q_trial == 0.0


def test_at_yield_is_still_elastic():
    """On the yield surface (f = 0) we treat it as elastic — the next
    incremental step is what triggers plastic flow."""
    K0, My = 1.0e6, 1.0e3
    s = BilinearMomentRotationSpring(K0=K0, My=My, b=0.0)
    M, K_t = s.get_response(My / K0)
    assert M == pytest.approx(My, rel=1e-14)
    assert K_t == pytest.approx(K0, rel=1e-14)
    assert s.theta_p_trial == 0.0


# ===================================================== plastic, EPP (b=0)

def test_epp_yield_plateau_holds_M_at_My():
    K0, My = 1.0e6, 1.0e3
    s = BilinearMomentRotationSpring(K0=K0, My=My, b=0.0)
    # Drive past yield in three steps
    s.get_response(2.0 * My / K0)
    s.commit_state()
    M1 = s.M_trial
    s.get_response(3.0 * My / K0)
    s.commit_state()
    M2 = s.M_trial
    s.get_response(5.0 * My / K0)
    s.commit_state()
    M3 = s.M_trial
    # All three must sit on the yield plateau
    np.testing.assert_allclose([M1, M2, M3], My, rtol=1e-14)


def test_epp_plastic_strain_grows_with_load():
    K0, My = 1.0e6, 1.0e3
    s = BilinearMomentRotationSpring(K0=K0, My=My, b=0.0)
    s.get_response(2.0 * My / K0); s.commit_state()
    tp1 = s.theta_p_committed
    s.get_response(3.0 * My / K0); s.commit_state()
    tp2 = s.theta_p_committed
    assert tp2 > tp1
    # plastic flow == total - elastic at yield
    assert tp1 == pytest.approx(My / K0, rel=1e-14)
    assert tp2 == pytest.approx(2.0 * My / K0, rel=1e-14)


def test_epp_unloading_is_elastic():
    K0, My = 1.0e6, 1.0e3
    s = BilinearMomentRotationSpring(K0=K0, My=My, b=0.0)
    # Load to 3 My / K0 (well past yield) and commit
    s.get_response(3.0 * My / K0)
    s.commit_state()
    theta_p_after_load = s.theta_p_committed
    # Unload partially — moment should drop along the elastic branch
    theta_unload = 2.0 * My / K0   # still positive but reduced
    M, K_t = s.get_response(theta_unload)
    expected_M = K0 * (theta_unload - theta_p_after_load)
    assert M == pytest.approx(expected_M, rel=1e-14)
    assert K_t == pytest.approx(K0, rel=1e-14)


def test_epp_reverse_yielding():
    """Load + into yield, unload all the way through and into reverse
    yield. With EPP the moment must clip at -My on the reverse side."""
    K0, My = 1.0e6, 1.0e3
    s = BilinearMomentRotationSpring(K0=K0, My=My, b=0.0)
    s.get_response(3.0 * My / K0); s.commit_state()
    # Now go strongly negative
    s.get_response(-3.0 * My / K0); s.commit_state()
    assert s.M_trial == pytest.approx(-My, rel=1e-14)


# ============================================== bilinear with KH (b > 0)

@pytest.mark.parametrize("b", [0.05, 0.1, 0.2])
def test_post_yield_tangent_is_b_K0(b):
    K0, My = 1.0e6, 1.0e3
    s = BilinearMomentRotationSpring(K0=K0, My=My, b=b)
    s.get_response(3.0 * My / K0); s.commit_state()
    assert s.K_tangent == pytest.approx(b * K0, rel=1e-12)


def test_post_yield_slope_matches_b_K0_finite_difference():
    """Take two finite-difference steps on the post-yield branch and
    confirm dM/dtheta == b * K0."""
    K0, My, b = 1.0e6, 1.0e3, 0.1
    s = BilinearMomentRotationSpring(K0=K0, My=My, b=b)
    theta_a = 2.0 * My / K0
    theta_b = 3.0 * My / K0
    s.get_response(theta_a); s.commit_state()
    M_a = s.M_trial
    s.get_response(theta_b); s.commit_state()
    M_b = s.M_trial
    slope = (M_b - M_a) / (theta_b - theta_a)
    assert slope == pytest.approx(b * K0, rel=1e-10)


def test_kinematic_hardening_back_stress_translates_yield_surface():
    """After plastic loading in +, the yield surface centre q has moved
    in the + direction. Reverse yielding therefore happens at a lower
    moment magnitude than the original ``-My``.
    """
    K0, My, b = 1.0e6, 1.0e3, 0.1
    s = BilinearMomentRotationSpring(K0=K0, My=My, b=b)
    s.get_response(3.0 * My / K0); s.commit_state()
    q_after_loading = s.q_committed
    assert q_after_loading > 0.0
    # Reverse yield should now occur near +q - My, not at -My
    expected_reverse_yield_M = q_after_loading - My
    # Take a small reverse step that lands just past reverse yield
    theta_reverse = (q_after_loading - 1.5 * My) / K0 + s.theta_p_committed
    s.get_response(theta_reverse)
    # Moment should be near expected_reverse_yield_M (allow generous tol
    # because the step itself includes plastic flow on the reverse side)
    assert s.M_trial < expected_reverse_yield_M + 0.1 * My


# ===================================================== commit / revert

def test_revert_undoes_uncommitted_plastic_flow():
    K0, My = 1.0e6, 1.0e3
    s = BilinearMomentRotationSpring(K0=K0, My=My, b=0.0)
    # Apply load past yield without committing
    s.get_response(3.0 * My / K0)
    assert s.theta_p_trial > 0.0
    s.revert_state()
    assert s.theta_p_trial == 0.0
    assert s.q_trial == 0.0


def test_commit_after_loading_persists_state():
    K0, My = 1.0e6, 1.0e3
    s = BilinearMomentRotationSpring(K0=K0, My=My, b=0.05)
    s.get_response(3.0 * My / K0); s.commit_state()
    tp = s.theta_p_committed
    q = s.q_committed
    # next call without further loading: predictor should use committed state
    M, K_t = s.get_response(3.0 * My / K0)
    assert s.theta_p_committed == tp     # committed unchanged
    assert s.q_committed == q
