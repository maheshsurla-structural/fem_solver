"""Tests for the RayleighDamping helper.

Rayleigh damping is the construction ``C = alpha_M M + alpha_K K`` with
two free coefficients. The :meth:`from_modes` classmethod solves the
2x2 system that picks ``alpha_M, alpha_K`` so the damping ratio takes
prescribed values at two target angular frequencies. The damping ratio
at *any* frequency then follows the Rayleigh curve

    zeta(omega) = alpha_M / (2 omega) + alpha_K omega / 2.

These tests pin down the algebra and the from-modes inverse relation.
"""
import numpy as np
import pytest

from femsolver import RayleighDamping


def test_rayleigh_default_is_zero():
    damp = RayleighDamping()
    assert damp.alpha_M == 0.0
    assert damp.alpha_K == 0.0


def test_rayleigh_damping_ratio_curve():
    damp = RayleighDamping(alpha_M=0.1, alpha_K=0.001)
    omega = 5.0
    zeta_expected = 0.5 * (0.1 / 5.0 + 0.001 * 5.0)
    assert damp.damping_ratio_at(omega) == pytest.approx(zeta_expected, rel=1e-14)


@pytest.mark.parametrize(
    "omega_1,omega_2,zeta_1,zeta_2",
    [
        (1.0, 10.0, 0.05, 0.05),    # equal damping at two modes
        (2.0, 20.0, 0.02, 0.05),    # different
        (1.0, 100.0, 0.01, 0.01),   # wide separation
    ],
)
def test_rayleigh_from_modes_recovers_targets(omega_1, omega_2, zeta_1, zeta_2):
    """Inverse relation: the damping ratios at the two target
    frequencies must match the prescribed values."""
    damp = RayleighDamping.from_modes(omega_1, zeta_1, omega_2, zeta_2)
    assert damp.damping_ratio_at(omega_1) == pytest.approx(zeta_1, rel=1e-12)
    assert damp.damping_ratio_at(omega_2) == pytest.approx(zeta_2, rel=1e-12)


def test_rayleigh_from_modes_rejects_equal_omegas():
    with pytest.raises(ValueError):
        RayleighDamping.from_modes(5.0, 0.05, 5.0, 0.05)


def test_rayleigh_from_modes_rejects_nonpositive_omegas():
    with pytest.raises(ValueError):
        RayleighDamping.from_modes(0.0, 0.05, 5.0, 0.05)
    with pytest.raises(ValueError):
        RayleighDamping.from_modes(1.0, 0.05, -1.0, 0.05)


def test_rayleigh_build_is_linear_combination():
    """C = alpha_M M + alpha_K K — bit-for-bit."""
    import scipy.sparse as sp
    M = sp.diags([1.0, 2.0, 3.0]).tocsc()
    K = sp.diags([10.0, 20.0, 30.0]).tocsc()
    damp = RayleighDamping(alpha_M=0.5, alpha_K=0.01)
    C = damp.build(M, K).toarray()
    expected = 0.5 * M.toarray() + 0.01 * K.toarray()
    np.testing.assert_allclose(C, expected, rtol=1e-14)
