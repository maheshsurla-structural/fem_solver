"""Tests for DruckerPrager3D (Phase 16.9).

The Drucker-Prager cone is pinned down by:

1. **Construction guard rails** — invalid E, nu, alpha, k.
2. **Mohr-Coulomb conversion** matches the closed-form outer-cone
   formula ``alpha = 2 sin(phi)/(sqrt(3)(3 - sin(phi)))``.
3. **Yield function sign** below and above the cone.
4. **Elastic below yield** — small strain produces sigma = D_elastic * eps
   and no plastic strain.
5. **Plastic-step consistency** — after a plastic step, ``f(sigma_n+1) = 0``
   exactly on the smooth cone face.
6. **Pure-cohesion (phi = 0) limit** — DP reduces to a J2-style cone
   with ``k = 2 c / sqrt(3)``.
7. **Hydrostatic compression cannot yield** — purely-compressive
   triaxial stress moves *into* the cone (f decreases).
8. **State lifecycle** — commit/revert/clone behave correctly.
"""
import math

import numpy as np
import pytest

from femsolver import DruckerPrager3D


# ====================================================== construction

def test_dp_rejects_invalid_E():
    with pytest.raises(ValueError, match="E"):
        DruckerPrager3D(E=0.0, nu=0.3, alpha=0.1, k=1e4)


def test_dp_rejects_invalid_nu():
    with pytest.raises(ValueError, match="nu"):
        DruckerPrager3D(E=2e7, nu=0.5, alpha=0.1, k=1e4)


def test_dp_rejects_negative_alpha():
    with pytest.raises(ValueError, match="alpha"):
        DruckerPrager3D(E=2e7, nu=0.3, alpha=-0.1, k=1e4)


def test_dp_rejects_zero_k():
    with pytest.raises(ValueError, match="k"):
        DruckerPrager3D(E=2e7, nu=0.3, alpha=0.1, k=0.0)


# ====================================================== M-C conversion

def test_dp_from_mc_outer_cone_formula():
    """alpha = 2 sin(phi) / (sqrt(3) (3 - sin(phi))) for outer cone."""
    phi_deg = 30.0
    c = 50e3
    mat = DruckerPrager3D.from_mohr_coulomb(E=2e7, nu=0.3, cohesion=c,
                                              phi_deg=phi_deg, outer_cone=True)
    sp = math.sin(math.radians(phi_deg))
    cp = math.cos(math.radians(phi_deg))
    alpha_expected = 2.0 * sp / (math.sqrt(3.0) * (3.0 - sp))
    k_expected = 6.0 * c * cp / (math.sqrt(3.0) * (3.0 - sp))
    assert mat.alpha == pytest.approx(alpha_expected, rel=1e-12)
    assert mat.k == pytest.approx(k_expected, rel=1e-12)


def test_dp_from_mc_zero_friction_reduces_to_j2_style():
    """phi = 0 -> alpha = 0, k = 2 c / sqrt(3) (J2-style cohesion)."""
    c = 100e3
    mat = DruckerPrager3D.from_mohr_coulomb(E=2e7, nu=0.3, cohesion=c,
                                              phi_deg=0.0)
    assert mat.alpha == pytest.approx(0.0, abs=1e-12)
    assert mat.k == pytest.approx(2.0 * c / math.sqrt(3.0), rel=1e-12)


def test_dp_from_mc_inner_cone():
    phi_deg = 30.0
    c = 50e3
    mat = DruckerPrager3D.from_mohr_coulomb(E=2e7, nu=0.3, cohesion=c,
                                              phi_deg=phi_deg, outer_cone=False)
    sp = math.sin(math.radians(phi_deg))
    alpha_expected = 2.0 * sp / (math.sqrt(3.0) * (3.0 + sp))
    assert mat.alpha == pytest.approx(alpha_expected, rel=1e-12)


# ====================================================== yield function

def test_dp_yield_function_zero_inside_cone():
    mat = DruckerPrager3D(E=2e7, nu=0.3, alpha=0.18, k=60e3)
    # Zero stress -> f = -k < 0
    sigma = np.zeros(6)
    assert mat.yield_function(sigma) == pytest.approx(-60e3, rel=1e-12)


def test_dp_below_yield_is_elastic():
    """Small strain stays within the cone -> no plastic strain."""
    mat = DruckerPrager3D(E=2e7, nu=0.3, alpha=0.18, k=60e3)
    nu = 0.3
    eps = np.array([0.0001, -nu * 0.0001, -nu * 0.0001, 0, 0, 0])
    sigma, D = mat.get_response(eps)
    assert mat.yield_function(sigma) < 0.0
    assert np.allclose(mat.eps_p_trial, 0.0, atol=1e-15)


# ====================================================== plastic consistency

def test_dp_plastic_step_lands_on_yield_surface():
    """After radial return on the smooth cone face, f(sigma_n+1) = 0
    exactly."""
    mat = DruckerPrager3D(E=2e7, nu=0.3, alpha=0.18, k=60e3)
    nu = 0.3
    # Pure shear strain large enough to push past yield
    eps = np.array([0, 0, 0, 0.05, 0, 0])
    sigma, _ = mat.get_response(eps)
    f = mat.yield_function(sigma)
    assert f == pytest.approx(0.0, abs=1e-3), f"f after return = {f}"


def test_dp_compressive_pressure_dependence():
    """Under uniaxial compression, the DP material yields at a higher
    stress than its zero-friction (J2-only) counterpart -- friction
    angle adds confining strength."""
    nu = 0.3
    # Friction material
    mat_fric = DruckerPrager3D.from_mohr_coulomb(
        E=2e7, nu=nu, cohesion=50e3, phi_deg=30.0)
    # Pure cohesion
    mat_coh = DruckerPrager3D.from_mohr_coulomb(
        E=2e7, nu=nu, cohesion=50e3, phi_deg=0.0)

    def yield_compression(mat):
        # Sweep eps_xx in compression with elastic transverse contraction
        for eps_xx_mag in np.linspace(0.0001, 0.05, 200):
            mat2 = type(mat)(E=mat.E, nu=mat.nu,
                              alpha=mat.alpha, k=mat.k)
            eps = np.array([-eps_xx_mag, nu * eps_xx_mag, nu * eps_xx_mag,
                              0, 0, 0])
            sigma, _ = mat2.get_response(eps)
            if abs(mat2.yield_function(sigma)) < 1e-3:
                return abs(sigma[0])
        return float("inf")

    sigma_fric = yield_compression(mat_fric)
    sigma_coh = yield_compression(mat_coh)
    # Friction material yields at higher compressive stress
    assert sigma_fric > sigma_coh, (
        f"Friction: {sigma_fric:.3e}, cohesion-only: {sigma_coh:.3e}"
    )


def test_dp_hydrostatic_compression_stays_elastic():
    """Pure hydrostatic compression (I1 << 0) can never reach the
    Drucker-Prager cone (the cone opens outward toward positive I1).
    Hence sigma is purely elastic, no matter how large the volumetric
    compression."""
    mat = DruckerPrager3D(E=2e7, nu=0.3, alpha=0.18, k=60e3)
    for eps_v_mag in (0.001, 0.01, 0.1):
        mat_fresh = DruckerPrager3D(E=2e7, nu=0.3, alpha=0.18, k=60e3)
        eps = np.array([-eps_v_mag, -eps_v_mag, -eps_v_mag, 0, 0, 0])
        sigma, _ = mat_fresh.get_response(eps)
        assert mat_fresh.yield_function(sigma) < 0.0
        # All shear-like / deviatoric components should be near zero
        s, _ = DruckerPrager3D._deviator(sigma)
        assert np.max(np.abs(s)) < 1e-3 * abs(sigma[0])


# ====================================================== state lifecycle

def test_dp_commit_revert():
    mat = DruckerPrager3D(E=2e7, nu=0.3, alpha=0.18, k=60e3)
    eps_yield = np.array([0, 0, 0, 0.05, 0, 0])
    mat.get_response(eps_yield); mat.commit_state()
    eps_p_after = mat.eps_p_committed.copy()
    sigma_after = mat.sigma_committed.copy()
    # Trial further
    eps_more = np.array([0, 0, 0, 0.1, 0, 0])
    mat.get_response(eps_more)
    # Revert
    mat.revert_state()
    assert np.allclose(mat.eps_p_trial, eps_p_after)
    assert np.allclose(mat.sigma_trial, sigma_after)


def test_dp_clone_is_independent():
    mat = DruckerPrager3D(E=2e7, nu=0.3, alpha=0.18, k=60e3)
    eps = np.array([0, 0, 0, 0.05, 0, 0])
    mat.get_response(eps); mat.commit_state()
    clone = mat.clone()
    # Push original further
    eps2 = np.array([0, 0, 0, 0.2, 0, 0])
    mat.get_response(eps2); mat.commit_state()
    # Clone state unchanged
    assert not np.allclose(mat.sigma_committed, clone.sigma_committed)


# ====================================================== continuum tangent

def test_dp_plastic_tangent_is_softer_than_elastic():
    """The continuum elasto-plastic tangent must be 'softer' than the
    elastic D in the loading direction (eigenvalues less than elastic)."""
    mat = DruckerPrager3D(E=2e7, nu=0.3, alpha=0.18, k=60e3)
    # Yield in shear
    eps = np.array([0, 0, 0, 0.05, 0, 0])
    _, D_plastic = mat.get_response(eps)
    D_elastic = mat.D_elastic()
    # Frobenius norm should be smaller in the plastic tangent
    assert np.linalg.norm(D_plastic) < np.linalg.norm(D_elastic)


def test_dp_plastic_tangent_symmetric_and_psd():
    """The continuum tangent for associated flow is symmetric and
    positive semi-definite."""
    mat = DruckerPrager3D(E=2e7, nu=0.3, alpha=0.18, k=60e3)
    eps = np.array([0, 0, 0, 0.05, 0, 0])
    _, D = mat.get_response(eps)
    assert np.allclose(D, D.T, rtol=1e-12)
    eigvals = np.linalg.eigvalsh(D)
    # Allow tiny negative eigenvalue from floating-point in the
    # plastic-flow direction
    assert min(eigvals) > -1.0e-6 * np.max(np.abs(eigvals))
