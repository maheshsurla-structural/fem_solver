"""Tests for J2Plasticity3D — Phase 16.7.

The radial-return J2 model is pinned down by:

1. **Construction guard rails** — invalid E, nu, sigma_y, K_iso.
2. **Elastic D shape** — D_elastic is 6x6 symmetric with the right
   isotropic structure (diagonal = lambda + 2 mu on normals, mu on
   shears; off-diagonal block = lambda).
3. **Below-yield response is purely elastic** — no plastic strain
   accumulates as long as the von-Mises stress stays below sigma_y.
4. **At-yield consistency** — for any plastic step, the von-Mises
   stress equals sigma_y(alpha), so the stress always lies on the
   yield surface (with the radial-return projection).
5. **Hydrostatic loading produces no plasticity** — J2 only depends
   on the deviator, so equi-triaxial loading stays elastic forever.
6. **Isotropic hardening grows the yield surface** — the equivalent
   plastic strain alpha accumulates and sigma_y(alpha) increases
   linearly with K_iso.
7. **Commit / revert lifecycle** preserves and rolls back plastic
   state correctly.
8. **Pure shear loading hits yield at sigma_y / sqrt(3)** — the
   classical J2 result that shear yield is sigma_y / sqrt(3).
"""
import numpy as np
import pytest

from femsolver import J2Plasticity3D


# ====================================================== construction

def test_j2_rejects_invalid_E():
    with pytest.raises(ValueError, match="E"):
        J2Plasticity3D(E=0.0, nu=0.3, sigma_y=400e6)


def test_j2_rejects_invalid_nu():
    with pytest.raises(ValueError, match="nu"):
        J2Plasticity3D(E=2e11, nu=0.5, sigma_y=400e6)
    with pytest.raises(ValueError, match="nu"):
        J2Plasticity3D(E=2e11, nu=-1.5, sigma_y=400e6)


def test_j2_rejects_invalid_sigma_y():
    with pytest.raises(ValueError, match="sigma_y"):
        J2Plasticity3D(E=2e11, nu=0.3, sigma_y=0.0)


def test_j2_rejects_negative_K_iso():
    with pytest.raises(ValueError, match="K_iso"):
        J2Plasticity3D(E=2e11, nu=0.3, sigma_y=400e6, K_iso=-1e9)


# ====================================================== D matrix

def test_j2_elastic_D_is_isotropic_6x6():
    E, nu = 2.0e11, 0.3
    mat = J2Plasticity3D(E=E, nu=nu, sigma_y=400e6)
    D = mat.D_elastic()
    assert D.shape == (6, 6)
    assert np.allclose(D, D.T, rtol=1e-12)
    # Diagonal structure
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    assert D[0, 0] == pytest.approx(lam + 2 * mu, rel=1e-12)
    assert D[3, 3] == pytest.approx(mu, rel=1e-12)
    assert D[0, 1] == pytest.approx(lam, rel=1e-12)


# ====================================================== elastic phase

def test_j2_below_yield_is_purely_elastic():
    """For uniaxial strain at half-yield, no plastic strain should
    accumulate."""
    E, nu, sigma_y = 2e11, 0.3, 400e6
    mat = J2Plasticity3D(E=E, nu=nu, sigma_y=sigma_y, K_iso=0.0)
    # Apply eps_xx well below yield
    eps_xx = 0.001        # < yield strain ~ 0.002
    eps = np.array([eps_xx, -nu * eps_xx, -nu * eps_xx, 0.0, 0.0, 0.0])
    sigma, _ = mat.get_response(eps)
    # Hooke's law: sigma_xx = E * eps_xx (with transverse contraction)
    assert sigma[0] == pytest.approx(E * eps_xx, rel=1e-12)
    assert abs(sigma[1]) < 1.0e-6 * sigma_y
    # Plastic strain stays zero
    assert np.allclose(mat.eps_p_trial, 0.0, atol=1e-15)
    assert mat.alpha_trial == 0.0


# ====================================================== yield consistency

def test_j2_stress_lies_on_yield_surface_after_plastic_step():
    """Whenever a plastic step occurs, the radial return places the
    stress exactly on ``||s|| = sqrt(2/3) sigma_y(alpha)``."""
    E, nu, sigma_y = 2e11, 0.3, 400e6
    mat = J2Plasticity3D(E=E, nu=nu, sigma_y=sigma_y, K_iso=2e9)
    eps_xx = 0.01
    eps = np.array([eps_xx, -nu * eps_xx, -nu * eps_xx, 0.0, 0.0, 0.0])
    sigma, _ = mat.get_response(eps)
    sigma_vm = mat.von_mises_stress(sigma)
    sigma_y_current = mat.yield_stress(mat.alpha_trial)
    assert sigma_vm == pytest.approx(sigma_y_current, rel=1e-10)


def test_j2_hardening_grows_yield_surface():
    """Under monotonic loading, alpha increases and sigma_y(alpha) grows
    linearly with K_iso * alpha."""
    E, nu, sigma_y = 2e11, 0.3, 400e6
    K_iso = 5e9
    mat = J2Plasticity3D(E=E, nu=nu, sigma_y=sigma_y, K_iso=K_iso)
    eps_xx_sequence = np.linspace(0.0, 0.02, 100)
    for eps_xx in eps_xx_sequence:
        eps = np.array([eps_xx, -nu * eps_xx, -nu * eps_xx, 0, 0, 0])
        mat.get_response(eps)
        mat.commit_state()
    sigma_y_final = mat.yield_stress(mat.alpha_committed)
    expected = sigma_y + K_iso * mat.alpha_committed
    assert sigma_y_final == pytest.approx(expected, rel=1e-12)
    # Yield surface grew (alpha > 0)
    assert mat.alpha_committed > 0.0


# ====================================================== invariance

def test_j2_hydrostatic_strain_produces_no_plasticity():
    """J2 yield depends only on the deviator; equi-triaxial strain
    produces only volumetric stress and zero plastic strain."""
    E, nu, sigma_y = 2e11, 0.3, 400e6
    mat = J2Plasticity3D(E=E, nu=nu, sigma_y=sigma_y)
    eps_hyd = 0.01     # huge volumetric strain
    eps = np.array([eps_hyd, eps_hyd, eps_hyd, 0, 0, 0])
    sigma, _ = mat.get_response(eps)
    # All three normal stresses should be equal (equi-triaxial)
    assert sigma[0] == pytest.approx(sigma[1], rel=1e-12)
    assert sigma[0] == pytest.approx(sigma[2], rel=1e-12)
    # Shear stresses zero
    assert np.allclose(sigma[3:], 0.0, atol=1e-9)
    # No plastic strain
    assert np.allclose(mat.eps_p_trial, 0.0, atol=1e-15)
    # von-Mises stress is zero (pure hydrostatic)
    assert mat.von_mises_stress(sigma) < 1.0e-6 * sigma_y


def test_j2_pure_shear_yields_at_sigma_y_over_sqrt3():
    """For pure shear gamma_xy, yield happens when tau_xy = sigma_y / sqrt(3).
    This is the classical J2 prediction for shear yield."""
    E, nu, sigma_y = 2e11, 0.3, 400e6
    G = E / (2 * (1 + nu))
    mat = J2Plasticity3D(E=E, nu=nu, sigma_y=sigma_y, K_iso=0.0)
    # Apply a shear strain just below the yield value:
    # tau_xy = G * gamma_xy ; yield when tau_xy = sigma_y/sqrt(3)
    # -> gamma_xy_yield = sigma_y / (G * sqrt(3))
    gamma_yield = sigma_y / (G * np.sqrt(3.0))
    # Probe at 99% of yield
    eps_below = np.array([0, 0, 0, 0.99 * gamma_yield, 0, 0])
    sigma, _ = mat.get_response(eps_below)
    assert mat.alpha_trial == 0.0, "should still be elastic at 99% of yield shear"
    tau_below = sigma[3]
    assert tau_below == pytest.approx(G * 0.99 * gamma_yield, rel=1e-10)
    # And at 1.5x yield: should hit the yield surface
    mat2 = J2Plasticity3D(E=E, nu=nu, sigma_y=sigma_y, K_iso=0.0)
    eps_above = np.array([0, 0, 0, 1.5 * gamma_yield, 0, 0])
    sigma_above, _ = mat2.get_response(eps_above)
    sigma_vm = mat2.von_mises_stress(sigma_above)
    assert sigma_vm == pytest.approx(sigma_y, rel=1e-10)


# ====================================================== state lifecycle

def test_j2_commit_revert_lifecycle():
    """Reverting must roll back trial state to the last commit."""
    mat = J2Plasticity3D(E=2e11, nu=0.3, sigma_y=400e6, K_iso=2e9)
    nu = 0.3
    # Plastic step + commit
    eps_xx = 0.005
    eps = np.array([eps_xx, -nu * eps_xx, -nu * eps_xx, 0, 0, 0])
    mat.get_response(eps); mat.commit_state()
    eps_p_after_commit = mat.eps_p_committed.copy()
    alpha_after_commit = mat.alpha_committed
    # Trial further
    eps_xx2 = 0.010
    eps2 = np.array([eps_xx2, -nu * eps_xx2, -nu * eps_xx2, 0, 0, 0])
    mat.get_response(eps2)
    # Revert
    mat.revert_state()
    assert np.allclose(mat.eps_p_trial, eps_p_after_commit)
    assert mat.alpha_trial == alpha_after_commit


def test_j2_clone_is_independent():
    """Cloned material has its own state — pushing one further must
    not affect the other."""
    mat = J2Plasticity3D(E=2e11, nu=0.3, sigma_y=400e6, K_iso=2e9)
    nu = 0.3
    eps = np.array([0.005, -nu * 0.005, -nu * 0.005, 0, 0, 0])
    mat.get_response(eps); mat.commit_state()
    alpha_at_clone_time = mat.alpha_committed
    clone = mat.clone()
    # Push original further
    eps2 = np.array([0.02, -nu * 0.02, -nu * 0.02, 0, 0, 0])
    mat.get_response(eps2); mat.commit_state()
    # Clone's committed alpha is unchanged from the time of cloning
    assert clone.alpha_committed == pytest.approx(alpha_at_clone_time,
                                                    rel=1e-12)
    # Original advanced further
    assert mat.alpha_committed > clone.alpha_committed


# ====================================================== unloading

def test_j2_unloading_after_yield_is_elastic():
    """After a plastic excursion, unloading takes place along the
    elastic D matrix until the opposite yield surface is hit."""
    E, nu, sigma_y = 2e11, 0.3, 400e6
    mat = J2Plasticity3D(E=E, nu=nu, sigma_y=sigma_y, K_iso=2e9)
    # Push deep into yield, commit
    eps_xx_peak = 0.01
    eps_peak = np.array([eps_xx_peak, -nu * eps_xx_peak, -nu * eps_xx_peak,
                           0, 0, 0])
    sigma_peak, _ = mat.get_response(eps_peak)
    mat.commit_state()
    alpha_peak = mat.alpha_committed
    eps_p_peak = mat.eps_p_committed.copy()
    # Unload slightly (still positive but smaller)
    eps_xx_lo = eps_xx_peak * 0.9
    eps_lo = np.array([eps_xx_lo, -nu * eps_xx_lo, -nu * eps_xx_lo, 0, 0, 0])
    sigma_lo, _ = mat.get_response(eps_lo)
    # Plastic strain unchanged (elastic unloading)
    assert np.allclose(mat.eps_p_trial, eps_p_peak)
    assert mat.alpha_trial == alpha_peak
    # Stress drop is along elastic D matrix
    # sigma_lo = D_elastic @ (eps_lo - eps_p_peak)
    sigma_pred = mat.D_elastic() @ (eps_lo - eps_p_peak)
    assert np.allclose(sigma_lo, sigma_pred, rtol=1e-10)
