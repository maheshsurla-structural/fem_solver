"""Phase 51 tests -- Theme R advanced constitutive materials.

Coverage:

* :class:`MohrCoulomb3D` -- Phase 51.1
* :class:`ModifiedCamClay3D` -- Phase 51.2
* :class:`ConcreteDamage3D` -- Phase 51.4
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from femsolver import (
    ConcreteDamage3D,
    MohrCoulomb3D,
    ModifiedCamClay3D,
)


# ============================================================ Mohr-Coulomb

class TestMohrCoulomb:
    def test_apex_value(self):
        mat = MohrCoulomb3D(E=30e6, nu=0.3, cohesion=10e3, phi_deg=30.0)
        # apex = c * cot(phi) = 10e3 / tan(30°) = 17.32 kPa
        assert mat._apex == pytest.approx(10e3 / math.tan(math.radians(30)),
                                           rel=1e-9)

    def test_elastic_compression_no_yield(self):
        mat = MohrCoulomb3D(E=30e6, nu=0.3, cohesion=10e3, phi_deg=30.0)
        # Small uniform compression: well inside the cone
        eps = np.array([-1e-4, -1e-4, -1e-4, 0, 0, 0])
        sigma, _ = mat.get_response(eps)
        assert mat.yield_function(sigma) < 0.0
        # Verify D_e * eps = sigma (no plastic correction)
        np.testing.assert_allclose(sigma, mat._D_elastic @ eps, rtol=1e-10)

    def test_apex_projection_at_high_tension(self):
        mat = MohrCoulomb3D(E=30e6, nu=0.3, cohesion=10e3, phi_deg=30.0)
        # Large hydrostatic tension: should land at apex
        eps = np.array([0.005, 0.005, 0.005, 0, 0, 0])
        sigma, _ = mat.get_response(eps)
        # All principal stresses should equal the apex
        from femsolver.materials.multiaxial.mohr_coulomb import _principal_decomposition
        s, _ = _principal_decomposition(sigma)
        for si in s:
            assert si == pytest.approx(mat._apex, rel=1e-6)
        # Yield function should be ~0 at the apex
        assert abs(mat.yield_function(sigma)) < 1.0

    def test_commit_revert(self):
        mat = MohrCoulomb3D(E=30e6, nu=0.3, cohesion=10e3, phi_deg=30.0)
        eps = np.array([0.005, 0.005, 0.005, 0, 0, 0])
        mat.get_response(eps)
        # Trial state has non-zero plastic strain
        assert np.any(mat.eps_p_trial != 0)
        mat.revert_state()
        assert np.allclose(mat.eps_p_trial, 0)

    def test_validates_inputs(self):
        with pytest.raises(ValueError, match="E"):
            MohrCoulomb3D(E=0, nu=0.3, cohesion=10e3, phi_deg=30)
        with pytest.raises(ValueError, match="nu"):
            MohrCoulomb3D(E=1, nu=0.5, cohesion=10e3, phi_deg=30)
        with pytest.raises(ValueError, match="cohesion"):
            MohrCoulomb3D(E=1, nu=0.3, cohesion=-1, phi_deg=30)
        with pytest.raises(ValueError, match="phi_deg"):
            MohrCoulomb3D(E=1, nu=0.3, cohesion=1, phi_deg=90)
        with pytest.raises(ValueError, match="psi_deg"):
            MohrCoulomb3D(E=1, nu=0.3, cohesion=1,
                           phi_deg=30, psi_deg=45)

    def test_clone_is_independent(self):
        mat = MohrCoulomb3D(E=30e6, nu=0.3, cohesion=10e3, phi_deg=30)
        clone = mat.clone()
        mat.eps_p_committed = np.ones(6)
        assert np.allclose(clone.eps_p_committed, 0)


# ============================================================ Modified Cam-Clay

class TestModifiedCamClay:
    def test_initial_yield_function_negative(self):
        mat = ModifiedCamClay3D(E=10e6, nu=0.3, M=1.0,
                                  lambda_cc=0.20, kappa_cc=0.05,
                                  p_c0=100e3)
        # At zero stress, f = 0^2 + M^2 * 0 * (0 - p_c) = 0 -- on
        # the apex of the yield surface, so f = 0 exactly. Move
        # slightly into the elastic region.
        sigma = np.array([-50e3, -50e3, -50e3, 0, 0, 0])    # p_eff = 50 kPa
        f = mat.yield_function(sigma)
        # q = 0, p_eff = 50, p_c = 100: f = 50*(50-100)*1 = -2500 (in M^2*kPa^2 units)
        assert f < 0.0

    def test_elastic_response_inside_surface(self):
        mat = ModifiedCamClay3D(E=10e6, nu=0.3, M=1.0,
                                  lambda_cc=0.20, kappa_cc=0.05,
                                  p_c0=100e3)
        # Move from in-situ to slightly larger compression
        eps = np.array([-0.002, -0.002, -0.002, 0, 0, 0])
        sigma, _ = mat.get_response(eps)
        # Should equal elastic prediction
        np.testing.assert_allclose(sigma, mat._D_elastic @ eps, rtol=1e-10)

    def test_yielding_triggers_hardening(self):
        mat = ModifiedCamClay3D(E=10e6, nu=0.3, M=1.0,
                                  lambda_cc=0.20, kappa_cc=0.05,
                                  p_c0=100e3)
        # First reach in-situ state
        mat.get_response(np.array([-0.002, -0.002, -0.002, 0, 0, 0]))
        mat.commit_state()
        # Apply enough deviatoric strain to plastify
        eps = np.array([-0.002, -0.002, -0.008, 0, 0, 0])
        mat.get_response(eps)
        mat.commit_state()
        # p_c should have increased (NC clay hardens)
        assert mat.p_c >= 100e3

    def test_validates_inputs(self):
        with pytest.raises(ValueError, match="M"):
            ModifiedCamClay3D(E=1, nu=0.3, M=-1, lambda_cc=0.1,
                                kappa_cc=0.01, p_c0=1)
        with pytest.raises(ValueError, match="kappa_cc"):
            ModifiedCamClay3D(E=1, nu=0.3, M=1, lambda_cc=0.05,
                                kappa_cc=0.10, p_c0=1)
        with pytest.raises(ValueError, match="p_c0"):
            ModifiedCamClay3D(E=1, nu=0.3, M=1, lambda_cc=0.1,
                                kappa_cc=0.01, p_c0=-1)


# ============================================================ concrete damage

class TestConcreteDamage:
    def test_elastic_at_low_strain(self):
        mat = ConcreteDamage3D(E=30e9, nu=0.20,
                                 eps_t0=1.0e-4, eps_c0=1.0e-3)
        # Strain below tension threshold: no damage
        eps = np.array([5e-5, 0, 0, 0, 0, 0])
        sigma, _ = mat.get_response(eps)
        assert mat.d_trial == pytest.approx(0.0)
        # Linear elastic: sigma_xx = E * eps_xx (approx, with Poisson coupling)
        # Use 1D uniaxial: eps = (1, -nu, -nu, 0, 0, 0) * eps_ax
        eps_ax = 5e-5
        eps_uni = np.array([eps_ax, -0.2*eps_ax, -0.2*eps_ax, 0, 0, 0])
        sigma, _ = mat.get_response(eps_uni)
        assert sigma[0] == pytest.approx(30e9 * eps_ax, rel=1e-3)

    def test_tension_damage_triggers_at_threshold(self):
        mat = ConcreteDamage3D(E=30e9, nu=0.20,
                                 eps_t0=1.0e-4, eps_c0=1.0e-3)
        # At eps_t0 exactly, damage just starts (d=0)
        eps = np.array([1.0e-4, -0.2e-4, -0.2e-4, 0, 0, 0])
        mat.get_response(eps)
        assert mat.d_trial < 0.1
        mat.commit_state()
        # 2x the threshold: clear softening
        eps = np.array([2.0e-4, -0.4e-4, -0.4e-4, 0, 0, 0])
        sigma, _ = mat.get_response(eps)
        assert mat.d_trial > 0.5      # significant tensile damage
        assert sigma[0] < 30e9 * 1.0e-4   # softened below peak

    def test_compression_damage_softens(self):
        mat = ConcreteDamage3D(E=30e9, nu=0.20,
                                 eps_t0=1.0e-4, eps_c0=1.0e-3)
        # Apply strong compression in 1-direction with Poisson coupling
        peak = None
        last_sigma = 0.0
        for k in range(10):
            eps_ax = -(k + 1) * 0.5e-3
            eps = np.array([eps_ax, -0.2*eps_ax, -0.2*eps_ax, 0, 0, 0])
            sigma, _ = mat.get_response(eps)
            mat.commit_state()
            if peak is None or abs(sigma[0]) > peak:
                peak = abs(sigma[0])
            last_sigma = abs(sigma[0])
        # Past the peak, the stress should have softened
        assert last_sigma < peak

    def test_damage_never_decreases(self):
        mat = ConcreteDamage3D(E=30e9, nu=0.20,
                                 eps_t0=1.0e-4, eps_c0=1.0e-3)
        # Load to ~ 3 * eps_t0
        eps_load = np.array([3.0e-4, -0.6e-4, -0.6e-4, 0, 0, 0])
        mat.get_response(eps_load)
        d_at_peak = mat.d_trial
        mat.commit_state()
        # Unload partially -- damage must not decrease
        eps_unload = np.array([1.0e-4, -0.2e-4, -0.2e-4, 0, 0, 0])
        mat.get_response(eps_unload)
        assert mat.d_trial >= d_at_peak - 1e-12

    def test_validates_inputs(self):
        with pytest.raises(ValueError, match="E"):
            ConcreteDamage3D(E=0, nu=0.2)
        with pytest.raises(ValueError, match="eps_t0"):
            ConcreteDamage3D(E=1, nu=0.2, eps_t0=-1)
        with pytest.raises(ValueError, match="A_t"):
            ConcreteDamage3D(E=1, nu=0.2, A_t=1.5)
        with pytest.raises(ValueError, match="B_t"):
            ConcreteDamage3D(E=1, nu=0.2, B_t=0)
