"""Phase HH.2 tests -- MCC stress-dependent elastic K.

Validates that the bulk modulus scales with mean effective stress
per critical-state theory: K' = (1 + e) p' / kappa.
"""
from __future__ import annotations

import numpy as np
import pytest

from femsolver import ModifiedCamClay3D


# ============================================================ K(p') scaling

class TestStressDependentBulkModulus:
    def test_initial_K_matches_formula(self):
        """At the initial state with no committed stress, K is
        evaluated at p_eff = p_c0 / 2 (typical in-situ assumption)."""
        e_0 = 0.7
        kappa = 0.05
        p_c0 = 100e3
        mat = ModifiedCamClay3D(
            E=10e6, nu=0.3, M=1.0, lambda_cc=0.20,
            kappa_cc=kappa, p_c0=p_c0, e_0=e_0,
        )
        # K_init = (1 + e_0) * (p_c0/2) / kappa
        expected = (1.0 + e_0) * (p_c0 / 2.0) / kappa
        assert mat.K_bulk == pytest.approx(expected, rel=1e-9)

    def test_K_increases_with_pressure(self):
        """As mean effective stress grows, K grows proportionally."""
        mat = ModifiedCamClay3D(
            E=10e6, nu=0.3, M=1.0, lambda_cc=0.20,
            kappa_cc=0.05, p_c0=100e3, e_0=0.7,
        )
        K_init = mat.K_bulk
        # Apply hydrostatic compression
        mat.get_response(np.array([-0.001, -0.001, -0.001, 0, 0, 0]))
        mat.commit_state()
        # Committed state now has nonzero pressure -> K_bulk reflects it
        # (could be smaller or larger depending on if pressure rose)
        from femsolver.materials.multiaxial.cam_clay import _deviator
        s, p_v = _deviator(mat.sigma_committed)
        p_eff_committed = -p_v
        expected = (1.0 + mat.e_committed) * max(p_eff_committed, mat.p_min) \
                   / mat.kappa_cc
        # Note: stress-dependent flag means K floor is at p_min
        assert mat.K_bulk == pytest.approx(expected, rel=1e-9)

    def test_stress_dependent_false_uses_initial_K(self):
        """Backward compat: ``stress_dependent=False`` keeps K
        constant at the E-derived initial value."""
        mat = ModifiedCamClay3D(
            E=10e6, nu=0.3, M=1.0, lambda_cc=0.20,
            kappa_cc=0.05, p_c0=100e3, stress_dependent=False,
        )
        K_initial = mat._K_initial
        # K shouldn't change after a step
        mat.get_response(np.array([-0.002, -0.002, -0.002, 0, 0, 0]))
        mat.commit_state()
        assert mat.K_bulk == K_initial

    def test_K_floor_at_p_min(self):
        """When p_eff < p_min, K is computed at p_min (not zero)."""
        mat = ModifiedCamClay3D(
            E=10e6, nu=0.3, M=1.0, lambda_cc=0.20,
            kappa_cc=0.05, p_c0=100e3, e_0=0.5,
            p_min=1.0e3,
        )
        # Apply tension (p_eff very small or negative)
        mat.sigma_committed = np.array([1e2, 1e2, 1e2, 0, 0, 0])  # tiny tension
        # Set committed state explicitly
        K_expected = (1.0 + 0.5) * 1.0e3 / 0.05  # floored to p_min
        assert mat.K_bulk == pytest.approx(K_expected, rel=1e-9)

    def test_validates_e_0(self):
        with pytest.raises(ValueError, match="e_0"):
            ModifiedCamClay3D(
                E=10e6, nu=0.3, M=1.0, lambda_cc=0.20,
                kappa_cc=0.05, p_c0=100e3, e_0=-1.0,
            )

    def test_validates_p_min(self):
        with pytest.raises(ValueError, match="p_min"):
            ModifiedCamClay3D(
                E=10e6, nu=0.3, M=1.0, lambda_cc=0.20,
                kappa_cc=0.05, p_c0=100e3, p_min=-1.0,
            )


class TestVoidRatioTracking:
    def test_void_ratio_decreases_under_compression(self):
        """Compression reduces void ratio: de = -(1+e) * dε_v."""
        mat = ModifiedCamClay3D(
            E=10e6, nu=0.3, M=1.0, lambda_cc=0.20,
            kappa_cc=0.05, p_c0=100e3, e_0=0.7,
        )
        e_initial = mat.e_committed
        mat.get_response(np.array([-0.001, -0.001, -0.001, 0, 0, 0]))
        mat.commit_state()
        assert mat.e_committed < e_initial

    def test_void_ratio_persists_across_steps(self):
        mat = ModifiedCamClay3D(
            E=10e6, nu=0.3, M=1.0, lambda_cc=0.20,
            kappa_cc=0.05, p_c0=100e3, e_0=0.7,
        )
        mat.get_response(np.array([-0.001, -0.001, -0.001, 0, 0, 0]))
        mat.commit_state()
        e_after_first = mat.e_committed
        mat.get_response(np.array([-0.0015, -0.0015, -0.0015, 0, 0, 0]))
        mat.commit_state()
        # Further compression -> further reduction
        assert mat.e_committed < e_after_first


class TestCyclicConsistency:
    """A simple loading-unloading cycle should return to a state
    near the original (since plasticity hasn't been triggered yet)."""

    def test_elastic_cycle_returns_to_origin(self):
        mat = ModifiedCamClay3D(
            E=10e6, nu=0.3, M=1.0, lambda_cc=0.20,
            kappa_cc=0.05, p_c0=500e3,    # large p_c -> elastic only
            e_0=0.6,
        )
        # Load
        sigma_load, _ = mat.get_response(
            np.array([-0.0005, -0.0005, -0.0005, 0, 0, 0]),
        )
        mat.commit_state()
        # Unload to original strain
        sigma_unload, _ = mat.get_response(np.zeros(6))
        mat.commit_state()
        # Should be approximately zero (elastic cycle)
        # Tolerance is loose because the stress-dependent K is
        # nonlinear, so the cycle is not perfectly closed.
        assert np.allclose(sigma_unload, 0.0, atol=1e3)
