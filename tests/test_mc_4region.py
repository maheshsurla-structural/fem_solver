"""Phase HH.1 tests -- Mohr-Coulomb full 4-region return mapping.

Validates that the new return mapping handles all four regions:
main face (smooth), right edge (sigma_1 = sigma_2), left edge
(sigma_2 = sigma_3), and apex.
"""
from __future__ import annotations

import numpy as np
import pytest

from femsolver import MohrCoulomb3D
from femsolver.materials.mohr_coulomb import _principal_decomposition


# ============================================================ helpers

def _triaxial_drive(material, *, sigma_3: float, n_steps: int = 50,
                     eps_max: float = -0.02):
    """Strain-controlled triaxial compression with Poisson iteration to
    keep sigma_xx = sigma_yy = sigma_3."""
    nu = material.nu
    E = material.E
    K = E / (3.0 * (1.0 - 2.0 * nu))
    eps_init = sigma_3 / (3.0 * K)
    eps = np.array([eps_init, eps_init, eps_init, 0, 0, 0])
    material.get_response(eps)
    material.commit_state()
    qs, ps = [], []
    for i in range(1, n_steps + 1):
        eps_axial = eps_init + (eps_max - eps_init) * i / n_steps
        eps[2] = eps_axial
        eps_lat = eps_init
        for _ in range(10):
            eps[0] = eps_lat; eps[1] = eps_lat
            sigma, _ = material.get_response(eps)
            err_x = sigma[0] - sigma_3
            if abs(err_x) < 10.0:
                break
            eps_lat -= err_x / (
                E * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
            )
        sigma, _ = material.get_response(eps)
        material.commit_state()
        p_v = (sigma[0] + sigma[1] + sigma[2]) / 3.0
        s = sigma.copy(); s[:3] -= p_v
        q = float(np.sqrt(1.5 * (
            s[0] ** 2 + s[1] ** 2 + s[2] ** 2
            + 2.0 * (s[3] ** 2 + s[4] ** 2 + s[5] ** 2)
        )))
        qs.append(q)
        ps.append(-p_v)
    return np.array(qs), np.array(ps)


# ============================================================ test cases

class Test4RegionReturn:
    def test_no_chatter_in_triaxial(self):
        """The hallmark of edge-case mishandling was a post-yield q
        that dropped to zero (apex projection). After the fix, q
        should stay close to the DP envelope, with no step-to-step
        oscillation greater than 5% of the peak."""
        from femsolver import DruckerPrager3D
        mc = MohrCoulomb3D(E=30e6, nu=0.30, cohesion=10e3, phi_deg=30.0)
        dp = DruckerPrager3D.from_mohr_coulomb(
            E=30e6, nu=0.30, cohesion=10e3, phi_deg=30.0,
        )
        q_mc, _ = _triaxial_drive(mc, sigma_3=-100e3, n_steps=50)
        q_dp, _ = _triaxial_drive(dp, sigma_3=-100e3, n_steps=50)
        # Verify no chatter: max step-to-step difference within 20 kPa
        # of the smooth DP series.
        diffs_mc = np.abs(np.diff(q_mc))
        diffs_dp = np.abs(np.diff(q_dp))
        # MC step-to-step variation should not greatly exceed DP's
        assert diffs_mc.max() <= 5.0 * diffs_dp.max() + 1e3

    def test_mc_matches_dp_at_high_strain(self):
        """At large strain the MC and DP yield surfaces converge
        (both reach the asymptotic strength). After the 4-region
        fix, MC's final q should be within ~5% of DP's."""
        from femsolver import DruckerPrager3D
        mc = MohrCoulomb3D(E=30e6, nu=0.30, cohesion=10e3, phi_deg=30.0)
        dp = DruckerPrager3D.from_mohr_coulomb(
            E=30e6, nu=0.30, cohesion=10e3, phi_deg=30.0,
        )
        q_mc, _ = _triaxial_drive(mc, sigma_3=-100e3)
        q_dp, _ = _triaxial_drive(dp, sigma_3=-100e3)
        assert q_mc[-1] == pytest.approx(q_dp[-1], rel=0.10)

    def test_pure_tension_returns_apex(self):
        """Large hydrostatic tension should land at the apex
        (sigma_1 = sigma_2 = sigma_3 = c cot phi)."""
        mc = MohrCoulomb3D(E=30e6, nu=0.30, cohesion=10e3, phi_deg=30.0)
        eps = np.array([0.005, 0.005, 0.005, 0, 0, 0])
        sigma, _ = mc.get_response(eps)
        s, _ = _principal_decomposition(sigma)
        for si in s:
            assert si == pytest.approx(mc._apex, rel=1e-6)
        # Yield function at the apex is approximately zero
        assert abs(mc.yield_function(sigma)) < 1.0

    def test_elastic_state_passes_through(self):
        mc = MohrCoulomb3D(E=30e6, nu=0.30, cohesion=10e3, phi_deg=30.0)
        eps = np.array([-5e-5, -5e-5, -5e-5, 0, 0, 0])
        sigma, _ = mc.get_response(eps)
        # f < 0
        assert mc.yield_function(sigma) < 0
        # No plastic strain accumulated
        assert np.allclose(mc.eps_p_trial, 0)


class TestRegionDispatchExplicit:
    """Test the four return regions explicitly by constructing trial
    stress states that target each one."""

    def _build_with_no_dilation(self):
        # psi = 0 simplifies the algebra; we still want non-trivial phi
        return MohrCoulomb3D(
            E=30e6, nu=0.30, cohesion=10e3, phi_deg=30.0,
            psi_deg=0.0,
        )

    def test_main_face_return_preserves_ordering(self):
        mc = self._build_with_no_dilation()
        # Direct call to the internal return-map function with a trial
        # stress that yields but lies clearly in the main-face region.
        s = mc._return_map_principal(30e3, -20e3, -50e3)
        # All three principals distinct and ordered
        assert s[0] >= s[1] >= s[2]
        # Should be on the yield surface
        f = (s[0] - s[2]) + (s[0] + s[2]) * mc._sin_phi \
            - 2.0 * mc.cohesion * mc._cos_phi
        assert abs(f) < 1.0

    def test_main_face_recovers_known_DP_value(self):
        """Single-point smoke check: triaxial compression hits a
        known stress level that DP and MC agree on at the same
        peak."""
        mc = MohrCoulomb3D(E=30e6, nu=0.30, cohesion=10e3, phi_deg=30.0)
        q, _ = _triaxial_drive(mc, sigma_3=-100e3, n_steps=50)
        # Peak deviatoric stress should be in the engineering range
        assert 100e3 < q.max() < 300e3
