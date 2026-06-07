"""Phase HH.3 tests -- Lubliner-Lee-Fenves concrete damage-plasticity."""
from __future__ import annotations

import math

import numpy as np
import pytest

from femsolver import ConcreteDamagePlasticity3D


class TestCalibration:
    def test_alpha_from_f_b_ratio(self):
        """Lubliner alpha = (r-1)/(2r-1), r = f_b/f_c."""
        mat = ConcreteDamagePlasticity3D(
            E=30e9, nu=0.20, f_c=30e6, f_t=3e6,
            f_b_over_f_c=1.16,
        )
        # (1.16 - 1) / (2*1.16 - 1) = 0.16 / 1.32 = 0.1212
        assert mat.alpha == pytest.approx(0.1212, rel=1e-3)

    def test_k_0_calibrates_to_uniaxial_compression(self):
        """The yield surface should fire exactly at sigma = -f_c
        under uniaxial compression."""
        mat = ConcreteDamagePlasticity3D(
            E=30e9, nu=0.20, f_c=30e6, f_t=3e6,
        )
        # In uniaxial compression sigma = [-f_c, 0, 0, 0, 0, 0]:
        # f = alpha*(-f_c) + sqrt(J_2) - k_0 = 0
        # sqrt(J_2) = f_c/sqrt(3)
        # => k_0 = -alpha*f_c + f_c/sqrt(3) = f_c (1/sqrt(3) - alpha)
        expected = 30e6 * (1.0 / math.sqrt(3.0) - mat.alpha)
        assert mat.k_0 == pytest.approx(expected, rel=1e-9)


class TestPlasticityAndDamage:
    def test_elastic_below_thresholds(self):
        """Below eps_t0 in tension, no damage; sigma = E * eps."""
        mat = ConcreteDamagePlasticity3D(
            E=30e9, nu=0.20, f_c=30e6, f_t=3e6,
        )
        eps = np.array([5e-5, -1e-5, -1e-5, 0, 0, 0])
        sigma, _ = mat.get_response(eps)
        assert mat.d_t_trial == 0.0
        assert mat.d_c_trial == 0.0
        # sigma_xx ~ E * 5e-5 = 1.5 MPa (with Poisson coupling)
        assert sigma[0] == pytest.approx(1.5e6, rel=0.05)

    def test_tension_peak_at_f_t(self):
        """Tension peak should be at f_t = E * eps_t0."""
        mat = ConcreteDamagePlasticity3D(
            E=30e9, nu=0.20, f_c=30e6, f_t=3e6,
        )
        # Apply eps at exactly eps_t0
        eps_ax = mat.eps_t0
        eps = np.array([eps_ax, -0.2*eps_ax, -0.2*eps_ax, 0, 0, 0])
        sigma, _ = mat.get_response(eps)
        assert sigma[0] == pytest.approx(3.0e6, rel=0.01)
        # Just past eps_t0, damage starts
        eps_ax2 = eps_ax * 1.5
        eps2 = np.array([eps_ax2, -0.2*eps_ax2, -0.2*eps_ax2, 0, 0, 0])
        sigma2, _ = mat.get_response(eps2)
        assert mat.d_t_trial > 0.0
        # Stress is reduced below the elastic projection
        assert sigma2[0] < 30e9 * eps_ax2 * 0.95

    def test_compression_yields_at_f_c(self):
        """Uniaxial compression peaks near -f_c under Poisson coupling."""
        mat = ConcreteDamagePlasticity3D(
            E=30e9, nu=0.20, f_c=30e6, f_t=3e6,
        )
        eps_ax = -mat.eps_c0
        eps = np.array([eps_ax, -0.2*eps_ax, -0.2*eps_ax, 0, 0, 0])
        sigma, _ = mat.get_response(eps)
        # Yield should be hit around here
        assert abs(sigma[0]) >= 29e6

    def test_plastic_strain_accumulates_in_compression(self):
        """After yielding under compression, eps_p has a non-trivial
        deviatoric and (with non-zero dilation) volumetric part."""
        mat = ConcreteDamagePlasticity3D(
            E=30e9, nu=0.20, f_c=30e6, f_t=3e6, psi_deg=30.0,
        )
        # Push well into the plastic regime
        eps_ax = -3e-3
        eps = np.array([eps_ax, -0.2*eps_ax, -0.2*eps_ax, 0, 0, 0])
        sigma, _ = mat.get_response(eps)
        mat.commit_state()
        assert np.any(np.abs(mat.eps_p_committed) > 1e-6)
        assert mat.kappa_c_committed > 0.0

    def test_damage_monotonically_non_decreasing(self):
        """Damage variables never decrease on unloading."""
        mat = ConcreteDamagePlasticity3D(
            E=30e9, nu=0.20, f_c=30e6, f_t=3e6,
        )
        # Apply tension well past threshold
        eps_load = np.array([3e-4, -0.6e-4, -0.6e-4, 0, 0, 0])
        mat.get_response(eps_load)
        d_t_peak = mat.d_t_trial
        mat.commit_state()
        # Unload
        eps_unload = np.array([5e-5, -1e-5, -1e-5, 0, 0, 0])
        mat.get_response(eps_unload)
        assert mat.d_t_trial >= d_t_peak


class TestStiffnessRecovery:
    def test_pure_tension_factor(self):
        """In pure tension all principals positive -> s = 1."""
        mat = ConcreteDamagePlasticity3D(
            E=30e9, nu=0.20, f_c=30e6, f_t=3e6,
        )
        sigma_bar = np.array([10e6, 1e6, 1e6, 0, 0, 0])
        s = mat._stiffness_recovery(sigma_bar)
        assert s == pytest.approx(1.0, abs=1e-6)

    def test_pure_compression_factor(self):
        """In pure compression all principals negative -> s = 0."""
        mat = ConcreteDamagePlasticity3D(
            E=30e9, nu=0.20, f_c=30e6, f_t=3e6,
        )
        sigma_bar = np.array([-30e6, -5e6, -5e6, 0, 0, 0])
        s = mat._stiffness_recovery(sigma_bar)
        assert s == pytest.approx(0.0, abs=1e-6)

    def test_cyclic_tension_to_compression(self):
        """After tension cracking, reapplying compression should
        recover most of the compressive stiffness (crack closure).
        The apparent compressive stress should be close to the
        undamaged-elastic prediction, modulo any compression damage."""
        mat = ConcreteDamagePlasticity3D(
            E=30e9, nu=0.20, f_c=30e6, f_t=3e6,
        )
        # Crack in tension
        eps_t = np.array([3e-4, -0.6e-4, -0.6e-4, 0, 0, 0])
        mat.get_response(eps_t); mat.commit_state()
        assert mat.d_t_trial > 0.5
        # Now apply small compression -- below f_c so no compressive damage
        eps_c = np.array([-5e-4, 1.0e-4, 1.0e-4, 0, 0, 0])
        sigma, _ = mat.get_response(eps_c)
        # Effective E * eps_xx = 30e9 * -5e-4 = -15 MPa
        # With full stiffness recovery, apparent ~ -15 MPa
        # With NO recovery (d_t = 0.7 active), apparent ~ -4.5 MPa
        # Recovery factor s ~ 0 in compression -> use d_c only (= 0)
        # so sigma ~ -15 MPa
        assert sigma[0] == pytest.approx(-15e6, rel=0.1)


class TestInputValidation:
    def test_rejects_negative_E(self):
        with pytest.raises(ValueError):
            ConcreteDamagePlasticity3D(E=-1, nu=0.2, f_c=30e6, f_t=3e6)

    def test_rejects_invalid_psi(self):
        with pytest.raises(ValueError):
            ConcreteDamagePlasticity3D(
                E=30e9, nu=0.2, f_c=30e6, f_t=3e6, psi_deg=90.0,
            )

    def test_rejects_invalid_f_b_ratio(self):
        with pytest.raises(ValueError):
            ConcreteDamagePlasticity3D(
                E=30e9, nu=0.2, f_c=30e6, f_t=3e6, f_b_over_f_c=1.0,
            )

    def test_rejects_invalid_damage_params(self):
        with pytest.raises(ValueError):
            ConcreteDamagePlasticity3D(
                E=30e9, nu=0.2, f_c=30e6, f_t=3e6, A_t=2.0,
            )
        with pytest.raises(ValueError):
            ConcreteDamagePlasticity3D(
                E=30e9, nu=0.2, f_c=30e6, f_t=3e6, B_c=0,
            )

    def test_clone_is_independent(self):
        mat = ConcreteDamagePlasticity3D(
            E=30e9, nu=0.2, f_c=30e6, f_t=3e6,
        )
        c = mat.clone()
        mat.eps_p_committed = np.ones(6)
        assert np.allclose(c.eps_p_committed, 0)
