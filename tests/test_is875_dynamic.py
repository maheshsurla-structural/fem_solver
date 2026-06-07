"""Phase HH.7 tests -- IS 875 Part 3 dynamic response factor."""
from __future__ import annotations

import math

import pytest

from femsolver.hazard.wind import (
    Is875DynamicFactor,
    background_factor,
    gust_energy_factor,
    integral_length_scale,
    is875_dynamic_factor,
    size_reduction_factor,
    turbulence_intensity,
)


class TestTurbulenceIntensity:
    def test_at_10m_matches_table(self):
        for cat, exp in [(1, 0.155), (2, 0.180), (3, 0.230), (4, 0.270)]:
            I = turbulence_intensity(10.0, cat)
            assert I == pytest.approx(exp, rel=1e-9)

    def test_decreases_with_height(self):
        I_10 = turbulence_intensity(10.0, 2)
        I_100 = turbulence_intensity(100.0, 2)
        assert I_100 < I_10

    def test_urban_more_turbulent_than_open(self):
        assert turbulence_intensity(50.0, 4) > turbulence_intensity(50.0, 1)

    def test_floor_at_10m(self):
        assert (turbulence_intensity(3.0, 2)
                == pytest.approx(turbulence_intensity(10.0, 2), rel=1e-12))

    def test_rejects_invalid(self):
        with pytest.raises(ValueError):
            turbulence_intensity(-1, 2)
        with pytest.raises(ValueError):
            turbulence_intensity(10, 5)


class TestLengthScale:
    def test_at_10m_is_85m(self):
        assert integral_length_scale(10.0) == pytest.approx(85.0, rel=1e-9)

    def test_grows_with_height(self):
        L_10 = integral_length_scale(10.0)
        L_100 = integral_length_scale(100.0)
        # L_h(100) = 85 * (100/10)^0.25 = 85 * 1.778 = 151.2 m
        assert L_100 > L_10
        assert L_100 == pytest.approx(151.2, rel=1e-2)


class TestBackgroundFactor:
    def test_small_building_high_B(self):
        # Small building (compared to L_h) -> B_s near 1
        B = background_factor(h=10.0, b=10.0, L_h=200.0)
        assert B > 0.5

    def test_large_building_low_B(self):
        # Large building -> B_s small
        B = background_factor(h=200.0, b=100.0, L_h=100.0)
        assert B < 0.5

    def test_rejects_invalid(self):
        with pytest.raises(ValueError):
            background_factor(h=-1, b=10, L_h=100)


class TestSizeReduction:
    def test_formula(self):
        # f_a=0.5, h=50, b=20, V=30: denom_h = 1 + 3.5*0.5*50/30 = 1+2.917 = 3.917
        #                            denom_b = 1 + 4*0.5*20/30 = 1+1.333 = 2.333
        # S = 1 / (3.917 * 2.333) = 0.1094
        S = size_reduction_factor(f_a=0.5, h=50.0, b=20.0, V_h_bar=30.0)
        expected = 1.0 / (3.917 * 2.333)
        assert S == pytest.approx(expected, rel=0.01)

    def test_decreases_with_frequency(self):
        S_low = size_reduction_factor(f_a=0.2, h=50.0, b=20.0, V_h_bar=30.0)
        S_high = size_reduction_factor(f_a=2.0, h=50.0, b=20.0, V_h_bar=30.0)
        # Higher frequency -> more size reduction (smaller S)
        assert S_high < S_low


class TestGustEnergyFactor:
    def test_peaks_at_intermediate_N(self):
        # E(N) = pi*N / (1+70.8*N^2)^(5/6) peaks at N ~ 0.145
        # (where d/dN(num/denom) = 0 ; numerator -> 0 at 70.8 N^2 + 1 = 118 N^2 ~ N=0.146)
        # Choose scenarios bracketing this peak:
        E_low = gust_energy_factor(f_a=0.001, L_h=150, V_h_bar=30)  # N=0.005 (tiny)
        E_peak = gust_energy_factor(f_a=0.03, L_h=150, V_h_bar=30)   # N=0.15 (peak)
        E_high = gust_energy_factor(f_a=5.0, L_h=150, V_h_bar=30)    # N=25 (huge)
        # E(N->0) ~ pi*N -> 0; E(N->inf) ~ 1/N^(2/3) -> 0
        assert E_peak > E_low
        assert E_peak > E_high


class TestDynamicFactor:
    def test_rigid_building_gives_low_amplification(self):
        """For very stiff building (f_a >> 1), resonance is negligible
        and only background contributes."""
        r_stiff = is875_dynamic_factor(
            f_a=20.0, h=100.0, b=30.0, V_h_bar=50.0, beta=0.02,
        )
        r_flex = is875_dynamic_factor(
            f_a=0.2, h=100.0, b=30.0, V_h_bar=50.0, beta=0.02,
        )
        assert r_stiff.C_dyn < r_flex.C_dyn

    def test_lower_damping_increases_C_dyn(self):
        """Lower damping -> larger resonant amplification."""
        r_low_zeta = is875_dynamic_factor(
            f_a=0.4, h=100.0, b=30.0, V_h_bar=50.0, beta=0.01,
        )
        r_high_zeta = is875_dynamic_factor(
            f_a=0.4, h=100.0, b=30.0, V_h_bar=50.0, beta=0.05,
        )
        assert r_low_zeta.C_dyn > r_high_zeta.C_dyn

    def test_tall_building_engineering_range(self):
        """A 100-m tall building with f=0.4 Hz, beta=2%, category 2
        should produce C_dyn in [1.2, 1.8]."""
        r = is875_dynamic_factor(
            f_a=0.4, h=100.0, b=30.0, V_h_bar=50.0, beta=0.02,
        )
        assert 1.2 < r.C_dyn < 1.8

    def test_rejects_invalid(self):
        with pytest.raises(ValueError):
            is875_dynamic_factor(f_a=-1, h=100, b=30, V_h_bar=50, beta=0.02)
        with pytest.raises(ValueError):
            is875_dynamic_factor(f_a=0.4, h=100, b=30, V_h_bar=50, beta=1.0)

    def test_returns_all_components(self):
        r = is875_dynamic_factor(
            f_a=0.4, h=100.0, b=30.0, V_h_bar=50.0, beta=0.02,
        )
        assert r.I_h > 0
        assert r.L_h > 0
        assert r.B_s > 0
        assert r.S > 0
        assert r.E > 0
        assert r.N > 0
        assert r.g_R > 0
