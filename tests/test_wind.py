"""Phase 52 tests -- Theme S wind engineering.

Covers all four wind modules:

* ASCE 7-22 -- velocity pressure, pressure coefficients, gust factor, MWFRS
* IS 875 Part 3 (2015) -- design wind pressure with k_2 from Table 2
* EN 1991-1-4 (EC1) -- peak velocity pressure
* Vortex shedding -- Strouhal, Scruton, lock-in heuristic
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from femsolver.hazard.wind import (
    asce7_Kz,
    asce7_exposure_constants,
    asce7_gust_factor_flexible,
    asce7_gust_factor_rigid,
    asce7_mwfrs_design_pressures,
    asce7_roof_Cp_flat,
    asce7_velocity_pressure,
    asce7_wall_Cp,
    ec1_peak_velocity_pressure,
    ec1_roughness_factor,
    is875_design_wind_pressure,
    is875_terrain_category_factor,
    is_lock_in_risk,
    scruton_number,
    vortex_shedding_frequency,
)


# ============================================================ ASCE 7

class TestASCE7Exposure:
    def test_exposure_C_constants(self):
        alpha, z_g = asce7_exposure_constants("C")
        assert alpha == pytest.approx(9.5)
        assert z_g == pytest.approx(274.32, rel=1e-3)

    def test_unknown_exposure_rejected(self):
        with pytest.raises(ValueError, match="exposure"):
            asce7_exposure_constants("X")

    def test_Kz_at_15ft_floor_is_constant_below(self):
        # Below 15ft = 4.572 m, K_z is floored
        K_below = asce7_Kz(3.0, "C")
        K_at_floor = asce7_Kz(4.572, "C")
        assert K_below == pytest.approx(K_at_floor, rel=1e-9)

    def test_Kz_increases_with_height(self):
        # K_z should grow monotonically with z (within zg)
        Kz_10 = asce7_Kz(10.0, "C")
        Kz_30 = asce7_Kz(30.0, "C")
        Kz_100 = asce7_Kz(100.0, "C")
        assert Kz_10 < Kz_30 < Kz_100

    def test_Kz_exposure_ordering(self):
        # At a given z above gradient, K_z(B) < K_z(C) < K_z(D)?
        # Below gradient at a typical 30 m height, exposure D is most
        # exposed (open water) -> highest Kz.
        Kz_B = asce7_Kz(30.0, "B")
        Kz_C = asce7_Kz(30.0, "C")
        Kz_D = asce7_Kz(30.0, "D")
        assert Kz_B < Kz_C < Kz_D


class TestASCE7VelocityPressure:
    def test_formula_matches_hand_calc(self):
        # V = 50 m/s, exposure C, z = 30 m -> K_z = 1.512 (from helper)
        # q_z = 0.613 * K_z * K_zt * K_d * K_e * V^2
        #     = 0.613 * 1.512 * 1.0 * 0.85 * 1.0 * 2500 = 1970 Pa
        vp = asce7_velocity_pressure(z=30.0, V=50.0, exposure="C")
        K_z = asce7_Kz(30.0, "C")
        expected = 0.613 * K_z * 1.0 * 0.85 * 1.0 * 50.0 ** 2
        assert vp.q_z == pytest.approx(expected, rel=1e-9)

    def test_rejects_negative_inputs(self):
        with pytest.raises(ValueError):
            asce7_velocity_pressure(z=30, V=-1)
        with pytest.raises(ValueError):
            asce7_velocity_pressure(z=30, V=50, K_d=-1)


class TestASCE7PressureCoefficients:
    def test_wall_Cp_at_LB1(self):
        c = asce7_wall_Cp(1.0)
        assert c.windward == 0.8
        assert c.leeward == -0.5
        assert c.side == -0.7

    def test_wall_Cp_at_LB4_or_greater(self):
        c = asce7_wall_Cp(4.0)
        assert c.leeward == -0.2
        c5 = asce7_wall_Cp(5.0)
        assert c5.leeward == -0.2

    def test_wall_Cp_interpolates(self):
        # L/B = 1.5 -> leeward = -0.5 + 0.2*0.5 = -0.4
        c = asce7_wall_Cp(1.5)
        assert c.leeward == pytest.approx(-0.4, abs=1e-12)
        # L/B = 3 -> -0.3 + 0.1*(3-2)/2 = -0.25
        c = asce7_wall_Cp(3.0)
        assert c.leeward == pytest.approx(-0.25, abs=1e-12)

    def test_roof_Cp_flat_steep_aspect(self):
        cp_near, cp_far = asce7_roof_Cp_flat(1.0)
        assert cp_near == pytest.approx(-1.3)
        assert cp_far == pytest.approx(-0.7)

    def test_roof_Cp_rejects_zero_aspect(self):
        with pytest.raises(ValueError):
            asce7_roof_Cp_flat(0)


class TestASCE7GustFactor:
    def test_rigid_is_0_85(self):
        assert asce7_gust_factor_rigid() == 0.85

    def test_flexible_returns_finite_positive(self):
        G = asce7_gust_factor_flexible(
            f1=0.5, zeta=0.02, h=100.0, B=30.0, L=30.0,
            V_bar_z=40.0, exposure="C",
        )
        assert math.isfinite(G)
        assert 0.5 < G < 2.0    # engineering range

    def test_flexible_rejects_invalid(self):
        with pytest.raises(ValueError):
            asce7_gust_factor_flexible(
                f1=-1, zeta=0.02, h=100, B=30, L=30, V_bar_z=40,
            )
        with pytest.raises(ValueError):
            asce7_gust_factor_flexible(
                f1=1, zeta=1.5, h=100, B=30, L=30, V_bar_z=40,
            )


class TestASCE7MWFRS:
    def test_windward_positive_leeward_negative(self):
        res = asce7_mwfrs_design_pressures(
            z=30.0, h=30.0, V=50.0, L=60.0, B=30.0,
        )
        # Windward (positive Cp = 0.8) should produce inward pressure
        # AFTER subtracting internal -- normally positive net.
        assert res.p_windward > 0
        # Leeward and side are suctions
        assert res.p_leeward < 0
        assert res.p_side < 0
        # Roof is suction on a flat low roof
        assert res.p_roof_near < 0

    def test_velocity_pressures_recorded(self):
        res = asce7_mwfrs_design_pressures(
            z=15.0, h=30.0, V=50.0, L=60.0, B=30.0,
        )
        # q_z (at z=15) should be smaller than q_h (at h=30)
        assert res.q_z < res.q_h


# ============================================================ IS 875

class TestIS875:
    def test_k2_table_category_2_at_30m(self):
        # From Table 2: terrain category 2 at z=30m -> k_2 = 1.12
        k2 = is875_terrain_category_factor(z=30.0, category=2)
        assert k2 == pytest.approx(1.12, rel=1e-9)

    def test_k2_interpolates(self):
        # Between z=20 (k2=1.07) and z=30 (k2=1.12), z=25 -> k2 ≈ 1.095
        k2 = is875_terrain_category_factor(z=25.0, category=2)
        assert k2 == pytest.approx(1.095, abs=1e-3)

    def test_k2_at_very_low_z(self):
        # Below the first tabulated value, return that value
        k2 = is875_terrain_category_factor(z=5.0, category=2)
        assert k2 == pytest.approx(1.00, rel=1e-9)

    def test_k2_rejects_invalid_category(self):
        with pytest.raises(ValueError, match="category"):
            is875_terrain_category_factor(z=10, category=5)

    def test_design_pressure_formula(self):
        # V_z = V_b * k_1 * k_2 * k_3 * k_4 = 50 * 1.0 * 1.12 * 1 * 1 = 56
        # p_z = 0.6 * 56^2 = 1881.6 Pa
        res = is875_design_wind_pressure(z=30.0, V_b=50.0, category=2)
        assert res.V_z == pytest.approx(56.0, rel=1e-9)
        assert res.p_z == pytest.approx(0.6 * 56.0 ** 2, rel=1e-9)

    def test_rejects_negative_inputs(self):
        with pytest.raises(ValueError):
            is875_design_wind_pressure(z=10, V_b=-1)
        with pytest.raises(ValueError):
            is875_design_wind_pressure(z=10, V_b=50, k_1=0)


# ============================================================ EC1

class TestEC1:
    def test_roughness_factor_terrain_II_at_10m(self):
        # c_r(z=10, terrain II) = k_r * ln(10 / 0.05) where k_r = 0.19
        # = 0.19 * ln(200) = 0.19 * 5.298 = 1.007
        c_r = ec1_roughness_factor(z=10.0, terrain="II")
        assert c_r == pytest.approx(1.0066, abs=1e-3)

    def test_terrain_IV_below_zmin_uses_floor(self):
        # Terrain IV has z_min = 10. At z=5, should compute as if z=10.
        c_r_5 = ec1_roughness_factor(z=5.0, terrain="IV")
        c_r_10 = ec1_roughness_factor(z=10.0, terrain="IV")
        assert c_r_5 == pytest.approx(c_r_10, rel=1e-9)

    def test_peak_velocity_pressure_terrain_II(self):
        res = ec1_peak_velocity_pressure(z=30.0, v_b=27.0, terrain="II")
        # v_m = c_r * c_o * v_b; expected c_r ~ 1.21 at 30m
        # q_p = (1 + 7*I_v) * 0.5 * 1.25 * v_m^2
        assert 1000.0 < res.q_p < 2000.0
        assert res.v_m > 27.0       # boosted by roughness

    def test_unknown_terrain_rejected(self):
        with pytest.raises(ValueError, match="terrain"):
            ec1_roughness_factor(z=10.0, terrain="V")


# ============================================================ vortex

class TestVortexShedding:
    def test_strouhal_frequency_formula(self):
        # f_s = St * U / D
        res = vortex_shedding_frequency(U=10.0, D=2.0, St=0.20)
        assert res.f_s == pytest.approx(1.0)

    def test_default_St_circular_cylinder(self):
        res = vortex_shedding_frequency(U=10.0, D=2.0)
        assert res.St == 0.20

    def test_scruton_number_formula(self):
        # Sc = 2 m_e zeta / (rho D^2)
        # For m_e=300, zeta=0.005, D=3, rho=1.25: Sc = 2*300*0.005/(1.25*9) = 0.267
        Sc = scruton_number(m_e=300.0, zeta=0.005, D=3.0)
        assert Sc == pytest.approx(2 * 300 * 0.005 / (1.25 * 9.0), rel=1e-9)

    def test_lock_in_when_close_and_low_Sc(self):
        # f_s ≈ f_n (within 20%) and Sc small -> lock-in
        assert is_lock_in_risk(f_s=1.05, f_n=1.0, Sc=5.0)

    def test_no_lock_in_when_high_Sc(self):
        # f_s ≈ f_n but Sc above threshold -> no lock-in
        assert not is_lock_in_risk(f_s=1.05, f_n=1.0, Sc=50.0)

    def test_no_lock_in_when_frequencies_far_apart(self):
        # Far outside bandwidth -> no lock-in
        assert not is_lock_in_risk(f_s=3.0, f_n=1.0, Sc=5.0)

    def test_vortex_rejects_invalid(self):
        with pytest.raises(ValueError):
            vortex_shedding_frequency(U=-1, D=1)
        with pytest.raises(ValueError):
            scruton_number(m_e=0, zeta=0.01, D=1)
