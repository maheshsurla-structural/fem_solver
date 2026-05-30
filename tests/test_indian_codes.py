"""Phase 36 tests -- Indian design codes (IS 456, 800, 1893, 13920).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from femsolver.design import is456, is800, is1893, is13920


# ============================================================ IS 456

class TestIS456BeamFlexure:
    def test_singly_under_reinforced(self):
        # SP-16 ex: M20, Fe415, 230x500 with d=460, M_u=100 kN.m
        res = is456.is456_beam_flexure(
            M_u=100.0e3,
            f_ck=is456.fck_M(20), f_y=is456.fy_Fe(415),
            b=0.230, d=0.460,
        )
        assert not res.is_doubly
        # SP-16 ~ 670 mm^2; we get ~700 from exact quadratic
        assert res.A_st * 1.0e6 == pytest.approx(700.0, rel=0.10)
        assert res.x_u_over_d < 0.48      # well under x_u,max/d for Fe415

    def test_x_u_max_correct_for_grades(self):
        assert is456.xu_max_over_d(250.0e6) == 0.53
        assert is456.xu_max_over_d(415.0e6) == 0.48
        assert is456.xu_max_over_d(500.0e6) == 0.46

    def test_doubly_reinforced_triggers(self):
        # Very high M_u for a small section -> doubly required
        res = is456.is456_beam_flexure(
            M_u=300.0e3,
            f_ck=is456.fck_M(20), f_y=is456.fy_Fe(415),
            b=0.230, d=0.460, d_prime=0.040,
        )
        assert res.is_doubly
        assert res.A_sc > 0.0

    def test_rejects_invalid(self):
        with pytest.raises(ValueError, match="M_u"):
            is456.is456_beam_flexure(M_u=-1.0, f_ck=20e6, f_y=415e6,
                                      b=0.2, d=0.4)

    def test_min_max_steel(self):
        A_min, A_max = is456.is456_min_max_tension_steel(
            f_y=is456.fy_Fe(415), b=0.230, d=0.460,
        )
        # 0.85/415 * 230 * 460 = 217 mm^2
        assert A_min * 1.0e6 == pytest.approx(217.0, rel=0.02)
        # 4% * 230 * 460 = 4232 mm^2
        assert A_max * 1.0e6 == pytest.approx(4232.0, rel=0.02)


class TestIS456Shear:
    def test_tau_c_table_lookup(self):
        # IS 456 Table 19: p_t=1.0%, f_ck=25 MPa -> tau_c = 0.64 MPa
        tc = is456.is456_tau_c(p_t_pct=1.0, f_ck=25e6)
        assert tc == pytest.approx(0.64e6, rel=0.02)

    def test_tau_c_max(self):
        # Table 20: M25 -> 3.1 MPa
        assert is456.is456_tau_c_max(f_ck=25e6) == pytest.approx(
            3.1e6, rel=0.02
        )

    def test_no_stirrups_when_tauv_below_tauc(self):
        # Small shear, large section
        res = is456.is456_beam_shear(
            V_u=20.0e3, f_ck=is456.fck_M(25),
            f_y_sv=is456.fy_Fe(415),
            b=0.230, d=0.460, A_st=1000e-6,
        )
        assert not res.requires_stirrups

    def test_redesign_required_above_tauc_max(self):
        # Very high shear
        res = is456.is456_beam_shear(
            V_u=500.0e3, f_ck=is456.fck_M(20),
            f_y_sv=is456.fy_Fe(415),
            b=0.230, d=0.460, A_st=500e-6,
        )
        assert math.isinf(res.V_us_required)


class TestIS456Column:
    def test_pm_curve_pure_compression_positive(self):
        pts = is456.is456_column_pm_curve(
            f_ck=is456.fck_M(20), f_y=is456.fy_Fe(415),
            b=0.300, D=0.500, A_st_total=1256e-6,
        )
        assert pts[0].P > 0.0
        assert pts[0].M == 0.0

    def test_pm_curve_balanced_point_exists(self):
        """Mid-range x_u should give a non-zero M."""
        pts = is456.is456_column_pm_curve(
            f_ck=is456.fck_M(20), f_y=is456.fy_Fe(415),
            b=0.300, D=0.500, A_st_total=1256e-6,
        )
        Ms = [p.M for p in pts]
        assert max(Ms) > 0.0

    def test_low_demand_passes(self):
        pts = is456.is456_column_pm_curve(
            f_ck=is456.fck_M(20), f_y=is456.fy_Fe(415),
            b=0.300, D=0.500, A_st_total=1256e-6,
        )
        passes, util = is456.is456_column_pm_check(
            P_u=500e3, M_u=10.0e3, points=pts,
        )
        assert passes
        assert util < 0.5


# ============================================================ IS 800

class TestIS800Tension:
    def test_basic(self):
        res = is800.is800_tension(
            T_u=300.0e3, A_g=1903e-6, f_y=250e6, f_u=410e6,
        )
        # T_dg = 1903e-6 * 250e6 / 1.10 = 432.5 kN
        assert res.T_d_gross == pytest.approx(432.5e3, rel=0.01)
        assert res.utilisation < 1.0
        assert res.governing in ("gross", "net")


class TestIS800Compression:
    def test_perry_robertson_at_lambda_1_curve_b(self):
        chi = is800.perry_robertson_chi(lambda_bar=1.0, curve="b")
        # Textbook: chi ~ 0.60
        assert chi == pytest.approx(0.60, abs=0.02)

    def test_perry_robertson_at_lambda_0_5(self):
        chi = is800.perry_robertson_chi(lambda_bar=0.5, curve="b")
        # Textbook: chi ~ 0.88
        assert chi == pytest.approx(0.88, abs=0.02)

    def test_low_slenderness_chi_one(self):
        chi = is800.perry_robertson_chi(lambda_bar=0.0, curve="b")
        assert chi == pytest.approx(1.0, abs=1.0e-12)

    def test_invalid_curve_raises(self):
        with pytest.raises(ValueError):
            is800.perry_robertson_chi(lambda_bar=0.5, curve="z")

    def test_compression_full_calc(self):
        res = is800.is800_compression(
            P_u=500.0e3, A_g=53.83e-4, f_y=250e6,
            r_min=0.0498, K_L=4.0, curve="b",
        )
        assert res.lambda_bar == pytest.approx(0.90, abs=0.02)
        assert 0.5 < res.chi < 0.7


class TestIS800Flexure:
    def test_no_LTB_returns_plastic_capacity(self):
        res = is800.is800_flexure(
            M_u=100.0e3, Z_p=651e-6, f_y=250e6,
        )
        # M_dp = 651e-6 * 250e6 / 1.10 = 147.95 kN.m
        assert res.M_d == pytest.approx(147.95e3, rel=0.01)
        assert res.chi_LT == 1.0

    def test_LTB_reduces_capacity(self):
        res = is800.is800_flexure(
            M_u=100.0e3, Z_p=651e-6, f_y=250e6,
            L_LT=3.0, M_cr=200.0e3,
        )
        assert res.chi_LT < 1.0
        assert res.M_d_LTB < res.M_d_plastic


class TestIS800Shear:
    def test_shear_formula(self):
        res = is800.is800_shear(V_u=200e3, A_v=2310e-6, f_yw=250e6)
        # V_d = 2310e-6 * 250e6 / (sqrt(3) * 1.10) = 303.1 kN
        assert res.V_d == pytest.approx(303.1e3, rel=0.01)


class TestIS800Combined:
    def test_linear_interaction(self):
        res = is800.is800_combined_pm(
            P_u=400e3, M_u_z=80e3, M_u_y=0.0,
            P_d=805e3, M_d_z=148e3, M_d_y=1e30,
        )
        # P_r = 0.497, Mz_r = 0.541, total = 1.038 -> fails
        assert res.total > 1.0
        assert not res.passes


# ============================================================ IS 1893

class TestIS1893:
    def test_zone_factor_table(self):
        assert is1893.zone_factor(2) == 0.10
        assert is1893.zone_factor(3) == 0.16
        assert is1893.zone_factor(4) == 0.24
        assert is1893.zone_factor(5) == 0.36

    def test_zone_factor_invalid(self):
        with pytest.raises(ValueError):
            is1893.zone_factor(1)

    def test_spectrum_plateau(self):
        # T in plateau region for soil 1 -> Sa/g = 2.50
        assert is1893.design_spectrum_Sa_g(0.25, soil_type=1) == \
            pytest.approx(2.50, abs=1e-9)

    def test_spectrum_short_period(self):
        # T < 0.10: Sa/g = 1 + 15T
        assert is1893.design_spectrum_Sa_g(0.05, soil_type=1) == \
            pytest.approx(1.75, abs=1e-9)

    def test_spectrum_long_period_soil2(self):
        # Soil 2: T > 0.55 -> Sa/g = 1.36 / T
        assert is1893.design_spectrum_Sa_g(1.0, soil_type=2) == \
            pytest.approx(1.36, abs=1e-9)

    def test_empirical_period_rc_mrf(self):
        # 0.075 * 15^0.75
        T = is1893.empirical_period(h=15.0, system="RC_MRF")
        assert T == pytest.approx(0.075 * 15.0 ** 0.75, rel=1.0e-12)

    def test_base_shear(self):
        T = 0.5
        res = is1893.is1893_base_shear(
            T=T, W=1000.0e3, zone=4, importance=1.0, R=5.0,
            soil_type=1,
        )
        # Z=0.24, I=1, R=5, Sa/g(0.5) for soil 1 = 1/0.5 = 2.0
        # A_h = 0.24/2 * 1/5 * 2.0 = 0.048
        assert res.A_h == pytest.approx(0.048, abs=1e-9)
        assert res.V_B == pytest.approx(0.048 * 1000e3, rel=1e-9)

    def test_vertical_distribution_sums(self):
        Q = is1893.vertical_force_distribution(
            V_B=100.0e3,
            storey_weights=np.array([200e3]*3),
            storey_heights=np.array([3.0, 6.0, 9.0]),
        )
        assert Q.sum() == pytest.approx(100.0e3, rel=1.0e-12)

    def test_drift_check_R_amplification(self):
        u = np.array([0.001, 0.002, 0.003])
        h = np.array([3.0, 3.0, 3.0])
        # Elastic drifts: 0.001, 0.001, 0.001 -> ratios = 0.000333
        # With R=5: 5*0.001 = 0.005, ratio 5*0.000333 = 0.00167 < 0.004
        res = is1893.is1893_drift_check(
            floor_disp=u, storey_heights=h, R=5.0,
        )
        assert res.passes


# ============================================================ IS 13920

class TestIS13920:
    def test_scwb_pass(self):
        res = is13920.is13920_scwb_check(sum_Mc=600e3, sum_Mb=400e3)
        assert res.passes
        assert res.ratio == pytest.approx(1.5, rel=1e-12)

    def test_scwb_fail(self):
        res = is13920.is13920_scwb_check(sum_Mc=300e3, sum_Mb=400e3)
        assert not res.passes
        assert res.ratio == 0.75

    def test_capacity_shear_beam(self):
        res = is13920.is13920_capacity_shear_beam(
            M_n_pos_left=80e3, M_n_neg_left=120e3,
            M_n_pos_right=80e3, M_n_neg_right=120e3,
            L_n=5.0, V_gravity=50e3,
        )
        # 1.25*(80+120)/5 = 50 kN
        assert res.V_p == pytest.approx(50.0e3, rel=1e-12)
        assert res.V_design == 50.0e3

    def test_capacity_shear_column(self):
        res = is13920.is13920_capacity_shear_column(
            M_n_top=100e3, M_n_bot=100e3,
            h_clear=3.0, V_analysis=50e3,
        )
        # 1.4 * (100+100) / 3 = 93.33 kN; max with 50 -> 93.33
        assert res.V_p == pytest.approx(93.33e3, rel=0.01)
        assert res.V_design == res.V_p

    def test_confinement_rho_increases_with_axial_ratio(self):
        # Larger A_g/A_k -> larger required rho
        small = is13920.is13920_confinement(
            A_g=0.250, A_k=0.20, f_ck=25e6, f_yh=415e6,
            h_clear=3.0, D=0.500,
        )
        big = is13920.is13920_confinement(
            A_g=0.500, A_k=0.20, f_ck=25e6, f_yh=415e6,
            h_clear=3.0, D=0.500,
        )
        assert big.rho_st_required > small.rho_st_required

    def test_confinement_validates(self):
        with pytest.raises(ValueError, match="A_k"):
            is13920.is13920_confinement(
                A_g=0.100, A_k=0.200, f_ck=25e6, f_yh=415e6,
                h_clear=3.0, D=0.500,
            )
