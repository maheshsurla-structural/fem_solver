"""Phase 46 tests -- Eurocodes EC2 / EC3 / EC8.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from femsolver.design import ec2, ec3, ec8


# ============================================================ EC2

class TestEC2Materials:
    def test_fck_class_lookup(self):
        assert ec2.fck_class("C30/37") == 30.0e6
        assert ec2.fck_class("C50/60") == 50.0e6

    def test_fck_class_unknown(self):
        with pytest.raises(ValueError, match="unknown"):
            ec2.fck_class("C100/120")

    def test_fyk_grade_B500(self):
        assert ec2.fyk_grade("B500") == 500.0e6

    def test_stress_block_normal_strength(self):
        # f_ck = 30 MPa: lambda = 0.8, eta = 1.0
        lam, eta = ec2._stress_block_params(30.0e6)
        assert lam == 0.8 and eta == 1.0

    def test_stress_block_HSC_60MPa(self):
        # f_ck = 60 MPa: lambda = 0.8 - 10/400 = 0.775, eta = 1 - 10/200 = 0.95
        lam, eta = ec2._stress_block_params(60.0e6)
        assert lam == pytest.approx(0.775, rel=1e-9)
        assert eta == pytest.approx(0.95, rel=1e-9)


class TestEC2Flexure:
    def test_singly_reinforced_normal_case(self):
        res = ec2.ec2_beam_flexure(
            M_Ed=150e3,
            f_ck=ec2.fck_class("C30/37"),
            f_yk=ec2.fyk_grade("B500"),
            b=0.300, d=0.460,
        )
        assert not res.is_doubly
        # Sanity: A_s should be in the 500-1500 mm^2 range for this moment
        assert 500e-6 < res.A_s < 1500e-6
        assert res.x_over_d < 0.45

    def test_doubly_reinforced_triggers_above_M_Rd_max(self):
        # Very high M -> doubly required (small section)
        res = ec2.ec2_beam_flexure(
            M_Ed=500e3,
            f_ck=ec2.fck_class("C25/30"),
            f_yk=ec2.fyk_grade("B500"),
            b=0.250, d=0.450, d2=0.050,
        )
        assert res.is_doubly
        assert res.A_s2 > 0

    def test_rejects_invalid(self):
        with pytest.raises(ValueError):
            ec2.ec2_beam_flexure(
                M_Ed=-1.0, f_ck=30e6, f_yk=500e6, b=0.3, d=0.4,
            )

    def test_min_max_steel(self):
        # C30/37 (f_ctm = 0.30 * 30^(2/3) = 2.90 MPa), B500
        A_min, A_max = ec2.ec2_min_max_tension_steel(
            f_ck=ec2.fck_class("C30/37"),
            f_yk=ec2.fyk_grade("B500"),
            b=0.300, d=0.460,
        )
        # 0.26 * 2.90 / 500 = 0.00151 -> 0.00151 * 300 * 460 = 208 mm^2
        assert A_min * 1e6 == pytest.approx(208.0, rel=0.05)
        # 0.04 * 300 * 460 = 5520 mm^2
        assert A_max * 1e6 == pytest.approx(5520.0, rel=0.05)


class TestEC2Shear:
    def test_no_stirrups_when_V_below_V_Rd_c(self):
        res = ec2.ec2_beam_shear(
            V_Ed=20e3, f_ck=ec2.fck_class("C30/37"),
            f_yk_sv=ec2.fyk_grade("B500"),
            b_w=0.300, d=0.460, A_s=1000e-6,
        )
        assert not res.requires_stirrups

    def test_stirrups_required_above_V_Rd_c(self):
        res = ec2.ec2_beam_shear(
            V_Ed=200e3, f_ck=ec2.fck_class("C30/37"),
            f_yk_sv=ec2.fyk_grade("B500"),
            b_w=0.300, d=0.460, A_s=500e-6,
        )
        assert res.requires_stirrups
        assert res.A_sw_over_s_required > 0

    def test_crushing_redesign_when_above_V_Rd_max(self):
        res = ec2.ec2_beam_shear(
            V_Ed=2.0e6, f_ck=ec2.fck_class("C25/30"),
            f_yk_sv=ec2.fyk_grade("B500"),
            b_w=0.250, d=0.450, A_s=500e-6,
        )
        assert math.isinf(res.A_sw_over_s_required)

    def test_cot_theta_validation(self):
        with pytest.raises(ValueError, match="cot_theta"):
            ec2.ec2_beam_shear(
                V_Ed=100e3, f_ck=30e6, f_yk_sv=500e6,
                b_w=0.3, d=0.4, A_s=500e-6, cot_theta=3.0,
            )


# ============================================================ EC3

class TestEC3:
    def test_fy_grade_S355(self):
        assert ec3.fy_grade("S355") == 355.0e6

    def test_perry_robertson_curve_a0(self):
        # Curve a0 is the most generous (best buckling performance)
        chi_a0 = ec3.perry_robertson_chi(lambda_bar=1.0, curve="a0")
        chi_d = ec3.perry_robertson_chi(lambda_bar=1.0, curve="d")
        # a0 should give higher chi than d
        assert chi_a0 > chi_d

    def test_perry_robertson_matches_textbook(self):
        # Textbook reference for curve b at lambda=1.0: chi ~ 0.597
        chi = ec3.perry_robertson_chi(lambda_bar=1.0, curve="b")
        assert chi == pytest.approx(0.597, abs=0.005)

    def test_perry_robertson_at_zero_is_one(self):
        chi = ec3.perry_robertson_chi(lambda_bar=0.0, curve="b")
        assert chi == pytest.approx(1.0, abs=1e-9)

    def test_tension_gross_governs_when_A_net_is_full(self):
        res = ec3.ec3_tension(
            N_Ed=300e3, A=1903e-6,
            f_y=ec3.fy_grade("S235"), f_u=ec3.fu_grade("S235"),
        )
        # When A_net = A, the gross-yield (gamma_M0=1.0) vs ultimate
        # rupture (0.9 A f_u / 1.25) -- compare which is smaller
        # for S235: f_y/1.0 = 235; 0.9·360/1.25 = 259 -> gross govern
        assert res.governing == "gross"

    def test_compression_basic(self):
        res = ec3.ec3_compression(
            N_Ed=500e3, A=53.83e-4,
            f_y=ec3.fy_grade("S355"),
            r_min=0.0498, L_cr=4.0, curve="b",
        )
        assert 0.5 < res.chi < 0.7
        assert res.lambda_bar == pytest.approx(1.05, abs=0.1)

    def test_flexure_no_LTB_returns_M_pl(self):
        res = ec3.ec3_flexure(
            M_Ed=100e3, W_pl=651e-6,
            f_y=ec3.fy_grade("S275"),
        )
        # M_pl = 651e-6 * 275e6 / 1.0 = 179 kN.m
        assert res.M_pl_Rd == pytest.approx(179.0e3, rel=1e-3)
        assert res.chi_LT == 1.0

    def test_flexure_LTB_reduces(self):
        res = ec3.ec3_flexure(
            M_Ed=100e3, W_pl=651e-6,
            f_y=ec3.fy_grade("S275"),
            L_LT=3.0, M_cr=200e3,
        )
        assert res.chi_LT < 1.0
        assert res.M_b_Rd < res.M_pl_Rd

    def test_shear_formula(self):
        res = ec3.ec3_shear(
            V_Ed=200e3, A_v=2310e-6,
            f_y=ec3.fy_grade("S275"),
        )
        # V_pl = 2310e-6 * 275e6 / (sqrt(3) * 1.0) = 366.7 kN
        assert res.V_pl_Rd == pytest.approx(366.7e3, rel=1e-3)


# ============================================================ EC8

class TestEC8GroundTypes:
    def test_type_1_A_default(self):
        S, T_B, T_C, T_D = ec8.ground_type_parameters("A", spectrum_type=1)
        assert S == 1.0 and T_B == 0.15 and T_C == 0.4 and T_D == 2.0

    def test_type_1_D_soft(self):
        S, T_B, T_C, T_D = ec8.ground_type_parameters("D", spectrum_type=1)
        # Soft ground: higher S (amplification), longer corner periods
        assert S == 1.35
        assert T_C == 0.8

    def test_unknown_ground_type(self):
        with pytest.raises(ValueError, match="ground_type"):
            ec8.ground_type_parameters("F", spectrum_type=1)


class TestEC8DesignSpectrum:
    def test_plateau_value(self):
        # T in plateau region: S_d = a_g · S · 2.5 / q
        Sd = ec8.design_spectrum_Sd(
            T=0.3, a_g=2.5, ground_type="A", q=1.5,
        )
        # 2.5 * 1.0 * 2.5 / 1.5 = 4.17
        assert Sd == pytest.approx(4.167, rel=1e-3)

    def test_short_period_below_plateau(self):
        # Below plateau, linear rising
        Sd_low = ec8.design_spectrum_Sd(
            T=0.05, a_g=2.5, ground_type="A", q=1.5,
        )
        Sd_at_TB = ec8.design_spectrum_Sd(
            T=0.15, a_g=2.5, ground_type="A", q=1.5,
        )
        assert Sd_low < Sd_at_TB

    def test_decay_for_long_periods(self):
        # T > T_C: 1/T decay
        Sd_short = ec8.design_spectrum_Sd(
            T=0.4, a_g=2.5, ground_type="A", q=1.5,
        )
        Sd_long = ec8.design_spectrum_Sd(
            T=1.5, a_g=2.5, ground_type="A", q=1.5,
        )
        assert Sd_long < Sd_short


class TestEC8BehaviourFactors:
    def test_DCL_low(self):
        assert ec8.behaviour_factor_default("RC_DCL") == 1.5

    def test_DCH_high_RC(self):
        assert ec8.behaviour_factor_default("RC_DCH_FRAME") == 4.5

    def test_unknown_system(self):
        with pytest.raises(ValueError, match="unknown"):
            ec8.behaviour_factor_default("BLAH")


class TestEC8BaseShear:
    def test_base_shear_formula(self):
        # T_1 = 0.3 is in plateau region (T < T_C = 0.4 for type A)
        res = ec8.ec8_base_shear(
            T_1=0.3, m_total=1000e3, a_g=2.5,
            ground_type="A", q=1.5, n_storeys=4,
        )
        # Plateau: Sd = a_g · S · 2.5 / q = 2.5 * 1.0 * 2.5/1.5 = 4.17
        # F_b = Sd * m * lambda = 4.17 * 1000e3 * 0.85 = 3542 kN
        assert res.F_b == pytest.approx(3542e3, rel=0.01)

    def test_lambda_for_short_period_low_storey_buildings(self):
        # n=2 -> lambda = 1.0
        res = ec8.ec8_base_shear(
            T_1=0.3, m_total=500e3, a_g=2.5,
            ground_type="C", q=3.0, n_storeys=2,
        )
        assert res.lambda_factor == 1.0


class TestEC8VerticalDistribution:
    def test_distribution_sums_to_F_b(self):
        F = ec8.vertical_force_distribution(
            F_b=1000e3,
            storey_masses=np.array([200e3, 200e3, 200e3]),
            storey_heights=np.array([3.0, 6.0, 9.0]),
        )
        assert F.sum() == pytest.approx(1000e3, rel=1e-12)

    def test_higher_floor_gets_more_force(self):
        F = ec8.vertical_force_distribution(
            F_b=1000e3,
            storey_masses=np.array([200e3, 200e3, 200e3]),
            storey_heights=np.array([3.0, 6.0, 9.0]),
        )
        assert F[0] < F[1] < F[2]


class TestEC8DriftCheck:
    def test_drift_amplified_by_q_nu(self):
        # Elastic drift 5 mm over 3 m height -> ratio 0.00167
        # With q=3.0, nu=0.5: design ratio = 3*0.5*0.00167 = 0.0025
        # Below brittle limit (0.005) -> PASS
        u = np.array([0.005, 0.010])
        h = np.array([3.0, 3.0])
        res = ec8.ec8_drift_check(
            floor_disp=u, storey_heights=h, q=3.0,
            importance_class="II", infill_type="brittle",
        )
        assert res.passes
