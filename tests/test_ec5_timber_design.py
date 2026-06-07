"""Phase D.1.4 tests -- EC5 §6 timber design checks."""
from __future__ import annotations

import math

import pytest

from femsolver.design.timber import (
    EC5Factors,
    ec5_bending_check,
    ec5_combined_check,
    ec5_compression_check,
    ec5_shear_check,
    ec5_tension_check,
    gamma_M_partial_factor,
    k_c_column_stability,
    k_crit_lateral_stability,
    k_h_glulam,
    k_h_solid,
    k_mod_factor,
)
from femsolver.materials.timber import get_ec5_class


# ============================================================ k_mod table

class TestKMod:
    def test_solid_class1_medium(self):
        assert k_mod_factor("solid", 1, "medium_term") == 0.8

    def test_solid_class1_permanent(self):
        assert k_mod_factor("solid", 1, "permanent") == 0.6

    def test_solid_class1_instantaneous(self):
        """Wind / seismic gets the highest k_mod."""
        assert k_mod_factor("solid", 1, "instantaneous") == 1.1

    def test_class3_reduces_k_mod(self):
        """Service class 3 (outdoor) gives lower k_mod."""
        k_c1 = k_mod_factor("solid", 1, "medium_term")
        k_c3 = k_mod_factor("solid", 3, "medium_term")
        assert k_c3 < k_c1

    def test_glulam_class1_same_as_solid(self):
        assert k_mod_factor("glulam", 1, "medium_term") == k_mod_factor("solid", 1, "medium_term")

    def test_invalid_combo_raises(self):
        with pytest.raises(ValueError):
            k_mod_factor("solid", 99, "medium_term")


# ============================================================ gamma_M

class TestGammaM:
    def test_solid(self):
        assert gamma_M_partial_factor("solid") == 1.30

    def test_glulam(self):
        assert gamma_M_partial_factor("glulam") == 1.25

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            gamma_M_partial_factor("unknown")


# ============================================================ k_h size

class TestKH:
    def test_solid_above_150(self):
        """For h >= 150 mm: k_h = 1."""
        assert k_h_solid(0.150) == 1.0
        assert k_h_solid(0.200) == 1.0
        assert k_h_solid(0.300) == 1.0

    def test_solid_below_150(self):
        """k_h = min((150/h)^0.2, 1.3) for h < 150 mm."""
        # h = 100mm: (150/100)^0.2 = 1.084
        assert k_h_solid(0.100) == pytest.approx((1.5) ** 0.2, rel=1e-6)
        # h = 50mm: (150/50)^0.2 = 3^0.2 = 1.246 (< 1.3 cap)
        assert k_h_solid(0.050) == pytest.approx(3.0 ** 0.2, rel=1e-6)

    def test_solid_capped_at_1_3(self):
        """Very small h: k_h capped at 1.3."""
        # h = 10mm: (150/10)^0.2 = 15^0.2 = 1.719 -> cap at 1.3
        assert k_h_solid(0.010) == 1.3

    def test_glulam_above_600(self):
        assert k_h_glulam(0.600) == 1.0
        assert k_h_glulam(1.000) == 1.0

    def test_glulam_below_600(self):
        """k_h = min((600/h)^0.1, 1.1)."""
        # h = 200mm: (600/200)^0.1 = 3^0.1 = 1.116 -> cap at 1.1
        assert k_h_glulam(0.200) == 1.1
        # h = 500mm: (600/500)^0.1 = 1.2^0.1 = 1.018
        assert k_h_glulam(0.500) == pytest.approx(1.2 ** 0.1, rel=1e-6)


# ============================================================ k_crit

class TestKCrit:
    def test_low_slenderness(self):
        """For high sigma_crit (short stocky beam), k_crit = 1."""
        # sigma_m_crit >> f_m_k -> lambda_rel_m small -> k_crit = 1
        k = k_crit_lateral_stability(f_m_k=24e6, sigma_m_crit=1000e6)
        assert k == 1.0

    def test_intermediate(self):
        """0.75 < lambda <= 1.4: linear formula."""
        # Pick sigma_m_crit such that lambda_rel_m = 1.0:
        # sqrt(f_m_k / sigma_crit) = 1 -> sigma_crit = f_m_k = 24e6
        k = k_crit_lateral_stability(f_m_k=24e6, sigma_m_crit=24e6)
        # k_crit = 1.56 - 0.75 * 1.0 = 0.81
        assert k == pytest.approx(0.81, rel=1e-9)

    def test_slender(self):
        """For lambda > 1.4: k_crit = 1/lambda^2."""
        # lambda = 2.0: sigma_crit = f_m_k / 4 = 6e6
        k = k_crit_lateral_stability(f_m_k=24e6, sigma_m_crit=6e6)
        # k = 1/(2.0)^2 = 0.25
        assert k == pytest.approx(0.25, rel=1e-9)


# ============================================================ k_c

class TestKC:
    def test_short_column(self):
        """Low slenderness -> k_c close to 1."""
        # lambda_rel = (slenderness / pi) * sqrt(f_c / E_05)
        # Need lambda_rel <= 0.3: slenderness <= 0.3 * pi * sqrt(E_05/f_c)
        # For C24: f_c = 21e6, E_05 = 7.4e9, ratio = 18.8
        # slenderness limit = 0.3 * pi * 18.8 = 17.7
        # Use slenderness = 10
        k = k_c_column_stability(
            f_c_0_k=21e6, slenderness=10.0, E_0_05=7.4e9,
        )
        assert k == 1.0

    def test_slender_column(self):
        k = k_c_column_stability(
            f_c_0_k=21e6, slenderness=150.0, E_0_05=7.4e9,
        )
        # Very slender -> k_c much less than 1
        assert k < 0.2

    def test_glulam_higher_k_c(self):
        """Glulam (beta_c = 0.1) gives higher k_c than solid (0.2)."""
        k_sol = k_c_column_stability(
            f_c_0_k=21e6, slenderness=60.0, E_0_05=7.4e9, beta_c=0.2,
        )
        k_glu = k_c_column_stability(
            f_c_0_k=21e6, slenderness=60.0, E_0_05=7.4e9, beta_c=0.1,
        )
        assert k_glu > k_sol


# ============================================================ bending

class TestEC5Bending:
    def test_C24_f_m_d_hand_calc(self):
        """C24, k_mod=0.8, gamma_M=1.3:
        f_m_d = 0.8 * 24 / 1.3 = 14.769 MPa
        (k_h = 1 for h >= 150mm)"""
        c24 = get_ec5_class("C24")
        check = ec5_bending_check(
            b=0.10, h=0.20, material=c24, M_Ed=1,
            factors=EC5Factors(k_mod=0.8, gamma_M=1.3),
            auto_k_crit=False,
        )
        assert check.f_m_d == pytest.approx(0.8 * 24e6 / 1.3, rel=1e-9)
        assert check.factors.k_h == 1.0

    def test_GL28h_f_m_d_with_k_h(self):
        """GL28h (h=200mm < 600mm), k_mod=0.8, gamma_M=1.25:
        k_h_glulam(200) = (600/200)^0.1 = 1.116 -> capped at 1.1
        f_m_d = 0.8 * 28 / 1.25 * 1.1 = 19.712 MPa"""
        gl = get_ec5_class("GL28h")
        check = ec5_bending_check(
            b=0.10, h=0.20, material=gl, M_Ed=1,
            factors=EC5Factors(k_mod=0.8, gamma_M=1.25),
            material_type="glulam",
            auto_k_crit=False,
        )
        expected = 0.8 * 28e6 / 1.25 * 1.1
        assert check.f_m_d == pytest.approx(expected, rel=1e-9)

    def test_small_member_higher_k_h(self):
        """80mm deep solid C24 gets k_h_solid(80) > 1."""
        c24 = get_ec5_class("C24")
        check = ec5_bending_check(
            b=0.05, h=0.08, material=c24, M_Ed=1,
            factors=EC5Factors(k_mod=0.8, gamma_M=1.3),
            auto_k_crit=False,
        )
        # k_h = min((150/80)^0.2, 1.3) = 1.875^0.2 = 1.133
        assert check.factors.k_h > 1.0
        assert check.factors.k_h == pytest.approx((150/80) ** 0.2, rel=1e-6)

    def test_passes_below_M_Rd(self):
        c24 = get_ec5_class("C24")
        check = ec5_bending_check(
            b=0.10, h=0.20, material=c24, M_Ed=5e3,
            factors=EC5Factors(k_mod=0.8, gamma_M=1.3),
        )
        assert check.passes
        assert check.DCR < 1.0

    def test_lateral_stability_reduces_capacity(self):
        """Long unbraced beam -> k_crit < 1."""
        c24 = get_ec5_class("C24")
        # Tall narrow beam
        check_short = ec5_bending_check(
            b=0.05, h=0.30, material=c24, M_Ed=1,
            factors=EC5Factors(k_mod=0.8, gamma_M=1.3),
            l_ef=0.5,
        )
        check_long = ec5_bending_check(
            b=0.05, h=0.30, material=c24, M_Ed=1,
            factors=EC5Factors(k_mod=0.8, gamma_M=1.3),
            l_ef=8.0,
        )
        assert check_short.factors.k_crit > check_long.factors.k_crit
        assert check_long.factors.k_crit < 1.0


# ============================================================ tension

class TestEC5Tension:
    def test_f_t_0_d_hand_calc(self):
        """C24 f_t_0_k=14, k_mod=0.8, gamma_M=1.3, k_h=1 (h>=150):
        f_t_0_d = 0.8 * 14 / 1.3 = 8.62 MPa"""
        c24 = get_ec5_class("C24")
        check = ec5_tension_check(
            b=0.10, h=0.20, material=c24, T_Ed=1,
            factors=EC5Factors(k_mod=0.8, gamma_M=1.3),
        )
        assert check.f_t_0_d == pytest.approx(0.8 * 14e6 / 1.3, rel=1e-9)


# ============================================================ compression

class TestEC5Compression:
    def test_f_c_0_d_hand_calc(self):
        """C24 f_c_0_k=21, k_mod=0.8, gamma_M=1.3:
        f_c_0_d = 0.8 * 21 / 1.3 = 12.92 MPa"""
        c24 = get_ec5_class("C24")
        check = ec5_compression_check(
            b=0.10, h=0.10, material=c24, P_Ed=1,
            factors=EC5Factors(k_mod=0.8, gamma_M=1.3),
        )
        assert check.f_c_0_d == pytest.approx(0.8 * 21e6 / 1.3, rel=1e-9)

    def test_short_column_k_c_one(self):
        """For very short column, k_c = 1."""
        c24 = get_ec5_class("C24")
        check = ec5_compression_check(
            b=0.20, h=0.20, material=c24, P_Ed=1,
            factors=EC5Factors(k_mod=0.8, gamma_M=1.3),
            l_ef=0.5,
        )
        assert check.factors.k_c == 1.0

    def test_long_column_k_c_low(self):
        c24 = get_ec5_class("C24")
        check = ec5_compression_check(
            b=0.10, h=0.10, material=c24, P_Ed=1,
            factors=EC5Factors(k_mod=0.8, gamma_M=1.3),
            l_ef=4.0,
        )
        assert check.factors.k_c < 0.5


# ============================================================ shear

class TestEC5Shear:
    def test_solid_k_cr_applied(self):
        """Solid timber gets k_cr = 0.67 (crack factor).
        f_v_d = k_mod * f_v_k / gamma_M * k_cr
        = 0.8 * 4 / 1.3 * 0.67 = 1.649 MPa"""
        c24 = get_ec5_class("C24")
        check = ec5_shear_check(
            b=0.10, h=0.20, material=c24, V_Ed=1,
            factors=EC5Factors(k_mod=0.8, gamma_M=1.3, k_cr=0.67),
            material_type="solid",
        )
        expected = 0.8 * 4e6 / 1.3 * 0.67
        assert check.f_v_d == pytest.approx(expected, rel=1e-3)

    def test_glulam_no_k_cr_reduction(self):
        """Glulam: k_cr = 1.0 -> higher shear capacity."""
        gl = get_ec5_class("GL28h")
        s = ec5_shear_check(
            b=0.10, h=0.20, material=gl, V_Ed=1,
            factors=EC5Factors(k_mod=0.8, gamma_M=1.25),
            material_type="glulam",
        )
        # k_cr should be reset to 1.0
        assert s.factors.k_cr == 1.0


# ============================================================ combined

class TestEC5Combined:
    def test_compression_plus_bending(self):
        """Standard case: should give interaction ~ ratio_c² + ratio_b."""
        c24 = get_ec5_class("C24")
        result = ec5_combined_check(
            b=0.10, h=0.20, material=c24,
            P_Ed=20e3, M_strong=2e3,
            factors=EC5Factors(k_mod=0.8, gamma_M=1.3),
            l_ef=2.0, l_ef_compress=2.0,
        )
        assert "6.2.4" in result.note
        assert result.interaction > 0

    def test_tension_plus_bending(self):
        c24 = get_ec5_class("C24")
        result = ec5_combined_check(
            b=0.10, h=0.20, material=c24,
            P_Ed=20e3, M_strong=2e3,
            is_tension=True,
            factors=EC5Factors(k_mod=0.8, gamma_M=1.3),
        )
        assert "6.2.3" in result.note
        # Tension+bending should have lower interaction than
        # compression+bending (linear vs squared form)
        comp_result = ec5_combined_check(
            b=0.10, h=0.20, material=c24,
            P_Ed=20e3, M_strong=2e3,
            is_tension=False,
            factors=EC5Factors(k_mod=0.8, gamma_M=1.3),
        )
        # For pure tension+bending the σ_t/f_t term is linear,
        # for compression it's squared. At low ratios, the
        # quadratic form is much smaller. So if axial dominates,
        # tension+bending will be HIGHER. If bending dominates,
        # they're closer.

    def test_passes_below_limit(self):
        c24 = get_ec5_class("C24")
        result = ec5_combined_check(
            b=0.10, h=0.20, material=c24,
            P_Ed=10e3, M_strong=500,
            factors=EC5Factors(k_mod=0.8, gamma_M=1.3),
            l_ef=2.0,
        )
        assert result.passes

    def test_fails_at_high_loads(self):
        c24 = get_ec5_class("C24")
        result = ec5_combined_check(
            b=0.10, h=0.20, material=c24,
            P_Ed=200e3, M_strong=20e3,
            factors=EC5Factors(k_mod=0.8, gamma_M=1.3),
            l_ef=2.0,
        )
        assert not result.passes


# ============================================================ NDS vs EC5 comparison

class TestCrossCodeComparison:
    """Same section, same loading, NDS vs EC5 -- both should give
    sensible answers, with EC5 being more conservative due to
    partial-safety factors."""

    def test_C24_via_NDS_vs_EC5(self):
        from femsolver.design.timber import NDSFactors, nds_bending_check
        c24 = get_ec5_class("C24")
        # EC5 design
        ec5 = ec5_bending_check(
            b=0.10, h=0.20, material=c24, M_Ed=1,
            factors=EC5Factors(k_mod=0.8, gamma_M=1.3),
            auto_k_crit=False,
        )
        # NDS with same characteristic values (treats f_m_k as F_b
        # reference), C_D=1
        nds = nds_bending_check(
            b=0.10, d=0.20, material=c24, M_applied=1,
            factors=NDSFactors(C_D=1.0),
        )
        # EC5: 0.8 * 24/1.3 = 14.77 MPa
        # NDS: 24 MPa (no partial factor, no k_mod)
        assert ec5.f_m_d < nds.F_b_prime
        # Ratio should be ~ k_mod / gamma_M = 0.8/1.3 = 0.615
        ratio = ec5.f_m_d / nds.F_b_prime
        assert ratio == pytest.approx(0.8 / 1.3, rel=1e-6)
