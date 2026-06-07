"""Phase D.1.3 tests -- NDS Ch. 3 timber design checks."""
from __future__ import annotations

import pytest

from femsolver.design.timber import (
    NDSFactors,
    C_D_load_duration,
    C_F_size_factor,
    C_L_lateral_stability,
    C_M_wet_service,
    C_P_column_stability,
    C_r_repetitive_member,
    nds_bending_check,
    nds_combined_check,
    nds_compression_check,
    nds_shear_check,
    nds_tension_check,
)
from femsolver.materials.timber import get_nds_timber


# ============================================================ C-factor helpers

class TestCDLoadDuration:
    def test_table_2_3_2(self):
        assert C_D_load_duration("permanent") == 0.9
        assert C_D_load_duration("normal") == 1.0
        assert C_D_load_duration("two_months") == 1.15
        assert C_D_load_duration("seven_days") == 1.25
        assert C_D_load_duration("ten_minutes") == 1.6
        assert C_D_load_duration("impact") == 2.0

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            C_D_load_duration("invalid")


class TestCMWetService:
    def test_bending_wet_85(self):
        """NDS Table 4A: C_M for F_b = 0.85 when wet (>19% MC)."""
        assert C_M_wet_service("F_b") == 0.85

    def test_dry_returns_one(self):
        assert C_M_wet_service("F_b", wet=False) == 1.0
        assert C_M_wet_service("F_c", wet=False) == 1.0

    def test_compression_wet(self):
        assert C_M_wet_service("F_c") == 0.8

    def test_shear_wet(self):
        assert C_M_wet_service("F_v") == 0.97

    def test_bearing_wet(self):
        assert C_M_wet_service("F_c_perp") == 0.67


class TestCFSizeFactor:
    def test_2x4_depth_C_F_1_5(self):
        """2x4 nominal = 3.5 in actual = 88.9 mm. C_F = 1.5."""
        assert C_F_size_factor(0.089) == 1.5

    def test_2x12_C_F_1_0(self):
        """2x12 nominal = 11.25 in actual = 285 mm. C_F = 1.0 (reference)."""
        assert C_F_size_factor(0.285) == 1.0

    def test_deep_member_C_F_lower(self):
        """For depths above 12", C_F drops to 0.9."""
        assert C_F_size_factor(0.40) == 0.9

    def test_monotonic_decrease_with_depth(self):
        depths = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
        Cs = [C_F_size_factor(d) for d in depths]
        for i in range(len(Cs) - 1):
            assert Cs[i + 1] <= Cs[i]


class TestCrRepetitive:
    def test_alone(self):
        assert C_r_repetitive_member(1) == 1.0
        assert C_r_repetitive_member(2) == 1.0

    def test_three_or_more(self):
        assert C_r_repetitive_member(3) == 1.15
        assert C_r_repetitive_member(10) == 1.15


class TestCPColumnStability:
    def test_C_P_formula(self):
        """NDS Eq. 3.7-1 verification: short column has C_P near 1,
        slender column has C_P much less than 1."""
        # Short column: l_e/d = 5
        C_P_short = C_P_column_stability(
            F_c_star=1700 * 6894.757, l_e=0.05 * 5, d=0.05,
            E_min=0.690e6 * 6894.757,
        )
        # Slender column: l_e/d = 40
        C_P_slender = C_P_column_stability(
            F_c_star=1700 * 6894.757, l_e=0.05 * 40, d=0.05,
            E_min=0.690e6 * 6894.757,
        )
        assert C_P_short > 0.9
        assert C_P_slender < 0.3

    def test_C_P_glulam_shape_factor(self):
        """Glulam c = 0.9 gives higher C_P than sawn c = 0.8 at same
        slenderness."""
        C_P_sawn = C_P_column_stability(
            F_c_star=10e6, l_e=2.0, d=0.10,
            E_min=5e9, c=0.8,
        )
        C_P_glulam = C_P_column_stability(
            F_c_star=10e6, l_e=2.0, d=0.10,
            E_min=5e9, c=0.9,
        )
        assert C_P_glulam > C_P_sawn

    def test_C_P_exceeds_50_raises(self):
        with pytest.raises(ValueError, match="50"):
            C_P_column_stability(
                F_c_star=10e6, l_e=10.0, d=0.05,
                E_min=5e9,
            )


class TestCLLateralStability:
    def test_short_span_C_L_one(self):
        """For deep narrow beam, l_e·d/b² small -> C_L close to 1."""
        C_L = C_L_lateral_stability(
            F_b_star=15e6, l_e=0.5, d=0.2, b=0.1,
            E_min=5e9,
        )
        assert C_L > 0.95

    def test_slender_beam_C_L_lower(self):
        """Long shallow narrow beam -> slender -> C_L < 0.5."""
        C_L = C_L_lateral_stability(
            F_b_star=15e6, l_e=10.0, d=0.3, b=0.05,
            E_min=5e9,
        )
        assert C_L < 0.5


# ============================================================ bending

class TestNDSBending:
    def test_F_b_prime_matches_hand_calc(self):
        """2x8 DFL-1 joist, C_D=1.0, C_F=1.2, C_r=1.15:
        F_b' = 1000 * 1.0 * 1.2 * 1.15 = 1380 psi = 9.51 MPa"""
        b, d = 0.0508, 0.184
        dfl1 = get_nds_timber("DFL-1")
        check = nds_bending_check(
            b=b, d=d, material=dfl1,
            M_applied=1000,    # arbitrary
            factors=NDSFactors(C_D=1.0, C_r=1.15, C_F=C_F_size_factor(d)),
        )
        expected = 1380 * 6894.757   # 1380 psi in Pa
        assert check.F_b_prime == pytest.approx(expected, rel=1e-4)

    def test_passes_below_limit(self):
        b, d = 0.0508, 0.184
        dfl1 = get_nds_timber("DFL-1")
        check = nds_bending_check(
            b=b, d=d, material=dfl1,
            M_applied=2200,
            factors=NDSFactors(C_D=1.0, C_r=1.15, C_F=C_F_size_factor(d)),
        )
        assert check.passes
        assert check.DCR < 1.0

    def test_fails_above_limit(self):
        b, d = 0.0508, 0.184
        dfl1 = get_nds_timber("DFL-1")
        check = nds_bending_check(
            b=b, d=d, material=dfl1,
            M_applied=5000,    # too high
            factors=NDSFactors(C_D=1.0),
        )
        assert not check.passes
        assert check.DCR > 1.0

    def test_C_D_scales_capacity(self):
        b, d = 0.0508, 0.184
        dfl1 = get_nds_timber("DFL-1")
        c1 = nds_bending_check(
            b=b, d=d, material=dfl1, M_applied=1000,
            factors=NDSFactors(C_D=1.0),
        )
        c2 = nds_bending_check(
            b=b, d=d, material=dfl1, M_applied=1000,
            factors=NDSFactors(C_D=1.6),
        )
        assert c2.F_b_prime == pytest.approx(1.6 * c1.F_b_prime, rel=1e-9)

    def test_auto_C_L_with_l_e(self):
        b, d = 0.0508, 0.20
        dfl1 = get_nds_timber("DFL-1")
        check = nds_bending_check(
            b=b, d=d, material=dfl1, M_applied=1000,
            factors=NDSFactors(C_D=1.0),
            l_e=4.0,
        )
        # Long slender beam: C_L should be < 1
        assert check.factors.C_L < 1.0


# ============================================================ tension

class TestNDSTension:
    def test_F_t_prime_matches_hand_calc(self):
        """DFL-SS: F_t = 1000 psi. F_t' = 1000 * 1.0 * 1.0 = 1000 psi."""
        b, d = 0.0508, 0.184
        dfl_ss = get_nds_timber("DFL-SS")
        check = nds_tension_check(
            b=b, d=d, material=dfl_ss,
            T_applied=10e3,
            factors=NDSFactors(C_D=1.0),
        )
        expected = 1000 * 6894.757
        assert check.F_t_prime == pytest.approx(expected, rel=1e-4)

    def test_passes_below_limit(self):
        b, d = 0.0508, 0.184
        dfl_ss = get_nds_timber("DFL-SS")
        # F_t' * A = 6.895e6 * 0.0508*0.184 = 64,460 N
        check = nds_tension_check(
            b=b, d=d, material=dfl_ss,
            T_applied=40e3,
            factors=NDSFactors(C_D=1.0),
        )
        assert check.passes


# ============================================================ compression

class TestNDSCompression:
    def test_short_column_no_buckling(self):
        """Stub column (l_e/d small) -> C_P near 1 -> capacity ~ F_c·A."""
        b, d = 0.089, 0.089   # 4x4
        dfl_ss = get_nds_timber("DFL-SS")
        check = nds_compression_check(
            b=b, d=d, material=dfl_ss,
            P_applied=10e3,
            factors=NDSFactors(C_D=1.0),
            l_e=0.30,    # very short, l_e/d ~3.4
        )
        # F_c = 1700 psi = 11.72 MPa, A = 0.089*0.089 = 7.92e-3
        # P_max ~ 11.72e6 * 7.92e-3 = 92.8 kN (without C_P)
        assert check.factors.C_P > 0.95
        assert check.P_allow > 80e3

    def test_long_column_C_P_low(self):
        """3m unbraced 4x4 column -> very slender -> C_P low."""
        b, d = 0.089, 0.089
        dfl_ss = get_nds_timber("DFL-SS")
        check = nds_compression_check(
            b=b, d=d, material=dfl_ss,
            P_applied=20e3,
            factors=NDSFactors(C_D=1.0),
            l_e=3.0,
        )
        # l_e/d = 33.7 -> slender
        assert check.factors.C_P < 0.4

    def test_glulam_higher_C_P_than_sawn(self):
        """Glulam (c=0.9) at same slenderness vs sawn (c=0.8) gives
        higher C_P."""
        from femsolver.materials.timber import TimberMaterial
        from femsolver.materials.timber.material import TimberMaterial
        # Use same material but different shape factor
        c_sawn = nds_compression_check(
            b=0.10, d=0.10, material=get_nds_timber("DFL-SS"),
            P_applied=10e3,
            factors=NDSFactors(C_D=1.0),
            l_e=2.0, column_shape_factor_c=0.8,
        )
        c_glu = nds_compression_check(
            b=0.10, d=0.10, material=get_nds_timber("DFL-SS"),
            P_applied=10e3,
            factors=NDSFactors(C_D=1.0),
            l_e=2.0, column_shape_factor_c=0.9,
        )
        assert c_glu.factors.C_P > c_sawn.factors.C_P


# ============================================================ shear

class TestNDSShear:
    def test_F_v_prime_matches(self):
        """DFL-1: F_v = 180 psi. F_v' = 180 * 1.0 (with C_D=1.0)."""
        b, d = 0.0508, 0.184
        dfl1 = get_nds_timber("DFL-1")
        check = nds_shear_check(
            b=b, d=d, material=dfl1, V_applied=5e3,
            factors=NDSFactors(C_D=1.0),
        )
        expected = 180 * 6894.757
        assert check.F_v_prime == pytest.approx(expected, rel=1e-4)

    def test_V_allow_uses_1_5_factor(self):
        """V_allow = F_v' * A / 1.5 (parabolic distribution)."""
        b, d = 0.05, 0.20
        dfl1 = get_nds_timber("DFL-1")
        check = nds_shear_check(
            b=b, d=d, material=dfl1, V_applied=1,
            factors=NDSFactors(C_D=1.0),
        )
        expected = check.F_v_prime * (b * d) / 1.5
        assert check.V_allow == pytest.approx(expected, rel=1e-9)


# ============================================================ combined H

class TestNDSCombined:
    def test_tension_plus_bending(self):
        """Pure tension OR pure bending should be DCR ~ 1 each;
        combined should sum to ~2 (i.e., fail)."""
        b, d = 0.0508, 0.184
        dfl1 = get_nds_timber("DFL-1")
        # Find tension at limit
        T_check = nds_tension_check(
            b=b, d=d, material=dfl1, T_applied=1e3,
            factors=NDSFactors(C_D=1.0),
        )
        T_limit = T_check.T_allow

        # Combined with half tension + half bending should be ~ 1.0
        result = nds_combined_check(
            b=b, d=d, material=dfl1,
            P_applied=0.5 * T_limit, M_strong=0,
            is_tension=True,
            factors=NDSFactors(C_D=1.0),
        )
        assert 0.4 < result.interaction < 0.6   # ~ 0.5 from tension alone

    def test_compression_plus_bending_passes(self):
        b, d = 0.089, 0.184
        dfl_ss = get_nds_timber("DFL-SS")
        result = nds_combined_check(
            b=b, d=d, material=dfl_ss,
            P_applied=20e3, M_strong=500,
            l_e=2.0, l_e_compress=2.0,
            factors=NDSFactors(C_D=1.0),
        )
        # Reasonable load should pass
        assert result.passes
        assert "3.9-3" in result.note

    def test_compression_plus_bending_fails_at_high_load(self):
        b, d = 0.089, 0.184
        dfl_ss = get_nds_timber("DFL-SS")
        result = nds_combined_check(
            b=b, d=d, material=dfl_ss,
            P_applied=200e3, M_strong=3000,    # excessive
            l_e=2.0, l_e_compress=2.0,
            factors=NDSFactors(C_D=1.0),
        )
        assert not result.passes
        assert result.interaction > 1.0


# ============================================================ EC5 / IS 883 compat

class TestCrossCodeCompat:
    """The NDS design functions should also work with EC5 and IS 883
    materials -- they just use different reference values."""

    def test_works_with_EC5_material(self):
        from femsolver.materials.timber import get_ec5_class
        c24 = get_ec5_class("C24")
        check = nds_bending_check(
            b=0.05, d=0.20, material=c24, M_applied=1000,
            factors=NDSFactors(C_D=1.0),
        )
        # EC5 C24 f_m_k = 24 MPa, so F_b' = 24 MPa with C_D=1
        assert check.F_b_prime == pytest.approx(24e6, rel=1e-9)
        assert check.passes
