"""Phase D.1.5 tests -- CLT-specific design checks."""
from __future__ import annotations

import math

import pytest

from femsolver.design.timber import (
    EC5Factors,
    clt_deflection_check,
    clt_rolling_shear_check,
    clt_two_way_bending_check,
    clt_vibration_check,
    k_def_factor,
    k_sys_clt,
)
from femsolver.materials.timber import get_ec5_class
from femsolver.shell_sections.clt import CLTLayer, CLTSection


# ============================================================ fixture

def _make_5ply_C24(layer_t: float = 0.020) -> CLTSection:
    """Symmetric 5-ply 100 mm CLT C24 panel (20-20-20-20-20)."""
    c24 = get_ec5_class("C24")
    return CLTSection(
        [
            CLTLayer(layer_t, c24, 0.0),
            CLTLayer(layer_t, c24, 90.0),
            CLTLayer(layer_t, c24, 0.0),
            CLTLayer(layer_t, c24, 90.0),
            CLTLayer(layer_t, c24, 0.0),
        ],
        name="100-5L-C24",
    )


# ============================================================ k_sys

class TestKSys:
    def test_geq_4_with_distribution_is_1_10(self):
        assert k_sys_clt(4) == 1.10
        assert k_sys_clt(10) == 1.10

    def test_under_4_is_1_0(self):
        assert k_sys_clt(3) == 1.0
        assert k_sys_clt(1) == 1.0

    def test_no_distribution_is_1_0(self):
        assert k_sys_clt(10, has_load_distribution=False) == 1.0

    def test_zero_raises(self):
        with pytest.raises(ValueError):
            k_sys_clt(0)


# ============================================================ k_def

class TestKDef:
    def test_solid_sc1(self):
        assert k_def_factor("solid", 1) == 0.60

    def test_solid_sc3_jumps_to_2(self):
        """SC3 (exterior exposure) is much higher."""
        assert k_def_factor("solid", 3) == 2.00

    def test_CLT_sc1(self):
        assert k_def_factor("CLT", 1) == 0.60

    def test_OSB_higher_creep(self):
        """OSB creeps much more than solid timber."""
        assert k_def_factor("OSB", 1) > k_def_factor("solid", 1)

    def test_invalid_combo_raises(self):
        with pytest.raises(ValueError):
            k_def_factor("CLT", 3)  # PRG-320 doesn't allow SC3


# ============================================================ rolling shear

class TestRollingShear:
    def test_f_R_d_hand_calc(self):
        """f_R_d = k_mod * f_R_k / gamma_M
        = 0.8 * 1.1 / 1.25 = 0.704 MPa"""
        clt = _make_5ply_C24()
        rs = clt_rolling_shear_check(
            clt, V_Ed_per_width=10e3,
            factors=EC5Factors(k_mod=0.8, gamma_M=1.25),
            f_R_k=1.1e6,
        )
        assert rs.f_R_d == pytest.approx(0.8 * 1.1e6 / 1.25, rel=1e-9)

    def test_zero_shear_zero_stress(self):
        clt = _make_5ply_C24()
        rs = clt_rolling_shear_check(
            clt, V_Ed_per_width=0.0,
            factors=EC5Factors(k_mod=0.8, gamma_M=1.25),
        )
        assert rs.tau_R_max == 0.0
        assert rs.passes

    def test_high_shear_fails(self):
        clt = _make_5ply_C24()
        rs = clt_rolling_shear_check(
            clt, V_Ed_per_width=100e3,
            factors=EC5Factors(k_mod=0.8, gamma_M=1.25),
            f_R_k=1.1e6,
        )
        # 100 kN/m on a 100mm CLT will overstress rolling shear
        assert rs.DCR > 1.0
        assert not rs.passes

    def test_governing_layer_is_cross(self):
        """The reported governing layer should be one whose
        interface borders a cross (90°) layer."""
        clt = _make_5ply_C24()
        rs = clt_rolling_shear_check(
            clt, V_Ed_per_width=15e3,
            factors=EC5Factors(k_mod=0.8, gamma_M=1.25),
        )
        gov = rs.governing_layer_idx
        # The interface "below" layer `gov` should be a cross interface
        assert gov >= 0
        is_cross_interface = (
            clt.layers[gov].angle_deg == 90.0
            or clt.layers[gov + 1].angle_deg == 90.0
        )
        assert is_cross_interface

    def test_negative_shear_raises(self):
        clt = _make_5ply_C24()
        with pytest.raises(ValueError):
            clt_rolling_shear_check(
                clt, V_Ed_per_width=-1.0,
                factors=EC5Factors(k_mod=0.8, gamma_M=1.25),
            )


# ============================================================ two-way bending

class TestTwoWayBending:
    def test_strong_only_recovers_uniaxial(self):
        """Pure strong-axis moment -> ratio_weak = 0, interaction =
        ratio_strong."""
        clt = _make_5ply_C24()
        tw = clt_two_way_bending_check(
            clt, M_strong_per_width=5e3, M_weak_per_width=0.0,
            factors=EC5Factors(k_mod=0.8, gamma_M=1.25),
            k_sys=1.0,
        )
        assert tw.ratio_weak == 0.0
        assert tw.interaction == pytest.approx(tw.ratio_strong)

    def test_f_m_d_with_k_sys(self):
        """f_m_d = k_mod * k_sys * k_h * f_m_k / gamma_M
        For 100mm panel, k_h on the outer-layer thickness should
        be > 1 (each layer is only 20mm)... actually k_h is on the
        WHOLE section depth for bending strength; we use factors.k_h
        provided by caller. Default 1.0 here."""
        clt = _make_5ply_C24()
        tw = clt_two_way_bending_check(
            clt, M_strong_per_width=1e3, M_weak_per_width=0.0,
            factors=EC5Factors(k_mod=0.8, gamma_M=1.25, k_h=1.0),
            k_sys=1.1,
        )
        # 0.8 * 1.1 * 24 / 1.25 = 16.896 MPa
        assert tw.f_m_d_strong == pytest.approx(
            0.8 * 1.1 * 24e6 / 1.25, rel=1e-9
        )

    def test_passes_at_low_loads(self):
        clt = _make_5ply_C24()
        tw = clt_two_way_bending_check(
            clt, M_strong_per_width=2e3, M_weak_per_width=500.0,
            factors=EC5Factors(k_mod=0.8, gamma_M=1.25),
        )
        assert tw.passes

    def test_fails_at_high_loads(self):
        clt = _make_5ply_C24()
        tw = clt_two_way_bending_check(
            clt, M_strong_per_width=30e3, M_weak_per_width=10e3,
            factors=EC5Factors(k_mod=0.8, gamma_M=1.25),
        )
        assert not tw.passes

    def test_interaction_is_linear_sum(self):
        clt = _make_5ply_C24()
        tw = clt_two_way_bending_check(
            clt, M_strong_per_width=4e3, M_weak_per_width=1e3,
            factors=EC5Factors(k_mod=0.8, gamma_M=1.25),
        )
        assert tw.interaction == pytest.approx(
            tw.ratio_strong + tw.ratio_weak, rel=1e-9
        )

    def test_weak_axis_lower_W(self):
        """For a CLT with outer 0° layers, weak-axis section modulus
        is much smaller than strong-axis -> for same moment, weak
        sigma >> strong sigma."""
        clt = _make_5ply_C24()
        tw = clt_two_way_bending_check(
            clt, M_strong_per_width=1e3, M_weak_per_width=1e3,
            factors=EC5Factors(k_mod=0.8, gamma_M=1.25),
        )
        assert tw.sigma_weak > tw.sigma_strong


# ============================================================ deflection

class TestDeflection:
    def test_zero_load_zero_deflection(self):
        clt = _make_5ply_C24()
        d = clt_deflection_check(
            clt, span=4.0, w_perm=0.0, w_var=0.0,
        )
        assert d.u_inst == 0.0
        assert d.u_fin == 0.0
        assert d.passes_inst
        assert d.passes_fin

    def test_inst_formula(self):
        """5wL^4/(384 EI_eff). Verify for w_perm=0, w_var=1 kPa,
        L=4m."""
        clt = _make_5ply_C24()
        d = clt_deflection_check(
            clt, span=4.0, w_perm=0.0, w_var=1e3,
            use_gamma_method=False,  # use full EI for exact match
        )
        EI = clt.EI_eff_per_width(strong_axis=True)
        expected = 5.0 * 1e3 * 4.0 ** 4 / (384.0 * EI)
        assert d.u_inst == pytest.approx(expected, rel=1e-9)

    def test_creep_increases_final(self):
        clt = _make_5ply_C24()
        d = clt_deflection_check(
            clt, span=4.0, w_perm=1e3, w_var=1e3,
            psi_2=0.30, material_type="CLT", service_class=1,
        )
        assert d.u_fin > d.u_inst

    def test_SC2_more_creep_than_SC1(self):
        clt = _make_5ply_C24()
        d1 = clt_deflection_check(
            clt, span=4.0, w_perm=1e3, w_var=0.0,
            material_type="CLT", service_class=1,
        )
        d2 = clt_deflection_check(
            clt, span=4.0, w_perm=1e3, w_var=0.0,
            material_type="CLT", service_class=2,
        )
        # k_def SC2 (0.80) > SC1 (0.60) -> larger u_fin
        assert d2.u_fin > d1.u_fin

    def test_limits_match_L_over_n(self):
        clt = _make_5ply_C24()
        d = clt_deflection_check(
            clt, span=4.0, w_perm=1e3, w_var=1e3,
            span_over_limit_inst=300.0, span_over_limit_fin=250.0,
        )
        assert d.u_limit_inst == pytest.approx(4.0 / 300, rel=1e-12)
        assert d.u_limit_fin == pytest.approx(4.0 / 250, rel=1e-12)

    def test_short_span_passes(self):
        """Very short span (2m) with modest load -> easily passes."""
        clt = _make_5ply_C24()
        d = clt_deflection_check(
            clt, span=2.0, w_perm=500, w_var=500,
        )
        assert d.passes_inst
        assert d.passes_fin


# ============================================================ vibration

class TestVibration:
    def test_short_span_passes_8hz(self):
        """2m span 100mm CLT: easily exceeds 8 Hz."""
        clt = _make_5ply_C24()
        v = clt_vibration_check(
            clt, span=2.0, additional_mass_per_area=0.0,
        )
        assert v.passes
        assert v.f_1 > 8.0

    def test_long_span_fails(self):
        """8m span 100mm CLT: should fail 8 Hz."""
        clt = _make_5ply_C24()
        v = clt_vibration_check(
            clt, span=8.0, additional_mass_per_area=30.0,
        )
        assert v.f_1 < 8.0
        assert not v.passes

    def test_formula(self):
        """f_1 = (pi/2) * sqrt(EI / (m L^4))"""
        clt = _make_5ply_C24()
        L = 4.0
        m_add = 30.0
        v = clt_vibration_check(
            clt, span=L, additional_mass_per_area=m_add,
            use_gamma_method=False,
        )
        EI = clt.EI_eff_per_width(strong_axis=True)
        m = clt.mass_per_area() + m_add
        expected = (math.pi / 2.0) * math.sqrt(EI / (m * L ** 4))
        assert v.f_1 == pytest.approx(expected, rel=1e-9)

    def test_added_mass_lowers_frequency(self):
        clt = _make_5ply_C24()
        v1 = clt_vibration_check(clt, span=4.0, additional_mass_per_area=0.0)
        v2 = clt_vibration_check(clt, span=4.0, additional_mass_per_area=100.0)
        assert v2.f_1 < v1.f_1

    def test_user_can_set_higher_threshold(self):
        """Some codes (FIB, residential offices) want 10 Hz."""
        clt = _make_5ply_C24()
        v = clt_vibration_check(
            clt, span=4.0, additional_mass_per_area=30.0,
            f_1_required=10.0,
        )
        # f_1 = 9.72 Hz from smoke test, just below 10
        assert v.f_1 < 10.0
        assert not v.passes

    def test_negative_span_raises(self):
        clt = _make_5ply_C24()
        with pytest.raises(ValueError):
            clt_vibration_check(clt, span=-1.0)
