"""Phase 41 tests -- Winkler beam, pile groups, liquefaction,
dynamic Gazetas impedance.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from femsolver import (
    BeamOnWinklerFoundation2D,
    CRR_from_N1_60cs,
    ElasticIsotropic,
    HalfspaceSoil,
    K_sigma,
    LinearStaticAnalysis,
    Model,
    cyclic_stress_ratio,
    dimensionless_frequency,
    dynamic_footing_impedance,
    evaluate_liquefaction,
    fines_content_correction,
    gazetas_dynamic_coefficients,
    gazetas_surface_footing,
    group_efficiency_converse_labarre,
    group_p_multipliers,
    group_settlement_elastic,
    hetenyi_characteristic_length,
    hetenyi_infinite_beam_point_load,
    magnitude_scaling_factor,
    p_multiplier,
    stress_reduction_coefficient,
    subgrade_modulus_table,
)


# ============================================================ Winkler

class TestHetenyiClosedForm:
    def test_characteristic_length_formula(self):
        # L_c = (4 E I / (k_s b))^0.25
        E, I, k_s, b = 2e11, 1e-4, 50e6, 0.5
        L_c = hetenyi_characteristic_length(E=E, I=I, k_s=k_s, b=b)
        expected = (4 * E * I / (k_s * b)) ** 0.25
        assert L_c == pytest.approx(expected, rel=1e-12)

    def test_w_max_M_max_relations(self):
        # w(0) = P beta / (2 k_s b),  M(0) = P / (4 beta)
        E, I, k_s, b, P = 2e11, 1e-4, 50e6, 0.5, 10000.0
        res = hetenyi_infinite_beam_point_load(P=P, E=E, I=I, k_s=k_s, b=b)
        beta = 1.0 / res.L_c
        assert res.w_max == pytest.approx(P * beta / (2 * k_s * b), rel=1e-12)
        assert res.M_max == pytest.approx(P / (4 * beta), rel=1e-12)

    def test_validates_positive(self):
        with pytest.raises(ValueError):
            hetenyi_characteristic_length(E=-1, I=1e-4, k_s=50e6, b=0.5)


class TestWinklerBeamElement:
    def test_finite_beam_under_point_load_matches_hetenyi(self):
        """Long beam (L > 5*L_c) on Winkler bed under midpoint
        point load should match the Hetenyi infinite-beam reference
        within ~10% (finite-length end effect)."""
        E, I, k_s, b = 2.0e11, 1.0e-4, 50e6, 0.5
        L_c = hetenyi_characteristic_length(E=E, I=I, k_s=k_s, b=b)
        # Use a 6-L_c long beam so the ends are far enough away
        L = 6.0 * L_c
        n_elem = 30
        P = 10000.0

        mat = ElasticIsotropic(1, E=E, nu=0.3, rho=0.0)
        m = Model(ndm=2, ndf=3)
        m.add_material(mat)
        for i in range(n_elem + 1):
            m.add_node(i + 1, i * L / n_elem, 0.0)
        for i in range(n_elem):
            m.add_element(BeamOnWinklerFoundation2D(
                i + 1, (i + 1, i + 2), mat,
                area=0.01, Iz=I,
                k_s=k_s, b=b,
            ))
        # Constrain just rigid-body axial motion (left end u=0,
        # midpoint loaded transversely)
        mid_node = n_elem // 2 + 1
        m.fix(1, [1, 0, 0])
        m.add_nodal_load(mid_node, [0.0, -P, 0.0])
        LinearStaticAnalysis(m).run()
        w_mid = -m.node(mid_node).disp[1]    # downward deflection
        # Compare to Hetenyi
        ref = hetenyi_infinite_beam_point_load(
            P=P, E=E, I=I, k_s=k_s, b=b,
        )
        # Allow 15% (end-effect, mesh discretization, no free-end
        # rotation constraint)
        assert w_mid == pytest.approx(ref.w_max, rel=0.15)

    def test_validates_inputs(self):
        mat = ElasticIsotropic(1, E=2e11, nu=0.3, rho=0.0)
        with pytest.raises(ValueError, match="k_s"):
            BeamOnWinklerFoundation2D(
                1, (1, 2), mat, area=0.01, Iz=1e-4,
                k_s=-1, b=0.5,
            )


class TestSubgradeModulusTable:
    def test_known_soil_types(self):
        for soil in ("loose_sand", "dense_sand", "hard_clay", "sound_rock"):
            lo, hi = subgrade_modulus_table(soil)
            assert lo > 0 and hi > lo

    def test_unknown_soil_raises(self):
        with pytest.raises(ValueError, match="unknown"):
            subgrade_modulus_table("martian_regolith")


# ============================================================ pile group

class TestPileGroup:
    def test_p_mult_at_5D_no_reduction_for_leading(self):
        assert p_multiplier(row=1, s_over_D=5.0) == pytest.approx(1.0)

    def test_p_mult_at_3D_minimum_values(self):
        assert p_multiplier(row=1, s_over_D=3.0) == pytest.approx(0.7)
        assert p_multiplier(row=2, s_over_D=3.0) == pytest.approx(0.5)
        assert p_multiplier(row=3, s_over_D=3.0) == pytest.approx(0.35)

    def test_p_mult_interpolation_at_4D(self):
        # Linear interpolation between (3, 0.7) and (5, 1.0) -> 0.85 at s/D=4
        assert p_multiplier(row=1, s_over_D=4.0) == pytest.approx(0.85)

    def test_p_mult_clamped_below_3D(self):
        # At s/D=2 (very tight), clamp to s/D=3 value
        assert p_multiplier(row=1, s_over_D=2.0) == pytest.approx(0.7)

    def test_group_p_multipliers_shape(self):
        fm = group_p_multipliers(
            n_rows=3, n_cols=4, s_x=2.0, s_y=2.0, D=0.5,
        )
        assert fm.shape == (4, 3)
        # Front column has highest mults; back column lowest
        assert (fm[0, :] >= fm[-1, :]).all()

    def test_group_efficiency_in_range(self):
        E_g = group_efficiency_converse_labarre(
            n_rows=3, n_cols=3, s_x=2.0, s_y=2.0, D=0.5,
        )
        assert 0.6 < E_g < 1.0

    def test_group_settlement_R_s_sqrt(self):
        # R_s = sqrt(B_g / D); s_g = R_s * s_single
        res = group_settlement_elastic(
            P_group=1.0e6, n_piles=4,
            s_single_per_unit_load=1.0e-9,
            B_g=4.0, D=0.5,
        )
        expected_R_s = math.sqrt(4.0 / 0.5)
        assert res.R_s == pytest.approx(expected_R_s, rel=1e-12)


# ============================================================ liquefaction

class TestLiquefaction:
    def test_MSF_decreases_with_M(self):
        msf_6 = magnitude_scaling_factor(6.0)
        msf_7p5 = magnitude_scaling_factor(7.5)
        msf_8p5 = magnitude_scaling_factor(8.5)
        assert msf_6 > msf_7p5 > msf_8p5    # smaller M -> larger MSF

    def test_MSF_at_M_7p5_near_one(self):
        msf = magnitude_scaling_factor(7.5)
        assert msf == pytest.approx(1.0, abs=0.05)

    def test_K_sigma_one_at_atmospheric(self):
        # At sigma_v_eff = 101.3 kPa: log(1) = 0 -> K_sigma = 1
        K_s = K_sigma(sigma_v_eff=101.3e3, N1_60cs=15.0)
        assert K_s == pytest.approx(1.0, abs=1e-9)

    def test_K_sigma_decreases_at_high_stress(self):
        K_s = K_sigma(sigma_v_eff=400.0e3, N1_60cs=15.0)
        assert K_s < 1.0    # high stress -> smaller K_sigma

    def test_CRR_finite_for_loose_sand(self):
        # N1_60cs = 12 -> small CRR (loose)
        crr = CRR_from_N1_60cs(12.0)
        assert 0.0 < crr < 0.5

    def test_CRR_clamps_at_high_N(self):
        # N1_60cs >= 37 -> CRR plateau (very high resistance)
        crr_37 = CRR_from_N1_60cs(37.0)
        crr_50 = CRR_from_N1_60cs(50.0)
        assert crr_37 == pytest.approx(crr_50, rel=1e-12)

    def test_FS_high_for_dense_sand(self):
        # Dense sand: N_60 = 30, low PGA, low water table -> high FS
        res = evaluate_liquefaction(
            z=3.0, M=6.0, a_max_g=0.15,
            sigma_v_total=3.0 * 19000,
            sigma_v_eff=3.0 * 19000,    # water table below 3 m
            N_60=30.0,
        )
        assert res.FS > 2.0
        assert not res.liquefies

    def test_FS_low_for_loose_sand(self):
        # Loose sand below water, strong shaking -> liquefies
        res = evaluate_liquefaction(
            z=5.0, M=7.5, a_max_g=0.3,
            sigma_v_total=5.0 * 19000,
            sigma_v_eff=5.0 * (19000 - 9810),
            N_60=10.0, FC_percent=5.0,
        )
        assert res.FS < 1.0
        assert res.liquefies


# ============================================================ dynamic Gazetas

class TestDynamicGazetas:
    def test_a_0_formula(self):
        a_0 = dimensionless_frequency(omega=10.0, B=2.0, V_s=200.0)
        assert a_0 == pytest.approx(10.0 * 2.0 / 200.0, rel=1e-12)

    def test_static_limit(self):
        # At omega = 0 (a_0 = 0): k = 1, c = 0
        coef = gazetas_dynamic_coefficients(a_0=0.0, L_over_B=1.0)
        assert coef.k_z == pytest.approx(1.0, abs=1e-12)
        assert coef.c_z == pytest.approx(0.0, abs=1e-12)
        assert coef.k_rx == pytest.approx(1.0, abs=1e-12)
        assert coef.c_t == pytest.approx(0.0, abs=1e-12)

    def test_higher_frequency_more_damping(self):
        coef_lo = gazetas_dynamic_coefficients(a_0=0.5, L_over_B=2.0)
        coef_hi = gazetas_dynamic_coefficients(a_0=1.5, L_over_B=2.0)
        assert coef_hi.c_z > coef_lo.c_z
        assert coef_hi.c_y > coef_lo.c_y

    def test_dimensional_dynamic_impedance(self):
        soil = HalfspaceSoil(G=50e6, nu=0.35, rho=1900.0)
        imp_static = gazetas_surface_footing(soil, B=2.0, L=3.0)
        # At omega = 0+, dynamic should equal static
        imp_dyn = dynamic_footing_impedance(
            static_impedance=imp_static, soil=soil, omega=1.0e-6,
        )
        assert imp_dyn.K_z == pytest.approx(imp_static.K_z, rel=1e-6)
        assert imp_dyn.C_z == pytest.approx(0.0, abs=1.0)

    def test_validates_inputs(self):
        with pytest.raises(ValueError, match="L_over_B"):
            gazetas_dynamic_coefficients(a_0=0.5, L_over_B=0.5)
