"""Phase 53 tests -- Theme W slab / diaphragm / punching engineering."""
from __future__ import annotations

import math

import numpy as np
import pytest

from femsolver.design.diaphragm import (
    classify_diaphragm,
    flexible_transfer,
    rigid_transfer,
)
from femsolver.design.punching import (
    aci318_punching_capacity,
    aci318_punching_demand,
    eurocode_punching_capacity,
    is456_punching_capacity,
)
from femsolver.design.two_way_slab import (
    ddm_minimum_thickness,
    ddm_panel,
)


# ============================================================ ACI 318 punching

class TestACI318Punching:
    def test_interior_square_governs_by_eq_a(self):
        # 400x400 square column, d = 200 mm, f_c' = 30 MPa
        res = aci318_punching_capacity(
            c_x=0.4, c_y=0.4, d=0.2, f_c=30e6, position="interior",
        )
        # v_c = (1/3) * sqrt(30) * lambda_s (for size effect with d=200mm)
        # lambda_s = sqrt(2 / (1 + 0.004*200)) = sqrt(2/1.8) = 1.054 -> capped at 1.0
        expected_v_c = (1.0 / 3.0) * math.sqrt(30.0) * 1.0e6
        assert res.v_c == pytest.approx(expected_v_c, rel=1e-9)
        # b_0 = 2 * (400+200) + 2 * (400+200) = 2400 mm
        assert res.b_0 == pytest.approx(2.4, rel=1e-9)
        assert "22.6.5.2(a)" in res.notes

    def test_critical_perimeter_edge(self):
        # Edge column: three sides active (one side at slab edge)
        res = aci318_punching_capacity(
            c_x=0.4, c_y=0.4, d=0.2, f_c=30e6, position="edge",
        )
        # b_0 = 2 * (c_x + d) + (c_y + d) = 2*0.6 + 0.6 = 1.8 m
        assert res.b_0 == pytest.approx(1.8, rel=1e-9)

    def test_critical_perimeter_corner(self):
        res = aci318_punching_capacity(
            c_x=0.4, c_y=0.4, d=0.2, f_c=30e6, position="corner",
        )
        # b_0 = (c_x + d) + (c_y + d) = 1.2 m
        assert res.b_0 == pytest.approx(1.2, rel=1e-9)

    def test_long_thin_column_triggers_eq_b(self):
        # beta = 4.0 -> equation (b) (1/6)*(1 + 2/beta) = (1/6)*1.5 = 0.25
        # vs equation (a) = 1/3 = 0.333  -> (b) governs
        res = aci318_punching_capacity(
            c_x=0.8, c_y=0.2, d=0.2, f_c=30e6, position="interior",
        )
        assert "(b)" in res.notes

    def test_size_effect_kicks_in_for_deep_slab(self):
        # d = 1000 mm -> lambda_s = sqrt(2/5) = 0.632
        res = aci318_punching_capacity(
            c_x=0.4, c_y=0.4, d=1.0, f_c=30e6, position="interior",
        )
        # Capacity stress should be reduced by lambda_s ~ 0.632
        v_c_no_size = (1.0/3.0) * math.sqrt(30.0) * 1.0e6
        assert res.v_c < 0.7 * v_c_no_size
        assert res.v_c > 0.6 * v_c_no_size

    def test_demand_with_unbalanced_moment_higher(self):
        v_pure = aci318_punching_demand(
            V_u=400e3, c_x=0.4, c_y=0.4, d=0.2, position="interior",
        )
        v_combo = aci318_punching_demand(
            V_u=400e3, M_unb=80e3,
            c_x=0.4, c_y=0.4, d=0.2, position="interior",
        )
        assert v_combo > v_pure


# ============================================================ EC2 + IS 456

class TestEC2Punching:
    def test_basic_capacity(self):
        res = eurocode_punching_capacity(
            c_x=0.4, c_y=0.4, d=0.2, f_ck=30e6, rho_l=0.01,
            position="interior",
        )
        # v_Rd,c = (0.18/1.5) * k * (100*0.01*30)^(1/3)
        # k = 1 + sqrt(200/200) = 2.0
        # = 0.12 * 2.0 * 30^(1/3) = 0.12 * 2.0 * 3.107 = 0.7458 MPa
        assert res.v_c / 1e6 == pytest.approx(0.7458, rel=1e-3)

    def test_rho_l_capped(self):
        # rho_l above 0.02 should be capped
        res_high = eurocode_punching_capacity(
            c_x=0.4, c_y=0.4, d=0.2, f_ck=30e6, rho_l=0.05,
        )
        res_cap = eurocode_punching_capacity(
            c_x=0.4, c_y=0.4, d=0.2, f_ck=30e6, rho_l=0.02,
        )
        assert res_high.v_c == pytest.approx(res_cap.v_c, rel=1e-9)

    def test_k_capped_at_2(self):
        # Very thin slab d=50 mm: 1 + sqrt(200/50) = 1 + 2 = 3.0, capped at 2.0
        res = eurocode_punching_capacity(
            c_x=0.4, c_y=0.4, d=0.05, f_ck=30e6,
        )
        assert "k = 2.000" in res.notes


class TestIS456Punching:
    def test_square_column_ks_one(self):
        # beta_c = 1.0 -> k_s = min(0.5 + 1.0, 1.0) = 1.0
        res = is456_punching_capacity(
            c_x=0.4, c_y=0.4, d=0.2, f_ck=30e6,
        )
        # tau_c = 1.0 * 0.25 * sqrt(30) = 1.369 MPa
        assert res.v_c / 1e6 == pytest.approx(1.369, rel=1e-3)

    def test_thin_column_reduces_ks(self):
        # beta_c = 200/800 = 0.25 -> k_s = 0.75
        res = is456_punching_capacity(
            c_x=0.2, c_y=0.8, d=0.2, f_ck=30e6,
        )
        # tau_c = 0.75 * 0.25 * sqrt(30) = 1.027 MPa
        assert res.v_c / 1e6 == pytest.approx(1.027, rel=1e-3)


# ============================================================ DDM

class TestDDM:
    def test_static_moment_formula(self):
        # w_u = 10 kPa, l_long = 8 m, l_short = 6 m, col 0.5 m
        # l_n = 8 - 0.5 = 7.5
        # M_o = 10000 * 6 * 7.5^2 / 8 = 42188 N.m
        res = ddm_panel(
            w_u=10e3, l_long=8.0, l_short=6.0,
            direction="long", col_size=0.5,
        )
        assert res.M_o == pytest.approx(10e3 * 6.0 * 7.5**2 / 8.0)

    def test_interior_split_factors(self):
        res = ddm_panel(
            w_u=10e3, l_long=6.0, l_short=6.0, col_size=0.4,
        )
        # Interior: negative 0.65, positive 0.35
        assert res.M_neg_int == pytest.approx(0.65 * res.M_o)
        assert res.M_pos_int == pytest.approx(0.35 * res.M_o)
        # Strip split at interior support: 75 / 25 for negative
        assert res.M_col_strip_neg_int == pytest.approx(0.75 * res.M_neg_int)
        assert res.M_mid_strip_neg_int == pytest.approx(0.25 * res.M_neg_int)

    def test_min_thickness_grade60(self):
        # f_y = 420 MPa, interior panel: h_min = l_n / 33 * (0.4 + 420/700)
        # = l_n / 33 * 1.0 = l_n / 33
        h = ddm_minimum_thickness(l_n=6.0, f_y=420e6, interior_panel=True)
        assert h == pytest.approx(6.0 / 33.0, rel=1e-9)

    def test_min_thickness_grade80_thicker(self):
        # higher f_y -> thicker slab (steel can take more strain so
        # deflection rises -> need more concrete depth)
        h60 = ddm_minimum_thickness(l_n=6.0, f_y=420e6)
        h80 = ddm_minimum_thickness(l_n=6.0, f_y=550e6)
        assert h80 > h60

    def test_rejects_invalid(self):
        with pytest.raises(ValueError, match="w_u"):
            ddm_panel(w_u=-1, l_long=6, l_short=6)
        with pytest.raises(ValueError, match="direction"):
            ddm_panel(w_u=1e3, l_long=6, l_short=6, direction="diagonal")
        with pytest.raises(ValueError, match="col_size"):
            ddm_panel(w_u=1e3, l_long=6, l_short=6, col_size=10.0)


# ============================================================ diaphragm

class TestDiaphragmClassification:
    def test_concrete_low_span_depth_auto_rigid(self):
        assert classify_diaphragm(
            delta_d=0.10, delta_drift_avg=0.001,
            span_over_depth=2.5, material="concrete",
        ) == "rigid"

    def test_flexible_when_ratio_ge_2(self):
        # delta_d / drift = 2 -> flexible
        assert classify_diaphragm(
            delta_d=0.010, delta_drift_avg=0.005,
        ) == "flexible"

    def test_semi_rigid_when_ratio_one(self):
        assert classify_diaphragm(
            delta_d=0.005, delta_drift_avg=0.005,
        ) == "semi_rigid"

    def test_rigid_when_ratio_low(self):
        # delta_d / drift = 0.3 -> rigid
        assert classify_diaphragm(
            delta_d=0.003, delta_drift_avg=0.010,
        ) == "rigid"


class TestFlexibleTransfer:
    def test_three_walls_equally_spaced(self):
        # 3 walls at 0, 10, 20 m -> tributary 5, 10, 5
        shares = flexible_transfer(
            F_total=1000.0,
            elements=[("A", 0.0, 1.0), ("B", 10.0, 1.0), ("C", 20.0, 1.0)],
        )
        # Tributary fractions: 5/20, 10/20, 5/20 = 25%, 50%, 25%
        ids = {s.element_id: s.force for s in shares}
        assert ids["A"] == pytest.approx(250.0, rel=1e-9)
        assert ids["B"] == pytest.approx(500.0, rel=1e-9)
        assert ids["C"] == pytest.approx(250.0, rel=1e-9)

    def test_total_force_preserved(self):
        shares = flexible_transfer(
            F_total=1000.0,
            elements=[("A", 0.0, 1.0), ("B", 7.0, 1.0), ("C", 20.0, 1.0)],
        )
        assert sum(s.force for s in shares) == pytest.approx(1000.0, rel=1e-9)


class TestRigidTransfer:
    def test_centered_F_no_torsion(self):
        # Walls at 0, 10, 20 m, all K=1; F at x=10
        shares, e_x = rigid_transfer(
            F_total=1000.0,
            elements=[("A", 0.0, 1.0), ("B", 10.0, 1.0), ("C", 20.0, 1.0)],
            F_position=10.0,
        )
        assert e_x == pytest.approx(0.0, abs=1e-9)
        # Equal stiffness, no torsion -> each wall gets 1/3
        for s in shares:
            assert s.F_direct == pytest.approx(1000.0 / 3.0, rel=1e-9)
            assert abs(s.F_torsion) < 1e-9

    def test_eccentric_F_adds_torsion(self):
        shares, e_x = rigid_transfer(
            F_total=1000.0,
            elements=[("A", 0.0, 1.0), ("B", 10.0, 1.0), ("C", 20.0, 1.0)],
            F_position=15.0,
        )
        assert e_x == pytest.approx(5.0)
        # Total force conserved
        total = sum(s.F_total for s in shares)
        assert total == pytest.approx(1000.0, rel=1e-9)
        # A (farther from F) gets less; C (closer) gets more
        ids = {s.element_id: s.F_total for s in shares}
        assert ids["C"] > ids["A"]

    def test_centre_of_rigidity(self):
        # Stiffness 1, 2, 1 at x=0, 10, 20 -> CR at (0+20+20)/4 = 10
        shares, _ = rigid_transfer(
            F_total=1000.0,
            elements=[("A", 0.0, 1.0), ("B", 10.0, 2.0), ("C", 20.0, 1.0)],
            F_position=10.0,
        )
        # No torsion when F at CR
        for s in shares:
            assert abs(s.F_torsion) < 1e-9
        # Direct share by stiffness: 25% / 50% / 25%
        ids = {s.element_id: s.F_direct for s in shares}
        assert ids["A"] == pytest.approx(250.0)
        assert ids["B"] == pytest.approx(500.0)
        assert ids["C"] == pytest.approx(250.0)
