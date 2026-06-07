"""Phase HH.6 tests -- ASCE 7-22 Components-and-Cladding (C&C) wind."""
from __future__ import annotations

import math

import pytest

from femsolver.wind import (
    CCDesignPressure,
    CCPressureCoefficient,
    cc_design_pressure,
    cc_edge_distance,
    cc_roof_GCp,
    cc_wall_GCp,
    gcpi_for_enclosure,
)


# ============================================================ edge distance

class TestEdgeDistance:
    def test_formula(self):
        # B=20, h=10: min(0.1*20, 0.4*10) = min(2, 4) = 2.0
        a = cc_edge_distance(B=20.0, h=10.0)
        assert a == pytest.approx(2.0, rel=1e-9)

    def test_floor_at_3_ft(self):
        # Very small building -> floor at 3 ft = 0.9144 m
        a = cc_edge_distance(B=5.0, h=2.0)
        assert a >= 0.9144

    def test_min_4_percent_B(self):
        # h is very tall -> min(0.1*B, 0.4*h) might exceed 0.04*B floor.
        # Here we verify 0.04*B floor kicks in for shallow h cases.
        a = cc_edge_distance(B=50.0, h=4.0)
        # min(5, 1.6) = 1.6, but >= 0.04 * 50 = 2 -> a = 2
        assert a == pytest.approx(2.0, rel=1e-9)

    def test_rejects_invalid(self):
        with pytest.raises(ValueError):
            cc_edge_distance(B=-1, h=5)
        with pytest.raises(ValueError):
            cc_edge_distance(B=5, h=-1)


# ============================================================ wall GC_p

class TestWallGCp:
    def test_zone_5_at_small_area_matches_table(self):
        """ASCE 7-22 Fig 30.5-1: Zone 5 GC_p = +1.0 / -1.4 at A_e = 10 sf."""
        c = cc_wall_GCp(A_e=0.93, zone="wall_5")
        assert c.GC_p_pos == pytest.approx(1.0, abs=0.01)
        assert c.GC_p_neg == pytest.approx(-1.4, abs=0.01)

    def test_zone_4_smaller_magnitude_than_zone_5(self):
        c4 = cc_wall_GCp(A_e=0.93, zone="wall_4")
        c5 = cc_wall_GCp(A_e=0.93, zone="wall_5")
        # Zone 5 (corner) has higher suction magnitude
        assert abs(c5.GC_p_neg) > abs(c4.GC_p_neg)

    def test_GC_p_decreases_with_area(self):
        small = cc_wall_GCp(A_e=1.0, zone="wall_5")
        large = cc_wall_GCp(A_e=40.0, zone="wall_5")
        assert abs(large.GC_p_neg) < abs(small.GC_p_neg)
        assert large.GC_p_pos < small.GC_p_pos

    def test_GC_p_clamped_at_large_area(self):
        # Beyond 500 sf the value should equal the large-area asymptote
        c = cc_wall_GCp(A_e=200.0, zone="wall_5")
        c_max = cc_wall_GCp(A_e=46.5, zone="wall_5")
        assert c.GC_p_neg == pytest.approx(c_max.GC_p_neg, rel=1e-9)

    def test_rejects_invalid_zone(self):
        with pytest.raises(ValueError):
            cc_wall_GCp(A_e=1.0, zone="zone_X")

    def test_rejects_negative_area(self):
        with pytest.raises(ValueError):
            cc_wall_GCp(A_e=-1.0, zone="wall_5")


# ============================================================ roof GC_p

class TestRoofGCp:
    def test_zone_3_corner_largest_uplift(self):
        c1 = cc_roof_GCp(A_e=1.0, zone="roof_1")
        c2 = cc_roof_GCp(A_e=1.0, zone="roof_2")
        c3 = cc_roof_GCp(A_e=1.0, zone="roof_3")
        # Magnitude ordering: roof_3 > roof_2 > roof_1
        assert abs(c3.GC_p_neg) > abs(c2.GC_p_neg) > abs(c1.GC_p_neg)

    def test_zone_3_matches_table(self):
        """ASCE 7-22 Fig 30.5-1 flat roof Zone 3 at A_e = 10 sf: GC_p = -3.2."""
        c = cc_roof_GCp(A_e=0.93, zone="roof_3")
        assert c.GC_p_neg == pytest.approx(-3.2, abs=0.05)

    def test_only_flat_supported(self):
        with pytest.raises(ValueError, match="flat"):
            cc_roof_GCp(A_e=1.0, zone="roof_1", roof_type="gable")


# ============================================================ GC_pi enclosure

class TestEnclosure:
    def test_enclosed(self):
        assert gcpi_for_enclosure("enclosed") == 0.18

    def test_partially_enclosed(self):
        assert gcpi_for_enclosure("partially_enclosed") == 0.55

    def test_open(self):
        assert gcpi_for_enclosure("open") == 0.0

    def test_rejects_unknown(self):
        with pytest.raises(ValueError):
            gcpi_for_enclosure("airtight")


# ============================================================ design pressure

class TestDesignPressure:
    def test_combined_formula(self):
        """p_min = q_h * (GC_p_neg - GC_pi) for the worst-outward case."""
        c = cc_wall_GCp(A_e=1.0, zone="wall_5")
        q_h = 1.0e3        # 1 kPa for easy arithmetic
        p = cc_design_pressure(coeff=c, q_h=q_h, enclosure="enclosed")
        expected_min = q_h * (c.GC_p_neg - 0.18)
        assert p.p_min == pytest.approx(expected_min, rel=1e-9)
        expected_max = q_h * (c.GC_p_pos + 0.18)
        assert p.p_max == pytest.approx(expected_max, rel=1e-9)

    def test_partially_enclosed_higher_demand(self):
        c = cc_wall_GCp(A_e=1.0, zone="wall_5")
        p_e = cc_design_pressure(coeff=c, q_h=1e3, enclosure="enclosed")
        p_p = cc_design_pressure(
            coeff=c, q_h=1e3, enclosure="partially_enclosed",
        )
        # Partially enclosed -> more negative p_min (worse suction)
        assert p_p.p_min < p_e.p_min
        # And higher p_max
        assert p_p.p_max > p_e.p_max

    def test_rejects_invalid_q_h(self):
        c = cc_wall_GCp(A_e=1.0)
        with pytest.raises(ValueError):
            cc_design_pressure(coeff=c, q_h=-1.0)


class TestEngineeringScenarios:
    def test_window_panel_design(self):
        """Small wall panel (1 m^2) near a building corner at V=50 m/s,
        exposure C, h=10 m. The peak suction on the cladding should
        be ~2.5 kPa."""
        from femsolver.wind import asce7_velocity_pressure
        q_h = asce7_velocity_pressure(
            z=10.0, V=50.0, exposure="C",
        ).q_z
        c = cc_wall_GCp(A_e=1.0, zone="wall_5")
        p = cc_design_pressure(coeff=c, q_h=q_h, enclosure="enclosed")
        # In the 2-3 kPa range (typical for cladding design)
        assert 2.0e3 < abs(p.p_min) < 3.0e3

    def test_roof_corner_fastener_design(self):
        """Small fastener (0.93 m^2 tributary) at a flat-roof corner.
        The uplift at V=50 m/s, h=10 m should approach -5 kPa."""
        from femsolver.wind import asce7_velocity_pressure
        q_h = asce7_velocity_pressure(
            z=10.0, V=50.0, exposure="C",
        ).q_z
        c = cc_roof_GCp(A_e=0.93, zone="roof_3")
        p = cc_design_pressure(coeff=c, q_h=q_h)
        # 4-6 kPa uplift range
        assert -6.0e3 < p.p_min < -4.0e3
