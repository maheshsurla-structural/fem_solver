"""Phase 30.1 tests -- AISC Shapes Database and steel material grades.

Verifies unit conversions, lookup behaviour, and that the embedded
properties round-trip back to the AISC v15.0 tabulated imperial
values within a tight tolerance (machine-precision for direct
imperial-to-SI conversion).
"""
from __future__ import annotations

import math

import pytest

from femsolver.design.steel import (
    SteelMaterial,
    SteelSection,
    all_designations,
    astm_a36,
    astm_a992,
    get_section,
    w_series,
)


IN = 0.0254
IN2 = IN * IN
IN3 = IN2 * IN
IN4 = IN3 * IN
PSI = 6894.757


# ============================================================ database basics

def test_database_has_curated_selection():
    """At least 30 W-shapes spanning W4 through W36 should be present."""
    designations = all_designations()
    assert len(designations) >= 30
    # Spans W4 to W36
    depths = {int(d.split("x")[0][1:]) for d in designations}
    assert min(depths) == 4
    assert max(depths) >= 36


def test_designations_sorted_by_depth_then_weight():
    """all_designations() returns sections sorted W4 → W36, lighter
    first within each depth."""
    designations = all_designations()
    prev = (0, 0)
    for d in designations:
        depth = int(d.split("x")[0][1:])
        weight = int(d.split("x")[1])
        assert (depth, weight) >= prev, (
            f"{d} out of order vs previous"
        )
        prev = (depth, weight)


# ============================================================ lookup

def test_get_section_returns_steel_section():
    s = get_section("W14x90")
    assert isinstance(s, SteelSection)
    assert s.designation == "W14x90"


def test_get_section_raises_on_unknown():
    with pytest.raises(KeyError):
        get_section("W42x999")


def test_get_section_suggests_alternatives_in_message():
    """When the designation is wrong but the series exists, the error
    message should list similar designations."""
    with pytest.raises(KeyError, match="W14"):
        get_section("W14x999")


# ============================================================ unit conversions

def test_W14x90_properties_match_aisc_table():
    """Verify W14x90 round-trips against the AISC v15.0 tabulated
    imperial values.

    Per AISC Manual 15th ed. Table 1-1:
      A = 26.5 in², d = 14.02 in, bf = 14.52 in, tf = 0.710 in,
      tw = 0.440 in, Ix = 999 in⁴, Iy = 362 in⁴, Sx = 143 in³,
      Zx = 157 in³, rx = 6.14 in, ry = 3.70 in, J = 4.06 in⁴,
      Cw = 16000 in⁶.
    """
    s = get_section("W14x90")
    assert s.A == pytest.approx(26.5 * IN2, rel=1e-12)
    assert s.d == pytest.approx(14.02 * IN, rel=1e-12)
    assert s.bf == pytest.approx(14.52 * IN, rel=1e-12)
    assert s.tf == pytest.approx(0.710 * IN, rel=1e-12)
    assert s.tw == pytest.approx(0.440 * IN, rel=1e-12)
    assert s.Ix == pytest.approx(999.0 * IN4, rel=1e-12)
    assert s.Iy == pytest.approx(362.0 * IN4, rel=1e-12)
    assert s.Sx == pytest.approx(143.0 * IN3, rel=1e-12)
    assert s.Zx == pytest.approx(157.0 * IN3, rel=1e-12)
    assert s.rx == pytest.approx(6.14 * IN, rel=1e-12)
    assert s.ry == pytest.approx(3.70 * IN, rel=1e-12)
    assert s.J == pytest.approx(4.06 * IN4, rel=1e-12)


def test_radius_of_gyration_consistency():
    """rx = sqrt(Ix / A); same for ry."""
    for designation in ("W14x90", "W12x65", "W24x84"):
        s = get_section(designation)
        # AISC tabulated r is rounded to 3 digits; we allow 0.5% slack
        rx_computed = math.sqrt(s.Ix / s.A)
        ry_computed = math.sqrt(s.Iy / s.A)
        assert s.rx == pytest.approx(rx_computed, rel=5e-3)
        assert s.ry == pytest.approx(ry_computed, rel=5e-3)


def test_section_moduli_consistency():
    """For symmetric I-shapes, Sx ≈ Ix / (d/2)."""
    for designation in ("W14x90", "W12x65", "W24x84"):
        s = get_section(designation)
        Sx_computed = s.Ix / (s.d / 2.0)
        # Allow ~3% slack for fillet/tapered-flange refinements in the
        # AISC tabulated values
        assert s.Sx == pytest.approx(Sx_computed, rel=3e-2)


def test_plastic_modulus_exceeds_elastic_modulus():
    """Zx > Sx for any W-shape (plastic shape factor > 1)."""
    for designation in all_designations():
        s = get_section(designation)
        assert s.Zx > s.Sx


def test_weight_derived_from_area_and_density():
    """weight_per_length = A · ρ_steel · g (with default density 7850 kg/m³)."""
    s = get_section("W14x90")
    expected_N_per_m = s.A * 7850.0 * 9.81
    # AISC W14x90 weighs 90 lb/ft; convert to N/m for sanity
    # 90 lb/ft * 14.5939 N/m per lb/ft = 1313 N/m
    assert s.weight_per_length == pytest.approx(expected_N_per_m, rel=1e-12)
    # Confirm physical reasonableness vs AISC nominal
    aisc_lb_per_ft = 90.0
    N_per_m_from_imperial = aisc_lb_per_ft * 14.5939
    # Within 1% of imperial nominal (since A is rounded)
    assert s.weight_per_length == pytest.approx(N_per_m_from_imperial, rel=0.01)


# ============================================================ w_series

def test_w_series_returns_only_matching_depth():
    sections = w_series("W14")
    assert len(sections) >= 2
    for s in sections:
        assert s.designation.startswith("W14x")


def test_w_series_sorted_lightest_first():
    sections = w_series("W14")
    for i in range(len(sections) - 1):
        assert sections[i].A <= sections[i + 1].A


def test_w_series_empty_for_unknown_prefix():
    assert w_series("W99") == []


# ============================================================ SteelMaterial

def test_astm_a992_values():
    mat = astm_a992()
    # Fy = 50 ksi = 344.74 MPa
    assert mat.Fy == pytest.approx(50.0e3 * PSI, rel=1e-12)
    assert mat.Fu == pytest.approx(65.0e3 * PSI, rel=1e-12)
    assert mat.E == pytest.approx(200.0e9)
    assert mat.G == pytest.approx(77.2e9)


def test_astm_a36_values():
    mat = astm_a36()
    assert mat.Fy == pytest.approx(36.0e3 * PSI, rel=1e-12)
    assert mat.Fu == pytest.approx(58.0e3 * PSI, rel=1e-12)


def test_material_validation_rejects_Fu_less_than_Fy():
    with pytest.raises(ValueError, match="Fu"):
        SteelMaterial(Fy=400e6, Fu=200e6)


def test_material_validation_rejects_negative_Fy():
    with pytest.raises(ValueError, match="Fy must be positive"):
        SteelMaterial(Fy=-1, Fu=100e6)
