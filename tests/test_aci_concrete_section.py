"""Phase 29.1 tests -- ConcreteSection / RebarLayout / ACI 318-19
material constants and dataclasses."""
from __future__ import annotations

import math

import pytest

from femsolver.design.concrete import (
    EPSILON_CU,
    E_STEEL,
    ConcreteMaterial,
    ConcreteSection,
    PhiFactors,
    RebarLayout,
    beta_1_aci,
    phi_for_strain,
    rebar_area,
    rebar_diameter,
    standard_rebar_designations,
)


# ============================================================ rebar table

def test_rebar_table_has_standard_sizes():
    """All US-customary standard rebar sizes #3 through #18 are present."""
    sizes = standard_rebar_designations()
    for label in ("#3", "#4", "#5", "#6", "#7", "#8", "#9", "#10",
                  "#11", "#14", "#18"):
        assert label in sizes


def test_rebar_diameters_match_aci_table():
    """ACI 318 Appendix A nominal diameters: #4 = 0.500 in, #8 = 1.0 in."""
    assert rebar_diameter("#4") == pytest.approx(0.500 * 0.0254, rel=1e-12)
    assert rebar_diameter("#8") == pytest.approx(1.000 * 0.0254, rel=1e-12)
    assert rebar_diameter("#11") == pytest.approx(1.410 * 0.0254, rel=1e-12)


def test_rebar_areas_match_aci_table():
    """ACI 318 Appendix A nominal areas: #4 = 0.20 in², #8 = 0.79 in²."""
    in2 = 0.0254 ** 2
    assert rebar_area("#4") == pytest.approx(0.20 * in2, rel=1e-12)
    assert rebar_area("#8") == pytest.approx(0.79 * in2, rel=1e-12)
    assert rebar_area("#11") == pytest.approx(1.56 * in2, rel=1e-12)


def test_rebar_lookup_rejects_unknown():
    with pytest.raises(ValueError, match="unknown rebar"):
        rebar_diameter("#42")
    with pytest.raises(ValueError, match="unknown rebar"):
        rebar_area("D25")     # SI bar designations -- not in US table


# ============================================================ ACI material constants

def test_beta_1_low_strength():
    """β_1 = 0.85 for f_c' <= 4000 psi (~27.6 MPa)."""
    assert beta_1_aci(20e6) == pytest.approx(0.85)
    assert beta_1_aci(27.0e6) == pytest.approx(0.85)


def test_beta_1_high_strength():
    """β_1 = 0.65 for f_c' >= 8000 psi (~55.2 MPa)."""
    assert beta_1_aci(56e6) == pytest.approx(0.65)
    assert beta_1_aci(80e6) == pytest.approx(0.65)


def test_beta_1_interpolation():
    """β_1 = 0.85 - 0.05·(f_c'[psi] - 4000)/1000 between."""
    # At 5000 psi (~34.5 MPa): β_1 = 0.85 - 0.05 = 0.80
    fc_5000psi = 5000 * 6894.757
    assert beta_1_aci(fc_5000psi) == pytest.approx(0.80, rel=1e-6)
    # At 6000 psi: β_1 = 0.75
    fc_6000psi = 6000 * 6894.757
    assert beta_1_aci(fc_6000psi) == pytest.approx(0.75, rel=1e-6)


def test_phi_compression_controlled():
    """ε_t <= ε_ty -> φ = 0.65 (tied) or 0.75 (spiral)."""
    eps_ty = 0.00207     # Grade 60 ε_ty
    assert phi_for_strain(0.001, epsilon_ty=eps_ty) == pytest.approx(0.65)
    assert phi_for_strain(0.001, epsilon_ty=eps_ty,
                            spiral=True) == pytest.approx(0.75)


def test_phi_tension_controlled():
    """ε_t >= 0.005 -> φ = 0.90."""
    assert phi_for_strain(0.005) == pytest.approx(0.90)
    assert phi_for_strain(0.010) == pytest.approx(0.90)


def test_phi_transition_zone_linear():
    """Linear interpolation in the transition zone."""
    eps_ty = 0.002
    # Midway: φ should be (0.65 + 0.90)/2 = 0.775
    eps_mid = 0.5 * (eps_ty + 0.005)
    assert phi_for_strain(eps_mid, epsilon_ty=eps_ty) == pytest.approx(
        0.5 * (0.65 + 0.90), rel=1e-6
    )


# ============================================================ ConcreteMaterial

def test_concrete_material_validation():
    with pytest.raises(ValueError, match="fc_prime"):
        ConcreteMaterial(fc_prime=-1, fy=1)
    with pytest.raises(ValueError, match="fy"):
        ConcreteMaterial(fc_prime=1, fy=-1)


def test_concrete_material_default_Ec():
    """Default E_c = 4700 · sqrt(f_c'[MPa]) MPa per ACI 19.2.2.1."""
    mat = ConcreteMaterial(fc_prime=28.0e6, fy=420.0e6)
    Ec_expected = 4700.0 * math.sqrt(28.0) * 1.0e6     # Pa
    assert mat.Ec == pytest.approx(Ec_expected, rel=1e-6)


def test_concrete_material_user_Ec_overrides():
    custom_Ec = 30.0e9
    mat = ConcreteMaterial(fc_prime=28e6, fy=420e6, Ec=custom_Ec)
    assert mat.Ec == pytest.approx(custom_Ec)


def test_concrete_material_beta_1_and_epsilon_ty():
    mat = ConcreteMaterial(fc_prime=28e6, fy=420e6)
    assert mat.beta_1 == pytest.approx(beta_1_aci(28e6))
    assert mat.epsilon_ty == pytest.approx(420e6 / E_STEEL)


# ============================================================ RebarLayout

def test_rebar_layout_areas():
    layout = RebarLayout(
        bottom_bars=("#8", "#8", "#8"),
        bottom_cover=0.040,
        top_bars=("#6", "#6"),
    )
    in2 = 0.0254 ** 2
    assert layout.As_bottom == pytest.approx(3 * 0.79 * in2, rel=1e-12)
    assert layout.As_top == pytest.approx(2 * 0.44 * in2, rel=1e-12)


def test_rebar_layout_default_top_cover_matches_bottom():
    layout = RebarLayout(bottom_bars=("#8",), bottom_cover=0.040)
    assert layout.top_cover == pytest.approx(0.040)


def test_rebar_layout_validates_cover():
    with pytest.raises(ValueError, match="bottom_cover"):
        RebarLayout(bottom_bars=("#5",), bottom_cover=-0.01)


def test_rebar_layout_av_with_two_legs():
    layout = RebarLayout(
        bottom_bars=("#5",), bottom_cover=0.040,
        stirrup_designation="#3", stirrup_spacing=0.150, stirrup_legs=2,
    )
    in2 = 0.0254 ** 2
    assert layout.Av == pytest.approx(2 * 0.11 * in2, rel=1e-12)


# ============================================================ ConcreteSection

def _ts_section(**kwargs):
    """Build a baseline 300x500 section with 4 #8 bars."""
    mat = ConcreteMaterial(fc_prime=28e6, fy=420e6)
    rebar = RebarLayout(
        bottom_bars=("#8", "#8", "#8", "#8"),
        bottom_cover=0.050,
    )
    return ConcreteSection(b=0.30, h=0.50, material=mat, rebar=rebar)


def test_concrete_section_validation():
    mat = ConcreteMaterial(fc_prime=28e6, fy=420e6)
    rebar = RebarLayout(bottom_bars=("#5",), bottom_cover=0.040)
    with pytest.raises(ValueError, match="b must be"):
        ConcreteSection(b=-0.1, h=0.5, material=mat, rebar=rebar)
    with pytest.raises(ValueError, match="h must be"):
        ConcreteSection(b=0.3, h=0.0, material=mat, rebar=rebar)
    # Cover deeper than h
    rebar_bad = RebarLayout(bottom_bars=("#5",), bottom_cover=0.6)
    with pytest.raises(ValueError, match="bottom_cover"):
        ConcreteSection(b=0.3, h=0.5, material=mat, rebar=rebar_bad)


def test_concrete_section_effective_depth():
    sec = _ts_section()
    assert sec.d == pytest.approx(0.45)
    assert sec.d_prime == pytest.approx(0.05)
    assert sec.Ag == pytest.approx(0.15)


def test_concrete_section_As_min_flexure_matches_aci_formula():
    """ACI 9.6.1.2 (SI): As,min = max(0.25 sqrt(fc'[MPa])/fy[MPa], 1.4/fy[MPa]) bd.
    For f_c'=28 MPa, f_y=420 MPa, b=300, d=450:
        max(0.25*sqrt(28)/420, 1.4/420) = max(0.00315, 0.00333) = 0.00333
        × b × d = 0.00333 × 0.30 × 0.45 = 4.5e-4 m² = 450 mm².
    """
    sec = _ts_section()
    assert sec.As_min_flexure() == pytest.approx(450e-6, rel=1e-3)


def test_concrete_section_As_max_tension_controlled():
    """As_max so that ε_t = 0.005 (lower bound of tension-controlled).
    c = 0.003 / 0.008 · d = 0.375 · d.
    a = β_1 · c.
    As_max = 0.85·fc'·b·a / fy.
    """
    sec = _ts_section()
    c_max = 0.003 / 0.008 * sec.d
    a_max = sec.material.beta_1 * c_max
    As_max_expected = (0.85 * sec.material.fc_prime * sec.b * a_max
                        / sec.material.fy)
    assert sec.As_max_tension_controlled() == pytest.approx(
        As_max_expected, rel=1e-10
    )


def test_concrete_section_balanced_neutral_axis():
    """c_b = ε_cu / (ε_cu + ε_ty) · d per ACI 22.2.1.
    For ε_ty = 420/200000 = 0.0021: c_b = 0.003 / 0.0051 · 0.45 = 0.2647 m.
    """
    sec = _ts_section()
    expected = EPSILON_CU / (EPSILON_CU + sec.material.epsilon_ty) * sec.d
    assert sec.neutral_axis_balanced() == pytest.approx(expected)
