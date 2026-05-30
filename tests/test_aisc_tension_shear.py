"""Phase 30.4 tests -- tension (Ch. D) + shear (Ch. G) per AISC 360-22.
"""
from __future__ import annotations

import math

import pytest

from femsolver.design.steel import (
    PHI_TENSION_RUPTURE,
    PHI_TENSION_YIELD,
    astm_a36,
    astm_a992,
    get_section,
    shear_strength,
    tension_strength,
)


IN = 0.0254
FT = 12 * IN
KIP = 4448.222
KSI = 6894.757 * 1000


# ============================================================ TENSION

def test_tension_W14x90_yielding_governs():
    """W14x90 A992 with A_e = A_g: yielding (φP_n = 0.90·F_y·A_g)
    governs over rupture (φP_n = 0.75·F_u·A_g)."""
    res = tension_strength(get_section("W14x90"), astm_a992())
    sec = get_section("W14x90"); mat = astm_a992()
    expected_yield = 0.90 * mat.Fy * sec.A
    expected_rupture = 0.75 * mat.Fu * sec.A
    assert res.phi_P_n_yield == pytest.approx(expected_yield, rel=1e-12)
    assert res.phi_P_n_rupture == pytest.approx(expected_rupture, rel=1e-12)
    # Yielding should govern (1192 < 1292 kips)
    assert res.governing_limit_state == "yielding"
    assert res.phi_P_n == res.phi_P_n_yield
    # ≈ 1192 kips per AISC Manual
    assert res.phi_P_n / KIP == pytest.approx(1192.5, rel=1e-3)


def test_tension_with_bolt_holes_rupture_governs():
    """A_e = 0.85·A_g (typical bolt-hole reduction): rupture may govern."""
    sec = get_section("W14x90"); mat = astm_a992()
    res = tension_strength(sec, mat, A_e=0.85 * sec.A)
    # Yield strength unchanged; rupture strength reduced
    expected_rupture = 0.75 * mat.Fu * 0.85 * sec.A
    assert res.phi_P_n_rupture == pytest.approx(expected_rupture, rel=1e-12)
    # For W14x90 A992 with Ae=0.85·Ag: rupture = 1098 < yield = 1192
    assert res.governing_limit_state == "rupture"
    assert res.phi_P_n == res.phi_P_n_rupture


def test_tension_validates_A_e():
    with pytest.raises(ValueError, match="A_e"):
        tension_strength(get_section("W14x90"), astm_a992(), A_e=-1.0)


def test_tension_phi_factors():
    res = tension_strength(get_section("W14x90"), astm_a992())
    # Yielding limit state
    assert PHI_TENSION_YIELD == pytest.approx(0.90)
    # Rupture limit state
    assert PHI_TENSION_RUPTURE == pytest.approx(0.75)


def test_tension_slenderness_warning_above_300():
    """KL/r > 300 should be flagged in notes."""
    sec = get_section("W14x90")
    # ry = 94 mm. KL/r = 300 → L = 300 * 0.094 = 28.2 m
    res = tension_strength(sec, astm_a992(), L=35.0, K=1.0)
    assert "300" in res.notes


def test_tension_no_slenderness_warning_below_300():
    res = tension_strength(get_section("W14x90"), astm_a992(),
                            L=5.0, K=1.0)
    assert "300" not in res.notes


def test_tension_a36_steel():
    """A36 has lower Fy/Fu -> proportionally lower strengths."""
    sec = get_section("W14x90")
    res992 = tension_strength(sec, astm_a992())
    res36 = tension_strength(sec, astm_a36())
    assert res36.phi_P_n < res992.phi_P_n
    # Ratio matches Fy ratio for the yielding limit (both materials,
    # yielding governs if Ae=Ag)
    assert res36.phi_P_n / res992.phi_P_n == pytest.approx(
        astm_a36().Fy / astm_a992().Fy, rel=1e-12
    )


# ============================================================ SHEAR

def test_shear_W14x90_matches_AISC():
    """W14x90 A992: A_w = d·t_w = 14.02 · 0.440 = 6.17 in²,
    V_n = 0.6·50·6.17 = 185.0 kips, φ_v = 1.00 → φV_n = 185 kips."""
    res = shear_strength(get_section("W14x90"), astm_a992())
    sec = get_section("W14x90")
    expected_Aw = sec.d * sec.tw
    assert res.A_w == pytest.approx(expected_Aw, rel=1e-12)
    assert res.web_compact
    assert res.C_v1 == 1.0
    assert res.phi == 1.0
    # φV_n ≈ 185 kips
    assert res.phi_V_n / KIP == pytest.approx(185.0, rel=1.0e-2)


def test_shear_compact_web_uses_phi_1p00():
    """Rolled W-shapes with h/tw < 2.24√(E/Fy) ≈ 54: φ_v = 1.00."""
    for designation in ("W14x90", "W18x60", "W24x84", "W36x150"):
        res = shear_strength(get_section(designation), astm_a992())
        assert res.web_compact, f"{designation} expected compact web"
        assert res.phi == 1.00


def test_shear_C_v1_is_one_for_compact_web():
    """Compact web → no shear-buckling reduction (C_v1 = 1.0)."""
    res = shear_strength(get_section("W14x90"), astm_a992())
    assert res.C_v1 == pytest.approx(1.0)


def test_shear_formula_scales_with_Fy():
    """φV_n ∝ F_y (for fixed section)."""
    sec = get_section("W14x90")
    r992 = shear_strength(sec, astm_a992())
    r36 = shear_strength(sec, astm_a36())
    ratio = r36.phi_V_n / r992.phi_V_n
    assert ratio == pytest.approx(astm_a36().Fy / astm_a992().Fy, rel=1e-10)


def test_shear_aw_equals_d_times_tw():
    """A_w = d · t_w per AISC G2.1 for I-shapes."""
    for designation in ("W14x90", "W18x60", "W24x84"):
        sec = get_section(designation)
        res = shear_strength(sec, astm_a992())
        assert res.A_w == pytest.approx(sec.d * sec.tw, rel=1e-12)


def test_shear_Vn_formula():
    """V_n = 0.6·F_y·A_w·C_v1."""
    sec = get_section("W14x90"); mat = astm_a992()
    res = shear_strength(sec, mat)
    expected = 0.6 * mat.Fy * sec.d * sec.tw * res.C_v1
    assert res.V_n == pytest.approx(expected, rel=1e-12)


def test_shear_W12x14_check():
    """W12x14 is a slender beam (small flanges). Verify the algorithm
    runs without errors and returns reasonable values."""
    res = shear_strength(get_section("W12x14"), astm_a992())
    assert res.V_n > 0
    assert res.phi_V_n > 0
    assert res.phi_V_n == res.phi * res.V_n
