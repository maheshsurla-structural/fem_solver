"""Phase 30.3 tests -- flexural strength of W-shapes per AISC 360-22
Ch. F2 (compact major-axis bending with lateral-torsional buckling).

Verification: W18x60 worked example, hand-computed M_p, L_p, L_r, and
the three LTB regime regions.
"""
from __future__ import annotations

import math

import pytest

from femsolver.design.steel import (
    PHI_FLEXURE,
    astm_a36,
    astm_a992,
    c_b_from_moments,
    flexural_strength,
    get_section,
)


IN = 0.0254
FT = 12 * IN
KIP = 4448.222
KIPIN = KIP * IN
KIPFT = KIPIN * 12
KSI = 6894.757 * 1000.0


# ============================================================ W18x60 reference

def test_W18x60_plastic_regime():
    """Lb = 4 ft < Lp ≈ 5.93 ft → M_n = M_p = 50 ksi · 123 in³ = 512.5 k-ft."""
    res = flexural_strength(
        get_section("W18x60"), astm_a992(), L_b=4 * FT, C_b=1.0,
    )
    assert res.regime == "plastic"
    M_p_expected = 512.5 * KIPFT
    assert res.M_p == pytest.approx(M_p_expected, rel=2.0e-3)
    assert res.M_n == pytest.approx(M_p_expected, rel=2.0e-3)
    assert res.phi_M_n == pytest.approx(0.90 * M_p_expected, rel=2.0e-3)


def test_W18x60_Lp_matches_F2_5():
    """L_p = 1.76 · ry · √(E/Fy) per Eq F2-5.
    For W18x60 A992: ry = 1.68 in, Fy = 50 ksi, E = 29000 ksi.
    L_p = 1.76·1.68·√(580) = 71.2 in = 5.93 ft."""
    res = flexural_strength(
        get_section("W18x60"), astm_a992(), L_b=2 * FT,
    )
    assert res.L_p / FT == pytest.approx(5.93, rel=1.0e-2)


def test_W18x60_Lr_about_18_2_ft():
    """L_r per Eq F2-6 ≈ 18.2 ft for W18x60 A992 (hand-computed)."""
    res = flexural_strength(
        get_section("W18x60"), astm_a992(), L_b=2 * FT,
    )
    assert res.L_r / FT == pytest.approx(18.2, rel=2.0e-2)


def test_W18x60_inelastic_LTB_linear_interpolation():
    """In the inelastic-LTB regime, M_n linearly interpolates between
    M_p (at L_p) and 0.7 F_y S_x (at L_r). Verify at L_b = 10 ft, C_b = 1."""
    sec = get_section("W18x60"); mat = astm_a992()
    res = flexural_strength(sec, mat, L_b=10 * FT, C_b=1.0)
    assert res.regime == "inelastic-LTB"
    # Hand calc: M_p = 512.5 k-ft; 0.7·Fy·Sx = 0.7·50·108/12 = 315 k-ft
    # At Lb=10: Mn = 512.5 - (512.5-315)·(10-5.93)/(18.19-5.93) = 447 k-ft
    M_n_expected = 447.0 * KIPFT
    assert res.M_n == pytest.approx(M_n_expected, rel=1.0e-2)


def test_W18x60_elastic_LTB():
    """Lb = 25 ft > Lr → elastic LTB regime. M_n ≈ 201 k-ft (hand calc)."""
    res = flexural_strength(
        get_section("W18x60"), astm_a992(), L_b=25 * FT, C_b=1.0,
    )
    assert res.regime == "elastic-LTB"
    M_n_expected = 201.0 * KIPFT
    assert res.M_n == pytest.approx(M_n_expected, rel=2.0e-2)


# ============================================================ regime continuity

def test_regime_continuity_at_Lp():
    """M_n should be continuous at L_b = L_p."""
    sec = get_section("W18x60"); mat = astm_a992()
    # Just below Lp
    res_below = flexural_strength(sec, mat, L_b=0.99 * 5.93 * FT, C_b=1.0)
    # Just above Lp
    res_above = flexural_strength(sec, mat, L_b=1.01 * 5.93 * FT, C_b=1.0)
    assert res_below.regime == "plastic"
    assert res_above.regime == "inelastic-LTB"
    # Both should give ≈ M_p
    assert res_below.M_n == pytest.approx(res_below.M_p, rel=2.0e-3)
    assert res_above.M_n == pytest.approx(res_above.M_p, rel=2.0e-2)


def test_regime_continuity_at_Lr():
    """M_n at L_b = L_r should equal 0.7 F_y S_x from both sides."""
    sec = get_section("W18x60"); mat = astm_a992()
    Sx = sec.Sx; Fy = mat.Fy
    M_n_at_Lr = 0.7 * Fy * Sx     # both formulas give this at Lb=Lr
    Lr = flexural_strength(sec, mat, L_b=2 * FT).L_r
    res_below = flexural_strength(sec, mat, L_b=0.99 * Lr, C_b=1.0)
    res_above = flexural_strength(sec, mat, L_b=1.01 * Lr, C_b=1.0)
    assert res_below.regime == "inelastic-LTB"
    assert res_above.regime == "elastic-LTB"
    assert res_below.M_n == pytest.approx(M_n_at_Lr, rel=3.0e-2)
    assert res_above.M_n == pytest.approx(M_n_at_Lr, rel=3.0e-2)


# ============================================================ C_b effect

def test_C_b_boosts_inelastic_capacity():
    """In the inelastic-LTB regime, C_b > 1 boosts M_n (capped at M_p)."""
    sec = get_section("W18x60"); mat = astm_a992()
    r1 = flexural_strength(sec, mat, L_b=15 * FT, C_b=1.0)
    r2 = flexural_strength(sec, mat, L_b=15 * FT, C_b=1.5)
    assert r2.M_n > r1.M_n
    # Both must respect M_n <= M_p
    assert r1.M_n <= r1.M_p
    assert r2.M_n <= r2.M_p


def test_C_b_boost_capped_at_Mp():
    """C_b cannot push M_n past M_p."""
    sec = get_section("W18x60"); mat = astm_a992()
    res = flexural_strength(sec, mat, L_b=7 * FT, C_b=3.0)
    assert res.M_n <= res.M_p + 1.0e-6


def test_C_b_no_effect_in_plastic_regime():
    """In the plastic regime, M_n = M_p regardless of C_b."""
    sec = get_section("W18x60"); mat = astm_a992()
    r1 = flexural_strength(sec, mat, L_b=4 * FT, C_b=1.0)
    r2 = flexural_strength(sec, mat, L_b=4 * FT, C_b=2.0)
    assert r1.M_n == pytest.approx(r2.M_n)


# ============================================================ c_b_from_moments

def test_c_b_uniform_moment_is_one():
    """Uniform moment along segment -> C_b = 1.0."""
    assert c_b_from_moments(100, 100, 100, 100) == pytest.approx(1.0)


def test_c_b_simply_supported_point_load():
    """Triangular moment (point load at midspan): C_b ≈ 1.316."""
    cb = c_b_from_moments(M_max=100, M_a=50, M_b=100, M_c=50)
    # AISC tabulated value: 1.32
    assert cb == pytest.approx(1.32, rel=1.0e-2)


def test_c_b_clipped_at_3():
    """C_b is clipped at 3.0 per AISC."""
    # Construct very high gradient
    cb = c_b_from_moments(M_max=1000, M_a=0, M_b=0, M_c=0)
    assert cb == pytest.approx(3.0)


def test_c_b_handles_zero_segment_moments():
    """Degenerate input -> C_b = 1.0 (safe default)."""
    cb = c_b_from_moments(0, 0, 0, 0)
    assert cb == pytest.approx(1.0)


# ============================================================ section compactness

def test_W18x60_is_compact():
    res = flexural_strength(
        get_section("W18x60"), astm_a992(), L_b=4 * FT,
    )
    assert res.section_compact


# ============================================================ phi factor

def test_phi_value():
    res = flexural_strength(
        get_section("W18x60"), astm_a992(), L_b=4 * FT,
    )
    assert res.phi == pytest.approx(PHI_FLEXURE)
    assert res.phi == pytest.approx(0.90)


# ============================================================ input validation

def test_rejects_nonpositive_Lb():
    with pytest.raises(ValueError, match="L_b"):
        flexural_strength(get_section("W18x60"), astm_a992(), L_b=0.0)


def test_rejects_nonpositive_Cb():
    with pytest.raises(ValueError, match="C_b"):
        flexural_strength(
            get_section("W18x60"), astm_a992(),
            L_b=10 * FT, C_b=0.0,
        )


# ============================================================ monotonicity

def test_flexural_capacity_decreases_with_Lb():
    """φM_n monotonically non-increasing in L_b."""
    sec = get_section("W18x60"); mat = astm_a992()
    caps = []
    for L_b_ft in (3, 6, 10, 15, 20, 30, 50):
        r = flexural_strength(sec, mat, L_b=L_b_ft * FT, C_b=1.0)
        caps.append(r.phi_M_n)
    for i in range(len(caps) - 1):
        assert caps[i + 1] <= caps[i] + 1e-9
