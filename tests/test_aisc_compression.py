"""Phase 30.2 tests -- AISC 360-22 Ch. E compression strength.

Verification cases follow AISC Manual 15th ed. Part E (Design of
Members for Compression) worked examples.
"""
from __future__ import annotations

import math

import pytest

from femsolver.design.steel import (
    PHI_COMPRESSION,
    astm_a36,
    astm_a992,
    compression_strength,
    get_section,
)


IN = 0.0254
FT = 12 * IN
KIP = 4448.222
KSI = 6894.757 * 1000     # Pa per ksi
PSI = 6894.757


# ============================================================ AISC Example E.1a

def test_W14x90_at_14ft_matches_AISC_manual():
    """AISC Manual Example E.1a: W14x90 (A992), K=1, L=14 ft (y-axis
    governs since ry = 3.70 in < rx = 6.14 in).

    Expected (AISC Manual Table 4-1 and E.1a):
        KL/r ≈ 45.4
        F_e ≈ 138.5 ksi
        F_cr ≈ 43.0 ksi (inelastic, since KL/r < 113.4)
        P_n  ≈ 1140 kips
        φP_n ≈ 1025 kips
    """
    res = compression_strength(
        get_section("W14x90"), astm_a992(), L=14 * FT,
    )
    assert res.KL_over_r == pytest.approx(45.4, rel=1.0e-2)
    assert res.governing_axis == "y"
    assert res.buckling_regime == "inelastic"
    # F_e ≈ 138.5 ksi
    assert res.F_e / KSI == pytest.approx(138.5, rel=2.0e-2)
    # F_cr ≈ 43.0 ksi
    assert res.F_cr / KSI == pytest.approx(43.0, rel=2.0e-2)
    # P_n ≈ 1140 kips
    assert res.P_n / KIP == pytest.approx(1140.0, rel=1.0e-2)
    # φP_n ≈ 1025 kips
    assert res.phi_P_n / KIP == pytest.approx(1025.0, rel=1.0e-2)
    assert res.section_nonslender


# ============================================================ regime selection

def test_short_column_yields_short_of_inelastic_buckling():
    """Very short column: KL/r small, F_cr approaches F_y."""
    res = compression_strength(
        get_section("W14x90"), astm_a992(), L=1.0 * FT,    # 1 ft
    )
    assert res.buckling_regime == "inelastic"
    assert res.F_cr > 0.95 * astm_a992().Fy


def test_long_column_elastic_buckling():
    """Long column: KL/r > 4.71 √(E/Fy) → elastic regime."""
    # For W14x90 A992: transition at KL/r = 113.4
    # ry = 3.70 in. Solve K·L/r > 113.4 → L > 113.4 * 3.70 in = 419.6 in = 35 ft
    res = compression_strength(
        get_section("W14x90"), astm_a992(), L=40 * FT,
    )
    assert res.buckling_regime == "elastic"
    # F_cr / F_e should equal 0.877 exactly
    assert res.F_cr / res.F_e == pytest.approx(0.877, rel=1.0e-12)


def test_transition_KLr_matches_AISC():
    """The transition between inelastic and elastic occurs at
    KL/r = 4.71 √(E/Fy) per E3."""
    mat = astm_a992()
    transition_expected = 4.71 * math.sqrt(mat.E / mat.Fy)
    # Build a hypothetical condition at exactly the transition
    # by tuning L. Use W14x90: ry = 3.70 in.
    sec = get_section("W14x90")
    L_transition = transition_expected * sec.ry      # K = 1
    res = compression_strength(sec, mat, L=L_transition)
    # At the boundary, both formulas should give the same F_cr.
    # Eq E3-2: F_cr = 0.658^(Fy/F_e) * Fy.
    # At transition: Fy/F_e = (KL/r)² · Fy / (π² E) = 4.71²·Fy/(π² E) · Fy/E·1
    # Simpler: just verify F_cr/F_e ≈ 0.877 (within ~3%).
    # Hmm actually they should match exactly when KL/r equals transition.
    # 0.658^(Fy/F_e_t) at boundary: Fy/F_e_t = (KL/r)²·Fy/π²E = transition²·Fy/π²E
    # = (4.71)²·Fy²/(π²·E²)·E/Fy = 4.71²/π² · Fy/E
    # For Fy=345 MPa, E=200000 MPa: 4.71²/π² · 345/200000 = 22.18/9.87·0.001725 = 0.00388
    # F_cr/Fy = 0.658^0.00388 ≈ 0.998 — hmm that doesn't match elastic.
    # OK actually at the transition the formulas should be CONTINUOUS:
    # F_cr_inel = 0.658^(Fy/F_e) * Fy
    # F_cr_elas = 0.877 * F_e
    # At Fy/F_e = 2.25 (the equivalent transition): 0.658^2.25 = 0.390
    # F_cr_inel = 0.390 * Fy
    # F_cr_elas = 0.877 * F_e = 0.877 * Fy/2.25 = 0.390 * Fy ✓ continuous
    # So I had the threshold wrong above. Let me just verify the regime
    # switches around the transition.
    L_below = transition_expected * sec.ry * 0.95
    L_above = transition_expected * sec.ry * 1.05
    res_below = compression_strength(sec, mat, L=L_below)
    res_above = compression_strength(sec, mat, L=L_above)
    assert res_below.buckling_regime == "inelastic"
    assert res_above.buckling_regime == "elastic"


# ============================================================ axis selection

def test_y_axis_governs_for_pinned_W_shape():
    """For W-shapes ry < rx, so under equal K and L the y-axis governs."""
    res = compression_strength(
        get_section("W14x90"), astm_a992(),
        L=10 * FT, K_x=1.0, K_y=1.0,
    )
    assert res.governing_axis == "y"


def test_x_axis_governs_when_Ly_is_short():
    """If the weak axis is heavily braced (small L_y) but the strong
    axis has full length, x-axis can govern."""
    res = compression_strength(
        get_section("W14x90"), astm_a992(),
        L=20 * FT, L_y=2 * FT,        # heavy weak-axis bracing
    )
    assert res.governing_axis == "x"


def test_K_factor_increases_slenderness():
    """Larger K → larger KL/r → lower F_cr."""
    sec = get_section("W14x90"); mat = astm_a992()
    r1 = compression_strength(sec, mat, L=10 * FT, K_y=1.0)
    r2 = compression_strength(sec, mat, L=10 * FT, K_y=2.0)
    assert r2.KL_over_r == pytest.approx(2.0 * r1.KL_over_r, rel=1e-12)
    assert r2.F_cr < r1.F_cr
    assert r2.phi_P_n < r1.phi_P_n


# ============================================================ slenderness E2

def test_E2_recommended_limit_flagged():
    """KL/r > 200 should be flagged in notes."""
    # W4x13 has ry ≈ 1.00 in. KL/r=200 → L = 200 in = 16.7 ft
    res = compression_strength(
        get_section("W4x13"), astm_a992(), L=25 * FT,
    )
    assert res.KL_over_r > 200
    assert not res.slenderness_ok
    assert "E2" in res.notes


def test_E2_recommended_limit_passes_for_normal_lengths():
    res = compression_strength(
        get_section("W14x90"), astm_a992(), L=14 * FT,
    )
    assert res.slenderness_ok
    assert "E2" not in res.notes


# ============================================================ section slenderness

def test_W14x90_is_nonslender():
    res = compression_strength(
        get_section("W14x90"), astm_a992(), L=10 * FT,
    )
    assert res.section_nonslender


def test_phi_value():
    res = compression_strength(
        get_section("W14x90"), astm_a992(), L=10 * FT,
    )
    assert res.phi == pytest.approx(PHI_COMPRESSION)
    assert res.phi == pytest.approx(0.90)
    assert res.phi_P_n == pytest.approx(0.90 * res.P_n)


# ============================================================ input validation

def test_rejects_nonpositive_L():
    with pytest.raises(ValueError):
        compression_strength(
            get_section("W14x90"), astm_a992(), L=0.0,
        )


def test_rejects_nonpositive_K():
    with pytest.raises(ValueError):
        compression_strength(
            get_section("W14x90"), astm_a992(),
            L=10 * FT, K_x=-1.0,
        )


# ============================================================ sanity sweep

def test_compression_capacity_decreases_with_length():
    """φP_n is monotonically non-increasing in L (longer = more buckling)."""
    sec = get_section("W14x90"); mat = astm_a992()
    capacities = []
    for L_ft in (5, 10, 15, 20, 30, 40, 50):
        res = compression_strength(sec, mat, L=L_ft * FT)
        capacities.append(res.phi_P_n)
    for i in range(len(capacities) - 1):
        assert capacities[i + 1] <= capacities[i]
