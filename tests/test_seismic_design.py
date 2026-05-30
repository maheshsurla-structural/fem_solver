"""Phase 32 tests -- seismic capacity-design detailing.

Covers:
* §32.1 Strong-Column-Weak-Beam check (ACI 18.7.3 / AISC 341 E3.4a)
* §32.2 Capacity-design beam shear (ACI 18.6.5)
* §32.3 Confined-concrete reinforcement detailing (ACI 18.7.5)
"""
from __future__ import annotations

import math

import pytest

from femsolver.design.concrete import (
    ConcreteMaterial,
    ConcreteSection,
    RebarLayout,
    beam_flexural_strength,
)
from femsolver.design.seismic import (
    SCWB_RATIO_ACI_SMF,
    SCWB_RATIO_AISC_SMF,
    capacity_design_shear,
    confined_concrete_detailing,
    probable_moment,
    scwb_check,
)


def _ts_mat():
    return ConcreteMaterial(fc_prime=28e6, fy=420e6)


def _beam_section():
    rebar = RebarLayout(
        bottom_bars=("#7", "#7", "#7"),
        bottom_cover=0.050,
        top_bars=("#7", "#7"),
        top_cover=0.050,
        stirrup_designation="#3",
        stirrup_spacing=0.150,
    )
    return ConcreteSection(b=0.30, h=0.55, material=_ts_mat(), rebar=rebar)


def _column_section(stirrup_spacing=0.10, stirrup_legs=4):
    rebar = RebarLayout(
        top_bars=("#9", "#9", "#9", "#9"),
        top_cover=0.060,
        bottom_bars=("#9", "#9", "#9", "#9"),
        bottom_cover=0.060,
        stirrup_designation="#4",
        stirrup_spacing=stirrup_spacing,
        stirrup_legs=stirrup_legs,
    )
    return ConcreteSection(b=0.50, h=0.50, material=_ts_mat(), rebar=rebar)


# ============================================================ SCWB

def test_scwb_aci_passes_when_ratio_above_6_over_5():
    """sum(M_nc)/sum(M_nb) = 1.30 > 1.20 -> passes."""
    res = scwb_check(
        column_M_n=[260e3, 260e3], beam_M_n=[200e3, 200e3], code="ACI",
    )
    assert res.ratio == pytest.approx(1.30, rel=1e-3)
    assert res.ratio_required == pytest.approx(SCWB_RATIO_ACI_SMF)
    assert res.passes


def test_scwb_aci_fails_when_ratio_below():
    """ratio 0.75 < 1.20 -> fails."""
    res = scwb_check(
        column_M_n=[150e3, 150e3], beam_M_n=[200e3, 200e3], code="ACI",
    )
    assert res.ratio == pytest.approx(0.75, rel=1e-3)
    assert not res.passes
    assert "SCWB ratio" in res.notes


def test_scwb_aisc_default_required_is_1p0():
    """AISC SMF: ratio >= 1.0 required."""
    res = scwb_check(
        column_M_n=[100e3, 100e3], beam_M_n=[100e3, 100e3], code="AISC",
    )
    assert res.ratio_required == pytest.approx(SCWB_RATIO_AISC_SMF)
    assert res.ratio == pytest.approx(1.0)
    assert res.passes


def test_scwb_custom_ratio_required():
    """User can override the required ratio."""
    res = scwb_check(
        column_M_n=[100e3], beam_M_n=[100e3],
        ratio_required=2.0, code="ACI",
    )
    assert res.ratio_required == 2.0
    assert not res.passes


def test_scwb_rejects_unknown_code():
    with pytest.raises(ValueError, match="code"):
        scwb_check(
            column_M_n=[1.0], beam_M_n=[1.0], code="IS 13920",
        )


def test_scwb_zero_beam_moments_trivially_passes():
    """If no beams frame in, the ratio is +inf -> passes."""
    res = scwb_check(column_M_n=[100e3], beam_M_n=[], code="ACI")
    assert math.isinf(res.ratio)
    assert res.passes


def test_scwb_uses_absolute_values():
    """Signed moments treated as magnitudes."""
    res = scwb_check(
        column_M_n=[-260e3, +260e3], beam_M_n=[-200e3, +200e3],
        code="ACI",
    )
    assert res.sum_M_nc == 520e3
    assert res.sum_M_nb == 400e3


# ============================================================ probable_moment

def test_probable_moment_increases_with_fy_boost():
    """M_pr > M_n because of the 1.25 f_y boost."""
    sec = _beam_section()
    fc = beam_flexural_strength(sec)
    M_n = fc.M_n
    M_pr = probable_moment(sec)
    assert M_pr > M_n
    # Should be close to (but not exactly) 1.25 * M_n for tension-
    # controlled sections (the (d - a/2) lever arm shifts a bit
    # because a depends on f_y).
    assert 1.15 < M_pr / M_n < 1.30


# ============================================================ capacity shear

def test_capacity_shear_combines_sway_and_gravity():
    """V_e = (M_pr_L + M_pr_R) / L_n + w_u * L_n / 2."""
    sec = _beam_section()
    L_n = 5.0
    w_u = 30e3
    res = capacity_design_shear(sec, sec, L_n=L_n, w_u=w_u)
    M_pr = probable_moment(sec)
    expected_sway = 2 * M_pr / L_n
    expected_grav = w_u * L_n / 2.0
    expected_V_e = expected_sway + expected_grav
    assert res.V_e_sway == pytest.approx(expected_sway, rel=1e-10)
    assert res.V_g == pytest.approx(expected_grav, rel=1e-10)
    assert res.V_e == pytest.approx(expected_V_e, rel=1e-10)


def test_capacity_shear_returns_max_of_Ve_and_Vu():
    """V_design = max(V_e, V_u_analysis)."""
    sec = _beam_section()
    # Make V_u_analysis huge -> it should govern
    huge_Vu = 1000e3
    res = capacity_design_shear(sec, sec, L_n=5.0, V_u_analysis=huge_Vu)
    assert res.V_design == pytest.approx(huge_Vu)


def test_capacity_shear_Ve_governs_typical_case():
    """For typical SMF demand, V_e (capacity-design) governs over V_u."""
    sec = _beam_section()
    res = capacity_design_shear(
        sec, sec, L_n=5.0, w_u=30e3, V_u_analysis=50e3,
    )
    assert res.V_e > res.V_u_analysis
    assert res.V_design == res.V_e
    assert "governs over analysis" in res.notes


def test_capacity_shear_rejects_nonpositive_Ln():
    with pytest.raises(ValueError, match="L_n"):
        capacity_design_shear(_beam_section(), _beam_section(), L_n=0.0)


def test_capacity_shear_asymmetric_sections():
    """Different M_pr at left and right -> different V_e contributions."""
    # Left section: heavily reinforced
    left_rebar = RebarLayout(
        bottom_bars=("#8", "#8", "#8", "#8"), bottom_cover=0.050,
    )
    left = ConcreteSection(b=0.30, h=0.55, material=_ts_mat(), rebar=left_rebar)
    # Right section: lightly reinforced
    right_rebar = RebarLayout(
        bottom_bars=("#5", "#5"), bottom_cover=0.050,
    )
    right = ConcreteSection(b=0.30, h=0.55, material=_ts_mat(), rebar=right_rebar)
    res = capacity_design_shear(left, right, L_n=5.0)
    assert res.M_pr_left > res.M_pr_right
    expected_sway = (res.M_pr_left + res.M_pr_right) / 5.0
    assert res.V_e_sway == pytest.approx(expected_sway, rel=1e-10)


# ============================================================ confinement

def test_confinement_l_o_takes_maximum_of_three():
    """l_o = max(member depth, clear/6, 450 mm)."""
    sec = _column_section()
    # For 500x500 col with H_clear = 3 m: max(500, 500, 450) = 500
    cd = confined_concrete_detailing(sec, column_clear_height=3.0)
    assert cd.l_o == pytest.approx(0.500)
    # Tall column: 500/6 = 83 mm < 500 mm, 450 mm; still 500 governs
    cd2 = confined_concrete_detailing(sec, column_clear_height=10.0)
    # max(500, 10/6=1.667, 0.450) = 1.667
    assert cd2.l_o == pytest.approx(10.0 / 6.0, rel=1e-3)


def test_confinement_s_o_governed_by_b_min_over_4_for_narrow_columns():
    """For a small column, b_min/4 typically governs s_o."""
    rebar = RebarLayout(
        top_bars=("#5", "#5"),
        bottom_bars=("#5", "#5"),
        top_cover=0.040, bottom_cover=0.040,
        stirrup_designation="#3", stirrup_spacing=0.080,
    )
    small_col = ConcreteSection(b=0.25, h=0.25, material=_ts_mat(),
                                  rebar=rebar)
    cd = confined_concrete_detailing(
        small_col, column_clear_height=3.0, longitudinal_bar="#5",
    )
    # b_min/4 = 250/4 = 62.5 mm; 6 d_b = 6*15.9 = 95 mm
    # s_x bounded 100-150 mm; b_min/4 governs
    assert cd.s_o_required == pytest.approx(0.0625)


def test_confinement_Ash_per_s_includes_both_AcAg_term_and_floor():
    """A_sh/s = max(0.3·(A_g/A_ch-1)·fc'/fy, 0.09·fc'/fy)·b_c."""
    sec = _column_section()
    cd = confined_concrete_detailing(sec, column_clear_height=3.0)
    # Recompute manually
    fc = sec.material.fc_prime; fy = sec.material.fy
    cover = max(sec.rebar.top_cover, sec.rebar.bottom_cover)
    b_c = sec.b - 2 * cover; h_c = sec.h - 2 * cover
    A_ch = b_c * h_c; A_g = sec.b * sec.h
    ratio_a = 0.3 * (A_g / A_ch - 1.0) * fc / fy
    ratio_b = 0.09 * fc / fy
    expected = max(ratio_a, ratio_b) * b_c
    assert cd.Ash_per_s_required == pytest.approx(expected, rel=1e-10)


def test_confinement_passes_when_spacing_and_area_satisfied():
    """Tight stirrups + adequate area -> passes."""
    sec = _column_section(stirrup_spacing=0.05, stirrup_legs=4)
    cd = confined_concrete_detailing(sec, column_clear_height=3.0)
    # Check passes are both true
    assert cd.spacing_ok
    assert cd.reinforcement_ok
    assert cd.passes


def test_confinement_fails_for_wide_stirrups():
    """Wide stirrup spacing -> spacing_ok false."""
    sec = _column_section(stirrup_spacing=0.300, stirrup_legs=2)
    cd = confined_concrete_detailing(sec, column_clear_height=3.0)
    assert not cd.spacing_ok
    assert not cd.passes
    assert "stirrup spacing" in cd.notes.lower()


def test_confinement_rejects_nonpositive_clear_height():
    sec = _column_section()
    with pytest.raises(ValueError, match="column_clear_height"):
        confined_concrete_detailing(sec, column_clear_height=0.0)
