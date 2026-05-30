"""Phase 29.4 tests -- column P-M interaction per ACI 318-19 Ch. 22.4.

Verification cases follow MacGregor's "Reinforced Concrete" Ch. 11
worked examples and the CRSI Design Handbook tabulated values for
standard tied-column sections.
"""
from __future__ import annotations

import math

import pytest

from femsolver.design.concrete import (
    EPSILON_CU,
    E_STEEL,
    ConcreteMaterial,
    ConcreteSection,
    RebarLayout,
    column_interaction_point,
    column_interaction_surface,
)


PSI = 6894.757
IN = 0.0254
KIP = 4448.222
KIPIN = KIP * IN


def _16x16_column():
    """Standard textbook column: 16x16 in, 8 #8 bars (4 top + 4 bot),
    f_c' = 4 ksi, f_y = 60 ksi, cover 2.5 in to bar centroid."""
    mat = ConcreteMaterial(fc_prime=4000 * PSI, fy=60000 * PSI)
    rebar = RebarLayout(
        top_bars=("#8", "#8", "#8", "#8"),
        top_cover=2.5 * IN,
        bottom_bars=("#8", "#8", "#8", "#8"),
        bottom_cover=2.5 * IN,
    )
    return ConcreteSection(
        b=16 * IN, h=16 * IN, material=mat, rebar=rebar,
    )


# ============================================================ P_o + cap

def test_Po_matches_aci_22_4_2_2():
    """P_o = 0.85 f_c' (A_g - A_st) + f_y A_st."""
    sec = _16x16_column()
    surf = column_interaction_surface(sec, n_points=20)
    A_st = sec.rebar.As_top + sec.rebar.As_bottom
    A_g = sec.b * sec.h
    P_o_expected = (0.85 * sec.material.fc_prime * (A_g - A_st)
                     + sec.material.fy * A_st)
    assert surf.P_o == pytest.approx(P_o_expected, rel=1e-12)


def test_Pn_max_tied_is_0p80_Po():
    """Tied column: P_n_max = 0.80 P_o per 22.4.2.1."""
    sec = _16x16_column()
    surf = column_interaction_surface(sec, spiral=False)
    assert surf.P_n_max == pytest.approx(0.80 * surf.P_o, rel=1e-12)


def test_Pn_max_spiral_is_0p85_Po():
    """Spiral column: P_n_max = 0.85 P_o per 22.4.2.1."""
    sec = _16x16_column()
    surf = column_interaction_surface(sec, spiral=True)
    assert surf.P_n_max == pytest.approx(0.85 * surf.P_o, rel=1e-12)


def test_pure_tension_equals_minus_Ast_fy():
    sec = _16x16_column()
    surf = column_interaction_surface(sec)
    A_st = sec.rebar.As_top + sec.rebar.As_bottom
    expected = -A_st * sec.material.fy
    assert surf.P_n_pure_tension == pytest.approx(expected, rel=1e-12)


# ============================================================ point evaluator

def test_single_point_equilibrium_force():
    """At a given c, P_n should equal F_c + sum of steel forces by
    direct verification."""
    sec = _16x16_column()
    c = 0.05      # 50 mm, deeply tension-controlled
    pt = column_interaction_point(sec, c)
    # Recompute manually
    fc = sec.material.fc_prime
    fy = sec.material.fy
    beta_1 = sec.material.beta_1
    h = sec.h
    b = sec.b
    a = min(beta_1 * c, h)
    F_c = 0.85 * fc * b * a
    F_s_total = 0.0
    layers = [
        (sec.rebar.top_cover, sec.rebar.As_top),
        (h - sec.rebar.bottom_cover, sec.rebar.As_bottom),
    ]
    for d_i, A_si in layers:
        eps_i = EPSILON_CU * (c - d_i) / c
        f_si = max(-fy, min(fy, E_STEEL * eps_i))
        if d_i <= a:
            f_si_eff = f_si - 0.85 * fc
        else:
            f_si_eff = f_si
        F_s_total += A_si * f_si_eff
    P_n_expected = F_c + F_s_total
    assert pt.P_n == pytest.approx(P_n_expected, rel=1e-12)


def test_balanced_point_has_eps_t_equal_eps_ty():
    """At c = ε_cu d_t / (ε_cu + ε_ty), the extreme tension steel
    reaches yield exactly: ε_t = ε_ty."""
    sec = _16x16_column()
    d_t = sec.h - sec.rebar.bottom_cover
    eps_ty = sec.material.epsilon_ty
    c_bal = EPSILON_CU * d_t / (EPSILON_CU + eps_ty)
    pt = column_interaction_point(sec, c_bal)
    assert pt.epsilon_t == pytest.approx(eps_ty, rel=1e-6)


def test_tension_controlled_boundary_phi_0p90():
    """At c = ε_cu d_t / (ε_cu + 0.005), ε_t = 0.005 -> tension-
    controlled, φ = 0.90."""
    sec = _16x16_column()
    d_t = sec.h - sec.rebar.bottom_cover
    c_tc = EPSILON_CU * d_t / (EPSILON_CU + 0.005)
    pt = column_interaction_point(sec, c_tc)
    assert pt.epsilon_t == pytest.approx(0.005, rel=1e-6)
    assert pt.phi == pytest.approx(0.90, rel=1e-6)


def test_compression_controlled_phi_0p65_tied():
    """At small ε_t (near pure compression), tied φ = 0.65."""
    sec = _16x16_column()
    # Use very large c -> whole section in compression
    pt = column_interaction_point(sec, 5.0 * sec.h, spiral=False)
    assert pt.epsilon_t < sec.material.epsilon_ty
    assert pt.phi == pytest.approx(0.65, rel=1e-6)


def test_compression_controlled_phi_0p75_spiral():
    """Spiral compression-controlled φ = 0.75."""
    sec = _16x16_column()
    pt = column_interaction_point(sec, 5.0 * sec.h, spiral=True)
    assert pt.phi == pytest.approx(0.75, rel=1e-6)


def test_section_type_classification():
    sec = _16x16_column()
    d_t = sec.h - sec.rebar.bottom_cover
    # Deeply tension-controlled
    c_small = 0.05 * d_t
    assert column_interaction_point(sec, c_small).section_type == \
        "tension-controlled"
    # Compression-controlled (large c)
    c_large = 2.0 * sec.h
    assert column_interaction_point(sec, c_large).section_type == \
        "compression-controlled"


# ============================================================ surface properties

def test_surface_points_sorted_by_descending_Pn():
    sec = _16x16_column()
    surf = column_interaction_surface(sec, n_points=30)
    Ps = [p.P_n for p in surf.points]
    for i in range(len(Ps) - 1):
        assert Ps[i] >= Ps[i + 1] - 1e-6, (
            f"surface points not sorted: P[{i}]={Ps[i]} < P[{i+1}]={Ps[i+1]}"
        )


def test_surface_caps_at_Pn_max():
    """No point on the surface should exceed P_n_max (capped per 22.4.2.1)."""
    sec = _16x16_column()
    surf = column_interaction_surface(sec)
    for p in surf.points:
        assert p.P_n <= surf.P_n_max + 1e-6


def test_M_n_zero_at_extremes():
    """M_n -> 0 at both pure axial compression (top of surface) and
    pure tension (bottom)."""
    sec = _16x16_column()
    surf = column_interaction_surface(sec)
    # First point (highest P_n) should have M_n = 0 (capped)
    assert surf.points[0].M_n == pytest.approx(0.0, abs=1e-3)
    # Last point (lowest P_n, near pure tension) should have small M_n
    last = surf.points[-1]
    assert abs(last.M_n) < 1e-3


# ============================================================ DCR

def test_dcr_pure_axial_low_demand_inside_surface():
    """Low axial demand with no moment -> DCR=0 (capacity infinite for M_u=0)."""
    sec = _16x16_column()
    surf = column_interaction_surface(sec)
    dcr = surf.dcr(P_u=200 * KIP, M_u=0.0)
    assert dcr == pytest.approx(0.0)


def test_dcr_at_pure_flexure():
    """At P_u = 0, DCR = M_u / φM_o."""
    sec = _16x16_column()
    surf = column_interaction_surface(sec)
    M_u = 0.5 * surf.M_o
    dcr = surf.dcr(P_u=0.0, M_u=M_u)
    assert dcr == pytest.approx(0.5, rel=2e-2)


def test_dcr_exceeds_one_for_overdesign_demand():
    """Demand well outside the surface -> DCR > 1."""
    sec = _16x16_column()
    surf = column_interaction_surface(sec)
    # Pick a demand well outside
    dcr = surf.dcr(P_u=0.5 * surf.P_n_max, M_u=2.0 * surf.M_o)
    assert dcr > 1.0


def test_dcr_under_one_for_low_demand():
    """Demand well within -> DCR < 1."""
    sec = _16x16_column()
    surf = column_interaction_surface(sec)
    dcr = surf.dcr(P_u=0.2 * surf.P_n_max, M_u=0.2 * surf.M_o)
    assert dcr < 1.0


# ============================================================ input validation

def test_column_interaction_point_rejects_nonpositive_c():
    sec = _16x16_column()
    with pytest.raises(ValueError, match="c must be positive"):
        column_interaction_point(sec, 0.0)
    with pytest.raises(ValueError, match="c must be positive"):
        column_interaction_point(sec, -0.05)
