"""Phase 29.3 tests -- beam shear strength + design per ACI 318-19
Ch. 22.5 + 9.6.3 + 9.7.6.2.
"""
from __future__ import annotations

import math

import pytest

from femsolver.design.concrete import (
    PHI_SHEAR,
    ConcreteMaterial,
    ConcreteSection,
    RebarLayout,
    beam_shear_strength,
    design_stirrup_spacing,
)

PSI = 6894.757
IN = 0.0254


def _make_section(*, stirrup="#3", stirrup_spacing=0.150, stirrup_legs=2,
                   fc_prime_psi=4000, fy_psi=60000,
                   b_in=12, h_in=23.5, cover_in=2.5):
    """Helper: build a baseline beam section."""
    mat = ConcreteMaterial(
        fc_prime=fc_prime_psi * PSI, fy=fy_psi * PSI,
    )
    rebar = RebarLayout(
        bottom_bars=("#8", "#8", "#8"),
        bottom_cover=cover_in * IN,
        stirrup_designation=stirrup,
        stirrup_spacing=stirrup_spacing,
        stirrup_legs=stirrup_legs,
    )
    return ConcreteSection(
        b=b_in * IN, h=h_in * IN, material=mat, rebar=rebar,
    )


# ============================================================ V_c

def test_Vc_matches_ACI_SI_formula():
    """V_c per ACI 22.5.5.1 simplified: V_c = 0.17 √f_c'[MPa] b d (in N
    with b·d in m²). Verify by direct computation."""
    sec = _make_section()
    r = beam_shear_strength(sec)
    fc_MPa = sec.material.fc_prime / 1.0e6
    expected = 0.17 * math.sqrt(fc_MPa) * sec.b * 1000.0 * sec.d * 1000.0
    assert r.V_c == pytest.approx(expected, rel=1e-10)


def test_Vc_independent_of_stirrup_layout():
    """V_c only depends on concrete strength and geometry."""
    r1 = beam_shear_strength(_make_section(stirrup_spacing=0.10))
    r2 = beam_shear_strength(_make_section(stirrup_spacing=0.25))
    assert r1.V_c == pytest.approx(r2.V_c)


def test_Vc_scales_with_sqrt_fc():
    """V_c ∝ √f_c'. Doubling f_c' should multiply V_c by √2."""
    sec_low = _make_section(fc_prime_psi=3000)
    sec_high = _make_section(fc_prime_psi=6000)
    r_low = beam_shear_strength(sec_low)
    r_high = beam_shear_strength(sec_high)
    assert r_high.V_c / r_low.V_c == pytest.approx(math.sqrt(2.0), rel=1e-6)


# ============================================================ V_s

def test_Vs_matches_ACI_formula():
    """V_s = A_v · f_yt · d / s (ACI 22.5.10.5.3)."""
    sec = _make_section(stirrup="#3", stirrup_spacing=0.20, stirrup_legs=2)
    r = beam_shear_strength(sec)
    Av = sec.rebar.Av
    expected = Av * sec.material.fy * sec.d / 0.20
    assert r.V_s == pytest.approx(expected, rel=1e-10)


def test_Vs_inverse_in_spacing():
    """Halving spacing should double V_s."""
    r_wide = beam_shear_strength(_make_section(stirrup_spacing=0.20))
    r_tight = beam_shear_strength(_make_section(stirrup_spacing=0.10))
    assert r_tight.V_s == pytest.approx(2.0 * r_wide.V_s, rel=1e-6)


def test_Vs_vanishes_as_spacing_grows():
    """Very-wide spacing -> V_s/V_c approaches zero (V_s ~ 1/s)."""
    sec_close = _make_section(stirrup_spacing=0.10)
    sec_wide = _make_section(stirrup_spacing=10.0)     # 100x wider
    r_close = beam_shear_strength(sec_close)
    r_wide = beam_shear_strength(sec_wide)
    # V_s should scale inversely with spacing -> 100x smaller
    assert r_wide.V_s == pytest.approx(r_close.V_s / 100.0, rel=1e-6)


# ============================================================ phi factor

def test_phi_shear_is_0p75():
    sec = _make_section()
    r = beam_shear_strength(sec)
    assert r.phi == pytest.approx(0.75)
    assert r.phi_V_n == pytest.approx(0.75 * r.V_n)


# ============================================================ max spacing

def test_s_max_low_Vs_uses_d_over_2():
    """When V_s is low, s_max = min(d/2, 600 mm)."""
    # Use a wide-spacing section so V_s is small
    sec = _make_section(stirrup_spacing=0.300)
    r = beam_shear_strength(sec)
    expected = min(sec.d / 2.0, 0.600)
    assert r.s_max == pytest.approx(expected, rel=1e-6)


def test_s_max_high_Vs_uses_d_over_4():
    """When V_s exceeds threshold (0.33 √f_c' b d), s_max = d/4 or
    300 mm."""
    # Tight spacing to drive V_s above threshold
    sec = _make_section(stirrup="#5", stirrup_spacing=0.050)
    r = beam_shear_strength(sec)
    expected = min(sec.d / 4.0, 0.300)
    assert r.s_max == pytest.approx(expected, rel=1e-6)


# ============================================================ min reinforcement

def test_min_reinforcement_formula():
    """Av_min/s = max(0.062 √f_c'[MPa] b / f_yt[MPa], 0.35 b / f_yt[MPa])."""
    sec = _make_section()
    r = beam_shear_strength(sec)
    fc_MPa = sec.material.fc_prime / 1.0e6
    fyt_MPa = sec.material.fy / 1.0e6
    b_mm = sec.b * 1000.0
    a1 = 0.062 * math.sqrt(fc_MPa) * b_mm / fyt_MPa
    a2 = 0.35 * b_mm / fyt_MPa
    expected = max(a1, a2) * 1.0e-3       # m²/m
    assert r.Av_min_per_s == pytest.approx(expected, rel=1e-10)


# ============================================================ design forward

def test_design_Vc_alone_sufficient():
    """If V_u <= φ V_c then V_s_required <= 0 and s_required = inf."""
    sec = _make_section()
    r_check = beam_shear_strength(sec)
    # V_u must be below φ V_c specifically (not just φ V_n), since
    # the baseline stirrups already contribute V_s.
    V_u = 0.5 * PHI_SHEAR * r_check.V_c
    d = design_stirrup_spacing(sec, V_u=V_u)
    assert d.V_s_required <= 0.0
    assert math.isinf(d.s_required)
    # Recommended is bounded by s_max or s_min_reinforcement
    assert d.s_recommended == min(d.s_max, d.s_min_reinforcement)


def test_design_requires_stirrups_when_demand_exceeds_phi_Vc():
    """When V_u > φ V_c the required stirrup spacing is finite and
    governs."""
    sec = _make_section()
    r_check = beam_shear_strength(sec)
    V_u = 0.9 * r_check.phi_V_n     # high demand
    d = design_stirrup_spacing(sec, V_u=V_u)
    assert d.V_s_required > 0.0
    assert math.isfinite(d.s_required)
    # Recommended spacing must be at most s_required
    assert d.s_recommended <= d.s_required + 1.0e-12


def test_design_spacing_satisfies_strength_at_recommendation():
    """Using s_recommended should yield φ V_n >= V_u."""
    sec = _make_section()
    r_check = beam_shear_strength(sec)
    V_u = 0.85 * r_check.phi_V_n
    d = design_stirrup_spacing(sec, V_u=V_u)
    # Construct a new section with the recommended spacing and verify
    new_rebar = RebarLayout(
        bottom_bars=sec.rebar.bottom_bars,
        bottom_cover=sec.rebar.bottom_cover,
        stirrup_designation=sec.rebar.stirrup_designation,
        stirrup_spacing=d.s_recommended,
        stirrup_legs=sec.rebar.stirrup_legs,
    )
    new_sec = ConcreteSection(
        b=sec.b, h=sec.h, material=sec.material, rebar=new_rebar,
    )
    new_check = beam_shear_strength(new_sec, V_u=V_u)
    assert new_check.phi_V_n >= V_u - 1.0e-6


def test_design_max_spacing_governs_for_low_demand():
    """For very low demand, s_required = inf and s_max bounds the
    recommendation."""
    sec = _make_section()
    d = design_stirrup_spacing(sec, V_u=1.0e3)     # tiny demand
    assert d.s_recommended <= d.s_max + 1.0e-12


# ============================================================ user-supplied V_u

def test_undercapacity_flagged_in_notes():
    """When V_u > φ V_n, notes should warn."""
    sec = _make_section(stirrup_spacing=0.40)     # weak shear capacity
    r = beam_shear_strength(sec)
    V_u_high = 2.0 * r.phi_V_n
    r2 = beam_shear_strength(sec, V_u=V_u_high)
    assert "undercapacity" in r2.notes


def test_excess_spacing_flagged():
    """Provide stirrups at a spacing exceeding s_max; spacing_ok=False."""
    sec = _make_section(stirrup_spacing=1.0)      # 1 m spacing
    r = beam_shear_strength(sec)
    assert not r.spacing_ok
    assert "s_max" in r.notes or "9.7.6" in r.notes
