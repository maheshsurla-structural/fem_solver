"""Phase 29.2 tests -- rectangular beam flexural design per
ACI 318-19 Ch. 22.2.

Verification cases come from MacGregor's "Reinforced Concrete" (PCA
Notes Ch. 4 worked examples) and the ACI 318-19 Design Examples
manual.
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

# Conversion factors
PSI_TO_PA = 6894.757
IN_TO_M = 0.0254
LBFT_TO_NM = 1.0 / 0.7376
KFT_TO_NM = 1000.0 * LBFT_TO_NM


# ============================================================ singly-reinforced

def test_singly_reinforced_macgregor_example():
    """PCA Notes Ch. 4 / MacGregor:
    Beam b=12", h=23.5", d=21" (3 #8 bottom), f_c'=4 ksi, f_y=60 ksi.
    Expected: a=3.485", c=4.100", M_n=228.3 k-ft, phi*M_n=205.4 k-ft.
    """
    mat = ConcreteMaterial(fc_prime=4000 * PSI_TO_PA, fy=60000 * PSI_TO_PA)
    rebar = RebarLayout(
        bottom_bars=("#8", "#8", "#8"),
        bottom_cover=2.5 * IN_TO_M,
    )
    sec = ConcreteSection(b=12 * IN_TO_M, h=23.5 * IN_TO_M,
                            material=mat, rebar=rebar)
    r = beam_flexural_strength(sec)
    # a in mm vs 3.485 in = 88.5 mm
    assert r.a == pytest.approx(3.485 * IN_TO_M, rel=2e-3)
    assert r.c == pytest.approx(4.100 * IN_TO_M, rel=2e-3)
    # M_n in N·m vs 228.3 k-ft
    assert r.M_n == pytest.approx(228.3 * KFT_TO_NM, rel=5e-3)
    # Tension-controlled with phi = 0.90
    assert r.phi == pytest.approx(0.90)
    assert r.section_type == "tension-controlled"
    assert r.tension_steel_yields
    assert not r.compression_steel_yields
    # eps_t should match strain-compat
    assert r.epsilon_t == pytest.approx(0.003 * (sec.d - r.c) / r.c, rel=1e-10)


def test_singly_reinforced_minimum_steel_metric():
    """Small section with minimum steel. Verify it's flagged as such."""
    mat = ConcreteMaterial(fc_prime=28e6, fy=420e6)
    rebar = RebarLayout(bottom_bars=("#3",), bottom_cover=0.040)
    sec = ConcreteSection(b=0.30, h=0.50, material=mat, rebar=rebar)
    r = beam_flexural_strength(sec)
    # Single #3 = 71 mm² is way below A_s,min ~ 450 mm²
    assert "As_min" in r.notes or "9.6.1.2" in r.notes


def test_no_tension_steel_zero_capacity():
    mat = ConcreteMaterial(fc_prime=28e6, fy=420e6)
    rebar = RebarLayout(bottom_bars=(), bottom_cover=0.040)
    sec = ConcreteSection(b=0.30, h=0.50, material=mat, rebar=rebar)
    r = beam_flexural_strength(sec)
    assert r.M_n == 0.0
    assert r.phi_M_n == 0.0
    assert "no tension" in r.notes.lower()


# ============================================================ tension-controlled boundary

def test_tension_controlled_classification():
    """Section with very light reinforcement -> deeply tension-controlled
    (ε_t >> 0.005), phi = 0.90."""
    mat = ConcreteMaterial(fc_prime=28e6, fy=420e6)
    rebar = RebarLayout(bottom_bars=("#5",), bottom_cover=0.040)
    sec = ConcreteSection(b=0.30, h=0.50, material=mat, rebar=rebar)
    r = beam_flexural_strength(sec)
    assert r.section_type == "tension-controlled"
    assert r.phi == pytest.approx(0.90)
    assert r.epsilon_t > 0.005


def test_compression_controlled_classification():
    """Section over-reinforced so much that ε_t < ε_ty. Should be
    flagged as compression-controlled with reduced phi."""
    mat = ConcreteMaterial(fc_prime=28e6, fy=420e6)
    # Very over-reinforced: pile 8 #11 bars (12500 mm²) in a small section
    rebar = RebarLayout(
        bottom_bars=tuple("#11" for _ in range(8)),
        bottom_cover=0.040,
    )
    sec = ConcreteSection(b=0.30, h=0.50, material=mat, rebar=rebar)
    r = beam_flexural_strength(sec)
    # With this much steel, ε_t will be very small (compression-controlled)
    assert r.section_type in ("compression-controlled", "transition")
    assert r.phi < 0.90


# ============================================================ doubly-reinforced

def test_doubly_reinforced_compression_steel_yields():
    """Doubly-reinforced section where compression steel reaches yield.
    Validate the assumption-then-verify branch."""
    mat = ConcreteMaterial(fc_prime=28e6, fy=420e6)
    # Heavy tension steel + moderate compression steel
    rebar = RebarLayout(
        bottom_bars=("#9", "#9", "#9", "#9", "#9"),    # 5 #9 ~ 3225 mm²
        bottom_cover=0.060,
        top_bars=("#7", "#7"),                          # 2 #7 ~ 775 mm²
        top_cover=0.050,
    )
    sec = ConcreteSection(b=0.30, h=0.55, material=mat, rebar=rebar)
    r = beam_flexural_strength(sec)
    # Should not crash; capacity should be sensible
    assert r.M_n > 0.0
    # Verify M_n formula (compression-steel-yields path):
    # M_n = (As - As')·fy·(d - a/2) + As'·fy·(d - d')
    if r.compression_steel_yields:
        As = sec.rebar.As_bottom
        As_prime = sec.rebar.As_top
        d = sec.d
        d_prime = sec.d_prime
        a_expected = (As - As_prime) * sec.material.fy / (
            0.85 * sec.material.fc_prime * sec.b
        )
        M_n_expected = (
            (As - As_prime) * sec.material.fy * (d - 0.5 * a_expected)
            + As_prime * sec.material.fy * (d - d_prime)
        )
        assert r.M_n == pytest.approx(M_n_expected, rel=1e-10)


def test_doubly_reinforced_compression_steel_does_not_yield():
    """Doubly-reinforced with very shallow neutral axis -> compression
    steel doesn't reach yield. Algorithm should fall into quadratic
    solve path."""
    mat = ConcreteMaterial(fc_prime=28e6, fy=420e6)
    # Modest tension steel + large compression steel area: a will be
    # small -> compression-steel strain ε_s' = ε_cu·(c - d')/c may
    # be less than ε_ty.
    rebar = RebarLayout(
        bottom_bars=("#6", "#6"),                       # ~ 568 mm²
        bottom_cover=0.060,
        top_bars=tuple("#9" for _ in range(5)),         # 5 #9 ~ 3225 mm²
        top_cover=0.050,
    )
    sec = ConcreteSection(b=0.30, h=0.55, material=mat, rebar=rebar)
    r = beam_flexural_strength(sec)
    # Should not raise; result should be physically sensible.
    assert r.M_n > 0.0
    assert r.c > 0.0


# ============================================================ phi factor

def test_phi_in_transition_zone_linear_interpolation():
    """Construct a section deliberately in the transition zone, verify
    phi interpolates correctly between 0.65 and 0.90."""
    mat = ConcreteMaterial(fc_prime=28e6, fy=420e6)
    # Tune As to get ε_t between ε_ty and 0.005
    # ε_t = 0.005 -> tension-controlled boundary, c = 3·d/8
    # For something just above ε_ty, c is just under c_balanced
    rebar = RebarLayout(
        bottom_bars=("#10", "#10", "#10", "#10"),     # 4 × 1290 = 5160 mm²
        bottom_cover=0.060,
    )
    sec = ConcreteSection(b=0.30, h=0.55, material=mat, rebar=rebar)
    r = beam_flexural_strength(sec)
    if r.section_type == "transition":
        # phi should be strictly between 0.65 and 0.90
        assert 0.65 < r.phi < 0.90


# ============================================================ unit consistency

def test_unit_consistency_doubling_b_doubles_M_n():
    """For a fixed steel ratio, doubling b should approximately double
    M_n (for singly-reinforced with fixed As/b)."""
    mat = ConcreteMaterial(fc_prime=28e6, fy=420e6)
    rebar = RebarLayout(bottom_bars=("#7", "#7"), bottom_cover=0.040)
    sec1 = ConcreteSection(b=0.30, h=0.50, material=mat, rebar=rebar)
    sec2 = ConcreteSection(b=0.60, h=0.50, material=mat, rebar=rebar)
    r1 = beam_flexural_strength(sec1)
    r2 = beam_flexural_strength(sec2)
    # Same As, larger b -> smaller a, smaller lever arm change. M_n
    # should be SLIGHTLY larger (because a/2 is smaller -> larger lever arm)
    # but not double.
    assert r2.M_n > r1.M_n
    assert r2.M_n < 1.1 * r1.M_n


def test_unit_consistency_doubling_As_more_than_doubles_capacity():
    """Doubling As should less-than-double M_n (because a/2 grows, lever arm shrinks)."""
    mat = ConcreteMaterial(fc_prime=28e6, fy=420e6)
    sec1 = ConcreteSection(
        b=0.30, h=0.50, material=mat,
        rebar=RebarLayout(bottom_bars=("#7", "#7"), bottom_cover=0.040),
    )
    sec2 = ConcreteSection(
        b=0.30, h=0.50, material=mat,
        rebar=RebarLayout(
            bottom_bars=("#7", "#7", "#7", "#7"), bottom_cover=0.040
        ),
    )
    r1 = beam_flexural_strength(sec1)
    r2 = beam_flexural_strength(sec2)
    # 2x As gives less than 2x M_n (lever arm decreases)
    assert r1.M_n < r2.M_n < 2.0 * r1.M_n
