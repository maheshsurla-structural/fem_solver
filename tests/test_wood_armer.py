"""Wood-Armer slab design moments + required reinforcement (slab plan S9)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from femsolver.design.wood_armer import (  # noqa: E402
    required_reinforcement,
    wood_armer_moments,
)


# ------------------------------------------------------ design moments

def test_pure_twist_needs_equal_top_and_bottom_steel():
    r = wood_armer_moments(0.0, 0.0, 5.0)
    assert r.mx_bot == pytest.approx(5.0)
    assert r.my_bot == pytest.approx(5.0)
    assert r.mx_top == pytest.approx(-5.0)
    assert r.my_top == pytest.approx(-5.0)


def test_no_twist_all_sagging_passes_through():
    r = wood_armer_moments(10.0, 5.0, 0.0)
    assert (r.mx_bot, r.my_bot) == pytest.approx((10.0, 5.0))
    assert (r.mx_top, r.my_top) == pytest.approx((0.0, 0.0))   # no hogging


def test_no_twist_all_hogging_only_top_steel():
    r = wood_armer_moments(-10.0, -5.0, 0.0)
    assert (r.mx_bot, r.my_bot) == pytest.approx((0.0, 0.0))
    assert (r.mx_top, r.my_top) == pytest.approx((-10.0, -5.0))


def test_bottom_correction_when_one_direction_goes_negative():
    # Mx + |Mxy| < 0 → Mx* clamps to 0 and My* is boosted by |Mxy²/Mx|
    r = wood_armer_moments(-6.0, 5.0, 4.0)
    assert r.mx_bot == pytest.approx(0.0)
    assert r.my_bot == pytest.approx(5.0 + 16.0 / 6.0)


def test_bottom_design_moment_never_below_raw_moment():
    r = wood_armer_moments(8.0, 3.0, 2.5)
    assert r.mx_bot >= 8.0 and r.my_bot >= 3.0     # twist only adds demand


# --------------------------------------------------- required steel

def test_required_reinforcement_matches_stress_block():
    As = required_reinforcement(100e3, d=0.15, fy=420e6, fc=30e6)
    k = 0.85 * 30e6
    disc = 0.15 ** 2 - 2 * 100e3 / (0.9 * k)
    expected = (k / 420e6) * (0.15 - math.sqrt(disc))
    assert As == pytest.approx(expected, rel=1e-9)
    assert 1.5e-3 < As < 2.5e-3                    # ~1979 mm²/m, sane


def test_required_reinforcement_zero_moment():
    assert required_reinforcement(0.0, d=0.15, fy=420e6, fc=30e6) == 0.0


def test_required_reinforcement_uses_magnitude():
    a = required_reinforcement(80e3, d=0.15, fy=420e6, fc=30e6)
    b = required_reinforcement(-80e3, d=0.15, fy=420e6, fc=30e6)
    assert a == pytest.approx(b)                    # top or bottom, same area


def test_required_reinforcement_inadequate_section_raises():
    with pytest.raises(ValueError):
        required_reinforcement(2000e3, d=0.15, fy=420e6, fc=30e6)
