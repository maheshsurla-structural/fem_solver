"""Dynamic unit system core (plan U1) — pure conversion + label coverage.

No Qt: ``units.py`` is GUI-free, so this runs anywhere.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "desktop"))

from units import (FORCE_UNITS, LENGTH_UNITS, Quantity,  # noqa: E402
                   UnitSystem)


def test_default_is_kn_m():
    u = UnitSystem()
    assert (u.force, u.length) == ("kN", "m")
    assert u.pair_label == "kN · m"


def test_unknown_units_rejected():
    with pytest.raises(ValueError):
        UnitSystem(force="slug")
    with pytest.raises(ValueError):
        UnitSystem(length="league")


def test_force_roundtrip_kn():
    u = UnitSystem("kN", "m")
    # 5 kN entered → 5000 N stored → shown back as 5 kN
    assert u.to_si(5.0, Quantity.FORCE) == pytest.approx(5000.0)
    assert u.to_display(5000.0, Quantity.FORCE) == pytest.approx(5.0)


def test_length_mm():
    u = UnitSystem("N", "mm")
    assert u.to_si(1000.0, Quantity.LENGTH) == pytest.approx(1.0)  # 1000 mm = 1 m
    assert u.to_display(0.25, Quantity.LENGTH) == pytest.approx(250.0)


def test_moment_is_force_times_length():
    u = UnitSystem("kN", "m")
    # 1 kN·m = 1000 N·m
    assert u.factor(Quantity.MOMENT) == pytest.approx(1000.0)
    assert u.to_display(1000.0, Quantity.MOMENT) == pytest.approx(1.0)


def test_stress_force_per_area():
    u = UnitSystem("kN", "m")           # kN/m² = 1000 Pa
    assert u.factor(Quantity.STRESS) == pytest.approx(1000.0)
    # 25 MPa (=25e6 Pa) shown in kN/m²
    assert u.to_display(25.0e6, Quantity.STRESS) == pytest.approx(25000.0)


def test_stress_n_mm2_is_mpa():
    u = UnitSystem("N", "mm")           # N/mm² == MPa == 1e6 Pa
    assert u.factor(Quantity.STRESS) == pytest.approx(1.0e6)
    assert u.to_display(30.0e6, Quantity.STRESS) == pytest.approx(30.0)


def test_dist_load_force_per_length():
    u = UnitSystem("kN", "m")           # kN/m = 1000 N/m
    assert u.factor(Quantity.DIST_LOAD) == pytest.approx(1000.0)


def test_area_and_inertia_exponents():
    u = UnitSystem("N", "mm")
    assert u.factor(Quantity.AREA) == pytest.approx(1.0e-6)   # mm² = 1e-6 m²
    assert u.factor(Quantity.INERTIA) == pytest.approx(1.0e-12)  # mm⁴


def test_kgf_and_tonf_factors():
    assert FORCE_UNITS["kgf"] == pytest.approx(9.80665)
    u = UnitSystem("tonf", "m")
    assert u.to_si(1.0, Quantity.FORCE) == pytest.approx(9806.65)


def test_imperial_force_factors():
    # 1 kip = 1000 lbf; lbf uses the exact international pound-force
    assert FORCE_UNITS["lbf"] == pytest.approx(4.4482216152605)
    assert FORCE_UNITS["kip"] == pytest.approx(1000.0 * FORCE_UNITS["lbf"])
    assert UnitSystem("kip", "ft").to_si(1.0, Quantity.FORCE) == \
        pytest.approx(4448.2216152605)


def test_imperial_length_factors():
    assert LENGTH_UNITS["in"] == pytest.approx(0.0254)
    assert LENGTH_UNITS["ft"] == pytest.approx(0.3048)
    # 12 in == 1 ft
    assert 12 * LENGTH_UNITS["in"] == pytest.approx(LENGTH_UNITS["ft"])
    u = UnitSystem("kip", "ft")
    assert u.to_display(3.048, Quantity.LENGTH) == pytest.approx(10.0)  # 3.048 m


def test_kip_ft_moment_roundtrip():
    u = UnitSystem("kip", "ft")
    # 1 kip·ft = 4448.2216152605 * 0.3048 N·m
    si = 4448.2216152605 * 0.3048
    assert u.factor(Quantity.MOMENT) == pytest.approx(si)
    assert u.to_display(si, Quantity.MOMENT) == pytest.approx(1.0)


def test_ksi_and_psi_are_compositional():
    # kip/in² == "ksi" (composed, not a special name); 1 ksi = 6.894757e6 Pa
    u = UnitSystem("kip", "in")
    assert u.label(Quantity.STRESS) == "kip/in²"
    assert u.factor(Quantity.STRESS) == pytest.approx(6.894757293168e6, rel=1e-9)
    # psi = lbf/in²
    p = UnitSystem("lbf", "in")
    assert p.label(Quantity.STRESS) == "lbf/in²"
    assert p.factor(Quantity.STRESS) == pytest.approx(6894.757293168, rel=1e-9)


def test_inch_fourth_inertia():
    u = UnitSystem("kip", "in")
    assert u.factor(Quantity.INERTIA) == pytest.approx(0.0254 ** 4)


def test_rotation_is_unit_invariant():
    for f in FORCE_UNITS:
        for l in LENGTH_UNITS:
            u = UnitSystem(f, l)
            assert u.factor(Quantity.ROTATION) == 1.0
            assert u.label(Quantity.ROTATION) == "rad"


@pytest.mark.parametrize("force,length,qty,expected", [
    ("kN", "m", Quantity.FORCE, "kN"),
    ("kN", "m", Quantity.MOMENT, "kN·m"),
    ("kN", "m", Quantity.STRESS, "kN/m²"),
    ("kN", "m", Quantity.DIST_LOAD, "kN/m"),
    ("N", "mm", Quantity.AREA, "mm²"),
    ("N", "mm", Quantity.INERTIA, "mm⁴"),
    ("N", "mm", Quantity.STRESS, "N/mm²"),
    ("kN", "cm", Quantity.LENGTH, "cm"),
])
def test_labels(force, length, qty, expected):
    assert UnitSystem(force, length).label(qty) == expected


def test_fmt_with_and_without_label():
    u = UnitSystem("kN", "m")
    assert u.fmt(1234.0, Quantity.FORCE, decimals=2) == "1.23 kN"
    assert u.fmt(1234.0, Quantity.FORCE, decimals=2, with_label=False) == "1.23"


def test_from_project_tolerates_legacy():
    class P:
        force_unit = "N"
        length_unit = "m"
    assert UnitSystem.from_project(P()) == UnitSystem("N", "m")

    class Bad:
        force_unit = "bogus"
        length_unit = "m"
    # falls back to default force, keeps valid length
    assert UnitSystem.from_project(Bad()) == UnitSystem("kN", "m")


def test_factor_matches_manual_composition():
    # spot-check that factor() == force^fe · length^le for a tricky case
    u = UnitSystem("tonf", "cm")
    # stress = F/L² = 9806.65 / (0.01)² = 9806.65 / 1e-4
    assert u.factor(Quantity.STRESS) == pytest.approx(9806.65 / 1.0e-4)
