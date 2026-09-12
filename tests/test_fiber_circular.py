"""Fiber-hinge plan P1 (G1) — circular fiber-section builders.

Covers:

* ``circular_sector_fibers`` — exact annulus area.
* ``FiberSection2D.circular`` / ``FiberSection3D.circular`` — exact gross
  area, monotone convergence of ``Iz`` (and ``Iy`` in 3-D) to the analytic
  circle, symmetric centroid, error handling.
* ``rc_circular_column_section`` — core/cover/rebar assembly, exact total
  area with rebar subtracted from the core, transformed axial stiffness,
  2-D and 3-D variants, and argument validation.

See ``docs/source/fiber_hinge_implementation_plan.md`` §6/§7.4.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from femsolver.materials.uniaxial.elastic import UniaxialElastic
from femsolver.sections.response.fiber import (
    FiberSection2D,
    FiberSection3D,
    circular_sector_fibers,
)
from femsolver.sections.response.fiber_build import rc_circular_column_section

D = 84.0  # benchmark column diameter (in)
A_EXACT = math.pi * D * D / 4.0
I_EXACT = math.pi * D**4 / 64.0


# ------------------------------------------------------ low-level meshing

def test_annulus_area_exact():
    m = UniaxialElastic(1.0)
    r_in, r_out = 10.0, 42.0
    fibers = circular_sector_fibers(r_in, r_out, 5, 24, m)
    area = sum(f.area for f in fibers)
    assert area == pytest.approx(math.pi * (r_out**2 - r_in**2), rel=1e-12)
    assert len(fibers) == 5 * 24


def test_sector_fibers_validation():
    m = UniaxialElastic(1.0)
    with pytest.raises(ValueError):
        circular_sector_fibers(42.0, 10.0, 4, 12, m)   # r_out <= r_in
    with pytest.raises(ValueError):
        circular_sector_fibers(0.0, 42.0, 0, 12, m)    # n_rings < 1
    with pytest.raises(ValueError):
        circular_sector_fibers(0.0, 42.0, 4, 1, m)     # n_wedges < 2


# ------------------------------------------------------ FiberSection2D.circular

def test_circular2d_area_exact_all_meshes():
    m = UniaxialElastic(3605.0)
    for nr, nw in [(2, 8), (4, 12), (8, 24), (16, 48)]:
        s = FiberSection2D.circular(D, nr, nw, m)
        assert s.gross_area == pytest.approx(A_EXACT, rel=1e-10)
        assert len(s.fibers) == nr * nw
        assert s.centroid_y == pytest.approx(0.0, abs=1e-9)


def test_circular2d_Iz_converges_monotonically():
    m = UniaxialElastic(3605.0)
    errs = []
    for nr, nw in [(2, 8), (4, 16), (8, 32), (16, 64)]:
        s = FiberSection2D.circular(D, nr, nw, m)
        errs.append(abs(s.gross_Iz - I_EXACT) / I_EXACT)
    # strictly decreasing error, and fine mesh within 0.5%
    assert all(errs[i + 1] < errs[i] for i in range(len(errs) - 1)), errs
    assert errs[-1] < 5e-3, errs


def test_circular2d_validation():
    m = UniaxialElastic(1.0)
    with pytest.raises(ValueError):
        FiberSection2D.circular(0.0, 8, 24, m)         # diameter <= 0
    with pytest.raises(ValueError):
        FiberSection2D.circular(D, 8, 1, m)            # n_wedges < 2


# ------------------------------------------------------ FiberSection3D.circular

def test_circular3d_symmetry_and_area():
    m = UniaxialElastic(3605.0)
    s = FiberSection3D.circular(D, 8, 24, m, GJ=1.0e9)
    assert s.gross_area == pytest.approx(A_EXACT, rel=1e-10)
    # Iz == Iy by symmetry; product of inertia ~ 0
    assert s.gross_Iz == pytest.approx(s.gross_Iy, rel=1e-9)
    assert s.gross_Iyz == pytest.approx(0.0, abs=1e-6 * I_EXACT)
    assert s.GJ == 1.0e9


def test_circular3d_independent_fiber_state():
    m = UniaxialElastic(3605.0)
    s = FiberSection3D.circular(D, 4, 12, m, GJ=1.0)
    mats = {id(f.material) for f in s.fibers}
    assert len(mats) == len(s.fibers)   # each fiber cloned


# ------------------------------------------------------ rc_circular_column_section

def _benchmark_section(**kw):
    """Caltrans 84in column: 56 #14, 2in cover, distinct core/cover E."""
    return rc_circular_column_section(
        diameter=84.0, cover=2.0,
        core_concrete=UniaxialElastic(3605.0),
        cover_concrete=UniaxialElastic(3000.0),
        n_bars=56, bar_area=2.25, steel=UniaxialElastic(29000.0),
        **kw,
    )


def test_rc_section_fiber_counts_and_total_area():
    s = _benchmark_section(n_core_rings=8, n_cover_rings=2, n_wedges=24)
    # 8*24 core + 2*24 cover + 56 bars
    assert len(s.fibers) == 8 * 24 + 2 * 24 + 56
    # steel fibers = 56, each area 2.25
    steel = [f for f in s.fibers if f.area == pytest.approx(2.25)]
    assert len(steel) == 56
    # total (concrete + steel) area == gross section area exactly
    assert s.gross_area == pytest.approx(A_EXACT, rel=1e-9)


def test_rc_section_transformed_axial_stiffness():
    """EA (= ks[0,0]) must equal Ec_core*A_core + Ec_cover*A_cover + Es*As."""
    Ecore, Ecover, Es = 3605.0, 3000.0, 29000.0
    s = _benchmark_section()
    _, ks = s.get_response(np.array([1.0e-6, 0.0]))
    EA = ks[0, 0]

    R = 42.0
    core_r = R - 2.0
    A_core_gross = math.pi * core_r**2
    A_cover = math.pi * (R**2 - core_r**2)
    As = 56 * 2.25
    A_core_conc = A_core_gross - As        # rebar subtracted from core
    EA_expected = Ecore * A_core_conc + Ecover * A_cover + Es * As
    assert EA == pytest.approx(EA_expected, rel=1e-9)


def test_rc_section_no_subtract_keeps_full_core_concrete():
    s = _benchmark_section(subtract_rebar_from_core=False)
    # total area = gross concrete + steel (steel double-counts its hole)
    assert s.gross_area == pytest.approx(A_EXACT + 56 * 2.25, rel=1e-9)


def test_rc_section_3d_variant():
    s = rc_circular_column_section(
        diameter=84.0, cover=2.0,
        core_concrete=UniaxialElastic(3605.0),
        cover_concrete=UniaxialElastic(3000.0),
        n_bars=56, bar_area=2.25, steel=UniaxialElastic(29000.0),
        three_d=True, GJ=2.0e9,
    )
    assert isinstance(s, FiberSection3D)
    assert s.gross_area == pytest.approx(A_EXACT, rel=1e-9)
    assert s.gross_Iz == pytest.approx(s.gross_Iy, rel=1e-6)


def test_rc_section_validation():
    args = dict(
        diameter=84.0,
        core_concrete=UniaxialElastic(3605.0),
        cover_concrete=UniaxialElastic(3000.0),
        n_bars=56, bar_area=2.25, steel=UniaxialElastic(29000.0),
    )
    with pytest.raises(ValueError):
        rc_circular_column_section(cover=50.0, **args)        # cover >= D/2
    with pytest.raises(ValueError):
        rc_circular_column_section(cover=2.0, three_d=True, **args)   # GJ missing
    with pytest.raises(ValueError):
        # steel area exceeds gross core area
        rc_circular_column_section(cover=2.0, n_core_rings=8, n_cover_rings=2,
                                   bar_circle_diameter=70.0, **{**args, "bar_area": 200.0})
