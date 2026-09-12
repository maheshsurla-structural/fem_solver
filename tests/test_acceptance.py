"""ASCE 41 fibre-strain acceptance criteria (plan §16 C5)."""
from __future__ import annotations

import pytest

from femsolver.materials.uniaxial.concrete import ConcreteKentPark
from femsolver.materials.uniaxial.elastic import UniaxialElastic
from femsolver.materials.uniaxial.reinforcing import UniaxialReinforcingSteel
from femsolver.performance.acceptance import (CONCRETE_LIMITS, LEVELS,
                                              STEEL_LIMITS, classify_strain,
                                              default_limits, level_name,
                                              section_state)
from femsolver.sections.response.fiber import FiberSection2D


def test_concrete_compression_bands():
    L = CONCRETE_LIMITS
    assert classify_strain(-0.001, L) == 0            # below IO
    assert classify_strain(-0.004, L) == 1            # IO (>=0.003)
    assert classify_strain(-0.007, L) == 2            # LS (>=0.006)
    assert classify_strain(-0.020, L) == 3            # CP (>=0.015)


def test_concrete_tension_ignored():
    assert classify_strain(0.05, CONCRETE_LIMITS) == 0     # concrete cracks, no limit


def test_steel_tension_and_compression_bands():
    L = STEEL_LIMITS
    assert classify_strain(0.003, L) == 0
    assert classify_strain(0.012, L) == 1             # tension IO (>=0.01)
    assert classify_strain(0.03, L) == 2              # LS (>=0.02)
    assert classify_strain(0.06, L) == 3              # CP (>=0.05)
    assert classify_strain(-0.006, L) == 1            # compression IO (>=0.005)
    assert classify_strain(-0.025, L) == 3            # compression CP (>=0.02)


def test_default_limits_dispatch():
    assert default_limits(ConcreteKentPark(30e6, 0.002, 6e6, 0.004)) is CONCRETE_LIMITS
    assert default_limits(UniaxialReinforcingSteel(2e11, 5e8, 6.5e8,
                                                   0.008, 0.09)) is STEEL_LIMITS
    assert default_limits(UniaxialElastic(2e11)) is STEEL_LIMITS   # non-concrete


def test_section_state_escalates_with_curvature():
    mat = ConcreteKentPark(30e6, 0.002, 6e6, 0.02)
    sec = FiberSection2D.rectangular(width=0.3, height=0.6, n_fibers=20,
                                     material=mat)
    ymax = max(f.y for f in sec.fibers)               # extreme fibre offset
    # kappa chosen so the extreme compression strain hits each band
    assert section_state(sec.fibers, 0.0, 0.001 / ymax) == 0
    assert section_state(sec.fibers, 0.0, 0.004 / ymax) == 1   # IO
    assert section_state(sec.fibers, 0.0, 0.007 / ymax) == 2   # LS
    assert section_state(sec.fibers, 0.0, 0.02 / ymax) == 3    # CP


def test_level_name():
    assert level_name(0) == "Elastic"
    assert level_name(3) == "CP"
    assert level_name(99) == LEVELS[-1]               # clamped
