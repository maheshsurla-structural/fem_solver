"""P-Δ modal — ``EigenAnalysis(stiffness="tangent")`` (E2d).

Modes of a preloaded structure use the tangent stiffness ``K + K_g`` at the
committed state, so an axial compression softens the natural frequencies:
``omega(P) = omega0 * sqrt(1 - P/Pcr)`` (the classical beam-column interaction).
On an unstressed model the tangent option coincides with the elastic one.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from femsolver import (BeamColumn2D, ElasticIsotropic,  # noqa: E402
                       LinearBucklingAnalysis, LinearStaticAnalysis, Model)
from femsolver.analysis.eigen import EigenAnalysis  # noqa: E402


def _column(n=10):
    """A cantilever column of ``n`` elastic beam-column elements (fixed at 1)."""
    E, A, Iz, L, rho = 2.0e11, 1.0e-2, 8.333e-7, 3.0, 7850.0
    mat = ElasticIsotropic(1, E=E, nu=0.3, rho=rho)
    m = Model(ndm=2, ndf=3)
    m.add_material(mat)
    for i in range(n + 1):
        m.add_node(i + 1, i * L / n, 0.0)
    for i in range(n):
        m.add_element(BeamColumn2D(i + 1, (i + 1, i + 2), mat, A, Iz))
    m.fix(1, [1, 1, 1])
    return m, n


def _omega1(m, **kw):
    eig = EigenAnalysis(m, num_modes=2, **kw)
    eig.run()
    return math.sqrt(eig.eigenvalues[0])


def _pcr():
    m, n = _column()
    m.add_nodal_load(n + 1, [-1.0, 0.0, 0.0])
    b = LinearBucklingAnalysis(m, num_modes=1)
    b.run()
    return float(b.load_factors[0])


def _omega_under_axial(P, n=10):
    m, nn = _column(n)
    m.add_nodal_load(nn + 1, [-P, 0.0, 0.0])   # axial compression at free tip
    LinearStaticAnalysis(m).run()              # commit the stressed state
    return _omega1(m, stiffness="tangent")


def test_tangent_equals_elastic_when_unstressed():
    """No stress state → K_g = 0 → tangent modal coincides with elastic modal."""
    m1, _ = _column()
    m2, _ = _column()
    assert _omega1(m1, stiffness="tangent") == pytest.approx(
        _omega1(m2, stiffness="elastic"), rel=1e-9)


def test_pdelta_softening_follows_euler_formula():
    w0 = _omega1(_column()[0])
    Pcr = _pcr()
    ws = {f: _omega_under_axial(f * Pcr) for f in (0.25, 0.5, 0.75)}
    assert ws[0.25] > ws[0.5] > ws[0.75]            # monotonic softening
    for f, w in ws.items():                          # classical formula (±5%)
        assert w == pytest.approx(w0 * math.sqrt(1.0 - f), rel=0.05)


def test_pdelta_frequency_vanishes_near_buckling():
    """As P → Pcr the tangent stiffness → singular, so omega → 0."""
    w0 = _omega1(_column()[0])
    w = _omega_under_axial(0.95 * _pcr())
    assert w / w0 < 0.3


def test_unknown_stiffness_rejected():
    with pytest.raises(ValueError, match="stiffness"):
        EigenAnalysis(_column()[0], num_modes=1, stiffness="bogus")
