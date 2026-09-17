"""Buckling from a committed state — ``LinearBucklingAnalysis(prestress=
"current_state")`` (E2e).

Buckling from an already-committed (e.g. nonlinear-preload) state must give the
same critical factor as the standalone reference-load buckling under that same
load — the option only changes *where* the prestress comes from, not the physics.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from femsolver import (BeamColumn2D, ElasticIsotropic,  # noqa: E402
                       LinearBucklingAnalysis, LinearStaticAnalysis, Model)


def _column(n=10):
    E, A, Iz, L = 2.0e11, 1.0e-2, 8.333e-7, 3.0
    mat = ElasticIsotropic(1, E=E, nu=0.3, rho=0.0)
    m = Model(ndm=2, ndf=3)
    m.add_material(mat)
    for i in range(n + 1):
        m.add_node(i + 1, i * L / n, 0.0)
    for i in range(n):
        m.add_element(BeamColumn2D(i + 1, (i + 1, i + 2), mat, A, Iz))
    m.fix(1, [1, 1, 1])
    return m, n


def test_buckling_from_state_matches_reference_and_euler():
    P = 3.0e4
    m1, n = _column()
    m1.add_nodal_load(n + 1, [-P, 0.0, 0.0])
    lam_ref = float(LinearBucklingAnalysis(
        m1, num_modes=1).run()["load_factors"][0])

    m2, n = _column()
    m2.add_nodal_load(n + 1, [-P, 0.0, 0.0])
    LinearStaticAnalysis(m2).run()               # commit the stressed state
    lam_state = float(LinearBucklingAnalysis(
        m2, num_modes=1, prestress="current_state").run()["load_factors"][0])

    assert lam_state == pytest.approx(lam_ref, rel=1e-6)      # consistency
    Pcr = math.pi**2 * 2.0e11 * 8.333e-7 / (4 * 3.0**2)       # cantilever Euler
    assert lam_state * P == pytest.approx(Pcr, rel=0.02)


def test_unknown_prestress_rejected():
    with pytest.raises(ValueError, match="prestress"):
        LinearBucklingAnalysis(_column()[0], prestress="bogus")
