"""Force-based 3-D beam-column (plan P7 / G5) — P-M2-M3.

Elastic equivalence + integration-point invariance (§7.4 cross-checks), and
the nonlinear checks: reduces to the 2-D force-based element under uniaxial
bending, and produces genuine biaxial P-M2-M3 under skew loading.
"""
from __future__ import annotations

import numpy as np
import pytest

from femsolver import (
    BeamColumn3D,
    ElasticIsotropic,
    FiberSection2D,
    FiberSection3D,
    ForceBeamColumn2DCorotational,
    ForceBeamColumn3D,
    Model,
    NonlinearStaticAnalysis,
)
from femsolver.materials.uniaxial import UniaxialBilinear
from femsolver.sections.response.elastic import ElasticSection3D

E, NU = 30000.0, 0.3
G = E / (2.0 * (1.0 + NU))
A, IY, IZ, JT = 100.0, 800.0, 600.0, 300.0
L = 120.0


def _elastic_elem(cls, n_int=5):
    m = Model(ndm=3, ndf=6)
    mat = ElasticIsotropic(1, E=E, nu=NU)
    m.add_material(mat)
    m.add_node(1, 0.0, 0.0, 0.0)
    m.add_node(2, L, 0.0, 0.0)
    e = cls(1, (1, 2), mat, section=ElasticSection3D(E, G, A, IY, IZ, JT))
    e.n_int = n_int
    m.add_element(e)
    return e


# ------------------------------------------------------ elastic cross-checks

def test_elastic_stiffness_matches_displacement_based():
    """For an elastic prismatic section the force-based and displacement-based
    stiffnesses are identical (both exact)."""
    Kdb = _elastic_elem(BeamColumn3D).K_global()
    Kfb = _elastic_elem(ForceBeamColumn3D).K_global()
    assert np.max(np.abs(Kdb - Kfb)) / np.max(np.abs(Kdb)) < 1e-12
    assert np.allclose(Kfb, Kfb.T, atol=1e-6)          # symmetric


def test_n_ip_invariant_for_elastic():
    """The elastic result is invariant to the number of integration points
    (>= 3, needed to integrate the quadratic flexibility exactly)."""
    K3 = _elastic_elem(ForceBeamColumn3D, 3).K_global()
    K5 = _elastic_elem(ForceBeamColumn3D, 5).K_global()
    K6 = _elastic_elem(ForceBeamColumn3D, 6).K_global()
    assert np.max(np.abs(K3 - K5)) < 1e-6
    assert np.max(np.abs(K5 - K6)) < 1e-6


# ------------------------------------------------------ nonlinear behaviour

def _steel():
    return UniaxialBilinear(E=29000.0, sigma_y=60.0, b=0.02)


def _run_2d(P, dia=24.0, span=100.0):
    sec = FiberSection2D.circular(dia, 8, 24, _steel())
    m = Model(ndm=2, ndf=3)
    mat = ElasticIsotropic(1, E=29000.0, nu=NU)
    m.add_material(mat)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, span, 0.0)
    m.add_element(ForceBeamColumn2DCorotational(1, (1, 2), mat, section=sec))
    m.fix(1, [1, 1, 1])
    m.add_nodal_load(2, [0.0, -P, 0.0])
    r = NonlinearStaticAnalysis(m, num_steps=15, dlambda=1 / 15,
                                integrator="load_control", track=(2, 1),
                                tol=1e-8).run()
    return r["tracked"][-1], m.nodes[1].reaction[2]        # tip u_y, base Mz


def _run_3d(Py, Pz, dia=24.0, span=100.0):
    Jt = np.pi * dia ** 4 / 32.0
    sec = FiberSection3D.circular(dia, 8, 24, _steel(), GJ=G * Jt)
    m = Model(ndm=3, ndf=6)
    mat = ElasticIsotropic(1, E=29000.0, nu=NU)
    m.add_material(mat)
    m.add_node(1, 0.0, 0.0, 0.0)
    m.add_node(2, span, 0.0, 0.0)
    m.add_element(ForceBeamColumn3D(1, (1, 2), mat, section=sec))
    m.fix(1, [1, 1, 1, 1, 1, 1])
    m.add_nodal_load(2, [0.0, -Py, -Pz, 0.0, 0.0, 0.0])
    r = NonlinearStaticAnalysis(m, num_steps=15, dlambda=1 / 15,
                                integrator="load_control", track=(2, 1),
                                tol=1e-8).run()
    return r["tracked"][-1], m.nodes[1].reaction     # tip u_y, base reactions


def test_reduces_to_2d_under_uniaxial_bending():
    """Pushing the 3-D fiber column in one transverse direction reproduces the
    2-D force-based response (tip displacement and base moment), with the
    off-axis moment essentially zero."""
    P = 25.0
    uy2, Mz2 = _run_2d(P)
    uy3, R3 = _run_3d(P, 0.0)
    assert uy3 == pytest.approx(uy2, rel=1e-6)
    assert R3[5] == pytest.approx(Mz2, rel=1e-6)      # base Mz matches 2-D
    assert abs(R3[4]) < 1e-6 * abs(Mz2)               # My ~ 0


def test_biaxial_pm2m3():
    """Skew loading develops both base moments (genuine P-M2-M3). For the
    symmetric circular section a 45-degree push splits the capacity equally and
    the resultant matches the uniaxial base moment."""
    P = 25.0
    _uy, R = _run_3d(P / np.sqrt(2.0), P / np.sqrt(2.0))
    Mz, My = R[5], R[4]
    assert abs(Mz) > 1.0 and abs(My) > 1.0            # both non-trivial
    assert abs(Mz) == pytest.approx(abs(My), rel=1e-3)   # symmetric split
    _uy0, R0 = _run_3d(P, 0.0)
    resultant = np.hypot(Mz, My)
    assert resultant == pytest.approx(abs(R0[5]), rel=1e-3)
