"""Tests for ShellDKT3 (Phase 22.8) -- thin-plate Discrete-Kirchhoff
triangular shell.

Validation: ShellDKT3 must give zero shear locking (scaled deflection
constant across all L/t) and must converge to the Kirchhoff thin-
plate answer with mesh refinement. Compared against ShellTri3 (which
locks as t -> 0) and against ShellMITC4 / ShellDKMQ4 on the same
problem (which serve as reference).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from femsolver import (
    ElasticIsotropic,
    Model,
    ShellDKMQ4,
    ShellDKT3,
    ShellMITC4,
    ShellTri3,
)
from femsolver.analysis.linear_static import LinearStaticAnalysis


# ======================================================== helpers

def _ss_plate_tri(element_cls, *, N: int, L: float = 1.0, t: float = 0.01,
                    E: float = 2.0e11, nu: float = 0.3):
    """N x N SS plate as 2 N x N triangles with central point load."""
    mat = ElasticIsotropic(1, E=E, nu=nu, rho=0.0)
    m = Model(ndm=3, ndf=6); m.add_material(mat)
    nL = N + 1
    for j in range(nL):
        for i in range(nL):
            m.add_node(j * nL + i + 1, i * L / N, j * L / N, 0.0)
    etag = 1
    for je in range(N):
        for ie in range(N):
            n1 = je * nL + ie + 1
            n2 = n1 + 1; n3 = n2 + nL; n4 = n1 + nL
            m.add_element(element_cls(etag, (n1, n2, n3), mat, thickness=t))
            etag += 1
            m.add_element(element_cls(etag, (n1, n3, n4), mat, thickness=t))
            etag += 1
    for j in range(nL):
        for i in range(nL):
            tag = j * nL + i + 1
            if i == 0 or i == nL - 1 or j == 0 or j == nL - 1:
                m.fix(tag, [1, 1, 1, 0, 0, 0])
    center = (nL // 2) * nL + nL // 2 + 1
    m.add_nodal_load(center, [0, 0, -1.0, 0, 0, 0])
    return m, center


def _unit_tri(*, E: float = 2.0e11, nu: float = 0.3, t: float = 0.01):
    """A single triangle with corners at (0,0), (1,0), (0,1)."""
    mat = ElasticIsotropic(1, E=E, nu=nu, rho=0.0)
    m = Model(ndm=3, ndf=6); m.add_material(mat)
    m.add_node(1, 0.0, 0.0, 0.0)
    m.add_node(2, 1.0, 0.0, 0.0)
    m.add_node(3, 0.0, 1.0, 0.0)
    m.add_element(ShellDKT3(1, (1, 2, 3), mat, thickness=t))
    return m


# ======================================================== construction

def test_dkt3_rejects_zero_thickness():
    mat = ElasticIsotropic(1, E=2e11, nu=0.3)
    with pytest.raises(ValueError, match="thickness"):
        ShellDKT3(1, [1, 2, 3], mat, thickness=0.0)


def test_dkt3_rejects_negative_drilling():
    mat = ElasticIsotropic(1, E=2e11, nu=0.3)
    with pytest.raises(ValueError, match="drilling_factor"):
        ShellDKT3(1, [1, 2, 3], mat, thickness=0.01, drilling_factor=-1.0)


# ======================================================== K properties

def test_dkt3_K_is_18x18_and_symmetric():
    m = _unit_tri()
    K = m.elements[1].K_global()
    assert K.shape == (18, 18)
    np.testing.assert_allclose(K, K.T, atol=1e-6 * np.max(np.abs(K)))


def test_dkt3_rigid_translation_zero_force():
    m = _unit_tri()
    K = m.elements[1].K_global()
    for dof in range(3):
        u = np.zeros(18)
        for i in range(3):
            u[6 * i + dof] = 1.0
        fint = K @ u
        assert np.linalg.norm(fint) < 1.0e-3 * np.max(np.abs(K))


# ======================================================== no shear locking

def test_dkt3_no_shear_locking():
    """DKT3 must give EXACTLY constant scaled deflection w*D across
    L/t -- the design intent. Any drift would indicate shear leakage
    in the formulation."""
    L = 1.0; E = 2e11; nu = 0.3
    scaled = []
    for t in (0.01, 0.001, 0.0001):
        D = E * t ** 3 / (12 * (1 - nu ** 2))
        m, c = _ss_plate_tri(ShellDKT3, N=4, L=L, t=t, E=E, nu=nu)
        LinearStaticAnalysis(m).run()
        w = -m.node(c).disp[2]
        scaled.append(w * D)
    s = np.asarray(scaled)
    # Exact constancy: ratio between max and min < 1e-8
    assert np.max(s) / np.min(s) == pytest.approx(1.0, abs=1e-8)


def test_dkt3_beats_tri3_on_thin_plate():
    """At L/t = 1000, DKT3 should give a sensible deflection while
    ShellTri3 locks to near-zero. The ratio DKT3/Tri3 should be > 100."""
    L = 1.0; t = 0.001; E = 2e11; nu = 0.3
    m_dk, c_dk = _ss_plate_tri(ShellDKT3, N=4, L=L, t=t, E=E, nu=nu)
    LinearStaticAnalysis(m_dk).run()
    w_dk = -m_dk.node(c_dk).disp[2]
    m_t, c_t = _ss_plate_tri(ShellTri3, N=4, L=L, t=t, E=E, nu=nu)
    LinearStaticAnalysis(m_t).run()
    w_t = -m_t.node(c_t).disp[2]
    assert w_dk / w_t > 100.0, (
        f"DKT3 ({w_dk:.3e}) should be >> ShellTri3 ({w_t:.3e}) at "
        f"L/t = 1000; ratio = {w_dk / w_t:.1f}"
    )


# ======================================================== convergence

def test_dkt3_converges_to_kirchhoff():
    """SS plate central deflection should converge to Kirchhoff theory
    0.01160 * P L^2 / D as N -> infinity."""
    L = 1.0; t = 0.01; E = 2e11; nu = 0.3
    D = E * t ** 3 / (12 * (1 - nu ** 2))
    w_kirch = 0.01160 * 1.0 * L ** 2 / D
    errors = []
    for N in (4, 8, 16):
        m, c = _ss_plate_tri(ShellDKT3, N=N, L=L, t=t, E=E, nu=nu)
        LinearStaticAnalysis(m).run()
        w = -m.node(c).disp[2]
        errors.append(abs(w - w_kirch) / w_kirch)
    # Monotonic decrease + N=16 within 2%
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 0.02


def test_dkt3_agrees_with_dkmq4_on_thin_plate():
    """DKT3 (triangles) and DKMQ4 (quads) should give the same answer
    on the same thin SS plate when both are refined enough."""
    L = 1.0; t = 0.001; E = 2e11; nu = 0.3

    # DKT3 N=8 (128 triangles)
    m_dkt, c_dkt = _ss_plate_tri(ShellDKT3, N=8, L=L, t=t, E=E, nu=nu)
    LinearStaticAnalysis(m_dkt).run()
    w_dkt = -m_dkt.node(c_dkt).disp[2]

    # DKMQ4 N=8 (64 quads, same node grid)
    mat = ElasticIsotropic(1, E=E, nu=nu, rho=0.0)
    m_dkmq = Model(ndm=3, ndf=6); m_dkmq.add_material(mat)
    nL = 8 + 1
    for j in range(nL):
        for i in range(nL):
            m_dkmq.add_node(j * nL + i + 1, i * L / 8, j * L / 8, 0.0)
    etag = 1
    for je in range(8):
        for ie in range(8):
            n1 = je * nL + ie + 1
            n2 = n1 + 1; n3 = n2 + nL; n4 = n1 + nL
            m_dkmq.add_element(ShellDKMQ4(etag, (n1, n2, n3, n4), mat,
                                            thickness=t))
            etag += 1
    for j in range(nL):
        for i in range(nL):
            tag = j * nL + i + 1
            if i == 0 or i == nL - 1 or j == 0 or j == nL - 1:
                m_dkmq.fix(tag, [1, 1, 1, 0, 0, 0])
    center = (nL // 2) * nL + nL // 2 + 1
    m_dkmq.add_nodal_load(center, [0, 0, -1.0, 0, 0, 0])
    LinearStaticAnalysis(m_dkmq).run()
    w_dkmq = -m_dkmq.node(center).disp[2]
    # Same answer within 5%
    assert w_dkt == pytest.approx(w_dkmq, rel=5.0e-2)


# ======================================================== mass

def test_dkt3_mass_matrix_shape_and_total():
    mat = ElasticIsotropic(1, E=2e11, nu=0.3, rho=7800.0)
    m = Model(ndm=3, ndf=6); m.add_material(mat)
    m.add_node(1, 0.0, 0.0, 0.0)
    m.add_node(2, 1.0, 0.0, 0.0)
    m.add_node(3, 0.0, 1.0, 0.0)
    el = ShellDKT3(1, (1, 2, 3), mat, thickness=0.01)
    m.add_element(el)
    M = el.M_global()
    assert M.shape == (18, 18)
    # Translational lump check (unit-triangle area = 0.5)
    M_lumped = el.M_global(lumped=True)
    total_x = sum(M_lumped[6 * i, 6 * i] for i in range(3))
    expected = 7800.0 * 0.01 * 0.5    # rho * t * A
    assert total_x == pytest.approx(expected, rel=1e-10)


# ======================================================== recovery

def test_dkt3_recover_populates_gp_arrays():
    m, c = _ss_plate_tri(ShellDKT3, N=2)
    LinearStaticAnalysis(m).run()
    el = m.elements[1]
    # 3-point Hammer quadrature for bending
    assert len(el.gp_membrane_strain) == 3
    assert len(el.gp_bending_curvature) == 3
    assert len(el.gp_resultants) == 3
    # Each resultants vector: 3 membrane (N) + 3 bending (M) = 6
    assert el.gp_resultants[0].size == 6
