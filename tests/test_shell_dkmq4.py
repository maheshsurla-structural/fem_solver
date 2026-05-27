"""Tests for ShellDKMQ4 (Phase 22.7) -- thin-plate Discrete-Kirchhoff
quadrilateral.

Validation strategy: compare against the proven :class:`ShellMITC4`
on identical thin-plate problems, where both elements should
converge to the same answer (the Kirchhoff thin-plate solution).
DKMQ4 is expected to give the Kirchhoff answer immediately at
coarse meshes (its design intent) while MITC4 needs refinement.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from femsolver import (
    ElasticIsotropic,
    Model,
    ShellDKMQ4,
    ShellMITC4,
)
from femsolver.analysis.linear_static import LinearStaticAnalysis


# ======================================================== helpers

def _build_cantilever(element_cls, *, N: int, L: float = 1.0,
                       b: float = 1.0, t: float = 0.01,
                       E: float = 2.0e11, nu: float = 0.3):
    """N x N cantilever plate clamped at x=0, with unit total tip
    load distributed along the x=L edge."""
    mat = ElasticIsotropic(1, E=E, nu=nu, rho=0.0)
    m = Model(ndm=3, ndf=6); m.add_material(mat)
    nL = N + 1
    for j in range(nL):
        for i in range(nL):
            m.add_node(j * nL + i + 1, i * L / N, j * b / N, 0.0)
    etag = 1
    for je in range(N):
        for ie in range(N):
            n1 = je * nL + ie + 1
            n2 = n1 + 1; n3 = n2 + nL; n4 = n1 + nL
            m.add_element(element_cls(etag, (n1, n2, n3, n4), mat, thickness=t))
            etag += 1
    for j in range(nL):
        m.fix(j * nL + 1, [1, 1, 1, 1, 1, 1])
    F_per = 1.0 / nL
    for j in range(nL):
        m.add_nodal_load(j * nL + N + 1, [0, 0, -F_per, 0, 0, 0])
    return m, nL


def _build_ss_plate(element_cls, *, N: int, L: float = 1.0,
                     t: float = 0.01,
                     E: float = 2.0e11, nu: float = 0.3):
    """N x N simply-supported plate with central point load."""
    mat = ElasticIsotropic(1, E=E, nu=nu, rho=0.0)
    m = Model(ndm=3, ndf=6); m.add_material(mat)
    nL = N + 1
    h = L / N
    for j in range(nL):
        for i in range(nL):
            m.add_node(j * nL + i + 1, i * h, j * h, 0.0)
    etag = 1
    for je in range(N):
        for ie in range(N):
            n1 = je * nL + ie + 1
            n2 = n1 + 1; n3 = n2 + nL; n4 = n1 + nL
            m.add_element(element_cls(etag, (n1, n2, n3, n4), mat, thickness=t))
            etag += 1
    for j in range(nL):
        for i in range(nL):
            tag = j * nL + i + 1
            if i == 0 or i == nL - 1 or j == 0 or j == nL - 1:
                m.fix(tag, [1, 1, 1, 0, 0, 0])
    center = (nL // 2) * nL + nL // 2 + 1
    m.add_nodal_load(center, [0, 0, -1.0, 0, 0, 0])
    return m, center


# ======================================================== construction

def test_dkmq4_rejects_zero_thickness():
    mat = ElasticIsotropic(1, E=2e11, nu=0.3, rho=0.0)
    with pytest.raises(ValueError, match="thickness"):
        ShellDKMQ4(1, [1, 2, 3, 4], mat, thickness=0.0)


def test_dkmq4_rejects_negative_drilling_factor():
    mat = ElasticIsotropic(1, E=2e11, nu=0.3, rho=0.0)
    with pytest.raises(ValueError, match="drilling_factor"):
        ShellDKMQ4(1, [1, 2, 3, 4], mat,
                     thickness=0.01, drilling_factor=-0.1)


# ======================================================== K properties

def test_dkmq4_K_is_symmetric():
    mat = ElasticIsotropic(1, E=2e11, nu=0.3, rho=0.0)
    m = Model(ndm=3, ndf=6); m.add_material(mat)
    for i, (x, y) in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)], 1):
        m.add_node(i, x, y, 0.0)
    m.add_element(ShellDKMQ4(1, (1, 2, 3, 4), mat, thickness=0.01))
    K = m.elements[1].K_global()
    assert K.shape == (24, 24)
    np.testing.assert_allclose(K, K.T, atol=1e-6 * np.max(np.abs(K)))


def test_dkmq4_rigid_translation_gives_zero_force():
    mat = ElasticIsotropic(1, E=2e11, nu=0.3, rho=0.0)
    m = Model(ndm=3, ndf=6); m.add_material(mat)
    for i, (x, y) in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)], 1):
        m.add_node(i, x, y, 0.0)
    m.add_element(ShellDKMQ4(1, (1, 2, 3, 4), mat, thickness=0.01))
    K = m.elements[1].K_global()
    # All three rigid translations
    for dof in range(3):
        u = np.zeros(24)
        for i in range(4):
            u[6 * i + dof] = 1.0
        fint = K @ u
        assert np.linalg.norm(fint) < 1.0e-3 * np.max(np.abs(K)), (
            f"rigid translation in DOF {dof} should give ~zero force"
        )


# ======================================================== thin-plate convergence

def test_dkmq4_cantilever_matches_mitc4_on_refined_mesh():
    """At N=16, DKMQ4 and MITC4 should give the same cantilever tip
    deflection within 1%."""
    m_dk, nL = _build_cantilever(ShellDKMQ4, N=16)
    LinearStaticAnalysis(m_dk).run()
    w_dk = -m_dk.node(nL).disp[2]
    m_mt, _ = _build_cantilever(ShellMITC4, N=16)
    LinearStaticAnalysis(m_mt).run()
    w_mt = -m_mt.node(nL).disp[2]
    assert w_dk == pytest.approx(w_mt, rel=1.0e-2)


def test_dkmq4_converges_with_refinement():
    """Successive refinements should approach a limit (within 1%
    change between N=8 and N=16)."""
    deflections = []
    for N in (4, 8, 16):
        m, nL = _build_cantilever(ShellDKMQ4, N=N)
        LinearStaticAnalysis(m).run()
        deflections.append(-m.node(nL).disp[2])
    # Successive refinements give a converging sequence (decreasing rate)
    delta_4_8 = abs(deflections[1] - deflections[0])
    delta_8_16 = abs(deflections[2] - deflections[1])
    # Rate of change should diminish (Cauchy-like)
    assert delta_8_16 < delta_4_8 or delta_8_16 < 0.01 * abs(deflections[2])


def test_dkmq4_no_shear_locking_as_thin_limit():
    """DKMQ4 has NO transverse-shear strain energy by construction
    (DK constraint), so the scaled deflection w*D/P should be exactly
    constant across L/t -- there is no locking to occur.
    """
    L = 1.0; E = 2e11; nu = 0.3; P = 1.0
    scaled = []
    for t in (0.01, 0.001, 0.0001):
        D = E * t ** 3 / (12 * (1 - nu ** 2))
        m, center = _build_ss_plate(ShellDKMQ4, N=4, L=L, t=t, E=E, nu=nu)
        LinearStaticAnalysis(m).run()
        w = -m.node(center).disp[2]
        scaled.append(w * D / (P * L ** 2))
    s = np.asarray(scaled)
    # Constant to machine precision (no shear contribution -> only
    # bending energy, which scales as t^3 == D)
    assert np.max(s) / np.min(s) == pytest.approx(1.0, abs=1.0e-10)


# ======================================================== convergence vs MITC4

def test_dkmq4_ss_plate_converges_to_mitc4_answer():
    """On refined SS plate (N=16), DKMQ4 and MITC4 should agree
    within 5% (point-load problem is singular -> slower convergence
    for both)."""
    m_dk, c_dk = _build_ss_plate(ShellDKMQ4, N=16)
    LinearStaticAnalysis(m_dk).run()
    w_dk = -m_dk.node(c_dk).disp[2]
    m_mt, c_mt = _build_ss_plate(ShellMITC4, N=16)
    LinearStaticAnalysis(m_mt).run()
    w_mt = -m_mt.node(c_mt).disp[2]
    assert w_dk == pytest.approx(w_mt, rel=5.0e-2)


# ======================================================== mass

def test_dkmq4_mass_matrix_shape_and_total():
    mat = ElasticIsotropic(1, E=2e11, nu=0.3, rho=7800.0)
    m = Model(ndm=3, ndf=6); m.add_material(mat)
    for i, (x, y) in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)], 1):
        m.add_node(i, x, y, 0.0)
    el = ShellDKMQ4(1, (1, 2, 3, 4), mat, thickness=0.01)
    m.add_element(el)
    M = el.M_global()
    assert M.shape == (24, 24)
    M_lumped = el.M_global(lumped=True)
    total_x_lumped = sum(M_lumped[6 * i, 6 * i] for i in range(4))
    expected = 7800.0 * 0.01 * 1.0     # rho * t * area
    assert total_x_lumped == pytest.approx(expected, rel=1e-10)


# ======================================================== recovery

def test_dkmq4_recover_populates_gp_arrays():
    m, nL = _build_cantilever(ShellDKMQ4, N=2)
    LinearStaticAnalysis(m).run()
    el = m.elements[1]
    # 3x3 Gauss for bending recovery
    assert len(el.gp_membrane_strain) == 9
    assert len(el.gp_bending_curvature) == 9
    assert len(el.gp_resultants) == 9
    # Each resultants vector: 3 membrane (N) + 3 bending (M) = 6
    assert el.gp_resultants[0].size == 6
