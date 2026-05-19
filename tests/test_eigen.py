"""Modal analysis tests — frequencies validated against textbook
analytical solutions for cantilever, simply-supported, and axial bars."""
from __future__ import annotations

import math

import numpy as np
import pytest

from femsolver import (
    BeamColumn2D,
    BeamColumn3D,
    EigenAnalysis,
    ElasticIsotropic,
    Model,
    Quad4,
    Truss2D,
)


# ---------------------------------------------------------------------------
# Euler-Bernoulli beam reference frequencies


# (beta_n L)^2 for cantilever modes 1..3 (eigenvalues of cosh(x)cos(x)+1=0)
CANTILEVER_BETA_L_SQ = (1.875104068711961 ** 2,
                        4.694091132974175 ** 2,
                        7.854757438237613 ** 2)


def _make_cantilever(n_elem: int, L: float, E: float, rho: float, A: float, Iz: float) -> Model:
    m = Model(ndm=2, ndf=3)
    mat = ElasticIsotropic(1, E=E, nu=0.3, rho=rho)
    m.add_material(mat)
    dx = L / n_elem
    for i in range(n_elem + 1):
        m.add_node(i + 1, i * dx, 0.0)
    m.fix(1, [1, 1, 1])
    for i in range(n_elem):
        m.add_element(BeamColumn2D(i + 1, (i + 1, i + 2), mat, A, Iz))
    return m


def test_cantilever_first_three_frequencies_consistent():
    L, E, rho, A, Iz = 2.0, 200e9, 7850.0, 1.0e-3, 8.333e-7
    m = _make_cantilever(n_elem=20, L=L, E=E, rho=rho, A=A, Iz=Iz)
    a = EigenAnalysis(m, num_modes=3)
    a.run()
    base = math.sqrt(E * Iz / (rho * A * L ** 4))
    expected = np.array([blsq * base for blsq in CANTILEVER_BETA_L_SQ]) / (2.0 * math.pi)
    # 20 elements: first mode <0.1% error, second <0.5%, third <2%
    rel = (a.frequencies - expected) / expected
    assert abs(rel[0]) < 1e-3
    assert abs(rel[1]) < 5e-3
    assert abs(rel[2]) < 2e-2


def test_cantilever_lumped_converges():
    """With rotational lumped mass = 0, the first frequency converges from
    above on this scheme — but on a 40-element mesh it should be within
    1% of the analytical value."""
    L, E, rho, A, Iz = 1.5, 200e9, 7850.0, 1.0e-3, 8.333e-7
    m = _make_cantilever(40, L, E, rho, A, Iz)
    a = EigenAnalysis(m, num_modes=2, lumped=True)
    a.run()
    base = math.sqrt(E * Iz / (rho * A * L ** 4))
    f1_exact = CANTILEVER_BETA_L_SQ[0] * base / (2.0 * math.pi)
    assert abs(a.frequencies[0] - f1_exact) / f1_exact < 1e-2


# ---------------------------------------------------------------------------
# simply-supported beam: omega_n = (n pi)^2 * sqrt(EI / (rho A L^4))


def test_ss_beam_first_three_frequencies():
    L, E, rho, A, Iz = 4.0, 200e9, 7850.0, 1.0e-3, 8.333e-7
    n_elem = 30
    m = Model(ndm=2, ndf=3)
    mat = ElasticIsotropic(1, E=E, nu=0.3, rho=rho)
    m.add_material(mat)
    dx = L / n_elem
    for i in range(n_elem + 1):
        m.add_node(i + 1, i * dx, 0.0)
    # SS: pin at left (uy fixed), roller at right (uy fixed). theta_z is free
    # at both ends. The axial DOF u_x at the left is fixed too to remove
    # the rigid translation; right end u_x is free.
    m.fix(1, [1, 1, 0])
    m.fix(n_elem + 1, [0, 1, 0])
    for i in range(n_elem):
        m.add_element(BeamColumn2D(i + 1, (i + 1, i + 2), mat, A, Iz))
    a = EigenAnalysis(m, num_modes=3)
    a.run()
    base = math.sqrt(E * Iz / (rho * A * L ** 4))
    expected = np.array([(n * math.pi) ** 2 * base / (2.0 * math.pi) for n in (1, 2, 3)])
    rel = (a.frequencies - expected) / expected
    assert abs(rel[0]) < 1e-3
    assert abs(rel[1]) < 5e-3
    assert abs(rel[2]) < 1.5e-2


# ---------------------------------------------------------------------------
# axial truss vibration: fixed-free rod, omega_n = (2n-1) pi / (2 L) * sqrt(E/rho)


def test_axial_truss_modes():
    L, E, rho, A = 5.0, 2.0e11, 7850.0, 1e-3
    n_elem = 40
    m = Model(ndm=2, ndf=2)
    mat = ElasticIsotropic(1, E=E, nu=0.3, rho=rho)
    m.add_material(mat)
    dx = L / n_elem
    for i in range(n_elem + 1):
        m.add_node(i + 1, i * dx, 0.0)
    m.fix(1, [1, 1])
    for i in range(2, n_elem + 2):
        m.fix(i, [0, 1])  # constrain transverse motion (truss-only model)
    for i in range(n_elem):
        m.add_element(Truss2D(i + 1, (i + 1, i + 2), mat, A))
    a = EigenAnalysis(m, num_modes=3)
    a.run()
    c = math.sqrt(E / rho)
    expected = np.array([(2 * n - 1) * math.pi * c / (2.0 * L) / (2.0 * math.pi) for n in (1, 2, 3)])
    rel = (a.frequencies - expected) / expected
    assert abs(rel[0]) < 5e-4
    assert abs(rel[1]) < 5e-3
    assert abs(rel[2]) < 2e-2


# ---------------------------------------------------------------------------
# mass-orthonormal mode shapes


def test_modes_are_M_orthonormal():
    L, E, rho, A, Iz = 1.0, 2.0e11, 7850.0, 1e-3, 8e-7
    m = _make_cantilever(15, L, E, rho, A, Iz)
    a = EigenAnalysis(m, num_modes=4)
    a.run()
    # phi^T M phi should be identity (within numerical tolerance)
    M = a.M.toarray()
    phi = a.mode_shapes
    G = phi.T @ M @ phi
    np.testing.assert_allclose(G, np.eye(a.num_modes), atol=1e-8)


# ---------------------------------------------------------------------------
# 3D beam — at least matches the 2D cantilever first mode about its weak axis


def test_cantilever_3d_first_mode():
    L, E, rho, A = 3.0, 200e9, 7850.0, 1e-3
    Iy = Iz = 8.333e-7
    J = 2.0 * Iy
    n_elem = 20
    m = Model(ndm=3, ndf=6)
    mat = ElasticIsotropic(1, E=E, nu=0.3, rho=rho)
    m.add_material(mat)
    dx = L / n_elem
    for i in range(n_elem + 1):
        m.add_node(i + 1, i * dx, 0.0, 0.0)
    m.fix(1, [1, 1, 1, 1, 1, 1])
    for i in range(n_elem):
        m.add_element(
            BeamColumn3D(i + 1, (i + 1, i + 2), mat, area=A, Iy=Iy, Iz=Iz, J=J,
                         vecxz=(0.0, 0.0, 1.0))
        )
    a = EigenAnalysis(m, num_modes=2)
    a.run()
    base = math.sqrt(E * Iz / (rho * A * L ** 4))
    f1_exact = CANTILEVER_BETA_L_SQ[0] * base / (2.0 * math.pi)
    # In 3D the symmetric Iy=Iz means the first two modes are degenerate
    # bending pairs — both should match the analytical value
    assert abs(a.frequencies[0] - f1_exact) / f1_exact < 1e-3
    assert abs(a.frequencies[1] - f1_exact) / f1_exact < 1e-3


# ---------------------------------------------------------------------------
# Quad4 — at least solves and produces positive sorted frequencies


def test_quad4_plate_returns_positive_sorted_frequencies():
    L = 1.0
    nx = ny = 8
    m = Model(ndm=2, ndf=2)
    mat = ElasticIsotropic(1, E=2.0e11, nu=0.3, rho=7850.0)
    m.add_material(mat)
    tag = 1
    for j in range(ny + 1):
        for i in range(nx + 1):
            m.add_node(tag, i * L / nx, j * L / ny)
            tag += 1

    def nidx(i, j):
        return j * (nx + 1) + i + 1

    etag = 1
    for j in range(ny):
        for i in range(nx):
            m.add_element(
                Quad4(etag,
                      (nidx(i, j), nidx(i + 1, j), nidx(i + 1, j + 1), nidx(i, j + 1)),
                      mat, thickness=0.01)
            )
            etag += 1
    # left edge fixed
    for j in range(ny + 1):
        m.fix(nidx(0, j), [1, 1])

    a = EigenAnalysis(m, num_modes=4)
    a.run()
    f = a.frequencies
    assert (f > 0.0).all()
    assert (np.diff(f) >= 0.0).all()


# ---------------------------------------------------------------------------
# eigen with MP constraints


def test_eigen_with_rigid_link_runs():
    """Cantilever with a rigid offset at the tip — analysis should run and
    return positive sorted frequencies, plus the slave-node mode shape
    must satisfy the rigid-link kinematic relation in 2D."""
    L, E, rho, A, Iz = 2.0, 2.0e11, 7850.0, 1e-3, 8e-7
    n_elem = 10
    m = _make_cantilever(n_elem, L, E, rho, A, Iz)
    # add an offset slave node at the tip
    tip = n_elem + 1
    offset_tag = tip + 1
    m.add_node(offset_tag, L, 0.3)
    m.rigid_link(retained=tip, constrained=offset_tag, kind="beam")
    a = EigenAnalysis(m, num_modes=2)
    a.run()
    assert (a.frequencies > 0.0).all()
    # Verify the rigid-link kinematic relation in each mode shape
    nr = m.node(tip)
    nc = m.node(offset_tag)
    dy = nc.coords[1] - nr.coords[1]  # dx is 0 here
    for k in range(a.num_modes):
        u_r = nr.mode_disp[0, k]
        v_r = nr.mode_disp[1, k]
        th_r = nr.mode_disp[2, k]
        u_c = nc.mode_disp[0, k]
        v_c = nc.mode_disp[1, k]
        th_c = nc.mode_disp[2, k]
        assert u_c == pytest.approx(u_r - dy * th_r, abs=1e-10, rel=1e-10)
        assert v_c == pytest.approx(v_r, abs=1e-10, rel=1e-10)
        assert th_c == pytest.approx(th_r, abs=1e-10, rel=1e-10)


# ---------------------------------------------------------------------------
# error paths


def test_eigen_zero_density_raises():
    L, n_elem = 1.0, 4
    m = Model(ndm=2, ndf=3)
    mat = ElasticIsotropic(1, E=2.0e11, nu=0.3, rho=0.0)
    m.add_material(mat)
    for i in range(n_elem + 1):
        m.add_node(i + 1, i * L / n_elem, 0.0)
    m.fix(1, [1, 1, 1])
    for i in range(n_elem):
        m.add_element(BeamColumn2D(i + 1, (i + 1, i + 2), mat, 1e-3, 1e-7))
    with pytest.raises(RuntimeError, match="rho"):
        EigenAnalysis(m, num_modes=2).run()


def test_eigen_too_many_modes_raises():
    m = _make_cantilever(2, 1.0, 2.0e11, 7850.0, 1e-3, 1e-7)
    # 2 elements -> 3 nodes, 1 fixed -> 6 free DOFs. Asking for 7 modes is invalid.
    with pytest.raises(RuntimeError, match="DOFs"):
        EigenAnalysis(m, num_modes=7).run()


def test_eigen_penalty_handler_rejected():
    m = _make_cantilever(2, 1.0, 2.0e11, 7850.0, 1e-3, 1e-7)
    with pytest.raises(ValueError, match="transformation"):
        EigenAnalysis(m, num_modes=1, constraints="penalty")
