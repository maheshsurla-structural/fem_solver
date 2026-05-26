"""Tests for ShellMITC4 geometric stiffness and linear buckling
(Phase 14.3).

Three properties pin down the geometric-stiffness implementation:

1. **K_tangent_global equals K_global at zero displacement** — with
   no internal forces, K_g must be exactly zero.
2. **K_g is symmetric** — the geometric stiffness preserves symmetry.
3. **SS square plate uniaxial buckling matches Bryan's formula** —
   the canonical closed-form for a simply-supported square plate
   under uniaxial edge compression is
       sigma_cr = 4 pi^2 E t^2 / [12 (1 - nu^2) b^2]
   On a 12x12 mesh, MITC4 should match this within ~1%.
"""
import math

import numpy as np
import pytest

from femsolver import ElasticIsotropic, Model, ShellMITC4
from femsolver.analysis.buckling import LinearBucklingAnalysis


def _build_buckling_plate(N: int, *, a: float, b: float, t: float,
                          E: float, nu: float, P_ref: float = 1.0):
    """SS-SS-SS-SS plate of dims (a x b), uniaxial edge compression
    on the right edge totaling P_ref."""
    mat = ElasticIsotropic(1, E=E, nu=nu)
    m = Model(ndm=3, ndf=6); m.add_material(mat)
    nx = N + 1
    ny = max(2, int(round(N * b / a))) + 1
    for j in range(ny):
        for i in range(nx):
            tag = j * nx + i + 1
            m.add_node(tag, i * a / N, j * b / (ny - 1), 0.0)
    etag = 1
    for j in range(ny - 1):
        for i in range(N):
            n1 = j * nx + i + 1; n2 = n1 + 1
            n3 = n2 + nx; n4 = n1 + nx
            m.add_element(ShellMITC4(etag, (n1, n2, n3, n4), mat, t))
            etag += 1
    # SS: w=0 on all four edges
    for j in range(ny):
        for i in range(nx):
            tag = j * nx + i + 1
            on_edge = (i == 0 or i == N or j == 0 or j == ny - 1)
            if on_edge:
                m.fix(tag, [0, 0, 1, 0, 0, 0])
    # Left edge: u = 0 (so the membrane-prestress field is uniaxial)
    for j in range(ny):
        m.fix(j * nx + 1, [1, 0, 0, 0, 0, 0])
    # Pin one node fully in-plane to remove the v rigid-body
    m.fix(1, [1, 1, 1, 0, 0, 0])
    # Right-edge compressive distribution, corner half-weights
    weights = [0.5 if j in (0, ny - 1) else 1.0 for j in range(ny)]
    w_total = sum(weights)
    for j in range(ny):
        tag = j * nx + N + 1
        m.add_nodal_load(tag, [-P_ref * weights[j] / w_total, 0, 0, 0, 0, 0])
    return m


# ====================================================== invariants

def test_shell_K_tangent_equals_K_at_zero_disp():
    """With no displacements, the geometric stiffness must be exactly
    zero — K_tangent == K_elastic."""
    mat = ElasticIsotropic(1, E=2.0e11, nu=0.3)
    m = Model(ndm=3, ndf=6); m.add_material(mat)
    m.add_node(1, 0.0, 0.0, 0.0)
    m.add_node(2, 1.0, 0.0, 0.0)
    m.add_node(3, 1.0, 1.0, 0.0)
    m.add_node(4, 0.0, 1.0, 0.0)
    e = ShellMITC4(1, (1, 2, 3, 4), mat, thickness=0.01)
    m.add_element(e)
    m.number_dofs()
    K_e = e.K_global()
    K_T = e.K_tangent_global()
    assert np.allclose(K_T, K_e, atol=1e-8 * np.max(np.abs(K_e)))


def test_shell_K_geometric_is_symmetric():
    """Even with nonzero membrane stress, K_g preserves symmetry."""
    mat = ElasticIsotropic(1, E=2.0e11, nu=0.3)
    m = Model(ndm=3, ndf=6); m.add_material(mat)
    m.add_node(1, 0.0, 0.0, 0.0)
    m.add_node(2, 1.0, 0.0, 0.0)
    m.add_node(3, 1.0, 1.0, 0.0)
    m.add_node(4, 0.0, 1.0, 0.0)
    e = ShellMITC4(1, (1, 2, 3, 4), mat, thickness=0.01)
    m.add_element(e)
    m.number_dofs()
    # Impose a small uniaxial-tension displacement state
    m.node(2).disp[0] = 1.0e-4
    m.node(3).disp[0] = 1.0e-4
    K_T = e.K_tangent_global()
    K_e = e.K_global()
    K_g = K_T - K_e
    rel = np.max(np.abs(K_g - K_g.T)) / max(np.max(np.abs(K_g)), 1.0)
    assert rel < 1e-10


# ====================================================== Bryan formula

def test_shell_ss_square_plate_buckling_matches_bryan():
    """SS-SS-SS-SS square plate under uniaxial edge compression. The
    closed-form critical edge load (per unit edge) is
        N_cr = 4 pi^2 D / b^2,  D = E t^3 / (12 (1 - nu^2))
    Multiplying by edge length b gives the total compressive force
    that triggers buckling.

    On a 12x12 mesh, MITC4 should land within 1.5% of this.
    """
    E, nu = 2.0e11, 0.3
    t = 0.01
    a = b = 1.0
    sigma_cr = 4.0 * math.pi ** 2 * E * t ** 2 / (12.0 * (1.0 - nu ** 2) * b ** 2)
    P_bryan = sigma_cr * t * b
    m = _build_buckling_plate(N=12, a=a, b=b, t=t, E=E, nu=nu, P_ref=1.0)
    res = LinearBucklingAnalysis(m, num_modes=1).run()
    P_fem = res["critical_load_factor"] * 1.0   # P_ref = 1
    assert P_fem / P_bryan == pytest.approx(1.0, rel=1.5e-2), \
        f"got ratio {P_fem / P_bryan:.4f}"


def test_shell_buckling_converges_under_mesh_refinement():
    """Refining 4 -> 8 -> 12 should reduce the buckling-load error."""
    E, nu = 2.0e11, 0.3
    t = 0.01
    b = 1.0
    sigma_cr = 4.0 * math.pi ** 2 * E * t ** 2 / (12.0 * (1.0 - nu ** 2) * b ** 2)
    P_bryan = sigma_cr * t * b
    errors = []
    for N in (4, 8, 12):
        m = _build_buckling_plate(N=N, a=1.0, b=1.0, t=t, E=E, nu=nu, P_ref=1.0)
        res = LinearBucklingAnalysis(m, num_modes=1).run()
        err = abs(res["critical_load_factor"] / P_bryan - 1.0)
        errors.append(err)
    assert errors[1] < errors[0], f"non-monotonic: {errors}"
    assert errors[2] < errors[1], f"non-monotonic: {errors}"
    assert errors[2] < 1.5e-2
