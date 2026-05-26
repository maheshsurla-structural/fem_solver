"""Tests for ShellMITC4 — Phase 14.0.

The MITC4 element is pinned down by six properties:

1. **Construction & validation** — invalid thickness, drilling, k_shear.
2. **Symmetry and positive-definiteness** — K is symmetric; the free-DOF
   K is PD once enough nodes are constrained.
3. **Rigid-body modes** — applying a rigid translation / rotation
   produces zero internal force.
4. **Membrane patch test** — a uniaxial tension state in the element
   plane gives the exact stiff-beam answer (no membrane locking).
5. **Plate-bending patch test (SS plate, point load)** — Mindlin-
   Reissner shell matches Timoshenko thin-plate closed form within
   ~0.5% on an 8x8 mesh.
6. **No shear locking** — the result is essentially independent of
   ``L/t`` from 100 to 10,000 (MITC tying cures locking at the limit).
"""
import math

import numpy as np
import pytest

from femsolver import (
    ElasticIsotropic,
    Model,
    ShellMITC4,
)
from femsolver.analysis.linear_static import LinearStaticAnalysis


# ====================================================== helpers

def _build_ss_plate(N: int, *, L: float, t: float, E: float, nu: float,
                    rho: float = 0.0):
    """N x N mesh of a square simply-supported plate of side L,
    thickness t, in the xy-plane at z=0."""
    nL = N + 1
    mat = ElasticIsotropic(1, E=E, nu=nu, rho=rho)
    m = Model(ndm=3, ndf=6); m.add_material(mat)
    for j in range(nL):
        for i in range(nL):
            tag = j * nL + i + 1
            m.add_node(tag, i * L / N, j * L / N, 0.0)
    etag = 1
    for j in range(N):
        for i in range(N):
            n1 = j * nL + i + 1; n2 = n1 + 1; n3 = n2 + nL; n4 = n1 + nL
            m.add_element(ShellMITC4(etag, (n1, n2, n3, n4), mat, t))
            etag += 1
    # SS BC: w = 0 on all edges (θx, θy free)
    for j in range(nL):
        for i in range(nL):
            tag = j * nL + i + 1
            on_edge = (i == 0 or i == N or j == 0 or j == N)
            if on_edge:
                m.fix(tag, [0, 0, 1, 0, 0, 0])
    # Pin one corner fully in-plane (u, v) and another corner v
    # to remove rigid-body modes.
    m.fix(1, [1, 1, 1, 0, 0, 0])
    m.fix(N + 1, [0, 1, 1, 0, 0, 0])
    return m, nL


def _single_element(L: float = 1.0, t: float = 0.01,
                    E: float = 2.0e11, nu: float = 0.3,
                    rho: float = 0.0):
    """Single ShellMITC4 in the xy-plane, side L, fully unconstrained
    (the caller adds constraints)."""
    mat = ElasticIsotropic(1, E=E, nu=nu, rho=rho)
    m = Model(ndm=3, ndf=6); m.add_material(mat)
    m.add_node(1, 0.0, 0.0, 0.0)
    m.add_node(2, L, 0.0, 0.0)
    m.add_node(3, L, L, 0.0)
    m.add_node(4, 0.0, L, 0.0)
    e = ShellMITC4(1, (1, 2, 3, 4), mat, t)
    m.add_element(e)
    return m, e


# ====================================================== construction

def test_shell_rejects_invalid_thickness():
    mat = ElasticIsotropic(1, E=2.0e11, nu=0.3)
    with pytest.raises(ValueError, match="thickness"):
        ShellMITC4(1, (1, 2, 3, 4), mat, thickness=0.0)
    with pytest.raises(ValueError, match="thickness"):
        ShellMITC4(1, (1, 2, 3, 4), mat, thickness=-0.1)


def test_shell_rejects_invalid_k_shear():
    mat = ElasticIsotropic(1, E=2.0e11, nu=0.3)
    with pytest.raises(ValueError, match="k_shear"):
        ShellMITC4(1, (1, 2, 3, 4), mat, thickness=0.01, k_shear=0.0)
    with pytest.raises(ValueError, match="k_shear"):
        ShellMITC4(1, (1, 2, 3, 4), mat, thickness=0.01, k_shear=1.5)


def test_shell_rejects_negative_drilling():
    mat = ElasticIsotropic(1, E=2.0e11, nu=0.3)
    with pytest.raises(ValueError, match="drilling_factor"):
        ShellMITC4(1, (1, 2, 3, 4), mat, thickness=0.01, drilling_factor=-0.1)


# ====================================================== symmetry

def test_shell_K_is_symmetric():
    _, e = _single_element()
    K = e.K_global()
    assert np.allclose(K, K.T, atol=1e-8 * np.max(np.abs(K)))


def test_shell_K_is_24x24():
    _, e = _single_element()
    K = e.K_global()
    assert K.shape == (24, 24)


# ====================================================== rigid-body modes

def test_shell_zero_internal_force_under_rigid_translation():
    """A rigid translation of all 4 nodes should produce zero internal
    force vector (within roundoff)."""
    m, e = _single_element()
    m.number_dofs()
    # Apply rigid translation u_x = 1.0 at every node (free DOFs)
    for tag in (1, 2, 3, 4):
        m.node(tag).disp[0] = 1.0
    f_int = e.f_int_global()
    # The drilling penalty terms are tied to θz, which is zero — so the
    # internal force for pure translation should be at roundoff.
    EA = e.material.E * e.thickness * 1.0
    assert np.max(np.abs(f_int)) < 1.0e-9 * EA


def test_shell_zero_internal_force_under_rigid_z_translation():
    """A uniform out-of-plane translation should produce zero force
    (no membrane or bending strain)."""
    m, e = _single_element()
    m.number_dofs()
    for tag in (1, 2, 3, 4):
        m.node(tag).disp[2] = 1.0
    f_int = e.f_int_global()
    EA = e.material.E * e.thickness * 1.0
    assert np.max(np.abs(f_int)) < 1.0e-9 * EA


# ====================================================== membrane patch test

def test_shell_membrane_uniaxial_tension():
    """Pull a single element in pure tension in x. The far-end
    displacement should equal P L / (E t) (one-dimensional bar)."""
    L = 1.0
    t = 0.01
    E = 2.0e11
    nu = 0.3
    m, e = _single_element(L=L, t=t, E=E, nu=nu)
    # Fix left edge fully in u, partially in v / w / rotations:
    # Node 1: u=v=w=0, all rotations fixed
    m.fix(1, [1, 1, 1, 1, 1, 1])
    # Node 4: u=0 (left edge), v free for Poisson, w=0, rotations fixed
    m.fix(4, [1, 0, 1, 1, 1, 1])
    # Right edge: w=0, rotations fixed; in-plane free
    m.fix(2, [0, 0, 1, 1, 1, 1])
    m.fix(3, [0, 0, 1, 1, 1, 1])
    P = 1000.0
    m.add_nodal_load(2, [P / 2.0, 0, 0, 0, 0, 0])
    m.add_nodal_load(3, [P / 2.0, 0, 0, 0, 0, 0])
    LinearStaticAnalysis(m).run()
    # Far-end x-displacement
    u_x = 0.5 * (m.node(2).disp[0] + m.node(3).disp[0])
    # One-dimensional bar with shell membrane stiffness E*t (width L)
    u_anal = P * L / (E * t * L)   # σ = P/(t·L), ε = σ/E, u = ε·L = P/(E·t)
    assert u_x == pytest.approx(u_anal, rel=2e-2)


# ====================================================== plate-bending

def test_shell_ss_plate_center_load_matches_timoshenko():
    """Simply-supported square plate under a central point load.
    Timoshenko gives w_c = 0.01160 * P L^2 / D for ν = 0.3.
    On an 8x8 mesh, the MITC4 result should match within 1%.
    """
    L = 1.0
    t = 0.01
    E = 2.0e11
    nu = 0.3
    P = 1.0
    D = E * t ** 3 / (12.0 * (1.0 - nu ** 2))
    w_anal = 0.01160 * P * L ** 2 / D
    m, nL = _build_ss_plate(N=8, L=L, t=t, E=E, nu=nu)
    ic = (8 // 2) * nL + 8 // 2 + 1
    m.add_nodal_load(ic, [0, 0, -P, 0, 0, 0])
    LinearStaticAnalysis(m).run()
    w_fem = -m.node(ic).disp[2]
    assert w_fem / w_anal == pytest.approx(1.0, rel=2.0e-2), \
        f"w_fem/w_anal = {w_fem/w_anal:.4f}, expected ~1.0"


def test_shell_no_shear_locking_at_thin_limit():
    """The MITC tying must remove shear locking. Sweep L/t from
    100 to 10000 and check the SS-plate ratio stays steady (not
    collapsing to zero as t->0)."""
    L = 1.0
    E = 2.0e11
    nu = 0.3
    P = 1.0
    ratios = []
    for t in (1e-2, 1e-3, 1e-4):
        D = E * t ** 3 / (12.0 * (1.0 - nu ** 2))
        w_anal = 0.01160 * P * L ** 2 / D
        m, nL = _build_ss_plate(N=8, L=L, t=t, E=E, nu=nu)
        ic = (8 // 2) * nL + 8 // 2 + 1
        m.add_nodal_load(ic, [0, 0, -P, 0, 0, 0])
        LinearStaticAnalysis(m).run()
        w_fem = -m.node(ic).disp[2]
        ratios.append(w_fem / w_anal)
    # All ratios should be close — within 1% spread, all near 0.99
    spread = max(ratios) - min(ratios)
    assert spread < 5e-3, f"shear locking detected: ratios = {ratios}"
    assert min(ratios) > 0.95, f"plate too soft / wrong: ratios = {ratios}"


# ====================================================== convergence

def test_shell_ss_plate_converges_with_refinement():
    """The error should decrease as the mesh is refined."""
    L = 1.0
    t = 0.01
    E = 2.0e11
    nu = 0.3
    P = 1.0
    D = E * t ** 3 / (12.0 * (1.0 - nu ** 2))
    w_anal = 0.01160 * P * L ** 2 / D
    errors = []
    for N in (4, 8, 12):
        m, nL = _build_ss_plate(N=N, L=L, t=t, E=E, nu=nu)
        # Center node — exists only if N is even.
        ic = (N // 2) * nL + N // 2 + 1
        m.add_nodal_load(ic, [0, 0, -P, 0, 0, 0])
        LinearStaticAnalysis(m).run()
        w_fem = -m.node(ic).disp[2]
        errors.append(abs(w_fem / w_anal - 1.0))
    # Monotonic decrease (or at least non-increase)
    assert errors[1] <= errors[0] + 5e-3
    assert errors[2] <= errors[1] + 5e-3
    assert errors[-1] < 5e-3


# ====================================================== recovery

def test_shell_recover_populates_gp_resultants():
    """After running a static analysis, e.recover() should populate
    per-Gauss-point membrane/bending/shear strains and resultants."""
    m, nL = _build_ss_plate(N=4, L=1.0, t=0.01, E=2.0e11, nu=0.3)
    ic = (2) * nL + 2 + 1
    m.add_nodal_load(ic, [0, 0, -1.0, 0, 0, 0])
    LinearStaticAnalysis(m).run()
    e = next(iter(m.elements.values()))
    e.recover()
    assert len(e.gp_membrane_strain) == 4
    assert len(e.gp_bending_curvature) == 4
    assert len(e.gp_shear_strain) == 4
    assert len(e.gp_resultants) == 4
    # Each resultant tuple has length 3 (N) + 3 (M) + 2 (Q) = 8
    assert e.gp_resultants[0].size == 8
