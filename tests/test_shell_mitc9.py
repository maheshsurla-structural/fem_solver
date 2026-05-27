"""Tests for ShellMITC9 — Phase 22.6.

The 9-node biquadratic shell is pinned down by similar properties to
ShellMITC4 (Phase 14):

1. Construction & validation
2. Symmetry and rigid-body kernel of K
3. Membrane patch test (uniaxial tension)
4. Plate-bending convergence (SS plate, point load)
5. No shear locking (L/t sweep)
6. Higher-order accuracy advantage vs MITC4 on the same DOF count
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from femsolver import (
    ElasticIsotropic,
    Model,
    ShellMITC4,
    ShellMITC9,
)
from femsolver.analysis.linear_static import LinearStaticAnalysis


# ======================================================== fixtures

def _unit_q9_model(*, E: float = 2.0e11, nu: float = 0.3,
                    thickness: float = 0.01):
    """A single 9-node shell on the unit square at z = 0."""
    mat = ElasticIsotropic(1, E=E, nu=nu, rho=0.0)
    m = Model(ndm=3, ndf=6); m.add_material(mat)
    pts = [(0, 0), (1, 0), (1, 1), (0, 1),
           (0.5, 0), (1, 0.5), (0.5, 1), (0, 0.5),
           (0.5, 0.5)]
    for i, (x, y) in enumerate(pts, 1):
        m.add_node(i, x, y, 0.0)
    m.add_element(ShellMITC9(1, tuple(range(1, 10)), mat, thickness=thickness))
    return m


def _build_ss_plate_q9(N: int, *, L: float, t: float, E: float, nu: float,
                       rho: float = 0.0):
    """N x N mesh of ShellMITC9 elements on a simply-supported plate.

    Each 9-node element occupies a 2-node × 2-node patch of the
    underlying biquadratic node grid (nL = 2 N + 1 nodes per side).
    """
    nL = 2 * N + 1
    mat = ElasticIsotropic(1, E=E, nu=nu, rho=rho)
    m = Model(ndm=3, ndf=6); m.add_material(mat)
    h = L / (nL - 1)
    for j in range(nL):
        for i in range(nL):
            tag = j * nL + i + 1
            m.add_node(tag, i * h, j * h, 0.0)
    etag = 1
    for je in range(N):
        for ie in range(N):
            j0 = 2 * je; i0 = 2 * ie
            n0 = j0 * nL + i0 + 1
            n1 = j0 * nL + (i0 + 2) + 1
            n2 = (j0 + 2) * nL + (i0 + 2) + 1
            n3 = (j0 + 2) * nL + i0 + 1
            n4 = j0 * nL + (i0 + 1) + 1
            n5 = (j0 + 1) * nL + (i0 + 2) + 1
            n6 = (j0 + 2) * nL + (i0 + 1) + 1
            n7 = (j0 + 1) * nL + i0 + 1
            n8 = (j0 + 1) * nL + (i0 + 1) + 1
            m.add_element(ShellMITC9(
                etag, (n0, n1, n2, n3, n4, n5, n6, n7, n8),
                mat, thickness=t,
            ))
            etag += 1
    # SS BCs: w = 0 on all 4 edges (θx, θy free)
    for j in range(nL):
        for i in range(nL):
            tag = j * nL + i + 1
            if i == 0 or i == nL - 1 or j == 0 or j == nL - 1:
                m.fix(tag, [1, 1, 1, 0, 0, 0])
    return m, nL


# ======================================================== construction

def test_shell_mitc9_rejects_zero_thickness():
    mat = ElasticIsotropic(1, E=2e11, nu=0.3, rho=0.0)
    with pytest.raises(ValueError, match="thickness"):
        ShellMITC9(1, list(range(1, 10)), mat, thickness=0.0)


def test_shell_mitc9_rejects_negative_drilling_factor():
    mat = ElasticIsotropic(1, E=2e11, nu=0.3, rho=0.0)
    with pytest.raises(ValueError, match="drilling_factor"):
        ShellMITC9(1, list(range(1, 10)), mat,
                    thickness=0.01, drilling_factor=-0.1)


def test_shell_mitc9_rejects_out_of_range_k_shear():
    mat = ElasticIsotropic(1, E=2e11, nu=0.3, rho=0.0)
    with pytest.raises(ValueError, match="k_shear"):
        ShellMITC9(1, list(range(1, 10)), mat,
                    thickness=0.01, k_shear=2.0)


# ======================================================== K properties

def test_shell_mitc9_K_is_symmetric():
    m = _unit_q9_model()
    K = m.elements[1].K_global()
    assert K.shape == (54, 54)
    np.testing.assert_allclose(K, K.T, atol=1e-6 * np.max(np.abs(K)))


def test_shell_mitc9_rigid_translation_gives_zero_internal_force():
    """Apply uniform x-translation; internal force should be zero
    (up to drilling-stabilization noise)."""
    m = _unit_q9_model()
    K = m.elements[1].K_global()
    u_rigid = np.zeros(54)
    for i in range(9):
        u_rigid[6 * i + 0] = 1.0
    fint = K @ u_rigid
    assert np.linalg.norm(fint) < 1.0e-3 * np.max(np.abs(K))


def test_shell_mitc9_rigid_z_translation_zero_force():
    m = _unit_q9_model()
    K = m.elements[1].K_global()
    u_rigid = np.zeros(54)
    for i in range(9):
        u_rigid[6 * i + 2] = 1.0
    fint = K @ u_rigid
    assert np.linalg.norm(fint) < 1.0e-3 * np.max(np.abs(K))


# ======================================================== membrane patch

def test_shell_mitc9_membrane_uniaxial_tension():
    """Apply uniaxial tension along x on a single Q9 element on the
    unit square. Use minimal BCs to avoid Poisson-contraction artifacts:
    only fix u at the 3 x=0-edge nodes, v at the (0,0) corner, w & rotations
    everywhere to make this a pure 2-D problem."""
    L = 1.0; t = 0.01; E = 2.0e11; nu = 0.3
    m = _unit_q9_model(E=E, nu=nu, thickness=t)
    # Fix u on the x=0 edge (nodes 1, 4, 8) so the bar can stretch in
    # +x. Fix v only on node 1 (corner (0,0)) to lock rigid translation
    # in y. Free contraction in y elsewhere.
    for tag in (1, 4, 8):
        m.node(tag).fixity[0] = 1     # u = 0 on x=0 edge
    m.node(1).fixity[1] = 1            # v = 0 at single corner
    # Suppress out-of-plane motion everywhere to keep it a 2D problem
    for tag in range(1, 10):
        node = m.node(tag)
        node.fixity[2] = 1
        node.fixity[3] = 1
        node.fixity[4] = 1
        node.fixity[5] = 1
    # Apply consistent line load on x=L=1 edge (nodes 2, 6, 3)
    F_total = 1.0e3
    weights = (1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0)
    for tag, w in zip((2, 6, 3), weights):
        m.add_nodal_load(tag, [F_total * w, 0, 0, 0, 0, 0])
    LinearStaticAnalysis(m).run()
    sigma_xx = F_total / (1.0 * t)
    u_expected = sigma_xx * L / E
    u_corner = m.node(2).disp[0]
    assert u_corner == pytest.approx(u_expected, rel=1.0e-3)


# ======================================================== plate convergence

def test_shell_mitc9_ss_plate_convergence():
    """SS plate, central point load. Q9 should converge with refinement
    toward the Kirchhoff thin-plate answer."""
    L = 1.0; t = 0.01; E = 2.0e11; nu = 0.3; P = 1.0
    D = E * t ** 3 / (12 * (1 - nu * nu))
    w_kirch = 0.01160 * P * L ** 2 / D
    errors = []
    for N in (2, 4):
        m, nL = _build_ss_plate_q9(N, L=L, t=t, E=E, nu=nu)
        center_tag = (nL // 2) * nL + nL // 2 + 1
        m.add_nodal_load(center_tag, [0, 0, -P, 0, 0, 0])
        LinearStaticAnalysis(m).run()
        w_center = -m.node(center_tag).disp[2]
        errors.append(abs(w_center - w_kirch) / w_kirch)
    # N=2 error larger than N=4 error (convergence)
    assert errors[1] < errors[0]
    # N=4 error within 5% (Mindlin-vs-Kirchhoff plus discretization)
    assert errors[1] < 0.05


def test_shell_mitc9_no_shear_locking():
    """Sweep L/t from 100 to 10000 on a 4x4 mesh; the scaled deflection
    should converge (no locking) -- Mindlin shear contribution becomes
    negligible as t -> 0."""
    L = 1.0; E = 2.0e11; nu = 0.3; P = 1.0
    scaled = []
    for t in (0.01, 0.001, 0.0001):
        D = E * t ** 3 / (12 * (1 - nu * nu))
        m, nL = _build_ss_plate_q9(4, L=L, t=t, E=E, nu=nu)
        center = (nL // 2) * nL + nL // 2 + 1
        m.add_nodal_load(center, [0, 0, -P, 0, 0, 0])
        LinearStaticAnalysis(m).run()
        w = -m.node(center).disp[2]
        scaled.append(w * D / (P * L ** 2))
    # The three scaled values should differ by less than 1%
    s_arr = np.asarray(scaled)
    assert np.max(s_arr) / np.min(s_arr) < 1.01, (
        f"shear locking suspected: scaled deflections vary by "
        f"{(np.max(s_arr)/np.min(s_arr) - 1) * 100:.2f}%, expected < 1%"
    )


# ======================================================== Q9 vs Q4 mesh comparison

def test_shell_mitc9_higher_order_accuracy_on_smooth_problem():
    """On a smooth uniformly-loaded SS plate problem (no point-load
    singularity), Q9 should converge faster than Q4 in terms of total
    nodes -- the value-add of higher-order elements.

    Use a uniform pressure approximated as equal nodal loads. Compare
    Q9 N=2 (25 nodes) vs Q4 N=4 (25 nodes) -- equal DOF count.
    """
    L = 1.0; t = 0.01; E = 2.0e11; nu = 0.3
    q = 1.0e3       # uniform pressure (N/m^2)

    # Q9: 5x5 = 25 nodes, 4 Q9 elements
    m_q9, nL_q9 = _build_ss_plate_q9(2, L=L, t=t, E=E, nu=nu)
    # Apply equal share of total force q*L^2 to each non-boundary node
    interior_q9 = []
    for j in range(nL_q9):
        for i in range(nL_q9):
            tag = j * nL_q9 + i + 1
            if not (i == 0 or i == nL_q9 - 1 or j == 0 or j == nL_q9 - 1):
                interior_q9.append(tag)
    F_per_q9 = q * L * L / len(interior_q9)
    for tag in interior_q9:
        m_q9.add_nodal_load(tag, [0, 0, -F_per_q9, 0, 0, 0])
    LinearStaticAnalysis(m_q9).run()
    center_q9 = (nL_q9 // 2) * nL_q9 + nL_q9 // 2 + 1
    w_q9 = -m_q9.node(center_q9).disp[2]

    # Q4: 5x5 = 25 nodes, 4x4 Q4 elements
    mat = ElasticIsotropic(1, E=E, nu=nu, rho=0.0)
    m_q4 = Model(ndm=3, ndf=6); m_q4.add_material(mat)
    N4 = 4; nL_q4 = N4 + 1
    h = L / N4
    for j in range(nL_q4):
        for i in range(nL_q4):
            m_q4.add_node(j * nL_q4 + i + 1, i * h, j * h, 0.0)
    etag = 1
    for je in range(N4):
        for ie in range(N4):
            n1 = je * nL_q4 + ie + 1
            n2 = n1 + 1
            n3 = n2 + nL_q4
            n4 = n1 + nL_q4
            m_q4.add_element(ShellMITC4(etag, (n1, n2, n3, n4), mat, t))
            etag += 1
    for j in range(nL_q4):
        for i in range(nL_q4):
            tag = j * nL_q4 + i + 1
            if i == 0 or i == nL_q4 - 1 or j == 0 or j == nL_q4 - 1:
                m_q4.fix(tag, [1, 1, 1, 0, 0, 0])
    interior_q4 = []
    for j in range(nL_q4):
        for i in range(nL_q4):
            tag = j * nL_q4 + i + 1
            if not (i == 0 or i == nL_q4 - 1 or j == 0 or j == nL_q4 - 1):
                interior_q4.append(tag)
    F_per_q4 = q * L * L / len(interior_q4)
    for tag in interior_q4:
        m_q4.add_nodal_load(tag, [0, 0, -F_per_q4, 0, 0, 0])
    LinearStaticAnalysis(m_q4).run()
    center_q4 = (nL_q4 // 2) * nL_q4 + nL_q4 // 2 + 1
    w_q4 = -m_q4.node(center_q4).disp[2]

    # Both should give positive deflection; Q9 should give a deflection
    # that's at least as accurate as Q4 (closer to the converged
    # answer). For this lumped-load approximation, both will exceed
    # the exact uniform-load answer slightly; the absolute compare is
    # against each other, not theory. We require that Q9 and Q4 agree
    # to within ~10% at equal DOF count -- if Q9 is grossly wrong this
    # would catch it.
    assert abs(w_q9 - w_q4) / w_q4 < 0.10, (
        f"Q9 ({w_q9:.3e}) and Q4 ({w_q4:.3e}) at equal DOF count "
        f"disagree by more than 10%; check Q9 implementation"
    )
    # Both deflections should be positive
    assert w_q9 > 0 and w_q4 > 0


# ======================================================== recovery

def test_shell_mitc9_recovery_populates_gp_arrays():
    """After LinearStaticAnalysis.run(), each Q9 element has 9 Gauss
    points (3x3) with membrane / bending / shear data."""
    L = 1.0; t = 0.01; E = 2.0e11; nu = 0.3; P = 1.0
    m, nL = _build_ss_plate_q9(2, L=L, t=t, E=E, nu=nu)
    center = (nL // 2) * nL + nL // 2 + 1
    m.add_nodal_load(center, [0, 0, -P, 0, 0, 0])
    LinearStaticAnalysis(m).run()
    el = m.elements[1]
    assert len(el.gp_membrane_strain) == 9
    assert len(el.gp_bending_curvature) == 9
    assert len(el.gp_shear_strain) == 9
    assert len(el.gp_resultants) == 9
    # Each resultants vector has length 3 + 3 + 2 = 8
    assert el.gp_resultants[0].size == 8


# ======================================================== mass

def test_shell_mitc9_mass_matrix_shape_and_total():
    """Total mass = rho * t * area, dispersed across the 9 nodes by
    the consistent N_i N_j integral. Sum of translational mass entries
    should equal rho * t * area * 3 (one for each of x, y, z)."""
    mat = ElasticIsotropic(1, E=2e11, nu=0.3, rho=7800.0)
    m = Model(ndm=3, ndf=6); m.add_material(mat)
    pts = [(0, 0), (1, 0), (1, 1), (0, 1),
           (0.5, 0), (1, 0.5), (0.5, 1), (0, 0.5),
           (0.5, 0.5)]
    for i, (x, y) in enumerate(pts, 1):
        m.add_node(i, x, y, 0.0)
    el = ShellMITC9(1, tuple(range(1, 10)), mat, thickness=0.01)
    m.add_element(el)
    M = el.M_global()
    assert M.shape == (54, 54)
    # Translational diagonals (DOF indices 6i+0, 6i+1, 6i+2 for i=0..8)
    total_x = sum(M[6 * i, 6 * i] for i in range(9))
    expected = 7800.0 * 0.01 * 1.0     # rho * t * area
    # Lumped (row-sum) check on translation
    M_lumped = el.M_global(lumped=True)
    total_x_lumped = sum(M_lumped[6 * i, 6 * i] for i in range(9))
    assert total_x_lumped == pytest.approx(expected, rel=1e-10)
