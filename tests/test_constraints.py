"""Tests for multi-point constraints — EqualDOF, RigidLink, RigidDiaphragm,
MPConstraint, and the two constraint handlers."""
from __future__ import annotations

import numpy as np
import pytest

from femsolver import (
    BeamColumn2D,
    BeamColumn3D,
    ElasticIsotropic,
    EqualDOF,
    LinearStaticAnalysis,
    MPConstraint,
    Model,
    PenaltyHandler,
    RigidDiaphragm,
    RigidLink,
    TransformationHandler,
    Truss2D,
)


# ---------------------------------------------------------------------------
# EqualDOF — two parallel trusses sharing tip displacement


def _two_parallel_trusses_model(k_ratio: float = 1.0):
    """Two horizontal trusses at different heights with their right ends tied.

    Truss A: nodes 1->2, area a.
    Truss B: nodes 3->4, area a*k_ratio.
    Left ends (1, 3) fixed. Right ends (2, 4) tied via EqualDOF on the X DOF.
    Load P applied to node 2 in +X.
    """
    L = 2.0
    E = 1.0e7
    a = 1.0
    P = 100.0

    m = Model(ndm=2, ndf=2)
    m.add_material(ElasticIsotropic(tag=1, E=E, nu=0.3))
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, L, 0.0)
    m.add_node(3, 0.0, 1.0)
    m.add_node(4, L, 1.0)
    m.fix(1, [1, 1])
    m.fix(3, [1, 1])
    # truss B's tip y is fixed too (no element supports it transversely)
    m.fix(4, [0, 1])
    m.fix(2, [0, 1])

    m.add_element(Truss2D(tag=1, nodes=(1, 2), material=m.material(1), area=a))
    m.add_element(Truss2D(tag=2, nodes=(3, 4), material=m.material(1), area=a * k_ratio))

    m.equal_dof(retained=2, constrained=4, dofs=[0])
    m.add_nodal_load(2, [P, 0.0])
    return m, dict(L=L, E=E, a=a, P=P, k_ratio=k_ratio)


def test_equal_dof_parallel_trusses_share_displacement():
    m, p = _two_parallel_trusses_model(k_ratio=2.0)
    LinearStaticAnalysis(m).run()
    u2 = m.node(2).disp[0]
    u4 = m.node(4).disp[0]
    # tied DOF must agree to machine precision
    assert u2 == pytest.approx(u4, abs=1e-14)
    # parallel-stiffness analytic: u = P / (k_a + k_b), k = E*A/L
    k_total = p["E"] * p["a"] * (1.0 + p["k_ratio"]) / p["L"]
    assert u2 == pytest.approx(p["P"] / k_total, rel=1e-12)


def test_equal_dof_reactions_split_by_stiffness():
    m, p = _two_parallel_trusses_model(k_ratio=3.0)
    LinearStaticAnalysis(m).run()
    R1 = m.node(1).reaction[0]
    R3 = m.node(3).reaction[0]
    # the stiffer member carries 3x the load
    assert R3 / R1 == pytest.approx(p["k_ratio"], rel=1e-12)
    # reactions sum to -P
    assert R1 + R3 == pytest.approx(-p["P"], rel=1e-12)


# ---------------------------------------------------------------------------
# RigidLink in 2D — beam tip with offset load


def test_rigid_link_2d_beam_offset_load():
    """Cantilever with a rigid offset above the tip; horizontal force applied
    at the offset node creates a moment at the tip.

    Cantilever from (0,0) to (L,0), fixed at node 1. Beam tip is node 2.
    Node 3 sits at (L, e); rigid link ties it to node 2 (retained). Apply
    horizontal force F at node 3.
    """
    L = 5.0
    e = 0.5
    E = 2.0e11
    A = 1.0e-3
    Iz = 8.333e-6
    F = 1000.0

    m = Model(ndm=2, ndf=3)
    m.add_material(ElasticIsotropic(tag=1, E=E, nu=0.3))
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, L, 0.0)
    m.add_node(3, L, e)
    m.fix(1, [1, 1, 1])
    # Node 3's DOFs are slaved to node 2 by the rigid link, so we leave it free.

    m.add_element(BeamColumn2D(tag=1, nodes=(1, 2), material=m.material(1), area=A, Iz=Iz))
    m.rigid_link(retained=2, constrained=3, kind="beam")
    m.add_nodal_load(3, [F, 0.0, 0.0])

    LinearStaticAnalysis(m).run()

    # Equivalent loads at retained node 2 (cantilever tip):
    #   axial  = F
    #   moment = -e * F  (r x F with r=(0,e), F=(F,0) → M_z = -eF)
    M = -e * F
    ux_r = F * L / (E * A)
    uy_r = M * L * L / (2.0 * E * Iz)
    th_r = M * L / (E * Iz)

    n2 = m.node(2)
    assert n2.disp[0] == pytest.approx(ux_r, rel=1e-10)
    assert n2.disp[1] == pytest.approx(uy_r, rel=1e-10)
    assert n2.disp[2] == pytest.approx(th_r, rel=1e-10)

    # Constrained node 3 follows the rigid-body relation
    n3 = m.node(3)
    dx = 0.0
    dy = e
    assert n3.disp[0] == pytest.approx(ux_r - dy * th_r, rel=1e-10)
    assert n3.disp[1] == pytest.approx(uy_r + dx * th_r, rel=1e-10)
    assert n3.disp[2] == pytest.approx(th_r, rel=1e-10)


def test_rigid_link_3d_beam_kinematics():
    """3D RigidLink (beam type): after solving, the constrained-node
    displacement satisfies the rigid-body relation ``u_c = u_r + theta_r x r``
    and ``theta_c = theta_r``, where ``r = x_c - x_r``.

    A 3D cantilever (node 1 fixed at origin, node 2 free at tip) carries a
    constrained offset node 3 at ``x_2 + (rx, ry, rz)``. We apply a 6-component
    load (forces and moments) at node 3; the rigid link transmits them to
    node 2 and the kinematic identity must hold to machine precision.
    """
    from femsolver import BeamColumn3D

    L = 4.0
    rx, ry, rz = 0.6, -0.4, 0.3
    m = Model(ndm=3, ndf=6)
    mat = ElasticIsotropic(1, E=2e11, nu=0.3)
    m.add_material(mat)
    m.add_node(1, 0.0, 0.0, 0.0)
    m.add_node(2, L, 0.0, 0.0)
    m.add_node(3, L + rx, ry, rz)
    m.fix(1, [1, 1, 1, 1, 1, 1])
    m.add_element(
        BeamColumn3D(
            tag=1, nodes=(1, 2), material=mat,
            area=1e-2, Iy=2e-5, Iz=3e-5, J=4e-5, vecxz=(0.0, 0.0, 1.0),
        )
    )
    m.rigid_link(retained=2, constrained=3, kind="beam")
    # mixed force + moment at the offset node — exercises all 6 rows
    m.add_nodal_load(3, [1.0e3, -2.0e3, 5.0e2, 1.0e2, -50.0, 80.0])

    LinearStaticAnalysis(m).run()

    u_r = m.node(2).disp[:3]
    th_r = m.node(2).disp[3:]
    u_c = m.node(3).disp[:3]
    th_c = m.node(3).disp[3:]
    expected_u = u_r + np.cross(th_r, [rx, ry, rz])
    np.testing.assert_allclose(u_c, expected_u, atol=1e-14, rtol=1e-12)
    np.testing.assert_allclose(th_c, th_r, atol=1e-14, rtol=1e-12)


def test_rigid_link_2d_bar_translations_only():
    """type='bar' couples translations only; rotations remain independent."""
    L = 1.0
    E = 1.0e7
    A = 1.0
    Iz = 1.0e-3

    m = Model(ndm=2, ndf=3)
    m.add_material(ElasticIsotropic(tag=1, E=E, nu=0.3))
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, L, 0.0)
    m.add_node(3, L, 0.5)
    m.fix(1, [1, 1, 1])
    m.fix(3, [0, 0, 1])  # node 3 has no element touching it; pin its rotation
    m.add_element(BeamColumn2D(tag=1, nodes=(1, 2), material=m.material(1), area=A, Iz=Iz))
    m.rigid_link(retained=2, constrained=3, kind="bar")
    m.add_nodal_load(2, [0.0, -10.0, 0.0])

    LinearStaticAnalysis(m).run()
    n2, n3 = m.node(2), m.node(3)
    # translations equal, rotations independent (n3 fixed → rotation 0)
    assert n3.disp[0] == pytest.approx(n2.disp[0], abs=1e-14)
    assert n3.disp[1] == pytest.approx(n2.disp[1], abs=1e-14)
    assert n3.disp[2] == pytest.approx(0.0, abs=1e-14)
    assert n2.disp[2] != 0.0  # tip rotation is nonzero


# ---------------------------------------------------------------------------
# RigidDiaphragm — 4-column 3D frame with rigid floor


def _rigid_diaphragm_frame_model():
    """4 columns rising from a fixed base to a rigid diaphragm.

    Base nodes 1..4 at the corners of an L x L square at z=0, fixed.
    Top nodes 5..8 at the same XY but z=H, free.
    Master node 9 at (L/2, L/2, H), free.
    Diaphragm: master=9, slaves=[5,6,7,8], perp_dir=2 (XY-plane).

    Apply horizontal force F_x at the master.
    """
    L = 4.0
    H = 3.0
    E = 2.0e11
    nu = 0.3
    A = 0.05
    Iy = 4.17e-4
    Iz = 4.17e-4
    J = 8.33e-4
    F = 5.0e4

    m = Model(ndm=3, ndf=6)
    m.add_material(ElasticIsotropic(tag=1, E=E, nu=nu))
    base_xy = [(0.0, 0.0), (L, 0.0), (L, L), (0.0, L)]
    for i, (x, y) in enumerate(base_xy):
        m.add_node(i + 1, x, y, 0.0)
        m.fix(i + 1, [1, 1, 1, 1, 1, 1])
    for i, (x, y) in enumerate(base_xy):
        m.add_node(i + 5, x, y, H)
    m.add_node(9, L / 2.0, L / 2.0, H)
    # Master is not connected to any element; pin its out-of-plane DOFs
    # (u_z, theta_x, theta_y) — only in-plane DOFs participate in the
    # diaphragm.
    m.fix(9, [0, 0, 1, 1, 1, 0])

    for i in range(4):
        m.add_element(
            BeamColumn3D(
                tag=i + 1,
                nodes=(i + 1, i + 5),
                material=m.material(1),
                area=A,
                Iy=Iy,
                Iz=Iz,
                J=J,
                vecxz=(1.0, 0.0, 0.0),
            )
        )
    m.rigid_diaphragm(master=9, slaves=[5, 6, 7, 8], perp_dir=2)
    m.add_nodal_load(9, [F, 0.0, 0.0, 0.0, 0.0, 0.0])
    return m, dict(L=L, H=H, E=E, Iy=Iy, Iz=Iz, F=F, A=A, J=J, nu=nu)


def test_rigid_diaphragm_in_plane_translation():
    m, p = _rigid_diaphragm_frame_model()
    LinearStaticAnalysis(m).run()

    # All slave nodes share the master's in-plane translations and theta_z
    master = m.node(9)
    for s in (5, 6, 7, 8):
        slave = m.node(s)
        assert slave.disp[0] == pytest.approx(master.disp[0], abs=1e-14)
        # in-plane y is shifted by the lever arm * theta_z; with a symmetric
        # X load, theta_z = 0 and y-translations should also match
        assert master.disp[5] == pytest.approx(0.0, abs=1e-9)
        assert slave.disp[1] == pytest.approx(master.disp[1], abs=1e-9)
        assert slave.disp[5] == pytest.approx(master.disp[5], abs=1e-14)


def test_rigid_diaphragm_lateral_stiffness():
    """Each column is a fixed-free cantilever in bending: 3 EI / H^3 per
    column. Master sees 4x that in parallel — so u_x = F H^3 / (12 E I)."""
    m, p = _rigid_diaphragm_frame_model()
    LinearStaticAnalysis(m).run()
    expected = p["F"] * p["H"] ** 3 / (12.0 * p["E"] * p["Iz"])
    assert m.node(9).disp[0] == pytest.approx(expected, rel=1e-3)


# ---------------------------------------------------------------------------
# General MPConstraint — averaging and prescribed displacement


def test_mp_constraint_prescribed_nonzero_displacement():
    """Cantilever with a prescribed tip horizontal displacement (no retained
    terms, just g)."""
    L = 2.0
    E = 1.0e7
    A = 1.0
    Iz = 1.0e-3
    delta = 0.01

    m = Model(ndm=2, ndf=3)
    m.add_material(ElasticIsotropic(tag=1, E=E, nu=0.3))
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, L, 0.0)
    m.fix(1, [1, 1, 1])
    m.fix(2, [0, 1, 0])  # vertical & rotational free except the imposed u_x via MPConstraint
    m.add_element(BeamColumn2D(tag=1, nodes=(1, 2), material=m.material(1), area=A, Iz=Iz))
    m.add_mp_constraint(MPConstraint(constrained=(2, 0), retained=[], g=delta))
    LinearStaticAnalysis(m).run()
    assert m.node(2).disp[0] == pytest.approx(delta, abs=1e-14)
    # no transverse load, so reaction at node 1 in X equals -EA*delta/L
    assert m.node(1).reaction[0] == pytest.approx(-E * A * delta / L, rel=1e-10)


def test_mp_constraint_averaging():
    """Force the midpoint of a 3-node truss line to equal the average of the
    two ends. Stiffer trusses on the outer segments will not change that
    geometry constraint."""
    E = 1.0e7
    A = 1.0
    m = Model(ndm=2, ndf=2)
    m.add_material(ElasticIsotropic(tag=1, E=E, nu=0.3))
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, 1.0, 0.0)
    m.add_node(3, 2.0, 0.0)
    m.fix(1, [1, 1])
    m.fix(2, [0, 1])
    m.fix(3, [0, 1])
    m.add_element(Truss2D(tag=1, nodes=(1, 2), material=m.material(1), area=A))
    m.add_element(Truss2D(tag=2, nodes=(2, 3), material=m.material(1), area=A))
    # u_2x = 0.5*u_1x + 0.5*u_3x; u_1x is fixed → 0
    m.add_mp_constraint(
        MPConstraint(constrained=(2, 0), retained=[(1, 0, 0.5), (3, 0, 0.5)])
    )
    m.add_nodal_load(3, [50.0, 0.0])
    LinearStaticAnalysis(m).run()
    u1, u2, u3 = m.node(1).disp[0], m.node(2).disp[0], m.node(3).disp[0]
    assert u2 == pytest.approx(0.5 * (u1 + u3), abs=1e-14)


# ---------------------------------------------------------------------------
# Parity: penalty vs transformation


def test_penalty_matches_transformation_equal_dof():
    m_t, _ = _two_parallel_trusses_model(k_ratio=1.7)
    m_p, _ = _two_parallel_trusses_model(k_ratio=1.7)
    LinearStaticAnalysis(m_t, constraints="transformation").run()
    LinearStaticAnalysis(m_p, constraints="penalty").run()
    for tag in (1, 2, 3, 4):
        assert m_t.node(tag).disp == pytest.approx(m_p.node(tag).disp, rel=1e-6, abs=1e-12)


def test_penalty_matches_transformation_diaphragm():
    m_t, _ = _rigid_diaphragm_frame_model()
    m_p, _ = _rigid_diaphragm_frame_model()
    LinearStaticAnalysis(m_t, constraints="transformation").run()
    LinearStaticAnalysis(m_p, constraints="penalty").run()
    # penalty enforcement has finite accuracy proportional to 1/alpha_factor
    # (default 1e8). Tolerance must allow that on near-zero components.
    u_t_max = max(abs(m_t.node(t).disp).max() for t in range(1, 10))
    abs_tol = 1e-5 * u_t_max
    for tag in range(1, 10):
        assert m_t.node(tag).disp == pytest.approx(
            m_p.node(tag).disp, rel=1e-5, abs=abs_tol
        )


# ---------------------------------------------------------------------------
# Validation


def test_chained_constraints_rejected():
    """A retained DOF that is itself a slave of another constraint is not
    supported by the simple transformation handler."""
    m = Model(ndm=2, ndf=2)
    m.add_material(ElasticIsotropic(tag=1, E=1.0, nu=0.3))
    for i, x in enumerate((0.0, 1.0, 2.0)):
        m.add_node(i + 1, x, 0.0)
    m.fix(1, [1, 1])
    m.fix(2, [0, 1])
    m.fix(3, [0, 1])
    m.add_element(Truss2D(tag=1, nodes=(1, 2), material=m.material(1), area=1.0))
    m.add_element(Truss2D(tag=2, nodes=(2, 3), material=m.material(1), area=1.0))
    m.equal_dof(retained=1, constrained=2, dofs=[0])  # u2 = u1
    m.equal_dof(retained=2, constrained=3, dofs=[0])  # u3 = u2 (chained!)
    m.add_nodal_load(3, [10.0, 0.0])
    with pytest.raises(RuntimeError, match="chained"):
        LinearStaticAnalysis(m).run()


def test_double_slave_rejected():
    """A DOF cannot be the slave of two MP constraints simultaneously."""
    m = Model(ndm=2, ndf=2)
    m.add_material(ElasticIsotropic(tag=1, E=1.0, nu=0.3))
    for i, x in enumerate((0.0, 1.0, 2.0)):
        m.add_node(i + 1, x, 0.0)
    m.fix(1, [1, 1])
    m.fix(2, [0, 1])
    m.fix(3, [0, 1])
    m.add_element(Truss2D(tag=1, nodes=(1, 2), material=m.material(1), area=1.0))
    m.add_element(Truss2D(tag=2, nodes=(2, 3), material=m.material(1), area=1.0))
    m.equal_dof(retained=1, constrained=2, dofs=[0])
    m.equal_dof(retained=3, constrained=2, dofs=[0])  # node 2 dof 0 slaved twice
    m.add_nodal_load(3, [10.0, 0.0])
    with pytest.raises(RuntimeError, match="more than one"):
        LinearStaticAnalysis(m).run()


def test_equal_dof_validation():
    with pytest.raises(ValueError, match="must differ"):
        EqualDOF(retained=1, constrained=1, dofs=[0])
    with pytest.raises(ValueError, match="empty"):
        EqualDOF(retained=1, constrained=2, dofs=[])
    with pytest.raises(ValueError, match="duplicate"):
        EqualDOF(retained=1, constrained=2, dofs=[0, 0])


def test_rigid_diaphragm_requires_3d():
    m = Model(ndm=2, ndf=3)
    m.add_material(ElasticIsotropic(tag=1, E=1.0, nu=0.3))
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, 1.0, 0.0)
    m.fix(1, [1, 1, 1])
    m.fix(2, [0, 1, 1])
    m.add_element(BeamColumn2D(tag=1, nodes=(1, 2), material=m.material(1), area=1.0, Iz=1.0))
    m.rigid_diaphragm(master=1, slaves=[2], perp_dir=2)
    m.add_nodal_load(2, [1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="3D"):
        LinearStaticAnalysis(m).run()
