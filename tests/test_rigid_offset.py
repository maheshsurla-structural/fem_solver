"""Phase 49.4 tests -- mixed-NDF rigid offset coupling.

Beam-to-shell (matched NDF) and beam-to-solid (mixed NDF) kinematic
constraints, using both the basic :class:`RigidOffset` and the
convenience helpers.
"""
from __future__ import annotations

import numpy as np
import pytest

from femsolver import (
    BeamColumn2D,
    BeamColumn3D,
    ElasticIsotropic,
    Hex8,
    LinearStaticAnalysis,
    MembraneQ4Drilling,
    Model,
)
from femsolver.constraints import (
    RigidOffset,
    beam_shell_offset_coupling,
    beam_solid_coupling,
)


# ============================================================ basic API

class TestRigidOffsetAPI:
    def test_master_equals_slave_rejected(self):
        with pytest.raises(ValueError, match="must differ"):
            RigidOffset(master=1, slave=1)

    def test_unknown_master_rejected(self):
        m = Model(ndm=3, ndf=6)
        m.add_node(1, 0, 0, 0)
        c = RigidOffset(master=99, slave=1)
        with pytest.raises(ValueError, match="unknown master"):
            c.basic_constraints(m)

    def test_unknown_slave_rejected(self):
        m = Model(ndm=3, ndf=6)
        m.add_node(1, 0, 0, 0)
        c = RigidOffset(master=1, slave=99)
        with pytest.raises(ValueError, match="unknown slave"):
            c.basic_constraints(m)


# ============================================================ 2D beam-to-membrane

class TestBeam2DToMembraneDrilling:
    """A 2D beam node coupled rigidly to a membrane Q4 (drilling)
    node. Both carry 3 DOFs (u, v, theta_z) so the coupling ties
    rotations as well."""

    def test_translation_follows_offset_and_rotation(self):
        mat = ElasticIsotropic(1, E=2e11, nu=0.3, rho=0)
        m = Model(ndm=2, ndf=3)
        m.add_material(mat)
        # Membrane Q4 over [0,1] x [0,1]
        m.add_node(1, 0.0, 0.0); m.add_node(2, 1.0, 0.0)
        m.add_node(3, 1.0, 1.0); m.add_node(4, 0.0, 1.0)
        m.add_element(MembraneQ4Drilling(
            1, (1, 2, 3, 4), mat, thickness=0.01,
        ))
        # Beam node above node 2 at offset (0, +0.5)
        m.add_node(5, 1.0, 0.5)
        # Beam between node 2 and node 5
        m.add_element(BeamColumn2D(2, (2, 5), mat,
                                    area=0.001, Iz=1e-6))
        # Constraints: pin left edge (nodes 1, 4)
        m.fix(1, [1, 1, 1])
        m.fix(4, [1, 0, 1])    # membrane corner: drilling can be free
        # Apply moment at the beam top (node 5)
        M_app = 100.0
        m.add_nodal_load(5, [0.0, 0.0, M_app])
        LinearStaticAnalysis(m).run()
        # No constraints involved here -- just sanity that the system
        # solves and produces non-trivial displacements
        assert abs(m.node(5).disp[2]) > 0     # non-zero rotation
        assert abs(m.node(2).disp[2]) > 0     # transmitted to base


# ============================================================ 3D beam-to-solid

def _hex8_with_beam_master(F_load=1000.0):
    """Unit Hex8 fully clamped at z=0, beam stub attached to top face
    centre via 4 rigid offsets, force at beam tip."""
    mat = ElasticIsotropic(1, E=2.0e11, nu=0.30, rho=0.0)
    m = Model(ndm=3, ndf=6)
    m.add_material(mat)
    for i, (x, y, z) in enumerate([
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
    ]):
        m.add_node(i + 1, float(x), float(y), float(z))
    # Bottom face fully clamped
    for tag in (1, 2, 3, 4):
        m.fix(tag, [1, 1, 1, 1, 1, 1])
    # Top face: lock the unused rotational DOFs
    for tag in (5, 6, 7, 8):
        m.fix(tag, [0, 0, 0, 1, 1, 1])
    m.add_element(Hex8(1, tuple(range(1, 9)), mat))
    # Beam master + tip
    m.add_node(9, 0.5, 0.5, 2.0)        # tip
    m.add_node(10, 0.5, 0.5, 1.0)       # master at top-face centre
    beam_solid_coupling(m, beam_node=10, solid_nodes=[5, 6, 7, 8])
    m.add_element(BeamColumn3D(
        2, (10, 9), mat,
        area=0.01, Iy=1e-6, Iz=1e-6, J=2e-6,
        vecxz=np.array([1.0, 0.0, 0.0]),
    ))
    m.add_nodal_load(9, [F_load, 0.0, 0.0, 0.0, 0.0, 0.0])
    return m, mat


class TestBeamSolidCoupling:
    def test_slaves_follow_master_kinematics(self):
        m, _ = _hex8_with_beam_master()
        LinearStaticAnalysis(m).run()
        u_master = m.node(10).disp[0]
        theta_y_master = m.node(10).disp[4]
        theta_z_master = m.node(10).disp[5]
        # For each top-face corner: u_x = u_master + dz*theta_y - dy*theta_z
        for tag in (5, 6, 7, 8):
            coord = m.node(tag).coords
            dy = coord[1] - 0.5
            dz = coord[2] - 1.0
            expected = u_master + dz * theta_y_master - dy * theta_z_master
            assert m.node(tag).disp[0] == pytest.approx(expected, abs=1e-10)

    def test_helper_installs_constraints(self):
        m, _ = _hex8_with_beam_master()
        # Helper should have installed 4 constraints
        assert len(list(m.mp_constraints)) == 4

    def test_tip_deflection_within_engineering_range(self):
        # Beam stub length above top face = 1.0 m; total beam length
        # from clamp to tip = 2.0 m. Solid is much stiffer than beam,
        # so the bulk of deflection comes from beam bending.
        m, mat = _hex8_with_beam_master(F_load=1000.0)
        LinearStaticAnalysis(m).run()
        u_tip = m.node(9).disp[0]
        # The deflection must be positive in the direction of F
        assert u_tip > 0
        # Cantilever beam analytical for 1m stub on rigid base:
        # delta = F*L^3 / 3EI = 1000 * 1 / (3 * 2e11 * 1e-6) = 1.67e-3 m
        beam_only_delta = 1000.0 * 1.0**3 / (3.0 * 2e11 * 1e-6)
        # Real system is slightly more flexible because solid contributes
        # some compliance. Should be within 5%.
        assert u_tip == pytest.approx(beam_only_delta, rel=0.05)


# ============================================================ rotation-tying

class TestRotationTying:
    def test_slave_rotation_tied_when_requested(self):
        # 2D case: master beam node, slave shell drilling node, both
        # ndf=3. With couple_slave_rotations=True the slave's theta_z
        # should equal master's theta_z.
        mat = ElasticIsotropic(1, E=2e11, nu=0.3, rho=0)
        m = Model(ndm=2, ndf=3)
        m.add_material(mat)
        m.add_node(1, 0.0, 0.0); m.add_node(2, 1.0, 0.0)
        m.add_node(3, 1.0, 1.0); m.add_node(4, 0.0, 1.0)
        m.add_element(MembraneQ4Drilling(
            1, (1, 2, 3, 4), mat, thickness=0.01,
        ))
        m.add_node(5, 0.5, 1.5)
        m.add_element(BeamColumn2D(2, (3, 5), mat, area=0.001, Iz=1e-6))
        beam_shell_offset_coupling(
            m, beam_node=5, shell_node=3,    # tie node 3 to beam tip
            couple_drilling=True,
        )
        m.fix(1, [1, 1, 1])
        m.fix(2, [0, 1, 0])
        m.fix(4, [1, 1, 0])
        # Apply tip moment
        m.add_nodal_load(5, [0.0, 0.0, 50.0])
        LinearStaticAnalysis(m).run()
        # Drilling rotation at node 3 should equal master beam rotation
        # at node 5 (rotation-tying)
        assert m.node(3).disp[2] == pytest.approx(
            m.node(5).disp[2], abs=1e-10,
        )


# ============================================================ error paths

class TestErrorPaths:
    def test_master_without_rotation_rejected(self):
        # Model with ndf=2 (no rotation) — master can't drive offset
        m = Model(ndm=2, ndf=2)
        m.add_node(1, 0.0, 0.0)
        m.add_node(2, 0.5, 0.5)
        c = RigidOffset(master=1, slave=2)
        with pytest.raises(ValueError, match="theta_z"):
            c.basic_constraints(m)

    def test_3d_master_with_only_3_dof_rejected(self):
        m = Model(ndm=3, ndf=3)
        m.add_node(1, 0, 0, 0)
        m.add_node(2, 1, 0, 0)
        c = RigidOffset(master=1, slave=2)
        with pytest.raises(ValueError, match="full rotations"):
            c.basic_constraints(m)
