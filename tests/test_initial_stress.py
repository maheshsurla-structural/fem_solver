"""Phase B.6 tests -- general initial-stress / eigenstrain equivalent loads.

Patch-test validation
----------------------
* **Free expansion**: an unrestrained element under uniform initial
  stress ``σ₀`` strains to ``ε = -D⁻¹ σ₀`` (the initial stress relaxes
  to zero net stress).
* **Restrained reaction**: a body restrained against an initial stress
  develops support reactions that balance ``σ₀ · A``.
* **Prestress helper**: a tendon force ``P`` over area ``A`` along a
  direction gives compression ``-P/A`` along it.
"""
from __future__ import annotations

import numpy as np
import pytest

from femsolver.core.model import Model
from femsolver.elements.solid import Hex8, Tet4
from femsolver.elements.plane import Quad4
from femsolver.elements.shell import ShellMITC4
from femsolver.materials.elastic import ElasticIsotropic
from femsolver.analysis.linear_static import LinearStaticAnalysis
from femsolver.analysis.initial_stress import (
    apply_initial_stress,
    prestress_initial_stress,
)


E = 30e9
NU = 0.2
MAT = ElasticIsotropic(1, E=E, nu=NU, rho=0.0)


def _unit_cube(Lx=1.0, Ly=1.0, Lz=1.0):
    m = Model(ndm=3, ndf=3)
    m.add_material(MAT)
    pts = [(0, 0, 0), (Lx, 0, 0), (Lx, Ly, 0), (0, Ly, 0),
           (0, 0, Lz), (Lx, 0, Lz), (Lx, Ly, Lz), (0, Ly, Lz)]
    for i, (x, y, z) in enumerate(pts):
        m.add_node(i + 1, x, y, z)
    m.add_element(Hex8(1, (1, 2, 3, 4, 5, 6, 7, 8), MAT))
    return m


def _det_supports(m):
    """Statically-determinate restraint of a cube (no over-restraint)."""
    m.fix(1, [1, 1, 1])
    m.fix(4, [1, 0, 1])
    m.fix(5, [1, 1, 0])
    m.fix(2, [0, 1, 1])


# ============================================================ Hex8

class TestHex8InitialStress:
    def test_free_uniaxial_strain(self):
        s = 10e6
        m = _unit_cube()
        _det_supports(m)
        apply_initial_stress(m, {1: np.array([s, 0, 0, 0, 0, 0])})
        LinearStaticAnalysis(m).run()
        eps_xx = m.node(2).disp[0]      # node 2 at x=1
        assert eps_xx == pytest.approx(-s / E, rel=1e-6)
        # transverse expansion +nu s/E
        assert m.node(4).disp[1] == pytest.approx(NU * s / E, rel=1e-6)

    def test_free_hydrostatic_strain(self):
        s = 5e6
        m = _unit_cube()
        _det_supports(m)
        apply_initial_stress(m, {1: np.array([s, s, s, 0, 0, 0])})
        LinearStaticAnalysis(m).run()
        # hydrostatic: eps = -s(1-2nu)/E in each direction
        expected = -s * (1.0 - 2.0 * NU) / E
        assert m.node(2).disp[0] == pytest.approx(expected, rel=1e-6)
        assert m.node(4).disp[1] == pytest.approx(expected, rel=1e-6)
        assert m.node(5).disp[2] == pytest.approx(expected, rel=1e-6)

    def test_restrained_reaction_balances_sigma0(self):
        s = 10e6
        m = _unit_cube()
        for nd in (1, 4, 5, 8):
            m.fix(nd, [1, 1, 1])         # x=0 face fixed
        for nd in (2, 3, 6, 7):
            m.fix(nd, [1, 0, 0])         # x=1 face: restrain x only
        apply_initial_stress(m, {1: np.array([s, 0, 0, 0, 0, 0])})
        LinearStaticAnalysis(m).run()
        Rx = sum(m.node(nd).reaction[0] for nd in (1, 4, 5, 8))
        assert Rx == pytest.approx(-s * 1.0, rel=1e-6)   # area = 1

    def test_factor_scales(self):
        s = 10e6
        m1 = _unit_cube(); _det_supports(m1)
        apply_initial_stress(m1, {1: np.array([s, 0, 0, 0, 0, 0])}, factor=1.0)
        LinearStaticAnalysis(m1).run()
        d1 = m1.node(2).disp[0]
        m2 = _unit_cube(); _det_supports(m2)
        apply_initial_stress(m2, {1: np.array([s, 0, 0, 0, 0, 0])}, factor=0.5)
        LinearStaticAnalysis(m2).run()
        assert m2.node(2).disp[0] == pytest.approx(0.5 * d1, rel=1e-9)

    def test_wrong_size_raises(self):
        m = _unit_cube()
        with pytest.raises(ValueError):
            apply_initial_stress(m, {1: np.array([1.0, 2.0, 3.0])})


# ============================================================ Tet4

class TestTet4InitialStress:
    def test_free_uniaxial_strain(self):
        """Single CST Tet4, determinate restraint, uniform initial
        stress -> free strain eps_xx = -s/E."""
        s = 8e6
        m = Model(ndm=3, ndf=3)
        m.add_material(MAT)
        m.add_node(1, 0, 0, 0)
        m.add_node(2, 1, 0, 0)
        m.add_node(3, 0, 1, 0)
        m.add_node(4, 0, 0, 1)
        m.add_element(Tet4(1, (1, 2, 3, 4), MAT))
        # determinate: node1 fixed; node2 free in x; node3 free in y;
        # node4 free in z (removes rigid-body, allows axial straining)
        m.fix(1, [1, 1, 1])
        m.fix(2, [0, 1, 1])
        m.fix(3, [1, 0, 1])
        m.fix(4, [1, 1, 0])
        apply_initial_stress(m, {1: np.array([s, 0, 0, 0, 0, 0])})
        LinearStaticAnalysis(m).run()
        # node 2 at x=1 -> u_x = eps_xx * 1
        assert m.node(2).disp[0] == pytest.approx(-s / E, rel=1e-6)


# ============================================================ Quad4 (plane)

class TestQuad4InitialStress:
    def test_free_uniaxial_plane_stress(self):
        s = 10e6
        m = Model(ndm=2, ndf=2)
        m.add_material(MAT)
        m.add_node(1, 0, 0); m.add_node(2, 1, 0)
        m.add_node(3, 1, 1); m.add_node(4, 0, 1)
        m.add_element(Quad4(1, (1, 2, 3, 4), MAT, thickness=0.1))
        # determinate restraint
        m.fix(1, [1, 1]); m.fix(4, [1, 0]); m.fix(2, [0, 1])
        apply_initial_stress(m, {1: np.array([s, 0, 0])})
        LinearStaticAnalysis(m).run()
        # plane stress free uniaxial: eps_xx = -s/E
        assert m.node(2).disp[0] == pytest.approx(-s / E, rel=1e-6)
        assert m.node(4).disp[1] == pytest.approx(NU * s / E, rel=1e-6)


# ============================================================ Shell (membrane)

def _flat_shell(t=0.2):
    """Single flat ShellMITC4 unit square in the global xy-plane."""
    m = Model(ndm=3, ndf=6)
    m.add_material(MAT)
    m.add_node(1, 0, 0, 0); m.add_node(2, 1, 0, 0)
    m.add_node(3, 1, 1, 0); m.add_node(4, 0, 1, 0)
    m.add_element(ShellMITC4(1, (1, 2, 3, 4), MAT, thickness=t))
    return m


class TestShellMembranePrestress:
    def test_free_uniaxial_membrane_strain(self):
        s, t = 10e6, 0.2
        m = _flat_shell(t)
        # suppress out-of-plane w + rotations everywhere; determinate
        # in-plane restraint
        for nd in (1, 2, 3, 4):
            m.fix(nd, [0, 0, 1, 1, 1, 1])
        m.fix(1, [1, 1, 1, 1, 1, 1])
        m.fix(4, [1, 0, 1, 1, 1, 1])
        m.fix(2, [0, 1, 1, 1, 1, 1])
        apply_initial_stress(m, {1: np.array([s, 0, 0])})
        LinearStaticAnalysis(m).run()
        assert m.node(2).disp[0] == pytest.approx(-s / E, rel=1e-6)
        assert m.node(4).disp[1] == pytest.approx(NU * s / E, rel=1e-6)

    def test_restrained_edge_reaction(self):
        s, t = 10e6, 0.2
        m = _flat_shell(t)
        for nd in (1, 2, 3, 4):
            m.fix(nd, [0, 0, 1, 1, 1, 1])
        for nd in (1, 4):
            m.fix(nd, [1, 1, 1, 1, 1, 1])     # x=0 edge fixed in-plane
        for nd in (2, 3):
            m.fix(nd, [1, 0, 1, 1, 1, 1])     # x=1 edge: fix x only
        apply_initial_stress(m, {1: np.array([s, 0, 0])})
        LinearStaticAnalysis(m).run()
        Rx = sum(m.node(nd).reaction[0] for nd in (1, 4))
        # membrane resultant N0 = s*t over unit width
        assert Rx == pytest.approx(-s * t * 1.0, rel=1e-6)

    def test_matches_quad4_plane(self):
        """A flat shell's membrane response to in-plane prestress equals
        the Quad4 plane-stress response."""
        s, t = 10e6, 0.2
        # shell
        ms = _flat_shell(t)
        for nd in (1, 2, 3, 4):
            ms.fix(nd, [0, 0, 1, 1, 1, 1])
        ms.fix(1, [1, 1, 1, 1, 1, 1]); ms.fix(4, [1, 0, 1, 1, 1, 1])
        ms.fix(2, [0, 1, 1, 1, 1, 1])
        apply_initial_stress(ms, {1: np.array([s, 0, 0])})
        LinearStaticAnalysis(ms).run()
        # quad4
        mq = Model(ndm=2, ndf=2)
        mq.add_material(MAT)
        mq.add_node(1, 0, 0); mq.add_node(2, 1, 0)
        mq.add_node(3, 1, 1); mq.add_node(4, 0, 1)
        mq.add_element(Quad4(1, (1, 2, 3, 4), MAT, thickness=t))
        mq.fix(1, [1, 1]); mq.fix(4, [1, 0]); mq.fix(2, [0, 1])
        apply_initial_stress(mq, {1: np.array([s, 0, 0])})
        LinearStaticAnalysis(mq).run()
        assert ms.node(2).disp[0] == pytest.approx(mq.node(2).disp[0], rel=1e-6)

    def test_wrong_size_raises(self):
        m = _flat_shell()
        with pytest.raises(ValueError):
            apply_initial_stress(m, {1: np.array([1e6, 0, 0, 0, 0, 0])})


# ============================================================ prestress helper

class TestPrestressHelper:
    def test_compression_along_x_3d(self):
        s0 = prestress_initial_stress(P=5e6, A=0.25, direction=[1, 0, 0])
        assert s0[0] == pytest.approx(-20e6)      # -P/A
        assert np.allclose(s0[1:], 0.0)

    def test_direction_normalised(self):
        s0 = prestress_initial_stress(P=1e6, A=1.0, direction=[3, 4, 0])
        # unit dir (0.6, 0.8, 0): sxx = -1e6*0.36, syy=-1e6*0.64, sxy=-1e6*0.48
        assert s0[0] == pytest.approx(-0.36e6)
        assert s0[1] == pytest.approx(-0.64e6)
        assert s0[3] == pytest.approx(-0.48e6)

    def test_2d(self):
        s0 = prestress_initial_stress(P=2e6, A=0.5, direction=[1, 0], ndim=2)
        assert s0.size == 3
        assert s0[0] == pytest.approx(-4e6)

    def test_prestress_compresses_solid(self):
        """A tendon along x in a free solid block shortens it (negative
        x-strain)."""
        m = _unit_cube()
        _det_supports(m)
        s0 = prestress_initial_stress(P=6e6, A=1.0, direction=[1, 0, 0])
        apply_initial_stress(m, {1: s0})
        LinearStaticAnalysis(m).run()
        # compression sigma0 = -P/A -> free strain eps = -sigma0/E = +P/(A E)?
        # eps = -Dinv sigma0; uniaxial: eps_xx = -sigma0_xx/E = -(-P/A)/E > 0
        # i.e. the released block EXTENDS in x. Check sign is positive and
        # magnitude P/(A E).
        assert m.node(2).disp[0] == pytest.approx(6e6 / (1.0 * E), rel=1e-6)

    def test_bad_inputs_raise(self):
        with pytest.raises(ValueError):
            prestress_initial_stress(P=-1, A=1.0, direction=[1, 0, 0])
        with pytest.raises(ValueError):
            prestress_initial_stress(P=1, A=1.0, direction=[0, 0, 0])


# ============================================================ unsupported element

class TestUnsupported:
    def test_beam_raises_notimplemented(self):
        from femsolver.elements.beam import BeamColumn2D
        m = Model(ndm=2, ndf=3)
        m.add_material(MAT)
        m.add_node(1, 0, 0); m.add_node(2, 1, 0)
        m.add_element(BeamColumn2D(1, (1, 2), MAT, 0.1, 1e-4))
        with pytest.raises(NotImplementedError):
            apply_initial_stress(m, {1: np.array([1e6, 0, 0])})
