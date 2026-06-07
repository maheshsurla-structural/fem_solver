"""Phase B.2 tests -- 3-D cable element + force-density form-finding.

Validation
----------
* **Cable-beam analogy** (exact): a cable under vertical point loads
  takes the shape ``y(x) = -M_beam(x)/H`` where ``M_beam`` is the
  simply-supported bending moment under the same loads and ``H`` is the
  (constant) horizontal tension. FDM reproduces this to machine
  precision.
* **Parabola sag**: equal point loads -> midspan sag ``W·L/(8H)``-ish
  via the exact analogy.
* **Catenary limit**: a shallow self-weight cable's FDM sag matches the
  closed-form catenary within ~1 %.
* **CableElement3D** reduces to :class:`Truss3D` with no sag, applies
  the Ernst modulus with sag, and solves inside a real 3-D model.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from femsolver.core.model import Model
from femsolver.elements.truss import Truss3D
from femsolver.materials.elastic import ElasticIsotropic
from femsolver.analysis.linear_static import LinearStaticAnalysis
from femsolver.bridges.cable import (
    CableElement3D,
    catenary_sag,
    ernst_equivalent_modulus,
)
from femsolver.bridges.form_finding import (
    FormFindingResult,
    force_density_form_find,
)


# ============================================================ FDM core

def _horizontal_cable(L=40.0, n=8):
    xs = np.linspace(0, L, n + 1)
    coords = np.column_stack([xs, np.zeros(n + 1), np.zeros(n + 1)])
    branches = np.array([[i, i + 1] for i in range(n)])
    return coords, branches, xs


class TestForceDensityCableBeamAnalogy:
    def test_shape_equals_beam_moment_over_H(self):
        """Exact: cable z(x) = -M_beam(x)/H under vertical point loads."""
        L, n, P, q = 40.0, 8, 50e3, 200e3
        coords, branches, xs = _horizontal_cable(L, n)
        loads = np.zeros((n + 1, 3))
        loads[1:n, 2] = -P
        res = force_density_form_find(coords, branches, [0, n], q, loads=loads)

        dx = L / n
        H = q * dx
        W = (n - 1) * P
        RA = W / 2.0
        M = np.array([
            RA * xs[i] - sum(P * (xs[i] - xs[j]) for j in range(1, n)
                             if xs[j] < xs[i] - 1e-9)
            for i in range(n + 1)
        ])
        z_exact = -M / H
        assert np.allclose(res.coords[:, 2], z_exact, atol=1e-12)

    def test_equilibrium_residual_is_zero(self):
        L, n, P, q = 40.0, 8, 50e3, 200e3
        coords, branches, _ = _horizontal_cable(L, n)
        loads = np.zeros((n + 1, 3))
        loads[1:n, 2] = -P
        res = force_density_form_find(coords, branches, [0, n], q, loads=loads)
        assert res.residual < 1e-6

    def test_member_tension_equals_qL(self):
        L, n, P, q = 40.0, 8, 50e3, 200e3
        coords, branches, _ = _horizontal_cable(L, n)
        loads = np.zeros((n + 1, 3))
        loads[1:n, 2] = -P
        res = force_density_form_find(coords, branches, [0, n], q, loads=loads)
        assert np.allclose(res.tensions, q * res.lengths)
        # steeper end segments carry more tension than the flat middle
        assert res.tensions[0] > res.tensions[n // 2]

    def test_higher_q_less_sag(self):
        L, n, P = 40.0, 8, 50e3
        coords, branches, _ = _horizontal_cable(L, n)
        loads = np.zeros((n + 1, 3))
        loads[1:n, 2] = -P
        soft = force_density_form_find(coords, branches, [0, n], 100e3, loads=loads)
        stiff = force_density_form_find(coords, branches, [0, n], 400e3, loads=loads)
        assert abs(stiff.coords[n // 2, 2]) < abs(soft.coords[n // 2, 2])

    def test_midspan_sag_parabola(self):
        """Equal point loads -> midspan sag = M_max/H."""
        L, n, P, q = 40.0, 8, 50e3, 200e3
        coords, branches, _ = _horizontal_cable(L, n)
        loads = np.zeros((n + 1, 3))
        loads[1:n, 2] = -P
        res = force_density_form_find(coords, branches, [0, n], q, loads=loads)
        H = q * (L / n)
        W = (n - 1) * P
        # SS beam max moment under (n-1) equal point loads at equal spacing
        M_max = W / 2.0 * (L / 2.0) - sum(
            P * (L / 2.0 - j * L / n) for j in range(1, n // 2 + 1)
            if j * L / n < L / 2.0
        )
        assert abs(res.coords[n // 2, 2]) == pytest.approx(M_max / H, rel=1e-9)


class TestCatenaryLimit:
    def test_shallow_selfweight_matches_catenary(self):
        """A shallow self-weight cable: lump weight to nodes, FDM sag
        ~ closed-form catenary within 1 %."""
        L, n = 100.0, 40
        w = 200.0          # N/m self weight
        H = 2.0e5          # large horizontal tension -> shallow
        q = H / (L / n)    # force density giving this H
        coords, branches, xs = _horizontal_cable(L, n)
        loads = np.zeros((n + 1, 3))
        dx = L / n
        # tributary self-weight at each interior node
        loads[1:n, 2] = -w * dx
        res = force_density_form_find(coords, branches, [0, n], q, loads=loads)
        fdm_sag = -res.coords[n // 2, 2]
        cat_sag = catenary_sag(L_h=L, w=w, H=H)
        assert fdm_sag == pytest.approx(cat_sag, rel=0.01)


class TestForceDensity3D:
    def test_symmetric_net_symmetric_shape(self):
        """A symmetric square cable net under a central downward load
        finds a symmetric shape (centre node drops, stays centred)."""
        # 4 corner anchors + 1 centre free node
        coords = np.array([
            [-5.0, -5.0, 0.0],
            [5.0, -5.0, 0.0],
            [5.0, 5.0, 0.0],
            [-5.0, 5.0, 0.0],
            [0.0, 0.0, 0.0],     # centre (free)
        ])
        branches = np.array([[4, 0], [4, 1], [4, 2], [4, 3]])
        loads = np.zeros((5, 3))
        loads[4, 2] = -10e3
        res = force_density_form_find(coords, branches, [0, 1, 2, 3], 50e3,
                                       loads=loads)
        c = res.coords[4]
        assert c[0] == pytest.approx(0.0, abs=1e-9)   # stays centred x
        assert c[1] == pytest.approx(0.0, abs=1e-9)   # stays centred y
        assert c[2] < 0.0                              # drops
        # all four cables equal tension by symmetry
        assert np.allclose(res.tensions, res.tensions[0])


class TestFormFindingErrors:
    def test_all_fixed_raises(self):
        coords = np.zeros((2, 3))
        with pytest.raises(ValueError):
            force_density_form_find(coords, np.array([[0, 1]]), [0, 1], 1.0)

    def test_nonpositive_q_raises(self):
        coords, branches, _ = _horizontal_cable(10.0, 4)
        with pytest.raises(ValueError):
            force_density_form_find(coords, branches, [0, 4], -1.0)

    def test_scalar_and_array_q_equivalent(self):
        coords, branches, _ = _horizontal_cable(20.0, 5)
        loads = np.zeros((6, 3))
        loads[1:5, 2] = -1e3
        r_scalar = force_density_form_find(coords, branches, [0, 5], 100e3,
                                            loads=loads)
        r_array = force_density_form_find(
            coords, branches, [0, 5], np.full(5, 100e3), loads=loads
        )
        assert np.allclose(r_scalar.coords, r_array.coords)


# ============================================================ CableElement3D

class _Mat:
    """Minimal material stand-in (E, rho)."""
    def __init__(self, E, rho=7850.0):
        self.E = E
        self.rho = rho


class TestCableElement3D:
    def _two_node_model(self):
        m = Model(ndm=3, ndf=3)
        mat = ElasticIsotropic(1, E=200e9, nu=0.3, rho=7850.0)
        m.add_material(mat)
        m.add_node(1, 0.0, 0.0, 0.0)
        m.add_node(2, 3.0, 0.0, 4.0)   # length 5, inclined
        return m, mat

    def test_reduces_to_truss3d_without_sag(self):
        m, mat = self._two_node_model()
        cable = CableElement3D(1, (1, 2), mat, 0.01)  # no T_operating
        m.add_element(cable)
        truss = Truss3D(2, (1, 2), mat, 0.01)
        truss.bind(m)
        assert np.allclose(cable.K_global(), truss.K_global())

    def test_ernst_reduces_stiffness(self):
        m, mat = self._two_node_model()
        # low operating tension + heavy cable -> reduced modulus
        cable = CableElement3D(1, (1, 2), mat, 0.01,
                               gamma_eff=500.0, T_operating=2e5,
                               vertical_axis=2)
        m.add_element(cable)
        assert cable.effective_modulus() < mat.E
        # K should be scaled by E_eq/E vs the plain truss
        truss = Truss3D(2, (1, 2), mat, 0.01)
        truss.bind(m)
        ratio = cable.effective_modulus() / mat.E
        assert np.allclose(cable.K_global(), ratio * truss.K_global())

    def test_horizontal_projection(self):
        m, mat = self._two_node_model()
        cable = CableElement3D(1, (1, 2), mat, 0.01, vertical_axis=2)
        m.add_element(cable)
        # chord (3,0,4): horizontal projection (z vertical) = sqrt(3^2+0)=3
        assert cable.horizontal_projection() == pytest.approx(3.0)

    def test_vertical_axis_changes_projection(self):
        m, mat = self._two_node_model()
        # if Y is vertical, horizontal projection uses x,z = sqrt(9+16)=5
        cable = CableElement3D(1, (1, 2), mat, 0.01, vertical_axis=1)
        m.add_element(cable)
        assert cable.horizontal_projection() == pytest.approx(5.0)

    def test_solves_in_model_vcable(self):
        """Symmetric V of two cables, apex loaded down -> solves,
        apex drops, stays centred; Ernst sag increases the drop."""
        def build(T_op):
            m = Model(ndm=3, ndf=3)
            mat = ElasticIsotropic(1, E=200e9, nu=0.3, rho=7850.0)
            m.add_material(mat)
            m.add_node(1, -3.0, 0.0, 4.0)
            m.add_node(2, 3.0, 0.0, 4.0)
            m.add_node(3, 0.0, 0.0, 0.0)    # apex
            kw = {} if T_op is None else dict(gamma_eff=300.0, T_operating=T_op)
            m.add_element(CableElement3D(1, (1, 3), mat, 0.005, **kw))
            m.add_element(CableElement3D(2, (2, 3), mat, 0.005, **kw))
            m.fix(1, [1, 1, 1])
            m.fix(2, [1, 1, 1])
            m.fix(3, [0, 1, 0])             # restrain out-of-plane y
            m.add_nodal_load(3, [0.0, 0.0, -100e3])
            LinearStaticAnalysis(m).run()
            return m

        m_stiff = build(None)
        apex = m_stiff.node(3).disp
        assert apex[2] < 0.0                       # drops
        assert apex[0] == pytest.approx(0.0, abs=1e-9)   # symmetric

        m_sag = build(1e5)                          # low tension -> sag
        # softer cable -> larger downward displacement
        assert abs(m_sag.node(3).disp[2]) > abs(apex[2])
