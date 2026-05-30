"""Phase 49.5 tests -- 3-node curved Timoshenko beam (CurvedBeam2D).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from femsolver import (
    CurvedBeam2D,
    ElasticIsotropic,
    LinearStaticAnalysis,
    Model,
)


# ============================================================ shape functions

class TestShapeFunctions:
    def test_at_xi_eq_minus_one(self):
        N = CurvedBeam2D.shape_functions(-1.0)
        np.testing.assert_allclose(N, [1.0, 0.0, 0.0], atol=1e-14)

    def test_at_xi_eq_zero(self):
        N = CurvedBeam2D.shape_functions(0.0)
        np.testing.assert_allclose(N, [0.0, 1.0, 0.0], atol=1e-14)

    def test_at_xi_eq_plus_one(self):
        N = CurvedBeam2D.shape_functions(1.0)
        np.testing.assert_allclose(N, [0.0, 0.0, 1.0], atol=1e-14)

    def test_partition_of_unity(self):
        for xi in (-0.7, -0.2, 0.0, 0.3, 0.9):
            assert CurvedBeam2D.shape_functions(xi).sum() == \
                pytest.approx(1.0, abs=1e-14)

    def test_derivative_consistency(self):
        # Sum of dN/dxi = 0 (since sum N = const = 1)
        for xi in (-0.5, 0.0, 0.5):
            assert CurvedBeam2D.dN_dxi(xi).sum() == pytest.approx(0.0, abs=1e-14)


# ============================================================ validation

class TestStraightCantilever:
    """A 3-node curved beam laid out in a straight line should reproduce
    the analytical Euler-Bernoulli cantilever to within Timoshenko shear
    correction terms (<< 1% for typical slenderness)."""

    def test_tip_deflection(self):
        mat = ElasticIsotropic(1, E=2e11, nu=0.30, rho=0)
        m = Model(ndm=2, ndf=3)
        m.add_material(mat)
        m.add_node(1, 0.0, 0.0)
        m.add_node(2, 0.5, 0.0)
        m.add_node(3, 1.0, 0.0)
        m.add_element(CurvedBeam2D(
            1, (1, 2, 3), mat, area=0.01, Iz=1e-6,
        ))
        m.fix(1, [1, 1, 1])
        P = 1000.0
        m.add_nodal_load(3, [0.0, -P, 0.0])
        LinearStaticAnalysis(m).run()
        u_anal = -P * 1.0**3 / (3.0 * mat.E * 1e-6)
        # Timoshenko adds shear flexibility ~ PL/(GA_s). For thin beam
        # L=1, A=0.01, A_s = A/1.2 = 0.0083, G ~ E/2.6 -> shear deflection
        # ~ 1e3 * 1 / (7.7e10 * 0.0083) ~ 1.6e-6 m which is ~0.1%
        assert m.node(3).disp[1] == pytest.approx(u_anal, rel=0.005)

    def test_rigid_body_translation(self):
        mat = ElasticIsotropic(1, E=2e11, nu=0.30, rho=0)
        m = Model(ndm=2, ndf=3)
        m.add_material(mat)
        m.add_node(1, 0.0, 0.0)
        m.add_node(2, 0.5, 0.0)
        m.add_node(3, 1.0, 0.0)
        e = CurvedBeam2D(1, (1, 2, 3), mat, area=0.01, Iz=1e-6)
        m.add_element(e)
        K = e.K_global()
        # 3 rigid-body modes in 2D: 2 translations + 1 rotation.
        # Their eigenvalues should be at machine epsilon relative to the
        # K magnitude (which is O(EA) ~ 2e9, so round-off ~ 2e-7).
        eigs = np.sort(np.linalg.eigvalsh(K))
        k_scale = float(np.abs(K).max())
        assert (np.abs(eigs[:3]) < 1e-12 * k_scale).all()
        assert eigs[3] > 1.0    # next mode has stiffness


class TestQuarterCircleArch:
    """Classic textbook quarter-circle cantilever with tangential tip
    load. Castigliano gives::

        delta_along_force = P R^3 (3*pi/4 - 2) / (E*I)
    """

    def _build(self, n_elem: int):
        mat = ElasticIsotropic(1, E=2e11, nu=0.30, rho=0)
        m = Model(ndm=2, ndf=3)
        m.add_material(mat)
        R = 1.0
        n_nodes = 2 * n_elem + 1
        for i in range(n_nodes):
            theta = (i / (n_nodes - 1)) * (math.pi / 2)
            m.add_node(i + 1, R * math.cos(theta), R * math.sin(theta))
        for e in range(n_elem):
            n1 = 2 * e + 1
            n2 = 2 * e + 2
            n3 = 2 * e + 3
            m.add_element(CurvedBeam2D(
                e + 1, (n1, n2, n3), mat,
                area=0.001, Iz=1e-8,
            ))
        m.fix(1, [1, 1, 1])
        # Tip is the last node, at (0, R). Tangential load = -x.
        tip_tag = n_nodes
        P = 1.0
        m.add_nodal_load(tip_tag, [-P, 0.0, 0.0])
        return m, mat, R, P, tip_tag

    def test_eight_elements_converged(self):
        m, mat, R, P, tip_tag = self._build(n_elem=8)
        LinearStaticAnalysis(m).run()
        delta_fe = -m.node(tip_tag).disp[0]   # in direction of force
        delta_anal = P * R**3 * (3 * math.pi / 4 - 2) / (mat.E * 1e-8)
        assert delta_fe == pytest.approx(delta_anal, rel=0.005)

    def test_convergence_with_refinement(self):
        # Eight elements should be more accurate than four
        m4, mat, R, P, tip4 = self._build(n_elem=4)
        m8, _, _, _, tip8 = self._build(n_elem=8)
        LinearStaticAnalysis(m4).run()
        LinearStaticAnalysis(m8).run()
        d4 = -m4.node(tip4).disp[0]
        d8 = -m8.node(tip8).disp[0]
        anal = P * R**3 * (3 * math.pi / 4 - 2) / (mat.E * 1e-8)
        err4 = abs(d4 - anal) / anal
        err8 = abs(d8 - anal) / anal
        assert err8 <= err4 + 1e-6


# ============================================================ validation API

class TestValidation:
    def test_rejects_zero_area(self):
        mat = ElasticIsotropic(1, E=1, nu=0.3, rho=0)
        with pytest.raises(ValueError, match="area"):
            CurvedBeam2D(1, (1, 2, 3), mat, area=0.0, Iz=1e-6)

    def test_rejects_negative_Iz(self):
        mat = ElasticIsotropic(1, E=1, nu=0.3, rho=0)
        with pytest.raises(ValueError, match="Iz"):
            CurvedBeam2D(1, (1, 2, 3), mat, area=0.01, Iz=-1e-6)

    def test_rejects_invalid_integration(self):
        mat = ElasticIsotropic(1, E=1, nu=0.3, rho=0)
        with pytest.raises(ValueError, match="integration_points"):
            CurvedBeam2D(
                1, (1, 2, 3), mat, area=0.01, Iz=1e-6,
                integration_points=4,
            )

    def test_zero_tangent_raises_on_K(self):
        # Place all three nodes at the same point -> zero tangent
        mat = ElasticIsotropic(1, E=2e11, nu=0.3, rho=0)
        m = Model(ndm=2, ndf=3)
        m.add_material(mat)
        m.add_node(1, 0.0, 0.0); m.add_node(2, 0.0, 0.0); m.add_node(3, 0.0, 0.0)
        e = CurvedBeam2D(1, (1, 2, 3), mat, area=0.01, Iz=1e-6)
        m.add_element(e)
        with pytest.raises(ValueError, match="zero tangent"):
            e.K_global()


# ============================================================ mass

class TestMassMatrix:
    def test_consistent_mass_total(self):
        mat = ElasticIsotropic(1, E=2e11, nu=0.3, rho=7850.0)
        m = Model(ndm=2, ndf=3)
        m.add_material(mat)
        m.add_node(1, 0.0, 0.0)
        m.add_node(2, 0.5, 0.0)
        m.add_node(3, 1.0, 0.0)
        A = 0.01
        e = CurvedBeam2D(1, (1, 2, 3), mat, area=A, Iz=1e-6)
        m.add_element(e)
        M = e.M_global()
        # Total mass per direction = rho * A * L = 7850 * 0.01 * 1.0 = 78.5
        # M sum over each direction's rows
        total_x = float(M[::3, :].sum())
        total_y = float(M[1::3, :].sum())
        assert total_x == pytest.approx(78.5, rel=1e-9)
        assert total_y == pytest.approx(78.5, rel=1e-9)

    def test_lumped_diagonal(self):
        mat = ElasticIsotropic(1, E=2e11, nu=0.3, rho=7850.0)
        e = CurvedBeam2D(1, (1, 2, 3), mat, area=0.01, Iz=1e-6)
        # Need a model to call K/M
        m = Model(ndm=2, ndf=3)
        m.add_material(mat)
        m.add_node(1, 0, 0); m.add_node(2, 0.5, 0); m.add_node(3, 1.0, 0)
        m.add_element(e)
        M_l = e.M_global(lumped=True)
        # Lumped is purely diagonal
        np.testing.assert_allclose(
            M_l - np.diag(np.diag(M_l)), 0.0, atol=1e-12,
        )
