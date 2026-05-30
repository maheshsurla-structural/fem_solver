"""Phase 42 tests -- hyperelastic materials, TL Hex8, finite-strain J2,
and contact mechanics.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from femsolver import (
    ContactNodeToPlane3D,
    ElasticIsotropic,
    FiniteJ2Plasticity3D,
    Hex8,
    Hex8TL,
    Model,
    MooneyRivlin3D,
    NeoHookean3D,
)


# ============================================================ hyperelastic

class TestNeoHookean:
    def test_zero_stress_at_F_identity(self):
        m = NeoHookean3D(tag=1, E=2e6, nu=0.45)
        S, _ = m.response_S(np.eye(3))
        assert np.linalg.norm(S) < 1e-9
        sigma, _ = m.response_sigma(np.eye(3))
        assert np.linalg.norm(sigma) < 1e-9

    def test_validates_inputs(self):
        with pytest.raises(ValueError):
            NeoHookean3D(tag=1, E=-1, nu=0.3)
        with pytest.raises(ValueError):
            NeoHookean3D(tag=1, E=1e6, nu=0.6)

    def test_uniaxial_tension_positive_axial_stress(self):
        m = NeoHookean3D(tag=1, E=2e6, nu=0.45)
        lam = 1.5
        F = np.diag([lam, 1.0 / np.sqrt(lam), 1.0 / np.sqrt(lam)])
        sigma, _ = m.response_sigma(F)
        # axial component positive (tension), transverse non-positive
        assert sigma[0] > 0
        assert sigma[1] <= 0
        assert sigma[2] <= 0

    def test_negative_jacobian_raises(self):
        m = NeoHookean3D(tag=1, E=2e6, nu=0.45)
        F = -np.eye(3)            # det F = -1
        with pytest.raises(ValueError, match="J"):
            m.response_S(F)


class TestMooneyRivlin:
    def test_zero_stress_at_F_identity(self):
        m = MooneyRivlin3D(tag=1, c_10=0.3e6, c_01=0.05e6, K=1e9)
        S, _ = m.response_S(np.eye(3))
        assert np.linalg.norm(S) < 1.0
        sigma, _ = m.response_sigma(np.eye(3))
        assert np.linalg.norm(sigma) < 1.0

    def test_E_initial_consistent(self):
        # mu_0 = 2 (c10 + c01); E ≈ 3 mu_0 for incompressible
        m = MooneyRivlin3D(tag=1, c_10=0.3e6, c_01=0.05e6, K=1.0e9)
        assert m.mu_0 == pytest.approx(2.0 * (0.3 + 0.05) * 1e6, rel=1e-12)


# ============================================================ Hex8TL

def _unit_cube_model(material) -> Model:
    """Single Hex8TL cube on (0,0,0) - (1,1,1) with the supplied material."""
    m = Model(ndm=3, ndf=3)
    # Material in the registry is only used for bookkeeping; the
    # TL element uses its own (hyperelastic) material directly
    m.add_material(material)
    # Hex8 standard CCW ordering on bottom then top
    pts = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
           (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    for i, (x, y, z) in enumerate(pts, start=1):
        m.add_node(i, float(x), float(y), float(z))
    m.add_element(Hex8TL(1, (1, 2, 3, 4, 5, 6, 7, 8), material))
    return m


class TestHex8TL:
    def test_initial_K_symmetric_and_positive_definite(self):
        """At F = I, the TL Hex8 initial stiffness should be symmetric
        and positive-definite (apart from rigid-body modes)."""
        nh = NeoHookean3D(tag=1, E=2.0e6, nu=0.30)
        m = _unit_cube_model(nh)
        K = list(m.elements.values())[0].K_global()
        # Symmetry
        assert np.allclose(K, K.T, atol=1e-9)
        # Eigenvalues: 6 rigid-body zeros + 18 strictly positive
        w = np.linalg.eigvalsh(K)
        n_zero = int(np.sum(np.abs(w) < 1.0e-6 * abs(w).max()))
        assert n_zero == 6
        assert (w[-1] > 0) and (np.sort(w)[6] > 0)

    def test_internal_force_zero_at_F_identity(self):
        nh = NeoHookean3D(tag=1, E=2e6, nu=0.45)
        m = _unit_cube_model(nh)
        e = list(m.elements.values())[0]
        # All disp = 0 -> f_int should be ~ zero (round-off accumulation
        # across 8 Gauss points)
        f = e.f_int_global()
        assert np.linalg.norm(f) < 1e-8


# ============================================================ finite-strain J2

class TestFiniteJ2:
    def test_elastic_response_below_yield(self):
        m = FiniteJ2Plasticity3D(
            tag=1, E=2.0e11, nu=0.30, sigma_y0=400.0e6,
        )
        # Tiny uniaxial stretch -> elastic, alpha stays 0
        eps = 1e-3
        F = np.diag([1.0 + eps, 1.0, 1.0])
        S, _ = m.response_S(F)
        m.commit_state()
        # No accumulated plastic strain
        assert m.alpha_committed == 0.0

    def test_yielding_accumulates_plastic_strain(self):
        m = FiniteJ2Plasticity3D(
            tag=1, E=2e11, nu=0.30, sigma_y0=400e6, H=1e9,
        )
        # Stretch well past yield strain (~ 0.002)
        eps = 0.01
        F = np.diag([1.0 + eps, 1.0, 1.0])
        m.response_S(F)
        m.commit_state()
        assert m.alpha_committed > 0.0

    def test_validates_inputs(self):
        with pytest.raises(ValueError):
            FiniteJ2Plasticity3D(tag=1, E=-1, nu=0.3, sigma_y0=400e6)
        with pytest.raises(ValueError):
            FiniteJ2Plasticity3D(tag=1, E=2e11, nu=0.3, sigma_y0=-1)


# ============================================================ contact

def _single_node_contact_model(*, z_node: float, plane_z: float = 0.0,
                                  K_N: float = 1e6, mu: float = 0.0):
    """Build a 1-node 3D model + a contact element to a plane z = plane_z."""
    mat = ElasticIsotropic(1, E=1.0, nu=0.3, rho=0.0)   # dummy
    m = Model(ndm=3, ndf=3)
    m.add_material(mat)
    m.add_node(1, 0.0, 0.0, z_node)
    contact = ContactNodeToPlane3D(
        1, (1,), plane_point=[0.0, 0.0, plane_z],
        plane_normal=[0.0, 0.0, 1.0],
        K_N=K_N, mu=mu,
    )
    m.add_element(contact)
    return m, contact


class TestContact:
    def test_no_force_when_off_surface(self):
        m, c = _single_node_contact_model(z_node=0.1, plane_z=0.0)
        f = c.f_int_global()
        assert np.linalg.norm(f) == 0.0

    def test_normal_penalty_force_under_penetration(self):
        """If the slave node sits BELOW the plane (z=-0.01, plane at 0),
        the gap is negative, and a penalty pushes the node back up."""
        m, c = _single_node_contact_model(
            z_node=-0.01, plane_z=0.0, K_N=1.0e6,
        )
        # Node has no displacement; the gap comes from the
        # reference geometry (coords[2] - plane_z)
        f = c.f_int_global()
        # F_N = K_N * |-g_N| pointing in +z (the plane normal)
        assert f[2] == pytest.approx(1.0e6 * 0.01, rel=1e-12)
        assert f[0] == 0.0 and f[1] == 0.0

    def test_friction_stick_below_cap(self):
        """Apply small displacement: tangential stick branch."""
        m, c = _single_node_contact_model(
            z_node=-0.01, plane_z=0.0,
            K_N=1.0e6, mu=0.3,
        )
        m.node(1).disp[:] = [0.001, 0.0, 0.0]    # small lateral
        f = c.f_int_global()
        # Stick: F_T = K_T * u_T = 1e6 * 0.001 = 1000 N
        # F_N = 1e4 N; cap = 0.3 * 1e4 = 3000 N. Stick.
        assert abs(f[0] - 1000.0) < 1.0

    def test_friction_slip_above_cap(self):
        """Apply large displacement: tangential slip branch."""
        m, c = _single_node_contact_model(
            z_node=-0.01, plane_z=0.0,
            K_N=1.0e6, mu=0.3,
        )
        m.node(1).disp[:] = [0.5, 0.0, 0.0]      # huge lateral
        f = c.f_int_global()
        # Slip cap = 0.3 * 1e4 = 3000 N
        assert abs(np.linalg.norm(f[:2]) - 3000.0) < 1.0

    def test_validates_inputs(self):
        with pytest.raises(ValueError, match="K_N"):
            ContactNodeToPlane3D(
                1, (1,), plane_point=[0, 0, 0],
                plane_normal=[0, 0, 1], K_N=-1,
            )
        with pytest.raises(ValueError, match="mu"):
            ContactNodeToPlane3D(
                1, (1,), plane_point=[0, 0, 0],
                plane_normal=[0, 0, 1], K_N=1e6, mu=-1,
            )
