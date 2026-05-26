"""Tests for Phase 22 composite-laminate machinery:

* ``OrthotropicLamina`` -- single-ply constitutive
* ``ShellLayer`` with ``theta_deg`` for fiber orientation
* ``LayeredShellSection`` summing rotated Q_bar over thickness (CLT)
* Curved-shell mesh helpers
"""
import math

import numpy as np
import pytest

from femsolver import (
    ElasticIsotropic,
    LayeredShellSection,
    Model,
    OrthotropicLamina,
    ShellLayer,
    ShellMITC4,
    cylindrical_shell_mesh,
    spherical_cap_mesh,
)
from femsolver.analysis.linear_static import LinearStaticAnalysis


# ====================================================== orthotropic lamina

def test_lamina_validates_positive_moduli():
    with pytest.raises(ValueError, match="E1"):
        OrthotropicLamina(E1=0.0, E2=10e9, G12=5e9, nu12=0.3)
    with pytest.raises(ValueError, match="E2"):
        OrthotropicLamina(E1=181e9, E2=-1.0, G12=5e9, nu12=0.3)
    with pytest.raises(ValueError, match="G12"):
        OrthotropicLamina(E1=181e9, E2=10e9, G12=0.0, nu12=0.3)


def test_lamina_validates_poisson_ratio():
    # nu12 must satisfy -1 < nu12 < sqrt(E1/E2)
    with pytest.raises(ValueError, match="nu12"):
        OrthotropicLamina(E1=10e9, E2=10e9, G12=4e9, nu12=2.0)
    with pytest.raises(ValueError, match="nu12"):
        OrthotropicLamina(E1=181e9, E2=10.3e9, G12=7e9, nu12=-1.5)


def test_lamina_Q_at_theta_zero_matches_local():
    ply = OrthotropicLamina(E1=181e9, E2=10.3e9, G12=7.17e9, nu12=0.28)
    Q_local = ply.Q_lamina()
    Q_zero = ply.Q_bar(0.0)
    np.testing.assert_allclose(Q_local, Q_zero, atol=1e-6 * np.max(np.abs(Q_local)))


def test_lamina_Q_at_theta_90_swaps_E1_and_E2():
    """At 90 degrees the fiber direction aligns with the global y axis,
    so Q11 = E2 / (1 - nu12 nu21) and Q22 = E1 / ditto."""
    E1, E2, G12, nu12 = 181e9, 10.3e9, 7.17e9, 0.28
    ply = OrthotropicLamina(E1=E1, E2=E2, G12=G12, nu12=nu12)
    Q90 = ply.Q_bar(90.0)
    denom = 1.0 - nu12 * nu12 * E2 / E1
    Q11_expected = E2 / denom
    Q22_expected = E1 / denom
    assert Q90[0, 0] == pytest.approx(Q11_expected, rel=1e-10)
    assert Q90[1, 1] == pytest.approx(Q22_expected, rel=1e-10)
    # Q12 unchanged by 90-deg rotation
    assert Q90[0, 1] == pytest.approx(nu12 * E2 / denom, rel=1e-10)


def test_lamina_Q16_Q26_zero_at_theta_zero():
    """Shear-extension coupling Q16 / Q26 vanishes at 0 and 90 degrees
    (and is maximum near 45 degrees)."""
    ply = OrthotropicLamina(E1=181e9, E2=10.3e9, G12=7.17e9, nu12=0.28)
    Q = ply.Q_bar(0.0)
    assert abs(Q[0, 2]) < 1e-6 * np.max(np.abs(Q))
    assert abs(Q[1, 2]) < 1e-6 * np.max(np.abs(Q))
    Q90 = ply.Q_bar(90.0)
    assert abs(Q90[0, 2]) < 1e-6 * np.max(np.abs(Q90))


def test_lamina_Q45_has_shear_coupling():
    ply = OrthotropicLamina(E1=181e9, E2=10.3e9, G12=7.17e9, nu12=0.28)
    Q45 = ply.Q_bar(45.0)
    assert abs(Q45[0, 2]) > 1e8       # nontrivial Q16
    assert abs(Q45[1, 2]) > 1e8       # nontrivial Q26


def test_lamina_isotropic_equivalent_recovers_isotropic():
    """If E1 = E2 = E and G12 = E/(2(1+nu)), the lamina is isotropic
    and Q_bar should be independent of theta."""
    E, nu = 70e9, 0.3
    G = E / (2.0 * (1.0 + nu))
    ply = OrthotropicLamina(E1=E, E2=E, G12=G, nu12=nu)
    Q0 = ply.Q_bar(0.0)
    Q45 = ply.Q_bar(45.0)
    Q67 = ply.Q_bar(67.5)
    np.testing.assert_allclose(Q0, Q45, atol=1e-3)
    np.testing.assert_allclose(Q0, Q67, atol=1e-3)


# ====================================================== layer with theta

def test_shell_layer_accepts_theta():
    ply = OrthotropicLamina(E1=181e9, E2=10.3e9, G12=7.17e9, nu12=0.28)
    layer = ShellLayer(ply, thickness=0.5e-3, z_mid=0.0, theta_deg=45.0)
    assert layer.theta_deg == 45.0
    assert layer.thickness == 0.5e-3


def test_layered_section_isotropic_backwards_compat():
    """A layered section built from isotropic layers reproduces the
    pre-Phase-22 behaviour (no orientation, isotropic Q_bar)."""
    mat = ElasticIsotropic(1, E=70e9, nu=0.3)
    sec = LayeredShellSection.from_layers_centered([
        (mat, 1e-3),
        (mat, 1e-3),
    ])
    D_m = sec.D_membrane()
    # A = E*t/(1-nu^2); two layers -> 2 * E * 1e-3 / (1 - 0.09)
    A_expected = 2.0 * 70e9 * 1e-3 / (1.0 - 0.3 * 0.3)
    assert D_m[0, 0] == pytest.approx(A_expected, rel=1e-10)


def test_layered_section_orthotropic_at_zero_theta():
    """Single orthotropic layer at theta=0: membrane stiffness equals
    Q_lamina * t."""
    ply = OrthotropicLamina(E1=181e9, E2=10.3e9, G12=7.17e9, nu12=0.28)
    sec = LayeredShellSection.from_layers_centered([
        (ply, 1e-3, 0.0),
    ])
    D_m = sec.D_membrane()
    Q = ply.Q_lamina() * 1e-3
    np.testing.assert_allclose(D_m, Q, rtol=1e-10)


def test_cross_ply_symmetric_laminate_has_zero_coupling():
    """[0/90/90/0] laminate is symmetric -> B = 0 (to roundoff)."""
    ply = OrthotropicLamina(E1=181e9, E2=10.3e9, G12=7.17e9, nu12=0.28)
    sec = LayeredShellSection.from_layers_centered([
        (ply, 0.5e-3, 0.0),
        (ply, 0.5e-3, 90.0),
        (ply, 0.5e-3, 90.0),
        (ply, 0.5e-3, 0.0),
    ])
    B = sec.D_coupling()
    A = sec.D_membrane()
    # B should be zero up to floating-point noise
    assert np.max(np.abs(B)) < 1e-9 * np.max(np.abs(A))


def test_unbalanced_laminate_has_nonzero_coupling():
    """An asymmetric stack [0/90] gives nonzero B."""
    ply = OrthotropicLamina(E1=181e9, E2=10.3e9, G12=7.17e9, nu12=0.28)
    sec = LayeredShellSection.from_layers_centered([
        (ply, 0.5e-3, 0.0),
        (ply, 0.5e-3, 90.0),
    ])
    B = sec.D_coupling()
    A = sec.D_membrane()
    assert np.max(np.abs(B)) > 1e-4 * np.max(np.abs(A))


def test_symmetric_balanced_cross_ply_has_diagonal_A():
    """A balanced and symmetric [0/90]s laminate has A16 = A26 = 0
    (no shear-extension coupling in the membrane stiffness)."""
    ply = OrthotropicLamina(E1=181e9, E2=10.3e9, G12=7.17e9, nu12=0.28)
    sec = LayeredShellSection.from_layers_centered([
        (ply, 0.5e-3, 0.0),
        (ply, 0.5e-3, 90.0),
        (ply, 0.5e-3, 90.0),
        (ply, 0.5e-3, 0.0),
    ])
    A = sec.D_membrane()
    assert abs(A[0, 2]) < 1e-6 * np.max(np.abs(A))
    assert abs(A[1, 2]) < 1e-6 * np.max(np.abs(A))


def test_45_deg_laminate_has_off_diagonal_A():
    """A laminate with non-axis-aligned plies has A16, A26 != 0."""
    ply = OrthotropicLamina(E1=181e9, E2=10.3e9, G12=7.17e9, nu12=0.28)
    sec = LayeredShellSection.from_layers_centered([
        (ply, 0.5e-3, 45.0),
        (ply, 0.5e-3, 45.0),
    ])
    A = sec.D_membrane()
    assert abs(A[0, 2]) > 1e5     # tangible shear-extension coupling
    assert abs(A[1, 2]) > 1e5


def test_orthotropic_layer_runs_in_shellmitc4(tmp_path):
    """End-to-end: a single orthotropic ply driven through ShellMITC4
    gives a sensible deflection for a tip-loaded plate."""
    ply = OrthotropicLamina(E1=181e9, E2=10.3e9, G12=7.17e9, nu12=0.28)
    sec = LayeredShellSection.from_layers_centered([
        (ply, 1.0e-3, 0.0),
    ])
    mat = ElasticIsotropic(1, E=181e9, nu=0.3)  # dummy for mass / model
    m = Model(ndm=3, ndf=6); m.add_material(mat)
    L = 1.0
    N = 8
    for j in range(N + 1):
        for i in range(N + 1):
            m.add_node(j * (N + 1) + i + 1, i * L / N, j * L / N, 0.0)
    etag = 1
    for j in range(N):
        for i in range(N):
            n1 = j * (N + 1) + i + 1; n2 = n1 + 1
            n3 = n2 + (N + 1); n4 = n1 + (N + 1)
            m.add_element(ShellMITC4(
                etag, (n1, n2, n3, n4), mat, section=sec,
            ))
            etag += 1
    # Clamp left edge (i = 0)
    for j in range(N + 1):
        m.fix(j * (N + 1) + 1, [1, 1, 1, 1, 1, 1])
    # Tip load on right edge
    for j in range(N + 1):
        tag = j * (N + 1) + N + 1
        m.add_nodal_load(tag, [0, 0, -1.0, 0, 0, 0])
    LinearStaticAnalysis(m).run()
    tip_disp = m.node((N + 1) * (N + 1)).disp[2]
    # Sanity: tip deflects downward; not exploding
    assert tip_disp < 0.0
    assert abs(tip_disp) < 1.0       # finite, positive-stiffness response


# ====================================================== curved-shell mesh

def test_cylindrical_mesh_full_topology():
    nodes, quads = cylindrical_shell_mesh(
        radius=1.0, length=2.0, n_circ=8, n_long=4,
    )
    # 8 circ * 5 long = 40 nodes for a full cylinder
    assert nodes.shape == (40, 3)
    assert quads.shape == (32, 4)
    # Every node should be at radius 1.0 from the z axis
    radial = np.hypot(nodes[:, 0], nodes[:, 1])
    np.testing.assert_allclose(radial, 1.0, rtol=1e-10)


def test_cylindrical_mesh_partial_sweep():
    """A half-cylinder (theta from 0 to pi) gives one extra row of
    nodes (open at the seam)."""
    nodes, quads = cylindrical_shell_mesh(
        radius=2.0, length=1.0,
        n_circ=6, n_long=3,
        theta_start=0.0, theta_end=math.pi,
    )
    # (n_circ + 1) * (n_long + 1) = 7 * 4 = 28 nodes
    assert nodes.shape == (28, 3)
    # n_circ * n_long = 18 quads
    assert quads.shape == (18, 4)


def test_cylindrical_mesh_axis_y():
    nodes, _ = cylindrical_shell_mesh(
        radius=0.5, length=1.0, n_circ=4, n_long=2, axis="y",
    )
    # All nodes should be at radius 0.5 from the y axis -> (x, z) distance
    radial = np.hypot(nodes[:, 0], nodes[:, 2])
    np.testing.assert_allclose(radial, 0.5, rtol=1e-10)
    # And y should span [0, 1]
    assert nodes[:, 1].min() == pytest.approx(0.0, abs=1e-12)
    assert nodes[:, 1].max() == pytest.approx(1.0, rel=1e-12)


def test_spherical_cap_mesh_apex_at_radius():
    """The first node (apex) is at (0, 0, R)."""
    nodes, _ = spherical_cap_mesh(
        radius=1.0, half_angle_deg=30.0, n_radial=4, n_circ=8,
    )
    np.testing.assert_allclose(nodes[0], [0.0, 0.0, 1.0], atol=1e-12)
    # Every node should lie on the sphere of radius 1.
    r = np.linalg.norm(nodes, axis=1)
    np.testing.assert_allclose(r, 1.0, rtol=1e-10)


def test_spherical_cap_mesh_validates_inputs():
    with pytest.raises(ValueError, match="half_angle_deg"):
        spherical_cap_mesh(radius=1.0, half_angle_deg=0.0,
                            n_radial=4, n_circ=4)
    with pytest.raises(ValueError, match="n_radial"):
        spherical_cap_mesh(radius=1.0, half_angle_deg=30.0,
                            n_radial=0, n_circ=4)
