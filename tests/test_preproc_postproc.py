"""Phase 47 tests -- pre/post-processing utilities (Theme L)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from femsolver.mesh import (
    MeshQualityReport,
    Quad4Quality,
    NodalStressField,
    disk_quad4,
    mesh_quality_report,
    principal_stresses_2d,
    principal_stresses_3d,
    quad4_quality,
    rectangle_quad4,
    ring_quad4,
    shell_curved_cylinder,
    solid_hex8,
    von_mises_2d,
    von_mises_3d,
)


# ============================================================ generators

class TestRectangleQuad4:
    def test_node_and_element_counts(self):
        m = rectangle_quad4(W=10.0, H=4.0, nx=5, ny=2)
        assert m.nodes.shape == (18, 2)      # (nx+1)*(ny+1) = 6*3
        assert m.connectivity.shape == (10, 4)  # nx*ny

    def test_boundary_groups(self):
        m = rectangle_quad4(W=1.0, H=1.0, nx=3, ny=3)
        # 4 nodes on each face for a 3x3 grid
        assert len(m.boundary_nodes["left"]) == 4
        assert len(m.boundary_nodes["right"]) == 4
        assert len(m.boundary_nodes["top"]) == 4
        assert len(m.boundary_nodes["bottom"]) == 4

    def test_corner_coordinates(self):
        m = rectangle_quad4(W=2.0, H=1.0, nx=2, ny=1)
        # Node 1 should be at (0, 0); top-right at (2, 1)
        assert tuple(m.nodes[0]) == (0.0, 0.0)
        assert tuple(m.nodes[-1]) == (2.0, 1.0)

    def test_origin_offset(self):
        m = rectangle_quad4(W=2.0, H=1.0, nx=2, ny=1, x0=5.0, y0=3.0)
        assert tuple(m.nodes[0]) == (5.0, 3.0)


class TestRingQuad4:
    def test_node_count_no_pinch(self):
        # n_radial layers => (n_radial + 1) circles of n_tangent nodes
        r = ring_quad4(R_inner=1.0, R_outer=2.0, n_radial=4, n_tangent=8)
        assert r.nodes.shape == (40, 2)
        assert r.connectivity.shape == (32, 4)

    def test_inner_outer_boundaries(self):
        r = ring_quad4(R_inner=1.0, R_outer=2.0, n_radial=3, n_tangent=6)
        for tag in r.boundary_nodes["inner"]:
            x, y = r.nodes[tag - 1]
            assert math.hypot(x, y) == pytest.approx(1.0, abs=1e-9)
        for tag in r.boundary_nodes["outer"]:
            x, y = r.nodes[tag - 1]
            assert math.hypot(x, y) == pytest.approx(2.0, abs=1e-9)


class TestSolidHex8:
    def test_counts(self):
        h = solid_hex8(Lx=3.0, Ly=2.0, Lz=1.0, nx=3, ny=2, nz=1)
        assert h.nodes.shape == (24, 3)     # 4*3*2
        assert h.connectivity.shape == (6, 8)

    def test_six_face_groups(self):
        h = solid_hex8(Lx=1.0, Ly=1.0, Lz=1.0, nx=2, ny=2, nz=2)
        for face in ("x_minus", "x_plus",
                     "y_minus", "y_plus",
                     "z_minus", "z_plus"):
            assert face in h.boundary_nodes
            assert len(h.boundary_nodes[face]) == 9      # (2+1)x(2+1)


class TestShellCurvedCylinder:
    def test_node_count(self):
        s = shell_curved_cylinder(R=2.0, H=4.0, n_tangent=12, n_axial=4)
        assert s.nodes.shape == (60, 3)         # 12 * 5  (closed in theta)
        assert s.connectivity.shape == (48, 4)  # 12 * 4

    def test_nodes_on_cylinder(self):
        s = shell_curved_cylinder(R=2.0, H=4.0, n_tangent=8, n_axial=2)
        for (x, y, z) in s.nodes:
            assert math.hypot(x, y) == pytest.approx(2.0, abs=1e-9)
            assert -1e-9 <= z <= 4.0 + 1e-9


# ============================================================ quality

class TestQuad4Quality:
    def test_perfect_square(self):
        coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
        q = quad4_quality(coords)
        assert q.jacobian_ratio == pytest.approx(1.0)
        assert q.aspect_ratio == pytest.approx(1.0)
        assert q.skewness == pytest.approx(0.0, abs=1e-9)

    def test_rectangle_aspect(self):
        coords = np.array([[0, 0], [4, 0], [4, 1], [0, 1]], float)
        q = quad4_quality(coords)
        assert q.aspect_ratio == pytest.approx(4.0)

    def test_skewed_parallelogram(self):
        # 45° shear -> corner angle = 45°, |45-90|/90 = 0.5
        coords = np.array([[0, 0], [1, 0], [2, 1], [1, 1]], float)
        q = quad4_quality(coords)
        assert q.skewness == pytest.approx(0.5, rel=0.02)

    def test_validates_shape(self):
        with pytest.raises(ValueError):
            quad4_quality(np.zeros((3, 2)))


class TestMeshQualityReport:
    def test_uniform_rectangle_is_perfect(self):
        m = rectangle_quad4(W=2.0, H=2.0, nx=4, ny=4)
        r = mesh_quality_report(m)
        assert r.jacobian_ratio_min == pytest.approx(1.0, abs=1e-12)
        assert r.aspect_ratio_max == pytest.approx(1.0, abs=1e-12)
        assert r.skewness_max == pytest.approx(0.0, abs=1e-12)
        assert r.n_elements == 16

    def test_disk_mesh_quality_acceptable(self):
        # Polar disk has some distortion at the centre patch
        d = disk_quad4(R=1.0, n_radial=4, n_tangent=12)
        r = mesh_quality_report(d)
        # No degenerate elements
        assert r.jacobian_ratio_min > 0.1
        assert math.isfinite(r.aspect_ratio_max)


# ============================================================ stress recovery

class TestPrincipal2D:
    def test_pure_uniaxial(self):
        s1, s2, theta = principal_stresses_2d([100e6, 0, 0])
        assert s1 == pytest.approx(100e6)
        assert s2 == pytest.approx(0.0)
        assert theta == pytest.approx(0.0, abs=1e-9)

    def test_pure_shear(self):
        s1, s2, theta = principal_stresses_2d([0, 0, 50e6])
        assert s1 == pytest.approx(50e6)
        assert s2 == pytest.approx(-50e6)
        # 45 degrees
        assert theta == pytest.approx(math.pi / 4, abs=1e-9)

    def test_mohr_circle_check(self):
        # State: sxx=100, syy=50, txy=25 -> center=75, radius=sqrt(25^2+25^2)=35.36
        s1, s2, _ = principal_stresses_2d([100e6, 50e6, 25e6])
        radius = math.sqrt(25e6 ** 2 + 25e6 ** 2)
        assert s1 == pytest.approx(75e6 + radius, rel=1e-9)
        assert s2 == pytest.approx(75e6 - radius, rel=1e-9)


class TestVonMises:
    def test_2d_uniaxial(self):
        assert von_mises_2d([100e6, 0, 0]) == pytest.approx(100e6)

    def test_2d_pure_shear(self):
        # sigma_eq = sqrt(3) * tau
        v = von_mises_2d([0, 0, 50e6])
        assert v == pytest.approx(math.sqrt(3) * 50e6, rel=1e-9)

    def test_2d_hydrostatic_plane_stress(self):
        # Plane-stress hydro: sigma_xx = sigma_yy = sigma, tau = 0
        # vM = sqrt(s^2 + s^2 - s*s) = s
        assert von_mises_2d([100e6, 100e6, 0]) == pytest.approx(100e6, rel=1e-9)

    def test_3d_pure_shear(self):
        # tau_xy only: vM = sqrt(3) * tau
        v = von_mises_3d([0, 0, 0, 50e6, 0, 0])
        assert v == pytest.approx(math.sqrt(3) * 50e6, rel=1e-9)

    def test_3d_hydrostatic_zero(self):
        # Pure hydrostatic stress -> vM = 0
        v = von_mises_3d([100e6, 100e6, 100e6, 0, 0, 0])
        assert v == pytest.approx(0.0, abs=1e-3)


class TestPrincipal3D:
    def test_diagonal(self):
        ps = principal_stresses_3d([3, 1, 2, 0, 0, 0])
        np.testing.assert_allclose(ps, [3, 2, 1], atol=1e-12)

    def test_validates_size(self):
        with pytest.raises(ValueError, match="6-vector"):
            principal_stresses_3d([1, 2, 3])


# ============================================================ plotting smoke

class TestPlottingHelpers:
    """Just smoke-test that the plotting helpers can be called with a
    valid mesh and produce a matplotlib Axes object (no display)."""

    def test_plot_undeformed_smoke(self):
        try:
            import matplotlib                        # noqa: F401
            matplotlib.use("Agg", force=True)
            from femsolver.postproc import plot_undeformed
        except ImportError:
            pytest.skip("matplotlib not installed")
        m = rectangle_quad4(W=2.0, H=1.0, nx=4, ny=2)
        ax = plot_undeformed(m.nodes, m.connectivity)
        assert ax is not None

    def test_plot_contour_smoke(self):
        try:
            import matplotlib
            matplotlib.use("Agg", force=True)
            from femsolver.postproc import plot_contour
        except ImportError:
            pytest.skip("matplotlib not installed")
        m = rectangle_quad4(W=2.0, H=1.0, nx=4, ny=2)
        # Linear field
        f = m.nodes[:, 0] * 100.0
        ax = plot_contour(m.nodes, m.connectivity, f,
                          cbar_label="sigma_xx (Pa)")
        assert ax is not None

    def test_plot_deformed_smoke(self):
        try:
            import matplotlib
            matplotlib.use("Agg", force=True)
            from femsolver.postproc import plot_deformed
        except ImportError:
            pytest.skip("matplotlib not installed")
        m = rectangle_quad4(W=2.0, H=1.0, nx=4, ny=2)
        u = np.zeros_like(m.nodes)
        u[:, 0] = 0.01 * m.nodes[:, 0]
        ax = plot_deformed(m.nodes, m.connectivity, u, scale=10.0)
        assert ax is not None

    def test_plot_time_history_smoke(self):
        try:
            import matplotlib
            matplotlib.use("Agg", force=True)
            from femsolver.postproc import plot_time_history
        except ImportError:
            pytest.skip("matplotlib not installed")
        t = np.linspace(0, 5, 50)
        y = np.sin(2 * np.pi * t)
        ax = plot_time_history(t, y, ylabel="u")
        assert ax is not None
        # dict version
        ax = plot_time_history(t, {"top": y, "base": -0.5 * y})
        assert ax is not None
