"""Phase 48.2 tests -- PyVista 3D post-processing helpers."""
from __future__ import annotations

import os

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")
# Force off-screen so this works on CI / headless boxes
pv.OFF_SCREEN = True

from femsolver.mesh import solid_hex8                # noqa: E402
from femsolver.postproc.plot_3d import (              # noqa: E402
    _build_grid,
    _hex_connectivity_to_vtk_cells,
    plot_deformed_3d,
    plot_mode_shape_3d,
    plot_scalar_field_3d,
    plot_undeformed_3d,
)


@pytest.fixture
def hex_mesh():
    return solid_hex8(Lx=2.0, Ly=1.0, Lz=1.0, nx=2, ny=1, nz=1)


# ============================================================ low-level

class TestHexCellConversion:
    def test_cells_array_format(self):
        # 2 hex elements with sequential node tags
        conn = np.array([
            [1, 2, 3, 4, 5, 6, 7, 8],
            [2, 9, 10, 3, 6, 11, 12, 7],
        ])
        cells, types = _hex_connectivity_to_vtk_cells(conn)
        # 2 elems x 9 entries each
        assert cells.size == 18
        # Leading count is 8 for each cell
        assert cells[0] == 8 and cells[9] == 8
        # Element 0 corner 0 = node 1 -> 0-based 0
        assert cells[1] == 0
        # All cell types are 12 (VTK_HEXAHEDRON)
        assert np.all(types == 12)

    def test_rejects_non_hex(self):
        with pytest.raises(ValueError, match="8 cols"):
            _hex_connectivity_to_vtk_cells(np.zeros((3, 4), dtype=int))


class TestBuildGrid:
    def test_hex_grid_node_count(self, hex_mesh):
        grid = _build_grid(hex_mesh.nodes, hex_mesh.connectivity)
        assert grid.n_points == hex_mesh.nodes.shape[0]
        assert grid.n_cells == hex_mesh.connectivity.shape[0]

    def test_quad_grid_from_2d_nodes(self):
        # 2x1 quad mesh with 6 nodes
        nodes = np.array([
            [0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
            [0.0, 1.0], [1.0, 1.0], [2.0, 1.0],
        ])
        conn = np.array([
            [1, 2, 5, 4],
            [2, 3, 6, 5],
        ])
        grid = _build_grid(nodes, conn)
        assert grid.n_points == 6
        assert grid.n_cells == 2

    def test_rejects_bad_width(self):
        with pytest.raises(ValueError, match="unsupported"):
            _build_grid(np.zeros((4, 3)), np.zeros((1, 5), dtype=int))


# ============================================================ smoke

class TestPlotSmoke:
    """Off-screen smoke tests: render to a PNG and check the file exists
    and is non-empty."""

    def test_undeformed_smoke(self, hex_mesh, tmp_path):
        out = tmp_path / "undeformed.png"
        plot_undeformed_3d(
            hex_mesh.nodes, hex_mesh.connectivity,
            off_screen=True, screenshot=str(out),
            title="Hex8 mesh",
        )
        assert out.exists() and out.stat().st_size > 0

    def test_deformed_smoke(self, hex_mesh, tmp_path):
        # Linear axial stretch displacement
        u = np.zeros_like(hex_mesh.nodes)
        u[:, 0] = 0.01 * hex_mesh.nodes[:, 0]
        out = tmp_path / "deformed.png"
        plot_deformed_3d(
            hex_mesh.nodes, hex_mesh.connectivity, u,
            scale=10.0,
            off_screen=True, screenshot=str(out),
        )
        assert out.exists() and out.stat().st_size > 0

    def test_scalar_field_smoke(self, hex_mesh, tmp_path):
        # Linear pressure-like field
        f = hex_mesh.nodes[:, 2] * 1e6   # 0 at z=0, max at z=1
        out = tmp_path / "scalar.png"
        plot_scalar_field_3d(
            hex_mesh.nodes, hex_mesh.connectivity, f,
            field_name="P (Pa)",
            off_screen=True, screenshot=str(out),
        )
        assert out.exists() and out.stat().st_size > 0

    def test_scalar_field_size_check(self, hex_mesh):
        with pytest.raises(ValueError, match="entries"):
            plot_scalar_field_3d(
                hex_mesh.nodes, hex_mesh.connectivity,
                np.zeros(7),
                off_screen=True,
            )

    def test_mode_shape_smoke(self, hex_mesh, tmp_path):
        # Bending-like mode: w ~ x (linear cantilever)
        n_nodes = hex_mesh.nodes.shape[0]
        ndf = 3
        vec = np.zeros(n_nodes * ndf)
        # Set z-component of each node proportional to x
        for i in range(n_nodes):
            vec[i * ndf + 2] = 0.01 * hex_mesh.nodes[i, 0]
        out = tmp_path / "mode.png"
        plot_mode_shape_3d(
            hex_mesh.nodes, hex_mesh.connectivity, vec,
            ndf=ndf, off_screen=True, screenshot=str(out),
        )
        assert out.exists() and out.stat().st_size > 0
