"""PyVista-backed 3D post-processing helpers (Phase 48.2).

The 2D :mod:`femsolver.results.plot` module already gives clean
matplotlib-based views for 2D meshes and shells projected onto a
plane. For 3D solids (Hex8, Tet4) and curved shells, matplotlib is
clumsy -- PyVista (a VTK wrapper) is the right tool.

This module exposes::

    plot_undeformed_3d(nodes, hex_connectivity)
    plot_deformed_3d(nodes, hex_connectivity, displacements, scale=...)
    plot_scalar_field_3d(nodes, hex_connectivity, scalar_field)
    plot_mode_shape_3d(nodes, hex_connectivity, mode_vec, ndf=3)

PyVista is imported lazily so that ``femsolver`` remains usable on
environments without it. When PyVista IS installed but no display is
available (CI, headless servers), pass ``off_screen=True`` and the
helpers will use VTK's framebuffer and write to ``screenshot``.
"""
from __future__ import annotations

import numpy as np


def _ensure_pyvista():
    try:
        import pyvista as pv
    except ImportError as exc:                                  # pragma: no cover
        raise ImportError(
            "pyvista is required for femsolver.results.plot_3d. "
            "Install with `pip install pyvista`."
        ) from exc
    return pv


# ============================================================ mesh conversion

def _hex_connectivity_to_vtk_cells(
    connectivity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert an ``(n_elem, 8)`` 1-based hex connectivity array to
    VTK's flat cells array + a matching cell-type array.

    VTK uses cell-type 12 for VTK_HEXAHEDRON and the layout
    ``[8, n0, n1, ..., n7, 8, n0, ..., n7, ...]``.
    """
    import pyvista as pv     # local import; lazy
    n_elem, k = connectivity.shape
    if k != 8:
        raise ValueError(f"hex_connectivity must have 8 cols, got {k}")
    cells = np.empty(n_elem * 9, dtype=np.int64)
    for i, conn in enumerate(connectivity):
        cells[9 * i] = 8
        cells[9 * i + 1: 9 * i + 9] = conn - 1   # to 0-based
    cell_types = np.full(n_elem, 12, dtype=np.uint8)   # VTK_HEXAHEDRON
    return cells, cell_types


def _quad4_connectivity_to_vtk_cells(
    connectivity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert ``(n_elem, 4)`` 1-based quad connectivity to VTK cells."""
    n_elem, k = connectivity.shape
    if k != 4:
        raise ValueError(f"quad connectivity must have 4 cols, got {k}")
    cells = np.empty(n_elem * 5, dtype=np.int64)
    for i, conn in enumerate(connectivity):
        cells[5 * i] = 4
        cells[5 * i + 1: 5 * i + 5] = conn - 1
    cell_types = np.full(n_elem, 9, dtype=np.uint8)   # VTK_QUAD
    return cells, cell_types


def _build_grid(
    nodes: np.ndarray,
    connectivity: np.ndarray,
):
    """Build a PyVista UnstructuredGrid from nodes + connectivity.

    Connectivity column count selects Hex8 (8) or Quad4 (4).
    """
    pv = _ensure_pyvista()
    coords = np.asarray(nodes, dtype=np.float64)
    if coords.shape[1] == 2:
        coords = np.column_stack([coords, np.zeros(coords.shape[0])])
    k = connectivity.shape[1]
    if k == 8:
        cells, ct = _hex_connectivity_to_vtk_cells(connectivity)
    elif k == 4:
        cells, ct = _quad4_connectivity_to_vtk_cells(connectivity)
    else:
        raise ValueError(
            f"unsupported connectivity width {k}; expected 4 (quad) or 8 (hex)"
        )
    return pv.UnstructuredGrid(cells, ct, coords)


# ============================================================ undeformed

def plot_undeformed_3d(
    nodes: np.ndarray,
    connectivity: np.ndarray,
    *,
    show_edges: bool = True,
    color: str = "white",
    off_screen: bool = False,
    screenshot: str | None = None,
    title: str | None = None,
    return_plotter: bool = False,
):
    """Render the undeformed 3D mesh as a wireframe + face plot."""
    pv = _ensure_pyvista()
    grid = _build_grid(nodes, connectivity)
    p = pv.Plotter(off_screen=off_screen)
    p.add_mesh(grid, color=color, show_edges=show_edges)
    if title:
        p.add_text(title, font_size=10)
    if screenshot:
        p.show(screenshot=screenshot, auto_close=not return_plotter)
    elif not return_plotter:
        p.show()
    return p if return_plotter else None


# ============================================================ deformed

def plot_deformed_3d(
    nodes: np.ndarray,
    connectivity: np.ndarray,
    displacements: np.ndarray,
    *,
    scale: float = 1.0,
    show_undeformed: bool = True,
    deformed_color: str = "lightblue",
    undeformed_color: str = "lightgray",
    undeformed_opacity: float = 0.25,
    off_screen: bool = False,
    screenshot: str | None = None,
    title: str | None = None,
    return_plotter: bool = False,
):
    """Render the deformed shape; overlay the undeformed mesh if asked.

    ``displacements`` must have shape ``(n_nodes, 3)``. For 2D
    problems pad the third column with zeros.
    """
    pv = _ensure_pyvista()
    nodes = np.asarray(nodes, dtype=np.float64)
    u = np.asarray(displacements, dtype=np.float64)
    if u.shape[1] == 2:
        u = np.column_stack([u, np.zeros(u.shape[0])])
    deformed_nodes = nodes.copy()
    if deformed_nodes.shape[1] == 2:
        deformed_nodes = np.column_stack(
            [deformed_nodes, np.zeros(deformed_nodes.shape[0])]
        )
    deformed_nodes = deformed_nodes + scale * u
    deformed_grid = _build_grid(deformed_nodes, connectivity)
    p = pv.Plotter(off_screen=off_screen)
    if show_undeformed:
        undeformed_grid = _build_grid(nodes, connectivity)
        p.add_mesh(
            undeformed_grid,
            color=undeformed_color,
            opacity=undeformed_opacity,
            show_edges=True,
        )
    p.add_mesh(deformed_grid, color=deformed_color, show_edges=True)
    if title is None:
        title = f"Deformed (scale = {scale:g})"
    p.add_text(title, font_size=10)
    if screenshot:
        p.show(screenshot=screenshot, auto_close=not return_plotter)
    elif not return_plotter:
        p.show()
    return p if return_plotter else None


# ============================================================ contour

def plot_scalar_field_3d(
    nodes: np.ndarray,
    connectivity: np.ndarray,
    scalar_field: np.ndarray,
    *,
    field_name: str = "value",
    cmap: str = "viridis",
    n_levels: int = 12,
    off_screen: bool = False,
    screenshot: str | None = None,
    title: str | None = None,
    return_plotter: bool = False,
):
    """Render a nodal scalar field as a coloured 3D contour."""
    pv = _ensure_pyvista()
    grid = _build_grid(nodes, connectivity)
    scalar_field = np.asarray(scalar_field, dtype=np.float64).ravel()
    if scalar_field.size != grid.n_points:
        raise ValueError(
            f"scalar_field has {scalar_field.size} entries but the grid "
            f"has {grid.n_points} points."
        )
    grid[field_name] = scalar_field
    p = pv.Plotter(off_screen=off_screen)
    p.add_mesh(
        grid, scalars=field_name, cmap=cmap, n_colors=n_levels,
        show_edges=True, scalar_bar_args={"title": field_name},
    )
    if title:
        p.add_text(title, font_size=10)
    if screenshot:
        p.show(screenshot=screenshot, auto_close=not return_plotter)
    elif not return_plotter:
        p.show()
    return p if return_plotter else None


# ============================================================ mode shape

def plot_mode_shape_3d(
    nodes: np.ndarray,
    connectivity: np.ndarray,
    mode_vec: np.ndarray,
    *,
    ndf: int = 3,
    scale: float | None = None,
    off_screen: bool = False,
    screenshot: str | None = None,
    title: str | None = None,
    return_plotter: bool = False,
):
    """Render one mode shape from an eigenvector.

    Parameters
    ----------
    mode_vec : (n_eqn,) array
    ndf : int
        Total DOFs per node (e.g., 3 for 3D solid, 6 for shell). The
        first ``min(ndf, 3)`` components per node are used as the
        translational displacement.
    scale : float, optional
        Amplification factor. If ``None``, auto-scaled so the max
        nodal translation equals 10 % of the model's diagonal extent.
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    mode_vec = np.asarray(mode_vec, dtype=np.float64).ravel()
    n_nodes = nodes.shape[0]
    full_ndf = mode_vec.size // n_nodes
    if full_ndf < ndf:
        raise ValueError(
            f"mode_vec size ({mode_vec.size}) inconsistent with "
            f"n_nodes={n_nodes} and ndf={ndf}."
        )
    disp = mode_vec.reshape(n_nodes, full_ndf)[:, : min(ndf, 3)]
    if disp.shape[1] < 3:
        pad = np.zeros((n_nodes, 3 - disp.shape[1]))
        disp = np.concatenate([disp, pad], axis=1)
    if scale is None:
        diag = float(np.linalg.norm(
            nodes.max(axis=0) - nodes.min(axis=0)
        ))
        max_disp = float(np.max(np.abs(disp)) + 1e-30)
        scale = 0.1 * diag / max_disp
    return plot_deformed_3d(
        nodes, connectivity, disp, scale=scale,
        title=(title or f"Mode shape (scale = {scale:.2g})"),
        off_screen=off_screen, screenshot=screenshot,
        return_plotter=return_plotter,
    )
