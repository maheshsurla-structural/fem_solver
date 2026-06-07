"""Matplotlib quick-plot helpers for FE post-processing.

These are *convenience* wrappers around matplotlib for the most common
FE post-processing tasks:

* :func:`plot_undeformed`  -- mesh outline.
* :func:`plot_deformed`    -- deformed shape, optionally on top of
  the undeformed mesh.
* :func:`plot_contour`     -- nodal scalar field rendered as a
  filled contour over the (possibly deformed) mesh.
* :func:`plot_mode_shape`  -- single mode of a normal-modes result.
* :func:`plot_time_history`-- 1-D time-history line plot.

Matplotlib is imported lazily so that ``femsolver`` remains usable
in environments without it. If the helpers are called without
matplotlib installed, a clear ``ImportError`` is raised.

Coordinates are assumed 2-D (x, y); 3-D meshes are projected onto
their xy plane for plotting.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


def _ensure_matplotlib():
    """Lazy import of matplotlib.pyplot."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:                                   # pragma: no cover
        raise ImportError(
            "matplotlib is required for femsolver.results.plot. "
            "Install with `pip install matplotlib`."
        ) from e
    return plt


# ============================================================ mesh

def plot_undeformed(
    nodes: np.ndarray,
    connectivity: np.ndarray,
    *,
    ax=None,
    edge_color: str = "0.4",
    face_color: str = "none",
    linewidth: float = 0.7,
    title: str | None = None,
):
    """Plot the undeformed mesh outline (Q4 / Tri3 / Hex8 face)."""
    plt = _ensure_matplotlib()
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    patches = []
    for conn in connectivity:
        coords = nodes[conn - 1, : 2]
        patches.append(Polygon(coords, closed=True))
    pc = PatchCollection(
        patches, edgecolor=edge_color, facecolor=face_color,
        linewidths=linewidth,
    )
    ax.add_collection(pc)
    ax.set_aspect("equal")
    ax.autoscale_view()
    if title:
        ax.set_title(title)
    return ax


def plot_deformed(
    nodes: np.ndarray,
    connectivity: np.ndarray,
    displacements: np.ndarray,
    *,
    scale: float = 1.0,
    ax=None,
    show_undeformed: bool = True,
    deformed_color: str = "C0",
    undeformed_color: str = "0.6",
    linewidth: float = 0.8,
    title: str | None = None,
):
    """Plot deformed shape, optionally on top of the undeformed mesh.

    Parameters
    ----------
    nodes : (n_nodes, ndm) array
    connectivity : (n_elem, k) array of 1-based node tags
    displacements : (n_nodes, ndm) array
    scale : float
        Amplification factor for displacements; choose so the
        deformation is visible at the plot scale.
    """
    plt = _ensure_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    if show_undeformed:
        plot_undeformed(
            nodes, connectivity, ax=ax,
            edge_color=undeformed_color, linewidth=linewidth * 0.6,
        )
    deformed = nodes[:, :2] + scale * displacements[:, :2]
    plot_undeformed(
        deformed, connectivity, ax=ax,
        edge_color=deformed_color, linewidth=linewidth,
    )
    if title is None:
        title = f"Deformed shape (scale = {scale:g})"
    ax.set_title(title)
    return ax


# ============================================================ contour

def plot_contour(
    nodes: np.ndarray,
    connectivity: np.ndarray,
    scalar_field: np.ndarray,
    *,
    ax=None,
    cmap: str = "viridis",
    n_levels: int = 20,
    show_mesh: bool = True,
    title: str | None = None,
    cbar_label: str | None = None,
):
    """Filled-contour plot of a nodal scalar field over the mesh.

    Uses matplotlib ``tricontourf`` so it works on irregular meshes
    without requiring a structured grid. For Q4 meshes each quad is
    silently split into two triangles.

    Parameters
    ----------
    nodes : (n_nodes, ndm)
    connectivity : (n_elem, k)
    scalar_field : (n_nodes,) array of nodal values to contour.
    """
    plt = _ensure_matplotlib()
    import matplotlib.tri as mtri

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    # Convert Q4 connectivity to triangles
    triangles = []
    k = connectivity.shape[1]
    for conn in connectivity:
        if k == 3:
            triangles.append([conn[0] - 1, conn[1] - 1, conn[2] - 1])
        elif k == 4:
            # Split: (0, 1, 2) and (0, 2, 3)
            triangles.append([conn[0] - 1, conn[1] - 1, conn[2] - 1])
            triangles.append([conn[0] - 1, conn[2] - 1, conn[3] - 1])
        else:
            # Fan triangulation
            for i in range(1, k - 1):
                triangles.append([conn[0] - 1, conn[i] - 1, conn[i + 1] - 1])
    triangles = np.array(triangles)
    triang = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles)

    cs = ax.tricontourf(triang, scalar_field, levels=n_levels, cmap=cmap)
    if show_mesh:
        plot_undeformed(
            nodes, connectivity, ax=ax,
            edge_color="0.3", linewidth=0.4,
        )
    cbar = fig.colorbar(cs, ax=ax)
    if cbar_label:
        cbar.set_label(cbar_label)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    return ax


# ============================================================ mode shape

def plot_mode_shape(
    nodes: np.ndarray,
    connectivity: np.ndarray,
    mode_vec: np.ndarray,
    *,
    ndf_per_node: int = 2,
    scale: float | None = None,
    ax=None,
    title: str | None = None,
):
    """Plot a single normal-mode shape.

    Parameters
    ----------
    mode_vec : (n_eqn,) array
        Eigenvector in equation-numbering.
    ndf_per_node : int
        Translational DOFs per node (2 for 2D plane, 3 for 3D).
    scale : float or None
        Amplification factor. If ``None``, auto-scale so the maximum
        nodal displacement equals 10 % of the mesh diagonal.
    """
    plt = _ensure_matplotlib()
    n_nodes = nodes.shape[0]
    # Reshape mode vector to per-node displacements (skip rotations)
    n_eqn = mode_vec.size
    full_ndf = n_eqn // n_nodes
    disp = mode_vec.reshape(n_nodes, full_ndf)[:, :ndf_per_node]
    # Auto-scale
    if scale is None:
        diag = np.linalg.norm(nodes[:, :2].max(0) - nodes[:, :2].min(0))
        max_disp = float(np.abs(disp).max() + 1e-30)
        scale = 0.1 * diag / max_disp
    return plot_deformed(
        nodes, connectivity, disp,
        scale=scale, ax=ax, show_undeformed=True,
        title=title or f"Mode shape (scale = {scale:.2g})",
    )


# ============================================================ time history

def plot_time_history(
    t: Iterable[float],
    y: Iterable[float] | dict[str, Iterable[float]],
    *,
    ax=None,
    xlabel: str = "Time (s)",
    ylabel: str | None = None,
    title: str | None = None,
):
    """Plot one or several response time histories on a single axis.

    Parameters
    ----------
    t : iterable of floats
    y : iterable of floats *or* dict ``{label: series}``
        Multiple series labelled in the legend.
    """
    plt = _ensure_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    if isinstance(y, dict):
        for lbl, series in y.items():
            ax.plot(t, series, label=str(lbl))
        ax.legend(loc="best", frameon=False)
    else:
        ax.plot(t, y)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax
