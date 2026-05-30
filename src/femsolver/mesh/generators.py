"""Structured mesh generators.

Returns small standalone dataclasses ``(nodes, connectivity)`` that
the caller can scatter into a :class:`~femsolver.Model` using the
appropriate element class. We deliberately avoid creating the
:class:`~femsolver.Model` directly so the user can pick which
element type to instantiate (e.g., :class:`Quad4` for plane stress,
:class:`ShellMITC4` for plates).

Available generators:

* :func:`rectangle_quad4` -- 2D rectangle, nx by ny Quad4-style
  connectivity (counter-clockwise).
* :func:`disk_quad4`      -- 2D circular disk via pizza-slice
  partitioning (radial + tangential mesh).
* :func:`ring_quad4`      -- 2D annulus from inner to outer radius.
* :func:`solid_hex8`      -- 3D structured Hex8 brick.
* :func:`shell_curved_cylinder`  -- 3D cylindrical shell.

Each returns a :class:`StructuredMesh` with::

    nodes      : (n_nodes, dim) coords
    connectivity : (n_elem, n_nodes_per_elem) tags (1-based)
    node_tags  : (n_nodes,) 1-based tags
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class StructuredMesh:
    """A structured mesh ready to be scattered into a :class:`Model`.

    Attributes
    ----------
    nodes : (n_nodes, ndm) np.ndarray
        Node coordinates.
    connectivity : (n_elem, n_nodes_per_elem) np.ndarray
        1-based node tags per element.
    node_tags : (n_nodes,) np.ndarray
        1-based node tags (just ``range(1, n_nodes+1)``).
    ndm : int
        Spatial dimension.
    nodes_per_element : int
    boundary_nodes : dict[str, np.ndarray]
        Named groups of boundary node tags
        (e.g., ``"left"``, ``"right"``, ``"top"``, ``"bottom"``).
    """

    nodes: np.ndarray
    connectivity: np.ndarray
    node_tags: np.ndarray
    ndm: int
    nodes_per_element: int
    boundary_nodes: dict


# ============================================================ rectangle Q4

def rectangle_quad4(
    *, W: float, H: float,
    nx: int, ny: int,
    x0: float = 0.0, y0: float = 0.0,
) -> StructuredMesh:
    """Structured ``nx`` by ``ny`` Quad4 mesh of a rectangle.

    Counter-clockwise node ordering per element::

        n4 ─── n3
        │       │
        n1 ─── n2

    Parameters
    ----------
    W, H : float
        Rectangle width (x) and height (y).
    nx, ny : int
        Number of elements in each direction.
    x0, y0 : float, default 0
        Lower-left corner.
    """
    if nx < 1 or ny < 1:
        raise ValueError("nx, ny must be >= 1")
    if W <= 0.0 or H <= 0.0:
        raise ValueError("W, H must be > 0")

    nodes = []
    grid = {}
    tag = 1
    for j in range(ny + 1):
        for i in range(nx + 1):
            x = x0 + i * W / nx
            y = y0 + j * H / ny
            nodes.append([x, y])
            grid[(i, j)] = tag
            tag += 1
    nodes = np.array(nodes, dtype=float)
    n_nodes = nodes.shape[0]
    conn = np.empty((nx * ny, 4), dtype=int)
    e = 0
    for j in range(ny):
        for i in range(nx):
            conn[e] = [
                grid[(i, j)], grid[(i + 1, j)],
                grid[(i + 1, j + 1)], grid[(i, j + 1)],
            ]
            e += 1
    # Boundary node groups
    boundary = {
        "left":   np.array([grid[(0, j)]      for j in range(ny + 1)]),
        "right":  np.array([grid[(nx, j)]     for j in range(ny + 1)]),
        "bottom": np.array([grid[(i, 0)]      for i in range(nx + 1)]),
        "top":    np.array([grid[(i, ny)]     for i in range(nx + 1)]),
    }
    return StructuredMesh(
        nodes=nodes, connectivity=conn,
        node_tags=np.arange(1, n_nodes + 1),
        ndm=2, nodes_per_element=4,
        boundary_nodes=boundary,
    )


# ============================================================ disk + ring

def disk_quad4(
    *, R: float,
    n_radial: int, n_tangent: int,
) -> StructuredMesh:
    """Mesh a unit-radius (or radius ``R``) disk with quadrilaterals.

    A polar-grid mesh; the central node is a single shared apex with
    triangular "fans" to its neighbours -- to keep Quad4 throughout
    we offset the inner-most ring slightly and pack the inner ring
    by ``n_tangent`` triangles internally. For simplicity here we
    return Quad4 connectivity by including a small INNER hole (radius
    ``R/n_radial``) and connecting the inner ring around it.

    Parameters
    ----------
    R : float
        Outer radius (m).
    n_radial : int
        Number of radial rings.
    n_tangent : int
        Number of elements around the circumference.
    """
    if R <= 0.0 or n_radial < 1 or n_tangent < 3:
        raise ValueError("R > 0, n_radial >= 1, n_tangent >= 3 required")
    R_inner = R / (n_radial + 1)        # small central hole
    return ring_quad4(
        R_inner=R_inner, R_outer=R,
        n_radial=n_radial, n_tangent=n_tangent,
    )


def ring_quad4(
    *, R_inner: float, R_outer: float,
    n_radial: int, n_tangent: int,
) -> StructuredMesh:
    """Mesh a 2D annulus with Quad4 elements."""
    if R_inner <= 0.0 or R_outer <= R_inner:
        raise ValueError("require 0 < R_inner < R_outer")
    if n_radial < 1 or n_tangent < 3:
        raise ValueError("n_radial >= 1, n_tangent >= 3 required")
    nodes = []
    grid = {}
    tag = 1
    for j in range(n_radial + 1):
        r = R_inner + j * (R_outer - R_inner) / n_radial
        for i in range(n_tangent):
            theta = 2.0 * math.pi * i / n_tangent
            nodes.append([r * math.cos(theta), r * math.sin(theta)])
            grid[(i, j)] = tag
            tag += 1
    nodes = np.array(nodes, dtype=float)
    conn = np.empty((n_radial * n_tangent, 4), dtype=int)
    e = 0
    for j in range(n_radial):
        for i in range(n_tangent):
            i_next = (i + 1) % n_tangent       # wrap
            conn[e] = [
                grid[(i, j)], grid[(i_next, j)],
                grid[(i_next, j + 1)], grid[(i, j + 1)],
            ]
            e += 1
    boundary = {
        "inner": np.array([grid[(i, 0)]            for i in range(n_tangent)]),
        "outer": np.array([grid[(i, n_radial)]     for i in range(n_tangent)]),
    }
    return StructuredMesh(
        nodes=nodes, connectivity=conn,
        node_tags=np.arange(1, nodes.shape[0] + 1),
        ndm=2, nodes_per_element=4,
        boundary_nodes=boundary,
    )


# ============================================================ solid Hex8

def solid_hex8(
    *, Lx: float, Ly: float, Lz: float,
    nx: int, ny: int, nz: int,
    x0: float = 0.0, y0: float = 0.0, z0: float = 0.0,
) -> StructuredMesh:
    """Structured ``nx · ny · nz`` Hex8 mesh of a rectangular block."""
    if nx < 1 or ny < 1 or nz < 1:
        raise ValueError("nx, ny, nz must be >= 1")
    if Lx <= 0.0 or Ly <= 0.0 or Lz <= 0.0:
        raise ValueError("Lx, Ly, Lz must be > 0")
    nodes = []
    grid = {}
    tag = 1
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                nodes.append([
                    x0 + i * Lx / nx,
                    y0 + j * Ly / ny,
                    z0 + k * Lz / nz,
                ])
                grid[(i, j, k)] = tag
                tag += 1
    nodes = np.array(nodes, dtype=float)
    conn = np.empty((nx * ny * nz, 8), dtype=int)
    e = 0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                conn[e] = [
                    grid[(i,     j,     k    )],
                    grid[(i + 1, j,     k    )],
                    grid[(i + 1, j + 1, k    )],
                    grid[(i,     j + 1, k    )],
                    grid[(i,     j,     k + 1)],
                    grid[(i + 1, j,     k + 1)],
                    grid[(i + 1, j + 1, k + 1)],
                    grid[(i,     j + 1, k + 1)],
                ]
                e += 1
    # 6 boundary face groups
    boundary = {
        "x_minus": np.array([grid[(0, j, k)]
                              for k in range(nz + 1)
                              for j in range(ny + 1)]),
        "x_plus":  np.array([grid[(nx, j, k)]
                              for k in range(nz + 1)
                              for j in range(ny + 1)]),
        "y_minus": np.array([grid[(i, 0, k)]
                              for k in range(nz + 1)
                              for i in range(nx + 1)]),
        "y_plus":  np.array([grid[(i, ny, k)]
                              for k in range(nz + 1)
                              for i in range(nx + 1)]),
        "z_minus": np.array([grid[(i, j, 0)]
                              for j in range(ny + 1)
                              for i in range(nx + 1)]),
        "z_plus":  np.array([grid[(i, j, nz)]
                              for j in range(ny + 1)
                              for i in range(nx + 1)]),
    }
    return StructuredMesh(
        nodes=nodes, connectivity=conn,
        node_tags=np.arange(1, nodes.shape[0] + 1),
        ndm=3, nodes_per_element=8,
        boundary_nodes=boundary,
    )


# ============================================================ cylindrical shell

def shell_curved_cylinder(
    *, R: float, H: float,
    n_tangent: int, n_axial: int,
) -> StructuredMesh:
    """Cylindrical shell mesh as a 3D quad surface.

    Returns nodes in 3D arrayed around the cylinder axis (z), with
    Quad4-style connectivity suitable for :class:`ShellMITC4`.
    """
    if R <= 0.0 or H <= 0.0 or n_tangent < 3 or n_axial < 1:
        raise ValueError(
            "require R > 0, H > 0, n_tangent >= 3, n_axial >= 1"
        )
    nodes = []
    grid = {}
    tag = 1
    for k in range(n_axial + 1):
        for i in range(n_tangent):
            theta = 2.0 * math.pi * i / n_tangent
            nodes.append([
                R * math.cos(theta), R * math.sin(theta),
                k * H / n_axial,
            ])
            grid[(i, k)] = tag
            tag += 1
    nodes = np.array(nodes, dtype=float)
    conn = np.empty((n_axial * n_tangent, 4), dtype=int)
    e = 0
    for k in range(n_axial):
        for i in range(n_tangent):
            i_next = (i + 1) % n_tangent
            conn[e] = [
                grid[(i, k)], grid[(i_next, k)],
                grid[(i_next, k + 1)], grid[(i, k + 1)],
            ]
            e += 1
    boundary = {
        "bottom": np.array([grid[(i, 0)] for i in range(n_tangent)]),
        "top":    np.array([grid[(i, n_axial)] for i in range(n_tangent)]),
    }
    return StructuredMesh(
        nodes=nodes, connectivity=conn,
        node_tags=np.arange(1, nodes.shape[0] + 1),
        ndm=3, nodes_per_element=4,
        boundary_nodes=boundary,
    )
