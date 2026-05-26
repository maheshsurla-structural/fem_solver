"""Mesh helpers for curved shells using the flat-facet approximation.

For shells of moderate curvature, the standard practice in
commercial-FE software is to mesh the curved surface with flat
quadrilateral or triangular facets and rely on the constant-Jacobian
mapping inside each facet. ``ShellMITC4`` and ``ShellTri3`` already
support this through their internal local-frame construction (the
mid-surface plane is derived from the four / three node positions).

The helpers below generate the most common curved-shell topologies
and return ready-to-use node / element lists. They do not generate
the model -- they yield arrays that the caller can hand to
``Model.add_node`` / ``Model.add_element`` along with the chosen
shell element class and material / section.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def cylindrical_shell_mesh(
    *,
    radius: float,
    length: float,
    n_circ: int,
    n_long: int,
    theta_start: float = 0.0,
    theta_end: float = 2.0 * math.pi,
    axis: str = "z",
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a cylindrical-shell mesh with quadrilateral facets.

    Parameters
    ----------
    radius : float
    length : float
    n_circ : int
        Number of facets around the circumference.
    n_long : int
        Number of facets along the cylinder axis.
    theta_start, theta_end : float
        Angular extent (radians). Defaults to a full cylinder
        ``[0, 2 pi]``. A partial sweep (e.g. ``0 -> pi``) gives a
        half-cylinder.
    axis : ``"z"`` (default), ``"x"``, or ``"y"``
        Direction of the cylinder axis. Nodes are placed in the plane
        perpendicular to this axis.

    Returns
    -------
    nodes : (n, 3) array of node coordinates.
    quads : (m, 4) array of zero-indexed node tags forming each quad
        facet (CCW seen from the outside).
    """
    if n_circ < 1 or n_long < 1:
        raise ValueError("n_circ and n_long must be >= 1")
    if axis not in ("x", "y", "z"):
        raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis!r}")
    full = abs(theta_end - theta_start) >= 2.0 * math.pi - 1e-12
    n_circ_nodes = n_circ if full else n_circ + 1
    n_long_nodes = n_long + 1
    nodes = np.zeros((n_circ_nodes * n_long_nodes, 3))
    thetas = np.linspace(theta_start, theta_end, n_circ_nodes
                              + (1 if full else 0))[:n_circ_nodes]
    zs = np.linspace(0.0, length, n_long_nodes)
    axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
    # The two non-axis directions get the radial sweep
    other = [i for i in range(3) if i != axis_idx]
    for j, zj in enumerate(zs):
        for i, ti in enumerate(thetas):
            idx = j * n_circ_nodes + i
            nodes[idx, axis_idx] = zj
            nodes[idx, other[0]] = radius * math.cos(ti)
            nodes[idx, other[1]] = radius * math.sin(ti)
    quads = np.zeros((n_circ * n_long, 4), dtype=int)
    q = 0
    for j in range(n_long):
        for i in range(n_circ):
            n0 = j * n_circ_nodes + i
            n1 = j * n_circ_nodes + (i + 1) % n_circ_nodes
            n2 = (j + 1) * n_circ_nodes + (i + 1) % n_circ_nodes
            n3 = (j + 1) * n_circ_nodes + i
            quads[q] = [n0, n1, n2, n3]
            q += 1
    return nodes, quads


def spherical_cap_mesh(
    *,
    radius: float,
    half_angle_deg: float,
    n_radial: int,
    n_circ: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a spherical-cap mesh of quadrilaterals.

    The cap is centered on the +z axis with apex at (0, 0, radius);
    points sweep from the apex out to the rim at polar angle
    ``half_angle_deg`` from the axis.

    Parameters
    ----------
    radius : float
    half_angle_deg : float
        Half-angle (from apex to rim), in degrees. 90 gives a
        hemisphere; smaller angles give shallower caps.
    n_radial : int
        Number of facets along the radial (apex-to-rim) direction.
    n_circ : int
        Number of facets around the circumference.

    Returns
    -------
    nodes : (n, 3) array of node coordinates.
    quads : (m, 4) array of zero-indexed quad node tags.

    Notes
    -----
    The apex is a single point shared by all radial rings. The
    "quads" that touch the apex are degenerate (three coincident
    corner indices); shell elements at the apex effectively act as
    triangles. For higher-quality apex meshing, use a
    triangulation with the ``ShellTri3`` element instead.
    """
    if half_angle_deg <= 0.0 or half_angle_deg > 180.0:
        raise ValueError(
            f"half_angle_deg must lie in (0, 180], got {half_angle_deg}"
        )
    if n_radial < 1 or n_circ < 1:
        raise ValueError("n_radial and n_circ must be >= 1")
    n_circ_nodes = n_circ
    n_radial_nodes = n_radial + 1
    # Apex node + each ring of n_circ nodes
    nodes = []
    nodes.append([0.0, 0.0, radius])     # apex
    half_angle = math.radians(half_angle_deg)
    phis = np.linspace(0.0, half_angle, n_radial_nodes)[1:]
    thetas = np.linspace(0.0, 2.0 * math.pi, n_circ_nodes + 1)[:n_circ_nodes]
    for phi in phis:
        sphi = math.sin(phi)
        cphi = math.cos(phi)
        for theta in thetas:
            nodes.append([radius * sphi * math.cos(theta),
                            radius * sphi * math.sin(theta),
                            radius * cphi])
    nodes = np.array(nodes)
    # Quads
    quads = []
    # First ring: apex (index 0) + first sweep of n_circ nodes
    for i in range(n_circ_nodes):
        n0 = 0
        n1 = 1 + i
        n2 = 1 + (i + 1) % n_circ_nodes
        n3 = 0      # degenerate corner
        quads.append([n0, n1, n2, n3])
    # Subsequent rings (regular quads)
    for k in range(1, n_radial):
        base0 = 1 + (k - 1) * n_circ_nodes
        base1 = 1 + k * n_circ_nodes
        for i in range(n_circ_nodes):
            n0 = base0 + i
            n1 = base0 + (i + 1) % n_circ_nodes
            n2 = base1 + (i + 1) % n_circ_nodes
            n3 = base1 + i
            quads.append([n0, n1, n2, n3])
    return nodes, np.array(quads, dtype=int)
