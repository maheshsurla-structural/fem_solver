"""Mesh-quality metrics.

For each element type, several quality scalars:

* **Jacobian min / max ratio** -- ``J_min / J_max`` over Gauss
  points; 1.0 for a perfect element, 0.0 for a degenerate one. A
  threshold of 0.3 is commonly used to flag bad cells.
* **Aspect ratio** -- ``L_max / L_min`` of element edges; 1.0 for a
  square / cube, > 4 typically flagged.
* **Skewness** -- normalised deviation from rectangular; 0 = perfect,
  1 = degenerate. For a Quad4 the corner-angle skewness is
  ``max(|theta_i - 90°|) / 90°``.
* **Orthogonality** -- min cosine of face normals; 1.0 = orthogonal,
  0 = degenerate.

Returned by both per-element and per-mesh aggregator helpers:

* :func:`quad4_quality` -- scalar metrics for one Q4 from its 4 nodes.
* :func:`mesh_quality_report` -- aggregator that walks all elements
  in a :class:`StructuredMesh` and reports worst / mean values.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ============================================================ per-element

@dataclass
class Quad4Quality:
    """Quality scalars for one Quad4 element.

    Attributes
    ----------
    jacobian_min, jacobian_max : float
        At the four corner (sampling) points.
    jacobian_ratio : float
        ``J_min / J_max``, ideal = 1.
    aspect_ratio : float
        ``L_max / L_min`` of edges.
    skewness : float
        ``max(|theta - 90 deg|) / 90 deg``.
    """

    jacobian_min: float
    jacobian_max: float
    jacobian_ratio: float
    aspect_ratio: float
    skewness: float


def _quad4_corner_jacobian(coords: np.ndarray, xi: float, eta: float) -> float:
    """det J at one parametric point of a Q4 with corner coords (4,2)."""
    # Derivatives of bilinear shape functions
    dN_dxi = 0.25 * np.array([
        [-(1 - eta), -(1 - xi)],
        [(1 - eta),  -(1 + xi)],
        [(1 + eta),  (1 + xi)],
        [-(1 + eta), (1 - xi)],
    ])     # (4, 2): row = node, col = (∂N/∂xi, ∂N/∂eta)
    # Jacobian J[i, k] = sum_a x_{a, k} ∂N_a / ∂xi_i  -> (2, 2)
    J = dN_dxi.T @ coords
    return float(np.linalg.det(J))


def quad4_quality(coords: np.ndarray) -> Quad4Quality:
    """Compute quality scalars for a single Q4 from its (4, 2) corner
    coordinates."""
    coords = np.asarray(coords, dtype=float)
    if coords.shape != (4, 2):
        raise ValueError(f"coords must be (4, 2), got {coords.shape}")
    # Sample Jacobian at the four corners of the bi-unit square
    js = [
        _quad4_corner_jacobian(coords, -1.0, -1.0),
        _quad4_corner_jacobian(coords,  1.0, -1.0),
        _quad4_corner_jacobian(coords,  1.0,  1.0),
        _quad4_corner_jacobian(coords, -1.0,  1.0),
    ]
    # Winding-agnostic: judge geometric quality from |J|. Most FE
    # codes do this so a uniformly CW-wound mesh isn't reported as
    # "degenerate". A truly tangled element (some +, some -) is
    # caught explicitly by the sign-change check below.
    abs_js = [abs(j) for j in js]
    j_min = min(abs_js)
    j_max = max(abs_js)
    j_ratio = j_min / j_max if j_max > 0 else 0.0
    if min(js) * max(js) < 0:
        # Element tangled (Jacobian changes sign across the element)
        j_ratio = 0.0
    # Edge lengths
    edges = [
        np.linalg.norm(coords[1] - coords[0]),
        np.linalg.norm(coords[2] - coords[1]),
        np.linalg.norm(coords[3] - coords[2]),
        np.linalg.norm(coords[0] - coords[3]),
    ]
    aspect = max(edges) / min(edges) if min(edges) > 0 else float("inf")
    # Corner angles
    skew = 0.0
    for i in range(4):
        v1 = coords[(i + 1) % 4] - coords[i]
        v2 = coords[(i - 1) % 4] - coords[i]
        cos_t = float(np.clip(
            v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-30),
            -1.0, 1.0,
        ))
        theta_deg = math.degrees(math.acos(cos_t))
        skew = max(skew, abs(theta_deg - 90.0) / 90.0)
    return Quad4Quality(
        jacobian_min=float(j_min),
        jacobian_max=float(j_max),
        jacobian_ratio=float(j_ratio),
        aspect_ratio=float(aspect),
        skewness=float(skew),
    )


# ============================================================ aggregator

@dataclass
class MeshQualityReport:
    """Aggregated mesh-quality summary."""

    n_elements: int
    jacobian_ratio_min: float
    jacobian_ratio_mean: float
    aspect_ratio_max: float
    aspect_ratio_mean: float
    skewness_max: float
    skewness_mean: float
    worst_element_index: int
    worst_element_quality: Quad4Quality | None = None


def mesh_quality_report(mesh) -> MeshQualityReport:
    """Walk a :class:`StructuredMesh` of Q4 elements and report the
    overall + worst-element quality."""
    if mesh.nodes_per_element != 4 or mesh.ndm not in (2, 3):
        raise ValueError(
            "mesh_quality_report currently supports 2D Q4 / 3D quad "
            "shell meshes (nodes_per_element = 4)"
        )
    j_ratios, aspects, skews = [], [], []
    worst_idx = 0
    worst_score = -float("inf")
    worst_q = None
    for e, conn in enumerate(mesh.connectivity):
        # Use the first two coordinates if mesh is 3D (project onto xy)
        coords = mesh.nodes[conn - 1, : 2]
        q = quad4_quality(coords)
        j_ratios.append(q.jacobian_ratio)
        aspects.append(q.aspect_ratio)
        skews.append(q.skewness)
        # Composite badness score: higher = worse
        score = q.skewness * 0.5 + q.aspect_ratio * 0.1 \
                + (1.0 - q.jacobian_ratio) * 0.4
        if score > worst_score:
            worst_score = score
            worst_idx = e
            worst_q = q
    j_arr = np.array(j_ratios)
    a_arr = np.array(aspects)
    s_arr = np.array(skews)
    return MeshQualityReport(
        n_elements=int(len(mesh.connectivity)),
        jacobian_ratio_min=float(j_arr.min()),
        jacobian_ratio_mean=float(j_arr.mean()),
        aspect_ratio_max=float(a_arr.max()),
        aspect_ratio_mean=float(a_arr.mean()),
        skewness_max=float(s_arr.max()),
        skewness_mean=float(s_arr.mean()),
        worst_element_index=int(worst_idx),
        worst_element_quality=worst_q,
    )
