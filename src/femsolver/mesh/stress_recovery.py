"""Stress recovery: Gauss-point -> nodal averages, principal stresses,
von Mises.

Element-level stress (and strain) is computed natively at the Gauss
integration points; for visualisation and design purposes we usually
want **nodal** values. Two extrapolation methods are common:

* **Direct sample at corners** -- evaluate the constitutive law at
  ``(xi, eta) = (±1, ±1)`` and average across all elements sharing
  the node.
* **Superconvergent patch recovery (Zienkiewicz-Zhu)** -- more
  accurate but heavier; deferred to a future phase.

This module implements the direct method, plus principal-stress and
von-Mises calculators for 2D (plane stress / plane strain) and 3D
states.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ============================================================ 2D principal

def principal_stresses_2d(
    sigma: np.ndarray,
) -> tuple[float, float, float]:
    """Principal stresses + angle for a 2D stress tensor.

    Parameters
    ----------
    sigma : (3,) array
        ``[sigma_xx, sigma_yy, tau_xy]``.

    Returns
    -------
    (sigma_1, sigma_2, theta_p) : (float, float, float)
        ``sigma_1 >= sigma_2`` and ``theta_p`` is the angle of the
        major principal axis from the x-axis (rad).
    """
    sigma = np.asarray(sigma, dtype=float).ravel()
    s_xx, s_yy, t_xy = sigma[0], sigma[1], sigma[2]
    mean = 0.5 * (s_xx + s_yy)
    radius = math.sqrt((0.5 * (s_xx - s_yy)) ** 2 + t_xy ** 2)
    sig_1 = mean + radius
    sig_2 = mean - radius
    theta_p = 0.5 * math.atan2(2.0 * t_xy, s_xx - s_yy)
    return float(sig_1), float(sig_2), float(theta_p)


def von_mises_2d(sigma: np.ndarray) -> float:
    """von Mises equivalent stress in 2D (plane stress assumed)::

        sigma_eq = sqrt(sigma_xx^2 + sigma_yy^2 - sigma_xx sigma_yy
                        + 3 tau_xy^2).
    """
    sigma = np.asarray(sigma, dtype=float).ravel()
    s_xx, s_yy, t_xy = sigma[0], sigma[1], sigma[2]
    return float(math.sqrt(
        s_xx ** 2 + s_yy ** 2 - s_xx * s_yy + 3.0 * t_xy ** 2
    ))


# ============================================================ 3D principal

def principal_stresses_3d(sigma: np.ndarray) -> np.ndarray:
    """Principal stresses of a 3D stress tensor (sorted descending).

    Parameters
    ----------
    sigma : (6,) array
        Voigt order ``[sigma_xx, sigma_yy, sigma_zz, tau_xy, tau_yz,
        tau_zx]``.
    """
    sigma = np.asarray(sigma, dtype=float).ravel()
    if sigma.size != 6:
        raise ValueError("sigma must be a 6-vector in Voigt order")
    T = np.array([
        [sigma[0], sigma[3], sigma[5]],
        [sigma[3], sigma[1], sigma[4]],
        [sigma[5], sigma[4], sigma[2]],
    ])
    w = np.sort(np.linalg.eigvalsh(T))[::-1]
    return w


def von_mises_3d(sigma: np.ndarray) -> float:
    """von Mises equivalent stress in 3D::

        sigma_eq = sqrt(½ [(sigma_xx - sigma_yy)^2
                          + (sigma_yy - sigma_zz)^2
                          + (sigma_zz - sigma_xx)^2
                          + 6 (tau_xy^2 + tau_yz^2 + tau_zx^2)]).
    """
    sigma = np.asarray(sigma, dtype=float).ravel()
    s = sigma
    return float(math.sqrt(0.5 * (
        (s[0] - s[1]) ** 2 + (s[1] - s[2]) ** 2 + (s[2] - s[0]) ** 2
        + 6.0 * (s[3] ** 2 + s[4] ** 2 + s[5] ** 2)
    )))


# ============================================================ nodal averaging

@dataclass
class NodalStressField:
    """Nodal-averaged stress field.

    Attributes
    ----------
    sigma_at_nodes : np.ndarray
        Shape ``(n_nodes, n_components)`` -- e.g., 3 for plane
        stress (sigma_xx, sigma_yy, tau_xy), 6 for 3D.
    n_contributions : np.ndarray
        Number of elements contributing to each node (= valence).
    """

    sigma_at_nodes: np.ndarray
    n_contributions: np.ndarray


def average_quad4_stresses_to_nodes(
    model,
) -> NodalStressField:
    """Recover nodal-averaged stresses from a model of Q4 elements.

    For each Q4 element, the *current* stress (after analysis) at the
    four corner sampling points (xi, eta) = (±1, ±1) is assumed to be
    stored as ``element.gp_stress[4]``. Each corner's stress is
    accumulated into the corresponding node, divided by the valence
    (number of elements sharing that node).

    If an element only carries 2x2 Gauss stresses (not corner), this
    method **extrapolates** them via the bilinear shape functions
    evaluated at the corners. Each Gauss point at
    ``(±1/sqrt(3), ±1/sqrt(3))`` is mapped to the corner via::

        sigma_corner = sum_q  N_q(xi_corner, eta_corner) · sigma_q

    where ``N_q`` is the bilinear shape function with corners at
    ``(±sqrt(3), ±sqrt(3))`` (the extrapolation form for Q4).
    """
    n_nodes = len(model.nodes)
    n_comp = 3       # plane stress / plane strain Voigt
    accum = np.zeros((n_nodes, n_comp))
    valence = np.zeros(n_nodes, dtype=int)
    sqrt3 = math.sqrt(3.0)
    # Extrapolation matrix from 2x2 Gauss to 4 corners (Q4)
    # corner xi = (-1, +1, +1, -1), eta = (-1, -1, +1, +1)
    # Gauss xi = (-1/√3, +1/√3, +1/√3, -1/√3)
    # The extrapolation N values: when "shape functions" are evaluated
    # at sqrt(3) corners using Gauss-point coords:
    ngp = [(-1 / sqrt3, -1 / sqrt3),
           ( 1 / sqrt3, -1 / sqrt3),
           ( 1 / sqrt3,  1 / sqrt3),
           (-1 / sqrt3,  1 / sqrt3)]
    corners = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
    # Build 4x4 extrapolation matrix N_ij where row = corner, col = gp
    Nx = np.zeros((4, 4))
    for r, (xi_c, eta_c) in enumerate(corners):
        # Express corner coords in the "GP-as-nodes" parametric space,
        # which uses base sqrt(3); shape function for GP q at corner:
        for q, (xi_g, eta_g) in enumerate(ngp):
            # bilinear shape with "nodes" at ±1/√3 (so use scaled coords)
            xi_norm = sqrt3 * xi_c
            eta_norm = sqrt3 * eta_c
            xi_q = sqrt3 * xi_g
            eta_q = sqrt3 * eta_g
            Nx[r, q] = 0.25 * (1.0 + xi_norm * xi_q) \
                                * (1.0 + eta_norm * eta_q)

    for e in model.elements.values():
        if not hasattr(e, "gp_stress") or not e.gp_stress:
            # Element has no recovered stresses; skip
            continue
        sigma_gp = np.array(e.gp_stress)        # (4, 3)
        # If element stored stresses at 2x2 Gauss, extrapolate to corners
        if sigma_gp.shape == (4, n_comp):
            sigma_corners = Nx @ sigma_gp           # (4, 3)
        else:
            sigma_corners = sigma_gp
        for c, node_tag in enumerate(e.node_tags):
            idx = node_tag - 1          # assumes contiguous 1-based tags
            accum[idx] += sigma_corners[c]
            valence[idx] += 1
    # Average
    sigma_nodes = np.zeros_like(accum)
    for i in range(n_nodes):
        if valence[i] > 0:
            sigma_nodes[i] = accum[i] / valence[i]
    return NodalStressField(
        sigma_at_nodes=sigma_nodes,
        n_contributions=valence,
    )


def von_mises_field(nodal_stress: NodalStressField) -> np.ndarray:
    """Compute the von Mises field from a 2D nodal-stress array."""
    sigma = nodal_stress.sigma_at_nodes
    n_nodes = sigma.shape[0]
    out = np.zeros(n_nodes)
    for i in range(n_nodes):
        out[i] = von_mises_2d(sigma[i])
    return out
