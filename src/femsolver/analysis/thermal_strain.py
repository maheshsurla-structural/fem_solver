"""Thermo-mechanical coupling (one-way: thermal → mechanical).

For a material with linear thermal expansion ``alpha``, a temperature
change ``Delta T = T - T_ref`` produces an initial (eigen) strain::

    eps_th = alpha * Delta T * I

(in 2D / 3D, isotropic). In a constrained structure this drives
stresses; in an unconstrained one it drives deformation. The
coupling is one-way: the temperature field is solved by
:mod:`femsolver.thermal.heat_conduction` first, then this module
generates equivalent nodal loads ``f = integral B^T D eps_th dV``
that are applied to the mechanical model.

The pattern:

    1. Build a *thermal* model (``ndf=1``); solve for ``T(node)``.
    2. Build a *mechanical* model (``ndf=2`` or ``3``) with mirrored
       node coordinates / connectivity.
    3. Call :func:`apply_thermal_load` with the thermal temperatures
       and reference temperature; it adds equivalent nodal forces to
       the mechanical model.
    4. Solve the mechanical model in the normal way.

This module supports the standard 2D and 3D continuum elements
(``Quad4`` plane stress / plane strain, ``Hex8``); for beam / shell
elements thermal strain is generally handled at the section level
(``alpha · Delta T · A`` axial pre-strain) which is straightforward
to express directly via a nodal moment / axial-force pattern.
"""
from __future__ import annotations

import numpy as np

from femsolver.elements.plane import Quad4
from femsolver.elements.solid import Hex8, _hex8_dN_dxi
from femsolver.numerics.quadrature import gauss_legendre_2d_quad


# ============================================================ 2D thermal force

def _thermal_force_quad4(
    elem: Quad4,
    T_nodes: np.ndarray,
    *,
    T_ref: float,
    alpha: float,
) -> np.ndarray:
    """Equivalent nodal force on a Quad4 from thermal strain.

    Parameters
    ----------
    elem : Quad4
        Plane-stress / plane-strain element.
    T_nodes : (4,) array
        Temperatures at the four nodes.
    T_ref : float
        Reference temperature at which thermal strain is zero.
    alpha : float
        Linear thermal expansion coefficient (1/K).

    Returns
    -------
    f_th : (8,) array
        Equivalent nodal force vector in element-local DOF order
        ``[u1, v1, u2, v2, u3, v3, u4, v4]``.
    """
    X = elem.node_coords()
    D = elem.D()
    t = elem.thickness
    f = np.zeros(8)
    xi, eta, w = gauss_legendre_2d_quad(elem.quadrature)
    for q in range(xi.size):
        _, detJ, dN_dx = elem.jacobian(float(xi[q]), float(eta[q]), X)
        B = elem.B_matrix(dN_dx)
        N = elem.shape_functions(float(xi[q]), float(eta[q]))
        T_at_gp = float(N @ T_nodes)
        dT = T_at_gp - T_ref
        # Initial (eigen) strain in Voigt: [exx, eyy, gamma_xy]
        # For plane stress: eps_th = alpha * dT * [1, 1, 0]
        # For plane strain: same Voigt form; the constraint enters via D
        eps_th = alpha * dT * np.array([1.0, 1.0, 0.0])
        f += (B.T @ D @ eps_th) * (t * detJ * w[q])
    return f


# ============================================================ 3D thermal force

def _thermal_force_hex8(
    elem: Hex8,
    T_nodes: np.ndarray,
    *,
    T_ref: float,
    alpha: float,
) -> np.ndarray:
    """Equivalent nodal force on a Hex8 from thermal strain."""
    X = elem.node_coords()
    # Hex8.D() returns the 6x6 3D elastic matrix
    D = elem.D() if hasattr(elem, "D") else elem.material.D_3d()
    f = np.zeros(24)
    # 2x2x2 Gauss
    gp = 1.0 / np.sqrt(3.0)
    pts = [(-gp, -gp, -gp), (gp, -gp, -gp), (gp, gp, -gp), (-gp, gp, -gp),
           (-gp, -gp,  gp), (gp, -gp,  gp), (gp, gp,  gp), (-gp, gp,  gp)]
    w = 1.0
    for (xi, eta, zeta) in pts:
        dN = _hex8_dN_dxi(xi, eta, zeta)
        J = dN @ X
        detJ = float(np.linalg.det(J))
        dN_dx = np.linalg.solve(J, dN)
        # Build the 3D B-matrix manually (mirror solid.py)
        B = np.zeros((6, 24))
        for i in range(8):
            dNx, dNy, dNz = dN_dx[0, i], dN_dx[1, i], dN_dx[2, i]
            B[0, 3 * i + 0] = dNx
            B[1, 3 * i + 1] = dNy
            B[2, 3 * i + 2] = dNz
            B[3, 3 * i + 0] = dNy
            B[3, 3 * i + 1] = dNx
            B[4, 3 * i + 1] = dNz
            B[4, 3 * i + 2] = dNy
            B[5, 3 * i + 0] = dNz
            B[5, 3 * i + 2] = dNx
        # Shape functions at Gauss point for T_at_gp
        from femsolver.elements.solid import _hex8_shape
        N = _hex8_shape(xi, eta, zeta)
        T_at_gp = float(N @ T_nodes)
        dT = T_at_gp - T_ref
        # 3D thermal eigen strain in Voigt:
        # [exx, eyy, ezz, gxy, gyz, gzx] = alpha * dT * [1,1,1,0,0,0]
        eps_th = alpha * dT * np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        f += (B.T @ D @ eps_th) * (detJ * w)
    return f


# ============================================================ public API

def apply_thermal_load(
    mech_model,
    *,
    temperatures: dict,
    T_ref: float,
    alpha: float,
) -> int:
    """Add thermal-strain equivalent nodal loads to a mechanical model.

    For each supported element in the model, computes the equivalent
    nodal force from the thermal eigen strain
    ``alpha · (T - T_ref) · I`` and adds it to the corresponding nodal
    load vector.

    Parameters
    ----------
    mech_model : Model
        Mechanical model (``ndf >= 2``).
    temperatures : dict
        ``{node_tag: T}`` mapping. Must cover all mechanical nodes.
    T_ref : float
    alpha : float

    Returns
    -------
    int
        Number of elements processed.
    """
    n_done = 0
    for elem in mech_model.elements.values():
        if isinstance(elem, Quad4):
            T_nodes = np.array([float(temperatures[t])
                                  for t in elem.node_tags])
            f_th = _thermal_force_quad4(
                elem, T_nodes, T_ref=T_ref, alpha=alpha,
            )
            # Scatter to nodal loads
            for k, ntag in enumerate(elem.node_tags):
                mech_model.add_nodal_load(
                    ntag,
                    [f_th[2 * k], f_th[2 * k + 1]],
                )
            n_done += 1
        elif isinstance(elem, Hex8):
            T_nodes = np.array([float(temperatures[t])
                                  for t in elem.node_tags])
            f_th = _thermal_force_hex8(
                elem, T_nodes, T_ref=T_ref, alpha=alpha,
            )
            for k, ntag in enumerate(elem.node_tags):
                mech_model.add_nodal_load(
                    ntag,
                    [f_th[3 * k], f_th[3 * k + 1], f_th[3 * k + 2]],
                )
            n_done += 1
        # Other element types (beam, shell): user supplies thermal
        # loads directly via section axial pre-strain etc.
    return n_done


def beam_thermal_axial_force(
    *, alpha: float, dT: float, E: float, A: float,
) -> float:
    """Thermal-axial pre-force ``-E A alpha dT`` for a fully-restrained
    beam segment with temperature change ``dT`` from the reference
    state.

    For a beam fully restrained at both ends, the resulting axial
    force is ``F = -E A alpha (T - T_ref)`` (negative = compression
    for positive ``dT``).
    """
    return -E * A * alpha * dT


def beam_thermal_gradient_moment(
    *, alpha: float, dT_top: float, dT_bot: float,
    E: float, I: float, h: float,
) -> float:
    """Thermal-gradient moment for a beam with linear temperature
    gradient ``dT_top - dT_bot`` across depth ``h``.

    For a fully-restrained beam::

        M_th = -E I alpha (dT_top - dT_bot) / h

    Positive ``(dT_top - dT_bot)`` (top hotter) gives negative
    moment (sagging) on a restrained beam — typical thermal-warp
    sign.
    """
    return -E * I * alpha * (dT_top - dT_bot) / h
