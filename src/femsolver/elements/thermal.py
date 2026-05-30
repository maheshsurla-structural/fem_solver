"""Thermal isoparametric elements.

A thermal element has **one DOF per node** (temperature ``T``) and
produces:

* a **conductivity** matrix ``K_T = integral B^T (k I) B dV``, which
  is the thermal analogue of the mechanical stiffness;
* a **capacitance** matrix ``C_T = integral N^T (rho c) N dV``, the
  thermal analogue of the consistent mass;
* an **internal-flux** vector ``f = K_T T``.

Because the assembler keys off ``dofs_per_node`` and
``K_global()``, the thermal elements drop straight into the existing
``Model`` machinery as long as the model is constructed with
``ndf=1``.

This module ships:

* :class:`ThermalQuad4` — 2D bilinear isoparametric, 2x2 Gauss.
* :class:`ThermalHex8`  — 3D trilinear isoparametric, 2x2x2 Gauss.
* :class:`ConvectionEdge2D` — line-load boundary element that adds the
  Robin-BC contribution to the assembled system.
"""
from __future__ import annotations

import numpy as np

from femsolver.elements.base import Element
from femsolver.elements.plane import Quad4
from femsolver.elements.solid import _hex8_dN_dxi, _hex8_shape
from femsolver.numerics.quadrature import gauss_legendre_2d_quad


# ============================================================ 2D thermal Quad4

class ThermalQuad4(Element):
    """2D bilinear thermal element (1 DOF / node = temperature).

    Reuses the bi-unit-square shape functions and Jacobian helpers from
    :class:`~femsolver.elements.plane.Quad4`; only the constitutive
    matrix changes (isotropic conductivity ``k I``).
    """

    n_nodes = 4
    dofs_per_node = 1

    def __init__(
        self,
        tag: int,
        nodes,
        material,
        thickness: float = 1.0,
        quadrature: int = 2,
    ):
        super().__init__(tag, nodes, material)
        if thickness <= 0:
            raise ValueError(f"thickness must be > 0, got {thickness}")
        self.thickness = float(thickness)
        self.quadrature = int(quadrature)

    # Shape-function helpers reused from the mechanical Quad4
    shape_functions = staticmethod(Quad4.shape_functions)
    dN_dxi = staticmethod(Quad4.dN_dxi)

    def jacobian(self, xi: float, eta: float, X: np.ndarray):
        dN = self.dN_dxi(xi, eta)
        J = dN @ X
        detJ = float(np.linalg.det(J))
        if detJ <= 0.0:
            raise ValueError(
                f"ThermalQuad4 element {self.tag}: non-positive Jacobian "
                f"({detJ:g}) at xi={xi}, eta={eta}; check node ordering."
            )
        dN_dx = np.linalg.solve(J, dN)
        return detJ, dN_dx

    def K_global(self) -> np.ndarray:
        """Conductivity matrix (4 x 4)."""
        X = self.node_coords()
        k = self.material.k
        t = self.thickness
        K = np.zeros((4, 4))
        xi, eta, w = gauss_legendre_2d_quad(self.quadrature)
        for q in range(xi.size):
            detJ, dN_dx = self.jacobian(float(xi[q]), float(eta[q]), X)
            # B in 2D thermal is just dN_dx, shape (2, 4)
            B = dN_dx
            K += B.T @ B * (k * t * detJ * w[q])
        return K

    def M_global(self, *, lumped: bool = False) -> np.ndarray:
        """Heat-capacity matrix ``C_T`` (called M_global so it slots
        into the existing mass-assembler pipeline)."""
        X = self.node_coords()
        rho_c = self.material.rho_c
        t = self.thickness
        C = np.zeros((4, 4))
        xi, eta, w = gauss_legendre_2d_quad(self.quadrature)
        for q in range(xi.size):
            N = self.shape_functions(float(xi[q]), float(eta[q]))
            detJ, _ = self.jacobian(float(xi[q]), float(eta[q]), X)
            C += np.outer(N, N) * (rho_c * t * detJ * w[q])
        if lumped:
            # row-sum lumping
            C = np.diag(C.sum(axis=1))
        return C


# ============================================================ 3D thermal Hex8

class ThermalHex8(Element):
    """3D trilinear thermal element (1 DOF / node)."""

    n_nodes = 8
    dofs_per_node = 1

    def __init__(self, tag: int, nodes, material):
        super().__init__(tag, nodes, material)

    shape_functions = staticmethod(_hex8_shape)
    dN_dxi = staticmethod(_hex8_dN_dxi)

    def jacobian(self, xi: float, eta: float, zeta: float, X: np.ndarray):
        dN = self.dN_dxi(xi, eta, zeta)
        J = dN @ X
        detJ = float(np.linalg.det(J))
        if detJ <= 0.0:
            raise ValueError(
                f"ThermalHex8 element {self.tag}: non-positive Jacobian "
                f"({detJ:g}) at ({xi},{eta},{zeta})"
            )
        dN_dx = np.linalg.solve(J, dN)
        return detJ, dN_dx

    def _gauss_points_3d(self):
        """2x2x2 Gauss-Legendre on the bi-unit cube."""
        gp = 1.0 / np.sqrt(3.0)
        pts = [(-gp, -gp, -gp), (gp, -gp, -gp), (gp, gp, -gp), (-gp, gp, -gp),
               (-gp, -gp,  gp), (gp, -gp,  gp), (gp, gp,  gp), (-gp, gp,  gp)]
        return pts, 1.0

    def K_global(self) -> np.ndarray:
        X = self.node_coords()
        k = self.material.k
        K = np.zeros((8, 8))
        pts, w = self._gauss_points_3d()
        for (xi, eta, zeta) in pts:
            detJ, dN_dx = self.jacobian(xi, eta, zeta, X)
            B = dN_dx          # (3, 8)
            K += B.T @ B * (k * detJ * w)
        return K

    def M_global(self, *, lumped: bool = False) -> np.ndarray:
        X = self.node_coords()
        rho_c = self.material.rho_c
        C = np.zeros((8, 8))
        pts, w = self._gauss_points_3d()
        for (xi, eta, zeta) in pts:
            N = self.shape_functions(xi, eta, zeta)
            detJ, _ = self.jacobian(xi, eta, zeta, X)
            C += np.outer(N, N) * (rho_c * detJ * w)
        if lumped:
            C = np.diag(C.sum(axis=1))
        return C


# ============================================================ convection edge BC

class ConvectionEdge2D(Element):
    """Robin-BC element for 2D heat transfer along a straight edge.

    Imposes ``q_n = h (T - T_inf)`` on the boundary, contributing
    ``K_conv = h ∫ N^T N ds`` (added to the conductivity matrix) and
    ``f_conv = h T_inf ∫ N^T ds`` (added to the right-hand side).

    Use as a regular element: define two end nodes on the convection
    boundary and attach the edge element. Both nodes share the
    ``ThermalMaterial`` (only for typing — its ``k`` is not used here).

    Parameters
    ----------
    tag : int
    nodes : (2,) tuple
    material : any (only typed for the registry; unused)
    h : float
        Convection coefficient (W/(m^2·K)).  Natural ~ 5-25,
        forced ~ 25-250.
    T_inf : float
        Ambient temperature (K or °C, matching the model's T units).
    thickness : float, default 1.0
        Out-of-plane thickness (m).
    """

    n_nodes = 2
    dofs_per_node = 1

    def __init__(
        self,
        tag: int,
        nodes,
        material,
        h: float,
        T_inf: float,
        thickness: float = 1.0,
    ):
        super().__init__(tag, nodes, material)
        if h < 0.0:
            raise ValueError(f"h must be >= 0, got {h}")
        if thickness <= 0.0:
            raise ValueError(f"thickness must be > 0, got {thickness}")
        self.h = float(h)
        self.T_inf = float(T_inf)
        self.thickness = float(thickness)

    def edge_length(self) -> float:
        X = self.node_coords()
        return float(np.linalg.norm(X[1] - X[0]))

    def K_global(self) -> np.ndarray:
        """Convection contribution to conductivity::
            K_conv = (h t L / 6) * [[2, 1], [1, 2]]
        (exact integral of N^T N over a line, 2-node linear).
        """
        L = self.edge_length()
        c = self.h * self.thickness * L / 6.0
        return c * np.array([[2.0, 1.0], [1.0, 2.0]])

    def f_eq_global(self) -> np.ndarray:
        """Equivalent flux at the two end nodes from ambient convection::
            f_conv = (h T_inf t L / 2) * [1, 1]
        (lumping the integral of N^T over the line).
        """
        L = self.edge_length()
        c = self.h * self.T_inf * self.thickness * L / 2.0
        return c * np.array([1.0, 1.0])
