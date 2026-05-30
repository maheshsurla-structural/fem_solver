"""Hex8 Total-Lagrangian large-strain element.

The element evaluates the deformation gradient ``F`` at each Gauss
point from the current nodal displacements, queries a hyperelastic
material for ``(S, C^M)`` in the reference configuration, and assembles
the **material** + **geometric** tangent stiffness contributions:

    K_T = ∫_{Ω_0} ( B_L^T C^M B_L + B_NL^T τ B_NL ) dV_0,
    f_int = ∫_{Ω_0} B_L^T S dV_0,

with ``B_L`` the linear strain-displacement matrix in the reference
configuration and ``τ`` the 2nd PK stress matrix (used for the
geometric / initial-stress term).

The element uses standard 2 x 2 x 2 Gauss-Legendre quadrature on the
bi-unit cube and the same shape functions as the small-strain
:class:`~femsolver.elements.solid.Hex8` (re-uses the helpers in
:mod:`femsolver.elements.solid`).

This element is suitable for moderate-to-large strain analyses of
rubber components, packers, isolator pads, and finite-strain
plasticity (combined with :mod:`femsolver.materials.finite_j2`).
"""
from __future__ import annotations

import numpy as np

from femsolver.elements.base import Element
from femsolver.elements.solid import _hex8_dN_dxi, _hex8_shape


class Hex8TL(Element):
    """8-node hexahedron, Total-Lagrangian formulation, large strain.

    Parameters
    ----------
    tag : int
    nodes : (8,) sequence of node tags (standard hex ordering).
    material : hyperelastic material
        Must expose ``response_S(F) -> (S_voigt, C_M_voigt)``.
    """

    n_nodes = 8
    dofs_per_node = 3

    def __init__(self, tag: int, nodes, material):
        super().__init__(tag, nodes, material)
        # Cache reference geometry on bind (lazy)
        self._X_ref: np.ndarray | None = None

    # -------------------------------------------------- geometry

    @staticmethod
    def _gauss_points():
        gp = 1.0 / np.sqrt(3.0)
        pts = [(-gp, -gp, -gp), (gp, -gp, -gp), (gp, gp, -gp), (-gp, gp, -gp),
               (-gp, -gp,  gp), (gp, -gp,  gp), (gp, gp,  gp), (-gp, gp,  gp)]
        return pts

    def _reference_coords(self) -> np.ndarray:
        if self._X_ref is None:
            self._X_ref = self.node_coords()
        return self._X_ref

    def _current_coords(self) -> np.ndarray:
        """Reference coords + current nodal displacements."""
        X = self._reference_coords()
        u = self.gather_u().reshape(8, 3)
        return X + u

    # -------------------------------------------------- F and B at a Gauss point

    def _F_at_gp(self, xi, eta, zeta, X_ref, x_cur) -> tuple[np.ndarray, float, np.ndarray]:
        """Deformation gradient F (3, 3) and reference Jacobian det/dN_dX
        at the (xi, eta, zeta) Gauss point.

        Returns
        -------
        F : (3, 3)
        detJ_ref : float
        dN_dX : (3, 8) -- shape-function derivatives in REFERENCE coords
        """
        dN_dxi = _hex8_dN_dxi(xi, eta, zeta)
        J_ref = dN_dxi @ X_ref
        detJ_ref = float(np.linalg.det(J_ref))
        if detJ_ref <= 0.0:
            raise ValueError(
                f"Hex8TL element {self.tag}: non-positive reference "
                f"Jacobian at ({xi}, {eta}, {zeta})"
            )
        dN_dX = np.linalg.solve(J_ref, dN_dxi)        # (3, 8)
        # F_{iJ} = sum_a x_{a, i} dN_a/dX_J = (x_cur.T @ dN_dX.T)^T ?
        # Equivalent: F = x_cur^T @ dN_dX^T = (dN_dX @ x_cur).T
        F = (dN_dX @ x_cur).T                            # (3, 3)
        return F, detJ_ref, dN_dX

    @staticmethod
    def _BL_from_F_dN(F: np.ndarray, dN_dX: np.ndarray) -> np.ndarray:
        """Build the linear B_L (6, 24) at a Gauss point.

        Following Bathe (1996) Eq. 6.36b for Total Lagrangian, the
        contribution of node ``a`` (with shape gradient ``dN_a/dX``
        in reference coords) to ``B_L`` couples through ``F``:

            B_L^a = [ F_11 dN_a,1                  F_21 dN_a,1                  F_31 dN_a,1                ;
                      F_12 dN_a,2                  F_22 dN_a,2                  F_32 dN_a,2                ;
                      F_13 dN_a,3                  F_23 dN_a,3                  F_33 dN_a,3                ;
                      F_11 dN_a,2 + F_12 dN_a,1    F_21 dN_a,2 + F_22 dN_a,1    F_31 dN_a,2 + F_32 dN_a,1  ;
                      F_12 dN_a,3 + F_13 dN_a,2    F_22 dN_a,3 + F_23 dN_a,2    F_32 dN_a,3 + F_33 dN_a,2  ;
                      F_11 dN_a,3 + F_13 dN_a,1    F_21 dN_a,3 + F_23 dN_a,1    F_31 dN_a,3 + F_33 dN_a,1  ]
        """
        B = np.zeros((6, 24))
        for a in range(8):
            g = dN_dX[:, a]      # (3,) [g1, g2, g3]
            for i in range(3):
                col = 3 * a + i
                # eps_xx, eps_yy, eps_zz
                B[0, col] = F[i, 0] * g[0]
                B[1, col] = F[i, 1] * g[1]
                B[2, col] = F[i, 2] * g[2]
                # 2 eps_xy = ( F_i1 g_2 + F_i2 g_1 )  (Voigt engineering)
                B[3, col] = F[i, 0] * g[1] + F[i, 1] * g[0]
                # 2 eps_yz
                B[4, col] = F[i, 1] * g[2] + F[i, 2] * g[1]
                # 2 eps_zx
                B[5, col] = F[i, 0] * g[2] + F[i, 2] * g[0]
        return B

    @staticmethod
    def _BNL(dN_dX: np.ndarray) -> np.ndarray:
        """Build the nonlinear B_NL (9, 24) for the geometric-stiffness term.

        Each block of 3 rows holds the shape-function gradient with
        respect to one reference-coord direction; the geometric matrix
        is then ``S_hat`` (9, 9) with three copies of the 3x3 S on the
        diagonal.
        """
        B = np.zeros((9, 24))
        for a in range(8):
            for i in range(3):
                col = 3 * a + i
                B[0 + i, col] = dN_dX[0, a]    # row 0..2: ∂N/∂X_1
                B[3 + i, col] = dN_dX[1, a]    # row 3..5: ∂N/∂X_2
                B[6 + i, col] = dN_dX[2, a]    # row 6..8: ∂N/∂X_3
        return B

    @staticmethod
    def _S_hat(S_voigt: np.ndarray) -> np.ndarray:
        """Build the (9, 9) block-diagonal stress matrix for the
        geometric-stiffness assembly.

        ``S_hat = diag(S, S, S)`` (each block is the symmetric (3, 3)
        2nd PK stress).
        """
        S = np.zeros((3, 3))
        S[0, 0], S[1, 1], S[2, 2] = S_voigt[0], S_voigt[1], S_voigt[2]
        S[0, 1] = S[1, 0] = S_voigt[3]
        S[1, 2] = S[2, 1] = S_voigt[4]
        S[0, 2] = S[2, 0] = S_voigt[5]
        out = np.zeros((9, 9))
        out[0:3, 0:3] = S
        out[3:6, 3:6] = S
        out[6:9, 6:9] = S
        return out

    # -------------------------------------------------- internal force and tangent

    def f_int_global(self) -> np.ndarray:
        X_ref = self._reference_coords()
        x_cur = self._current_coords()
        f = np.zeros(24)
        for (xi, eta, zeta) in self._gauss_points():
            F, detJ_ref, dN_dX = self._F_at_gp(xi, eta, zeta, X_ref, x_cur)
            S_voigt, _ = self.material.response_S(F)
            B_L = self._BL_from_F_dN(F, dN_dX)
            f += B_L.T @ S_voigt * detJ_ref
        return f

    def K_global(self) -> np.ndarray:
        """Initial elastic tangent (at undeformed state).

        For analyses started from rest this equals the linear elastic
        ``K`` if the material has a meaningful small-strain limit.
        Subsequent Newton steps should use :meth:`K_tangent_global`.
        """
        X_ref = self._reference_coords()
        K = np.zeros((24, 24))
        I3 = np.eye(3)
        # Use F = I (undeformed) for the initial tangent
        for (xi, eta, zeta) in self._gauss_points():
            dN_dxi = _hex8_dN_dxi(xi, eta, zeta)
            J_ref = dN_dxi @ X_ref
            detJ_ref = float(np.linalg.det(J_ref))
            dN_dX = np.linalg.solve(J_ref, dN_dxi)
            _, C_M = self.material.response_S(I3)
            B_L = self._BL_from_F_dN(I3, dN_dX)
            K += B_L.T @ C_M @ B_L * detJ_ref
        return K

    def K_tangent_global(self) -> np.ndarray:
        """Tangent at the current deformation: material + geometric."""
        X_ref = self._reference_coords()
        x_cur = self._current_coords()
        K = np.zeros((24, 24))
        for (xi, eta, zeta) in self._gauss_points():
            F, detJ_ref, dN_dX = self._F_at_gp(xi, eta, zeta, X_ref, x_cur)
            S_voigt, C_M = self.material.response_S(F)
            B_L = self._BL_from_F_dN(F, dN_dX)
            B_NL = self._BNL(dN_dX)
            S_hat = self._S_hat(S_voigt)
            K += (B_L.T @ C_M @ B_L
                  + B_NL.T @ S_hat @ B_NL) * detJ_ref
        return K
