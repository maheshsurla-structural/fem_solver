"""4-node quadrilateral plane element (Q4) with bilinear shape functions."""
from __future__ import annotations

import numpy as np

from femsolver.elements.base import Element
from femsolver.numerics.quadrature import gauss_legendre_2d_quad


class Quad4(Element):
    """Bilinear isoparametric quadrilateral, plane stress or plane strain.

    Parameters
    ----------
    tag : int
    nodes : (4,) sequence of node tags, counter-clockwise.
    material : Material
    thickness : float
        Out-of-plane thickness. Defaults to 1.0.
    state : {"plane_stress", "plane_strain"}, default "plane_stress".
    quadrature : int, default 2 (2x2 Gauss).
    """

    n_nodes = 4
    dofs_per_node = 2

    def __init__(
        self,
        tag: int,
        nodes,
        material,
        thickness: float = 1.0,
        state: str = "plane_stress",
        quadrature: int = 2,
    ):
        super().__init__(tag, nodes, material)
        if thickness <= 0:
            raise ValueError(f"thickness must be positive, got {thickness}")
        if state not in ("plane_stress", "plane_strain"):
            raise ValueError(f"state must be plane_stress or plane_strain, got {state}")
        self.thickness = float(thickness)
        self.state = state
        self.quadrature = int(quadrature)
        self.gp_stress: list[np.ndarray] = []  # populated by recover()
        self.gp_strain: list[np.ndarray] = []
        self._body_force: np.ndarray = np.zeros(2)

    @staticmethod
    def shape_functions(xi: float, eta: float) -> np.ndarray:
        """N (4,) at (xi, eta) on the bi-unit square."""
        return 0.25 * np.array([
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ])

    @staticmethod
    def dN_dxi(xi: float, eta: float) -> np.ndarray:
        """dN/d(xi,eta) as a (2, 4) matrix. Row 0 = dN/dxi, row 1 = dN/deta."""
        dN = 0.25 * np.array([
            [-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)],
            [-(1.0 - xi), -(1.0 + xi), (1.0 + xi),  (1.0 - xi)],
        ])
        return dN

    def D(self) -> np.ndarray:
        if self.state == "plane_stress":
            return self.material.D_plane_stress()
        return self.material.D_plane_strain()

    def jacobian(self, xi: float, eta: float, X: np.ndarray):
        """Return (J, detJ, dN_dx) where dN_dx is the (2, 4) physical-derivative matrix."""
        dN = self.dN_dxi(xi, eta)  # (2, 4)
        J = dN @ X  # (2, 2)
        detJ = float(np.linalg.det(J))
        if detJ <= 0.0:
            raise ValueError(
                f"Quad4 element {self.tag}: non-positive Jacobian determinant "
                f"({detJ:g}) at xi={xi}, eta={eta}. Check node ordering "
                f"(must be counter-clockwise) and element shape."
            )
        Jinv = np.linalg.inv(J)
        dN_dx = Jinv @ dN  # (2, 4)
        return J, detJ, dN_dx

    def B_matrix(self, dN_dx: np.ndarray) -> np.ndarray:
        """Strain-displacement matrix B (3, 8). Voigt order [exx, eyy, gxy]."""
        B = np.zeros((3, 8))
        for i in range(4):
            dNx = dN_dx[0, i]
            dNy = dN_dx[1, i]
            B[0, 2 * i] = dNx
            B[1, 2 * i + 1] = dNy
            B[2, 2 * i] = dNy
            B[2, 2 * i + 1] = dNx
        return B

    def K_global(self) -> np.ndarray:
        X = self.node_coords()  # (4, 2)
        D = self.D()
        t = self.thickness
        K = np.zeros((8, 8))
        xi, eta, w = gauss_legendre_2d_quad(self.quadrature)
        for q in range(xi.size):
            _, detJ, dN_dx = self.jacobian(float(xi[q]), float(eta[q]), X)
            B = self.B_matrix(dN_dx)
            K += (B.T @ D @ B) * (t * detJ * w[q])
        return K

    def M_global(self, *, lumped: bool = False) -> np.ndarray:
        rho = self.material.rho
        if rho == 0.0:
            return np.zeros((8, 8))
        X = self.node_coords()
        t = self.thickness
        # consistent: M = integral rho * Nbar^T Nbar t dA, where Nbar (2x8) maps
        # nodal disp -> point disp. Element area is invariant of state
        # (plane stress vs plane strain) for the mass term.
        M = np.zeros((8, 8))
        xi, eta, w = gauss_legendre_2d_quad(self.quadrature)
        for q in range(xi.size):
            N = self.shape_functions(float(xi[q]), float(eta[q]))
            _, detJ, _ = self.jacobian(float(xi[q]), float(eta[q]), X)
            jw = rho * t * detJ * w[q]
            Nbar = np.zeros((2, 8))
            for i in range(4):
                Nbar[0, 2 * i] = N[i]
                Nbar[1, 2 * i + 1] = N[i]
            M += (Nbar.T @ Nbar) * jw
        if lumped:
            # row-sum lumping (HRZ for bilinear quad reduces to equal split
            # of the total mass across the four translational DOF pairs)
            row_sum = M.sum(axis=1)
            return np.diag(row_sum)
        return M

    # ---------------------------------------------------------------- loading
    def set_body_force(self, bx: float, by: float) -> None:
        """Body force per unit volume (in global x, y)."""
        self._body_force = np.array([float(bx), float(by)])

    def clear_distributed_loads(self) -> None:
        self._body_force = np.zeros(2)

    def f_eq_global(self) -> np.ndarray:
        if not np.any(self._body_force):
            return np.zeros(8)
        X = self.node_coords()
        f = np.zeros(8)
        bx, by = self._body_force
        t = self.thickness
        xi, eta, w = gauss_legendre_2d_quad(self.quadrature)
        for q in range(xi.size):
            N = self.shape_functions(float(xi[q]), float(eta[q]))
            _, detJ, _ = self.jacobian(float(xi[q]), float(eta[q]), X)
            jw = t * detJ * w[q]
            for i in range(4):
                f[2 * i] += N[i] * bx * jw
                f[2 * i + 1] += N[i] * by * jw
        return f

    def recover(self) -> None:
        X = self.node_coords()
        u = self.gather_u()
        D = self.D()
        self.gp_stress = []
        self.gp_strain = []
        xi, eta, _ = gauss_legendre_2d_quad(self.quadrature)
        for q in range(xi.size):
            _, _, dN_dx = self.jacobian(float(xi[q]), float(eta[q]), X)
            B = self.B_matrix(dN_dx)
            eps = B @ u
            sig = D @ eps
            self.gp_strain.append(eps)
            self.gp_stress.append(sig)
