"""Quadrilateral plane elements: Q4 (bilinear) and Q8 (serendipity quadratic).

Both support plane-stress and plane-strain via the material's
``D_plane_stress()`` / ``D_plane_strain()`` returns.

Quad8 (8-node serendipity)
--------------------------

Node numbering -- corners first, then mid-sides (CCW)::

    4 ----7---- 3
    |           |
    8           6
    |           |
    1 ----5---- 2

Default quadrature is 3x3 Gauss (exact for the quadratic shape
functions). Reduced 2x2 Gauss is supported for selective integration.
"""
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


# ============================================================ Quad8

# Corner master coords (xi, eta) for nodes 1..4 and mid-sides for 5..8
_Q8_XI = np.array([-1.0,  1.0, 1.0, -1.0,  0.0, 1.0, 0.0, -1.0])
_Q8_ETA = np.array([-1.0, -1.0, 1.0,  1.0, -1.0, 0.0, 1.0,  0.0])


class Quad8(Element):
    """8-node serendipity quadrilateral, plane stress or plane strain.

    Parameters
    ----------
    tag : int
    nodes : (8,) sequence of node tags. Corners 1-4 CCW, then mid-sides
        5 (bottom), 6 (right), 7 (top), 8 (left).
    material : Material
    thickness : float, default 1.0
    state : {"plane_stress", "plane_strain"}, default "plane_stress"
    quadrature : int, default 3
        3 = 3x3 Gauss (full, recommended); 2 = 2x2 (reduced — risk of
        spurious modes).
    """

    n_nodes = 8
    dofs_per_node = 2

    def __init__(
        self,
        tag: int,
        nodes,
        material,
        thickness: float = 1.0,
        state: str = "plane_stress",
        quadrature: int = 3,
    ):
        super().__init__(tag, nodes, material)
        if thickness <= 0:
            raise ValueError(f"thickness must be positive, got {thickness}")
        if state not in ("plane_stress", "plane_strain"):
            raise ValueError(
                f"state must be plane_stress or plane_strain, got {state}"
            )
        self.thickness = float(thickness)
        self.state = state
        self.quadrature = int(quadrature)
        self.gp_stress: list[np.ndarray] = []
        self.gp_strain: list[np.ndarray] = []
        self._body_force: np.ndarray = np.zeros(2)

    @staticmethod
    def shape_functions(xi: float, eta: float) -> np.ndarray:
        """Serendipity Q8 shape functions, shape (8,)."""
        N = np.empty(8)
        # Corner nodes
        for a in range(4):
            xa, ea = _Q8_XI[a], _Q8_ETA[a]
            N[a] = 0.25 * (1.0 + xi * xa) * (1.0 + eta * ea) \
                   * (xi * xa + eta * ea - 1.0)
        # Mid-side: nodes 5 (eta=-1) and 7 (eta=+1) have xi_a = 0
        for a in (4, 6):
            ea = _Q8_ETA[a]
            N[a] = 0.5 * (1.0 - xi * xi) * (1.0 + eta * ea)
        # Mid-side: nodes 6 (xi=+1) and 8 (xi=-1) have eta_a = 0
        for a in (5, 7):
            xa = _Q8_XI[a]
            N[a] = 0.5 * (1.0 - eta * eta) * (1.0 + xi * xa)
        return N

    @staticmethod
    def dN_dxi(xi: float, eta: float) -> np.ndarray:
        """dN/d(xi,eta) as (2, 8): row 0 = dN/dxi, row 1 = dN/deta."""
        dN = np.zeros((2, 8))
        # Corners
        for a in range(4):
            xa, ea = _Q8_XI[a], _Q8_ETA[a]
            dN[0, a] = 0.25 * xa * (1.0 + eta * ea) * (2.0 * xi * xa + eta * ea)
            dN[1, a] = 0.25 * ea * (1.0 + xi * xa) * (xi * xa + 2.0 * eta * ea)
        # Mid-side on eta = ±1 (xi_a = 0)
        for a in (4, 6):
            ea = _Q8_ETA[a]
            dN[0, a] = -xi * (1.0 + eta * ea)
            dN[1, a] = 0.5 * ea * (1.0 - xi * xi)
        # Mid-side on xi = ±1 (eta_a = 0)
        for a in (5, 7):
            xa = _Q8_XI[a]
            dN[0, a] = 0.5 * xa * (1.0 - eta * eta)
            dN[1, a] = -eta * (1.0 + xi * xa)
        return dN

    def D(self) -> np.ndarray:
        if self.state == "plane_stress":
            return self.material.D_plane_stress()
        return self.material.D_plane_strain()

    def jacobian(self, xi: float, eta: float, X: np.ndarray):
        dN = self.dN_dxi(xi, eta)         # (2, 8)
        J = dN @ X                         # (2, 2)
        detJ = float(np.linalg.det(J))
        if detJ <= 0.0:
            raise ValueError(
                f"Quad8 element {self.tag}: non-positive Jacobian "
                f"({detJ:g}) at xi={xi}, eta={eta}. Check node ordering "
                f"(corners CCW, then mid-sides 5-8)."
            )
        Jinv = np.linalg.inv(J)
        dN_dx = Jinv @ dN                  # (2, 8)
        return J, detJ, dN_dx

    def B_matrix(self, dN_dx: np.ndarray) -> np.ndarray:
        """Strain-displacement B matrix, shape (3, 16). Voigt
        [exx, eyy, gxy]."""
        B = np.zeros((3, 16))
        for i in range(8):
            dNx = dN_dx[0, i]
            dNy = dN_dx[1, i]
            B[0, 2 * i] = dNx
            B[1, 2 * i + 1] = dNy
            B[2, 2 * i] = dNy
            B[2, 2 * i + 1] = dNx
        return B

    def K_global(self) -> np.ndarray:
        X = self.node_coords()             # (8, 2)
        D = self.D()
        t = self.thickness
        K = np.zeros((16, 16))
        xi, eta, w = gauss_legendre_2d_quad(self.quadrature)
        for q in range(xi.size):
            _, detJ, dN_dx = self.jacobian(float(xi[q]), float(eta[q]), X)
            B = self.B_matrix(dN_dx)
            K += (B.T @ D @ B) * (t * detJ * w[q])
        return K

    def M_global(self, *, lumped: bool = False) -> np.ndarray:
        rho = self.material.rho
        if rho == 0.0:
            return np.zeros((16, 16))
        X = self.node_coords()
        t = self.thickness
        M = np.zeros((16, 16))
        xi, eta, w = gauss_legendre_2d_quad(self.quadrature)
        for q in range(xi.size):
            N = self.shape_functions(float(xi[q]), float(eta[q]))
            _, detJ, _ = self.jacobian(float(xi[q]), float(eta[q]), X)
            jw = rho * t * detJ * w[q]
            Nbar = np.zeros((2, 16))
            for i in range(8):
                Nbar[0, 2 * i] = N[i]
                Nbar[1, 2 * i + 1] = N[i]
            M += (Nbar.T @ Nbar) * jw
        if lumped:
            # HRZ row-sum for Q8: scale diagonal so trace matches total
            # translational mass per direction.
            diag = np.diag(M).copy()
            row_sums = M.sum(axis=1)
            total_mass = float(np.sum(row_sums[::2]))   # x DOFs
            # Use diagonal entries as proportional weights
            diag_x = diag[::2]
            scale = total_mass / float(np.sum(diag_x)) if diag_x.sum() > 0 else 0.0
            lumped_diag = diag.copy()
            lumped_diag[::2] *= scale
            lumped_diag[1::2] *= scale
            return np.diag(lumped_diag)
        return M

    # ---------------------------------------------------------------- loading
    def set_body_force(self, bx: float, by: float) -> None:
        self._body_force = np.array([float(bx), float(by)])

    def clear_distributed_loads(self) -> None:
        self._body_force = np.zeros(2)

    def f_eq_global(self) -> np.ndarray:
        if not np.any(self._body_force):
            return np.zeros(16)
        X = self.node_coords()
        f = np.zeros(16)
        bx, by = self._body_force
        t = self.thickness
        xi, eta, w = gauss_legendre_2d_quad(self.quadrature)
        for q in range(xi.size):
            N = self.shape_functions(float(xi[q]), float(eta[q]))
            _, detJ, _ = self.jacobian(float(xi[q]), float(eta[q]), X)
            jw = t * detJ * w[q]
            for i in range(8):
                f[2 * i]     += N[i] * bx * jw
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
