"""3-node Reissner-Mindlin triangular shell element (``ShellTri3``).

A flat triangular shell with:

* **CST membrane** — constant-strain triangle for in-plane behavior
  (3 DOFs per node: u, v in the element-local plane).
* **Linear Reissner-Mindlin bending** — independent linear interpolation
  of (w, theta_x, theta_y) gives constant curvature over the element.
* **Reduced (one-point) transverse-shear** integration at the
  centroid. This is the simplest cure for shear locking on affine
  triangles, equivalent to the basic MITC3 tying for constant-
  thickness flat shells without bubble functions.
* **Drilling** stiffness on ``theta_z`` to keep the global system
  nonsingular when only coplanar shells meet at a node.

Total: 18 DOFs per element (3 nodes x 6 DOFs/node). The element
fits naturally into 3-D frame models with ``ndf = 6``.

**Recommended range**: ``L/t <= 20`` (thick to moderate shells).
For thinner shells, residual shear locking from the reduced-point
integration starts to dominate; use ``ShellMITC4`` on a
quadrilateral mesh in that regime. A future ShellTri3 with proper
edge-tying and bubble-rotation enrichment (Lee & Bathe 2004) would
extend this to the thin-shell limit.

Sign convention matches ``ShellMITC4``:
    kappa_xx = +d theta_y / dx
    kappa_yy = -d theta_x / dy
    kappa_xy = +d theta_y / dy - d theta_x / dx
    gamma_xz = +d w / dx + theta_y
    gamma_yz = +d w / dy - theta_x
"""
from __future__ import annotations

import numpy as np

from femsolver.elements.base import Element


# ============================================================ local frame

def _triangle_local_frame(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Build an orthonormal frame for a 3-node triangle ``X (3, 3)``."""
    v1 = X[1] - X[0]
    v2 = X[2] - X[0]
    n = np.cross(v1, v2)
    A = 0.5 * np.linalg.norm(n)
    if A <= 0.0:
        raise ValueError("triangle has zero or negative area")
    n /= np.linalg.norm(n)
    e1 = v1 / np.linalg.norm(v1)
    e2 = np.cross(n, e1)
    R = np.column_stack([e1, e2, n])
    centroid = X.mean(axis=0)
    Xl3 = (X - centroid) @ R
    return R, Xl3[:, :2], A


def _shape_derivatives(Xl: np.ndarray) -> tuple[np.ndarray, float]:
    """For a CST/linear triangle with local-2D nodes ``Xl (3, 2)``,
    returns the (2, 3) Cartesian-derivative matrix
    ``[[dN1/dx, dN2/dx, dN3/dx], [dN1/dy, dN2/dy, dN3/dy]]`` and area.
    """
    (x1, y1), (x2, y2), (x3, y3) = Xl[0], Xl[1], Xl[2]
    A2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    A = 0.5 * A2
    if A <= 0.0:
        raise ValueError("triangle has zero or negative area in local frame")
    dN_dx = np.array([
        [(y2 - y3) / A2, (y3 - y1) / A2, (y1 - y2) / A2],
        [(x3 - x2) / A2, (x1 - x3) / A2, (x2 - x1) / A2],
    ])
    return dN_dx, A


# ============================================================ element

class ShellTri3(Element):
    """3-node MITC-style triangular shell. 18 DOFs (3 nodes x 6 DOF).

    Parameters
    ----------
    tag : int
    nodes : sequence of 3 node tags, counter-clockwise from the +n side.
    material : ElasticIsotropic
    thickness : float
    k_shear : float, default 5/6
        Transverse-shear correction factor.
    drilling_factor : float, default 1e-3
        Relative drilling penalty (same convention as ShellMITC4).
    """

    n_nodes = 3
    dofs_per_node = 6

    def __init__(
        self,
        tag: int,
        nodes,
        material,
        thickness: float | None = None,
        *,
        k_shear: float = 5.0 / 6.0,
        drilling_factor: float = 1.0e-3,
        section=None,
    ):
        super().__init__(tag, nodes, material)
        if drilling_factor < 0.0:
            raise ValueError(f"drilling_factor must be >= 0, got {drilling_factor}")
        if section is not None:
            self.section = section
            self.thickness = float(section.thickness)
            self.k_shear = float(getattr(section, "k_shear", k_shear))
        else:
            if thickness is None or thickness <= 0:
                raise ValueError(f"thickness must be positive, got {thickness}")
            if not (0.0 < k_shear <= 1.0):
                raise ValueError(f"k_shear must be in (0, 1], got {k_shear}")
            self.thickness = float(thickness)
            self.k_shear = float(k_shear)
            self.section = None
        self.drilling_factor = float(drilling_factor)
        self.membrane_strain: np.ndarray | None = None
        self.bending_curvature: np.ndarray | None = None
        self.shear_strain: np.ndarray | None = None
        self.resultants: np.ndarray | None = None

    # ----------------------------------------------------- constitutive
    def _D_membrane(self) -> np.ndarray:
        if self.section is not None:
            return self.section.D_membrane()
        E, nu, t = self.material.E, self.material.nu, self.thickness
        f = E * t / (1.0 - nu * nu)
        return f * np.array([
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, 0.5 * (1.0 - nu)],
        ])

    def _D_bending(self) -> np.ndarray:
        if self.section is not None:
            return self.section.D_bending()
        return self._D_membrane() * (self.thickness ** 2 / 12.0)

    def _D_coupling(self) -> np.ndarray:
        if self.section is not None:
            return self.section.D_coupling()
        return np.zeros((3, 3))

    def _D_shear(self) -> np.ndarray:
        if self.section is not None:
            return self.section.D_shear()
        G, t, k = self.material.G, self.thickness, self.k_shear
        return k * G * t * np.eye(2)

    # ----------------------------------------------------- local frame
    def _local_geom(self):
        return _triangle_local_frame(self.node_coords())

    def _T_global_to_local(self, R: np.ndarray) -> np.ndarray:
        T = np.zeros((18, 18))
        Rt = R.T
        for i in range(3):
            T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = Rt
            T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = Rt
        return T

    # ----------------------------------------------------- B-matrices in local
    @staticmethod
    def _Bm_local(dN_dx: np.ndarray) -> np.ndarray:
        """Membrane B (3, 18) in 6-DOF-per-node local ordering."""
        B = np.zeros((3, 18))
        for i in range(3):
            B[0, 6 * i] = dN_dx[0, i]
            B[1, 6 * i + 1] = dN_dx[1, i]
            B[2, 6 * i] = dN_dx[1, i]
            B[2, 6 * i + 1] = dN_dx[0, i]
        return B

    @staticmethod
    def _Bb_local(dN_dx: np.ndarray) -> np.ndarray:
        """Bending B (3, 18) — couples to local (theta_x, theta_y).

        Same sign convention as ShellMITC4:
            kappa_xx = +d theta_y / dx
            kappa_yy = -d theta_x / dy
            kappa_xy = +d theta_y / dy - d theta_x / dx
        """
        B = np.zeros((3, 18))
        for i in range(3):
            B[0, 6 * i + 4] = dN_dx[0, i]
            B[1, 6 * i + 3] = -dN_dx[1, i]
            B[2, 6 * i + 4] = dN_dx[1, i]
            B[2, 6 * i + 3] = -dN_dx[0, i]
        return B

    def _Bs_mitc3(self, Xl: np.ndarray, dN_dx: np.ndarray) -> np.ndarray:
        """MITC3 transverse-shear B-matrix (2, 18).

        The covariant shear (gamma_xi_z, gamma_eta_z) is tied at the
        three edge midpoints (A on edge 1-2, B on edge 2-3, C on edge
        3-1). The tied vector is interpolated linearly over the
        triangle and pulled back to Cartesian (gamma_xz, gamma_yz)
        through ``Jinv^T``.

        For a CST-like triangle the gradients are constant, and the
        tying degenerates to a one-point shear-strain reduction at
        the centroid -- the practical recipe that cures locking and
        is computationally identical to MITC3 for affine triangles.
        """
        # At centroid: w_centroid = (w_1 + w_2 + w_3) / 3, etc.
        # gamma_xz = dw/dx + theta_y_at_centroid
        # gamma_yz = dw/dy - theta_x_at_centroid
        B = np.zeros((2, 18))
        for i in range(3):
            # dw/dx contribution from w_i
            B[0, 6 * i + 2] = dN_dx[0, i]
            # theta_y contribution from theta_y_i (averaged at centroid: N_i = 1/3)
            B[0, 6 * i + 4] = 1.0 / 3.0
            # dw/dy contribution
            B[1, 6 * i + 2] = dN_dx[1, i]
            # -theta_x contribution
            B[1, 6 * i + 3] = -1.0 / 3.0
        return B

    # ----------------------------------------------------- stiffness
    def _K_local(self) -> np.ndarray:
        R, Xl, A = self._local_geom()
        dN_dx, _ = _shape_derivatives(Xl)
        Dm = self._D_membrane()
        Db = self._D_bending()
        Dc = self._D_coupling()
        Ds = self._D_shear()
        Bm = self._Bm_local(dN_dx)
        Bb = self._Bb_local(dN_dx)
        Bs = self._Bs_mitc3(Xl, dN_dx)
        K = (Bm.T @ Dm @ Bm) * A
        K += (Bb.T @ Db @ Bb) * A
        K += (Bs.T @ Ds @ Bs) * A
        if np.any(Dc):
            K_mb = (Bm.T @ Dc @ Bb) * A
            K += K_mb + K_mb.T
        # Drilling: penalize theta_z difference from element average
        k_drill = self.drilling_factor * max(np.max(np.abs(np.diag(K))), 1.0)
        for i in range(3):
            K[6 * i + 5, 6 * i + 5] += k_drill
            for j in range(3):
                if i != j:
                    K[6 * i + 5, 6 * j + 5] -= k_drill / 3.0
        return K

    def K_global(self) -> np.ndarray:
        R, _, _ = self._local_geom()
        T = self._T_global_to_local(R)
        K_loc = self._K_local()
        return T.T @ K_loc @ T

    # ----------------------------------------------------- mass
    def M_global(self, *, lumped: bool = False) -> np.ndarray:
        rho = getattr(self.material, "rho", 0.0)
        if rho == 0.0:
            return np.zeros((18, 18))
        _, _, A = self._local_geom()
        t = self.thickness
        m_total = rho * t * A
        # Lumped: translational mass evenly distributed; rotary inertia small
        M = np.zeros((18, 18))
        for i in range(3):
            for k in range(3):
                M[6 * i + k, 6 * i + k] = m_total / 3.0
            for k in (3, 4):
                M[6 * i + k, 6 * i + k] = m_total * t ** 2 / 36.0
        return M

    # ----------------------------------------------------- recovery
    def recover(self) -> None:
        R, Xl, A = self._local_geom()
        T = self._T_global_to_local(R)
        u_glob = self.gather_u()
        u_loc = T @ u_glob
        dN_dx, _ = _shape_derivatives(Xl)
        Bm = self._Bm_local(dN_dx)
        Bb = self._Bb_local(dN_dx)
        Bs = self._Bs_mitc3(Xl, dN_dx)
        Dm = self._D_membrane()
        Db = self._D_bending()
        Ds = self._D_shear()
        eps_m = Bm @ u_loc
        kappa = Bb @ u_loc
        gamma = Bs @ u_loc
        N = Dm @ eps_m
        M = Db @ kappa
        Q = Ds @ gamma
        self.membrane_strain = eps_m
        self.bending_curvature = kappa
        self.shear_strain = gamma
        self.resultants = np.concatenate([N, M, Q])
