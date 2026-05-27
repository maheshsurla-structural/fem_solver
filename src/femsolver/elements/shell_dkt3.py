"""3-node Discrete-Kirchhoff Triangle (``ShellDKT3``).

Phase 22.8 -- thin-plate triangular shell that complements
:class:`ShellTri3`. While ShellTri3 (Reissner-Mindlin triangle with
reduced one-point shear) handles thick plates (``L/t < 20``) but
suffers from residual shear locking for thinner shells, this
element uses the **discrete-Kirchhoff** approach (Batoz-Bathe-Ho
1980, "A study of three-node triangular plate bending elements"):
the rotation field is enriched with hierarchical edge-bubble modes
that are eliminated by enforcing zero transverse shear along each
edge. The result is a thin-plate element that gives the Kirchhoff
answer exactly at any thickness -- no shear locking by construction.

The DK formulation is the triangular sibling of :class:`ShellDKMQ4`
(Phase 22.7). The same convention applies:

    ψ_s_k = C_k θ_y - S_k θ_x       (effective tangential rotation
                                       in the Reissner-Mindlin
                                       shear γ_s_k = ∂w/∂s + ψ_s_k)

    Δψ_s_k = -(3 / (2 L_k)) (w_{j} - w_{i}) - (3/4) (ψ_s_i + ψ_s_j)

The bubble enrichment feeds back into rotations as

    θ_x += -S_k · N_(k+3) · Δψ_s_k
    θ_y += +C_k · N_(k+3) · Δψ_s_k

where N_(k+3) is the quadratic mid-edge "bubble" of a T6 serendipity
triangle (peaks at midpoint of edge k, vanishes at all 3 corners
and along the other 2 edges):

    N_3 = 4 L_0 L_1     (mid edge 0)
    N_4 = 4 L_1 L_2     (mid edge 1)
    N_5 = 4 L_2 L_0     (mid edge 2)

with L_0, L_1, L_2 the area coordinates.

Scope
-----
* **Thin plate only**: zero transverse-shear strain energy by
  construction. For thick plates (``L/t < 20``) where Mindlin shear
  matters, use :class:`ShellTri3` or :class:`ShellMITC4`.
* **Plate bending + membrane**: membrane stiffness uses the same
  CST B-matrix as ShellTri3; bending uses the DK rotation field.
* **Flat element**: a warped triangle has its three nodes projected
  onto the centroid plane.
"""
from __future__ import annotations

import numpy as np

from femsolver.elements.base import Element
from femsolver.elements.shell_tri import (
    _shape_derivatives,
    _triangle_local_frame,
)


# ============================================================ edge geometry

def _edge_geometry_tri(Xl: np.ndarray) -> tuple[np.ndarray, np.ndarray,
                                                  np.ndarray]:
    """For a triangle with local-2D node coords ``Xl (3, 2)``, return
    ``(L, C, S)`` arrays of length 3 where edge k goes from node k to
    node ``(k + 1) mod 3``.
    """
    L = np.empty(3)
    C = np.empty(3)
    S = np.empty(3)
    for k in range(3):
        i = k
        j = (k + 1) % 3
        dx = Xl[j, 0] - Xl[i, 0]
        dy = Xl[j, 1] - Xl[i, 1]
        Lk = float(np.hypot(dx, dy))
        if Lk <= 0.0:
            raise ValueError(
                f"ShellDKT3: zero-length edge {k} between nodes {i + 1} "
                f"and {j + 1}."
            )
        L[k] = Lk
        C[k] = dx / Lk
        S[k] = dy / Lk
    return L, C, S


def _dbs_matrix_tri(L: np.ndarray, C: np.ndarray,
                     S: np.ndarray) -> np.ndarray:
    """(3, 9) substitution matrix ``A`` such that
    ``[Δψ_s_0, Δψ_s_1, Δψ_s_2]^T = A @ d_pb`` where d_pb is the
    9-vector ``(w_0, θ_x_0, θ_y_0, w_1, θ_x_1, θ_y_1, w_2, θ_x_2, θ_y_2)``.

    Discrete-Kirchhoff substitution (same as DKMQ4 but for 3 edges):

        Δψ_s_k = -(3 / (2 L_k)) (w_j - w_i) - (3/4) (ψ_s_i + ψ_s_j)

    with ``ψ_s_n = C_k θ_y_n - S_k θ_x_n`` (matches MITC4's mixed
    rotation convention).
    """
    A = np.zeros((3, 9))
    for k in range(3):
        i = k
        j = (k + 1) % 3
        # w_i, w_j
        A[k, 3 * i]     += +1.5 / L[k]
        A[k, 3 * j]     += -1.5 / L[k]
        # θ_x and θ_y coefficients
        A[k, 3 * i + 1] += +0.75 * S[k]      # = -(3/4) * (-S_k)
        A[k, 3 * i + 2] += -0.75 * C[k]
        A[k, 3 * j + 1] += +0.75 * S[k]
        A[k, 3 * j + 2] += -0.75 * C[k]
    return A


# ============================================================ area-coord helpers

#: 3-point Hammer quadrature on the unit triangle (area = 1/2).
#: Points are the three edge midpoints in area coords ``(L_0, L_1, L_2)``;
#: each weight is ``1/3 * triangle_area``. We store weights as 1/3 and
#: multiply by triangle area at integration time.
_TRI_GAUSS_AREA_COORDS = np.array([
    [0.5, 0.5, 0.0],     # midpoint of edge 0 (between nodes 0,1)
    [0.0, 0.5, 0.5],     # midpoint of edge 1 (between nodes 1,2)
    [0.5, 0.0, 0.5],     # midpoint of edge 2 (between nodes 2,0)
])
_TRI_GAUSS_WEIGHTS = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])


def _bubble_values_and_grads(L_area: np.ndarray, dN_dx: np.ndarray,
                              Xl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For a Gauss point with area coords ``L_area = (L_0, L_1, L_2)``,
    return:

    * ``N3`` shape (3,): values of the mid-edge bubble functions
        ``N_3 = 4 L_0 L_1``, ``N_4 = 4 L_1 L_2``, ``N_5 = 4 L_2 L_0``.
    * ``dN3_dx`` shape (2, 3): their Cartesian-coordinate derivatives.

    Since ``L_i`` has constant Cartesian gradient ``∂L_i/∂x_j = dN_dx[j, i]``
    (because L_i are the area coords, identical to linear shape functions),
    we have

        ∂(L_a L_b)/∂x = (∂L_a/∂x) L_b + L_a (∂L_b/∂x)
    """
    L0, L1, L2 = L_area
    N3 = np.array([4.0 * L0 * L1, 4.0 * L1 * L2, 4.0 * L2 * L0])

    # ∂L_i/∂x = dN_dx[0, i], ∂L_i/∂y = dN_dx[1, i]
    dN3_dx = np.zeros((2, 3))
    # N_3 = 4 L_0 L_1
    dN3_dx[0, 0] = 4.0 * (dN_dx[0, 0] * L1 + L0 * dN_dx[0, 1])
    dN3_dx[1, 0] = 4.0 * (dN_dx[1, 0] * L1 + L0 * dN_dx[1, 1])
    # N_4 = 4 L_1 L_2
    dN3_dx[0, 1] = 4.0 * (dN_dx[0, 1] * L2 + L1 * dN_dx[0, 2])
    dN3_dx[1, 1] = 4.0 * (dN_dx[1, 1] * L2 + L1 * dN_dx[1, 2])
    # N_5 = 4 L_2 L_0
    dN3_dx[0, 2] = 4.0 * (dN_dx[0, 2] * L0 + L2 * dN_dx[0, 0])
    dN3_dx[1, 2] = 4.0 * (dN_dx[1, 2] * L0 + L2 * dN_dx[1, 0])
    return N3, dN3_dx


# ============================================================ element

class ShellDKT3(Element):
    """3-node Discrete-Kirchhoff triangular plate element.

    Parameters
    ----------
    tag : int
    nodes : sequence of 3 node tags, CCW from the +normal side.
    material : ElasticIsotropic
    thickness : float
    drilling_factor : float, default 1e-3

    Notes
    -----
    * **Thin-plate only**: zero transverse shear by construction. For
      thick plates use :class:`ShellTri3` or :class:`ShellMITC4`.
    * 6 DOFs per node (u, v, w, θ_x, θ_y, θ_z); 18 total.
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
        drilling_factor: float = 1.0e-3,
    ):
        super().__init__(tag, nodes, material)
        if drilling_factor < 0.0:
            raise ValueError(
                f"drilling_factor must be >= 0, got {drilling_factor}"
            )
        if thickness is None or thickness <= 0:
            raise ValueError(f"thickness must be positive, got {thickness}")
        self.thickness = float(thickness)
        self.drilling_factor = float(drilling_factor)
        # Recovery buffers
        self.gp_membrane_strain: list[np.ndarray] = []
        self.gp_bending_curvature: list[np.ndarray] = []
        self.gp_resultants: list[np.ndarray] = []

    # ----------------------------------------------------- constitutive
    def _D_membrane(self) -> np.ndarray:
        E, nu, t = self.material.E, self.material.nu, self.thickness
        f = E * t / (1.0 - nu * nu)
        return f * np.array([
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, 0.5 * (1.0 - nu)],
        ])

    def _D_bending(self) -> np.ndarray:
        return self._D_membrane() * (self.thickness ** 2 / 12.0)

    # ----------------------------------------------------- local-global
    def _local_geom(self):
        return _triangle_local_frame(self.node_coords())

    def _T_global_to_local(self, R: np.ndarray) -> np.ndarray:
        T = np.zeros((18, 18))
        Rt = R.T
        for i in range(3):
            T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = Rt
            T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = Rt
        return T

    # ----------------------------------------------------- B-matrices
    @staticmethod
    def _Bm_local(dN_dx: np.ndarray) -> np.ndarray:
        """Membrane B (3, 18). Same form as ShellTri3."""
        B = np.zeros((3, 18))
        for i in range(3):
            B[0, 6 * i] = dN_dx[0, i]
            B[1, 6 * i + 1] = dN_dx[1, i]
            B[2, 6 * i] = dN_dx[1, i]
            B[2, 6 * i + 1] = dN_dx[0, i]
        return B

    def _Bb_dk_local(self, L_area: np.ndarray, dN_dx: np.ndarray,
                       Xl: np.ndarray, L_edges: np.ndarray,
                       C: np.ndarray, S: np.ndarray,
                       A_dbs: np.ndarray) -> np.ndarray:
        """Bending B (3, 9) at a Gauss point with area coords
        ``L_area``, mapping the 9 plate-bending DOFs d_pb to
        curvatures (κ_x, κ_y, κ_xy).

        The rotation field is

            θ_x = Σ_i N_i θ_x_i + Σ_k N_(k+3) (-S_k) Δψ_s_k
            θ_y = Σ_i N_i θ_y_i + Σ_k N_(k+3) (+C_k) Δψ_s_k

        For a CST (linear N_i) the corner-node shape derivatives are
        constants -- ``dN_dx`` -- so the bilinear contributions to
        ∂θ/∂x are also constants (independent of L_area). The
        hierarchical contributions vary with L_area through dN3_dx.
        """
        _, dN3_dx = _bubble_values_and_grads(L_area, dN_dx, Xl)

        # Build 9-vector rows for each rotation derivative
        dbxdx = np.zeros(9)
        dbxdy = np.zeros(9)
        dbydx = np.zeros(9)
        dbydy = np.zeros(9)
        for i in range(3):
            dbxdx[3 * i + 1] = dN_dx[0, i]
            dbxdy[3 * i + 1] = dN_dx[1, i]
            dbydx[3 * i + 2] = dN_dx[0, i]
            dbydy[3 * i + 2] = dN_dx[1, i]
        # Hierarchical contributions:
        # θ_x_bubble = Σ_k (-S_k) N_(k+3) Δψ_s_k
        # θ_y_bubble = Σ_k (+C_k) N_(k+3) Δψ_s_k
        # so ∂(θ_x_bubble)/∂x = Σ_k (-S_k) ∂N_(k+3)/∂x Δψ_s_k
        # = Σ_k (-S_k) dN3_dx[0, k] * A_dbs[k]  (acting on d_pb)
        for k in range(3):
            dbxdx += dN3_dx[0, k] * (-S[k]) * A_dbs[k]
            dbxdy += dN3_dx[1, k] * (-S[k]) * A_dbs[k]
            dbydx += dN3_dx[0, k] * (+C[k]) * A_dbs[k]
            dbydy += dN3_dx[1, k] * (+C[k]) * A_dbs[k]
        Bb = np.zeros((3, 9))
        Bb[0, :] = +dbydx                # κ_x  = +∂θ_y/∂x
        Bb[1, :] = -dbxdy                # κ_y  = -∂θ_x/∂y
        Bb[2, :] = +dbydy - dbxdx        # κ_xy = +∂θ_y/∂y - ∂θ_x/∂x
        return Bb

    # ----------------------------------------------------- stiffness assembly
    def _K_local(self) -> np.ndarray:
        """18×18 stiffness in 6-DOF/node local ordering."""
        R, Xl, A = self._local_geom()
        dN_dx, _A_local = _shape_derivatives(Xl)
        L_edges, C, S = _edge_geometry_tri(Xl)
        A_dbs = _dbs_matrix_tri(L_edges, C, S)
        Dm = self._D_membrane()
        Db = self._D_bending()

        # Permutation matrix P: 9 plate-bending DOFs -> 18 full local
        # vector. d_pb = P @ d18 where d18 has 6-DOF/node ordering and
        # d_pb has (w, θ_x, θ_y) per node.
        P = np.zeros((9, 18))
        for i in range(3):
            P[3 * i,     6 * i + 2] = 1.0    # w
            P[3 * i + 1, 6 * i + 3] = 1.0    # θ_x
            P[3 * i + 2, 6 * i + 4] = 1.0    # θ_y

        K = np.zeros((18, 18))

        # Membrane: constant B over the triangle, single-point integration
        Bm = self._Bm_local(dN_dx)
        K += (Bm.T @ Dm @ Bm) * A

        # Bending: 3-point Hammer over the DK B-matrix
        for q in range(3):
            L_area = _TRI_GAUSS_AREA_COORDS[q]
            wq = _TRI_GAUSS_WEIGHTS[q]
            Bb_pb = self._Bb_dk_local(L_area, dN_dx, Xl, L_edges, C, S, A_dbs)
            Bb18 = Bb_pb @ P
            K += (Bb18.T @ Db @ Bb18) * (A * wq)

        # Drilling stiffness (relative penalty, same convention as Tri3/MITC4)
        k_drill = self.drilling_factor * max(
            float(np.max(np.abs(np.diag(K)))), 1.0
        )
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
        R, _, A = self._local_geom()
        t = self.thickness
        m_total = rho * t * A
        # Equal-share lump (consistent for symmetric CST triangle is
        # nearly identical for transitional DOFs; rotary terms are
        # small).
        M = np.zeros((18, 18))
        for i in range(3):
            for k in range(3):
                M[6 * i + k, 6 * i + k] = m_total / 3.0
            for k in (3, 4):
                M[6 * i + k, 6 * i + k] = m_total * t ** 2 / 36.0
        if lumped:
            return M
        # For consistent we'd build via shape-function products; equal-
        # share lump is sufficient for typical use and matches Tri3.
        T = self._T_global_to_local(R)
        return T.T @ M @ T

    # ----------------------------------------------------- recovery
    def recover(self) -> None:
        R, Xl, A = self._local_geom()
        T = self._T_global_to_local(R)
        u_glob = self.gather_u()
        u_loc = T @ u_glob
        dN_dx, _ = _shape_derivatives(Xl)
        L_edges, C, S = _edge_geometry_tri(Xl)
        A_dbs = _dbs_matrix_tri(L_edges, C, S)
        P = np.zeros((9, 18))
        for i in range(3):
            P[3 * i,     6 * i + 2] = 1.0
            P[3 * i + 1, 6 * i + 3] = 1.0
            P[3 * i + 2, 6 * i + 4] = 1.0
        Dm = self._D_membrane()
        Db = self._D_bending()
        self.gp_membrane_strain = []
        self.gp_bending_curvature = []
        self.gp_resultants = []
        Bm = self._Bm_local(dN_dx)
        eps_m = Bm @ u_loc
        # Sample at the 3 Hammer points
        for q in range(3):
            L_area = _TRI_GAUSS_AREA_COORDS[q]
            Bb_pb = self._Bb_dk_local(L_area, dN_dx, Xl, L_edges, C, S, A_dbs)
            Bb18 = Bb_pb @ P
            kappa = Bb18 @ u_loc
            self.gp_membrane_strain.append(eps_m)
            self.gp_bending_curvature.append(kappa)
            self.gp_resultants.append(
                np.concatenate([Dm @ eps_m, Db @ kappa])
            )
