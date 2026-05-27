"""4-node Discrete Kirchhoff plate element (``ShellDKMQ4``).

Phase 22.7 -- thin-plate Discrete Kirchhoff Quadrilateral
(Batoz-Tahar 1982).

Formulation
-----------
The rotation field is enriched with hierarchical edge-bubble modes
parametrized by 4 hierarchical degrees of freedom ``Δψ_s_k`` (one
per edge), eliminated by enforcing **zero transverse shear along
each edge** (discrete-Kirchhoff constraint):

    Δψ_s_k = -(3 / (2 L_k)) (w_{k+1} - w_k) - (3 / 4) (ψ_s_k + ψ_s_{k+1})

where ``ψ_s_n = C_k θ_y_n - S_k θ_x_n`` is the **effective**
tangential rotation in the Reissner-Mindlin tangential shear
``γ_s_k = (C_k ∂w/∂x + S_k ∂w/∂y) + ψ_s_k``. This follows the mixed
rotation convention of :class:`ShellMITC4` (γ_xz = ∂w/∂x + θ_y,
γ_yz = ∂w/∂y - θ_x), in which θ_y = -∂w/∂x and θ_x = +∂w/∂y in
the Kirchhoff limit. The result is a thin-plate element whose
stiffness matrix involves only the bending strain-displacement
matrix -- there is no separate transverse-shear stiffness
contribution.

After the substitution, the rotation field at any interior point
of the element is

    θ_x(ξ, η) = Σ N_i(ξ,η) θ_x_i + Σ_k N_(k+4)(ξ,η) (-S_k) Δψ_s_k
    θ_y(ξ, η) = Σ N_i(ξ,η) θ_y_i + Σ_k N_(k+4)(ξ,η) (+C_k) Δψ_s_k

so that the bubble enrichment of ψ_s by Δψ_s_k along each edge is
recovered with ``C_k θ_y_new - S_k θ_x_new = N_(k+4) Δψ_s_k``.

Curvatures use the standard Reissner-Mindlin form

    κ_x  = +∂θ_y/∂x
    κ_y  = -∂θ_x/∂y
    κ_xy = +∂θ_y/∂y - ∂θ_x/∂x

inherited from :class:`ShellMITC4`.

Scope and limitations
---------------------
* **Thin plate only.** This element omits transverse-shear strain
  energy by construction. For thick plates (``L/t < 20``) or where
  Mindlin shear contributions matter, use :class:`ShellMITC4`. A
  future enhancement could add the Katili (1993) Mindlin
  substitute-shear field to produce the full **DKMQ4** of
  Batoz & Katili 1992.
* **Plate bending only.** Membrane stiffness uses the same bilinear
  Q4 B-matrix as :class:`ShellMITC4`; membrane and bending decouple
  for a flat symmetric element.
* **Flat element.** Like MITC4, a warped quad is projected onto
  the element-centroid plane.

Naming -- the class is named ``ShellDKMQ4`` for API consistency
with the project's task labelling, although strictly this is a DKQ
(thin-plate) rather than the full Mindlin-shear DKMQ. Future work
can extend to DKMQ proper.
"""
from __future__ import annotations

import numpy as np

from femsolver.elements.base import Element
from femsolver.numerics.quadrature import gauss_legendre_2d_quad


# ============================================================ shape functions

def _N4(xi: float, eta: float) -> np.ndarray:
    """Bilinear Q4 Lagrange shape functions at (ξ, η)."""
    return 0.25 * np.array([
        (1.0 - xi) * (1.0 - eta),
        (1.0 + xi) * (1.0 - eta),
        (1.0 + xi) * (1.0 + eta),
        (1.0 - xi) * (1.0 + eta),
    ])


def _dN4_dxi(xi: float, eta: float) -> np.ndarray:
    """Returns (2, 4): row 0 = dN/dξ, row 1 = dN/dη."""
    return 0.25 * np.array([
        [-(1.0 - eta),  (1.0 - eta),  (1.0 + eta), -(1.0 + eta)],
        [-(1.0 - xi),  -(1.0 + xi),   (1.0 + xi),   (1.0 - xi)],
    ])


def _N8_mid(xi: float, eta: float) -> np.ndarray:
    """Q8-serendipity mid-edge shape functions at (ξ, η).

    Returns (4,) array: [N_5, N_6, N_7, N_8] for edges 1, 2, 3, 4.
    N_(k+4) is nonzero on edge k only (vanishes on the other 3 edges).
    These are the hierarchical edge-bubble modes used by DK to enrich
    the rotation field.
    """
    return np.array([
        0.5 * (1.0 - xi * xi) * (1.0 - eta),     # edge 1 (η=-1)
        0.5 * (1.0 + xi) * (1.0 - eta * eta),    # edge 2 (ξ=+1)
        0.5 * (1.0 - xi * xi) * (1.0 + eta),     # edge 3 (η=+1)
        0.5 * (1.0 - xi) * (1.0 - eta * eta),    # edge 4 (ξ=-1)
    ])


def _dN8_mid_dxi(xi: float, eta: float) -> np.ndarray:
    """(2, 4) derivative array: row 0 = ∂N_(k+4)/∂ξ, row 1 = ∂N_(k+4)/∂η,
    for k = 1, 2, 3, 4."""
    return np.array([
        # ∂/∂ξ
        [-xi * (1.0 - eta),          0.5 * (1.0 - eta * eta),
         -xi * (1.0 + eta),         -0.5 * (1.0 - eta * eta)],
        # ∂/∂η
        [-0.5 * (1.0 - xi * xi),    -eta * (1.0 + xi),
          0.5 * (1.0 - xi * xi),    -eta * (1.0 - xi)],
    ])


# ============================================================ local frame

def _local_frame_q4(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build an orthonormal local frame at the element centroid for a
    Q4. Returns (R, X_local2d).
    """
    r1 = 0.5 * ((X[1] - X[0]) + (X[2] - X[3]))
    r2 = 0.5 * ((X[3] - X[0]) + (X[2] - X[1]))
    n = np.cross(r1, r2)
    n /= np.linalg.norm(n)
    e1 = r1 / np.linalg.norm(r1)
    e2 = np.cross(n, e1)
    R = np.column_stack([e1, e2, n])
    centroid = X.mean(axis=0)
    Xc = X - centroid
    X_local_3 = Xc @ R
    return R, X_local_3[:, :2]


def _jacobian2d_4(xi: float, eta: float, Xl: np.ndarray):
    dN = _dN4_dxi(xi, eta)
    J = dN @ Xl
    detJ = float(np.linalg.det(J))
    if detJ <= 0.0:
        raise ValueError(
            f"ShellDKMQ4: non-positive Jacobian determinant ({detJ:g}) "
            f"at (ξ={xi:g}, η={eta:g}). Check node ordering (CCW)."
        )
    Jinv = np.linalg.inv(J)
    dN_dx = Jinv @ dN
    return J, detJ, dN_dx, Jinv


# ============================================================ edge geometry

def _edge_geometry(Xl: np.ndarray) -> tuple[np.ndarray, np.ndarray,
                                              np.ndarray]:
    """Compute (L_k, C_k = cos α_k, S_k = sin α_k) for each of the 4 edges.

    Edge k connects nodes (k-1) and (k mod 4) — i.e.:
        edge 1: node 0 -> node 1
        edge 2: node 1 -> node 2
        edge 3: node 2 -> node 3
        edge 4: node 3 -> node 0
    """
    L = np.empty(4)
    C = np.empty(4)
    S = np.empty(4)
    for k in range(4):
        i = k
        j = (k + 1) % 4
        dx = Xl[j, 0] - Xl[i, 0]
        dy = Xl[j, 1] - Xl[i, 1]
        Lk = float(np.hypot(dx, dy))
        if Lk <= 0.0:
            raise ValueError(
                f"ShellDKMQ4: zero-length edge {k + 1} (nodes "
                f"{i + 1}, {j + 1})."
            )
        L[k] = Lk
        C[k] = dx / Lk
        S[k] = dy / Lk
    return L, C, S


# ============================================================ DK substitution

def _dbs_matrix(L: np.ndarray, C: np.ndarray,
                 S: np.ndarray) -> np.ndarray:
    """Build the (4, 12) matrix ``A`` such that
    ``[Δψ_s_1, Δψ_s_2, Δψ_s_3, Δψ_s_4]^T = A @ d``,
    where ``d`` is the 12-vector of nodal DOFs in order
    ``(w_1, θ_x_1, θ_y_1, w_2, θ_x_2, θ_y_2, w_3, θ_x_3, θ_y_3,
    w_4, θ_x_4, θ_y_4)``.

    Discrete-Kirchhoff constraint on edge k (zero shear):

        γ̄_s_k = 0
        ⇒  Δψ_s_k = -(3 / (2 L_k)) (w_{j} - w_{i})
                    -(3 / 4) (ψ_s_i + ψ_s_j)

    where ``ψ_s_n = C_k θ_y_n - S_k θ_x_n`` -- the "effective
    tangential rotation" in the Reissner-Mindlin tangential shear

        γ_s_k = C_k γ_xz + S_k γ_yz
              = (C_k ∂w/∂x + S_k ∂w/∂y)
                + (C_k θ_y - S_k θ_x)

    matching the MITC4 mixed-rotation convention
    (γ_xz = ∂w/∂x + θ_y, γ_yz = ∂w/∂y - θ_x).
    """
    A = np.zeros((4, 12))
    for k in range(4):
        i = k                    # first endpoint
        j = (k + 1) % 4          # second endpoint
        # w coefficients
        A[k, 3 * i]     += +1.5 / L[k]
        A[k, 3 * j]     += -1.5 / L[k]
        # ψ_s = C θ_y - S θ_x : θ_x slot in d_pb is 3 n + 1, θ_y is 3 n + 2
        A[k, 3 * i + 1] += +0.75 * S[k]    # -0.75 * (-S_k)
        A[k, 3 * i + 2] += -0.75 * C[k]
        A[k, 3 * j + 1] += +0.75 * S[k]
        A[k, 3 * j + 2] += -0.75 * C[k]
    return A


# ============================================================ element

class ShellDKMQ4(Element):
    """4-node Discrete-Kirchhoff plate element (thin-plate DKQ).

    Parameters
    ----------
    tag : int
    nodes : (4,) sequence of node tags (CCW from the +normal side).
    material : ElasticIsotropic
    thickness : float
    drilling_factor : float, default 1e-3
        Fictitious θ_z stiffness fraction (analogous to MITC4) to
        prevent global singularity at nodes where only coplanar
        shells meet.

    Notes
    -----
    * **Thin-plate only**: zero transverse shear enforced at edges.
      For thick plates use :class:`ShellMITC4`.
    * 6 DOFs per node (u, v, w, θ_x, θ_y, θ_z) -- 24 DOFs total per
      element. Membrane stiffness is the same bilinear Q4 form used
      by :class:`ShellMITC4`; bending uses the DK rotation field
      with 3x3 Gauss integration.
    """

    n_nodes = 4
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
        self.gp_bending_curvature: list[np.ndarray] = []
        self.gp_membrane_strain: list[np.ndarray] = []
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
        X = self.node_coords()
        return _local_frame_q4(X)

    def _T_global_to_local(self, R: np.ndarray) -> np.ndarray:
        T = np.zeros((24, 24))
        Rt = R.T
        for i in range(4):
            T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = Rt
            T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = Rt
        return T

    # ----------------------------------------------------- B matrices
    @staticmethod
    def _Bm_local(dN_dx: np.ndarray) -> np.ndarray:
        """Membrane B (3, 20) in 5-DOF/node local ordering (u,v,w,θx,θy).
        Identical to MITC4's membrane B."""
        B = np.zeros((3, 20))
        for i in range(4):
            B[0, 5 * i] = dN_dx[0, i]
            B[1, 5 * i + 1] = dN_dx[1, i]
            B[2, 5 * i] = dN_dx[1, i]
            B[2, 5 * i + 1] = dN_dx[0, i]
        return B

    def _Bb_dk_local(self, xi: float, eta: float, Xl: np.ndarray,
                       L: np.ndarray, C: np.ndarray, S: np.ndarray,
                       A_dbs: np.ndarray) -> np.ndarray:
        """Bending B matrix at (ξ, η) for the DK rotation field.

        Returns (3, 12), mapping the 12 plate-bending DOFs ``d_pb``
        in order ``(w_i, β_x_i, β_y_i)`` for i=1..4 to the bending
        curvatures ``(κ_x, κ_y, κ_xy)``.

        The rotations are:
            β_x = Σ_i N_i β_x_i + Σ_k N_(k+4) Δβ_s_k C_k
            β_y = Σ_i N_i β_y_i + Σ_k N_(k+4) Δβ_s_k S_k

        With Δβ_s = A_dbs @ d_pb, the rotation gradients become:
            ∂β_x/∂x = Σ_i ∂N_i/∂x β_x_i + Σ_k ∂N_(k+4)/∂x Δβ_s_k C_k
            ...

        Curvatures (Reissner-Mindlin form):
            κ_x  = +∂β_y/∂x
            κ_y  = -∂β_x/∂y
            κ_xy = +∂β_y/∂y - ∂β_x/∂x
        """
        _, _, dN_dx, _ = _jacobian2d_4(xi, eta, Xl)         # (2, 4)
        dN8_xi = _dN8_mid_dxi(xi, eta)                        # (2, 4)
        # Push dN8_xi from natural to local-Cartesian via Jinv
        J = _dN4_dxi(xi, eta) @ Xl
        Jinv = np.linalg.inv(J)
        dN8_dx = Jinv @ dN8_xi                                 # (2, 4)

        # Bilinear contribution to ∂β_x/∂x etc.:
        # β_x = Σ_i N_i β_x_i  → ∂β_x/∂x = Σ_i dN_i/dx β_x_i
        # In the 12-DOF d_pb vector, β_x_i lives at index 3 i + 1.
        # We build separate (12,) row vectors for each of the four
        # rotation derivatives.
        dbxdx = np.zeros(12)
        dbxdy = np.zeros(12)
        dbydx = np.zeros(12)
        dbydy = np.zeros(12)
        for i in range(4):
            dbxdx[3 * i + 1] = dN_dx[0, i]
            dbxdy[3 * i + 1] = dN_dx[1, i]
            dbydx[3 * i + 2] = dN_dx[0, i]
            dbydy[3 * i + 2] = dN_dx[1, i]
        # Add hierarchical contributions: for each edge k, the row of
        # A_dbs[k, :] linearly maps d_pb to Δψ_s_k. The bubble enriches
        # the rotation field so that the *effective tangential rotation*
        # (C_k θ_y - S_k θ_x) picks up an additional N_(k+4) Δψ_s_k. To
        # achieve that with rotations (θ_x, θ_y), we add:
        #     θ_x += -S_k · N_(k+4) · Δψ_s_k
        #     θ_y += +C_k · N_(k+4) · Δψ_s_k
        # (so that C_k θ_y_new - S_k θ_x_new = (C_k² + S_k²) N_(k+4) Δψ_s_k
        # = N_(k+4) Δψ_s_k for unit-tangent (C_k, S_k).)
        for k in range(4):
            dbxdx += dN8_dx[0, k] * (-S[k]) * A_dbs[k]
            dbxdy += dN8_dx[1, k] * (-S[k]) * A_dbs[k]
            dbydx += dN8_dx[0, k] * (+C[k]) * A_dbs[k]
            dbydy += dN8_dx[1, k] * (+C[k]) * A_dbs[k]
        Bb = np.zeros((3, 12))
        Bb[0, :] = +dbydx                  # κ_x = +∂β_y/∂x
        Bb[1, :] = -dbxdy                  # κ_y = -∂β_x/∂y
        Bb[2, :] = +dbydy - dbxdx          # κ_xy
        return Bb

    # ----------------------------------------------------- stiffness assembly
    def _K_local_5dof(self) -> np.ndarray:
        """20×20 local stiffness in 5-DOF-per-node ordering
        (u, v, w, θx, θy). Drilling θ_z added in 24-DOF expansion.
        """
        _, Xl = self._local_geom()
        Dm = self._D_membrane()
        Db = self._D_bending()
        L, C, S = _edge_geometry(Xl)
        A_dbs = _dbs_matrix(L, C, S)        # (4, 12)
        K = np.zeros((20, 20))

        # Index map: 12 plate-bending DOFs (w, βx, βy per node) → which
        # entries of the 20-vector (5-DOF/node local ordering)? In the
        # 20-vector, node i: (u, v, w, θx, θy) at indices 5i, 5i+1,
        # 5i+2, 5i+3, 5i+4. The plate DOFs are at indices (5i+2, 5i+3,
        # 5i+4) for each node.
        # Build a (12, 20) permutation P such that d_pb = P @ d20
        P = np.zeros((12, 20))
        for i in range(4):
            P[3 * i,     5 * i + 2] = 1.0   # w
            P[3 * i + 1, 5 * i + 3] = 1.0   # β_x
            P[3 * i + 2, 5 * i + 4] = 1.0   # β_y

        # 2×2 Gauss for membrane and bending (same as MITC4)
        xi2, eta2, w2 = gauss_legendre_2d_quad(2)
        for q in range(xi2.size):
            xq, yq, wq = float(xi2[q]), float(eta2[q]), float(w2[q])
            _, detJ, dN_dx, _ = _jacobian2d_4(xq, yq, Xl)
            Bm = self._Bm_local(dN_dx)                       # (3, 20)
            K += (Bm.T @ Dm @ Bm) * (detJ * wq)
            # 3×3 Gauss recommended for DK bending; here 2×2 to match
            # the standard Q4 K_bend integration order. Reduced-quadrature
            # spurious modes are blocked by the bilinear part.

        # Bending: 3×3 Gauss for accurate DK-rotation field
        xi3, eta3, w3 = gauss_legendre_2d_quad(3)
        for q in range(xi3.size):
            xq, yq, wq = float(xi3[q]), float(eta3[q]), float(w3[q])
            _, detJ, _, _ = _jacobian2d_4(xq, yq, Xl)
            Bb_pb = self._Bb_dk_local(xq, yq, Xl, L, C, S, A_dbs)   # (3, 12)
            Bb20 = Bb_pb @ P                                      # (3, 20)
            K += (Bb20.T @ Db @ Bb20) * (detJ * wq)
        return K

    def _K_local_6dof(self) -> np.ndarray:
        K20 = self._K_local_5dof()
        K24 = np.zeros((24, 24))
        idx5 = np.array([6 * i + k for i in range(4) for k in range(5)])
        for a, ia in enumerate(idx5):
            K24[ia, idx5] = K20[a, :]
        k_drill = self.drilling_factor * max(
            float(np.max(np.abs(np.diag(K20)))), 1.0
        )
        for i in range(4):
            K24[6 * i + 5, 6 * i + 5] += k_drill
            for j in range(4):
                if i != j:
                    K24[6 * i + 5, 6 * j + 5] -= k_drill / 4.0
        return K24

    def K_global(self) -> np.ndarray:
        R, _ = self._local_geom()
        T = self._T_global_to_local(R)
        K_loc = self._K_local_6dof()
        return T.T @ K_loc @ T

    # ----------------------------------------------------- mass
    def M_global(self, *, lumped: bool = False) -> np.ndarray:
        rho = getattr(self.material, "rho", 0.0)
        if rho == 0.0:
            return np.zeros((24, 24))
        R, Xl = self._local_geom()
        t = self.thickness
        M = np.zeros((24, 24))
        xi, eta, w = gauss_legendre_2d_quad(2)
        for q in range(xi.size):
            xq, yq, wq = float(xi[q]), float(eta[q]), float(w[q])
            _, detJ, _, _ = _jacobian2d_4(xq, yq, Xl)
            N = _N4(xq, yq)
            jw = rho * t * detJ * wq
            for i in range(4):
                for j in range(4):
                    val = N[i] * N[j] * jw
                    for k in range(3):
                        M[6 * i + k, 6 * j + k] += val
                    rot_val = val * (t ** 2 / 12.0)
                    M[6 * i + 3, 6 * j + 3] += rot_val
                    M[6 * i + 4, 6 * j + 4] += rot_val
        if lumped:
            return np.diag(M.sum(axis=1))
        T = self._T_global_to_local(R)
        return T.T @ M @ T

    # ----------------------------------------------------- recovery
    def recover(self) -> None:
        R, Xl = self._local_geom()
        T = self._T_global_to_local(R)
        u_glob = self.gather_u()
        u_loc = T @ u_glob
        idx5 = np.array([6 * i + k for i in range(4) for k in range(5)])
        u5 = u_loc[idx5]
        L, C, S = _edge_geometry(Xl)
        A_dbs = _dbs_matrix(L, C, S)
        P = np.zeros((12, 20))
        for i in range(4):
            P[3 * i,     5 * i + 2] = 1.0
            P[3 * i + 1, 5 * i + 3] = 1.0
            P[3 * i + 2, 5 * i + 4] = 1.0
        Dm = self._D_membrane()
        Db = self._D_bending()
        self.gp_membrane_strain = []
        self.gp_bending_curvature = []
        self.gp_resultants = []
        xi3, eta3, _ = gauss_legendre_2d_quad(3)
        for q in range(xi3.size):
            xq, yq = float(xi3[q]), float(eta3[q])
            _, _, dN_dx, _ = _jacobian2d_4(xq, yq, Xl)
            Bm = self._Bm_local(dN_dx)
            Bb_pb = self._Bb_dk_local(xq, yq, Xl, L, C, S, A_dbs)
            Bb20 = Bb_pb @ P
            eps_m = Bm @ u5
            kappa = Bb20 @ u5
            self.gp_membrane_strain.append(eps_m)
            self.gp_bending_curvature.append(kappa)
            self.gp_resultants.append(
                np.concatenate([Dm @ eps_m, Db @ kappa])
            )
