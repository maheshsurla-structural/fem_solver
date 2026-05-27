"""9-node biquadratic shell element (``ShellMITC9``).

Phase 22.6 — higher-order curved-shell extension to the Phase 14 shell
family.

Motivation
----------
The 4-node MITC4 element (:class:`~femsolver.elements.shell.ShellMITC4`)
is excellent for general-purpose flat or mildly-warped meshes, but its
linear interpolation requires fine meshes to resolve curved geometries
accurately. A 9-node biquadratic shell captures parabolic in-plane
displacement variation per element, dramatically reducing the mesh
density required for curved shells like cylinders, spheres, and hyper-
boloids (Bathe-Bucalem 1991, Lee-Bathe 2004).

Locking cure (selective reduced integration)
--------------------------------------------
The original Bathe-Bucalem MITC9 element uses mixed-interpolation
tensorial-component tying at 6+6 sampling points to cure transverse
shear locking. This implementation uses an alternative, well-
established locking cure: **selective reduced integration (SRI)**.
The membrane and bending terms are integrated with the full 3x3
Gauss rule; the transverse-shear term uses a reduced 2x2 rule. SRI
is the standard Hughes-Cohen-Haroun (1978) treatment, robust against
shear locking down to L/t > 10^4, and widely cited in commercial
shell codes as an alternative to MITC tying. We keep the class name
``ShellMITC9`` for naming consistency with ``ShellMITC4``; users
familiar with the Bathe convention should read this as "9-node shell
with locking cure analogous to MITC4's purpose."

A future enhancement could replace SRI with the full Bathe-Bucalem
covariant-tying scheme; the membrane / bending / drilling
machinery here is independent of that choice and would be reused.

Element
-------
* 9 nodes: corners 0-3, mid-edges 4-7, center 8 (standard
  Lagrange / serendipity-9 ordering)
* 6 DOF per node (u, v, w, θ_x, θ_y, θ_z) — 54 DOFs total
* 3×3 Gauss for membrane and bending (full)
* 2×2 Gauss for transverse shear (reduced)
* Drilling stiffness on θ_z analogous to MITC4
"""
from __future__ import annotations

import numpy as np

from femsolver.elements.base import Element
from femsolver.numerics.quadrature import gauss_legendre_2d_quad


# ============================================================ shape functions

#: Local node numbering (consistent with most textbooks):
#:
#:    3-----6-----2
#:    |     |     |
#:    7-----8-----5
#:    |     |     |
#:    0-----4-----1
#:
#: Indices 0-3 are corners (CCW from bottom-left), 4-7 are mid-edges in
#: the same CCW order starting with the bottom edge, and 8 is the
#: center.

# 1-D quadratic Lagrange polynomials at nodes (-1, 0, +1)
def _L1(t: float) -> float:
    """Lagrange polynomial at -1: equals 1 at -1, 0 at 0 and +1."""
    return 0.5 * t * (t - 1.0)


def _L2(t: float) -> float:
    """Lagrange polynomial at 0: equals 0 at -1 and +1, 1 at 0."""
    return 1.0 - t * t


def _L3(t: float) -> float:
    """Lagrange polynomial at +1: equals 1 at +1, 0 at 0 and -1."""
    return 0.5 * t * (t + 1.0)


def _dL1(t: float) -> float:
    return t - 0.5


def _dL2(t: float) -> float:
    return -2.0 * t


def _dL3(t: float) -> float:
    return t + 0.5


def _N9(xi: float, eta: float) -> np.ndarray:
    """Biquadratic Lagrange shape functions at (ξ, η) ∈ [-1, 1]^2.

    Returns array of length 9 in the node ordering above:
    [N0, N1, N2, N3, N4, N5, N6, N7, N8]
        =  N(ξ_i, η_i) tensor product Lagrange.
    """
    Lxi = (_L1(xi), _L2(xi), _L3(xi))     # at ξ = -1, 0, +1
    Let = (_L1(eta), _L2(eta), _L3(eta))  # at η = -1, 0, +1
    # Corners: (Lxi, Let) products at corners (i_xi, i_et)
    # 0: (-1, -1) → Lxi[0] * Let[0]
    # 1: (+1, -1) → Lxi[2] * Let[0]
    # 2: (+1, +1) → Lxi[2] * Let[2]
    # 3: (-1, +1) → Lxi[0] * Let[2]
    # 4: ( 0, -1) → Lxi[1] * Let[0]
    # 5: (+1,  0) → Lxi[2] * Let[1]
    # 6: ( 0, +1) → Lxi[1] * Let[2]
    # 7: (-1,  0) → Lxi[0] * Let[1]
    # 8: ( 0,  0) → Lxi[1] * Let[1]
    N = np.empty(9, dtype=float)
    N[0] = Lxi[0] * Let[0]
    N[1] = Lxi[2] * Let[0]
    N[2] = Lxi[2] * Let[2]
    N[3] = Lxi[0] * Let[2]
    N[4] = Lxi[1] * Let[0]
    N[5] = Lxi[2] * Let[1]
    N[6] = Lxi[1] * Let[2]
    N[7] = Lxi[0] * Let[1]
    N[8] = Lxi[1] * Let[1]
    return N


def _dN9_dxi(xi: float, eta: float) -> np.ndarray:
    """Returns (2, 9): row 0 = dN/dξ, row 1 = dN/dη."""
    Lxi = (_L1(xi), _L2(xi), _L3(xi))
    Let = (_L1(eta), _L2(eta), _L3(eta))
    dLxi = (_dL1(xi), _dL2(xi), _dL3(xi))
    dLet = (_dL1(eta), _dL2(eta), _dL3(eta))

    # ξ and η indices for each node (0..8):
    i_xi  = (0, 2, 2, 0, 1, 2, 1, 0, 1)
    i_eta = (0, 0, 2, 2, 0, 1, 2, 1, 1)

    dN = np.empty((2, 9), dtype=float)
    for k in range(9):
        a, b = i_xi[k], i_eta[k]
        dN[0, k] = dLxi[a] * Let[b]       # ∂N_k/∂ξ
        dN[1, k] = Lxi[a]  * dLet[b]      # ∂N_k/∂η
    return dN


# ============================================================ local frame

def _local_frame_9(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build an orthonormal local frame at the element centroid.

    For a 9-node shell, the centroid is at (ξ, η) = (0, 0) — that's
    node 8 directly, but we compute via the shape functions in case
    of distortion. Tangent vectors come from the Jacobian at (0, 0).
    """
    dN0 = _dN9_dxi(0.0, 0.0)
    g_xi = dN0[0] @ X      # (3,) tangent in ξ-direction at center
    g_et = dN0[1] @ X
    n = np.cross(g_xi, g_et)
    n_norm = np.linalg.norm(n)
    if n_norm <= 0.0:
        raise ValueError(
            "ShellMITC9: degenerate element with zero normal vector at "
            "centroid -- check node coordinates / ordering."
        )
    n /= n_norm
    e1 = g_xi / np.linalg.norm(g_xi)
    e2 = np.cross(n, e1)            # e2 perpendicular to e1 and n
    R = np.column_stack([e1, e2, n])  # (3, 3) local→global
    centroid = _N9(0.0, 0.0) @ X      # (3,)
    X_local = (X - centroid) @ R     # project into local axes
    return R, X_local[:, :2]


def _jacobian2d_9(xi: float, eta: float, Xl: np.ndarray):
    """For local 2-D node coords ``Xl`` (9, 2) at (ξ, η).

    Returns (J, detJ, dN_dx) where dN_dx is (2, 9) in local-Cartesian
    space.
    """
    dN = _dN9_dxi(xi, eta)            # (2, 9)
    J = dN @ Xl                       # (2, 2)
    detJ = float(np.linalg.det(J))
    if detJ <= 0.0:
        raise ValueError(
            f"ShellMITC9: non-positive Jacobian determinant ({detJ:g}) "
            f"at (ξ={xi:g}, η={eta:g}). Check node ordering (CCW) and "
            "edge/center node placement."
        )
    Jinv = np.linalg.inv(J)
    dN_dx = Jinv @ dN                 # (2, 9)
    return J, detJ, dN_dx


# ============================================================ element

class ShellMITC9(Element):
    """9-node biquadratic shell with selective reduced integration.

    Parameters
    ----------
    tag : int
    nodes : sequence of 9 node tags (corners 0-3 CCW, mid-edges 4-7
        starting at bottom edge CCW, center 8).
    material : ElasticIsotropic
    thickness : float, optional
        Required if ``section`` is not given.
    k_shear : float, default 5/6
    drilling_factor : float, default 1e-3
        Fictitious drilling stiffness as a fraction of the max bending
        diagonal. Prevents singularity at nodes where only coplanar
        shells meet.
    section : ShellSectionBase, optional
        For composite / layered shells. Overrides material+thickness.

    Notes
    -----
    * Flat in its local centroidal plane; warped 9-node patches are
      projected onto that plane (as MITC4 does).
    * Membrane and bending use 3x3 Gauss; transverse shear uses 2x2
      reduced Gauss (selective reduced integration; Hughes 1978).
      This is a robust locking cure equivalent in role -- though not
      formulation -- to the Bathe-Bucalem MITC9 covariant tying scheme.
    * The element is a higher-order companion to ``ShellMITC4`` for
      curved-shell convergence studies.
    """

    n_nodes = 9
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
            raise ValueError(
                f"drilling_factor must be >= 0, got {drilling_factor}"
            )
        if section is not None:
            self.section = section
            self.thickness = float(section.thickness)
            self.k_shear = float(getattr(section, "k_shear", k_shear))
        else:
            if thickness is None or thickness <= 0:
                raise ValueError(
                    f"thickness must be positive, got {thickness}"
                )
            if not (0.0 < k_shear <= 1.0):
                raise ValueError(
                    f"k_shear must be in (0, 1], got {k_shear}"
                )
            self.thickness = float(thickness)
            self.k_shear = float(k_shear)
            self.section = None
        self.drilling_factor = float(drilling_factor)
        # Per-Gauss-point recovery buffers
        self.gp_membrane_strain: list[np.ndarray] = []
        self.gp_bending_curvature: list[np.ndarray] = []
        self.gp_shear_strain: list[np.ndarray] = []
        self.gp_resultants: list[np.ndarray] = []

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

    # ----------------------------------------------------- local-global
    def _local_geom(self):
        X = self.node_coords()
        return _local_frame_9(X)

    def _T_global_to_local(self, R: np.ndarray) -> np.ndarray:
        """Block-diagonal 54x54 transform: ``u_local = T @ u_global``.

        Each 6-DOF block uses R^T on translations and R^T on rotations.
        """
        T = np.zeros((54, 54))
        Rt = R.T
        for i in range(9):
            T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = Rt
            T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = Rt
        return T

    # ----------------------------------------------------- B-matrices
    @staticmethod
    def _Bm_local(dN_dx: np.ndarray) -> np.ndarray:
        """Membrane B (3, 45) in 5-DOF-per-node ordering (u, v, w, θx, θy)."""
        B = np.zeros((3, 45))
        for i in range(9):
            B[0, 5 * i] = dN_dx[0, i]
            B[1, 5 * i + 1] = dN_dx[1, i]
            B[2, 5 * i] = dN_dx[1, i]
            B[2, 5 * i + 1] = dN_dx[0, i]
        return B

    @staticmethod
    def _Bb_local(dN_dx: np.ndarray) -> np.ndarray:
        """Bending B (3, 45) — couples to (θx, θy).

        Curvatures in Reissner-Mindlin form:
            κ_xx = +dθ_y / dx
            κ_yy = -dθ_x / dy
            κ_xy = +dθ_y / dy - dθ_x / dx
        """
        B = np.zeros((3, 45))
        for i in range(9):
            B[0, 5 * i + 4] = dN_dx[0, i]      # +dθ_y/dx
            B[1, 5 * i + 3] = -dN_dx[1, i]     # -dθ_x/dy
            B[2, 5 * i + 4] = dN_dx[1, i]
            B[2, 5 * i + 3] = -dN_dx[0, i]
        return B

    @staticmethod
    def _Bs_local(xi: float, eta: float, Xl: np.ndarray) -> np.ndarray:
        """Transverse-shear B (2, 45) at (ξ, η), full biquadratic in
        local Cartesian coords.

        Reissner-Mindlin transverse shears:
            γ_xz = dw/dx + θ_y
            γ_yz = dw/dy - θ_x
        """
        _, _, dN_dx = _jacobian2d_9(xi, eta, Xl)
        N = _N9(xi, eta)
        B = np.zeros((2, 45))
        for i in range(9):
            B[0, 5 * i + 2] = dN_dx[0, i]      # dw/dx
            B[0, 5 * i + 4] = N[i]             # + θ_y
            B[1, 5 * i + 2] = dN_dx[1, i]      # dw/dy
            B[1, 5 * i + 3] = -N[i]            # - θ_x
        return B

    # ----------------------------------------------------- stiffness assembly
    def _K_local_5dof(self) -> np.ndarray:
        """Build the (45, 45) local stiffness over 5-DOF-per-node
        (u, v, w, θx, θy). Drilling θ_z is added after expansion to
        54 DOFs.
        """
        _, Xl = self._local_geom()
        Dm = self._D_membrane()
        Db = self._D_bending()
        Dc = self._D_coupling()
        Ds = self._D_shear()
        has_coupling = bool(np.any(Dc))
        K = np.zeros((45, 45))

        # Full 3x3 Gauss for membrane + bending
        xi3, eta3, w3 = gauss_legendre_2d_quad(3)
        for q in range(xi3.size):
            xq, yq, wq = float(xi3[q]), float(eta3[q]), float(w3[q])
            _, detJ, dN_dx = _jacobian2d_9(xq, yq, Xl)
            Bm = self._Bm_local(dN_dx)
            Bb = self._Bb_local(dN_dx)
            K += (Bm.T @ Dm @ Bm) * (detJ * wq)
            K += (Bb.T @ Db @ Bb) * (detJ * wq)
            if has_coupling:
                K_mb = (Bm.T @ Dc @ Bb) * (detJ * wq)
                K += K_mb + K_mb.T

        # Reduced 2x2 Gauss for transverse shear (selective reduced
        # integration to cure shear locking)
        xi2, eta2, w2 = gauss_legendre_2d_quad(2)
        for q in range(xi2.size):
            xq, yq, wq = float(xi2[q]), float(eta2[q]), float(w2[q])
            _, detJ, _ = _jacobian2d_9(xq, yq, Xl)
            Bs = self._Bs_local(xq, yq, Xl)
            K += (Bs.T @ Ds @ Bs) * (detJ * wq)
        return K

    def _K_local_6dof(self) -> np.ndarray:
        """Expand 45×45 (u,v,w,θx,θy) stiffness to 54×54 by adding a
        drilling stiffness on θ_z."""
        K45 = self._K_local_5dof()
        K54 = np.zeros((54, 54))
        idx5 = np.array([6 * i + k for i in range(9) for k in range(5)])
        for a, ia in enumerate(idx5):
            K54[ia, idx5] = K45[a, :]
        # Drilling: small relative stiffness on θ_z. Penalize deviation
        # of each node's θ_z from the average, not absolute -- so we
        # don't fight legitimate rigid-body rotations about the normal.
        k_drill = self.drilling_factor * max(
            float(np.max(np.abs(np.diag(K45)))), 1.0
        )
        for i in range(9):
            K54[6 * i + 5, 6 * i + 5] += k_drill
            for j in range(9):
                if i != j:
                    K54[6 * i + 5, 6 * j + 5] -= k_drill / 9.0
        return K54

    def K_global(self) -> np.ndarray:
        R, _ = self._local_geom()
        T = self._T_global_to_local(R)
        K_loc = self._K_local_6dof()
        return T.T @ K_loc @ T

    # ----------------------------------------------------- mass
    def M_global(self, *, lumped: bool = False) -> np.ndarray:
        rho = getattr(self.material, "rho", 0.0)
        if rho == 0.0:
            return np.zeros((54, 54))
        R, Xl = self._local_geom()
        t = self.thickness
        M = np.zeros((54, 54))
        xi3, eta3, w3 = gauss_legendre_2d_quad(3)
        for q in range(xi3.size):
            xq, yq, wq = float(xi3[q]), float(eta3[q]), float(w3[q])
            _, detJ, _ = _jacobian2d_9(xq, yq, Xl)
            N = _N9(xq, yq)
            jw = rho * t * detJ * wq
            for i in range(9):
                for j in range(9):
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
        idx5 = np.array([6 * i + k for i in range(9) for k in range(5)])
        u5 = u_loc[idx5]
        Dm, Db, Ds = self._D_membrane(), self._D_bending(), self._D_shear()
        self.gp_membrane_strain = []
        self.gp_bending_curvature = []
        self.gp_shear_strain = []
        self.gp_resultants = []
        # Recover at the same 3x3 grid used for K membrane/bending
        xi3, eta3, _ = gauss_legendre_2d_quad(3)
        for q in range(xi3.size):
            xq, yq = float(xi3[q]), float(eta3[q])
            _, _, dN_dx = _jacobian2d_9(xq, yq, Xl)
            Bm = self._Bm_local(dN_dx)
            Bb = self._Bb_local(dN_dx)
            Bs = self._Bs_local(xq, yq, Xl)
            eps_m = Bm @ u5
            kappa = Bb @ u5
            gamma = Bs @ u5
            N = Dm @ eps_m
            M = Db @ kappa
            Q = Ds @ gamma
            self.gp_membrane_strain.append(eps_m)
            self.gp_bending_curvature.append(kappa)
            self.gp_shear_strain.append(gamma)
            self.gp_resultants.append(np.concatenate([N, M, Q]))
