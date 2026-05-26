"""4-node MITC shell element (``ShellMITC4``).

The MITC ("Mixed Interpolation of Tensorial Components") family of
shell elements was introduced by Dvorkin & Bathe (1984) and is the
workhorse general-purpose shell in commercial structural-FE codes
(SAP2000, ETABS, MIDAS Civil/Gen, ADINA). The 4-node version handles
both **thin** (Kirchhoff) and **thick** (Mindlin-Reissner) regimes
without shear locking, by replacing the standard bilinear transverse-
shear interpolation with a tying scheme:

* In-plane membrane and bending strains use the standard isoparametric
  bilinear B-matrix → consistent rank, no membrane locking for a flat
  plate.
* Transverse shear ``γ_xz``, ``γ_yz`` is sampled at four tying points
  on the element edges and interpolated linearly back to Gauss points.
  This is the MITC4 cure for shear locking as ``t → 0``.

The element has **6 DOF per node** (``u_x, u_y, u_z, θ_x, θ_y, θ_z``)
to fit naturally into a 3-D frame model. The in-plane "drilling"
rotation ``θ_z`` carries a small fictitious stiffness so the global
system stays nonsingular when shells alone meet at a node.

For an isotropic single-layer shell of thickness ``t``:

    D_membrane  = (E t / (1 - ν²)) · [[1, ν, 0], [ν, 1, 0], [0, 0, (1-ν)/2]]
    D_bending   = D_membrane · t² / 12
    D_shear     = κ · G · t · I_2   (κ = 5/6, shear-correction factor)

Membrane and bending decouple for a symmetric mid-surface (the only
case implemented here). Composite layered shells are a future
extension.
"""
from __future__ import annotations

import numpy as np

from femsolver.elements.base import Element
from femsolver.numerics.quadrature import gauss_legendre_2d_quad


# ============================================================ local frame

def _local_frame(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build an orthonormal local frame at the element centroid.

    Returns
    -------
    R : (3, 3) array
        Columns are (e1_local, e2_local, n_local) in global coords.
        ``X_local = R^T @ (X_global - X_centroid)`` rotates a global
        vector into the local plane.
    X_local : (4, 2) array
        2-D local coordinates of the 4 nodes (z_local dropped — for a
        warped quad this is the projection onto the centroid plane,
        which is the standard MITC4 approximation).
    """
    # e1 from edge-12 → edge-43 average (the "1-3 diagonal of midsides")
    r1 = 0.5 * ((X[1] - X[0]) + (X[2] - X[3]))     # along ξ at η=0
    r2 = 0.5 * ((X[3] - X[0]) + (X[2] - X[1]))     # along η at ξ=0
    n = np.cross(r1, r2)
    n /= np.linalg.norm(n)
    e1 = r1 / np.linalg.norm(r1)
    # Orthogonalize e2 by removing e1 component
    e2 = np.cross(n, e1)
    R = np.column_stack([e1, e2, n])
    centroid = X.mean(axis=0)
    Xc = X - centroid
    X_local_3 = Xc @ R                              # (4, 3)
    return R, X_local_3[:, :2]


# ============================================================ shape & B

def _N(xi: float, eta: float) -> np.ndarray:
    return 0.25 * np.array([
        (1.0 - xi) * (1.0 - eta),
        (1.0 + xi) * (1.0 - eta),
        (1.0 + xi) * (1.0 + eta),
        (1.0 - xi) * (1.0 + eta),
    ])


def _dN_dxi(xi: float, eta: float) -> np.ndarray:
    """Returns (2, 4): row 0 = dN/dξ, row 1 = dN/dη."""
    return 0.25 * np.array([
        [-(1.0 - eta),  (1.0 - eta),  (1.0 + eta), -(1.0 + eta)],
        [-(1.0 - xi),  -(1.0 + xi),   (1.0 + xi),   (1.0 - xi)],
    ])


def _jacobian2d(xi: float, eta: float, Xl: np.ndarray):
    """For local 2-D node coords ``Xl`` (4, 2) at (ξ, η)."""
    dN = _dN_dxi(xi, eta)
    J = dN @ Xl                                     # (2, 2)
    detJ = float(np.linalg.det(J))
    if detJ <= 0.0:
        raise ValueError(
            f"ShellMITC4: non-positive Jacobian determinant ({detJ:g}) "
            f"at (ξ={xi}, η={eta}). Check node ordering (CCW seen from "
            f"the positive normal side) and avoid degenerate shapes."
        )
    Jinv = np.linalg.inv(J)
    dN_dx = Jinv @ dN                               # (2, 4)
    return J, detJ, dN_dx


# ============================================================ MITC4 shear

#: Tying points for transverse shear (Dvorkin & Bathe 1984):
#:   A: ( 0, -1)  carries γ_xz
#:   B: ( 1,  0)  carries γ_yz
#:   C: ( 0, +1)  carries γ_xz
#:   D: (-1,  0)  carries γ_yz
_TYING_PTS = {
    "A": (0.0, -1.0),
    "B": (+1.0, 0.0),
    "C": (0.0, +1.0),
    "D": (-1.0, 0.0),
}


def _Bs_at_tying_point(xi: float, eta: float, Xl: np.ndarray,
                        which: str) -> np.ndarray:
    """Build the transverse-shear B-matrix row at a single tying point.

    The "natural" transverse shear at the tying point is one of γ_ξz
    (at A, C) or γ_ηz (at B, D). We return a (1, 20) row in the
    *local* 5-DOF-per-node ordering (u, v, w, θx, θy) suitable for
    later transformation via the Jacobian.
    """
    dN = _dN_dxi(xi, eta)
    N = _N(xi, eta)
    row = np.zeros(20)
    if which in ("A", "C"):
        # γ_ξz = dw/dξ + N_i · θ_y_i? No — in shell convention
        # γ_xz_local = dw/dx + θ_y (rotation about y bends about y).
        # In natural coords, γ_ξz = dw/dξ + Σ N_i · (something).
        # Standard MITC4 expression: γ_ξz = dw/dξ + (J^T · θ)_ξ
        # which we evaluate by gathering θ in *natural* form.
        for i in range(4):
            row[5 * i + 2] = dN[0, i]            # dw/dξ on w_i
            row[5 * i + 3] = -N[i] * 0.0          # θx contribution (filled by Jacobian)
            row[5 * i + 4] = N[i]                # θy contribution (raw, transformed later)
    else:  # B, D
        for i in range(4):
            row[5 * i + 2] = dN[1, i]            # dw/dη on w_i
            row[5 * i + 3] = -N[i]               # θx contribution
            row[5 * i + 4] = N[i] * 0.0
    return row


# ============================================================ element

class ShellMITC4(Element):
    """4-node MITC shell element (flat or mildly warped).

    Parameters
    ----------
    tag : int
    nodes : (4,) sequence of node tags, counter-clockwise when viewed
        from the positive-normal side.
    material : ElasticIsotropic
    thickness : float
    k_shear : float, default 5/6
        Transverse-shear correction factor.
    drilling_factor : float, default 1e-3
        Fictitious drilling stiffness as a fraction of the smallest
        bending diagonal. Prevents singularity at nodes where only
        coplanar shells meet (no out-of-plane rotational coupling
        through other elements).

    Notes
    -----
    * The element is *flat* in its constructed local plane. For a
      warped input quad, the four nodes are projected onto the
      centroid plane — the standard MITC4 approximation. Severe
      warping degrades accuracy.
    * 2×2 Gauss quadrature is used for membrane and bending; the
      MITC tying-and-interpolation handles transverse shear without
      a separate quadrature rule.
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
        k_shear: float = 5.0 / 6.0,
        drilling_factor: float = 1.0e-3,
        section=None,
    ):
        super().__init__(tag, nodes, material)
        if drilling_factor < 0.0:
            raise ValueError(f"drilling_factor must be >= 0, got {drilling_factor}")
        # Two construction modes:
        #   (a) section= directly (the layered / composite path).
        #   (b) material + thickness — internally wrap as an isotropic
        #       single-layer ElasticShellSection for D-matrix retrieval.
        if section is not None:
            self.section = section
            self.thickness = float(section.thickness)
            self.k_shear = float(getattr(section, "k_shear", k_shear))
        else:
            if thickness is None or thickness <= 0:
                raise ValueError(f"thickness must be positive, got {thickness}")
            if not (0.0 < k_shear <= 1.0):
                raise ValueError(
                    f"k_shear (transverse-shear correction) must be in (0, 1], "
                    f"got {k_shear}"
                )
            self.thickness = float(thickness)
            self.k_shear = float(k_shear)
            # Lazy construction below: D_* methods read from material/thickness.
            self.section = None
        self.drilling_factor = float(drilling_factor)
        # Per-Gauss-point recovery buffers (populated by .recover())
        self.gp_membrane_strain: list[np.ndarray] = []
        self.gp_bending_curvature: list[np.ndarray] = []
        self.gp_shear_strain: list[np.ndarray] = []
        self.gp_resultants: list[np.ndarray] = []   # (N, M, Q) per GP

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
        """Membrane-bending coupling matrix (3, 3). Zero for symmetric
        sections; nonzero for asymmetric stacks (e.g. RC slab with
        bottom-only reinforcement)."""
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
        """Return (R, Xl) where R is the (3,3) local→global rotation
        and Xl is the (4, 2) local node coords."""
        X = self.node_coords()
        return _local_frame(X)

    def _T_global_to_local(self, R: np.ndarray) -> np.ndarray:
        """Block-diagonal 24x24 transform: ``u_local = T @ u_global``.

        Each 6-DOF block of T uses R^T on the 3 translations and R^T
        on the 3 rotations.
        """
        T = np.zeros((24, 24))
        Rt = R.T
        for i in range(4):
            T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = Rt
            T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = Rt
        return T

    # ----------------------------------------------------- B-matrices in local frame
    @staticmethod
    def _Bm_local(dN_dx: np.ndarray) -> np.ndarray:
        """Membrane B (3, 20) in 5-DOF-per-node local ordering
        (u, v, w, θx, θy)."""
        B = np.zeros((3, 20))
        for i in range(4):
            B[0, 5 * i] = dN_dx[0, i]              # ε_xx = du/dx
            B[1, 5 * i + 1] = dN_dx[1, i]          # ε_yy = dv/dy
            B[2, 5 * i] = dN_dx[1, i]              # γ_xy = du/dy + dv/dx
            B[2, 5 * i + 1] = dN_dx[0, i]
        return B

    @staticmethod
    def _Bb_local(dN_dx: np.ndarray) -> np.ndarray:
        """Bending B (3, 20) — couples to (θx, θy).

        Curvatures in Reissner-Mindlin form:
            κ_xx = +dθ_y / dx
            κ_yy = -dθ_x / dy
            κ_xy = +dθ_y / dy - dθ_x / dx
        """
        B = np.zeros((3, 20))
        for i in range(4):
            B[0, 5 * i + 4] = dN_dx[0, i]          # +dθ_y / dx
            B[1, 5 * i + 3] = -dN_dx[1, i]         # -dθ_x / dy
            B[2, 5 * i + 4] = dN_dx[1, i]
            B[2, 5 * i + 3] = -dN_dx[0, i]
        return B

    @staticmethod
    def _Bs_natural_local(xi: float, eta: float, Xl: np.ndarray) -> np.ndarray:
        """Build the (2, 20) shear B-matrix at (ξ, η) using MITC4 tying.

        Following Dvorkin-Bathe (1984): the covariant shear components
        ``γ_ξz`` and ``γ_ηz`` are tied at four edge-midpoint sampling
        points to remove shear locking, then interpolated linearly
        across the element and pulled back to Cartesian (γ_xz, γ_yz)
        through the inverse Jacobian.

        Returns ``Bs`` such that ``[γ_xz, γ_yz]^T = Bs @ u_local``.
        """
        # Natural shear at the four tying points (covariant components):
        # γ_ξz_A at (0, -1): γ_ξz = dw/dξ + Σ N_i^A · θ_y_local_i  (with sign)
        # γ_ξz_C at (0, +1)
        # γ_ηz_B at ( 1,  0)
        # γ_ηz_D at (-1,  0)
        Bxi = np.zeros((1, 20))
        Beta = np.zeros((1, 20))

        def _row_xi(xi_t: float, eta_t: float) -> np.ndarray:
            """γ_ξz = dw/dξ + (J_ξ · θ)_..., with J from local frame.
            In flat-shell MITC4 we use the local Cartesian θ directly
            via the Jacobian at the tying point.
            """
            dN = _dN_dxi(xi_t, eta_t)
            N = _N(xi_t, eta_t)
            J = dN @ Xl                            # (2, 2)
            r = np.zeros(20)
            for i in range(4):
                r[5 * i + 2] = dN[0, i]                 # dw/dξ
                # θ-contribution: -θ_x sin + θ_y cos in plane projection.
                # Standard form: J_ξ · [θ_y_loc, -θ_x_loc].
                r[5 * i + 4] = N[i] * J[0, 0]           # θ_y contrib
                r[5 * i + 3] = -N[i] * J[0, 1]          # θ_x contrib
            return r

        def _row_eta(xi_t: float, eta_t: float) -> np.ndarray:
            dN = _dN_dxi(xi_t, eta_t)
            N = _N(xi_t, eta_t)
            J = dN @ Xl
            r = np.zeros(20)
            for i in range(4):
                r[5 * i + 2] = dN[1, i]                 # dw/dη
                r[5 * i + 4] = N[i] * J[1, 0]
                r[5 * i + 3] = -N[i] * J[1, 1]
            return r

        # Tying points (Dvorkin-Bathe): A=(0,-1) for γ_ξz, C=(0,+1) γ_ξz
        rA = _row_xi(0.0, -1.0)
        rC = _row_xi(0.0, +1.0)
        rB = _row_eta(+1.0, 0.0)
        rD = _row_eta(-1.0, 0.0)
        # Bilinear interpolation back to (ξ, η):
        #   γ_ξz(ξ, η) = 0.5 * ((1 - η) · γ_ξz_A + (1 + η) · γ_ξz_C)
        #   γ_ηz(ξ, η) = 0.5 * ((1 + ξ) · γ_ηz_B + (1 - ξ) · γ_ηz_D)
        Bxi = 0.5 * ((1.0 - eta) * rA + (1.0 + eta) * rC)
        Beta = 0.5 * ((1.0 + xi) * rB + (1.0 - xi) * rD)
        # Pull covariant (γ_ξz, γ_ηz) back to Cartesian (γ_xz, γ_yz)
        # through Jinv: [γ_xz, γ_yz]^T = Jinv^T · [γ_ξz, γ_ηz]^T
        J = _dN_dxi(xi, eta) @ Xl
        Jinv = np.linalg.inv(J)
        # The transform is (γ_xy_cart) = Jinv^T · (γ_natural). The 2x2
        # acts on the rows of the (2, 20) covariant-shear matrix.
        cov = np.vstack([Bxi, Beta])                # (2, 20)
        Bs = Jinv.T @ cov                            # (2, 20)
        return Bs

    # ----------------------------------------------------- stiffness assembly
    def _K_local_5dof(self) -> np.ndarray:
        """Build the (20, 20) local stiffness over 5-DOF-per-node
        (u, v, w, θx, θy). Drilling θ_z is added after expansion to 24
        DOFs.
        """
        R, Xl = self._local_geom()
        Dm = self._D_membrane()
        Db = self._D_bending()
        Dc = self._D_coupling()
        Ds = self._D_shear()
        has_coupling = bool(np.any(Dc))
        K = np.zeros((20, 20))
        xi, eta, w = gauss_legendre_2d_quad(2)
        for q in range(xi.size):
            xq, yq, wq = float(xi[q]), float(eta[q]), float(w[q])
            _, detJ, dN_dx = _jacobian2d(xq, yq, Xl)
            Bm = self._Bm_local(dN_dx)
            Bb = self._Bb_local(dN_dx)
            Bs = self._Bs_natural_local(xq, yq, Xl)
            K += (Bm.T @ Dm @ Bm) * (detJ * wq)
            K += (Bb.T @ Db @ Bb) * (detJ * wq)
            K += (Bs.T @ Ds @ Bs) * (detJ * wq)
            if has_coupling:
                # Symmetric coupling: K_mb + K_mb^T = Bm^T Dc Bb + Bb^T Dc^T Bm
                K_mb = (Bm.T @ Dc @ Bb) * (detJ * wq)
                K += K_mb + K_mb.T
        return K

    def _K_local_6dof(self) -> np.ndarray:
        """Expand the 20×20 (u,v,w,θx,θy) stiffness to 24×24 by adding a
        drilling stiffness on θ_z."""
        K20 = self._K_local_5dof()
        K24 = np.zeros((24, 24))
        # 5-DOF → 6-DOF index map: local index 5*i + k → 6*i + k, k=0..4
        idx5 = np.array([6 * i + k for i in range(4) for k in range(5)])
        for a, ia in enumerate(idx5):
            K24[ia, idx5] = K20[a, :]
        # Drilling: small stiffness on θ_z to prevent singularity.
        # Calibrate to the smallest bending diagonal so it scales with E,t.
        k_drill = self.drilling_factor * max(
            np.max(np.abs(np.diag(K20))), 1.0
        )
        # Penalize θ_z = (θ_z_i - avg(θ_z)) — relative drilling, not
        # absolute, so it doesn't fight legitimate rigid-body rotations.
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

    # ----------------------------------------------------- geometric stiffness
    @staticmethod
    def _Gw_local(dN_dx: np.ndarray) -> np.ndarray:
        """Local out-of-plane displacement gradient.

        ``[dw/dx, dw/dy]^T = G_w · u5_local``, with the only nonzero
        columns at the ``w`` slot of each node (local index ``5 i + 2``).
        Shape ``(2, 20)``.
        """
        Gw = np.zeros((2, 20))
        for i in range(4):
            Gw[0, 5 * i + 2] = dN_dx[0, i]
            Gw[1, 5 * i + 2] = dN_dx[1, i]
        return Gw

    def _K_g_local_5dof(self, u5_local: np.ndarray) -> np.ndarray:
        """Geometric stiffness (20, 20) from the current membrane
        stress field. Dominant out-of-plane term — the contribution
        from in-plane displacement gradients of (u, v) is small at the
        pre-buckling state (Bathe 6.6.4).
        """
        _, Xl = self._local_geom()
        Dm = self._D_membrane()
        K_g = np.zeros((20, 20))
        xi, eta, w = gauss_legendre_2d_quad(2)
        for q in range(xi.size):
            xq, yq, wq = float(xi[q]), float(eta[q]), float(w[q])
            _, detJ, dN_dx = _jacobian2d(xq, yq, Xl)
            Bm = self._Bm_local(dN_dx)
            eps_m = Bm @ u5_local
            N = Dm @ eps_m          # [Nxx, Nyy, Nxy] resultants
            S = np.array([[N[0], N[2]], [N[2], N[1]]])
            Gw = self._Gw_local(dN_dx)
            K_g += (Gw.T @ S @ Gw) * (detJ * wq)
        return K_g

    def K_tangent_global(self) -> np.ndarray:
        """Tangent stiffness at the current state: elastic + geometric.

        Built so that ``K_tangent_global - K_global`` isolates the
        geometric part, exactly the form expected by
        :class:`~femsolver.analysis.buckling.LinearBucklingAnalysis`.
        """
        R, Xl = self._local_geom()
        T = self._T_global_to_local(R)
        u_glob = self.gather_u()
        u_loc = T @ u_glob
        idx5 = np.array([6 * i + k for i in range(4) for k in range(5)])
        u5 = u_loc[idx5]
        K_e_loc = self._K_local_6dof()
        K_g20 = self._K_g_local_5dof(u5)
        # Expand 20x20 K_g into 24x24 (drilling rows/cols stay zero)
        K_g24 = np.zeros((24, 24))
        for a, ia in enumerate(idx5):
            K_g24[ia, idx5] = K_g20[a, :]
        return T.T @ (K_e_loc + K_g24) @ T

    # ----------------------------------------------------- mass
    def M_global(self, *, lumped: bool = False) -> np.ndarray:
        rho = getattr(self.material, "rho", 0.0)
        if rho == 0.0:
            return np.zeros((24, 24))
        R, Xl = self._local_geom()
        t = self.thickness
        # Translational mass: lumped per node = rho*t*A_node, where
        # A_node = integral of N_i dA. For a Q4 with 2x2 quad, this gives
        # equal quarter-area lumps for a square; for a general quad we
        # integrate consistently and (optionally) row-sum.
        M = np.zeros((24, 24))
        xi, eta, w = gauss_legendre_2d_quad(2)
        for q in range(xi.size):
            xq, yq, wq = float(xi[q]), float(eta[q]), float(w[q])
            _, detJ, _ = _jacobian2d(xq, yq, Xl)
            N = _N(xq, yq)
            jw = rho * t * detJ * wq
            for i in range(4):
                for j in range(4):
                    val = N[i] * N[j] * jw
                    # translations on u, v, w
                    for k in range(3):
                        M[6 * i + k, 6 * j + k] += val
                    # rotary inertia (small — rho * t^3 / 12 on θx, θy)
                    rot_val = val * (t ** 2 / 12.0)
                    M[6 * i + 3, 6 * j + 3] += rot_val
                    M[6 * i + 4, 6 * j + 4] += rot_val
        if lumped:
            return np.diag(M.sum(axis=1))
        # Rotate to global (mass is invariant under orthogonal transform)
        # — but since rho is isotropic and the mass tensor is diagonal in
        # local axes per-node, rotating gives back the same global block.
        # For rotary terms the transform is also identity because we
        # use a diagonal rotary inertia. We still apply T for consistency.
        T = self._T_global_to_local(R)
        return T.T @ M @ T

    # ----------------------------------------------------- recovery
    def recover(self) -> None:
        R, Xl = self._local_geom()
        T = self._T_global_to_local(R)
        u_glob = self.gather_u()                    # (24,)
        u_loc = T @ u_glob                           # (24,)
        # Reduce to (20,) for the 5-DOF B-matrices
        idx5 = np.array([6 * i + k for i in range(4) for k in range(5)])
        u5 = u_loc[idx5]
        Dm, Db, Ds = self._D_membrane(), self._D_bending(), self._D_shear()
        self.gp_membrane_strain = []
        self.gp_bending_curvature = []
        self.gp_shear_strain = []
        self.gp_resultants = []
        xi, eta, _ = gauss_legendre_2d_quad(2)
        for q in range(xi.size):
            xq, yq = float(xi[q]), float(eta[q])
            _, _, dN_dx = _jacobian2d(xq, yq, Xl)
            Bm = self._Bm_local(dN_dx)
            Bb = self._Bb_local(dN_dx)
            Bs = self._Bs_natural_local(xq, yq, Xl)
            eps_m = Bm @ u5
            kappa = Bb @ u5
            gamma = Bs @ u5
            N = Dm @ eps_m
            M = Db @ kappa
            Q = Ds @ gamma
            self.gp_membrane_strain.append(eps_m)
            self.gp_bending_curvature.append(kappa)
            self.gp_shear_strain.append(gamma)
            self.gp_resultants.append(
                np.concatenate([N, M, Q])
            )
