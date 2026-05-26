"""3-D continuum elements -- Hex8 (trilinear brick) and Tet4 (linear
tetrahedron).

Both elements use the standard isoparametric formulation and the
6-component Voigt strain ordering

    eps = (eps_xx, eps_yy, eps_zz, gamma_xy, gamma_yz, gamma_zx)

which matches the ``ElasticIsotropic.D_3d`` (6, 6) constitutive
matrix already in the materials library (the off-diagonal G entries
are uncoupled and indifferent to the shear-component ordering).

Both elements have **3 DOFs per node** (u, v, w) so they fit
naturally into 3-D models with ``ndf = 3`` or ``ndf = 6`` (in the
latter, the assembler's ``eqn[:dofs_per_node]`` slot picks up the
three translations and leaves the rotations untouched).

Hex8 uses 2x2x2 Gauss quadrature (exact for polynomials up to bicubic
in any direction). Tet4 has constant strain so a single midpoint
evaluation suffices.

Future Phase 15.x candidates:

* B-bar / selective reduced integration to cure volumetric locking
  for nearly-incompressible materials (nu -> 0.5).
* Hex8 with reduced integration + hourglass stabilization (the
  industry workhorse for plasticity).
* Tet10 (10-node quadratic tet) for unstructured-mesh accuracy.
"""
from __future__ import annotations

import numpy as np

from femsolver.elements.base import Element
from femsolver.numerics.quadrature import gauss_legendre_3d_hex


# ============================================================ shape helpers

#: Local-coord positions of the 8 corner nodes of the bi-unit cube,
#: in the standard isoparametric ordering 1..8 (bottom CCW, then top CCW).
_HEX8_CORNERS = np.array([
    [-1, -1, -1],
    [+1, -1, -1],
    [+1, +1, -1],
    [-1, +1, -1],
    [-1, -1, +1],
    [+1, -1, +1],
    [+1, +1, +1],
    [-1, +1, +1],
], dtype=float)


def _hex8_shape(xi: float, eta: float, zeta: float) -> np.ndarray:
    """8-vector of trilinear shape function values at (xi, eta, zeta)."""
    N = np.empty(8)
    for i in range(8):
        xi_i, eta_i, zeta_i = _HEX8_CORNERS[i]
        N[i] = 0.125 * (1.0 + xi * xi_i) * (1.0 + eta * eta_i) * (1.0 + zeta * zeta_i)
    return N


def _hex8_dN_dxi(xi: float, eta: float, zeta: float) -> np.ndarray:
    """Returns (3, 8): row 0 = dN/dxi, row 1 = dN/deta, row 2 = dN/dzeta."""
    dN = np.empty((3, 8))
    for i in range(8):
        xi_i, eta_i, zeta_i = _HEX8_CORNERS[i]
        dN[0, i] = 0.125 * xi_i * (1.0 + eta * eta_i) * (1.0 + zeta * zeta_i)
        dN[1, i] = 0.125 * eta_i * (1.0 + xi * xi_i) * (1.0 + zeta * zeta_i)
        dN[2, i] = 0.125 * zeta_i * (1.0 + xi * xi_i) * (1.0 + eta * eta_i)
    return dN


def _voigt_B_3d(dN_dx: np.ndarray, n_nodes: int) -> np.ndarray:
    """Build the (6, 3*n_nodes) strain-displacement matrix in 3-D Voigt
    notation eps = (exx, eyy, ezz, gxy, gyz, gzx)."""
    B = np.zeros((6, 3 * n_nodes))
    for i in range(n_nodes):
        dNx, dNy, dNz = dN_dx[0, i], dN_dx[1, i], dN_dx[2, i]
        B[0, 3 * i + 0] = dNx                    # eps_xx
        B[1, 3 * i + 1] = dNy                    # eps_yy
        B[2, 3 * i + 2] = dNz                    # eps_zz
        B[3, 3 * i + 0] = dNy                    # gamma_xy
        B[3, 3 * i + 1] = dNx
        B[4, 3 * i + 1] = dNz                    # gamma_yz
        B[4, 3 * i + 2] = dNy
        B[5, 3 * i + 0] = dNz                    # gamma_zx
        B[5, 3 * i + 2] = dNx
    return B


# ============================================================ Hex8

class Hex8(Element):
    """8-node trilinear hexahedron.

    Parameters
    ----------
    tag : int
    nodes : sequence of 8 node tags in the standard ordering --
        bottom counter-clockwise (z = z_min) then top counter-
        clockwise (z = z_max), each face viewed from the +z side.
    material : ElasticIsotropic
        Provides ``D_3d()`` for the elastic (initial) stiffness and
        ``rho`` for the mass matrix. When ``material3d`` is provided,
        ``material`` is still used for mass-density.
    quadrature : int, default 2
        Number of Gauss points per direction (2 = 2x2x2 = 8 GPs,
        exact integration for trilinear shapes).
    material3d : optional 3-D plasticity material (e.g.
        ``J2Plasticity3D``, ``DruckerPrager3D``).
        When given, the element clones one independent copy per
        integration point and uses ``get_response`` for the stress /
        tangent. The element then carries its own committed / trial
        state and the standard Newton solve picks up the plastic
        response automatically.
    """

    n_nodes = 8
    dofs_per_node = 3

    def __init__(
        self,
        tag: int,
        nodes,
        material,
        *,
        quadrature: int = 2,
        material3d=None,
    ):
        super().__init__(tag, nodes, material)
        self.quadrature = int(quadrature)
        self.gp_stress: list[np.ndarray] = []
        self.gp_strain: list[np.ndarray] = []
        self._body_force: np.ndarray = np.zeros(3)
        self.material3d = material3d
        if material3d is not None:
            n_gp = self.quadrature ** 3
            self._ip_materials = [material3d.clone() for _ in range(n_gp)]
        else:
            self._ip_materials = None

    def _jacobian(self, xi: float, eta: float, zeta: float,
                   X: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        """Returns (J, detJ, dN/dx) at (xi, eta, zeta) for the given node
        coordinates ``X (8, 3)``."""
        dN = _hex8_dN_dxi(xi, eta, zeta)
        J = dN @ X                            # (3, 3)
        detJ = float(np.linalg.det(J))
        if detJ <= 0.0:
            raise ValueError(
                f"Hex8 element {self.tag}: non-positive Jacobian determinant "
                f"({detJ:g}) at (xi={xi}, eta={eta}, zeta={zeta}). Check "
                f"node ordering and element shape."
            )
        Jinv = np.linalg.inv(J)
        dN_dx = Jinv @ dN                     # (3, 8)
        return J, detJ, dN_dx

    def _D_initial(self) -> np.ndarray:
        """Initial (linear) D matrix. Uses material3d.D_elastic() if a
        stateful material is attached; otherwise material.D_3d()."""
        if self.material3d is not None:
            return self.material3d.D_elastic()
        return self.material.D_3d()

    def K_global(self) -> np.ndarray:
        X = self.node_coords()                 # (8, 3)
        D = self._D_initial()
        K = np.zeros((24, 24))
        xi, eta, zeta, w = gauss_legendre_3d_hex(self.quadrature)
        for q in range(xi.size):
            _, detJ, dN_dx = self._jacobian(
                float(xi[q]), float(eta[q]), float(zeta[q]), X,
            )
            B = _voigt_B_3d(dN_dx, n_nodes=8)
            K += (B.T @ D @ B) * (detJ * float(w[q]))
        return K

    def K_tangent_global(self) -> np.ndarray:
        """Tangent stiffness at the current displacement state.

        For ``material3d is None``: same as ``K_global`` (linear elastic).

        For ``material3d`` present: integrates ``B^T D_tangent B`` at each
        Gauss point, where ``D_tangent`` is the per-IP tangent returned
        by the plastic material at the current strain. (Currently the
        materials return the elastic D as their tangent, giving linear
        Newton convergence; an algorithmic-consistent tangent is a
        future refinement.)
        """
        if self._ip_materials is None:
            return self.K_global()
        X = self.node_coords()
        u = self.gather_u()
        K = np.zeros((24, 24))
        xi, eta, zeta, w = gauss_legendre_3d_hex(self.quadrature)
        for q in range(xi.size):
            _, detJ, dN_dx = self._jacobian(
                float(xi[q]), float(eta[q]), float(zeta[q]), X,
            )
            B = _voigt_B_3d(dN_dx, n_nodes=8)
            eps_voigt = B @ u
            _, D_t = self._ip_materials[q].get_response(eps_voigt)
            K += (B.T @ D_t @ B) * (detJ * float(w[q]))
        return K

    def f_int_global(self) -> np.ndarray:
        """Internal resisting force.

        For ``material3d is None`` falls back to the linear ``K u``.
        Otherwise integrates ``int B^T sigma dV`` with the *plastic*
        stresses returned by each IP's ``get_response``.
        """
        if self._ip_materials is None:
            return self.K_global() @ self.gather_u()
        X = self.node_coords()
        u = self.gather_u()
        f = np.zeros(24)
        xi, eta, zeta, w = gauss_legendre_3d_hex(self.quadrature)
        for q in range(xi.size):
            _, detJ, dN_dx = self._jacobian(
                float(xi[q]), float(eta[q]), float(zeta[q]), X,
            )
            B = _voigt_B_3d(dN_dx, n_nodes=8)
            eps_voigt = B @ u
            sigma_voigt, _ = self._ip_materials[q].get_response(eps_voigt)
            f += (B.T @ sigma_voigt) * (detJ * float(w[q]))
        return f

    def commit_state(self) -> None:
        if self._ip_materials is not None:
            for ip in self._ip_materials:
                ip.commit_state()

    def revert_state(self) -> None:
        if self._ip_materials is not None:
            for ip in self._ip_materials:
                ip.revert_state()

    def M_global(self, *, lumped: bool = False) -> np.ndarray:
        rho = getattr(self.material, "rho", 0.0)
        if rho == 0.0:
            return np.zeros((24, 24))
        X = self.node_coords()
        M = np.zeros((24, 24))
        xi, eta, zeta, w = gauss_legendre_3d_hex(self.quadrature)
        for q in range(xi.size):
            xq, yq, zq, wq = (float(xi[q]), float(eta[q]),
                                float(zeta[q]), float(w[q]))
            N = _hex8_shape(xq, yq, zq)
            _, detJ, _ = self._jacobian(xq, yq, zq, X)
            jw = rho * detJ * wq
            Nbar = np.zeros((3, 24))
            for i in range(8):
                Nbar[0, 3 * i + 0] = N[i]
                Nbar[1, 3 * i + 1] = N[i]
                Nbar[2, 3 * i + 2] = N[i]
            M += (Nbar.T @ Nbar) * jw
        if lumped:
            return np.diag(M.sum(axis=1))
        return M

    # ------------------------------------------------------------ loading
    def set_body_force(self, bx: float, by: float, bz: float) -> None:
        self._body_force = np.array([float(bx), float(by), float(bz)])

    def clear_distributed_loads(self) -> None:
        self._body_force = np.zeros(3)

    def f_eq_global(self) -> np.ndarray:
        if not np.any(self._body_force):
            return np.zeros(24)
        X = self.node_coords()
        f = np.zeros(24)
        bx, by, bz = self._body_force
        xi, eta, zeta, w = gauss_legendre_3d_hex(self.quadrature)
        for q in range(xi.size):
            xq, yq, zq, wq = (float(xi[q]), float(eta[q]),
                                float(zeta[q]), float(w[q]))
            N = _hex8_shape(xq, yq, zq)
            _, detJ, _ = self._jacobian(xq, yq, zq, X)
            jw = detJ * wq
            for i in range(8):
                f[3 * i + 0] += N[i] * bx * jw
                f[3 * i + 1] += N[i] * by * jw
                f[3 * i + 2] += N[i] * bz * jw
        return f

    def recover(self) -> None:
        X = self.node_coords()
        u = self.gather_u()
        D = self.material.D_3d()
        self.gp_stress = []
        self.gp_strain = []
        xi, eta, zeta, _ = gauss_legendre_3d_hex(self.quadrature)
        for q in range(xi.size):
            _, _, dN_dx = self._jacobian(
                float(xi[q]), float(eta[q]), float(zeta[q]), X,
            )
            B = _voigt_B_3d(dN_dx, n_nodes=8)
            eps = B @ u
            sig = D @ eps
            self.gp_strain.append(eps)
            self.gp_stress.append(sig)


# ============================================================ Tet4

class Tet4(Element):
    """4-node linear tetrahedron (constant-strain in 3-D).

    Parameters
    ----------
    tag : int
    nodes : sequence of 4 node tags. The convention is that nodes
        ``1, 2, 3`` form the base face seen from node 4 in
        counter-clockwise order; equivalently, the signed volume
        ``(x2 - x1) . ((x3 - x1) x (x4 - x1)) > 0``.
    material : ElasticIsotropic

    Notes
    -----
    Tet4 is the simplest 3-D continuum element. It is notoriously
    overly stiff on coarse meshes (no quadratic kinematic field for
    bending or shear) and is best used for filling unstructured
    meshes around bricks or for problems dominated by bulk volumetric
    response (e.g. soil with confining pressure). For bending-
    dominated problems prefer Hex8 or, future, Tet10.
    """

    n_nodes = 4
    dofs_per_node = 3

    def __init__(self, tag: int, nodes, material, *, material3d=None):
        super().__init__(tag, nodes, material)
        self.stress: np.ndarray | None = None
        self.strain: np.ndarray | None = None
        self._body_force: np.ndarray = np.zeros(3)
        self.material3d = material3d
        # CST has 1 integration point, so a single cloned material.
        self._ip_material = (
            material3d.clone() if material3d is not None else None
        )

    def _B_and_volume(self) -> tuple[np.ndarray, float]:
        """For a CST tet, B and V are constant. Returns (B, V)."""
        X = self.node_coords()                # (4, 3)
        # Jacobian: derivative of (x, y, z) w.r.t. (xi, eta, zeta) on
        # the reference tet (vertices (0,0,0), (1,0,0), (0,1,0), (0,0,1)).
        # Shape functions: N1 = 1 - xi - eta - zeta, N2 = xi, N3 = eta, N4 = zeta.
        # dN/dxi (constant): (-1, 1, 0, 0); dN/deta: (-1, 0, 1, 0);
        # dN/dzeta: (-1, 0, 0, 1).
        dN_dref = np.array([
            [-1.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0, 1.0],
        ])
        J = dN_dref @ X                       # (3, 3)
        detJ = float(np.linalg.det(J))
        if detJ <= 0.0:
            raise ValueError(
                f"Tet4 element {self.tag}: non-positive Jacobian determinant "
                f"({detJ:g}). Check node ordering -- the signed volume of "
                f"(x2-x1, x3-x1, x4-x1) must be positive."
            )
        V = detJ / 6.0
        Jinv = np.linalg.inv(J)
        dN_dx = Jinv @ dN_dref                # (3, 4)
        B = _voigt_B_3d(dN_dx, n_nodes=4)
        return B, V

    def _D_initial(self) -> np.ndarray:
        if self.material3d is not None:
            return self.material3d.D_elastic()
        return self.material.D_3d()

    def K_global(self) -> np.ndarray:
        B, V = self._B_and_volume()
        return (B.T @ self._D_initial() @ B) * V

    def K_tangent_global(self) -> np.ndarray:
        if self._ip_material is None:
            return self.K_global()
        B, V = self._B_and_volume()
        u = self.gather_u()
        eps_voigt = B @ u
        _, D_t = self._ip_material.get_response(eps_voigt)
        return (B.T @ D_t @ B) * V

    def f_int_global(self) -> np.ndarray:
        if self._ip_material is None:
            return self.K_global() @ self.gather_u()
        B, V = self._B_and_volume()
        u = self.gather_u()
        eps_voigt = B @ u
        sigma_voigt, _ = self._ip_material.get_response(eps_voigt)
        return B.T @ sigma_voigt * V

    def commit_state(self) -> None:
        if self._ip_material is not None:
            self._ip_material.commit_state()

    def revert_state(self) -> None:
        if self._ip_material is not None:
            self._ip_material.revert_state()

    def M_global(self, *, lumped: bool = False) -> np.ndarray:
        rho = getattr(self.material, "rho", 0.0)
        if rho == 0.0:
            return np.zeros((12, 12))
        _, V = self._B_and_volume()
        m_total = rho * V
        # Consistent mass for a Tet4 (from M_ij = integral N_i N_j dV):
        # diagonal entries get m/10, off-diagonal entries get m/20.
        # (For each of the 3 translations the same scalar pattern.)
        M = np.zeros((12, 12))
        for i in range(4):
            for j in range(4):
                coef = m_total * (1.0 / 10.0 if i == j else 1.0 / 20.0)
                for k in range(3):
                    M[3 * i + k, 3 * j + k] = coef
        if lumped:
            return np.diag(M.sum(axis=1))
        return M

    # ------------------------------------------------------------ loading
    def set_body_force(self, bx: float, by: float, bz: float) -> None:
        self._body_force = np.array([float(bx), float(by), float(bz)])

    def clear_distributed_loads(self) -> None:
        self._body_force = np.zeros(3)

    def f_eq_global(self) -> np.ndarray:
        if not np.any(self._body_force):
            return np.zeros(12)
        _, V = self._B_and_volume()
        # CST: body force distributes equally to all 4 nodes
        f = np.zeros(12)
        bx, by, bz = self._body_force
        per_node = V / 4.0
        for i in range(4):
            f[3 * i + 0] = bx * per_node
            f[3 * i + 1] = by * per_node
            f[3 * i + 2] = bz * per_node
        return f

    def recover(self) -> None:
        B, _ = self._B_and_volume()
        D = self.material.D_3d()
        u = self.gather_u()
        self.strain = B @ u
        self.stress = D @ self.strain
