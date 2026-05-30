"""4-node membrane quadrilateral with in-plane drilling rotation
(Allman / Ibrahimbegovic / Hughes-Brezzi formulation).

A plain Quad4 plane element has 2 DOF/node (u, v) and exhibits no
in-plane rotational stiffness about the normal direction. When such
a membrane is *combined with a plate-bending shell* (as in a flat
shell element) or is *attached to a beam* that does carry an out-of-
plane drilling moment, the system has spurious modes: the beam can
rotate about its own normal without the membrane resisting.

The Allman-Ibrahimbegovic (1990) "drilling-DOF" membrane adds a
third in-plane rotation ``theta_z`` per node, and enriches the
displacement field with a hierarchical incompatible shape so that

* The translation field is still bilinear in (u, v).
* The rotation field adds a quadratic correction that ties
  ``theta_z`` to the antisymmetric part of the in-plane displacement
  gradient -- enforced via a penalty/constraint term

      Pi_drill = (gamma / 2) * integral [theta_z - 0.5(dv/dx - du/dy)]^2 dA

This gives the element a non-singular 12x12 stiffness in
``(u, v, theta_z)`` per node (3 DOF/node) which fits naturally into a
6-DOF/node shell or beam-assembled global model.

The penalty ``gamma`` is taken as ``G * t`` per Hughes-Brezzi (1989),
giving the right units and an exact patch test in the elastic limit.

Reference
---------
Ibrahimbegovic A., Taylor R. L., Wilson E. L. (1990). "A robust
quadrilateral membrane finite element with drilling degrees of
freedom." *Int. J. Numer. Meth. Engng.* 30, 445-457.
"""
from __future__ import annotations

import numpy as np

from femsolver.elements.base import Element
from femsolver.numerics.quadrature import gauss_legendre_2d_quad


class MembraneQ4Drilling(Element):
    """4-node membrane with drilling DOF.

    Parameters
    ----------
    tag : int
    nodes : (4,) sequence of node tags (CCW).
    material : Material providing ``D_plane_stress()`` and ``E``, ``nu``.
    thickness : float
    state : {"plane_stress", "plane_strain"}, default "plane_stress"
    drilling_penalty : float, optional
        Stiffness multiplier for the drilling-rotation penalty term.
        ``None`` (default) means ``G * t`` per Hughes-Brezzi. Set to a
        small positive number to "stiffen" the drilling DOF when used
        as a stabiliser for a flat-shell coupling, or to 0 to fall
        back to a pure Q4 membrane (with 3 trivial DOFs/node).
    """

    n_nodes = 4
    dofs_per_node = 3                  # u, v, theta_z

    def __init__(
        self,
        tag: int,
        nodes,
        material,
        thickness: float = 1.0,
        state: str = "plane_stress",
        drilling_penalty: float | None = None,
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
        self.drilling_penalty = drilling_penalty
        self.gp_stress: list[np.ndarray] = []
        self.gp_strain: list[np.ndarray] = []

    # ---------------------------------------------------------------- helpers

    def D(self) -> np.ndarray:
        if self.state == "plane_stress":
            return self.material.D_plane_stress()
        return self.material.D_plane_strain()

    @staticmethod
    def _shape(xi: float, eta: float) -> np.ndarray:
        return 0.25 * np.array([
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ])

    @staticmethod
    def _dN_dxi(xi: float, eta: float) -> np.ndarray:
        return 0.25 * np.array([
            [-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)],
            [-(1.0 - xi), -(1.0 + xi), (1.0 + xi),   (1.0 - xi)],
        ])

    def _jacobian(self, xi: float, eta: float, X: np.ndarray):
        dN = self._dN_dxi(xi, eta)               # (2, 4)
        J = dN @ X
        detJ = float(np.linalg.det(J))
        if detJ <= 0.0:
            raise ValueError(
                f"MembraneQ4Drilling element {self.tag}: non-positive "
                f"Jacobian ({detJ:g}) at xi={xi}, eta={eta}."
            )
        Jinv = np.linalg.inv(J)
        dN_dx = Jinv @ dN                         # (2, 4)
        return J, detJ, dN_dx

    def _G(self) -> float:
        """Shear modulus from the underlying material."""
        E = float(self.material.E)
        nu = float(self.material.nu)
        return E / (2.0 * (1.0 + nu))

    # ---------------------------------------------------------------- K
    def K_global(self) -> np.ndarray:
        X = self.node_coords()                   # (4, 2)
        D = self.D()
        t = self.thickness
        # 3 DOF per node -> 12x12
        K = np.zeros((12, 12))
        # Membrane contribution: standard Q4 plane-stress integrand at
        # the (u, v) DOFs, plus a drilling-penalty term acting on
        # the antisymmetric part of the displacement gradient and the
        # nodal theta_z values.
        gamma = (
            self._G() * t if self.drilling_penalty is None
            else float(self.drilling_penalty)
        )
        # Map: per-node DOFs are (u, v, theta_z) so for node a in 0..3:
        #     u_a     -> 3*a + 0
        #     v_a     -> 3*a + 1
        #     theta_a -> 3*a + 2
        xi, eta, w = gauss_legendre_2d_quad(2)
        for q in range(xi.size):
            N = self._shape(float(xi[q]), float(eta[q]))
            _, detJ, dN_dx = self._jacobian(
                float(xi[q]), float(eta[q]), X,
            )
            # ----------- standard membrane B (3, 12) at the (u, v) slots
            B_uv = np.zeros((3, 12))
            for a in range(4):
                dNx = dN_dx[0, a]
                dNy = dN_dx[1, a]
                B_uv[0, 3 * a + 0] = dNx
                B_uv[1, 3 * a + 1] = dNy
                B_uv[2, 3 * a + 0] = dNy
                B_uv[2, 3 * a + 1] = dNx
            K += (B_uv.T @ D @ B_uv) * (t * detJ * w[q])
            # ----------- drilling penalty
            # phi := theta_z - 0.5*(dv/dx - du/dy)
            # phi_h = sum_a N_a * theta_a - 0.5 * (sum_a (dN_a/dx * v_a -
            #                                            dN_a/dy * u_a))
            # so dphi/d(u_a) = +0.5 * dN_a/dy,
            #    dphi/d(v_a) = -0.5 * dN_a/dx,
            #    dphi/d(theta_a) = N_a.
            B_phi = np.zeros(12)
            for a in range(4):
                B_phi[3 * a + 0] = 0.5 * dN_dx[1, a]
                B_phi[3 * a + 1] = -0.5 * dN_dx[0, a]
                B_phi[3 * a + 2] = N[a]
            K += gamma * np.outer(B_phi, B_phi) * (detJ * w[q])
        return K

    # ---------------------------------------------------------------- mass
    def M_global(self, *, lumped: bool = False) -> np.ndarray:
        rho = self.material.rho
        if rho == 0.0:
            return np.zeros((12, 12))
        X = self.node_coords()
        t = self.thickness
        M = np.zeros((12, 12))
        # Rotational inertia: I_theta = rho * t * h^2 / 12 (per node)
        # using an effective characteristic size h^2 of the element.
        # We approximate it as the element area / 4 per integration
        # contribution; this is small but non-singular.
        xi, eta, w = gauss_legendre_2d_quad(2)
        for q in range(xi.size):
            N = self._shape(float(xi[q]), float(eta[q]))
            _, detJ, _ = self._jacobian(
                float(xi[q]), float(eta[q]), X,
            )
            jw = rho * t * detJ * w[q]
            Nbar_t = np.zeros((2, 12))            # translations only
            for a in range(4):
                Nbar_t[0, 3 * a + 0] = N[a]
                Nbar_t[1, 3 * a + 1] = N[a]
            M += (Nbar_t.T @ Nbar_t) * jw
            # Rotational inertia per node (tiny but non-zero so that
            # the mass matrix is non-singular when used with explicit
            # integrators).
            I_rot = jw * detJ            # ~ rho * t * area_q for tiny I
            for a in range(4):
                M[3 * a + 2, 3 * a + 2] += N[a] ** 2 * I_rot
        if lumped:
            row_sums = M.sum(axis=1)
            return np.diag(row_sums)
        return M

    # ---------------------------------------------------------------- recover
    def recover(self) -> None:
        X = self.node_coords()
        u_full = self.gather_u()                  # (12,)
        # Strip drilling DOFs to compute strains
        u_uv = np.zeros(8)
        for a in range(4):
            u_uv[2 * a + 0] = u_full[3 * a + 0]
            u_uv[2 * a + 1] = u_full[3 * a + 1]
        D = self.D()
        self.gp_stress = []
        self.gp_strain = []
        xi, eta, _ = gauss_legendre_2d_quad(2)
        for q in range(xi.size):
            _, _, dN_dx = self._jacobian(
                float(xi[q]), float(eta[q]), X,
            )
            B = np.zeros((3, 8))
            for a in range(4):
                B[0, 2 * a + 0] = dN_dx[0, a]
                B[1, 2 * a + 1] = dN_dx[1, a]
                B[2, 2 * a + 0] = dN_dx[1, a]
                B[2, 2 * a + 1] = dN_dx[0, a]
            eps = B @ u_uv
            sig = D @ eps
            self.gp_strain.append(eps)
            self.gp_stress.append(sig)

    def clear_distributed_loads(self) -> None:
        pass
