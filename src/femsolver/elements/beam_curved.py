"""Curved Timoshenko beam in 2D -- 3-node isoparametric formulation.

The element uses **the same quadratic shape functions for both
geometry and displacement** (isoparametric formulation), letting it
represent arches and curved bridge girders with one element per
~30-60 degrees of arc.

Geometry
--------

Three nodes (1, 2, 3) lie along the beam axis. Node 1 is at the
start, node 3 at the end, node 2 anywhere between them (often the
midpoint, but the formulation handles asymmetric placement).

The natural coordinate ``xi in [-1, 1]`` parametrises the element.
Shape functions::

    N1(xi) = 0.5*xi*(xi - 1)
    N2(xi) = 1 - xi^2
    N3(xi) = 0.5*xi*(xi + 1)

At each Gauss point, the tangent vector
``dX/dxi = sum_i (dN_i/dxi) * X_i`` defines the local axial direction;
the local normal is perpendicular (rotated 90 deg CCW). The arc-length
metric is ``ds/dxi = |dX/dxi|``.

DOFs and strains
----------------

Each node carries ``(u, v, theta_z)`` in *global* coordinates, so
the element has 9 DOFs total. At each Gauss point we rotate the
displacement field into local axial/transverse components

* axial:        ``u_a = u_x cos(a) + u_y sin(a)``
* transverse:   ``u_t = -u_x sin(a) + u_y cos(a)``

where ``a`` is the local tangent angle. The Timoshenko strains are

* ``eps = d(u_a)/ds``                  (axial)
* ``kappa = d(theta_z)/ds``            (curvature change)
* ``gamma = d(u_t)/ds - theta_z``      (shear)

Stiffness
---------

``K = integral B^T D B ds``  with  ``D = diag(EA, EI, G*A_s)``
where ``A_s = A / k_s`` is the effective shear area
(``k_s`` = shear correction factor, default 6/5 for rectangular).

Integration uses **2-point Gauss reduced integration** to avoid
shear locking on slender elements. This is exact for the quadratic
axial / bending strains and *under-integrates* the shear by one
order, which is the classical fix for Timoshenko locking. The
result is rotation- and translation-invariant for any curvature.
"""
from __future__ import annotations

import numpy as np

from femsolver.elements.base import Element
from femsolver.numerics.quadrature import gauss_legendre_1d


class CurvedBeam2D(Element):
    """3-node isoparametric quadratic curved Timoshenko beam (2D).

    Parameters
    ----------
    tag : int
    nodes : (3,) sequence of node tags (1=start, 2=interior, 3=end)
    material : Material providing ``E``, ``nu`` (and optionally ``G``).
    area : float
        Cross-sectional area A.
    Iz : float
        Second moment of area about the out-of-plane axis.
    shear_correction : float, default 1.2
        ``k_s`` -- divides A to give the effective shear area
        ``A_s = A / k_s``. ``1.2`` (= 6/5) is the textbook value
        for rectangular sections; ``1.11`` for circular; ``A/A_web``
        for I-sections.
    integration_points : int, default 2
        Number of 1D Gauss points. 2 = reduced (recommended,
        cures shear locking); 3 = full (over-stiff on slender
        elements).
    """

    n_nodes = 3
    dofs_per_node = 3              # u, v, theta_z

    def __init__(
        self,
        tag: int,
        nodes,
        material,
        area: float,
        Iz: float,
        *,
        shear_correction: float = 1.2,
        integration_points: int = 2,
    ):
        super().__init__(tag, nodes, material)
        if area <= 0:
            raise ValueError(f"area must be positive, got {area}")
        if Iz <= 0:
            raise ValueError(f"Iz must be positive, got {Iz}")
        if shear_correction <= 0:
            raise ValueError(
                f"shear_correction must be positive, got {shear_correction}"
            )
        if integration_points not in (2, 3):
            raise ValueError(
                f"integration_points must be 2 (reduced) or 3 (full), "
                f"got {integration_points}"
            )
        self.area = float(area)
        self.Iz = float(Iz)
        self.shear_correction = float(shear_correction)
        self.integration_points = int(integration_points)

    # ---------------------------------------------------------------- shapes

    @staticmethod
    def shape_functions(xi: float) -> np.ndarray:
        """Quadratic Lagrange shape functions at ``xi in [-1, 1]``."""
        return np.array([
            0.5 * xi * (xi - 1.0),
            1.0 - xi * xi,
            0.5 * xi * (xi + 1.0),
        ])

    @staticmethod
    def dN_dxi(xi: float) -> np.ndarray:
        """Derivatives ``dN/dxi`` at ``xi``."""
        return np.array([xi - 0.5, -2.0 * xi, xi + 0.5])

    # ---------------------------------------------------------------- geometry

    def _frame_at(self, xi: float, X: np.ndarray):
        """Return ``(ds_dxi, c, s)`` at the parametric coord ``xi``.

        ``ds_dxi`` = arc-length metric, ``c, s`` = direction cosines of
        the local tangent (``c = cos(a), s = sin(a)``).
        """
        dN = self.dN_dxi(xi)                  # (3,)
        dX_dxi = dN @ X                        # (2,) tangent (un-normalised)
        ds_dxi = float(np.linalg.norm(dX_dxi))
        if ds_dxi <= 0.0:
            raise ValueError(
                f"CurvedBeam2D element {self.tag}: zero tangent at xi={xi}. "
                f"Check node ordering (1 - 2 - 3 along the curve)."
            )
        c = float(dX_dxi[0] / ds_dxi)
        s = float(dX_dxi[1] / ds_dxi)
        return ds_dxi, c, s

    # ---------------------------------------------------------------- B matrix

    def _B_at(self, xi: float, X: np.ndarray) -> tuple[np.ndarray, float]:
        """Strain-displacement matrix in *generalised* strain order
        ``[eps, kappa, gamma]`` and arc-length weight ``ds_dxi``.

        The local-to-global rotation is baked into B so the element
        stiffness can be assembled directly without explicit T.
        """
        N = self.shape_functions(xi)
        dN = self.dN_dxi(xi)
        ds_dxi, c, sn = self._frame_at(xi, X)
        # d/ds = (1 / ds_dxi) * d/dxi
        dN_ds = dN / ds_dxi
        # Each node has 3 DOFs (u, v, theta_z). Express the three
        # Timoshenko strains as linear combinations of nodal DOFs:
        #
        # eps   = d(u_a)/ds = sum_i dN_i/ds * (cos(a) u_i + sin(a) v_i)
        # kappa = d(theta)/ds = sum_i dN_i/ds * theta_i
        # gamma = d(u_t)/ds - theta
        #       = sum_i dN_i/ds * (-sin(a) u_i + cos(a) v_i) - sum_i N_i theta_i
        B = np.zeros((3, 9))
        for i in range(3):
            B[0, 3 * i + 0] = c * dN_ds[i]
            B[0, 3 * i + 1] = sn * dN_ds[i]
            B[1, 3 * i + 2] = dN_ds[i]
            B[2, 3 * i + 0] = -sn * dN_ds[i]
            B[2, 3 * i + 1] = c * dN_ds[i]
            B[2, 3 * i + 2] = -N[i]
        return B, ds_dxi

    # ---------------------------------------------------------------- D
    def _D(self) -> np.ndarray:
        """Section constitutive matrix ``diag(EA, EI, G*A/k_s)``."""
        E = float(self.material.E)
        nu = float(self.material.nu)
        G = E / (2.0 * (1.0 + nu))
        A_s = self.area / self.shear_correction
        return np.diag([E * self.area, E * self.Iz, G * A_s])

    # ---------------------------------------------------------------- K

    def K_global(self) -> np.ndarray:
        X = self.node_coords()              # (3, 2)
        D = self._D()
        K = np.zeros((9, 9))
        xi_g, w_g = gauss_legendre_1d(self.integration_points)
        for q in range(xi_g.size):
            B, ds_dxi = self._B_at(float(xi_g[q]), X)
            K += (B.T @ D @ B) * (ds_dxi * w_g[q])
        return K

    # ---------------------------------------------------------------- mass
    def M_global(self, *, lumped: bool = False) -> np.ndarray:
        rho = self.material.rho
        if rho == 0.0:
            return np.zeros((9, 9))
        X = self.node_coords()
        rA = rho * self.area
        # Polar mass moment per unit length: rho * Iz
        rI = rho * self.Iz
        M = np.zeros((9, 9))
        # Use full 3-point integration for the mass even in reduced mode
        xi_g, w_g = gauss_legendre_1d(3)
        for q in range(xi_g.size):
            N = self.shape_functions(float(xi_g[q]))
            ds_dxi, _, _ = self._frame_at(float(xi_g[q]), X)
            jw_t = rA * ds_dxi * w_g[q]
            jw_r = rI * ds_dxi * w_g[q]
            for a in range(3):
                for b in range(3):
                    M[3 * a + 0, 3 * b + 0] += N[a] * N[b] * jw_t
                    M[3 * a + 1, 3 * b + 1] += N[a] * N[b] * jw_t
                    M[3 * a + 2, 3 * b + 2] += N[a] * N[b] * jw_r
        if lumped:
            row_sums = M.sum(axis=1)
            return np.diag(row_sums)
        return M

    def clear_distributed_loads(self) -> None:
        pass
