"""Corotational 2D truss element with engineering strain.

Same kinematics as the linear :class:`Truss2D` (2 nodes, 2 DOF/node), but
the chord direction and length are recomputed from the *current*
configuration each iteration. The tangent stiffness picks up a
geometric-stiffness contribution proportional to the current axial
force, which is what allows snap-through and large-rotation behaviour
to be captured.

Formulation
-----------

Reference (initial) chord vector ``D = X2 - X1`` of length ``L0``;
current chord ``d = D + (u2 - u1)`` of length ``L`` and unit vector
``n_hat = d / L``. With engineering strain ``eps = (L - L0) / L0`` and
axial force ``N = E A eps``:

    f_int_e = N * [-n_hat; n_hat]    (4-vector, in global coords)

    K_T = [[A, -A], [-A, A]]         (block 4x4, each block 2x2)

with

    A = (E A / L0) * n_hat n_hat^T  +  (N / L) * (I - n_hat n_hat^T)

The first term is the material tangent (positive definite, present even
when N=0); the second is the geometric stiffness, which is
*destabilising* in compression (N<0) and stiffens transverse motion in
tension (the cable-stiffening effect).

At ``u=0`` the element reduces to :class:`Truss2D` exactly — used as a
free verification path.
"""
from __future__ import annotations

import numpy as np

from femsolver.elements.truss import Truss2D


class Truss2DCorotational(Truss2D):
    """Corotational truss in 2D — geometrically-nonlinear, materially-linear."""

    def _current_geometry(self) -> tuple[float, np.ndarray, float, float]:
        """Return ``(L0, n_hat, L, N)`` at the current state.

        ``L0`` and the initial direction come from node coordinates;
        ``L``, ``n_hat`` come from coords + current ``Node.disp``.
        ``N = E A eps`` with engineering strain.
        """
        L0, _ = self.length_and_dirs()
        coords = self.node_coords()
        D = coords[1] - coords[0]
        u = self.gather_u()
        u1 = u[0:2]
        u2 = u[2:4]
        d = D + (u2 - u1)
        L = float(np.linalg.norm(d))
        if L == 0.0:
            raise ValueError(
                f"corotational truss {self.tag}: nodes coincide at current state"
            )
        n_hat = d / L
        eps = (L - L0) / L0
        N = self.material.E * self.area * eps
        return L0, n_hat, L, N

    # K_global remains the *initial* stiffness inherited from Truss2D — used
    # when this element is asked for an initial-stiffness K (e.g., linear
    # static analysis or the first iteration of modified Newton).

    def K_tangent_global(self) -> np.ndarray:
        L0, n, L, N = self._current_geometry()
        I2 = np.eye(2)
        nn = np.outer(n, n)
        A = (self.material.E * self.area / L0) * nn + (N / L) * (I2 - nn)
        K = np.zeros((4, 4))
        K[0:2, 0:2] = A
        K[2:4, 2:4] = A
        K[0:2, 2:4] = -A
        K[2:4, 0:2] = -A
        return K

    def f_int_global(self) -> np.ndarray:
        _, n, _, N = self._current_geometry()
        f = np.empty(4)
        f[0:2] = -N * n
        f[2:4] = N * n
        return f

    # mass: inherit from Truss2D — the consistent mass is unchanged by
    # geometric nonlinearity at this level (still uses initial geometry,
    # which is a standard approximation).

    def recover(self) -> None:
        L0, _, L, N = self._current_geometry()
        self.axial_force = N
        self.axial_stress = N / self.area
