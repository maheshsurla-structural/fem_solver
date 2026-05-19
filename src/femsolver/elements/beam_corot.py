"""Corotational 2-D Euler-Bernoulli beam-column.

Same kinematics as :class:`BeamColumn2D` (2 nodes, 3 DOF/node), but the
chord direction and length are recomputed from the *current*
configuration each iteration. The tangent stiffness picks up a
geometric-stiffness contribution proportional to the current axial
force and bending moments, which is what allows P-Delta, snap-through,
and Euler-style buckling behaviour to be captured.

Formulation
-----------

Three "natural" DOFs are extracted from the current configuration:

    u_nat       = L - L0                  (axial elongation)
    theta1_nat  = theta1 - alpha          (end rotation relative to chord)
    theta2_nat  = theta2 - alpha

where ``alpha`` is the rotation of the chord vector from its initial
orientation to the current one, computed by ``atan2`` of the cross
product / dot product of (initial-chord, current-chord) so that it
remains a single-valued angle in ``(-pi, pi]``.

Constitutive response in the natural frame is obtained by integrating
the *section* response along the chord using Gauss-Lobatto quadrature.
At each integration point ``xi`` the section strain

    e(xi) = B_nat(xi) @ q_nat,    q_nat = [u_nat, theta1_nat, theta2_nat]

is fed to the section's ``get_response`` to obtain ``s = [N, Mz]`` and
the section tangent ``ks (2 x 2)``. These are then assembled into the
3 x 3 natural-frame stiffness ``K_l_nat`` and force ``f_nat`` via

    K_l_nat = sum_i w_i (L0/2) B_nat(xi_i)^T ks(xi_i) B_nat(xi_i)
    f_nat   = sum_i w_i (L0/2) B_nat(xi_i)^T s(xi_i)

For an :class:`ElasticSection2D` this reproduces the closed-form
``EA / L0`` and ``4 EI / L0`` / ``2 EI / L0`` stiffness to machine
precision once ``n_int >= 3``. For a :class:`FiberSection2D` the
natural-frame tangent picks up the per-fiber state-dependent ``ks``,
giving the corotational element full **P-Delta + distributed
plasticity** capability — including the off-diagonal axial-bending
coupling that emerges after asymmetric yielding.

Global internal force and tangent are obtained from the natural-frame
quantities via:

    f_int_g     = B(c, s, L)^T  f_nat
    K_tangent_g = B^T K_l_nat B  +  K_g(N, M1 + M2, c, s, L)

where ``B`` is the 3 x 6 mapping ``dq_nat = B du_g`` and ``K_g`` is the
geometric stiffness derived from the variation of ``B`` with respect
to the chord direction (depends only on N, M1 + M2, c, s, L).

At ``u = 0`` the element reduces to :class:`BeamColumn2D` exactly — a
test in ``tests/test_beam_corotational.py`` pins this down to machine
precision.
"""
from __future__ import annotations

import numpy as np

from femsolver.elements.beam import BeamColumn2D
from femsolver.numerics.quadrature import gauss_lobatto_1d
from femsolver.sections.base import SectionBase


class BeamColumn2DCorotational(BeamColumn2D):
    """Corotational 2-D beam-column — geometrically and (optionally)
    materially nonlinear.

    Construction follows the same three-way pattern as
    :class:`BeamColumn2D`:

    * ``(area, Iz)`` — legacy elastic constructor; an internal
      :class:`ElasticSection2D` is built.
    * ``section=ElasticSection2D(...)`` — explicit elastic section.
    * ``section=FiberSection2D(...)`` — stateful fiber section. The
      element clones the section per integration point and integrates
      constitutive response in the chord-aligned frame; combines
      P-Delta with distributed plasticity.
    """

    def __init__(
        self,
        tag: int,
        nodes,
        material,
        area: float | None = None,
        Iz: float | None = None,
        *,
        section: SectionBase | None = None,
    ):
        # Forward everything to BeamColumn2D's constructor; the parent
        # handles elastic vs stateful section detection, per-IP cloning,
        # area/Iz extraction, and the use_numerical_integration flag.
        super().__init__(tag, nodes, material, area, Iz, section=section)

    # ------------------------------------------------------------ geometry
    def _current_geometry(self) -> tuple[float, float, float, float, float, float, float]:
        """Return ``(L0, L, c, s, c0, s0, alpha)`` at the current state.

        ``(c0, s0)`` are the initial chord-direction cosines, ``(c, s)``
        the current ones, and ``alpha`` the chord rotation in radians
        in ``(-pi, pi]``.
        """
        coords = self.node_coords()
        D = coords[1] - coords[0]
        L0 = float(np.linalg.norm(D))
        if L0 == 0.0:
            raise ValueError(
                f"corotational beam {self.tag}: nodes coincide initially"
            )
        c0 = float(D[0] / L0)
        s0 = float(D[1] / L0)
        u_g = self.gather_u()
        d_x = float(D[0] + u_g[3] - u_g[0])
        d_y = float(D[1] + u_g[4] - u_g[1])
        L = float(np.hypot(d_x, d_y))
        if L == 0.0:
            raise ValueError(
                f"corotational beam {self.tag}: nodes coincide at current state"
            )
        c = d_x / L
        s = d_y / L
        # Chord rotation alpha from initial to current, single-valued
        # via atan2 of the cross / dot products.
        sin_alpha = c0 * s - s0 * c
        cos_alpha = c0 * c + s0 * s
        alpha = float(np.arctan2(sin_alpha, cos_alpha))
        return L0, L, c, s, c0, s0, alpha

    # ------------------------------------------------- natural quantities
    @staticmethod
    def _natural_B_matrix(xi: float, L0: float) -> np.ndarray:
        """Section-strain to natural-DOF map at natural coordinate ``xi``.

        Returns the 2 x 3 ``B_nat`` matrix such that

            [eps_axial, kappa_z] = B_nat @ [u_nat, theta1_nat, theta2_nat]

        Axial strain comes from a linear shape function on ``u`` so the
        first row is constant in ``xi``. Curvature comes from the
        second derivatives of the two Hermite cubics that survive after
        eliminating ``v(0) = v(L0) = 0`` in the chord-aligned frame —
        the same ``(3 xi -+ 1)/L0`` pattern that appears in the
        displacement-based :class:`BeamColumn2D` for the rotational
        DOFs.
        """
        return np.array([
            [1.0 / L0,     0.0,                  0.0                  ],
            [0.0,          (3.0 * xi - 1.0) / L0, (3.0 * xi + 1.0) / L0],
        ])

    def _natural_response(self) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
        """Return ``(f_nat, K_l_nat, c, s, L, L0)`` at the current state.

        ``f_nat = [N, M1, M2]`` and the 3 x 3 ``K_l_nat`` are obtained by
        integrating the section response along the chord with
        Gauss-Lobatto quadrature:

            f_nat   = sum_i w_i (L0/2) B_nat^T s(B_nat q_nat)
            K_l_nat = sum_i w_i (L0/2) B_nat^T ks B_nat

        For an elastic section ``ks`` is constant and the integral
        collapses to the standard ``diag(EA/L0, 4 EI/L0 / 2 EI/L0)``
        block. For a fiber section ``ks`` and ``s`` vary along the
        chord with the local section state, picking up axial-bending
        coupling and progressive yielding.
        """
        L0, L, c, s, _, _, alpha = self._current_geometry()
        u_g = self.gather_u()
        theta1 = float(u_g[2])
        theta2 = float(u_g[5])
        # Natural deformations
        u_nat = L - L0
        theta1_nat = theta1 - alpha
        theta2_nat = theta2 - alpha
        q_nat = np.array([u_nat, theta1_nat, theta2_nat])

        # Integrate section response along the chord. The per-IP
        # ``self.sections`` list was built by the parent constructor:
        # for an elastic section all entries share one object (cost-
        # free), for a fiber section each entry is an independent clone
        # carrying its own per-fiber state.
        self._ensure_sections_length()
        xi_pts, w_pts = gauss_lobatto_1d(self.n_int)
        K_l_nat = np.zeros((3, 3))
        f_nat = np.zeros(3)
        jac = 0.5 * L0
        for i, (xi, w) in enumerate(zip(xi_pts, w_pts)):
            B_nat = self._natural_B_matrix(xi, L0)
            e_section = B_nat @ q_nat
            s_section, ks_section = self.sections[i].get_response(e_section)
            f_nat += (w * jac) * (B_nat.T @ s_section)
            K_l_nat += (w * jac) * (B_nat.T @ ks_section @ B_nat)
        return f_nat, K_l_nat, c, s, L, L0

    # ---------------------------------------------------- transformation B
    @staticmethod
    def _B(c: float, s: float, L: float) -> np.ndarray:
        """3 x 6 transformation ``dq_nat = B du_g``.

        DOF order in global: ``[u1, v1, theta1, u2, v2, theta2]``.
        DOF order in natural: ``[u_nat, theta1_nat, theta2_nat]``.
        """
        return np.array([
            [-c,    -s,    0.0, c,    s,     0.0],
            [-s/L,  c/L,   1.0, s/L,  -c/L,  0.0],
            [-s/L,  c/L,   0.0, s/L,  -c/L,  1.0],
        ])

    # --------------------------------------- geometric stiffness K_g
    @staticmethod
    def _K_geometric(c: float, s: float, L: float,
                     N: float, M_sum: float) -> np.ndarray:
        """Geometric-stiffness matrix in global coords.

        ``K_g = (dB^T / du_g) f_nat``. For a 2-D corotational beam this
        splits cleanly into a string-stiffening part proportional to
        ``N/L`` and a moment part proportional to ``(M1 + M2) / L^2``.
        The two patterns are the standard "axial" and "moment" geometric
        matrices respectively; they involve only the translational rows
        and columns (rotational DOFs do not enter the chord-direction
        derivatives).
        """
        cs = c * s
        c2 = c * c
        s2 = s * s
        d  = c2 - s2          # cos(2 beta) up to sign
        a1 = N / L
        a2 = M_sum / (L * L)
        # Axial (string-stiffening) block. The 6 x 6 pattern is built
        # from the 2 x 2 perpendicular-projection matrix repeated on
        # the (u1, v1) and (u2, v2) blocks with anti-symmetric coupling.
        K_axial = a1 * np.array([
            [s2,   -cs,   0.0, -s2,    cs,   0.0],
            [-cs,   c2,   0.0,  cs,   -c2,   0.0],
            [0.0,   0.0,  0.0, 0.0,   0.0,   0.0],
            [-s2,   cs,   0.0,  s2,   -cs,   0.0],
            [cs,   -c2,   0.0, -cs,    c2,   0.0],
            [0.0,   0.0,  0.0, 0.0,   0.0,   0.0],
        ])
        # Moment block: comes from the d/du of (s/L, -c/L) in B,
        # multiplied by (M1 + M2). Shows up as a "sin(2 beta)" /
        # "cos(2 beta)" pattern between translational DOFs.
        K_moment = a2 * np.array([
            [-2*cs,   d,      0.0,  2*cs,  -d,      0.0],
            [ d,      2*cs,   0.0, -d,     -2*cs,   0.0],
            [ 0.0,    0.0,    0.0, 0.0,    0.0,     0.0],
            [ 2*cs,  -d,      0.0, -2*cs,   d,      0.0],
            [-d,     -2*cs,   0.0,  d,      2*cs,   0.0],
            [ 0.0,    0.0,    0.0, 0.0,    0.0,     0.0],
        ])
        return K_axial + K_moment

    # ------------------------------------------- public element interface
    def f_int_global(self) -> np.ndarray:
        """Internal nodal force in global coords at the current state."""
        f_nat, _, c, s, L, _ = self._natural_response()
        B = self._B(c, s, L)
        return B.T @ f_nat

    def K_tangent_global(self) -> np.ndarray:
        """Tangent stiffness at the current state — material + geometric."""
        f_nat, K_l_nat, c, s, L, _ = self._natural_response()
        B = self._B(c, s, L)
        K_mat = B.T @ K_l_nat @ B
        K_geo = self._K_geometric(c, s, L, f_nat[0], f_nat[1] + f_nat[2])
        return K_mat + K_geo

    # K_global() inherits from BeamColumn2D and returns the initial
    # elastic stiffness — used by LinearStaticAnalysis and as the first
    # iterate in modified Newton. At u = 0 it coincides with
    # K_tangent_global() by construction; a test in the suite pins
    # this equivalence down to machine precision.

    # ------------------------------------------------------------- recovery
    def recover(self) -> None:
        """Element-end forces (in local / chord-aligned frame) at the
        current state. Mirrors :meth:`BeamColumn2D.recover` but uses the
        natural-frame internal forces rather than ``K_local @ u_local``.
        """
        f_nat, _, c, s, L, _ = self._natural_response()
        N, M1, M2 = float(f_nat[0]), float(f_nat[1]), float(f_nat[2])
        # End forces in the chord-aligned local frame:
        #   [F_x1, F_y1, M_1, F_x2, F_y2, M_2]
        # Shear from end moments: F_y = -(M1 + M2) / L (top of beam),
        # opposite at the other end (sign chosen to match BeamColumn2D
        # convention so end_forces_local is comparable across element
        # types).
        V = (M1 + M2) / L
        self.end_forces_local = np.array([
            -N,    -V,     M1,
             N,     V,     M2,
        ])
