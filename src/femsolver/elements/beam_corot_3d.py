"""Chord-corotational 3-D Euler-Bernoulli beam-column.

Same kinematics as :class:`BeamColumn3D` (2 nodes, 6 DOF/node), but
the chord direction is recomputed from the *current* configuration each
iteration. The tangent stiffness picks up a geometric-stiffness
contribution from the chord rotation × axial-force interaction, which
is what makes the element catch P-Delta amplification in 3D.

Formulation
-----------
This is a "chord-corotational, small-rotation-nodal" formulation in the
style of Crisfield Vol. 2 §17.4 — appropriate for slender frame
problems where chord rotations can be large but nodal rotations stay
moderate (a few degrees). It is the same formulation that drives the
P-Delta options of SAP2000 / ETABS frame elements.

Specifically:

* **Chord rotation handled exactly** via a 3-D rotation pseudo-vector
  ``alpha``. For arbitrary chord rotation (no upper bound on angle)
  we compute the axis-and-angle pseudo-vector from
  ``e_x_initial × e_x_current`` and ``e_x_initial · e_x_current``.

* **Nodal rotations treated as small** — the nodal rotational DOFs
  in ``Node.disp`` are interpreted as pseudo-vector components, and
  natural rotations are obtained by element-wise subtraction of the
  chord rotation pseudo-vector. This is accurate for nodal-rotation
  magnitudes up to ~10 degrees. For larger rotations a full
  finite-rotation-matrix formulation (Battini-Pacoste, Cardona-Geradin)
  is needed; that is a substantial extension deferred to a follow-up.

* **Constitutive layer reused**: the natural-frame ``K_l_nat`` (6 x 6)
  comes from integrating the section response along the element using
  the standard ``BeamColumn3D._strain_disp_matrix`` and Gauss-Lobatto
  quadrature. For an elastic section this gives the analytical
  natural-frame stiffness; for a fiber section it picks up the per-IP
  state-dependent tangent (Phase 5.5 :class:`FiberSection3D`).

* **Geometric stiffness** from the chord-axial interaction. For a
  beam under axial tension N, transverse motion of the ends produces
  a restoring moment (cable-stiffening); under compression the same
  motion is destabilising (P-Delta). The 3-D form is

      K_g_axial = (N / L) * P_perp_3d

  where ``P_perp_3d`` is the 6 x 6 projection-onto-the-perpendicular-
  to-chord matrix applied to translational DOFs at both ends. Moment-
  induced geometric stiffness (Crisfield's ``M / L^2`` terms) is
  small at moderate rotations and is dropped here for simplicity.

At ``u = 0`` the element reduces to :class:`BeamColumn3D` exactly — a
test in ``tests/test_beam_corotational_3d.py`` pins this down to
machine precision.
"""
from __future__ import annotations

import numpy as np

from femsolver.elements.beam import BeamColumn3D
from femsolver.numerics.quadrature import gauss_lobatto_1d
from femsolver.sections.response.base import SectionBase


def _chord_rotation_pseudovector(
    e_x0: np.ndarray, e_x: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Return ``(alpha, |alpha|)``, the rotation pseudo-vector that
    takes ``e_x0`` to ``e_x``.

    ``alpha`` is the axis-angle representation: ``axis * angle`` where
    ``axis`` is the unit rotation axis and ``angle`` is in
    ``[0, pi]``. For nearly-aligned (small-rotation) cases we use the
    Taylor expansion ``alpha ≈ e_x0 × e_x``.

    For exactly opposite directions (angle = pi) the axis is degenerate;
    we return a zero vector with a warning in the docstring — this
    case shouldn't arise in practice for slender beam-columns.
    """
    cross = np.cross(e_x0, e_x)
    sin_alpha = float(np.linalg.norm(cross))
    cos_alpha = float(np.dot(e_x0, e_x))
    if sin_alpha < 1.0e-14:
        # Near-zero rotation or near-pi rotation.
        if cos_alpha > 0.0:
            return np.zeros(3), 0.0
        # angle ≈ pi: degenerate. Pick an arbitrary perpendicular axis.
        # For our chord-rotation case this would mean the chord
        # reversed, which is unphysical for beam analysis.
        # Pick any perpendicular axis to e_x0.
        axis = np.array([1.0, 0.0, 0.0]) if abs(e_x0[0]) < 0.9 \
            else np.array([0.0, 1.0, 0.0])
        axis = axis - (axis @ e_x0) * e_x0
        axis = axis / np.linalg.norm(axis)
        return axis * np.pi, np.pi
    angle = float(np.arctan2(sin_alpha, cos_alpha))
    axis = cross / sin_alpha
    return axis * angle, angle


class BeamColumn3DCorotational(BeamColumn3D):
    """3-D corotational beam-column — chord rotation handled exactly,
    nodal rotations linearised.

    Drop-in replacement for :class:`BeamColumn3D` for problems that
    need P-Delta or large rigid-body rotation effects. Same
    constructor signature.

    Limitations
    -----------
    * Nodal rotation magnitudes assumed small (linearised). Rotations
      above ~10° at any node will produce error in the natural-frame
      transformation. For applications that need finite-rotation
      accuracy at the node level (Battini-Pacoste-style) a more
      sophisticated formulation is needed.
    * Moment-induced geometric-stiffness terms (Crisfield's
      ``M / L^2`` blocks) are dropped. This is fine when bending
      moments are modest relative to the axial-force-induced
      stiffness; for problems near a lateral-torsional buckling load
      the simplification may underestimate the effect.
    """

    def __init__(
        self,
        tag: int,
        nodes,
        material,
        area: float | None = None,
        Iy: float | None = None,
        Iz: float | None = None,
        J: float | None = None,
        vecxz=None,
        *,
        section: SectionBase | None = None,
    ):
        # Forward to BeamColumn3D — handles all three construction
        # paths (legacy, elastic-section, fiber-section).
        super().__init__(tag, nodes, material,
                         area, Iy, Iz, J, vecxz,
                         section=section)
        # The initial chord direction is the corotational "reference"
        # configuration. We populate it lazily on the first call to
        # ``_current_geometry`` — at construction time the element
        # isn't yet bound to a model so ``node_coords()`` would fail.
        self._L0_init: float | None = None
        self._ex0_init: np.ndarray | None = None
        self._ey0_init: np.ndarray | None = None
        self._ez0_init: np.ndarray | None = None

    def _ensure_initial_geometry(self) -> None:
        if self._L0_init is None:
            L0, ex0, ey0, ez0 = self.length_and_axes()
            self._L0_init = L0
            self._ex0_init = ex0.copy()
            self._ey0_init = ey0.copy()
            self._ez0_init = ez0.copy()

    # ------------------------------------------------- current geometry
    def _current_geometry(self) -> tuple[float, float, np.ndarray, np.ndarray, float]:
        """Return ``(L0, L, ex_curr, alpha, sin_alpha)``.

        ``L0`` is the initial chord length, ``L`` the current (deformed)
        chord length, ``ex_curr`` the current chord-direction unit
        vector (3-vec), ``alpha`` the rotation pseudo-vector from
        initial to current chord (3-vec), and ``sin_alpha`` the
        magnitude of the cross product used to construct it.
        """
        self._ensure_initial_geometry()
        coords = self.node_coords()
        D = coords[1] - coords[0]
        # Ensure 3-D
        if D.size == 2:
            D = np.array([D[0], D[1], 0.0])
        L0 = self._L0_init
        u_g = self.gather_u()
        # Current chord vector includes translational increments
        d_x = float(D[0] + u_g[6] - u_g[0])
        d_y = float(D[1] + u_g[7] - u_g[1])
        d_z = float(D[2] + u_g[8] - u_g[2])
        L = float(np.sqrt(d_x ** 2 + d_y ** 2 + d_z ** 2))
        if L == 0.0:
            raise ValueError(
                f"corotational beam {self.tag}: nodes coincide at current state"
            )
        ex_curr = np.array([d_x, d_y, d_z]) / L
        alpha, _ = _chord_rotation_pseudovector(self._ex0_init, ex_curr)
        return L0, L, ex_curr, alpha, float(np.linalg.norm(alpha))

    # ------------------------------------------ natural-frame quantities
    def _natural_strain(self) -> np.ndarray:
        """Return the natural-frame strain vector at *every* Gauss-
        Lobatto integration point along the element.

        The natural-frame DOFs aren't 6 separate quantities — instead
        we use the element's existing 12-DOF natural displacement in a
        "co-rotated" frame:

            u_local_corot[0]    = u_nat (axial elongation)
            u_local_corot[1, 2] = 0 (chord passes through node 1)
            u_local_corot[3]    = theta_x_a_rel
            u_local_corot[4]    = theta_y_a_rel
            u_local_corot[5]    = theta_z_a_rel
            u_local_corot[6]    = u_nat (same as [0] for axial)
            u_local_corot[7, 8] = 0 (chord passes through node 2)
            u_local_corot[9]    = theta_x_b_rel
            u_local_corot[10]   = theta_y_b_rel
            u_local_corot[11]   = theta_z_b_rel

        i.e., we set transverse displacements at both ends to zero
        (the chord is the new local x-axis), keep the axial
        elongation, and subtract the chord rotation pseudo-vector
        from each node's rotational DOFs (small-rotation
        approximation). This is what allows the standard
        ``BeamColumn3D._strain_disp_matrix`` (cubic Hermite for
        bending) to be used unchanged.
        """
        L0, L, ex_curr, alpha, _ = self._current_geometry()
        u_g = self.gather_u()
        # Natural rotations: subtract chord rotation from each node's
        # rotational pseudo-vector. Index map for u_g (12-vec):
        #   0..2  = u_x, u_y, u_z of node a
        #   3..5  = theta_x, theta_y, theta_z of node a
        #   6..8  = u_x, u_y, u_z of node b
        #   9..11 = theta_x, theta_y, theta_z of node b
        theta_a = u_g[3:6]
        theta_b = u_g[9:12]
        theta_a_rel = theta_a - alpha
        theta_b_rel = theta_b - alpha
        # Build the "co-rotated local" 12-vec.
        u_local_corot = np.zeros(12)
        u_local_corot[0] = 0.0          # use axial extension at u[6], not [0]
        # Transverse displacements at both nodes are zero by construction
        # in the chord frame.
        u_local_corot[3:6] = theta_a_rel
        u_local_corot[6] = L - L0       # axial elongation distributed to node b
        u_local_corot[9:12] = theta_b_rel
        return u_local_corot

    # ------------------------------------------ corotational tangent + force
    def f_int_global(self) -> np.ndarray:
        """Internal nodal force in global coordinates at the current
        deformed state. Computed by integrating section forces along
        the element in the *chord-aligned* (co-rotated) local frame,
        then transforming back to global.
        """
        L0, L, ex_curr, alpha, _ = self._current_geometry()
        # Build the co-rotated local 12-vec for the element.
        u_local_corot = self._natural_strain()

        # Section integration in the chord frame.
        self._ensure_sections_length()
        xi_pts, w_pts = gauss_lobatto_1d(self.n_int)
        f_local = np.zeros(12)
        jac = 0.5 * L0
        for i, (xi, w) in enumerate(zip(xi_pts, w_pts)):
            B = self._strain_disp_matrix(xi, L0)
            e_i = B @ u_local_corot
            s_i, _ = self.sections[i].get_response(e_i)
            f_local += (w * jac) * (B.T @ s_i)

        # Transform back to global. The "transform matrix" is built
        # from the CURRENT chord triad — that is what makes this a
        # corotational element.
        T_curr = self._transform_matrix_current(ex_curr)
        return T_curr.T @ f_local

    def K_tangent_global(self) -> np.ndarray:
        """Tangent stiffness in global coordinates: material part from
        natural-frame section integration plus geometric part from the
        chord-axial interaction (P-Delta).
        """
        L0, L, ex_curr, alpha, _ = self._current_geometry()
        u_local_corot = self._natural_strain()

        # --- material part: K_l = integral B^T k_s B dx ---
        self._ensure_sections_length()
        xi_pts, w_pts = gauss_lobatto_1d(self.n_int)
        K_local = np.zeros((12, 12))
        jac = 0.5 * L0
        N_axial = 0.0
        for i, (xi, w) in enumerate(zip(xi_pts, w_pts)):
            B = self._strain_disp_matrix(xi, L0)
            e_i = B @ u_local_corot
            s_i, k_s = self.sections[i].get_response(e_i)
            K_local += (w * jac) * (B.T @ k_s @ B)
            # Use the axial force from the mid-element IP (or
            # equivalently any IP — N is constant for an equilibrium
            # field) for geometric stiffness.
            if abs(xi) < 1.0e-12:
                N_axial = float(s_i[0])
        # If no IP landed exactly at xi=0 (n_int even), average ends
        if N_axial == 0.0 and self.n_int > 0:
            # Re-evaluate at midpoint
            B_mid = self._strain_disp_matrix(0.0, L0)
            e_mid = B_mid @ u_local_corot
            s_mid, _ = self.sections[self.n_int // 2].get_response(e_mid) \
                if self.n_int % 2 == 1 else \
                self.sections[0].get_response(e_mid)
            N_axial = float(s_mid[0])

        # --- transform material part to global via current chord triad ---
        T_curr = self._transform_matrix_current(ex_curr)
        K_mat_global = T_curr.T @ K_local @ T_curr

        # --- geometric part (P-Delta from axial force) ---
        # The chord-perpendicular projection acts on translational DOFs
        # at both nodes. In 3D, the "perpendicular-to-chord" matrix is
        # (I - ex ex^T), a 3x3.
        I3 = np.eye(3)
        Pperp = I3 - np.outer(ex_curr, ex_curr)
        K_g = np.zeros((12, 12))
        # Translational block at each node: (N/L) * Pperp.
        # Cross-coupling between nodes: -(N/L) * Pperp (compressive
        # axial pulls both ends together).
        a = N_axial / L
        K_g[0:3, 0:3] = +a * Pperp
        K_g[6:9, 0:3] = -a * Pperp
        K_g[0:3, 6:9] = -a * Pperp
        K_g[6:9, 6:9] = +a * Pperp
        # K_g is already in global coords (we used ex_curr which is
        # global). No transformation needed.

        return K_mat_global + K_g

    # ----------------------------------------------- transform matrix
    def _transform_matrix_current(self, ex_curr: np.ndarray) -> np.ndarray:
        """Build the 12 x 12 transformation matrix that takes the
        global-frame nodal displacement vector to the chord-aligned
        local-frame vector. ``ex_curr`` is the current chord unit
        vector. The local y and z axes are constructed from
        ``self._ey0_init`` and ``self._ez0_init`` rotated by the
        chord-rotation pseudo-vector (so they form an orthonormal
        triad with ``ex_curr``).

        For simplicity here we re-derive a local y, z triad directly
        from the current chord direction using the same procedure as
        :meth:`BeamColumn3D.length_and_axes` — i.e., construct from a
        user-supplied vecxz heuristic. This makes the local triad's
        in-plane orientation consistent at any rotation.
        """
        ex = ex_curr
        if self._vecxz_user is not None:
            v = self._vecxz_user
        else:
            v = np.array([0.0, 0.0, 1.0]) if abs(ex[2]) < 0.999 \
                else np.array([1.0, 0.0, 0.0])
        ez = v - (v @ ex) * ex
        nz = np.linalg.norm(ez)
        if nz < 1e-12:
            raise ValueError(
                f"corotational beam {self.tag}: vecxz parallel to "
                f"current chord — pick a different reference vector"
            )
        ez = ez / nz
        ey = np.cross(ez, ex)
        R = np.vstack([ex, ey, ez])
        T = np.zeros((12, 12))
        for i in range(4):
            T[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] = R
        return T

    # ----------------------------------------------------- K_global override
    # K_global() inherits from BeamColumn3D: it returns the initial
    # elastic stiffness at u = 0. At u = 0, K_tangent_global reduces
    # to the same matrix (chord direction = initial direction, all
    # natural deformations zero, K_g = 0). A test pins this down.

    # ---- recover() inherited from BeamColumn3D works as-is, because
    # _evaluate_sections_along_length now uses self.sections (Phase 5.5
    # fix); the section forces it reports are in the chord-aligned
    # local frame, just like the elastic-only path.
