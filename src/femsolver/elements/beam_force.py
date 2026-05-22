"""Force-based 2-D Euler-Bernoulli beam-column (corotational kinematics).

A *force-based* (= flexibility-based) beam-column element assumes the
**force distribution** along the element axis a priori — not the
displacement field. In a 2-D beam-column without distributed load,
equilibrium between two end nodes makes the moment diagram exactly
*linear* and the axial force exactly *constant*. By interpolating
those forces and integrating section *flexibility* to recover
deformations, the element captures distributed plasticity with one
element per member, regardless of mesh density.

This is the canonical "OpenSees ``forceBeamColumn``" formulation: the
default beam-column for performance-based earthquake analysis since
the late 1990s. Crucially, the moment diagram is exact under
equilibrium even when the section response is highly nonlinear —
unlike displacement-based cubic-Hermite beams (Phase 5), where the
moment distribution is only as good as the assumed displacement
field's second derivative.

Formulation
-----------
Three natural-frame quantities at the element ends:

    v = [u_nat, theta_1_nat, theta_2_nat]^T          (deformations)
    q = [N, M_1, M_2]^T                                (basic forces)

Force interpolation ``b(x)`` distributes ``q`` to section forces
``s(x) = [N(x), M_z(x)]``:

    N(x) = N
    M(x) = -(1 - x/L) M_1  +  (x/L) M_2

(the sign on the M_1 column matches the convention from
:class:`BeamColumn2DCorotational._natural_response` so that
``K_b = (integral b^T f_s b dx)^{-1}`` reproduces the standard
``EI/L * [[4, 2], [2, 4]]`` bending stiffness for an elastic section.)

State determination algorithm (Neuenhofer-Filippou 1997):

  1. Predictor: ``q = q_committed + K_b_committed (v - v_committed)``
  2. At each Gauss-Lobatto point ``xi_i``:
        target ``s = b(xi_i) q``
        section returns ``(s_curr, k_s)`` at committed strain
        ``de = k_s^{-1} (s - s_curr)``
        ``e_trial = e_committed + de``
        recall section with ``e_trial`` to update ``(s_curr, k_s)``
  3. ``v_computed = sum_i w_i (L/2) b^T(xi_i) e(xi_i)``
  4. ``dv = v - v_computed``;  if ||dv|| < tol, done.
  5. ``K_b = (sum_i w_i (L/2) b^T(xi_i) k_s^{-1}(xi_i) b(xi_i))^{-1}``
  6. ``q += K_b dv``;  repeat from step 2.

Convergence is quadratic for smooth sections. For sections with
discontinuous tangent (perfect plasticity), the iteration may stall —
a line-search or smoothing of the section response usually recovers it.

After convergence, ``q`` is the natural force, ``K_b`` is the natural
stiffness. The element returns these to its corotational parent
(:class:`BeamColumn2DCorotational`), which then wraps them with the
geometric stiffness and global transformation to produce
``f_int_global`` and ``K_tangent_global``.

Compared with the displacement-based variant
:class:`BeamColumn2DCorotational`:

* Same external interface (6 global DOFs, ``K_global``,
  ``K_tangent_global``, ``f_int_global``, ``recover``).
* Same per-IP section handling (clones for stateful sections).
* Same corotational kinematics, geometric stiffness, mass matrix.
* The *only* difference is :meth:`_natural_response` which uses
  force-based state determination instead of B-matrix integration of
  section response.
"""
from __future__ import annotations

import numpy as np

from femsolver.elements.beam_corot import BeamColumn2DCorotational
from femsolver.numerics.quadrature import gauss_lobatto_1d
from femsolver.sections.base import SectionBase


class ForceBeamColumn2DCorotational(BeamColumn2DCorotational):
    """Force-based (= flexibility-based) 2-D corotational beam-column.

    Drop-in replacement for :class:`BeamColumn2DCorotational`. Same
    constructor signature; the only difference is the constitutive
    integration is force-based rather than displacement-based, giving
    one-element-per-member accuracy under distributed plasticity.

    Parameters
    ----------
    All parameters identical to :class:`BeamColumn2DCorotational`.
    Additionally:

    state_det_tol : float, default 1e-9
        Tolerance on ``||v - v_computed||`` in the element-level
        Neuenhofer-Filippou iteration. Loose tolerance is fine — the
        outer global Newton iteration will drive any residual
        equilibrium error to zero at the structural level.
    state_det_max_iter : int, default 30
        Maximum number of element-level iterations per state-
        determination call.
    """

    # Class-level defaults; the user can override these per instance.
    # The inner tolerance is set well below typical global Newton
    # tolerances (~1e-6 to 1e-8) so the state-determination noise
    # floor does not limit the global convergence rate.
    state_det_tol: float = 1.0e-12
    state_det_max_iter: int = 30

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
        super().__init__(tag, nodes, material, area, Iz, section=section)
        # Committed natural force and per-IP section strain at the end
        # of the last converged step. The element-level iteration uses
        # these as the "reference" state from which the incremental
        # algorithm starts.
        self._q_committed: np.ndarray = np.zeros(3)
        self._e_committed: np.ndarray | None = None
        # Last K_b (used as predictor for the next state determination
        # if the previous step's K_b is still in scope).
        self._K_b_last: np.ndarray | None = None

    # ------------------------------------------ force interpolation b(x)
    @staticmethod
    def _b_matrix(xi: float, L: float) -> np.ndarray:
        """Force-interpolation matrix at natural coordinate xi.

        Returns a ``(2, 3)`` matrix ``b`` such that
        ``[N(x), M(x)]^T = b @ [N, M_1, M_2]^T`` where x = L (1+xi) / 2.

        The M_1 column carries a negative sign so that the resulting
        element flexibility ``F_b = integral b^T f_s b dx`` inverts to
        the standard ``EI/L * [[4, 2], [2, 4]]`` bending block of
        :class:`BeamColumn2DCorotational`'s natural stiffness.
        """
        # x / L  in terms of xi in [-1, 1]:  x/L = (1 + xi) / 2
        s = 0.5 * (1.0 + xi)            # s = x / L  in [0, 1]
        return np.array([
            [1.0,    0.0,         0.0],
            [0.0,   -(1.0 - s),   s  ],
        ])

    # ------------------------------------------ section-level inversion
    @staticmethod
    def _section_strain_for_force(
        section, s_target: np.ndarray, e_committed: np.ndarray,
        *, max_iter: int = 20, tol: float = 1.0e-12,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find the section strain ``e`` such that ``section.get_response(e)``
        returns a force vector equal to ``s_target``, by Newton iteration
        starting from ``e_committed``.

        Returns ``(e, k_s)`` at convergence. For an elastic section the
        function returns after one step (``f_s = k_s^{-1}`` is constant).
        For nonlinear sections this is the "inner-inner" iteration
        used by Neuenhofer-Filippou's element-level algorithm.

        We use the *committed* state as starting point and accumulate
        increments; this avoids accidentally setting the section's
        trial state to inconsistent values during the outer iteration.
        """
        # First call to seed the trial state and grab the initial tangent.
        e = np.asarray(e_committed, dtype=float).copy()
        s_curr, k_s = section.get_response(e)
        for _ in range(max_iter):
            r = s_target - s_curr
            if float(np.max(np.abs(r))) < tol:
                return e, k_s
            try:
                de = np.linalg.solve(k_s, r)
            except np.linalg.LinAlgError:
                # Section tangent is singular (perfect plasticity); fall
                # back to least-squares (will give zero strain increment
                # along the singular direction — correct for the plastic
                # plateau).
                de, *_ = np.linalg.lstsq(k_s, r, rcond=None)
            e = e + de
            s_curr, k_s = section.get_response(e)
        # Did not converge; return the current state and let the outer
        # iteration accommodate.
        return e, k_s

    # ------------------------------------------ Neuenhofer-Filippou loop
    def _natural_response(self) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
        """Force-based version of the parent's
        :meth:`BeamColumn2DCorotational._natural_response`. Returns
        ``(f_nat, K_l_nat, c, s, L, L0)``. The signature is identical
        so the parent's ``f_int_global`` and ``K_tangent_global`` need
        no changes — the corotational wrapping and geometric stiffness
        are reused unchanged.
        """
        L0, L, c, s, _, _, alpha = self._current_geometry()
        u_g = self.gather_u()
        theta1 = float(u_g[2])
        theta2 = float(u_g[5])
        u_nat = L - L0
        theta1_nat = theta1 - alpha
        theta2_nat = theta2 - alpha
        v_target = np.array([u_nat, theta1_nat, theta2_nat])

        self._ensure_sections_length()
        xi_pts, w_pts = gauss_lobatto_1d(self.n_int)
        n_r = self.sections[0].n_resultants
        jac = 0.5 * L0
        n_ip = self.n_int

        # Initialize committed section strains on the first call
        if self._e_committed is None:
            self._e_committed = np.zeros((n_ip, n_r))

        # Predict q from committed state + last K_b.
        if self._K_b_last is not None:
            q = self._q_committed + self._K_b_last @ (
                v_target - self._v_at_last_commit()
            )
        else:
            q = self._q_committed.copy()

        # Working arrays: per-IP trial strain and tangent
        e_trial = self._e_committed.copy()
        # K_b cached across iterations
        K_b = None

        for _it in range(self.state_det_max_iter):
            # --- Section loop: bring section strains into agreement
            # with target forces s(x) = b(x) q, then integrate to get
            # the natural deformation v_computed and flexibility F_b.
            F_b = np.zeros((3, 3))
            v_computed = np.zeros(3)
            for i, (xi, w) in enumerate(zip(xi_pts, w_pts)):
                b = self._b_matrix(xi, L0)
                s_target = b @ q                          # (n_r,)
                # Solve section: find e_i such that section gives s_target.
                e_i, k_s_i = self._section_strain_for_force(
                    self.sections[i], s_target,
                    e_committed=self._e_committed[i],
                )
                e_trial[i] = e_i
                # Section flexibility f_s = k_s^{-1}; use solve for stability.
                try:
                    f_s_b = np.linalg.solve(k_s_i, b)
                except np.linalg.LinAlgError:
                    f_s_b, *_ = np.linalg.lstsq(k_s_i, b, rcond=None)
                F_b += (w * jac) * (b.T @ f_s_b)
                v_computed += (w * jac) * (b.T @ e_i)

            dv = v_target - v_computed
            if float(np.max(np.abs(dv))) < self.state_det_tol:
                K_b = np.linalg.inv(F_b)
                break

            # Newton update on q
            try:
                K_b = np.linalg.inv(F_b)
            except np.linalg.LinAlgError as exc:
                raise RuntimeError(
                    f"force-based beam {self.tag}: element flexibility "
                    f"became singular during state determination ({exc}). "
                    "Likely cause: every section has lost stiffness in "
                    "a direction the element needs (e.g. fully plastic). "
                    "Use a hardening section, refine the time step, or "
                    "switch to displacement-based for the post-collapse "
                    "regime."
                ) from exc
            q = q + K_b @ dv
        else:
            raise RuntimeError(
                f"force-based beam {self.tag}: state determination did "
                f"not converge in {self.state_det_max_iter} iterations "
                f"(||dv|| = {float(np.max(np.abs(dv))):.3e}, "
                f"tol = {self.state_det_tol:.3e})"
            )

        # Stash the trial state for the next iteration and for commit.
        self._q_trial = q
        self._e_trial = e_trial
        self._K_b_last = K_b
        return q, K_b, c, s, L, L0

    def _v_at_last_commit(self) -> np.ndarray:
        """Reconstruct the natural deformations corresponding to the
        committed state. Used by the predictor to take a fresh-step
        increment from the converged state."""
        # Stored at commit time in ``self._v_committed``; if not yet
        # populated (very first call), assume zero.
        return getattr(self, "_v_committed", np.zeros(3))

    # ---------------------------------------------------------- lifecycle
    def commit_state(self) -> None:
        # Forward section commit to per-IP sections via the parent.
        super().commit_state()
        # Snapshot the converged force / strain state for the next
        # step's predictor.
        self._q_committed = self._q_trial.copy() if hasattr(self, "_q_trial") else self._q_committed
        if hasattr(self, "_e_trial"):
            self._e_committed = self._e_trial.copy()
        # Remember the natural deformations at this commit for the
        # incremental predictor.
        L0, L, _, _, _, _, alpha = self._current_geometry()
        u_g = self.gather_u()
        self._v_committed = np.array([
            L - L0,
            float(u_g[2]) - alpha,
            float(u_g[5]) - alpha,
        ])

    def revert_state(self) -> None:
        super().revert_state()
        # Trial natural state reverts to committed
        # (no element-level trial-vs-committed plumbing here beyond
        # what super() does; the sections themselves revert via super).
        if hasattr(self, "_q_trial"):
            del self._q_trial
        if hasattr(self, "_e_trial"):
            del self._e_trial
